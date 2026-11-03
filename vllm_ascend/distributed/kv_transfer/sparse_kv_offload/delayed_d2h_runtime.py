# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V1 step boundaries for request-owned delayed Mooncake D2H."""

from collections.abc import Callable, Sequence

import numpy as np

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_decode_worker import DsaDecodeWorkerState
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    CopySfaTopkSlotAllocator,
    prepare_copy_sfa_dummy_slots,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h import corrected_computed_frontier
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h_executor import MooncakeDelayedD2HExecutor


class DelayedD2HRuntime:
    def __init__(
        self,
        executor: MooncakeDelayedD2HExecutor,
        *,
        restore_rows: Callable[[dict[int, tuple[int, int]], np.ndarray], None],
        synchronize_compute: Callable[[], None],
    ) -> None:
        self.executor = executor
        self.restore_rows = restore_rows
        self.synchronize_compute = synchronize_compute
        self.slots = CopySfaTopkSlotAllocator(executor.planner.capacity)
        self._generations: dict[str, int] = {}
        self._state: DsaDecodeWorkerState | None = None
        self._previous_computed = np.empty(0, dtype=np.int64)
        self.computed_tokens: np.ndarray | None = None
        self._prepared = False
        self._closed = False

    def begin_step(self, state: DsaDecodeWorkerState) -> None:
        """Retire before input-batch updates, including empty/terminal steps."""
        if self._closed:
            raise RuntimeError("delayed D2H runtime is closed")
        if self._state is not None and self._state is not state:
            raise RuntimeError("delayed D2H worker ownership source changed")
        self._state = state
        pending = state.releasable()
        self.executor.retire(tuple((request.request_id, request.generation) for request in pending))
        for request_id, slot in self.slots.bound_slots().items():
            generation = self._generations[request_id]
            prefix = self.executor.planner.prefixes(np.array([slot]), np.array([generation]))[0]
            state.publish(request_id, generation, int(prefix))
        agreed = set(self.executor.agreed_releases)
        pending = tuple(request for request in pending if (request.request_id, request.generation) in agreed)
        if any(request.resident_bound for request in pending):
            # Copy completion does not fence the final model forward. Wait
            # only at release, before either the Host blocks or row is reused.
            self.synchronize_compute()
        for request in pending:
            slot = self.slots.get(request.request_id)
            if request.resident_bound != (slot is not None):
                raise RuntimeError("decode release residency disagrees with row ownership")
            if slot is not None:
                generation = self._generations[request.request_id]
                if generation != request.generation:
                    raise RuntimeError("decode release generation differs from resident row")
                self.executor.planner.release(np.array([slot]), np.array([generation]))
                self.slots.release(request.request_id)
                del self._generations[request.request_id]
            state.confirm_drained(request.request_id, request.generation)
        self.computed_tokens = None
        self._prepared = False

    def prepare_frontier(
        self,
        optimistic: np.ndarray,
        *,
        previous_positions: np.ndarray | None = None,
        previous_drafts: np.ndarray | None = None,
        valid_sampled_counts: np.ndarray | None = None,
    ) -> None:
        """The caller waits for the existing accepted-count event beforehand."""
        optimistic = np.asarray(optimistic)
        if optimistic.ndim != 1 or optimistic.dtype.kind not in "iu" or np.any(optimistic < 0):
            raise ValueError("computed frontier must be a nonnegative integer vector")
        if previous_positions is None:
            corrected = optimistic.astype(np.int64, copy=True)
        else:
            if previous_drafts is None or valid_sampled_counts is None:
                raise RuntimeError("async MTP correction requires completed sampled counts and previous drafts")
            previous_size = self._previous_computed.size
            corrected = corrected_computed_frontier(
                optimistic,
                self._previous_computed,
                previous_positions,
                previous_drafts[:previous_size],
                valid_sampled_counts[:previous_size],
            )
        self.computed_tokens = corrected
        self._previous_computed = corrected.copy()

    def prepare_batch(
        self,
        request_ids: Sequence[str],
        *,
        block_table: np.ndarray,
        max_write_tokens: np.ndarray,
        slots: np.ndarray,
        generations: np.ndarray,
        stable_prefixes: np.ndarray,
        padded_requests: int,
    ) -> None:
        """Restore newly owned rows, freeze their geometry, then launch D2H."""
        state = self._state
        computed = self.computed_tokens
        if state is None or computed is None or self._prepared:
            raise RuntimeError("delayed D2H requires one prepared batch per begun step")
        count = len(request_ids)
        capacity = self.executor.planner.capacity
        if len(set(request_ids)) != count or computed.shape != (count,):
            raise ValueError("request identity/order must match the corrected frontier")
        if any(len(buffer) != capacity for buffer in (slots, generations, stable_prefixes)):
            raise ValueError("runner buffers must match resident row capacity")
        if not count <= padded_requests <= capacity:
            raise ValueError("padded request count exceeds resident metadata capacity")
        requests = [state.acquire(request_id) for request_id in request_ids]
        new_requests = [request for request in requests if self.slots.get(request.request_id) is None]
        if len(new_requests) + len(self._generations) > capacity:
            raise RuntimeError("delayed D2H resident rows remain owned; wait for decode release")
        for row, request in enumerate(requests):
            if request.resident_bound:
                if self._generations.get(request.request_id) != request.generation:
                    raise RuntimeError("resident row generation differs from decode lifetime")
            elif computed[row] > request.host_ready_tokens:
                raise RuntimeError("cannot restore a new resident row beyond verified Host coverage")
        prepare_copy_sfa_dummy_slots(slots, generations, padded_requests)
        stable_prefixes[:padded_requests] = 0
        restore = {}
        for row, request in enumerate(requests):
            slot = self.slots.bind(request.request_id)
            slots[row] = slot
            generations[row] = request.generation
            if not request.resident_bound:
                self._generations[request.request_id] = request.generation
                restore[slot] = (row, int(computed[row]))
        if restore:
            self.restore_rows(restore, block_table)
            new_slots = np.array(list(restore), dtype=np.int64)
            new_rows = np.array([row for row, _ in restore.values()], dtype=np.int64)
            initial_prefixes = computed[new_rows] // self.executor.planner.block_size * self.executor.planner.block_size
            self.executor.planner.bind(new_slots, generations[new_rows], initial_prefixes)
            for row in new_rows:
                request = requests[row]
                state.bind_resident(request.request_id, request.generation)
        stable_prefixes[:count] = self.executor.planner.prefixes(slots[:count], generations[:count])
        self.executor.planner.plan(
            computed_tokens=computed,
            pool_slots=slots[:count],
            generations=generations[:count],
            block_table=block_table[:count],
            max_write_tokens=max_write_tokens,
        )
        self.executor.launch()
        self._prepared = True

    def finish_forward(self) -> None:
        if not self._prepared:
            raise RuntimeError("delayed D2H source-ready requires the prepared target/draft batch")
        self.executor.record_source_ready()

    def close(self) -> None:
        """Shutdown drains locally; it never publishes or acknowledges success."""
        if self._closed:
            return
        self.executor.close()
        self.synchronize_compute()
        self._closed = True
