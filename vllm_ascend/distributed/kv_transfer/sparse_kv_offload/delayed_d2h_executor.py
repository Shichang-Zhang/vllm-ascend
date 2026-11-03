# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute one frozen D2H batch into local Mooncake/AscendVMM views.

Only TP0 writes the shared Host pool. Every rank joins ``retire`` at the same
runner boundary, including idle iterations, so divergent or failed plans
cannot skip the completion collective and publish a partial Host prefix.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.delayed_d2h import DelayedD2HPlanner

DESCRIPTOR_FIELDS = 3
SOURCE_EVENTS = 2


@dataclass(frozen=True)
class D2HCopyComponent:
    """Keep the actual resident and rank-local Host allocations alive."""

    name: str
    resident: torch.Tensor
    host: torch.Tensor


class MooncakeDelayedD2HExecutor:
    def __init__(
        self,
        planner: DelayedD2HPlanner,
        components: Sequence[D2HCopyComponent],
        *,
        runtime: Any,
        tp_group: Any,
        tp_rank: int,
        tp_size: int,
        copy_blocks: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    ) -> None:
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError("invalid delayed D2H TP rank/size")
        if tp_size > 1 and tp_group is None:
            raise ValueError("delayed D2H requires a TP group")
        if not components or len({component.name for component in components}) != len(components):
            raise ValueError("delayed D2H requires unique registered K/V components")
        self.planner = planner
        self.components = tuple(components)
        self.runtime = runtime
        self.tp_group = tp_group
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.copy_blocks = copy_blocks
        self._launched_epoch: int | None = None
        self._error: str | None = None
        self._latest_source_slot: int | None = None
        self._inflight_source_slot: int | None = None
        self._closed = False
        self.agreed_releases: tuple[tuple[str, int], ...] = ()
        self._source_bases, self._host_bases, self._token_bytes = self._validate_components()
        self.stream = None
        if tp_rank == 0:
            capacity = len(components) * planner.capacity
            device = components[0].resident.device
            self._descriptors_host = torch.empty(
                (DESCRIPTOR_FIELDS, capacity), dtype=torch.int64, device="cpu", pin_memory=True
            )
            self._descriptors_numpy = self._descriptors_host.numpy()
            self._descriptors_device = torch.empty((DESCRIPTOR_FIELDS, capacity), dtype=torch.int64, device=device)
            self.stream = runtime.Stream()
            self._source_events = tuple(runtime.Event() for _ in range(SOURCE_EVENTS))
            self._complete_event = runtime.Event()

    def _validate_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sources, destinations, widths = [], [], []
        device = self.components[0].resident.device
        for component in self.components:
            resident, host = component.resident, component.host
            if resident.dtype != torch.bfloat16 or host.dtype != resident.dtype:
                raise ValueError(f"{component.name}: delayed D2H requires BF16 main KV")
            if resident.device != device or host.device != device:
                raise ValueError(f"{component.name}: Host must use this rank's local device-addressable view")
            if not resident.is_contiguous() or not host.is_contiguous():
                raise ValueError(f"{component.name}: delayed D2H requires contiguous KV views")
            if resident.ndim != 4 or resident.shape[1] != self.planner.row_tokens:
                raise ValueError(f"{component.name}: invalid resident row geometry")
            if resident.shape[0] < self.planner.capacity:
                raise ValueError(f"{component.name}: resident pool is smaller than planner capacity")
            if host.shape != (self.planner.host_num_blocks, self.planner.block_size, *resident.shape[2:]):
                raise ValueError(f"{component.name}: Host block geometry does not match resident KV")
            width = resident.shape[2] * resident.shape[3] * resident.element_size()
            if width <= 0:
                raise ValueError(f"{component.name}: KV token size must be positive")
            sources.append(resident.data_ptr())
            destinations.append(host.data_ptr())
            widths.append(width)
        return tuple(np.asarray(values, dtype=np.int64)[:, None] for values in (sources, destinations, widths))

    def _check_eager(self) -> None:
        if self._closed:
            raise RuntimeError("delayed D2H executor is closed")
        if self.runtime.is_current_stream_capturing():
            raise RuntimeError("delayed D2H control must remain outside graph capture")

    def record_source_ready(self) -> None:
        """Call after the final target/draft writer on the compute stream."""
        self._check_eager()
        if self._error is not None:
            # Submission errors are propagated collectively by retire. Do
            # not strand peer ranks by raising at this compute-only hook.
            return
        if self.tp_rank != 0:
            return
        # Repeated records may replace the free event, never the event still
        # referenced by the copy stream's wait for the in-flight batch.
        slot = 0 if self._inflight_source_slot is None else 1 - self._inflight_source_slot
        self._source_events[slot].record(self.runtime.current_stream())
        self._latest_source_slot = slot

    def _prepare_descriptors(self) -> int:
        batch = self.planner.inflight
        if batch is None:
            raise RuntimeError("delayed D2H has no frozen batch")
        count = len(self.components) * batch.pool_slots.size
        self._descriptors_numpy[0, :count] = (
            self._source_bases + batch.source_token_slots[None, :] * self._token_bytes
        ).reshape(-1)
        self._descriptors_numpy[1, :count] = (
            self._host_bases + batch.destination_blocks[None, :] * self.planner.block_size * self._token_bytes
        ).reshape(-1)
        self._descriptors_numpy[2, :count] = np.broadcast_to(
            self.planner.block_size * self._token_bytes, (len(self.components), batch.pool_slots.size)
        ).reshape(-1)
        # The pinned staging and device descriptors remain untouched until
        # completion. No descriptors point at another process's Host VA.
        self._descriptors_device.copy_(self._descriptors_host, non_blocking=True)
        return count

    def launch(self) -> bool:
        """Submit on TP0; report submission errors to every rank at retire.

        The caller must reach retire even after its final/no-forward step.
        A failed submission keeps all ownership and buffers protected.
        """
        self._check_eager()
        batch = self.planner.inflight
        if self._error is not None:
            raise RuntimeError(f"delayed D2H is poisoned: {self._error}")
        if batch is None:
            return False
        if self._launched_epoch is not None:
            raise RuntimeError("delayed D2H batch was already submitted")
        self._launched_epoch = batch.epoch
        if self.tp_rank != 0:
            return False
        try:
            if self._latest_source_slot is None:
                raise RuntimeError("delayed D2H has no source-ready frontier")
            self._inflight_source_slot = self._latest_source_slot
            with self.runtime.stream(self.stream):
                count = self._prepare_descriptors()
                self.stream.wait_event(self._source_events[self._inflight_source_slot])
                self.copy_blocks(*(self._descriptors_device[field, :count] for field in range(DESCRIPTOR_FIELDS)))
                self._complete_event.record(self.stream)
        except Exception as exc:
            self._error = f"submission failed: {exc}"
            return False
        return True

    def _batch_signature(self) -> tuple[int, bytes] | None:
        batch = self.planner.inflight
        if batch is None:
            return None
        digest = sha256()
        for array in (
            batch.pool_slots,
            batch.generations,
            batch.source_token_slots,
            batch.destination_blocks,
            batch.target_prefixes,
        ):
            digest.update(array.tobytes())
        return batch.epoch, digest.digest()

    def retire(self, release_candidates: tuple[tuple[str, int], ...] = ()) -> int:
        """Confirm copies and agree which decode rows all ranks can release.

        Receive/notify completion may differ across ranks. Taking the common
        release set in this existing collective keeps resident free-list order
        identical without a second per-step completion collective.
        """
        self._check_eager()
        batch = self.planner.inflight
        error = self._error
        if batch is not None and self._launched_epoch != batch.epoch:
            error = error or "batch has not been submitted"
        if batch is not None and self.tp_rank == 0 and error is None:
            try:
                self._complete_event.synchronize()
            except Exception as exc:
                error = f"completion wait failed: {exc}"
        status = (self._batch_signature(), error, release_candidates)
        statuses = [status]
        if self.tp_size > 1:
            statuses = [None] * self.tp_size
            # Always participate, even without a local batch. This also
            # detects empty/nonempty, generation, and block-ID divergence.
            try:
                dist.all_gather_object(statuses, status, group=self.tp_group.cpu_group)
            except Exception as exc:
                self._error = f"completion collective failed: {exc}"
                raise RuntimeError(f"delayed D2H {self._error}") from exc
        failures = [(rank, entry[1]) for rank, entry in enumerate(statuses) if entry[1] is not None]
        if failures or any(entry[0] != status[0] for entry in statuses):
            self._error = f"TP completion failed or plans disagree: {statuses}"
            raise RuntimeError(f"delayed D2H {self._error}")
        common_releases = set(release_candidates)
        for entry in statuses:
            common_releases.intersection_update(entry[2])
        self.agreed_releases = tuple(sorted(common_releases))
        if batch is None:
            return 0
        self.planner.complete(batch.epoch)
        self._launched_epoch = None
        self._inflight_source_slot = None
        return batch.pool_slots.size

    def close(self) -> None:
        """Locally drain submitted work before releasing shared-pool memory.

        Shutdown is not a successful prefix publication and does not require
        other ranks to remain alive. A stream failure prevents pool release.
        """
        if self._closed:
            return
        if self.stream is not None:
            self.stream.synchronize()
        self._closed = True
