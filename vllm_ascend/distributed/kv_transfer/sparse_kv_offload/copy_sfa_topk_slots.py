# SPDX-License-Identifier: Apache-2.0
"""Request-owned fused_copy_sfa top-k row slots shared by the PD connector and runner."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


COPY_SFA_POOL_PADDING_ROWS = 2


def copy_sfa_pool_capacity(max_num_seqs: int) -> int:
    """Match the runner-owned fused_copy_sfa row arena: one row per request plus padding."""
    return max_num_seqs + COPY_SFA_POOL_PADDING_ROWS


def copy_sfa_tail_geometry(kv_tokens: int, block_size: int) -> tuple[int, int]:
    """Return ``(tail_tokens, tail_block_index)`` for a finished prefill prefix.

    The circular tail only stores the incomplete last block. A 128-aligned
    prefix has nothing to prefetch.
    """
    if kv_tokens <= 0 or block_size <= 0:
        return 0, 0
    tail_tokens = kv_tokens % block_size
    if tail_tokens == 0:
        return 0, 0
    return tail_tokens, kv_tokens // block_size


def copy_sfa_prefill_dest_geometry(
    kv_tokens: int,
    block_size: int,
    hot_tokens: int,
) -> tuple[bool, int, int]:
    """Return ``(dense, tail_tokens, tail_block_index)`` for a finished prefill.

    ``dense=True`` when the whole prompt fits the decode row's hot region
    (``kv_tokens <= hot_tokens``): blocks ``[0, ceil(kv_tokens / block_size))``
    are D2D'd to row offsets ``b * block_size`` so the request can decode in
    the -3 non-offload state. This also covers 128-aligned prompts that the
    circular-tail path would skip entirely. Otherwise the circular tail only
    prefetches the incomplete last block; a block-aligned ``kv_tokens`` keeps
    ``(False, 0, 0)`` and the hot region arrives via the decode-side -2 init.
    """
    if kv_tokens <= hot_tokens:
        return True, 0, 0
    tail_tokens, tail_block_index = copy_sfa_tail_geometry(kv_tokens, block_size)
    return False, tail_tokens, tail_block_index


def prepare_copy_sfa_dummy_slots(slots: np.ndarray, generations: np.ndarray, padded_reqs: int) -> None:
    """Fill CPU buffer views with private rows and inactive generations.

    Pass the full slot buffer: its length is the real-row pool capacity.
    """
    slots[:padded_reqs] = np.arange(padded_reqs, dtype=np.int32) + len(slots)
    generations[:padded_reqs] = -1


def prepare_copy_sfa_dummy_slots_active(slots: np.ndarray, active: np.ndarray, padded_reqs: int) -> None:
    """Fill CPU views with private slots and mark every padded row inactive (active model)."""
    slots[:padded_reqs] = np.arange(padded_reqs, dtype=np.int32) + len(slots)
    active[:padded_reqs] = False


def prepare_copy_sfa_request_slots(
    *,
    req_ids: Sequence[str],
    live_req_ids: Collection[str],
    slots: np.ndarray,
    generations: np.ndarray,
    request_slots: dict[str, int],
    slot_generations: dict[int, int],
    last_prefixes: dict[int, int],
    generation: int,
    prebound_slots: dict[str, int],
    computed_tokens: np.ndarray | None,
    padded_reqs: int,
    block_size: int,
    hot_tokens: int,
    dummy: bool,
    restore_new_requests: bool = False,
) -> tuple[dict[str, int], int, bool, dict[int, tuple[int, int]]]:
    """Prepare CPU row ownership and return updated state plus rollback work.

    ``req_ids`` contains the rows in this batch. Pass the full CPU slot and
    generation buffer views, whose length is the real-row pool capacity.
    Slot generations and prefix history update in place. The result contains
    the retained request mapping, generation counter, tail-restore flag and
    dense fills as ``slot -> (batch row, computed tokens)``. The caller owns
    these buffers/maps and submits any returned dense fills to the manager.
    """
    capacity = len(slots)
    prepare_copy_sfa_dummy_slots(slots, generations, padded_reqs)
    restore_tails = False
    dense_fills: dict[int, tuple[int, int]] = {}
    if not dummy:
        # PD binds rows at alloc time. Keep those reservations even when the
        # request is waiting for KV and is not in the current decode batch.
        request_slots = {
            req: slot for req, slot in request_slots.items() if req in live_req_ids or req in prebound_slots
        }
        used = set(request_slots.values()) | set(prebound_slots.values())
        available = iter(slot for slot in range(capacity) if slot not in used)
        for row, req in enumerate(req_ids):
            if req not in request_slots:
                slot = prebound_slots[req] if req in prebound_slots else next(available)
                request_slots[req] = slot
                generation += 1
                slot_generations[slot] = generation
                last_prefixes.pop(slot, None)
                if restore_new_requests and computed_tokens is not None:
                    computed = int(computed_tokens[row])
                    if computed <= hot_tokens:
                        dense_fills[slot] = (row, computed)
                    else:
                        restore_tails = True
                # Colocated prefill populates the row in exec_kv. Existing PD
                # connectors deliver the full prompt into the Host pool and
                # need one eager restore when Decode first owns the row.
            slot = request_slots[req]
            slots[row] = slot
            generations[row] = slot_generations[slot]
        if computed_tokens is not None:
            for row in range(len(req_ids)):
                slot = int(slots[row])
                prefix = (int(computed_tokens[row]) // block_size) * block_size
                last_prefix = last_prefixes.get(slot)
                if last_prefix is not None and prefix < last_prefix:
                    restore_tails = True
                    # A rollback across the hot boundary leaves a sparse-layout
                    # row under a dense (-3) reader: refill the whole row.
                    if hot_tokens and last_prefix >= hot_tokens > prefix:
                        dense_fills[slot] = (row, int(computed_tokens[row]))
                last_prefixes[slot] = prefix
    return request_slots, generation, restore_tails, dense_fills


@dataclass
class _CopySfaRequestState:
    slot: int
    last_prefix: int | None = None


class CopySfaRequestStates:
    """Request-owned history; batch buffers are views of these bindings.

    PD reservations remain authoritative. New bindings explicitly invalidate
    target/MTP LIM banks before metadata is built. Prefix history belongs
    to the same request incarnation, never to a reusable physical slot.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _CopySfaRequestState] = {}
        self._retired_requests: set[str] = set()

    def remove_request(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
        self._retired_requests.add(req_id)

    def prepare(
        self,
        *,
        req_ids: Sequence[str],
        live_req_ids: Collection[str],
        slots: np.ndarray,
        active: np.ndarray,
        prebound_slots: dict[str, int],
        computed_tokens: np.ndarray | None,
        padded_reqs: int,
        block_size: int,
        hot_tokens: int,
        dummy: bool,
        lim_cache_histories: Sequence[Tensor],
    ) -> tuple[bool, dict[int, tuple[int, int]], tuple[int, ...]]:
        """Return tail restoration, dense fills, and slots needing LIM reset."""
        capacity = len(slots)
        if not 0 <= len(req_ids) <= padded_reqs <= capacity or len(active) != capacity:
            raise ValueError("fused_copy_sfa request rows exceed the slot buffer capacity")
        if block_size <= 0:
            raise ValueError("fused_copy_sfa block size must be positive")
        prepare_copy_sfa_dummy_slots_active(slots, active, padded_reqs)
        if dummy:
            return False, {}, ()
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("fused_copy_sfa batch contains duplicate request IDs")
        # Worker connector cleanup runs after forward. The runner can therefore
        # see an already-finished owner's reservation alongside its replacement.
        # Forget retirement only once the connector drops it, or this ID is
        # explicitly admitted again with fresh request state.
        self._retired_requests.intersection_update(prebound_slots)
        self._retired_requests.difference_update(req_ids)
        prebound_slots = {req: slot for req, slot in prebound_slots.items() if req not in self._retired_requests}
        live = set(live_req_ids) | set(prebound_slots)
        if not set(req_ids) <= live:
            raise ValueError("fused_copy_sfa batch contains a request without a live owner")
        if computed_tokens is not None and len(computed_tokens) < len(req_ids):
            raise ValueError("fused_copy_sfa computed-token rows do not cover the batch")
        # Keep live batch bindings and PD reservations awaiting their next decode.
        retained = {req: state for req, state in self._requests.items() if req in live}
        owners: dict[int, str] = {}
        for req, slot in prebound_slots.items():
            if not 0 <= slot < capacity:
                raise ValueError(f"fused_copy_sfa invalid PD slot {slot} for {req}")
            if slot in owners:
                raise RuntimeError(f"fused_copy_sfa slot {slot} reserved by two requests")
            owners[slot] = req
        for req, state in retained.items():
            reserved = prebound_slots.get(req, state.slot)
            if reserved != state.slot:
                raise RuntimeError(f"fused_copy_sfa binding changed for live request {req}: {state.slot} -> {reserved}")
            if state.slot in owners and owners[state.slot] != req:
                raise RuntimeError(f"fused_copy_sfa slot {state.slot} still owned by {req}")
            owners[state.slot] = req
        available = iter(slot for slot in range(capacity) if slot not in owners)
        pending: dict[str, _CopySfaRequestState] = {}
        for req in req_ids:
            if req in retained:
                continue
            new_slot = prebound_slots.get(req)
            if new_slot is None:
                new_slot = next(available, None)
            if new_slot is None:
                raise RuntimeError(f"fused_copy_sfa topk slot pool exhausted (capacity={capacity})")
            pending[req] = _CopySfaRequestState(new_slot)
        # Publish bindings only after validating the entire ownership transition.
        invalidated_slots = tuple(state.slot for state in pending.values())
        if invalidated_slots:
            # Reset every supplied target/draft bank before publishing new
            # owners. Existing owners and inactive capture rows keep history.
            for last_cache in lim_cache_histories:
                last_cache[:, list(invalidated_slots)] = -1
        retained.update(pending)
        self._requests = retained
        restore_tails = False
        dense_fills: dict[int, tuple[int, int]] = {}
        for row, req in enumerate(req_ids):
            state = retained[req]
            slots[row] = state.slot
            active[row] = True
            if computed_tokens is None:
                continue
            prefix = (int(computed_tokens[row]) // block_size) * block_size
            if state.last_prefix is not None and prefix < state.last_prefix:
                if prebound_slots:
                    restore_tails = True
                if hot_tokens and state.last_prefix >= hot_tokens > prefix:
                    dense_fills[state.slot] = (row, int(computed_tokens[row]))
            state.last_prefix = prefix
        return restore_tails, dense_fills, invalidated_slots


class CopySfaTopkSlotAllocator:
    """Bind a stable top-k row to a request from PD alloc until it finishes."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"fused_copy_sfa topk slot capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._free: list[int] = list(range(capacity))
        self._req_to_slot: dict[str, int] = {}

    def bind(self, req_id: str) -> int:
        slot = self._req_to_slot.get(req_id)
        if slot is not None:
            return slot
        if not self._free:
            raise RuntimeError(f"fused_copy_sfa topk slot pool exhausted (capacity={self.capacity})")
        slot = self._free.pop(0)
        self._req_to_slot[req_id] = slot
        return slot

    def get(self, req_id: str) -> int | None:
        return self._req_to_slot.get(req_id)

    def can_bind(self, req_id: str) -> bool:
        """Check admission without reserving a row before KV allocation succeeds."""
        return req_id in self._req_to_slot or bool(self._free)

    def release(self, req_id: str) -> int | None:
        slot = self._req_to_slot.pop(req_id, None)
        if slot is not None:
            self._free.append(slot)
        return slot

    def bound_slots(self) -> dict[str, int]:
        return dict(self._req_to_slot)
