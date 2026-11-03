# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side ownership and full-block planning for delayed Copy-SFA writes.

Planning never publishes a Host prefix. The executor must confirm DMA and TP
completion before calling ``complete``. Its frozen batch remains valid until
then, including when the corresponding request leaves the scheduled batch.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

INVALID_GENERATION = -1
RESIDENT_TAIL_BLOCKS = 2


def _integer_array(values: np.ndarray, name: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a {ndim}-D integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the int64 range")
    return array.astype(np.int64, copy=False)


def corrected_computed_frontier(
    optimistic: np.ndarray,
    previous_computed: np.ndarray,
    previous_positions: np.ndarray,
    previous_drafts: np.ndarray,
    valid_sampled_counts: np.ndarray,
) -> np.ndarray:
    """Mirror V1's device-side async MTP correction on completed Host inputs.

    ``previous_computed`` is the previous iteration's corrected frontier, in
    the previous batch order. It must not be advanced by speculative writes.
    The caller waits for the existing sampled-count D2H event before reading
    ``valid_sampled_counts``. New and non-draft rows use the scheduler value.
    No caller-owned array is modified.
    """
    optimistic = _integer_array(optimistic, "optimistic")
    previous_computed = _integer_array(previous_computed, "previous_computed")
    previous_positions = _integer_array(previous_positions, "previous_positions")
    previous_drafts = _integer_array(previous_drafts, "previous_drafts")
    valid_sampled_counts = _integer_array(valid_sampled_counts, "valid_sampled_counts")
    if previous_positions.shape != optimistic.shape:
        raise ValueError("previous_positions must match the current batch")
    if previous_computed.shape != previous_drafts.shape or previous_computed.shape != valid_sampled_counts.shape:
        raise ValueError("previous frontier, drafts and sampled counts must share the previous batch order")
    if np.any(optimistic < 0) or np.any(previous_computed < 0) or np.any(previous_drafts < 0):
        raise ValueError("computed tokens and draft counts must be nonnegative")
    old_rows = previous_positions >= 0
    if np.any(previous_positions < -1) or np.any(previous_positions[old_rows] >= previous_computed.size):
        raise ValueError("previous_positions contains an invalid previous batch row")
    old_indices = previous_positions[old_rows]
    if np.unique(old_indices).size != old_indices.size:
        raise ValueError("current requests must not alias a previous batch row")
    result = optimistic.copy()
    participating = previous_drafts[old_indices] > 0
    current_rows = np.flatnonzero(old_rows)[participating]
    indices = old_indices[participating]
    counts = valid_sampled_counts[indices]
    if np.any(counts < 1) or np.any(counts > previous_drafts[indices] + 1):
        raise ValueError("sampled count must be in 1..previous draft count + 1")
    result[current_rows] = previous_computed[indices] + counts
    if np.any(result < 0):
        raise ValueError("corrected frontier exceeds the int64 range")
    return result


def _frozen_copy(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class DelayedD2HBatch:
    epoch: int
    pool_slots: np.ndarray
    generations: np.ndarray
    source_token_slots: np.ndarray
    destination_blocks: np.ndarray
    target_prefixes: np.ndarray


class DelayedD2HPlanner:
    """One in-flight block per request, independent of the DMA backend."""

    def __init__(self, *, capacity: int, block_size: int, hot_tokens: int, host_num_blocks: int):
        for name, value in (
            ("capacity", capacity),
            ("block_size", block_size),
            ("hot_tokens", hot_tokens),
            ("host_num_blocks", host_num_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hot_tokens % (RESIDENT_TAIL_BLOCKS * block_size):
            raise ValueError("hot_tokens must align to the resident tail ring")
        self.capacity = int(capacity)
        self.block_size = int(block_size)
        self.hot_tokens = int(hot_tokens)
        self.host_num_blocks = int(host_num_blocks)
        self.row_tokens = self.hot_tokens + RESIDENT_TAIL_BLOCKS * self.block_size
        self._generations = np.full(capacity, INVALID_GENERATION, dtype=np.int64)
        self._prefixes = np.zeros(capacity, dtype=np.int64)
        self._inflight: DelayedD2HBatch | None = None
        self._next_epoch = 0

    @property
    def inflight(self) -> DelayedD2HBatch | None:
        return self._inflight

    def _slots(self, slots: np.ndarray) -> np.ndarray:
        slots = _integer_array(slots, "pool_slots")
        if np.any(slots < 0) or np.any(slots >= self.capacity):
            raise ValueError("pool slot is outside resident capacity")
        if np.unique(slots).size != slots.size:
            raise ValueError("pool slots must be unique")
        return slots

    def _owners(self, slots: np.ndarray, generations: np.ndarray) -> np.ndarray:
        generations = _integer_array(generations, "generations")
        if generations.shape != slots.shape or np.any(generations < 0):
            raise ValueError("each pool slot needs a nonnegative generation")
        if np.any(self._generations[slots] != generations):
            raise RuntimeError("pool slot generation does not match the current owner")
        return generations

    def _require_idle(self, slots: np.ndarray) -> None:
        if self._inflight is not None and np.any(np.isin(slots, self._inflight.pool_slots)):
            raise RuntimeError("resident slot is referenced by an in-flight D2H batch")

    def bind(self, slots: np.ndarray, generations: np.ndarray, stable_prefixes: np.ndarray) -> None:
        """Bind rows only after the corresponding prompt KV is Host-visible."""
        slots = self._slots(slots)
        generations = _integer_array(generations, "generations")
        prefixes = _integer_array(stable_prefixes, "stable_prefixes")
        if generations.shape != slots.shape or prefixes.shape != slots.shape or np.any(generations < 0):
            raise ValueError("slot, generation and prefix arrays must have matching shapes")
        if np.any(prefixes < 0) or np.any(prefixes % self.block_size):
            raise ValueError("stable prefixes must be nonnegative and block aligned")
        self._require_idle(slots)
        if np.any(self._generations[slots] != INVALID_GENERATION):
            raise RuntimeError("resident slot must be released before binding a new owner")
        self._generations[slots] = generations
        self._prefixes[slots] = prefixes

    def prefixes(self, slots: np.ndarray, generations: np.ndarray) -> np.ndarray:
        slots = self._slots(slots)
        self._owners(slots, generations)
        return self._prefixes[slots].copy()

    def release(self, slots: np.ndarray, generations: np.ndarray) -> None:
        """Reject stale release notifications as well as outstanding DMA."""
        slots = self._slots(slots)
        self._owners(slots, generations)
        self._require_idle(slots)
        self._generations[slots] = INVALID_GENERATION
        self._prefixes[slots] = 0

    def plan(
        self,
        *,
        computed_tokens: np.ndarray,
        pool_slots: np.ndarray,
        generations: np.ndarray,
        block_table: np.ndarray,
        max_write_tokens: np.ndarray,
    ) -> DelayedD2HBatch | None:
        """Freeze a batch without publishing it or advancing any frontier.

        Inputs contain real requests only. ``max_write_tokens`` bounds all KV
        writes in the upcoming target plus draft passes, not just accepted
        tokens. The executor retires before a subsequent plan/row reuse.
        """
        if self._inflight is not None:
            raise RuntimeError("retire the in-flight D2H batch before planning another")
        slots = self._slots(pool_slots)
        owners = self._owners(slots, generations)
        computed = _integer_array(computed_tokens, "computed_tokens")
        writes = _integer_array(max_write_tokens, "max_write_tokens")
        table = _integer_array(block_table, "block_table", ndim=2)
        if computed.shape != slots.shape or writes.shape != slots.shape or table.shape[0] != slots.size:
            raise ValueError("planner inputs must match the real request count")
        if np.any(computed < 0) or np.any(writes < 0):
            raise ValueError("computed tokens and write bounds must be nonnegative")
        prefixes = self._prefixes[slots]
        lag = computed - prefixes
        if np.any(lag < 0):
            raise RuntimeError("computed frontier rolled back behind the completed Host prefix")
        tail_tokens = RESIDENT_TAIL_BLOCKS * self.block_size
        if np.any(lag >= tail_tokens):
            raise RuntimeError("two full blocks are pending; the resident tail may already be overwritten")
        if np.any(writes > tail_tokens - lag):
            raise RuntimeError("upcoming target/draft writes would overlap an unretired source block")
        ready = lag >= self.block_size
        if not np.any(ready):
            return None
        source_prefixes = prefixes[ready]
        logical_blocks = source_prefixes // self.block_size
        if np.any(logical_blocks >= table.shape[1]):
            raise ValueError("source prefix is outside the request block table")
        destinations = table[np.flatnonzero(ready), logical_blocks]
        if np.any(destinations < 0) or np.any(destinations >= self.host_num_blocks):
            raise ValueError("D2H destination is outside the Host allocation")
        if np.unique(destinations).size != destinations.size:
            raise RuntimeError("requests must not write to the same Host block")
        # Dense slots are used until the published prefix reaches hot_tokens.
        # At that transition hot_tokens is ring aligned, so the dense overflow
        # already occupies the first tail block without a relocation.
        offsets = np.where(
            source_prefixes < self.hot_tokens,
            source_prefixes,
            self.hot_tokens + source_prefixes % tail_tokens,
        )
        batch = DelayedD2HBatch(
            epoch=self._next_epoch,
            pool_slots=_frozen_copy(slots[ready]),
            generations=_frozen_copy(owners[ready]),
            source_token_slots=_frozen_copy(slots[ready] * self.row_tokens + offsets),
            destination_blocks=_frozen_copy(destinations),
            target_prefixes=_frozen_copy(source_prefixes + self.block_size),
        )
        self._inflight = batch
        self._next_epoch += 1
        return batch

    def complete(self, epoch: int) -> None:
        """Publish only after the executor confirms DMA and TP completion."""
        batch = self._inflight
        if batch is None or epoch != batch.epoch:
            raise RuntimeError("completion epoch does not match the in-flight batch")
        self._owners(batch.pool_slots, batch.generations)
        if np.any(self._prefixes[batch.pool_slots] + self.block_size != batch.target_prefixes):
            raise RuntimeError("completion is not the next Host block boundary")
        self._prefixes[batch.pool_slots] = batch.target_prefixes
        self._inflight = None
