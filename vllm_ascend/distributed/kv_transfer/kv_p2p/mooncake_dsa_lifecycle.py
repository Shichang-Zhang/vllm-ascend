# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned decode lifetimes, separate from one-shot PD receives."""

from dataclasses import dataclass, field, replace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaDecodeCommand,
    DsaDecodeResult,
    DsaDecodeResultKind,
)


@dataclass
class _DecodeLease:
    generation: int
    host_ready_tokens: int
    stable_prefix: int
    receive_pending: bool
    receive_failed: bool = False
    terminal: bool = False
    bind_emitted: bool = False
    release_emitted: bool = False
    rank_prefixes: dict[int, int] = field(default_factory=dict)
    released_ranks: set[int] = field(default_factory=set)
    error: str | None = None


class DsaDecodeLedger:
    """Keep Host blocks owned until every TP worker acknowledges release.

    Prefix reports only describe completed copies. A receive completion can
    make the prompt Host-visible, but never releases an active decode lease.
    All methods run on the scheduler thread; no process-global state is used.
    """

    def __init__(self, *, tp_size: int, block_size: int) -> None:
        if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 1:
            raise ValueError("tp_size must be a positive integer")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.ranks = frozenset(range(tp_size))
        self._next_generation = 1
        self._leases: dict[str, _DecodeLease] = {}

    def allocate(self, request_id: str, *, host_ready_tokens: int, receive_pending: bool) -> int:
        existing = self._leases.get(request_id)
        if existing is not None:
            if existing.terminal or existing.error is not None:
                raise RuntimeError("cannot allocate more blocks to a terminal/failed decode lifetime")
            return existing.generation
        generation = self._next_generation
        # Use the wire type's strict identity and integer validation.
        DsaDecodeCommand(request_id, generation, host_ready_tokens)
        if not isinstance(receive_pending, bool):
            raise TypeError("receive_pending must be bool")
        self._leases[request_id] = _DecodeLease(
            generation=generation,
            host_ready_tokens=host_ready_tokens,
            stable_prefix=host_ready_tokens // self.block_size * self.block_size,
            receive_pending=receive_pending,
        )
        self._next_generation += 1
        return generation

    def generation(self, request_id: str) -> int | None:
        lease = self._leases.get(request_id)
        return lease.generation if lease is not None else None

    def is_terminal(self, request_id: str) -> bool:
        lease = self._leases.get(request_id)
        return lease is not None and lease.terminal

    def completed_prefix(self, request_id: str) -> int:
        """Unknown/unallocated requests have no new Host data to publish."""
        lease = self._leases.get(request_id)
        return lease.stable_prefix if lease is not None else 0

    def request_release(self, request_id: str) -> bool:
        lease = self._leases.get(request_id)
        if lease is None:
            return False
        lease.terminal = True
        return True

    def build_commands(self) -> tuple[DsaDecodeCommand, ...]:
        commands = []
        for request_id, lease in sorted(self._leases.items()):
            if not lease.bind_emitted:
                commands.append(DsaDecodeCommand(request_id, lease.generation, lease.host_ready_tokens))
                lease.bind_emitted = True
            if lease.terminal and not lease.release_emitted:
                commands.append(DsaDecodeCommand(request_id, lease.generation, release=True))
                lease.release_emitted = True
        return tuple(commands)

    def _matching(self, request_id: str, generation: int) -> _DecodeLease | None:
        lease = self._leases.get(request_id)
        if lease is None or generation < lease.generation:
            return None
        if generation > lease.generation:
            raise ValueError("decode result generation is ahead of the current owner")
        return lease

    def receive_finished(self, request_id: str, generation: int, *, host_ready_tokens: int, success: bool) -> None:
        """Called only after every rank finished the PD receive, even on failure."""
        lease = self._matching(request_id, generation)
        if lease is None:
            return
        if not lease.receive_pending:
            raise RuntimeError("decode lifetime has no outstanding receive")
        DsaDecodeCommand(request_id, generation, host_ready_tokens)
        if not isinstance(success, bool):
            raise TypeError("receive success must be bool")
        if success:
            if host_ready_tokens < lease.host_ready_tokens:
                raise ValueError("received Host coverage cannot roll back the cached prefix")
            lease.host_ready_tokens = host_ready_tokens
            lease.stable_prefix = max(lease.stable_prefix, host_ready_tokens // self.block_size * self.block_size)
        lease.receive_failed = not success
        lease.receive_pending = False

    def apply_results(self, results: tuple[DsaDecodeResult, ...]) -> None:
        updates: dict[str, _DecodeLease] = {}
        for result in results:
            if not isinstance(result, DsaDecodeResult):
                raise TypeError("decode results must contain DsaDecodeResult values")
            lease = self._matching(result.request_id, result.generation)
            if lease is None:
                continue
            if result.request_id not in updates:
                updates[result.request_id] = replace(
                    lease, rank_prefixes=lease.rank_prefixes.copy(), released_ranks=lease.released_ranks.copy()
                )
            lease = updates[result.request_id]
            if result.tp_rank not in self.ranks:
                raise ValueError(f"illegal Decode TP rank {result.tp_rank}")
            if result.stable_prefix % self.block_size:
                raise ValueError("decode completed prefix must be block aligned")
            if result.kind is DsaDecodeResultKind.FAILED:
                lease.error = result.error
                self._leases[result.request_id].error = result.error
            if lease.error is not None:
                raise RuntimeError(f"decode lifetime failed for {result.request_id!r}: {lease.error}")
            if not lease.bind_emitted:
                raise RuntimeError("decode result arrived before lifetime registration")
            if result.kind is DsaDecodeResultKind.RELEASE_COMPLETE:
                if not lease.terminal or not lease.release_emitted:
                    raise RuntimeError("decode release acknowledgement arrived before a release command")
                if (
                    result.tp_rank in lease.released_ranks
                    and result.stable_prefix != lease.rank_prefixes[result.tp_rank]
                ):
                    raise RuntimeError("duplicate decode release changed its final completed prefix")
                if result.stable_prefix < lease.rank_prefixes.get(result.tp_rank, 0):
                    raise RuntimeError("decode release acknowledgement rolled back its completed prefix")
                lease.released_ranks.add(result.tp_rank)
            elif result.tp_rank in lease.released_ranks:
                # A delayed prefix report from before the final release may
                # be duplicated. It cannot advance a drained lifetime.
                if result.stable_prefix > lease.rank_prefixes[result.tp_rank]:
                    raise RuntimeError("decode prefix advanced after worker release")
            lease.rank_prefixes[result.tp_rank] = max(lease.rank_prefixes.get(result.tp_rank, 0), result.stable_prefix)
        # Publish only after the whole message has been validated, and only
        # through a frontier visible to every consuming rank. Partial or
        # out-of-order rank reports never decrease a published prefix.
        for lease in updates.values():
            if not lease.receive_pending and not lease.receive_failed and set(lease.rank_prefixes) == self.ranks:
                lease.stable_prefix = max(lease.stable_prefix, min(lease.rank_prefixes.values()))
        self._leases.update(updates)

    def take_released(self) -> set[str]:
        released = {
            request_id
            for request_id, lease in self._leases.items()
            if lease.terminal
            and not lease.receive_pending
            and lease.released_ranks == self.ranks
            and lease.error is None
        }
        for request_id in released:
            del self._leases[request_id]
        return released
