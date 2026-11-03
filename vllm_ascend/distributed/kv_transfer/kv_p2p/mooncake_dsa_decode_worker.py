# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Main-thread decode ownership, independent of Mooncake receive threads."""

from dataclasses import dataclass, replace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaConnectorMetadata,
    DsaDecodeCommand,
    DsaDecodeResult,
    DsaDecodeResultKind,
    DsaLocalResult,
    DsaLocalResultKind,
    DsaStepRequest,
)


@dataclass(frozen=True)
class DsaDecodeWorkerRequest:
    request_id: str
    generation: int
    host_ready_tokens: int
    stable_prefix: int
    resident_bound: bool


@dataclass
class _WorkerLease:
    command: DsaDecodeCommand
    host_ready_tokens: int
    stable_prefix: int
    receive_command: DsaStepRequest | None = None
    receive_pending: bool = False
    receive_dispatched: bool = False
    receive_failed: bool = False
    resident_bound: bool = False
    terminal: bool = False
    error: str | None = None

    def snapshot(self) -> DsaDecodeWorkerRequest:
        return DsaDecodeWorkerRequest(
            self.command.request_id,
            self.command.generation,
            self.host_ready_tokens,
            self.stable_prefix,
            self.resident_bound,
        )

    @property
    def host_receive_pending(self) -> bool:
        return self.receive_pending and self.receive_command is not None and not self.receive_command.notify_only


class DsaDecodeWorkerState:
    """Only the worker's execution thread may mutate this state.

    Receive callbacks enter through a queue owned by MooncakeConnectorWorker.
    The runner binds residency after restoring verified Host data, and confirms
    release only after its D2H/compute users have drained. No receive callback
    may recycle rows or acknowledge a decode release.
    """

    def __init__(self, *, tp_rank: int, block_size: int) -> None:
        if isinstance(tp_rank, bool) or not isinstance(tp_rank, int) or tp_rank < 0:
            raise ValueError("tp_rank must be a nonnegative integer")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.tp_rank = tp_rank
        self.block_size = block_size
        self._leases: dict[str, _WorkerLease] = {}
        self._results: dict[tuple, DsaDecodeResult] = {}
        # The scheduler allocates globally increasing generations. A single
        # watermark rejects replayed registrations after release, without an
        # unbounded set of request-ID tombstones. New commands in one metadata
        # batch may be ordered by request ID rather than generation.
        self._highest_generation = 0

    def apply_metadata(self, metadata: DsaConnectorMetadata) -> None:
        leases = {key: replace(value) for key, value in self._leases.items()}
        highest = self._highest_generation
        for command in metadata.decode_commands:
            if command.release:
                continue
            existing = leases.get(command.request_id)
            if existing is not None:
                if command.generation < existing.command.generation:
                    continue
                if command != existing.command:
                    raise RuntimeError("conflicting decode registration before the previous owner was drained")
                continue
            if command.generation <= self._highest_generation:
                continue
            leases[command.request_id] = _WorkerLease(
                command,
                command.host_ready_tokens,
                command.host_ready_tokens // self.block_size * self.block_size,
            )
            highest = max(highest, command.generation)
        for receive in metadata.requests:
            if receive.generation == 0 and receive.notify_only:
                # The scheduler can cancel before allocating any destination
                # blocks. This only tells P to release its send reservation.
                if receive.main_host_block_ids or receive.indexer_hbm_block_ids:
                    raise RuntimeError("unallocated notification must not reference destination blocks")
                continue
            lease = leases.get(receive.request_id)
            if lease is None or receive.generation < lease.command.generation:
                if 0 < receive.generation <= self._highest_generation:
                    continue
                raise RuntimeError("DSA receive has no matching decode registration")
            if receive.generation != lease.command.generation:
                raise RuntimeError("DSA receive generation differs from decode ownership")
            if lease.receive_command is not None:
                if lease.receive_command != receive:
                    raise RuntimeError("conflicting receive within a decode generation")
                continue
            if lease.resident_bound:
                raise RuntimeError("cannot receive into a resident decode lifetime")
            lease.receive_command = receive
            # Even notify-only cleanup must leave the receive thread before
            # this generation can be acknowledged and its textual ID reused.
            lease.receive_pending = True
        for command in metadata.decode_commands:
            if not command.release:
                continue
            lease = leases.get(command.request_id)
            if lease is None or command.generation < lease.command.generation:
                if command.generation <= self._highest_generation:
                    continue
                raise RuntimeError("decode release has no registered owner")
            if command.generation != lease.command.generation:
                raise RuntimeError("decode release generation differs from current owner")
            lease.terminal = True
        self._leases = leases
        self._highest_generation = highest

    def should_dispatch_receive(self, command: DsaStepRequest) -> bool:
        lease = self._leases.get(command.request_id)
        if command.generation == 0 and command.notify_only:
            return lease is None
        return (
            lease is not None
            and lease.receive_pending
            and not lease.receive_dispatched
            and lease.receive_command == command
            and lease.command.generation == command.generation
        )

    def claim_receive(self, command: DsaStepRequest) -> None:
        """Reserve dispatch on the execution thread before starting endpoint tasks.

        The receive callback may remove its active-thread entry before its
        queued completion is consumed. That gap must not permit a second
        dispatch of the same generation.
        """
        if not self.should_dispatch_receive(command):
            raise RuntimeError("decode receive has already been dispatched or is stale")
        if command.generation > 0:
            self._leases[command.request_id].receive_dispatched = True

    def is_terminal(self, request_id: str) -> bool:
        lease = self._leases.get(request_id)
        return lease is not None and lease.terminal

    def _matching(self, request_id: str, generation: int) -> _WorkerLease | None:
        lease = self._leases.get(request_id)
        if lease is None or generation < lease.command.generation:
            return None
        if generation != lease.command.generation:
            raise RuntimeError("worker completion generation is ahead of current owner")
        return lease

    def receive_finished(self, command: DsaStepRequest, result: DsaLocalResult) -> None:
        lease = self._matching(command.request_id, command.generation)
        if lease is None:
            return
        if (result.request_id, result.generation, result.tp_rank) != (
            command.request_id,
            command.generation,
            self.tp_rank,
        ):
            raise ValueError("receive completion does not match the local decode owner")
        if command != lease.receive_command or not lease.receive_pending:
            raise RuntimeError("receive completion has no outstanding command")
        success = result.kind is DsaLocalResultKind.RECEIVE_COMPLETE
        if success and not command.notify_only:
            ready = command.source.num_computed_tokens + command.source.num_external_tokens
            if ready < lease.host_ready_tokens:
                raise RuntimeError("received Host coverage rolled back the verified cached prefix")
            lease.host_ready_tokens = ready
            lease.stable_prefix = max(lease.stable_prefix, ready // self.block_size * self.block_size)
        lease.receive_pending = False
        # Notification failure cannot invalidate KV obtained through APC.
        lease.receive_failed = not success and not command.notify_only

    def acquire(self, request_id: str) -> DsaDecodeWorkerRequest:
        lease = self._leases.get(request_id)
        if lease is None:
            raise RuntimeError("scheduled decode request has no registered owner")
        if lease.terminal or lease.receive_failed or lease.error is not None:
            raise RuntimeError("scheduled decode request is terminal or failed")
        if lease.host_receive_pending:
            raise RuntimeError("scheduled decode request is still receiving Host KV")
        return lease.snapshot()

    def bind_resident(self, request_id: str, generation: int) -> None:
        snapshot = self.acquire(request_id)
        if snapshot.generation != generation:
            raise RuntimeError("resident generation differs from decode ownership")
        self._leases[request_id].resident_bound = True

    def _report(self, lease: _WorkerLease, kind: DsaDecodeResultKind) -> None:
        result = DsaDecodeResult(
            lease.command.request_id,
            lease.command.generation,
            self.tp_rank,
            kind,
            lease.stable_prefix,
            lease.error if kind is DsaDecodeResultKind.FAILED else None,
        )
        # Multiple completions before serialization coalesce to the latest
        # completed frontier, avoiding conflicting results in one message.
        self._results[result.identity] = result

    def publish(self, request_id: str, generation: int, stable_prefix: int) -> None:
        lease = self._matching(request_id, generation)
        if lease is None:
            return
        if not lease.resident_bound or lease.host_receive_pending or lease.receive_failed or lease.error is not None:
            raise RuntimeError("cannot publish from an unready or failed resident owner")
        if isinstance(stable_prefix, bool) or not isinstance(stable_prefix, int) or stable_prefix < 0:
            raise ValueError("completed prefix must be a nonnegative integer")
        if stable_prefix % self.block_size:
            raise ValueError("completed prefix must be block aligned")
        # The first forward may recompute the last received prompt token.
        # Its planner can start one block behind the already verified Host
        # prompt, but must never retract that existing publication.
        if stable_prefix > lease.stable_prefix:
            lease.stable_prefix = stable_prefix
            self._report(lease, DsaDecodeResultKind.PREFIX_READY)

    def releasable(self) -> tuple[DsaDecodeWorkerRequest, ...]:
        return tuple(
            lease.snapshot()
            for _, lease in sorted(self._leases.items())
            if lease.terminal and not lease.receive_pending and lease.error is None
        )

    def confirm_drained(self, request_id: str, generation: int) -> None:
        lease = self._matching(request_id, generation)
        if lease is None:
            return
        if not lease.terminal or lease.receive_pending or lease.error is not None:
            raise RuntimeError("cannot release a live, receiving or failed decode owner")
        self._report(lease, DsaDecodeResultKind.RELEASE_COMPLETE)
        del self._leases[request_id]

    def fail(self, request_id: str, generation: int, error: str) -> None:
        lease = self._matching(request_id, generation)
        if lease is None:
            return
        if not isinstance(error, str) or not error:
            raise ValueError("decode failure requires an error message")
        lease.error = error
        self._report(lease, DsaDecodeResultKind.FAILED)

    def take_results(self) -> tuple[DsaDecodeResult, ...]:
        results = tuple(self._results.values())
        self._results.clear()
        return results
