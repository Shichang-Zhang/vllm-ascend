# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for one-shot blockwise DSA Mooncake receives."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

MAX_TCP_PORT = 65535
DSA_GROUPED_TRANSFER_VERSION = 2


class DsaLocalResultKind(str, Enum):
    RECEIVE_COMPLETE = "receive_complete"
    TRANSFER_FAILED = "transfer_failed"


class DsaTransferPhase(str, Enum):
    INDEXER_D2D = "indexer_d2d"
    MAIN_D2RH = "main_d2rh"
    MAIN_D2D = "main_d2d"


class DsaDecodeResultKind(str, Enum):
    PREFIX_READY = "prefix_ready"
    RELEASE_COMPLETE = "release_complete"
    FAILED = "failed"


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _normalize_block_ids(name: str, values: object) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list")
    block_ids = tuple(values)
    for block_id in block_ids:
        _require_nonnegative_integer(name, block_id)
    if len(set(block_ids)) != len(block_ids):
        raise ValueError(f"{name} must not contain duplicates")
    return block_ids


@dataclass(frozen=True, slots=True)
class DsaCacheGroupBlocks:
    """Block IDs in one cache-manager group, independently on either peer."""

    group_id: int
    block_size: int
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_integer("group_id", self.group_id)
        _require_nonnegative_integer("block_size", self.block_size)
        if self.block_size == 0:
            raise ValueError("cache group block_size must be positive")
        object.__setattr__(self, "block_ids", _normalize_block_ids("cache group block_ids", self.block_ids))


def _normalize_cache_groups(groups: Iterable[DsaCacheGroupBlocks]) -> tuple[DsaCacheGroupBlocks, ...]:
    groups = tuple(groups)
    if not all(isinstance(group, DsaCacheGroupBlocks) for group in groups):
        raise TypeError("cache groups must contain DsaCacheGroupBlocks values")
    groups = tuple(sorted(groups, key=lambda group: group.group_id))
    if tuple(group.group_id for group in groups) != tuple(range(len(groups))):
        raise ValueError("cache groups must cover contiguous manager group IDs exactly once")
    return groups


@dataclass(frozen=True, slots=True)
class RemoteEndpoint:
    remote_host: str
    remote_port: int
    remote_engine_id: str

    def __post_init__(self) -> None:
        _require_nonempty_string("remote_host", self.remote_host)
        _require_nonnegative_integer("remote_port", self.remote_port)
        if self.remote_port == 0 or self.remote_port > MAX_TCP_PORT:
            raise ValueError("remote_port must be in range 1..65535")
        _require_nonempty_string("remote_engine_id", self.remote_engine_id)


@dataclass(frozen=True, slots=True)
class RemoteSource:
    remote_request_id: str
    endpoints_by_prefill_rank: tuple[RemoteEndpoint, ...]
    indexer_block_ids: tuple[int, ...]
    main_block_ids: tuple[int, ...]
    remote_dcp_size: int = 1
    remote_pcp_size: int = 1
    remote_ptp_size: int = 1
    remote_pp_size: int = 1
    remote_block_size: int = 1
    num_prompt_blocks: int = 0
    remote_main_group_id: int | None = None
    remote_indexer_group_id: int | None = None
    remote_main_block_size: int = 0
    remote_indexer_block_size: int = 0
    num_computed_tokens: int = 0
    num_external_tokens: int = 0
    protocol_version: int = 1
    cache_groups: tuple[DsaCacheGroupBlocks, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_integer("protocol_version", self.protocol_version)
        if self.protocol_version not in (1, DSA_GROUPED_TRANSFER_VERSION):
            raise ValueError("unsupported DSA transfer protocol version")
        groups = _normalize_cache_groups(self.cache_groups)
        if bool(groups) != (self.protocol_version == DSA_GROUPED_TRANSFER_VERSION):
            raise ValueError("DSA grouped protocol requires complete cache groups; legacy protocol cannot carry groups")
        object.__setattr__(self, "cache_groups", groups)
        for name in ("remote_dcp_size", "remote_pcp_size", "remote_ptp_size", "remote_pp_size", "remote_block_size"):
            _require_nonnegative_integer(name, getattr(self, name))
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "num_computed_tokens",
            "num_external_tokens",
            "remote_main_block_size",
            "num_prompt_blocks",
            "remote_indexer_block_size",
        ):
            _require_nonnegative_integer(name, getattr(self, name))
        for name in ("remote_main_group_id", "remote_indexer_group_id"):
            if getattr(self, name) is not None:
                _require_nonnegative_integer(name, getattr(self, name))
        _require_nonempty_string(
            "remote_request_id",
            self.remote_request_id,
        )
        endpoints = tuple(self.endpoints_by_prefill_rank)
        if not endpoints:
            raise ValueError("endpoints_by_prefill_rank must not be empty")
        if not all(isinstance(endpoint, RemoteEndpoint) for endpoint in endpoints):
            raise TypeError("endpoints_by_prefill_rank must contain RemoteEndpoint values")
        object.__setattr__(self, "endpoints_by_prefill_rank", endpoints)
        object.__setattr__(
            self,
            "indexer_block_ids",
            _normalize_block_ids(
                "indexer_block_ids",
                self.indexer_block_ids,
            ),
        )
        object.__setattr__(
            self,
            "main_block_ids",
            _normalize_block_ids("main_block_ids", self.main_block_ids),
        )
        if groups:
            for kind in ("main", "indexer"):
                group_id = getattr(self, f"remote_{kind}_group_id")
                if group_id is None or group_id >= len(groups):
                    raise ValueError(f"DSA {kind} alias must identify a source cache group")
                if getattr(self, f"{kind}_block_ids") != groups[group_id].block_ids:
                    raise ValueError(f"DSA {kind} alias block IDs disagree with its source group")
                if getattr(self, f"remote_{kind}_block_size") != groups[group_id].block_size:
                    raise ValueError(f"DSA {kind} alias block size disagrees with its source group")


@dataclass(frozen=True, slots=True)
class DsaStepRequest:
    """One PD receive into vLLM-owned Decode blocks."""

    request_id: str
    source: RemoteSource
    main_host_block_ids: tuple[int, ...]
    indexer_hbm_block_ids: tuple[int, ...]
    invalid_block_ids: tuple[int, ...] = ()
    notify_only: bool = False
    generation: int = 0
    destination_groups: tuple[DsaCacheGroupBlocks, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        _require_nonnegative_integer("generation", self.generation)
        if not isinstance(self.source, RemoteSource):
            raise TypeError("source must be RemoteSource")
        if not isinstance(self.notify_only, bool):
            raise TypeError("notify_only must be bool")
        groups = _normalize_cache_groups(self.destination_groups)
        if groups and self.source.protocol_version != DSA_GROUPED_TRANSFER_VERSION:
            raise ValueError("grouped destinations require a versioned grouped source")
        if self.source.protocol_version == DSA_GROUPED_TRANSFER_VERSION and not groups and not self.notify_only:
            raise ValueError("DSA grouped receive requires all destination cache groups")
        object.__setattr__(self, "destination_groups", groups)
        object.__setattr__(self, "invalid_block_ids", _normalize_block_ids("invalid_block_ids", self.invalid_block_ids))
        object.__setattr__(
            self,
            "main_host_block_ids",
            _normalize_block_ids(
                "main_host_block_ids",
                self.main_host_block_ids,
            ),
        )
        object.__setattr__(
            self,
            "indexer_hbm_block_ids",
            _normalize_block_ids(
                "indexer_hbm_block_ids",
                self.indexer_hbm_block_ids,
            ),
        )


@dataclass(frozen=True, slots=True)
class DsaDecodeCommand:
    """Register or release one decode lifetime, independently of PD receive."""

    request_id: str
    generation: int
    host_ready_tokens: int = 0
    release: bool = False

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        _require_nonnegative_integer("generation", self.generation)
        _require_nonnegative_integer("host_ready_tokens", self.host_ready_tokens)
        if self.generation == 0:
            raise ValueError("decode lifetime generation must be positive")
        if not isinstance(self.release, bool):
            raise TypeError("release must be bool")


@dataclass(frozen=True, slots=True)
class DsaConnectorMetadata(KVConnectorMetadata):
    """One-shot receive requests issued to Decode workers."""

    requests: tuple[DsaStepRequest, ...] = ()
    cancelled_requests: tuple[str, ...] = ()
    decode_commands: tuple[DsaDecodeCommand, ...] = ()

    def __post_init__(self) -> None:
        requests_by_id: dict[str, DsaStepRequest] = {}
        for request in self.requests:
            if not isinstance(request, DsaStepRequest):
                raise TypeError("requests must contain DsaStepRequest values")
            existing = requests_by_id.get(request.request_id)
            if existing is not None and existing != request:
                raise ValueError(f"conflicting DSA requests for {request.request_id!r}")
            requests_by_id[request.request_id] = request
        object.__setattr__(
            self,
            "requests",
            tuple(requests_by_id[key] for key in sorted(requests_by_id)),
        )
        commands = {}
        for command in self.decode_commands:
            if not isinstance(command, DsaDecodeCommand):
                raise TypeError("decode_commands must contain DsaDecodeCommand values")
            key = (command.request_id, command.generation, command.release)
            if key in commands and commands[key] != command:
                raise ValueError(f"conflicting decode command for {key}")
            commands[key] = command
        object.__setattr__(self, "decode_commands", tuple(commands[key] for key in sorted(commands)))


@dataclass(frozen=True, slots=True)
class DsaLocalResult:
    request_id: str
    tp_rank: int
    kind: DsaLocalResultKind
    failure_phase: DsaTransferPhase | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        _require_nonnegative_integer("tp_rank", self.tp_rank)
        _require_nonnegative_integer("generation", self.generation)
        if not isinstance(self.kind, DsaLocalResultKind):
            raise TypeError("kind must be DsaLocalResultKind")
        if self.failure_phase is not None and not isinstance(
            self.failure_phase,
            DsaTransferPhase,
        ):
            raise TypeError("failure_phase must be DsaTransferPhase")
        if self.kind is DsaLocalResultKind.TRANSFER_FAILED and self.failure_phase is None:
            raise ValueError("TRANSFER_FAILED requires a failure phase")
        if self.kind is DsaLocalResultKind.RECEIVE_COMPLETE and self.failure_phase is not None:
            raise ValueError("RECEIVE_COMPLETE must not carry a failure phase")

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.request_id, self.generation, self.tp_rank


def _merge_results(
    groups: Iterable[Iterable[DsaLocalResult]],
) -> tuple[DsaLocalResult, ...]:
    merged: dict[tuple[str, int, int], DsaLocalResult] = {}
    for results in groups:
        for result in results:
            if not isinstance(result, DsaLocalResult):
                raise TypeError("results must contain DsaLocalResult values")
            existing = merged.get(result.identity)
            if existing is not None and existing != result:
                raise ValueError(f"conflicting DSA local results for {result.identity}")
            merged[result.identity] = result
    return tuple(merged[key] for key in sorted(merged))


@dataclass(frozen=True, slots=True)
class DsaDecodeResult:
    """A TP rank's Host frontier or drained decode release acknowledgement."""

    request_id: str
    generation: int
    tp_rank: int
    kind: DsaDecodeResultKind
    stable_prefix: int
    error: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string("request_id", self.request_id)
        for name in ("generation", "tp_rank", "stable_prefix"):
            _require_nonnegative_integer(name, getattr(self, name))
        if self.generation == 0:
            raise ValueError("decode lifetime generation must be positive")
        if not isinstance(self.kind, DsaDecodeResultKind):
            raise TypeError("kind must be DsaDecodeResultKind")
        if self.kind is DsaDecodeResultKind.FAILED:
            _require_nonempty_string("error", self.error)
        elif self.error is not None:
            raise ValueError("successful decode results must not carry an error")

    @property
    def identity(self) -> tuple[str, int, int, DsaDecodeResultKind]:
        return self.request_id, self.generation, self.tp_rank, self.kind


def _merge_decode_results(groups: Iterable[Iterable[DsaDecodeResult]]) -> tuple[DsaDecodeResult, ...]:
    merged = {}
    for results in groups:
        for result in results:
            if not isinstance(result, DsaDecodeResult):
                raise TypeError("decode_results must contain DsaDecodeResult values")
            previous = merged.get(result.identity)
            if previous is not None and previous != result:
                raise ValueError(f"conflicting decode result for {result.identity}")
            merged[result.identity] = result
    return tuple(merged[key] for key in sorted(merged))


@dataclass(frozen=True, slots=True)
class DsaWorkerResultMetadata(KVConnectorWorkerMetadata):
    """Rank-aware completion results for one-shot PD receives."""

    results: tuple[DsaLocalResult, ...] = ()
    decode_results: tuple[DsaDecodeResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            _merge_results((self.results,)),
        )
        object.__setattr__(self, "decode_results", _merge_decode_results((self.decode_results,)))

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> DsaWorkerResultMetadata:
        if not isinstance(other, DsaWorkerResultMetadata):
            raise TypeError("aggregate expects DsaWorkerResultMetadata")
        return DsaWorkerResultMetadata(
            _merge_results((self.results, other.results)),
            _merge_decode_results((self.decode_results, other.decode_results)),
        )
