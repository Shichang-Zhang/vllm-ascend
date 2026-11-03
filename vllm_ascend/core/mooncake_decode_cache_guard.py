# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate APC publication and block release on Mooncake decode completion.

Install on the scheduler's coordinator instance so both allocate_slots' direct
publication and explicit cache_blocks calls use the same frontier. This also
covers hybrid coordinators and synchronous/asynchronous recompute schedulers.
The upstream replacement for this adapter would be connector-owned publication
and release fences in KVCacheManager, rather than per-scheduler output hooks.
"""

from typing import Protocol

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request


class DecodeCompletionSource(Protocol):
    def get_completed_host_prefix(self, request_id: str) -> int | None: ...

    def owns_decode_blocks(self, request_id: str) -> bool: ...


class MooncakeDecodeCacheGuard:
    def __init__(self, coordinator: KVCacheCoordinator, connector: DecodeCompletionSource) -> None:
        self.connector = connector
        self._cache_blocks = coordinator.cache_blocks
        self._free = coordinator.free
        self._pop_blocks_for_free = coordinator.pop_blocks_for_free

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        prefix = self.connector.get_completed_host_prefix(request.request_id)
        if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
            raise RuntimeError("Mooncake delayed D2H requires a valid completed Host prefix")
        self._cache_blocks(request, min(num_computed_tokens, prefix))

    def _require_released(self, request_id: str) -> None:
        if self.connector.owns_decode_blocks(request_id):
            raise RuntimeError(
                f"Mooncake delayed D2H still owns blocks for {request_id!r}; "
                "wait for decode release acknowledgements before freeing blocks. "
                "Preemption/resume is not supported in this mode."
            )

    def free(self, request_id: str) -> None:
        self._require_released(request_id)
        self._free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._require_released(request_id)
        return self._pop_blocks_for_free(request_id)


def install_mooncake_decode_cache_guard(scheduler) -> None:
    """Called after scheduler construction, before any request is admitted."""
    connector = scheduler.connector
    if getattr(connector, "delayed_d2h_enabled", False) is not True:
        return
    coordinator = scheduler.kv_cache_manager.coordinator
    if getattr(coordinator, "_mooncake_decode_cache_guard", None) is not None:
        return
    guard = MooncakeDecodeCacheGuard(coordinator, connector)
    coordinator.cache_blocks = guard.cache_blocks
    coordinator.free = guard.free
    coordinator.pop_blocks_for_free = guard.pop_blocks_for_free
    coordinator._mooncake_decode_cache_guard = guard
