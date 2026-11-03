# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt vLLM's single-group load-failure scan for mixed SFA caches.

Mooncake DSA requires failure_policy=fail. All groups share one block pool,
but have independent block tables. Failing any group invalidates the request
at its earliest failed logical block. This instance adapter can be removed
when upstream's load-failure handler supports full-attention cache groups.
It performs no device work and runs only after a transfer failure.
"""

from collections.abc import Iterable

from vllm.v1.request import Request


class MooncakeGroupFailureHandler:
    def __init__(self, manager, block_size: int) -> None:
        self.manager = manager
        self.block_size = block_size

    def update_requests(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected = set()
        tokens = 0
        evict = set()
        for request in requests:
            computed = max(0, request.num_computed_tokens - num_scheduled_tokens.get(request.request_id, 0))
            computed_blocks = (computed + self.block_size - 1) // self.block_size
            groups = self.manager.get_block_ids(request.request_id)
            first_bad = min(
                (
                    ordinal
                    for blocks in groups
                    for ordinal, block_id in enumerate(blocks[:computed_blocks])
                    if block_id in invalid_block_ids
                ),
                default=None,
            )
            if first_bad is None:
                continue
            # All affected requests fail, including those sharing a bad cached
            # block. There is no recompute owner to keep other requests alive.
            affected.add(request.request_id)
            request.num_computed_tokens = first_bad * self.block_size
            tokens += computed - request.num_computed_tokens
            if evict_blocks:
                evict.update(block_id for blocks in groups for block_id in blocks[first_bad:])
        return affected, tokens, evict


def install_mooncake_group_failure_handler(scheduler) -> None:
    if getattr(scheduler.connector, "dsa_grouped_transfer_enabled", False) is not True:
        return
    if getattr(scheduler, "_mooncake_group_failure_handler", None) is not None:
        return
    if scheduler.recompute_kv_load_failures:
        raise ValueError("Mooncake grouped DSA requires kv_load_failure_policy='fail'")
    manager = scheduler.kv_cache_manager
    if any(group.kv_cache_spec.block_size != scheduler.block_size for group in manager.kv_cache_config.kv_cache_groups):
        raise ValueError("Mooncake grouped DSA requires equal scheduler and cache-group block sizes")
    handler = MooncakeGroupFailureHandler(manager, scheduler.block_size)
    scheduler._update_requests_with_invalid_blocks = handler.update_requests
    scheduler._mooncake_group_failure_handler = handler
