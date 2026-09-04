# SPDX-License-Identifier: Apache-2.0
"""Unified Host-pool View for blockwise DSA Main KV.

Prefill DCP shards Main into ``cp_kv_cache_interleave_size`` (typically 128)
token chunks. Decode fused_overlap / Host pool address **global token order**
pages (Decode DCP=1 layout). This module maps one Prefill-CP local page onto
that Host view so:

* P DCP>1 / D DCP=1 (asymmetric): concatenate shards into consecutive Host pages.
* P DCP == D DCP (symmetric): pack shards into one Decode scheduler row at
  token offsets ``rank * kernel``, which fused_overlap still reads as a
  contiguous Host block.
* P DCP=1 / D DCP=1: identity (``dest_index=src_index``, offset 0).

Indexer stays rank-local D2D. Main uses this view: after Host pages
are allocated, each Decode TP writes the CP shards it owns (disjoint
Host token ranges) through the shared pool.
"""

from __future__ import annotations

import math


def prefill_rank_for_cp_rank(
    cp_rank: int,
    *,
    prefill_tp_size: int,
    remote_cp_size: int,
    remote_pcp_size: int = 1,
) -> int:
    """MLA Main: CP rank ``r`` lives on Prefill TP ``r`` (replicas at ``r+k*cp``).

    Requires Prefill PCP=1. ``prefill_tp_size`` must be a multiple of
    ``remote_cp_size`` (DCP must divide TP).
    """
    if remote_pcp_size != 1:
        raise ValueError(
            "DSA unified view MAIN gather requires Prefill PCP=1, "
            f"got remote_pcp_size={remote_pcp_size}"
        )
    if remote_cp_size <= 0:
        raise ValueError(f"remote_cp_size must be positive, got {remote_cp_size}")
    if prefill_tp_size <= 0:
        raise ValueError(f"prefill_tp_size must be positive, got {prefill_tp_size}")
    if prefill_tp_size % remote_cp_size != 0:
        raise ValueError(
            "prefill TP must be a multiple of remote CP: "
            f"tp={prefill_tp_size} cp={remote_cp_size}"
        )
    if cp_rank < 0 or cp_rank >= remote_cp_size:
        raise ValueError(
            f"cp_rank out of range: rank={cp_rank}, remote_cp_size={remote_cp_size}"
        )
    return cp_rank


def decode_tp_owned_cp_ranks(
    decode_tp_rank: int,
    *,
    decode_tp_size: int,
    prefill_tp_size: int,
    remote_cp_size: int,
    remote_pcp_size: int = 1,
) -> tuple[int, ...]:
    """CP ranks this Decode TP should MAIN_D2RH into the shared Host pool.

    Prefill CP ``r`` lives on Prefill TP ``prefill_rank_for_cp_rank(r)``.
    Decode TP ``i`` owns Prefill ranks ``[i * stride, (i + 1) * stride)``
    where ``stride = prefill_tp_size // decode_tp_size`` (same pairing as
    Indexer D2D ``leader_rank``). Live 1P1D P_tp=D_tp=DCP=8 is TP_i ↔ cp_i;
    P8/D1 keeps Decode TP0 gathering every CP.
    """
    if decode_tp_size <= 0:
        raise ValueError(f"decode_tp_size must be positive, got {decode_tp_size}")
    if decode_tp_rank < 0 or decode_tp_rank >= decode_tp_size:
        raise ValueError(
            f"decode_tp_rank out of range: rank={decode_tp_rank}, "
            f"decode_tp_size={decode_tp_size}"
        )
    if prefill_tp_size <= 0:
        raise ValueError(f"prefill_tp_size must be positive, got {prefill_tp_size}")
    if prefill_tp_size % decode_tp_size != 0:
        raise ValueError(
            "prefill TP must be a multiple of decode TP: "
            f"P_tp={prefill_tp_size} D_tp={decode_tp_size}"
        )
    stride = prefill_tp_size // decode_tp_size
    leader = decode_tp_rank * stride
    owned = []
    for cp_rank in range(remote_cp_size):
        prefill_rank = prefill_rank_for_cp_rank(
            cp_rank,
            prefill_tp_size=prefill_tp_size,
            remote_cp_size=remote_cp_size,
            remote_pcp_size=remote_pcp_size,
        )
        if leader <= prefill_rank < leader + stride:
            owned.append(cp_rank)
    return tuple(owned)


def unified_host_slot(
    src_index: int,
    cp_rank: int,
    *,
    remote_cp_size: int,
    p_kernel_tokens: int,
    d_block_tokens: int,
) -> tuple[int, int]:
    """Map Prefill CP-local page ``src_index`` on ``cp_rank`` to Host (block, offset).

    Global tokens for that page start at::

        src_index * remote_cp_size * p_kernel_tokens + cp_rank * p_kernel_tokens

    Decode Host rows are ``d_block_tokens`` long in the same global order.
    """
    if src_index < 0:
        raise ValueError(f"src_index must be nonnegative, got {src_index}")
    if remote_cp_size <= 0 or p_kernel_tokens <= 0 or d_block_tokens <= 0:
        raise ValueError(
            "remote_cp_size, p_kernel_tokens, and d_block_tokens must be positive: "
            f"cp={remote_cp_size} p_kernel={p_kernel_tokens} d_block={d_block_tokens}"
        )
    if cp_rank < 0 or cp_rank >= remote_cp_size:
        raise ValueError(
            f"cp_rank out of range: rank={cp_rank}, remote_cp_size={remote_cp_size}"
        )
    if (remote_cp_size * p_kernel_tokens) % d_block_tokens != 0 and d_block_tokens % p_kernel_tokens != 0:
        raise ValueError(
            "Host block tokens must tile Prefill kernel pages: "
            f"remote_cp={remote_cp_size} p_kernel={p_kernel_tokens} "
            f"d_block={d_block_tokens}"
        )
    global_start = src_index * remote_cp_size * p_kernel_tokens + cp_rank * p_kernel_tokens
    dest_index = global_start // d_block_tokens
    token_offset = global_start % d_block_tokens
    if token_offset % p_kernel_tokens != 0:
        raise ValueError(
            "unified view token_offset must be a Prefill kernel multiple: "
            f"offset={token_offset} p_kernel={p_kernel_tokens}"
        )
    return dest_index, token_offset


def cp_local_page_to_unified_index(
    local_block: int,
    dcp_rank: int,
    *,
    dcp_size: int,
) -> int:
    """Decode-side Host page index for a CP-local device page (128-token Host rows).

    Inverse of writing ``unified_host_slot`` when ``d_block_tokens == p_kernel``.
    fused_overlap with ``cache_config.block_size`` pages uses this when Decode
    DCP>1 still carries CP-local block ids.
    """
    if local_block < 0 or dcp_rank < 0:
        raise ValueError(
            f"local_block and dcp_rank must be nonnegative: "
            f"local_block={local_block} dcp_rank={dcp_rank}"
        )
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if dcp_rank >= dcp_size:
        raise ValueError(f"dcp_rank={dcp_rank} >= dcp_size={dcp_size}")
    return local_block * dcp_size + dcp_rank


def host_pages_for_tokens(num_tokens: int, *, host_page_tokens: int) -> int:
    """How many Host pages cover ``num_tokens``.

    ``host_page_tokens`` is the Main group's ``kv_cache_spec.block_size``
    (Host ``layout.block_size`` / ``cache_config.block_size``). Not a
    DCP-virtual scheduler row, and not a hardcoded 128.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be nonnegative, got {num_tokens}")
    if host_page_tokens <= 0:
        raise ValueError(f"host_page_tokens must be positive, got {host_page_tokens}")
    if num_tokens == 0:
        return 0
    return math.ceil(num_tokens / host_page_tokens)


def tokens_per_page(*, page_nbytes: int, host_block_nbytes: int, host_block_tokens: int) -> int:
    """Infer Prefill kernel tokens from handshake page bytes vs Host row bytes."""
    if page_nbytes <= 0 or host_block_nbytes <= 0 or host_block_tokens <= 0:
        raise ValueError(
            "page_nbytes, host_block_nbytes, and host_block_tokens must be positive"
        )
    if host_block_nbytes % page_nbytes != 0 and page_nbytes % host_block_nbytes != 0:
        raise ValueError(
            "handshake page bytes must divide Host row bytes (or vice versa): "
            f"page={page_nbytes} host_block={host_block_nbytes}"
        )
    if host_block_nbytes == page_nbytes:
        return host_block_tokens
    if host_block_nbytes > page_nbytes:
        ratio = host_block_nbytes // page_nbytes
        if host_block_tokens % ratio != 0:
            raise ValueError(
                "Host tokens not divisible by page packing ratio: "
                f"host_tokens={host_block_tokens} ratio={ratio}"
            )
        return host_block_tokens // ratio
    ratio = page_nbytes // host_block_nbytes
    return host_block_tokens * ratio
