# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_unified_view import (
    cp_local_page_to_unified_index,
    host_pages_for_tokens,
    prefill_rank_for_cp_rank,
    tokens_per_page,
    unified_host_slot,
)


def test_identity_p1_d1() -> None:
    for src in range(4):
        dest, off = unified_host_slot(
            src, 0, remote_cp_size=1, p_kernel_tokens=128, d_block_tokens=128
        )
        assert (dest, off) == (src, 0)


def test_asymmetric_p8_d1_concatenates_128_pages() -> None:
    """P DCP=8 local page -> consecutive Decode DCP=1 Host pages."""
    slots = [
        unified_host_slot(
            0, rank, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=128
        )
        for rank in range(8)
    ]
    assert slots == [(rank, 0) for rank in range(8)]
    dest, off = unified_host_slot(
        1, 0, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=128
    )
    assert (dest, off) == (8, 0)


def test_symmetric_p8_d8_packs_into_one_scheduler_row() -> None:
    """P DCP=8 into Decode Host row of 1024 tokens (scheduler_block = 128*8)."""
    for rank in range(8):
        dest, off = unified_host_slot(
            0, rank, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=1024
        )
        assert dest == 0
        assert off == rank * 128
    dest, off = unified_host_slot(
        1, 3, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=1024
    )
    assert (dest, off) == (1, 384)


def test_short_prompt_only_low_cp_ranks_land_in_first_pages() -> None:
    dest, off = unified_host_slot(
        0, 0, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=128
    )
    assert (dest, off) == (0, 0)
    dest, _ = unified_host_slot(
        0, 1, remote_cp_size=8, p_kernel_tokens=128, d_block_tokens=128
    )
    assert dest == 1


def test_prefill_rank_mla_tp_equals_dcp() -> None:
    for rank in range(8):
        assert (
            prefill_rank_for_cp_rank(
                rank, prefill_tp_size=8, remote_cp_size=8
            )
            == rank
        )


def test_prefill_rank_rejects_pcp() -> None:
    with pytest.raises(ValueError, match="PCP=1"):
        prefill_rank_for_cp_rank(
            0, prefill_tp_size=8, remote_cp_size=8, remote_pcp_size=2
        )


def test_decode_cp_local_to_unified_matches_write_path() -> None:
    for local in range(3):
        for rank in range(8):
            written, off = unified_host_slot(
                local,
                rank,
                remote_cp_size=8,
                p_kernel_tokens=128,
                d_block_tokens=128,
            )
            assert off == 0
            assert written == cp_local_page_to_unified_index(
                local, rank, dcp_size=8
            )


def test_tokens_per_page_equal_and_packed() -> None:
    assert tokens_per_page(
        page_nbytes=131072, host_block_nbytes=131072, host_block_tokens=128
    ) == 128
    assert tokens_per_page(
        page_nbytes=131072, host_block_nbytes=1048576, host_block_tokens=1024
    ) == 128


@pytest.mark.parametrize(
    ("tokens", "page", "expected"),
    [
        (0, 64, 0),
        (1, 64, 1),
        (64, 64, 1),
        (65, 64, 2),
        (2060, 64, 33),
        (2060, 256, 9),
        (2060, 128, 17),
    ],
)
def test_host_pages_for_tokens_uses_page_arg(tokens: int, page: int, expected: int) -> None:
    """Page length comes from spec/cache_config, not a hardcoded 128."""
    assert host_pages_for_tokens(tokens, host_page_tokens=page) == expected


def test_host_pages_for_tokens_rejects_bad_page() -> None:
    with pytest.raises(ValueError, match="host_page_tokens"):
        host_pages_for_tokens(16, host_page_tokens=0)
