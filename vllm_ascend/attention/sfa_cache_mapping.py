# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate SFA logical query positions to an independent cache group."""

import torch


def map_sfa_indexer_slots(
    block_table: torch.Tensor,
    query_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    main_slots: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Use device lengths for target and MTP, retaining all ignored writes.

    No CPU mirror of accepted lengths or device scalar reads are needed.
    Fixed-shape operations are captured with the forward graph, so later
    replays read current lengths, request boundaries, and physical block IDs.
    """
    requests = query_ends.numel()
    if requests == 0 or block_table.shape[0] < requests or block_table.shape[1] == 0:
        raise ValueError("SFA indexer table must cover every metadata request")
    tokens = torch.arange(main_slots.numel(), dtype=torch.int32, device=main_slots.device)
    rows = torch.searchsorted(query_ends.contiguous(), tokens, right=True)
    valid = (rows < requests) & (main_slots >= 0)
    rows = rows.clamp(max=requests - 1).long()
    # seq_len - query_end + token_index is the absolute logical position;
    # subtracting query_start from both query length and offset cancels out.
    positions = seq_lens[rows].long() - query_ends[rows].long() + tokens
    valid = valid & (positions >= 0) & (positions < block_table.shape[1] * block_size)
    columns = torch.div(positions, block_size, rounding_mode="floor").clamp(0, block_table.shape[1] - 1)
    blocks = block_table[rows, columns].long()
    valid = valid & (blocks >= 0)
    return torch.where(valid, blocks * block_size + positions.remainder(block_size), -1)
