# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn


def _iter_mtp_index_cache_impls(
    draft_model: nn.Module,
) -> Iterator[tuple[str, nn.Module, Any]]:
    """Yield Ascend MLA wrappers and their non-Module sparse implementations."""
    for name, module in draft_model.named_modules():
        impl = getattr(getattr(module, "mla_attn", None), "impl", None)
        if impl is not None and hasattr(impl, "skip_topk"):
            yield name, module, impl


def assert_mtp_index_cache_state(
    draft_model: nn.Module,
    *,
    expected_skip_topk: bool,
    phase: str,
) -> None:
    """Fail if MTP's public index-cache state did not reach the real impl."""
    found = False
    for name, wrapper, impl in _iter_mtp_index_cache_impls(draft_model):
        found = True
        actual_skip_topk = bool(impl.skip_topk)
        if actual_skip_topk != expected_skip_topk:
            raise AssertionError(
                "MTP index-cache gate did not reach the Ascend MLA implementation: "
                f"phase={phase}, module={name}, expected_skip_topk="
                f"{expected_skip_topk}, actual_skip_topk={actual_skip_topk}."
            )

        wrapper_buffer = getattr(wrapper, "topk_indices_buffer", None)
        impl_buffer = getattr(impl, "topk_indices_buffer", None)
        if impl_buffer is None:
            raise AssertionError(f"MTP index-cache implementation has no top-k buffer: phase={phase}, module={name}.")
        if wrapper_buffer is not impl_buffer:
            raise AssertionError(
                "MTP shared top-k buffer did not reach the Ascend MLA implementation: "
                f"phase={phase}, module={name}, wrapper_buffer_id="
                f"{id(wrapper_buffer)}, impl_buffer_id={id(impl_buffer)}."
            )

    if not found:
        raise AssertionError(
            f"MTP index sharing is enabled, but no Ascend sparse MLA implementation was found during {phase}."
        )


def assert_mtp_topk_rows_front_aligned(
    draft_model: nn.Module,
    token_indices_to_sample: torch.Tensor,
) -> None:
    """Fail when step-0 top-k rows are not ready for request-major reuse."""
    num_rows = token_indices_to_sample.numel()
    if num_rows == 0:
        return

    seen_buffers: set[int] = set()
    for name, _, impl in _iter_mtp_index_cache_impls(draft_model):
        buffer = getattr(impl, "topk_indices_buffer", None)
        if buffer is None or id(buffer) in seen_buffers:
            continue
        seen_buffers.add(id(buffer))
        indices = token_indices_to_sample.to(device=buffer.device, dtype=torch.int64)
        if indices.numel() > buffer.shape[0]:
            raise AssertionError(
                "MTP sample-index count exceeds the top-k buffer: "
                f"module={name}, sample_rows={indices.numel()}, "
                f"buffer_rows={buffer.shape[0]}."
            )
        expected_rows = buffer.index_select(0, indices)
        actual_rows = buffer[:num_rows]
        # This diagnostic intentionally synchronizes device state so a silent
        # accuracy failure becomes an actionable runtime assertion.
        if not torch.equal(actual_rows, expected_rows):
            raise AssertionError(
                "MTP step-0 top-k rows are not front-aligned for steps 1+: "
                f"module={name}, token_indices_to_sample="
                f"{token_indices_to_sample}. Compact the selected rows before "
                "enabling skip_topk."
            )


def assert_greedy_rejection_matches_reference(
    actual_token_ids: torch.Tensor,
    reference_token_ids: torch.Tensor,
) -> None:
    """Synchronously compare Triton rejection output with its safe reference."""
    if actual_token_ids.shape != reference_token_ids.shape:
        raise AssertionError(
            "[DEBUG] Ascend greedy rejection output shape differs from the "
            "PyTorch reference: "
            f"actual={tuple(actual_token_ids.shape)}, "
            f"reference={tuple(reference_token_ids.shape)}."
        )
    # torch.equal returns a Host bool and intentionally synchronizes the NPU.
    if not torch.equal(actual_token_ids, reference_token_ids):
        mismatch = actual_token_ids != reference_token_ids
        mismatch_count = int(mismatch.sum().item())
        first_mismatch = mismatch.nonzero(as_tuple=False)[0].tolist()
        row, column = first_mismatch
        raise AssertionError(
            "[DEBUG] Ascend Triton greedy rejection differs from the safe "
            "PyTorch reference; this run may be affected by the lane-zero "
            "prefix read fixed by 8e8d1cb28: "
            f"mismatches={mismatch_count}, first=({row}, {column}), "
            f"actual_token={int(actual_token_ids[row, column].item())}, "
            f"reference_token={int(reference_token_ids[row, column].item())}."
        )


def assert_rejection_kernel_completed(
    kernel_name: str,
    synchronize: Callable[[], None],
) -> None:
    """Surface an asynchronous rejection-kernel device fault at its caller."""
    try:
        synchronize()
    except RuntimeError as error:
        raise AssertionError(
            "[DEBUG] Ascend rejection sampling failed while synchronizing "
            f"immediately after {kernel_name}; this run may have triggered "
            "the out-of-bounds prefix read fixed by 8e8d1cb28."
        ) from error
