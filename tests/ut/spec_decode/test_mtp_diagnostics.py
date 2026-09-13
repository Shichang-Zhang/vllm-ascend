# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_MODULE_PATH = Path(__file__).parents[3] / "vllm_ascend" / "spec_decode" / "mtp_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "mtp_diagnostics_cpu_test_target",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
assert_mtp_index_cache_state = _MODULE.assert_mtp_index_cache_state
assert_mtp_topk_rows_front_aligned = _MODULE.assert_mtp_topk_rows_front_aligned


class FakeMTPAttention(nn.Module):
    def __init__(self, wrapper_buffer, impl_buffer, *, wrapper_skip, impl_skip):
        super().__init__()
        self.skip_topk = wrapper_skip
        self.topk_indices_buffer = wrapper_buffer
        self.mla_attn = SimpleNamespace(
            impl=SimpleNamespace(
                skip_topk=impl_skip,
                topk_indices_buffer=impl_buffer,
            )
        )


def _model(attention: nn.Module) -> nn.Module:
    model = nn.Module()
    model.add_module("attention", attention)
    return model


def test_asserts_when_skip_topk_does_not_reach_inner_impl():
    buffer = torch.arange(16, dtype=torch.int32).reshape(4, 4)
    model = _model(
        FakeMTPAttention(
            buffer,
            buffer,
            wrapper_skip=True,
            impl_skip=False,
        )
    )

    with pytest.raises(AssertionError, match="gate did not reach"):
        assert_mtp_index_cache_state(
            model,
            expected_skip_topk=True,
            phase="draft steps 1+",
        )


def test_asserts_when_shared_buffer_does_not_reach_inner_impl():
    wrapper_buffer = torch.zeros(4, 4, dtype=torch.int32)
    impl_buffer = torch.ones(4, 4, dtype=torch.int32)
    model = _model(
        FakeMTPAttention(
            wrapper_buffer,
            impl_buffer,
            wrapper_skip=False,
            impl_skip=False,
        )
    )

    with pytest.raises(AssertionError, match="shared top-k buffer did not reach"):
        assert_mtp_index_cache_state(
            model,
            expected_skip_topk=False,
            phase="draft step 0",
        )


def test_asserts_when_step_zero_rows_are_not_front_aligned():
    buffer = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    model = _model(
        FakeMTPAttention(
            buffer,
            buffer,
            wrapper_skip=False,
            impl_skip=False,
        )
    )

    with pytest.raises(AssertionError, match="not front-aligned"):
        assert_mtp_topk_rows_front_aligned(
            model,
            torch.tensor([3, 7], dtype=torch.int32),
        )


def test_accepts_forwarded_state_and_compacted_rows():
    buffer = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    selected = buffer[torch.tensor([3, 7])].clone()
    buffer[:2].copy_(selected)
    model = _model(
        FakeMTPAttention(
            buffer,
            buffer,
            wrapper_skip=True,
            impl_skip=True,
        )
    )

    assert_mtp_index_cache_state(
        model,
        expected_skip_topk=True,
        phase="draft steps 1+",
    )
    assert_mtp_topk_rows_front_aligned(
        model,
        torch.tensor([3, 7], dtype=torch.int32),
    )
