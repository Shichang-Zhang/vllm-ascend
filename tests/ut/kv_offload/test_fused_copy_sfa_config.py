# SPDX-License-Identifier: Apache-2.0
"""Serving constraints for the opt-in LIM and Copy-SFA path."""

from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_config import SparseKVOffloadConfig


def make_config(*, topk=2048, drafts=0, block_size=128, max_model_len=32768, method="mtp"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=topk), max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        cache_config=SimpleNamespace(block_size=block_size),
        speculative_config=SimpleNamespace(num_speculative_tokens=drafts, method=method) if drafts else None,
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
        use_v2_model_runner=False,
    )


def options(**kwargs):
    return dict(enabled=True, fused_op_type="fused_copy_sfa", host_backend="memfabric", topk_buffer_size=8192, **kwargs)


def test_fused_copy_sfa_selects_memfabric_without_v1_overlap():
    config = SparseKVOffloadConfig.from_additional_config(make_config(drafts=3), options())
    assert config.use_fused_copy_sfa
    assert not config.use_fused_overlap
    assert config.topk == 2048


@pytest.mark.parametrize("value", ["copy_sfa", "fused_overlap", None, 1])
def test_invalid_kernel_selector(value):
    with pytest.raises(ValueError, match="fused_op_type"):
        SparseKVOffloadConfig.from_additional_config(SimpleNamespace(), {"fused_op_type": value})


def test_fused_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SparseKVOffloadConfig.from_additional_config(make_config(), options(use_fused_overlap=True))


@pytest.mark.parametrize(
    ("config", "hot", "error"),
    [
        (make_config(topk=128), 8192, "TopK=2048"),
        (make_config(drafts=7), 16128, "1–7"),
        (make_config(drafts=3), 4096, "hot budget"),
        (make_config(), 8320, "256-aligned"),
        (make_config(), 16384, "hot budget"),
        (make_config(block_size=64), 8192, "block_size=128"),
        (make_config(max_model_len=(1 << 21) + 1), 8192, "max_model_len"),
        (make_config(drafts=1, method="dspark"), 8192, "DSpark"),
    ],
)
def test_native_limits_fail_before_cache_allocation(config, hot, error):
    user_config = options()
    user_config["topk_buffer_size"] = hot
    with pytest.raises(ValueError, match=error):
        SparseKVOffloadConfig.from_additional_config(config, user_config)


@pytest.mark.parametrize("hot", [2048, 16128])
def test_hot_budget_boundaries(hot):
    user_config = options()
    user_config["topk_buffer_size"] = hot
    assert SparseKVOffloadConfig.from_additional_config(make_config(), user_config).use_fused_copy_sfa
