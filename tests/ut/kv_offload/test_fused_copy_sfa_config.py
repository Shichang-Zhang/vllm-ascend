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
        kv_transfer_config=SimpleNamespace(
            is_kv_consumer=True,
            kv_connector="MooncakeConnectorV1",
            kv_connector_extra_config={"dsa_pd_offload": True},
        ),
        use_v2_model_runner=False,
    )


def options(**kwargs):
    return dict(enabled=True, fused_op_type="fused_copy_sfa", host_backend="mooncake", topk_buffer_size=8192, **kwargs)


def test_fused_copy_sfa_selects_mooncake_without_v1_overlap():
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


def test_offload_migration_defaults_preserve_existing_modes():
    for kernel_options in ({"fused_op_type": "fused_copy_sfa"}, {"use_fused_overlap": True}):
        config = SparseKVOffloadConfig(make_config(), dict(enabled=True, host_backend="mooncake", **kernel_options))
        assert not config.delayed_d2h


@pytest.mark.parametrize("connector", ["SfaRemoteD2HConnector", "MultiConnector"])
def test_disabled_migration_preserves_existing_non_mooncake_connectors(connector):
    vllm_config = make_config()
    vllm_config.kv_transfer_config.kv_connector = connector
    vllm_config.kv_transfer_config.kv_connector_extra_config = {}
    user_options = options(delayed_d2h=False)
    user_options["host_backend"] = "memfabric"
    config = SparseKVOffloadConfig(vllm_config, user_options)
    assert config.use_fused_copy_sfa
    assert config.host_backend == "memfabric"


@pytest.mark.parametrize("flag", ["delayed_d2h"])
@pytest.mark.parametrize("value,expected", [("false", False), ("true", True), (0, False), (1, True)])
def test_migration_flags_use_central_boolean_parser(flag, value, expected):
    assert getattr(SparseKVOffloadConfig(make_config(), options(**{flag: value})), flag) is expected


@pytest.mark.parametrize("flag", ["delayed_d2h"])
def test_migration_rejects_unparsed_boolean_values(flag):
    with pytest.raises(ValueError, match=f"{flag} must be a boolean"):
        SparseKVOffloadConfig(make_config(), options(**{flag: "sometimes"}))


@pytest.mark.parametrize("flag", ["delayed_d2h"])
@pytest.mark.parametrize("overrides", [{"enabled": False}, {"fused_op_type": "none"}])
def test_migration_requires_enabled_copy_sfa(flag, overrides):
    user_options = options(**{flag: True})
    user_options.update(overrides)
    with pytest.raises(ValueError, match="require sparse KV offload"):
        SparseKVOffloadConfig(make_config(), user_options)


@pytest.mark.parametrize("flag", ["delayed_d2h"])
def test_migration_requires_pd_decode(flag):
    with pytest.raises(ValueError, match="keep_device_kv_cache=false"):
        SparseKVOffloadConfig(make_config(), options(**{flag: True}, keep_device_kv_cache=True))


@pytest.mark.parametrize("flag", ["delayed_d2h"])
@pytest.mark.parametrize(
    "connector,dsa,backend",
    [
        ("SfaRemoteD2HConnector", True, "memfabric"),
        ("MultiConnector", True, "mooncake"),
        ("MooncakeConnectorV1", False, "mooncake"),
        ("MooncakeConnectorV1", "true", "mooncake"),
    ],
)
def test_migration_rejects_connectors_without_migrated_protocol(flag, connector, dsa, backend):
    vllm_config = make_config()
    vllm_config.kv_transfer_config.kv_connector = connector
    vllm_config.kv_transfer_config.kv_connector_extra_config = {"dsa_pd_offload": dsa}
    user_options = options(**{flag: True})
    user_options["host_backend"] = backend
    with pytest.raises(ValueError, match="require host_backend='mooncake'"):
        SparseKVOffloadConfig(vllm_config, user_options)


def test_actual_overlap_config_requires_larger_budget_for_copy_sfa_mtp3():
    config = make_config(drafts=3)
    existing = dict(enabled=True, use_fused_overlap=True, host_backend="mooncake", topk_buffer_size=4096)
    assert SparseKVOffloadConfig(config, existing).topk_buffer_size == 4096
    copy_sfa = dict(existing, use_fused_overlap=False, fused_op_type="fused_copy_sfa", delayed_d2h=True)
    with pytest.raises(ValueError, match="hot budget"):
        SparseKVOffloadConfig(config, copy_sfa)
    assert SparseKVOffloadConfig(config, dict(copy_sfa, topk_buffer_size=8192)).topk_buffer_size == 8192
