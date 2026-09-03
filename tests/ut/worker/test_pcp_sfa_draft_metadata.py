from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.attention.sfa_v1 import DCPContext
from vllm_ascend.worker.pcp_utils import PCPManager


def _manager(*, dcp: int = 8) -> PCPManager:
    mgr = MagicMock(spec=PCPManager)
    mgr.pcp_world_size = 1
    mgr.pcp_world_rank = 0
    mgr.dcp_world_size = dcp
    mgr.dcp_world_rank = 0
    mgr.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=128)
    )
    mgr._is_mla_kv_cache_spec = PCPManager._is_mla_kv_cache_spec.__get__(mgr, PCPManager)
    mgr._is_sfa_dcp_metadata_builder = PCPManager._is_sfa_dcp_metadata_builder.__get__(
        mgr, PCPManager
    )
    mgr._get_cp_local_seq_lens = PCPManager._get_cp_local_seq_lens.__get__(mgr, PCPManager)
    return mgr


def test_sfa_offload_metadata_does_not_require_nested_decode() -> None:
    mgr = _manager(dcp=8)
    metadata = SimpleNamespace(dcp_context=None, seq_lens=torch.tensor([2060], dtype=torch.int32))
    PCPManager.update_spec_decode_drafting_cp_metadata(
        mgr,
        attn_metadata=metadata,
        kv_cache_spec=object(),
        seq_lens=torch.tensor([2060], dtype=torch.int32),
        draft_index=0,
        attn_metadata_builder=SimpleNamespace(uses_unified_main_kv_view=True),
    )


def test_unknown_layout_fail_fast() -> None:
    mgr = _manager(dcp=8)
    metadata = SimpleNamespace(dcp_context=None)
    try:
        PCPManager.update_spec_decode_drafting_cp_metadata(
            mgr,
            attn_metadata=metadata,
            kv_cache_spec=object(),
            seq_lens=torch.tensor([2060], dtype=torch.int32),
            draft_index=0,
            attn_metadata_builder=object(),
        )
    except RuntimeError as exc:
        assert "Unknown spec-decode CP metadata layout" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_sfa_dcp_context_still_updates_local_seq_lens() -> None:
    mgr = _manager(dcp=8)
    dcp_seq = torch.zeros(2, dtype=torch.int32)
    metadata = SimpleNamespace(
        dcp_context=DCPContext(
            slot_mapping=torch.zeros(1, dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=dcp_seq,
        )
    )
    PCPManager.update_spec_decode_drafting_cp_metadata(
        mgr,
        attn_metadata=metadata,
        kv_cache_spec=object(),
        seq_lens=torch.tensor([2060], dtype=torch.int32),
        draft_index=0,
        attn_metadata_builder=object(),
    )
    assert int(dcp_seq[0].item()) > 0
    assert int(dcp_seq[1].item()) == 0


def test_mla_nested_decode_still_sets_cp_seq_len() -> None:
    mgr = _manager(dcp=8)
    decode = SimpleNamespace(cp_seq_len=None)
    metadata = SimpleNamespace(dcp_context=None, decode=decode)
    PCPManager.update_spec_decode_drafting_cp_metadata(
        mgr,
        attn_metadata=metadata,
        kv_cache_spec=object(),
        seq_lens=torch.tensor([2060], dtype=torch.int32),
        draft_index=0,
        attn_metadata_builder=object(),
    )
    assert decode.cp_seq_len is not None
    assert decode.cp_seq_len.shape[0] == 1
