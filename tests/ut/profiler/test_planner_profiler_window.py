# SPDX-License-Identifier: Apache-2.0
"""Exercise the production wrapper's window lifecycle without importing torch."""

import ast
import json
import os
import resource
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch


class WorkerProfilerStub:
    def __init__(self, config):
        pass


def wrapper_type(cpu_only):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/profiler/torch_npu_profiler.py"
    tree = ast.parse(source.read_text())
    # Compile the exact class, injecting hardware dependencies at its boundary.
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    npu = SimpleNamespace(npu=SimpleNamespace(synchronize=MagicMock()))
    namespace = {
        "WorkerProfiler": WorkerProfilerStub,
        "ProfilerConfig": SimpleNamespace,
        "Any": Any,
        "envs_ascend": SimpleNamespace(VLLM_ASCEND_PLANNER_TRACE_ONLY=cpu_only, VLLM_ASCEND_PLANNER_TRACE_CAPACITY=32),
        "torch_npu": npu,
        "snapshot": MagicMock(return_value={}),
        "buffer_mappings": MagicMock(return_value=[]),
        "json": json,
        "os": os,
        "resource": resource,
        "socket": socket,
        "sys": sys,
        "time": time,
        "Path": Path,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["TorchNPUProfilerWrapper"], namespace


class TestPlannerProfilerWindow(unittest.TestCase):
    def test_cpu_only_exports_dp_identity_without_starting_npu_profiler(self):
        wrapper, namespace = wrapper_type(True)
        extension = MagicMock()
        extension.debug_planner_clock_ns.return_value = 1000
        extension.debug_stop_planner_trace.return_value = [
            {"lru_state_ptr": 123, "begin_end_steady_ns": [[1000, 4000]], "calls": 1, "dropped": 0}
        ]
        manager = SimpleNamespace(
            use_fused_overlap=True,
            tp_rank=0,
            tp_size=2,
            sparse_kv_offload_cpp=extension,
            lru_last_req_ids_ptrs=[123],
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=7, data_parallel_size=16)),
        )
        name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {name: SimpleNamespace(_SPARSE_KV_OFFLOAD_MANAGER=manager)}),
            patch.object(wrapper, "_create_profiler", side_effect=AssertionError("NPU profiler must not be created")),
        ):
            instance = wrapper(SimpleNamespace(torch_profiler_dir=directory), "dp7_tp0")
            instance._start()
            instance._stop()
            extension.debug_start_planner_trace.assert_called_once_with(32)
            extension.debug_stop_planner_trace.assert_called_once()
            self.assertEqual(namespace["torch_npu"].npu.synchronize.call_count, 2)
            document = json.loads(next(Path(directory).glob("*_lru_callback.json")).read_text())
            self.assertEqual(document["dp_rank"], 7)
            self.assertEqual(document["tp_size"], 2)
            self.assertEqual(document["schema_version"], 2)
            self.assertEqual(document["records"][0]["p50_us"], 3)
            self.assertIsNone(instance._debug_planner_manager)

    def test_non_planner_rank_does_not_synchronize_or_collect(self):
        wrapper, namespace = wrapper_type(True)
        manager = SimpleNamespace(use_fused_overlap=True, tp_rank=1)
        name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager"
        with patch.dict(sys.modules, {name: SimpleNamespace(_SPARSE_KV_OFFLOAD_MANAGER=manager)}):
            instance = wrapper(SimpleNamespace(torch_profiler_dir="/unused"), "dp7_tp1")
            instance._start()
            instance._stop()
            namespace["snapshot"].assert_not_called()
            namespace["torch_npu"].npu.synchronize.assert_not_called()

    def test_combined_mode_preserves_profiler_start_stop(self):
        wrapper, _ = wrapper_type(False)
        profiler = MagicMock()
        name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager"
        with (
            patch.object(wrapper, "_create_profiler", return_value=profiler),
            patch.dict(sys.modules, {name: SimpleNamespace(_SPARSE_KV_OFFLOAD_MANAGER=None)}),
        ):
            instance = wrapper(SimpleNamespace(torch_profiler_dir="/unused"), "dp0_tp1")
            instance._start()
            instance._stop()
            profiler.start.assert_called_once()
            profiler.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
