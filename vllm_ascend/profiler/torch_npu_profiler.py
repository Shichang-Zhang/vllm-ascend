#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import json
import os
import resource
import socket
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch_npu
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import WorkerProfiler

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config


class TorchNPUProfilerWrapper(WorkerProfiler):
    """Subclass of vLLM ``WorkerProfiler`` that wires in ``torch_npu.profiler``."""

    def __init__(self, profiler_config: ProfilerConfig, trace_name: str) -> None:
        super().__init__(profiler_config)
        self._debug_trace_dir = profiler_config.torch_profiler_dir
        self._debug_trace_name = trace_name
        self._debug_planner_manager = None
        self.profiler: Any = self._create_profiler(profiler_config, trace_name)

    @staticmethod
    def _create_profiler(profiler_config: ProfilerConfig, trace_name: str) -> Any:
        if profiler_config.profiler != "torch":
            raise RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")
        if not profiler_config.torch_profiler_dir:
            raise RuntimeError("torch_profiler_dir cannot be empty.")
        msmonitor_use_daemon = envs_ascend.MSMONITOR_USE_DAEMON
        with suppress(RuntimeError):
            msmonitor_use_daemon = get_ascend_config().msmonitor_use_daemon
        if msmonitor_use_daemon:
            raise RuntimeError("MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time.")

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )

        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=False,
            profile_memory=profiler_config.torch_profiler_with_memory,
            # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
            # The with_stack option in torch_npu.profiler introduces significant time overhead.
            with_modules=profiler_config.torch_profiler_with_stack,
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiler_config.torch_profiler_dir,
                worker_name=trace_name,
            ),
        )

    def _start(self) -> None:
        # Temporary: use an already loaded manager; do not initialize offload here.
        module = sys.modules.get(
            "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager"
        )
        manager = getattr(module, "_SPARSE_KV_OFFLOAD_MANAGER", None)
        if manager is not None and manager.use_fused_overlap and manager.tp_rank == 0:
            torch_npu.npu.synchronize()
            self._debug_planner_manager = manager
            extension = manager.sparse_kv_offload_cpp
            extension.debug_start_planner_trace()
            self._debug_clock = self._debug_clock_anchor(extension)
            self._debug_faults = resource.getrusage(resource.RUSAGE_SELF)
        self.profiler.start()

    @staticmethod
    def _debug_clock_anchor(extension):
        # Bracket the clock sample: clocks from different hosts are not directly comparable.
        before = time.time_ns()
        steady = extension.debug_planner_clock_ns()
        after = time.time_ns()
        return {"steady_ns": steady, "unix_before_ns": before, "unix_after_ns": after}

    def _stop(self) -> None:
        self.profiler.stop()
        manager = self._debug_planner_manager
        if manager is not None:
            # Drain callbacks before reading/resetting their preallocated buffers.
            # This is outside the measured forward path, with no per-layer barriers.
            torch_npu.npu.synchronize()
            extension = manager.sparse_kv_offload_cpp
            records = extension.debug_stop_planner_trace()
            layers = {ptr: layer for layer, ptr in enumerate(manager.lru_last_req_ids_ptrs)}
            for record in records:
                record["layer_id"] = layers.get(record["lru_state_ptr"])
                durations = sorted((end - begin) / 1000 for begin, end in record["begin_end_steady_ns"])
                record["p50_us"] = durations[int((len(durations) - 1) * 0.50)]
                record["p99_us"] = durations[int((len(durations) - 1) * 0.99)]
                record["max_us"] = durations[-1]
            output = Path(self._debug_trace_dir)
            output.mkdir(parents=True, exist_ok=True)
            faults = resource.getrusage(resource.RUSAGE_SELF)
            document = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "cpu_minor_faults": faults.ru_minflt - self._debug_faults.ru_minflt,
                "cpu_major_faults": faults.ru_majflt - self._debug_faults.ru_majflt,
                "rank_trace_name": self._debug_trace_name,
                "start_clock_anchor": self._debug_clock,
                "stop_clock_anchor": self._debug_clock_anchor(extension),
                "scope": "existing captured planner callbacks; includes profile boundary drain",
                "records": records,
            }
            filename = f"{self._debug_trace_name}_{os.getpid()}_{time.time_ns()}_lru_callback.json"
            (output / filename).write_text(json.dumps(document, indent=2))
            self._debug_planner_manager = None

    def _profiler_step(self) -> bool:
        return True
