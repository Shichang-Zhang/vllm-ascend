# SPDX-License-Identifier: Apache-2.0
"""Optional CPU integration test against a real LLVM OpenMP runtime."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("g++"), "requires g++ and LLVM OpenMP development files")
class TestPlannerOmpt(unittest.TestCase):
    def test_real_wait_callbacks_and_delayed_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            # Probe prerequisites separately: a production-tool build failure must not be skipped.
            probe = subprocess.run(
                ["g++", "-x", "c++", "-", "-lomp", "-o", str(path / "probe")],
                input="#include <omp-tools.h>\nint main() {}\n",
                text=True,
                capture_output=True,
            )
            if probe.returncode:
                self.skipTest("requires omp-tools.h on include path and linkable libomp")
            library = path / "planner_ompt.so"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-shared",
                    "-fPIC",
                    str(ROOT / "tools/profiler/planner_ompt.cpp"),
                    "-o",
                    str(library),
                ],
                check=True,
            )
            source = path / "test.cpp"
            source.write_text(r"""
#include "planner_trace.h"
#include <omp.h>
#include <cassert>
#include <thread>
#include <chrono>
int main() {
  omp_set_dynamic(0);
  DebugPlannerTrace trace;
  trace.reset(2, 2);
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto* call = trace.begin();
    assert(call && call->bind_wait);
#pragma omp parallel num_threads(2)
    {
      const auto tid = omp_get_thread_num();
      auto& record = call->threads[tid];
      record.begin(call->bind_wait);
      if (tid == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
      record.end();
    }
    trace.end(call);
  }
  for (const auto& record : trace.records) {
    for (const auto& thread : record.threads) {
      assert(thread.wait && thread.wait->support == 5);
      assert(thread.wait->count.load() > 0);
      assert(thread.wait->intervals[0].begin_ns.load() >= thread.work_end_ns);
      const auto end = thread.wait->intervals[0].end_ns.load();
      assert(end == 0 || end >= thread.wait->intervals[0].begin_ns.load());
    }
  }
  // A worker's previous wait-end can be deferred until its next parallel task.
  // Resetting the recorder must not invalidate that callback's storage.
  auto retained = trace.records[1].threads[1].wait;
  trace.reset(2, 1);
#pragma omp parallel num_threads(2)
  { std::this_thread::yield(); }
  assert(retained->intervals[0].end_ns.load() >= retained->intervals[0].begin_ns.load());
}
""")
            binary = path / "test"
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-fopenmp",
                    "-pthread",
                    "-I",
                    str(ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"),
                    str(source),
                    "-lomp",
                    "-ldl",
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            environment = {
                **os.environ,
                "OMP_TOOL": "enabled",
                "OMP_TOOL_LIBRARIES": str(library),
                "LD_PRELOAD": str(library),
            }
            subprocess.run([str(binary)], env=environment, check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
