"""CPU-only check of the temporary C++ callback recorder, without CANN/torch."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestDebugPlannerTrace(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_disabled_bounded_and_restart(self):
        source = (
            Path(__file__).resolve().parents[3]
            / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp"
        ).read_text()
        # Compile the exact recorder used by the NPU callback, without hardware headers.
        begin = source.index("struct DebugPlannerTrace {")
        end = source.index("\nstruct LruResidentCompactWithPlanPayload", begin)
        program = "#include <array>\n#include <atomic>\n#include <chrono>\n#include <cassert>\n"
        program += source[begin:end]
        program += r"""
int main() {
  DebugPlannerTrace trace;
  trace.end(trace.begin());
  assert(trace.calls == 0);
  trace.reset();
  for (size_t i = 0; i < DebugPlannerTrace::kCapacity + 3; ++i) {
    const auto index = trace.begin();
    trace.end(index);
  }
  assert(trace.calls == DebugPlannerTrace::kCapacity + 3);
  for (const auto& record : trace.records) {
    assert(record[0] > 0);
    assert(record[1] >= record[0]);
  }
  trace.enabled.store(false);
  trace.end(trace.begin());
  assert(trace.calls == DebugPlannerTrace::kCapacity + 3);
  trace.reset();
  trace.end(trace.begin());
  assert(trace.calls == 1);
  assert(trace.records[0][1] >= trace.records[0][0]);
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test.cpp").write_text(program)
            subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(path / "test.cpp"), "-o", str(path / "test")],
                check=True,
            )
            subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
