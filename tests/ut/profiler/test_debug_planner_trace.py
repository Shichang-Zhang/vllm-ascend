"""Compile the production recorder and planner loop without CANN/torch."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"


@unittest.skipUnless(shutil.which("g++"), "requires g++")
class TestDebugPlannerTrace(unittest.TestCase):
    def compile_run(self, program):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "test.cpp").write_text(program)
            subprocess.run(
                [
                    "g++",
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fopenmp",
                    "-pthread",
                    "-I",
                    str(SOURCE_DIR),
                    str(path / "test.cpp"),
                    "-ldl",
                    "-o",
                    str(path / "test"),
                ],
                check=True,
            )
            subprocess.run([str(path / "test")], check=True, timeout=30)

    def test_disabled_bounded_restart_and_delayed_wait_lifetime(self):
        self.compile_run(r"""
#include "planner_trace.h"
#include <cassert>
#include <thread>
int main() {
  DebugPlannerTrace trace;
  trace.end(trace.begin());
  assert(trace.calls == 0 && trace.records.empty());
  trace.reset(2, 3);
  for (size_t i = 0; i < 5; ++i) {
    auto* record = trace.begin();
    if (i < 3) {
      assert(record);
      record->threads[0].begin(nullptr);
      record->threads[0].end();
    } else {
      assert(record == nullptr);
    }
    trace.end(record);
  }
  assert(trace.calls == 5);
  for (const auto& record : trace.records) {
    assert(record.begin_ns > 0 && record.end_ns >= record.begin_ns);
    const auto& thread = record.threads[0];
    assert(thread.tid > 0 && thread.cpu_begin >= 0);
    assert(thread.work_end_ns >= thread.work_begin_ns);
    assert(thread.cpu_end_ns >= thread.cpu_begin_ns);
  }
  trace.enabled.store(false);
  assert(trace.begin() == nullptr && trace.calls == 5);
  auto pending = std::make_shared<planner_trace::WaitTrace>();
  trace.records[0].threads[0].wait = pending;
  trace.reset();
  std::thread late_callback([pending]() {
    pending->intervals[0].begin_ns = 1;
    pending->intervals[0].end_ns = 2;
  });
  late_callback.join();
  assert(pending->intervals[0].end_ns == 2);
  assert(trace.calls == 0 && trace.records[0].threads[0].entry_ns == 0);
  trace.end(trace.begin());
  assert(trace.calls == 1);
}
""")

    def test_real_planner_outputs_unchanged_and_rows_accounted(self):
        source = (SOURCE_DIR / "sparse_kv_offload.cpp").read_text()
        begin = source.index("constexpr int32_t EPOCH_RESET_THRESHOLD")
        end = source.index("HOT_FUNCTION void lru_resident_compact(")
        program = r"""
#include "planner_trace.h"
#include <omp.h>
#include <cassert>
#include <numeric>
#define HOT_FUNCTION
#define FORCE_INLINE inline
#define RESTRICT __restrict__
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define TORCH_CHECK(condition, ...) assert(condition)
"""
        program += source[begin:end]
        program += r"""
struct State {
  static constexpr int rows = 5, topk = 4, capacity = 8, tokens = 32, workers = 2;
  std::vector<int64_t> ids = {1,2,3,4,5}, last = std::vector<int64_t>(rows, -1);
  std::vector<int32_t> indices = std::vector<int32_t>(rows * topk);
  std::vector<int32_t> prefix = std::vector<int32_t>(rows, tokens);
  std::vector<int32_t> slots = std::vector<int32_t>(rows * capacity);
  std::vector<int32_t> lru = slots, current = indices, count = prefix, misses = indices, miss_slots = indices;
  std::vector<int32_t> mark = std::vector<int32_t>(workers * tokens);
  std::vector<int32_t> pos = mark, work = std::vector<int32_t>(workers * capacity * 3);
  std::vector<int32_t> positions = std::vector<int32_t>(workers * topk), epochs = std::vector<int32_t>(workers);
  State() { for (int i = 0; i < rows * topk; ++i) indices[i] = i % topk; }
  template<class T> static uintptr_t ptr(std::vector<T>& values) { return reinterpret_cast<uintptr_t>(values.data()); }
  void run(int threads, planner_trace::CallRecord* trace) {
    lru_resident_compact_impl(ptr(ids), ptr(last), ptr(indices), ptr(prefix), ptr(slots), ptr(lru),
      ptr(current), ptr(count), ptr(misses), ptr(miss_slots), ptr(mark), ptr(pos), ptr(work), ptr(positions),
      ptr(epochs), rows, topk, capacity, tokens, workers, threads, 0, 0, 0, rows, 0, trace);
  }
};
int main() {
  omp_set_dynamic(0);
  for (int threads : {1, 2}) {
    State baseline, measured;
    DebugPlannerTrace trace;
    trace.reset(threads, 2);
    baseline.run(threads, nullptr);
    auto* call = trace.begin();
    measured.run(threads, call);
    trace.end(call);
    assert(baseline.current == measured.current);
    assert(baseline.misses == measured.misses && baseline.slots == measured.slots);
    assert(baseline.lru == measured.lru && baseline.count == measured.count);
    assert(call->actual_threads == threads);
    assert(call->begin_ns <= call->parallel_begin_ns);
    int rows = 0, misses = 0;
    for (const auto& thread : call->threads) {
      rows += thread.rows;
      misses += thread.misses;
      assert(thread.entry_ns >= call->parallel_begin_ns);
      assert(thread.work_end_ns <= call->parallel_end_ns);
    }
    assert(rows == State::rows && misses == State::rows * State::topk);
    assert(call->parallel_end_ns <= call->end_ns);
  }
}
"""
        self.compile_run(program)


if __name__ == "__main__":
    unittest.main()
