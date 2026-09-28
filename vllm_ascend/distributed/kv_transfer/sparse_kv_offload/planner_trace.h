// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace planner_trace {

// Accommodate both 64-byte and 128-byte cache lines on supported hosts.
constexpr size_t kRecordAlignment = 128;

inline int64_t clock_ns(clockid_t clock = CLOCK_MONOTONIC) noexcept {
  timespec value{};
  if (clock_gettime(clock, &value) != 0) return -1;
  constexpr int64_t kNanosecondsPerSecond = 1000000000;
  return value.tv_sec * kNanosecondsPerSecond + value.tv_nsec;
}

// OMPT may finish a worker's final barrier after the primary thread has left
// the parallel region. Shared ownership keeps these slots alive across reset
// and graph destruction; atomic endpoints make an incomplete export safe.
struct WaitInterval {
  std::atomic<int64_t> begin_ns{0};
  std::atomic<int64_t> end_ns{0};
};

struct alignas(kRecordAlignment) WaitTrace {
  static constexpr size_t kCapacity = 4;
  WaitInterval intervals[kCapacity];
  std::atomic<size_t> count{0};
  int support = 0;  // ompt_set_result_t; zero means no cooperating tool.
};

using BindWait = void (*)(const std::shared_ptr<WaitTrace>*);

// Separate cache lines prevent the recorder from creating worker contention.
struct alignas(kRecordAlignment) ThreadRecord {
  int64_t entry_ns = 0;
  int64_t work_begin_ns = 0;
  int64_t work_end_ns = 0;
  int64_t cpu_begin_ns = -1;
  int64_t cpu_end_ns = -1;
  int64_t tid = 0;
  int cpu_begin = -1;
  int cpu_end = -1;
  int rows = 0;
  int64_t misses = 0;
  std::shared_ptr<WaitTrace> wait;

  void begin(BindWait bind_wait) noexcept {
    entry_ns = clock_ns();
    tid = syscall(SYS_gettid);
    cpu_begin = sched_getcpu();
    if (bind_wait != nullptr && wait) bind_wait(&wait);
    cpu_begin_ns = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    work_begin_ns = clock_ns();
  }

  void end() noexcept {
    work_end_ns = clock_ns();
    cpu_end_ns = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    cpu_end = sched_getcpu();
  }
};

struct CallRecord {
  int64_t begin_ns = 0;
  int64_t parallel_begin_ns = 0;
  int64_t parallel_end_ns = 0;
  int64_t end_ns = 0;
  int64_t callback_tid = 0;
  int actual_threads = 0;
  BindWait bind_wait = nullptr;
  std::vector<ThreadRecord> threads;
};

}  // namespace planner_trace

struct DebugPlannerTrace {
  static constexpr size_t kCapacity = 256;
  static constexpr size_t kMaxCapacity = 65536;
  std::atomic<bool> enabled{false};
  size_t calls = 0;
  std::vector<planner_trace::CallRecord> records;
  std::vector<std::shared_ptr<planner_trace::WaitTrace>> retired_waits;

  static int64_t now_ns() noexcept { return planner_trace::clock_ns(); }

  // Call only after draining the stream. Allocate and touch storage outside
  // the measured window. No record allocation or output runs in a callback.
  void reset(int threads = 1, size_t capacity = kCapacity) {
    enabled.store(false);
    calls = 0;
    // A late OMPT endpoint must not free an old window's final record in the
    // next measured parallel region. Reclaim it here once the tool lets go.
    retired_waits.erase(std::remove_if(retired_waits.begin(), retired_waits.end(),
                                       [](const auto& wait) { return wait.use_count() == 1; }),
                        retired_waits.end());
    for (auto& record : records) {
      for (auto& thread : record.threads) {
        if (thread.wait && thread.wait.use_count() > 1) retired_waits.push_back(std::move(thread.wait));
      }
    }
    records.clear();
    records.resize(capacity);
    const auto bind = reinterpret_cast<planner_trace::BindWait>(dlsym(RTLD_DEFAULT, "ascend_planner_ompt_bind_v1"));
    for (auto& record : records) {
      record.bind_wait = bind;
      record.threads.resize(std::max(1, threads));
      if (bind != nullptr) {
        for (auto& thread : record.threads) thread.wait = std::make_shared<planner_trace::WaitTrace>();
      }
    }
    enabled.store(true);
  }

  planner_trace::CallRecord* begin() noexcept {
    if (!enabled.load(std::memory_order_acquire)) return nullptr;
    const size_t index = calls++;
    if (index >= records.size()) return nullptr;
    auto& record = records[index];
    record.begin_ns = now_ns();
    record.callback_tid = syscall(SYS_gettid);
    return &record;
  }

  void end(planner_trace::CallRecord* record) noexcept {
    if (record != nullptr) record->end_ns = now_ns();
  }
};
