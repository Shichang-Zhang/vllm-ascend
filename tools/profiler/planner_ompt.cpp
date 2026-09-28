// SPDX-License-Identifier: Apache-2.0
// Optional tool: build with the same C++ standard library as the planner.
// Preload before OpenMP initialization; see sparse_kv_cpu_profiling.md.
#include <omp-tools.h>
#include <new>

#include "../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/planner_trace.h"

namespace {

struct ThreadState {
  std::shared_ptr<planner_trace::WaitTrace> wait;
  size_t pending = planner_trace::WaitTrace::kCapacity;
};

void thread_begin(ompt_thread_t, ompt_data_t* data) { data->ptr = new (std::nothrow) ThreadState(); }
void thread_end(ompt_data_t* data) { delete static_cast<ThreadState*>(data->ptr); }

void wait_callback(ompt_sync_region_t, ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, const void*);
void task_callback(ompt_scope_endpoint_t, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int);

// Immutable runtime configuration; all mutable recording state is owned by
// OpenMP thread_data, never by a process-global recorder or shared hot lock.
struct Runtime {
  ompt_get_thread_data_t get_thread_data = nullptr;
  int wait_support = 0;
  bool supported = false;

  explicit Runtime(ompt_function_lookup_t lookup) {
    if (lookup == nullptr) return;
    get_thread_data = reinterpret_cast<ompt_get_thread_data_t>(lookup("ompt_get_thread_data"));
    auto set = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (get_thread_data == nullptr || set == nullptr) return;
    const auto begin = set(ompt_callback_thread_begin, reinterpret_cast<ompt_callback_t>(thread_begin));
    const auto end = set(ompt_callback_thread_end, reinterpret_cast<ompt_callback_t>(thread_end));
    const auto task = set(ompt_callback_implicit_task, reinterpret_cast<ompt_callback_t>(task_callback));
    wait_support = set(ompt_callback_sync_region_wait, reinterpret_cast<ompt_callback_t>(wait_callback));
    supported = begin == ompt_set_always && end == ompt_set_always && task == ompt_set_always;
  }
};

const Runtime& runtime(ompt_function_lookup_t lookup = nullptr) {
  static const Runtime configuration(lookup);
  return configuration;
}

ThreadState* state() {
  const auto& configuration = runtime();
  if (!configuration.supported) return nullptr;
  auto* data = configuration.get_thread_data();
  return data == nullptr ? nullptr : static_cast<ThreadState*>(data->ptr);
}

void wait_callback(ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, const void*) {
  // Only the final parallel barrier, not arbitrary runtime locks or workshare
  // barriers. Legacy runtimes report the generic implicit-barrier kind.
  if (kind != ompt_sync_region_barrier_implicit && kind != ompt_sync_region_barrier_implicit_parallel) return;
  auto* thread = state();
  if (thread == nullptr || !thread->wait) return;
  auto& wait = *thread->wait;
  if (endpoint == ompt_scope_begin) {
    thread->pending = wait.count.load(std::memory_order_relaxed);
    if (thread->pending < planner_trace::WaitTrace::kCapacity) {
      wait.intervals[thread->pending].begin_ns.store(planner_trace::clock_ns());
    }
    wait.count.store(thread->pending + 1, std::memory_order_release);
  } else if (endpoint == ompt_scope_end && thread->pending < planner_trace::WaitTrace::kCapacity) {
    wait.intervals[thread->pending].end_ns.store(planner_trace::clock_ns(), std::memory_order_release);
    thread->pending = planner_trace::WaitTrace::kCapacity;
  }
}

void task_callback(ompt_scope_endpoint_t endpoint, ompt_data_t*, ompt_data_t*, unsigned int, unsigned int, int) {
  if (endpoint != ompt_scope_end) return;
  auto* thread = state();
  if (thread != nullptr) thread->wait.reset();
}

int initialize(ompt_function_lookup_t lookup, int, ompt_data_t*) {
  const auto& configuration = runtime(lookup);
  // Do not accidentally use one OpenMP runtime's thread data in another.
  return configuration.supported &&
         configuration.get_thread_data == reinterpret_cast<ompt_get_thread_data_t>(lookup("ompt_get_thread_data"));
}

}  // namespace

extern "C" void ascend_planner_ompt_bind_v1(const std::shared_ptr<planner_trace::WaitTrace>* wait) {
  auto* thread = state();
  if (thread == nullptr) return;
  thread->wait = *wait;
  thread->pending = planner_trace::WaitTrace::kCapacity;
  thread->wait->support = runtime().wait_support;
}

extern "C" ompt_start_tool_result_t* ompt_start_tool(unsigned int, const char*) {
  // The runtime retains this small descriptor for the lifetime of the tool.
  // It owns tool_data; no mutable static storage is needed.
  return new (std::nothrow) ompt_start_tool_result_t{initialize, nullptr, {0}};
}
