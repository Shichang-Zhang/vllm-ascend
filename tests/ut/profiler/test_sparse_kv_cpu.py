# SPDX-License-Identifier: Apache-2.0
"""No torch/CANN required: test observability semantics with synthetic evidence."""

import importlib.util
import os
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/profiler/sparse_kv_cpu.py"
SPEC = importlib.util.spec_from_file_location("sparse_kv_cpu", SOURCE)
CPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPU)


class TestSparseKvCpu(unittest.TestCase):
    def test_cgroup_v2_nested_and_namespace_root(self):
        mount = "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
        groups = CPU.cgroup_locations("0::/kubepods/pod/container\n", mount)
        self.assertEqual(groups[0]["path"], "/sys/fs/cgroup/kubepods/pod/container")
        nested = mount.replace("0:32 / ", "0:32 /kubepods/pod/container ")
        self.assertEqual(CPU.cgroup_locations("0::/\n", nested)[0]["path"], "/sys/fs/cgroup")
        self.assertEqual(CPU.cgroup_locations("0::/../../other\n", mount), [])

    def test_cgroup_v1_separate_controllers_and_mount_escape(self):
        mounts = (
            "31 25 0:29 /pods /sys/fs/cgroup/cpu\\040acct rw - cgroup cgroup rw,cpu,cpuacct\n"
            "32 25 0:30 / /sys/fs/cgroup/cpuset rw - cgroup cgroup rw,cpuset\n"
        )
        groups = CPU.cgroup_locations("4:cpu,cpuacct:/pods/a\n3:cpuset:/pods/a\n", mounts)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["path"], "/sys/fs/cgroup/cpu acct/a")
        self.assertEqual(groups[1]["path"], "/sys/fs/cgroup/cpuset/pods/a")

    def test_unavailable_is_not_zero_and_snapshot_works_without_privilege(self):
        self.assertIn("error", CPU.read_file("/nonexistent/planner-test"))
        result = CPU.snapshot(os.getpid())
        self.assertEqual(result["identity"]["Pid"], str(os.getpid()))
        self.assertIn(str(os.getpid()), result["threads"])
        self.assertGreaterEqual(result["end_ns"], result["begin_ns"])

    def test_memory_mapping_is_vma_evidence_not_exact_buffer_distribution(self):
        context = {
            "maps": {"text": "1000-3000 rw-p 0 00:00 0\n3000-4000 rw-p 0 00:00 0\n"},
            "numa_maps": {"text": "1000 default N0=1 N1=1\n3000 default N1=1\n"},
        }
        result = CPU.buffer_mappings([{"name": "workspace", "address": 0x2800, "bytes": 0x1000}], context)
        self.assertEqual(len(result[0]["mappings"]), 2)
        self.assertIn("not exact", result[0]["scope"])

    def test_scheduler_preemption_block_and_wakeup_are_distinct(self):
        events = [
            {"ts_ns": 0, "kind": "switch", "prev": 0, "next": 10, "runnable": True},
            {"ts_ns": 10, "kind": "switch", "prev": 10, "next": 20, "runnable": True},
            {"ts_ns": 20, "kind": "switch", "prev": 20, "next": 10, "runnable": True},
            {"ts_ns": 30, "kind": "switch", "prev": 10, "next": 20, "runnable": False},
            {"ts_ns": 50, "kind": "wakeup", "tid": 10},
            {"ts_ns": 60, "kind": "switch", "prev": 20, "next": 10, "runnable": True},
            {"ts_ns": 70, "kind": "switch", "prev": 10, "next": 0, "runnable": False},
        ]
        intervals = CPU.scheduler_intervals(events)
        self.assertEqual(
            CPU.scheduler_overlap(intervals[10], 0, 80),
            {"running_ns": 30, "runnable_ns": 20, "blocked_ns": 20, "unknown_ns": 10},
        )

    def test_perf_parser_retains_nanoseconds_and_reports_loss(self):
        raw = (
            "1.000000123: sched:sched_switch: prev_comm=a prev_pid=10 prev_prio=120 prev_state=R+ "
            "==> next_comm=b next_pid=20 next_prio=120\n"
            "1.000000200: sched:sched_wakeup: comm=a pid=10 prio=120 target_cpu=2\n"
            "LOST 20 events\n"
        )
        result = CPU.normalize_perf(raw)
        self.assertEqual(result["events"][0]["ts_ns"], 1_000_000_123)
        self.assertTrue(result["events"][0]["runnable"])
        self.assertEqual(result["events"][1]["tid"], 10)
        self.assertEqual(result["unparsed_lines"], ["LOST 20 events"])

    def test_incomplete_wait_is_not_zero_idle_and_tail_is_not_barrier(self):
        thread = {
            "tid": 10,
            "entry_ns": 110,
            "work_begin_ns": 120,
            "work_end_ns": 220,
            "cpu_begin_ns": 1000,
            "cpu_end_ns": 1040,
            "rows": 2,
            "misses": 8,
            "ompt_support": 5,
            "ompt_dropped": 0,
            "ompt_wait_intervals_ns": [[230, 0]],
        }
        document = {
            "hostname": "pod-a",
            "pid": 123,
            "dp_rank": 7,
            "tp_rank": 0,
            "records": [
                {
                    "num_reqs": 2,
                    "topk": 4,
                    "dropped": 3,
                    "details": [
                        {
                            "call_index": 0,
                            "begin_ns": 90,
                            "parallel_begin_ns": 100,
                            "parallel_end_ns": 400,
                            "end_ns": 410,
                            "threads": [thread],
                        }
                    ],
                }
            ],
        }
        result = CPU.analyze(document)
        worker = result["calls"][0]["workers"][0]
        self.assertFalse(worker["wait_complete"])
        self.assertEqual(worker["tail_to_parallel_return_ns"], 180)
        self.assertEqual(worker["work_cpu_ns"], 40)
        self.assertNotIn("scheduler_work", worker)
        self.assertEqual(result["dropped_calls"], 3)
        self.assertEqual(result["identity"]["dp_rank"], 7)
        self.assertEqual(len(CPU.chrome_trace(document)["traceEvents"]), 1)

        # Missing namespace mappings must not accidentally attribute another
        # host thread with the same numeric TID to this container worker.
        scheduler = {
            "events": [
                {"ts_ns": 100, "kind": "switch", "prev": 0, "next": 10, "runnable": True},
                {"ts_ns": 300, "kind": "switch", "prev": 10, "next": 0, "runnable": False},
            ]
        }
        mapped = CPU.analyze(document, scheduler, tid_map={})["calls"][0]["workers"][0]
        self.assertIsNone(mapped["scheduler_tid"])
        self.assertEqual(mapped["scheduler_work"]["unknown_ns"], 100)
        direct = CPU.analyze(document, scheduler)["calls"][0]["workers"][0]
        self.assertEqual(direct["scheduler_work"]["running_ns"], 100)

        # A completed OMPT interval may extend past the callback's critical path.
        thread["ompt_wait_intervals_ns"] = [[230, 600]]
        worker = CPU.analyze(document)["calls"][0]["workers"][0]
        self.assertEqual(worker["observed_wait_ns"], 370)
        self.assertEqual(worker["observed_wait_in_parallel_ns"], 170)

    def test_cgroup_deltas_preserve_scope_and_missing_counters(self):
        group = {
            "controllers": "",
            "path": "/cg",
            "version": 2,
            "files": {"cpu.stat": {"text": "nr_throttled 4\nthrottled_usec 200\n"}},
        }
        after = {**group, "files": {"cpu.stat": {"text": "nr_throttled 6\nthrottled_usec 500\n"}}}
        result = CPU.cgroup_deltas({"cgroups": [group]}, {"cgroups": [after]})
        self.assertEqual(result[0]["cpu_stat_delta"], {"nr_throttled": 2, "throttled_usec": 300})
        self.assertEqual(CPU.cgroup_deltas({}, {"cgroups": [after]}), [])


if __name__ == "__main__":
    unittest.main()
