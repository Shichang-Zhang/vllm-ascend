# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run directly with Python as well as pytest; no NPU packages required."""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock


class TestSFASyncTiming(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/profiler/sfa_sync_timing.py"
        spec = importlib.util.spec_from_file_location("sfa_sync_timing", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.timing_class = module.SFASyncTiming
        self.report_samples = module.TIMING_REPORT_SAMPLES

    def test_separates_backlog_host_call_and_completion(self):
        order = []
        emit = Mock()
        timer = self.timing_class(
            lambda: order.append("sync"),
            lambda: False,
            emit,
            {"tp_rank": 1},
            clock=Mock(side_effect=[0, 10000, 13000, 20000]),
        )
        with timer.measure("broadcast"):
            order.append("operation")
        self.assertEqual(order, ["sync", "operation", "sync"])
        report = json.loads(emit.call_args.args[0].removeprefix("SFA_SYNC_TIMING "))
        self.assertEqual(report["tp_rank"], 1)
        self.assertEqual(report["prior_work_us"]["mean"], 10)
        self.assertEqual(report["host_call_us"]["mean"], 3)
        self.assertEqual(report["completion_wait_us"]["mean"], 7)
        self.assertEqual(report["phase_us"]["mean"], 10)

    def test_never_synchronizes_or_reads_clock_during_capture(self):
        for explicitly_capturing, runtime_capturing in ((False, True), (True, False)):
            with self.subTest(explicitly_capturing=explicitly_capturing):
                sync, emit, clock = Mock(), Mock(), Mock()
                timer = self.timing_class(sync, Mock(return_value=runtime_capturing), emit, {}, clock=clock)
                body = Mock()
                with timer.measure("capture", capturing=explicitly_capturing):
                    body()
                body.assert_called_once()
                sync.assert_not_called()
                emit.assert_not_called()
                clock.assert_not_called()

    def test_reports_bounded_windows_without_mixing_phases(self):
        emit = Mock()
        timer = self.timing_class(Mock(), lambda: False, emit, {}, clock=Mock(return_value=0))
        with timer.measure("other"):
            pass
        for _ in range(1 + self.report_samples * 2):
            with timer.measure("planner"):
                pass
            self.assertLess(len(timer.samples["planner"]), self.report_samples)
        reports = [json.loads(c.args[0].removeprefix("SFA_SYNC_TIMING ")) for c in emit.call_args_list]
        self.assertEqual([r["phase"] for r in reports], ["other", "planner", "planner", "planner"])
        self.assertEqual([r["samples"] for r in reports], [1, 1, self.report_samples, self.report_samples])

    def test_operation_error_propagates_without_extra_sync_or_success_record(self):
        sync, emit = Mock(), Mock()
        timer = self.timing_class(sync, lambda: False, emit, {}, clock=Mock(return_value=0))
        with self.assertRaisesRegex(ValueError, "operation failed"), timer.measure("planner"):
            raise ValueError("operation failed")
        sync.assert_called_once()
        emit.assert_not_called()
        self.assertEqual(timer.samples, {})


if __name__ == "__main__":
    unittest.main()
