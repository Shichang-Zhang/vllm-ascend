# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Temporary serialized timings. These are wall times, not kernel durations."""

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

TIMING_REPORT_SAMPLES = 32


class SFASyncTiming:
    def __init__(
        self,
        synchronize: Callable[[], None],
        is_capturing: Callable[[], bool],
        emit: Callable[[str], None],
        identity: dict,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.synchronize = synchronize
        self.is_capturing = is_capturing
        self.emit = emit
        self.identity = {**identity, "pid": os.getpid()}
        self.clock = clock
        self.samples: dict[str, list[tuple[float, float, float]]] = {}
        self.calls: dict[str, int] = {}

    @contextmanager
    def measure(self, phase: str, *, capturing: bool = False) -> Iterator[None]:
        # Python synchronization is illegal during capture. Captured Python code
        # is not rerun at replay, so timing it would not measure graph execution.
        if capturing or self.is_capturing():
            yield
            return
        before = self.clock()
        self.synchronize()
        ready = self.clock()
        yield
        submitted = self.clock()
        self.synchronize()
        completed = self.clock()
        samples = self.samples.setdefault(phase, [])
        samples.append(((ready - before) / 1000, (submitted - ready) / 1000, (completed - submitted) / 1000))
        self.calls[phase] = self.calls.get(phase, 0) + 1
        # Emit the first observation too, so short runs still produce evidence.
        if self.calls[phase] != 1 and len(samples) < TIMING_REPORT_SAMPLES:
            return
        report = {
            **self.identity,
            "phase": phase,
            "calls": self.calls[phase],
            "samples": len(samples),
            "scope": "serialized wall time; collective phases include peer waiting",
        }
        for name, values in (
            ("prior_work_us", [s[0] for s in samples]),
            ("host_call_us", [s[1] for s in samples]),
            ("completion_wait_us", [s[2] for s in samples]),
            ("phase_us", [s[1] + s[2] for s in samples]),
        ):
            report[name] = {"mean": sum(values) / len(values), "max": max(values)}
        samples.clear()
        self.emit("SFA_SYNC_TIMING " + json.dumps(report, sort_keys=True))
