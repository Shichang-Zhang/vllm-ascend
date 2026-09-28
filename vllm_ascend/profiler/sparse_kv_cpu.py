# SPDX-License-Identifier: Apache-2.0
"""CPU-only collection/analysis. Also runnable directly, without torch or NPU.

Snapshots are deliberately taken outside the planner. All times in planner
records and normalized scheduler events use Linux CLOCK_MONOTONIC.
"""

import argparse
import bisect
import json
import os
import re
import socket
import subprocess
import time
from collections import defaultdict
from contextlib import suppress
from pathlib import Path, PurePosixPath

OMPT_SET_ALWAYS = 5
NANOSECONDS_PER_SECOND = 1_000_000_000
NANOSECONDS_PER_MICROSECOND = 1000


def read_file(path):
    try:
        return {"text": Path(path).read_text()}
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def status_fields(text):
    wanted = {
        "Name",
        "Pid",
        "Tgid",
        "NSpid",
        "Cpus_allowed_list",
        "Mems_allowed_list",
        "voluntary_ctxt_switches",
        "nonvoluntary_ctxt_switches",
    }
    return {
        key: value.strip()
        for line in text.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
        if key in wanted
    }


def mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def cgroup_locations(membership, mountinfo):
    """Resolve v1/v2 membership against mount roots, including cgroup namespaces.

    Return paths in the target's mount namespace, not assumed host paths.
    """
    groups = [line.split(":", 2) for line in membership.splitlines() if line.count(":") >= 2]
    result = []
    for line in mountinfo.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, filesystem = left.split(), right.split()
        if len(fields) < 5 or len(filesystem) < 3 or filesystem[0] not in {"cgroup", "cgroup2"}:
            continue
        root, mount = PurePosixPath(mount_path(fields[3])), PurePosixPath(mount_path(fields[4]))
        for _, controllers, member in groups:
            if filesystem[0] == "cgroup2":
                if controllers:
                    continue
            elif not set(controllers.split(",")) & set(filesystem[2].split(",")):
                continue
            member_path = PurePosixPath(member)
            if ".." in member_path.parts:
                continue  # Ancestor outside this cgroup namespace is not accessible.
            try:
                relative = member_path.relative_to(root)
            except ValueError:
                if member_path != PurePosixPath("/"):
                    continue
                relative = PurePosixPath(".")  # Namespace membership '/' at a subtree mount.
            result.append(
                {
                    "version": 2 if filesystem[0] == "cgroup2" else 1,
                    "controllers": controllers,
                    "path": str(mount / relative),
                    "mount": str(mount),
                }
            )
    return result


def snapshot(pid, memory=False):
    """Best-effort read-only snapshot; unavailable data retains its error."""
    proc = Path("/proc") / str(pid)
    result = {
        "pid": pid,
        "hostname": socket.gethostname(),
        "begin_ns": time.monotonic_ns(),
        "unix_ns": time.time_ns(),
        "status": read_file(proc / "status"),
        "cgroup_membership": read_file(proc / "cgroup"),
        "mountinfo": read_file(proc / "mountinfo"),
    }
    result["identity"] = status_fields(result["status"].get("text", ""))
    result["time_namespace_offsets"] = read_file(proc / "timens_offsets")
    result["threads"] = {}
    try:
        for task in (proc / "task").iterdir():
            status = read_file(task / "status")
            result["threads"][task.name] = {
                "status": status_fields(status.get("text", "")),
                "status_error": status.get("error"),
                "schedstat": read_file(task / "schedstat"),
            }
    except OSError as exc:
        result["threads_error"] = str(exc)
    result["cgroups"] = []
    for location in cgroup_locations(result["cgroup_membership"].get("text", ""), result["mountinfo"].get("text", "")):
        names = (
            (
                "cpu.max",
                "cpu.stat",
                "cpu.pressure",
                "cpuset.cpus.effective",
                "cpuset.mems.effective",
                "memory.events",
                "memory.pressure",
                "memory.current",
                "memory.max",
                "memory.numa_stat",
            )
            if location["version"] == 2
            else (
                "cpu.cfs_quota_us",
                "cpu.cfs_period_us",
                "cpu.stat",
                "cpuset.cpus",
                "cpuset.mems",
                "cpuset.effective_cpus",
                "cpuset.effective_mems",
                "memory.failcnt",
                "memory.usage_in_bytes",
                "memory.limit_in_bytes",
                "memory.numa_stat",
            )
        )
        path = PurePosixPath(location["path"])
        while True:
            # The target root accounts for a different mount namespace when collecting from a host.
            base = proc / "root" / str(path).lstrip("/")
            result["cgroups"].append(
                {**location, "path": str(path), "files": {name: read_file(base / name) for name in names}}
            )
            if str(path) == location["mount"] or path == path.parent:
                break
            path = path.parent
    result["ancestor_visibility"] = "Only mounted ancestors are visible; hidden parent limits are unknown."
    result["numa_nodes"] = {}
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        result["numa_nodes"][node.name] = {
            "cpulist": read_file(node / "cpulist"),
            "distance": read_file(node / "distance"),
        }
    maps = read_file(proc / "maps")
    result["openmp_mappings"] = [
        line
        for line in maps.get("text", "").splitlines()
        if "libomp" in line or "libgomp" in line or "planner_ompt" in line
    ]
    if memory:
        result["maps"] = maps
        result["numa_maps"] = read_file(proc / "numa_maps")
    result["end_ns"] = time.monotonic_ns()
    return result


def buffer_mappings(buffers, context):
    """VMA evidence, not an exact per-buffer page query (VMAs may be shared)."""
    numa = {int(line.split()[0], 16): line for line in context.get("numa_maps", {}).get("text", "").splitlines()}
    mappings = []
    for line in context.get("maps", {}).get("text", "").splitlines():
        start, end = (int(value, 16) for value in line.split()[0].split("-"))
        mappings.append((start, end, line))
    return [
        {
            **buffer,
            "scope": "overlapping VMA, not exact buffer pages",
            "mappings": [
                {"maps": line, "numa_maps": numa.get(start)}
                for start, end, line in mappings
                if start < buffer["address"] + buffer["bytes"] and end > buffer["address"]
            ],
        }
        for buffer in buffers
    ]


def integer_fields(file):
    result = {}
    for line in file.get("text", "").splitlines():
        fields = line.split()
        if len(fields) == 2:
            with suppress(ValueError):
                result[fields[0]] = int(fields[1])
    return result


def cgroup_deltas(before, after):
    previous = {(group["controllers"], group["path"]): group for group in before.get("cgroups", [])}
    result = []
    for group in after.get("cgroups", []):
        old = previous.get((group["controllers"], group["path"]))
        if old is None:
            continue
        left = integer_fields(old["files"].get("cpu.stat", {}))
        right = integer_fields(group["files"].get("cpu.stat", {}))
        result.append(
            {
                "path": group["path"],
                "version": group["version"],
                "cpu_stat_delta": {key: value - left[key] for key, value in right.items() if key in left},
                "scope": "cgroup/window aggregate; not a per-thread or per-callback attribution",
            }
        )
    return result


def normalize_perf(text):
    """Parse `perf script --ns -F time,event,trace`; retain completeness warnings."""
    events = []
    rejected = []
    pattern = re.compile(r"^\s*(\d+)\.(\d+):\s+(\S+):\s+(.*)$")
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            rejected.append(line)
            continue
        seconds, fraction, name, payload = match.groups()
        fields = dict(re.findall(r"(\w+)=([^\s]+)", payload))
        event = {"ts_ns": int(seconds) * NANOSECONDS_PER_SECOND + int(fraction.ljust(9, "0")[:9])}
        try:
            if name.endswith("sched_switch"):
                event.update(
                    kind="switch",
                    prev=int(fields["prev_pid"]),
                    next=int(fields["next_pid"]),
                    runnable=fields["prev_state"].startswith("R"),
                )
            elif name.endswith(("sched_wakeup", "sched_wakeup_new")):
                event.update(kind="wakeup", tid=int(fields["pid"]))
            else:
                rejected.append(line)
                continue
        except (KeyError, ValueError):
            rejected.append(line)
            continue
        events.append(event)
    return {
        "clock": "CLOCK_MONOTONIC",
        "clock_requires_verification": True,
        "events": sorted(events, key=lambda event: event["ts_ns"]),
        "unparsed_lines": rejected,
        "note": "Use the collector metadata to verify clock, host and event loss.",
    }


def scheduler_intervals(events):
    """Conservative reconstruction: prefixes/tails without transitions are unknown."""
    state = {}
    intervals = defaultdict(list)

    def transition(tid, timestamp, new_state):
        if tid == 0:
            return
        if tid in state:
            begin, old_state = state[tid]
            if timestamp > begin:
                intervals[tid].append((begin, timestamp, old_state))
        state[tid] = (timestamp, new_state)

    for event in events:
        timestamp = event["ts_ns"]
        if event["kind"] == "switch":
            transition(event["prev"], timestamp, "runnable" if event["runnable"] else "blocked")
            transition(event["next"], timestamp, "running")
        elif event["kind"] == "wakeup":
            # A duplicate wakeup must not turn an already running task into runnable.
            if event["tid"] not in state or state[event["tid"]][1] != "running":
                transition(event["tid"], timestamp, "runnable")
    return dict(intervals)


def scheduler_overlap(intervals, begin, end, starts=None):
    totals = {"running_ns": 0, "runnable_ns": 0, "blocked_ns": 0}
    first = max(0, bisect.bisect_right(starts, begin) - 1) if starts else 0
    for index in range(first, len(intervals)):
        left, right, state = intervals[index]
        if left >= end:
            break
        duration = max(0, min(right, end) - max(left, begin))
        totals[f"{state}_ns"] += duration
    totals["unknown_ns"] = max(0, end - begin - sum(totals.values()))
    return totals


def analyze(document, scheduler=None, tid_map=None):
    intervals = scheduler_intervals(scheduler["events"]) if scheduler else {}
    starts = {tid: [interval[0] for interval in values] for tid, values in intervals.items()}

    def host_tid(tid):
        if tid_map is None:
            return tid
        mapped = tid_map.get(str(tid))
        return None if mapped is None else int(mapped)

    result = {
        "identity": {
            key: document.get(key)
            for key in ("hostname", "pid", "dp_rank", "tp_rank", "dp_size", "tp_size", "rank_trace_name")
        },
        "calls": [],
        "cgroup_deltas": cgroup_deltas(document.get("system_start", {}), document.get("system_stop", {})),
        "limitations": [
            "Call indices are local to a graph payload, not globally aligned decode steps.",
            "Different DP groups may have different requests and miss counts.",
            "CPU time includes memory stalls/spinning; it is not useful instruction time.",
            "Wall minus CPU time does not isolate scheduler delay or NUMA latency.",
        ],
    }
    if scheduler:
        result["scheduler_quality"] = {key: value for key, value in scheduler.items() if key != "events"}
        result["limitations"].append(
            "Scheduler intervals are on-CPU/runnable/blocked observations, not pure OpenMP runtime overhead; "
            "event loss or a wrong host-TID mapping invalidates attribution."
        )
    for record in document.get("records", []):
        for call in record.get("details", []):
            workers = []
            for thread in call["threads"]:
                begin, end = thread["work_begin_ns"], thread["work_end_ns"]
                if begin <= 0 or end < begin:
                    continue
                cpu = (
                    thread["cpu_end_ns"] - thread["cpu_begin_ns"]
                    if thread["cpu_begin_ns"] >= 0 and thread["cpu_end_ns"] >= thread["cpu_begin_ns"]
                    else None
                )
                waits = thread["ompt_wait_intervals_ns"]
                complete = (
                    thread["ompt_support"] == OMPT_SET_ALWAYS
                    and not thread["ompt_dropped"]
                    and all(right >= left > 0 for left, right in waits)
                )
                worker = {
                    **thread,
                    "work_wall_ns": end - begin,
                    "work_cpu_ns": cpu,
                    "entry_delay_ns": thread["entry_ns"] - call["parallel_begin_ns"],
                    "tail_to_parallel_return_ns": call["parallel_end_ns"] - end,
                    "observed_wait_ns": sum(right - left for left, right in waits if right >= left > 0),
                    "observed_wait_in_parallel_ns": sum(
                        max(0, min(right, call["parallel_end_ns"]) - max(left, call["parallel_begin_ns"]))
                        for left, right in waits
                        if right >= left > 0
                    ),
                    "wait_complete": complete,
                    "wall_minus_cpu_ns": None if cpu is None else end - begin - cpu,
                }
                if scheduler:
                    tid = host_tid(thread["tid"])
                    worker["scheduler_tid"] = tid
                    worker["scheduler_work"] = scheduler_overlap(intervals.get(tid, []), begin, end, starts.get(tid))
                    worker["scheduler_entry"] = scheduler_overlap(
                        intervals.get(tid, []), call["parallel_begin_ns"], thread["entry_ns"], starts.get(tid)
                    )
                    worker["scheduler_tail"] = scheduler_overlap(
                        intervals.get(tid, []), end, call["parallel_end_ns"], starts.get(tid)
                    )
                workers.append(worker)
            result["calls"].append(
                {
                    "graph_id": record.get("graph_id"),
                    "layer_id": record.get("layer_id"),
                    "call_index": call["call_index"],
                    "begin_ns": call["begin_ns"],
                    "num_reqs": record["num_reqs"],
                    "topk": record["topk"],
                    "callback_ns": call["end_ns"] - call["begin_ns"],
                    "actual_threads": call.get("actual_threads"),
                    "serial_setup_ns": call["parallel_begin_ns"] - call["begin_ns"]
                    if call["parallel_begin_ns"] > 0
                    else None,
                    "parallel_ns": call["parallel_end_ns"] - call["parallel_begin_ns"]
                    if call["parallel_begin_ns"] > 0
                    else None,
                    "last_worker_tid": max(workers, key=lambda worker: worker["work_end_ns"])["tid"]
                    if workers
                    else None,
                    "workers": workers,
                }
            )
            if scheduler and call["parallel_begin_ns"] > 0:
                tid = host_tid(call.get("callback_tid"))
                result["calls"][-1]["scheduler_serial_setup"] = scheduler_overlap(
                    intervals.get(tid, []), call["begin_ns"], call["parallel_begin_ns"], starts.get(tid)
                )
    result["calls"].sort(key=lambda call: call["callback_ns"], reverse=True)
    result["dropped_calls"] = sum(record.get("dropped", 0) for record in document.get("records", []))
    return result


def chrome_trace(document):
    events = []
    for record in document.get("records", []):
        for call in record.get("details", []):
            for thread in call["threads"]:
                spans = [("planner work", thread["work_begin_ns"], thread["work_end_ns"])]
                spans += [
                    ("OMPT parallel barrier wait", left, right) for left, right in thread["ompt_wait_intervals_ns"]
                ]
                for name, begin, end in spans:
                    if end >= begin > 0:
                        events.append(
                            {
                                "name": name,
                                "ph": "X",
                                "ts": begin / NANOSECONDS_PER_MICROSECOND,
                                "dur": (end - begin) / NANOSECONDS_PER_MICROSECOND,
                                "pid": document["pid"],
                                "tid": thread["tid"],
                                "args": {
                                    "dp_rank": document.get("dp_rank"),
                                    "layer": record.get("layer_id"),
                                    "call_index": call["call_index"],
                                    "rows": thread["rows"],
                                },
                            }
                        )
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("snapshot")
    collect.add_argument("--pid", type=int, required=True)
    collect.add_argument("--memory", action="store_true")
    collect.add_argument("--output", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("trace", type=Path)
    analyze_parser.add_argument("--scheduler", type=Path)
    analyze_parser.add_argument("--tid-map", type=Path, help="JSON mapping container TID to scheduler/host TID")
    analyze_parser.add_argument(
        "--same-host",
        action="store_true",
        help="Confirm differing hostnames refer to one physical host and monotonic clock",
    )
    analyze_parser.add_argument(
        "--host-tids", action="store_true", help="Confirm planner TIDs already use the host PID namespace"
    )
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--chrome", type=Path)
    perf_parser = commands.add_parser("record-sched", help="Optional host-wide capture; requires kernel permission")
    perf_parser.add_argument("--seconds", type=float, default=10)
    perf_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        output = snapshot(args.pid, args.memory)
    elif args.command == "analyze":
        document = json.loads(args.trace.read_text())
        scheduler = json.loads(args.scheduler.read_text()) if args.scheduler else None
        if scheduler and scheduler.get("hostname") != document.get("hostname") and not args.same_host:
            parser.error("scheduler and planner must be captured on the same host/namespace identity")
        if scheduler and scheduler.get("clock") != "CLOCK_MONOTONIC":
            parser.error("scheduler clock must be CLOCK_MONOTONIC")
        if scheduler and "events" not in scheduler:
            parser.error("scheduler capture has no decoded events; inspect its error/returncode")
        if scheduler and not args.tid_map and not args.host_tids:
            parser.error("provide --tid-map or confirm --host-tids; tracepoints use host TIDs")
        offsets = document.get("system_start", {}).get("time_namespace_offsets", {}).get("text", "")
        if scheduler and any(
            any(int(value) != 0 for value in line.split()[1:])
            for line in offsets.splitlines()
            if line.startswith("monotonic")
        ):
            parser.error("nonzero monotonic time namespace offset: convert clocks before correlating")
        mapping = json.loads(args.tid_map.read_text()) if args.tid_map else None
        output = analyze(document, scheduler, mapping)
        if args.chrome:
            args.chrome.write_text(json.dumps(chrome_trace(document)))
    else:
        if args.seconds <= 0:
            parser.error("--seconds must be positive")
        raw = args.output.with_suffix(".perf.data")
        command = [
            "perf",
            "record",
            "-a",
            "--clockid",
            "mono",
            "-e",
            "sched:sched_switch",
            "-e",
            "sched:sched_wakeup",
            "-e",
            "sched:sched_wakeup_new",
            "-o",
            str(raw),
            "--",
            "sleep",
            str(args.seconds),
        ]
        output = {
            "hostname": socket.gethostname(),
            "clock": "CLOCK_MONOTONIC",
            "command": command,
            "collector_pid": os.getpid(),
            "begin_ns": time.monotonic_ns(),
        }
        try:
            capture = subprocess.run(command, capture_output=True, text=True)
            output.update(returncode=capture.returncode, stderr=capture.stderr, end_ns=time.monotonic_ns())
            if capture.returncode == 0:
                decoded = subprocess.run(
                    ["perf", "script", "-i", str(raw), "--ns", "-F", "time,event,trace"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                output.update(normalize_perf(decoded.stdout))
                output["clock_requires_verification"] = False
                output["decode_stderr"] = decoded.stderr
                output["event_loss_suspected"] = (
                    bool(output["unparsed_lines"]) or "lost" in (capture.stderr + decoded.stderr).lower()
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            output["error"] = str(exc)
    args.output.write_text(json.dumps(output, indent=2))
    if args.command == "record-sched" and (output.get("error") or output.get("returncode") != 0):
        parser.exit(1, f"Scheduler capture unavailable; details saved to {args.output}\n")


if __name__ == "__main__":
    main()
