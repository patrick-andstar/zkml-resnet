#!/usr/bin/env python3
"""Outer-process timer for one verification request (P2A method v2).

Why this exists: the previous wrapper polled `Popen.poll()` every 20 ms and
sampled RSS on the same path, so the measured wall carried a periodic
detection lag and the RSS peak could be missed.

Method v2:
  * log files are opened and fsync-irrelevant setup is done BEFORE t0;
  * t0 = time.monotonic() immediately before spawn;
  * the child is spawned with subprocess.Popen but never polled/waited through
    Popen; it is reaped exactly once with os.wait4(pid, 0);
  * t1 = time.monotonic() immediately after wait4 returns;
  * the resource monitor (per-process VmHWM + cgroup memory.current) runs in a
    separate thread and is joined only after t1, so it can never delay exit
    detection;
  * exit-time resource usage comes from wait4's rusage (ru_maxrss in KiB ->
    bytes). Linux exposes the maximum RSS of the *reaped child process*, which
    does not accumulate descendants' concurrent memory.

What is still included in wall: exec + dynamic linking, all real work, program
output being written, destructors and process teardown, plus small scheduler
noise. This is a full request wall, not a "core verify" number.

Exit status: the wrapper exits with the child's exit code. A child killed by
signal N yields 128+N and `"signal": N` in the record. Negative-case acceptance
must be judged by the caller, never by "non-zero means pass".
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time


def cgroup_mem():
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read().strip())
    except OSError:
        return None


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return f.read().split()[:3]
    except OSError:
        return None


def read_status(pid, keys):
    out = {}
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in keys:
                    out[k] = v.strip()
    except OSError:
        pass
    return out


def count_threads(pid):
    try:
        return len(os.listdir("/proc/%d/task" % pid))
    except OSError:
        return None


def self_affinity():
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--stdout-log", required=True)
    ap.add_argument("--stderr-log", required=True)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--poll-ms", type=int, default=5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("no command")

    # Everything below up to t0 is setup and is excluded from the wall clock.
    with open(a.stdout_log, "wb") as so, open(a.stderr_log, "wb") as se:
        mem_before = cgroup_mem()
        stop = threading.Event()
        samples = {"peak_vmhwm": 0, "peak_cgroup": mem_before or 0,
                   "cpu_seen": [], "cpu_last": None, "threads_max": 0}

        def monitor(pid):
            # Sampling must continue for the whole life of the child: the very
            # first sample can land before `taskset` has applied the mask (it
            # sets affinity and then execs), so a single early read would be
            # wrong evidence. The last read before exit is the useful one.
            while not stop.is_set():
                st = read_status(pid, {"VmHWM", "Cpus_allowed_list"})
                hwm = st.get("VmHWM")
                if hwm:
                    try:
                        samples["peak_vmhwm"] = max(
                            samples["peak_vmhwm"], int(hwm.split()[0]) * 1024
                        )
                    except ValueError:
                        pass
                cpu = st.get("Cpus_allowed_list")
                if cpu:
                    samples["cpu_last"] = cpu
                    if cpu not in samples["cpu_seen"]:
                        samples["cpu_seen"].append(cpu)
                nt = count_threads(pid)
                if nt:
                    samples["threads_max"] = max(samples["threads_max"], nt)
                cg = cgroup_mem()
                if cg:
                    samples["peak_cgroup"] = max(samples["peak_cgroup"], cg)
                stop.wait(a.poll_ms / 1000.0)

        load_before = loadavg()
        t0 = time.monotonic()
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=so,
            stderr=se,
            cwd=a.cwd,
            close_fds=True,
        )
        th = threading.Thread(target=monitor, args=(p.pid,), daemon=True)
        th.start()
        pid, status, ru = os.wait4(p.pid, 0)
        t1 = time.monotonic()
        stop.set()
        th.join(timeout=2.0)
        # Keep Popen from trying to reap again in __del__.
        p.returncode = os.waitstatus_to_exitcode(status)

    wall = t1 - t0
    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else 0
    raw = os.waitstatus_to_exitcode(status)
    # Shell convention for the reported code: 128+N for a signalled child.
    # The raw waitstatus_to_exitcode value (-N) is kept for traceability.
    code = 128 + sig if sig else raw
    rec = {
        "label": a.label,
        "method": "outer-v2/wait4",
        "cmd": cmd,
        "cwd": a.cwd,
        "wall_s": round(wall, 6),
        "exit_code": code,
        "exit_code_raw": raw,
        "signal": sig,
        "raw_wait_status": status,
        "ru_maxrss_bytes": ru.ru_maxrss * 1024,
        "ru_utime_s": round(ru.ru_utime, 6),
        "ru_stime_s": round(ru.ru_stime, 6),
        "ru_maxrss_semantics": "peak RSS of the reaped child process only",
        "monitor_peak_vmhwm_bytes": samples["peak_vmhwm"] or None,
        "monitor_peak_cgroup_bytes": samples["peak_cgroup"] or None,
        "child_cpus_allowed_last": samples["cpu_last"],
        "child_cpus_allowed_observed": samples["cpu_seen"],
        "child_threads_max": samples["threads_max"],
        "note_affinity": (
            "sampled from /proc/<pid>/status of the target pid; the first sample "
            "may precede taskset, the last one is after exec"
        ),
        "wrapper_affinity": self_affinity(),
        "poll_ms": a.poll_ms,
        "loadavg_before": load_before,
        "loadavg_after": loadavg(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.json_out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec))
    return code


if __name__ == "__main__":
    sys.exit(main())
