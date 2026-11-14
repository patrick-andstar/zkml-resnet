#!/usr/bin/env python3
"""Self-test for runwrap.py (P2A step 1).

Checks that the outer timer no longer suffers the old 20 ms polling lag:
  1. fast success  : /bin/true            -> wall must be well under 30 ms
  2. controlled    : sleep 0.300          -> wall within 0.29..0.45 s
  3. non-zero exit : sh -c 'exit 7'       -> exit_code == 7
  4. signal        : sh -c 'kill -TERM $$'-> signal == 15, exit_code == 143
  5. repeatability : 10x /bin/true        -> max-min spread < 25 ms,
                                             and no clustering at 20 ms steps
  6. affinity      : taskset -c 3 sleep 0.05
                        -> last sampled Cpus_allowed_list of the child is "3",
                           proving the monitor records the post-exec mask and
                           not the pre-taskset one

Writes a JSON summary; exits non-zero if any check fails.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
WRAP = os.path.join(HERE, "runwrap.py")

CASES = [
    ("fast", ["/bin/true"]),
    ("controlled", ["/bin/sleep", "0.300"]),
    ("exit7", ["/bin/sh", "-c", "exit 7"]),
    ("signal", ["/bin/sh", "-c", "kill -TERM $$"]),
]


def run(name, cmd, tag):
    j = os.path.join(OUT, "selftest_%s_%s.json" % (name, tag))
    o = os.path.join(OUT, "selftest_%s_%s.out" % (name, tag))
    e = os.path.join(OUT, "selftest_%s_%s.err" % (name, tag))
    r = subprocess.run(
        [sys.executable, WRAP, "--label", "selftest-" + name,
         "--json-out", j, "--stdout-log", o, "--stderr-log", e, "--"] + cmd,
        capture_output=True, text=True,
    )
    return json.load(open(j)), r.returncode


def main():
    results = {}
    ok = True

    rec, rc = run("fast", ["/bin/true"], "1")
    w = rec["wall_s"] * 1000
    good = w < 30.0
    ok &= good
    results["fast_ms"] = round(w, 3)
    results["fast_pass"] = good

    rec, rc = run("controlled", ["/bin/sleep", "0.300"], "1")
    w = rec["wall_s"]
    good = 0.29 <= w <= 0.45
    ok &= good
    results["controlled_s"] = round(w, 4)
    results["controlled_pass"] = good

    rec, rc = run("exit7", ["/bin/sh", "-c", "exit 7"], "1")
    good = rec["exit_code"] == 7 and rc == 7 and rec["signal"] == 0
    ok &= good
    results["exit7_code"] = rec["exit_code"]
    results["exit7_wrapper_rc"] = rc
    results["exit7_pass"] = good

    rec, rc = run("signal", ["/bin/sh", "-c", "kill -TERM $$"], "1")
    good = rec["signal"] == 15 and rec["exit_code"] == 143 and rc == 143
    ok &= good
    results["signal"] = rec["signal"]
    results["signal_exit_code"] = rec["exit_code"]
    results["signal_pass"] = good

    walls = []
    for i in range(10):
        rec, rc = run("fast", ["/bin/true"], "r%d" % i)
        walls.append(round(rec["wall_s"] * 1000, 3))
    spread = max(walls) - min(walls)
    good = spread < 25.0
    ok &= good
    results["fast_repeat_ms"] = walls
    results["fast_spread_ms"] = round(spread, 3)
    results["fast_repeat_pass"] = good

    rec, rc = run("affinity", ["taskset", "-c", "3", "/bin/sleep", "0.05"], "1")
    good = rec["child_cpus_allowed_last"] == "3" and rec["child_cpus_allowed_observed"][-1] == "3"
    ok &= good
    results["affinity_observed"] = rec["child_cpus_allowed_observed"]
    results["affinity_last"] = rec["child_cpus_allowed_last"]
    results["affinity_pass"] = good

    results["all_pass"] = bool(ok)
    with open(os.path.join(OUT, "selftest_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
