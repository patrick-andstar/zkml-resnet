#!/usr/bin/env python3
"""P2A controlled comparison: original binary vs three parameter-read modes.

Groups
  O  target/release/verify_circuit        (unmodified upstream binary)
  A  verify_profile2 --read-mode file      (File, upstream style)
  B  verify_profile2 --read-mode bufreader (BufReader, 1 MiB)
  C  verify_profile2 --read-mode memory    (fs::read + Cursor)

Controls (identical for every sample):
  * one available logical CPU via `taskset -c <cpu>`, RAYON_NUM_THREADS=1
  * a brand-new process per sample, its own working directory
  * one unchanged set of artefacts (SRS / VK / proof / public values / config)
  * the same calibrated outer harness (runwrap.py, wait4-based)

Design notes
  * Groups are interleaved and rotated each round so that no group is always
    first; that spreads slow drift (neighbour load, thermal, page cache) across
    groups instead of letting it bias one of them.
  * Group O resolves `./params_kzg/<k>.params` relative to its cwd. Upstream
    `get_kzg_params` *generates and writes* a fresh SRS when the file is
    missing, so O runs get a private directory whose `params_kzg` symlink
    points at a hash-verified copy, never at the canonical SRS.
  * Artefact digests are taken once, before the batch, and are not part of any
    measured request.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys

REPRO = "/root/autodl-tmp/zkml-repro"
TARGET = REPRO + "/target/release"
RUNROOT = REPRO + "/runs/p2a-calib1"
OUT = os.environ.get("P2A_OUT") or (RUNROOT + "/abc")
ORIG_SRS = RUNROOT + "/orig-srs"
CANON_SRS = REPRO + "/srs/mnist-initial/15.params"
BASE = REPRO + "/runs/mnist-baseline"
CONFIG = REPRO + "/src/examples/mnist/config.msgpack"
WRAP = REPRO + "/scripts/runwrap.py"

CPU = os.environ.get("P2A_CPU", "0")
N = int(os.environ.get("P2A_N", "10"))
PY = sys.executable

ARTEFACTS = [
    ("config", CONFIG),
    ("vkey", BASE + "/vkey"),
    ("proof", BASE + "/proof"),
    ("public_vals", BASE + "/public_vals"),
    ("srs", CANON_SRS),
    ("orig_srs_copy", ORIG_SRS + "/15.params"),
]


def sha256(path):
    h = subprocess.run(["sha256sum", path], capture_output=True, text=True, check=True)
    return h.stdout.split()[0]


def pctile(sorted_vals, q):
    """Linear interpolation on zero-based index (n-1)*q."""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats(vals):
    s = sorted(vals)
    if not s:
        return {}
    return {
        "n": len(s),
        "min": s[0],
        "median": statistics.median(s),
        "p95_linear_n1x095": pctile(s, 0.95),
        "max": s[-1],
        "mean": statistics.fmean(s),
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "all_values": vals,
        "all_below_1000ms": all(v < 1000.0 for v in vals),
    }


def prepare():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ORIG_SRS, exist_ok=True)
    # Private, hash-checked copy of the SRS for group O. A copy (not a symlink
    # to the canonical file) means upstream can never rewrite the canonical SRS.
    dst = ORIG_SRS + "/15.params"
    if not os.path.exists(dst):
        shutil.copyfile(CANON_SRS, dst)
    a, b = sha256(CANON_SRS), sha256(dst)
    if a != b:
        raise SystemExit("orig-srs copy digest mismatch: %s vs %s" % (a, b))
    bins = {
        "verify_circuit_upstream": sha256(TARGET + "/verify_circuit"),
        "verify_profile2": sha256(TARGET + "/verify_profile2"),
    }
    return {"canonical_srs": a, "orig_copy": b, "identical": a == b, "binaries": bins}


def build_cmd(group, cwd, tag):
    env = ["env", "RAYON_NUM_THREADS=1"]
    pin = ["taskset", "-c", CPU]
    if group == "O":
        return pin + env + [
            TARGET + "/verify_circuit",
            CONFIG,
            BASE + "/vkey",
            BASE + "/proof",
            BASE + "/public_vals",
            "kzg",
        ]
    mode = {"A": "file", "B": "bufreader", "C": "memory"}[group]
    return pin + env + [
        TARGET + "/verify_profile2",
        "--read-mode",
        mode,
        CONFIG,
        BASE + "/vkey",
        BASE + "/proof",
        BASE + "/public_vals",
        "kzg",
        BASE + "/params_kzg",
    ]


def parse_inner(stdout_log):
    """Last JSON object printed by the instrumented binary, if any."""
    try:
        with open(stdout_log, "r", errors="replace") as f:
            lines = [l for l in f.read().splitlines() if l.startswith("{")]
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def run_one(group, idx):
    d = "%s/%s/s%02d" % (OUT, group, idx)
    os.makedirs(d, exist_ok=True)
    if group == "O":
        link = d + "/params_kzg"
        if os.path.islink(link) or os.path.exists(link):
            if os.path.islink(link):
                os.unlink(link)
            else:
                shutil.rmtree(link)
        os.symlink(ORIG_SRS, link)
    cmd = build_cmd(group, d, "s%02d" % idx)
    jout = d + "/outer.json"
    so = d + "/stdout.log"
    se = d + "/stderr.log"
    r = subprocess.run(
        [PY, WRAP, "--label", "%s/s%02d" % (group, idx), "--json-out", jout,
         "--stdout-log", so, "--stderr-log", se, "--cwd", d, "--", *cmd],
        capture_output=True, text=True,
    )
    rec = {}
    if os.path.exists(jout):
        with open(jout) as f:
            rec = json.load(f)
    rec["group"] = group
    rec["sample"] = idx
    rec["run_dir"] = d
    rec["wrapper_rc"] = r.returncode
    rec["inner"] = parse_inner(so)
    return rec


def main():
    digests = prepare()
    print(json.dumps({"artefact_sha256": digests}, indent=2))

    groups = ["O", "A", "B", "C"]
    records = []
    for rnd in range(N):
        order = groups[rnd % len(groups):] + groups[: rnd % len(groups)]
        for g in order:
            rec = run_one(g, rnd)
            records.append(rec)
            inner = rec.get("inner") or {}
            print(
                "%-2s s%02d wall=%8.3f ms exit=%s ver=%s"
                % (
                    g,
                    rnd,
                    rec.get("wall_s", -1) * 1000.0,
                    rec.get("exit_code"),
                    inner.get("verify_result", "n/a"),
                ),
                flush=True,
            )

    summary = {}
    for g in groups:
        walls = [r["wall_s"] * 1000.0 for r in records if r["group"] == g]
        rss = [
            r.get("ru_maxrss_bytes") for r in records
            if r["group"] == g and r.get("ru_maxrss_bytes")
        ]
        inner_tot = [
            r["inner"]["total_ms"] for r in records
            if r["group"] == g and r.get("inner")
        ]
        summary[g] = {
            "wall_ms": stats(walls),
            "ru_maxrss_bytes": sorted(rss),
            "inner_total_ms": stats(inner_tot) if inner_tot else "not available upstream",
            "exit_codes": sorted({r.get("exit_code") for r in records if r["group"] == g}),
            "child_cpus_allowed_last": sorted(
                {str(r.get("child_cpus_allowed_last")) for r in records if r["group"] == g}
            ),
            "child_cpus_allowed_observed": sorted(
                {
                    ",".join(r.get("child_cpus_allowed_observed") or [])
                    for r in records
                    if r["group"] == g
                }
            ),
            "child_threads_max": sorted(
                {r.get("child_threads_max") for r in records if r["group"] == g}
            ),
            "rayon_requested": 1,
        }

    out = {
        "design": {
            "groups": {
                "O": "verify_circuit (unmodified upstream)",
                "A": "verify_profile2 --read-mode file",
                "B": "verify_profile2 --read-mode bufreader",
                "C": "verify_profile2 --read-mode memory",
            },
            "n_per_group": N,
            "interleaved": "rotated round-robin",
            "cpu": CPU,
            "rayon_num_threads": 1,
            "wrapper": "runwrap.py method outer-v2/wait4",
            "artefact_sha256": digests,
        },
        "summary": summary,
        "records": records,
    }
    with open(OUT + "/summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n== wall ms ==")
    for g in groups:
        w = summary[g]["wall_ms"]
        print(
            "%-2s n=%d min=%.3f median=%.3f p95=%.3f max=%.3f"
            % (g, w["n"], w["min"], w["median"], w["p95_linear_n1x095"], w["max"])
        )


if __name__ == "__main__":
    main()
