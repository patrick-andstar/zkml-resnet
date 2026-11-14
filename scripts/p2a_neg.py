#!/usr/bin/env python3
"""P2A negative / evidence batch.

Purpose
  1. identify which proof elements are curve points and which are field
     elements, using the decoder's own error as the classifier (no guessing
     from a layout diagram);
  2. show that a tampered proof can be *fully decoded* and only then rejected,
     which is a cryptographic rejection, not a decoding error;
  3. keep decode-stage failures visibly separate (truncation, invalid encoding);
  4. check that a non-canonical public value is rejected by all three read
     modes, and that a missing or truncated SRS fails fast instead of falling
     back to generating a new trusted setup.

Method
  Every candidate is a byte-for-byte copy of the real artefact; the canonical
  files are only ever read. Each run uses `--diag --trace-reads`, so we record
  the verifier's own `Result` plus how many elements it consumed before giving
  up. `n_reads == 105 && bytes_consumed == 3360` means the entire proof was
  decoded before the rejection.
"""
import hashlib
import json
import os
import subprocess
import sys

REPRO = "/root/autodl-tmp/zkml-repro"
BIN = REPRO + "/target/release/verify_profile2"
RUNROOT = REPRO + "/runs/p2a-calib1"
NEG = RUNROOT + "/neg"
BASE = REPRO + "/runs/mnist-baseline"
CONFIG = REPRO + "/src/examples/mnist/config.msgpack"
CANON_SRS = REPRO + "/srs/mnist-initial/15.params"

PROOF = BASE + "/proof"
PVALS = BASE + "/public_vals"
VKEY = BASE + "/vkey"
PARAMS_DIR = BASE + "/params_kzg"

CPU = os.environ.get("P2A_CPU", "0")
MODES = ["file", "bufreader", "memory"]
ELEM = 32


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(result, fully_decoded):
    """Map a run onto the stage that rejected it.

    The error *name* alone is not enough: SHPLONK maps a failed read of its
    trailing points onto `Opening`, so a truncated proof reports `Opening` even
    though nothing about the opening was checked. The reliable discriminator is
    whether the verifier consumed the whole proof (105 reads / 3360 bytes).
    """
    if result == "Ok":
        return "accepted"
    if result == "PANIC":
        return "load_error_panic"
    if not fully_decoded:
        return "decode_error"      # ran out of bytes or an element did not decode
    return "verification_error"    # every element decoded; a check rejected it


def run(mode, proof, pvals, params_dir, tag, vkey=VKEY):
    cmd = [
        "taskset", "-c", CPU, "env", "RAYON_NUM_THREADS=1", BIN,
        "--read-mode", mode, "--diag", "--trace-reads",
        CONFIG, vkey, proof, pvals, "kzg", params_dir,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = {"mode": mode, "tag": tag, "exit_code": r.returncode}
    trace = None
    inner = None
    for line in r.stdout.splitlines():
        if line.startswith("TRACE "):
            trace = json.loads(line[6:])
        elif line.startswith("{"):
            inner = json.loads(line)
    if inner:
        out["verify_result"] = inner.get("verify_result")
        out["n_reads"] = trace["n_reads"] if trace else None
        out["bytes_consumed"] = trace["bytes_consumed"] if trace else None
        out["read_lengths"] = sorted({x[1] for x in trace["reads"]}) if trace else None
    else:
        out["verify_result"] = "PANIC"
        out["stderr_tail"] = r.stderr.strip().splitlines()[-3:]
        out["n_reads"] = None
        out["bytes_consumed"] = None
    out["fully_decoded"] = bool(
        out.get("n_reads") == 105 and out.get("bytes_consumed") == 3360
    )
    out["stage"] = classify(str(out["verify_result"]), out["fully_decoded"])
    return out


def write_copy(name, src, mutate):
    dst = os.path.join(NEG, name)
    with open(src, "rb") as f:
        data = bytearray(f.read())
    mutate(data)
    with open(dst, "wb") as f:
        f.write(data)
    return dst, sha(src), sha(dst)


def main():
    os.makedirs(NEG, exist_ok=True)
    n_elems = os.path.getsize(PROOF) // ELEM
    report = {
        "method": {
            "binary": BIN,
            "binary_sha256": sha(BIN),
            "flags": "--diag --trace-reads --read-mode <mode>",
            "cpu": CPU,
            "rayon_num_threads": 1,
            "element_bytes": ELEM,
            "n_elements": n_elems,
            "stage_rule": (
                "decode_error: verifier reports a transcript/encoding failure at the "
                "element it could not decode; verification_error: the whole proof was "
                "decoded and the failure came from the constraint/opening checks"
            ),
        },
        "artefact_sha256": {
            "proof": sha(PROOF),
            "public_vals": sha(PVALS),
            "vkey": sha(VKEY),
            "srs": sha(CANON_SRS),
            "config": sha(CONFIG),
        },
        "cases": [],
    }

    # ---- 0. positive control, all three modes -----------------------------
    pos = {"case": "positive_control", "runs": []}
    for m in MODES:
        pos["runs"].append(run(m, PROOF, PVALS, PARAMS_DIR, "pos-" + m))
    report["cases"].append(pos)

    # ---- 1. classify proof elements by decoder behaviour -------------------
    # Replacing a 32-byte element with 32 zero bytes is a legal encoding of the
    # field element 0. Where the decoder then reports "invalid point encoding"
    # at that element, the element was a curve point; where the rejection comes
    # later from the constraint/opening checks, it was a field element.
    scan = {
        "case": "element_type_scan",
        "method": "overwrite element i with 32 zero bytes, classify by the error stage",
        "rows": [],
    }
    for idx in range(n_elems):
        def zero_at(data, i=idx):
            data[i * ELEM:(i + 1) * ELEM] = b"\x00" * ELEM
        p, before, after = write_copy("scan_zero_%03d.bin" % idx, PROOF, zero_at)
        r = run("memory", p, PVALS, PARAMS_DIR, "scan-%03d" % idx)
        scan["rows"].append({
            "index": idx,
            "offset": idx * ELEM,
            "sha256_before": before,
            "sha256_after": after,
            "n_reads": r["n_reads"],
            "bytes_consumed": r["bytes_consumed"],
            "verify_result": r["verify_result"],
            "stage": r["stage"],
            "fully_decoded": r["fully_decoded"],
            "exit_code": r["exit_code"],
        })
    # A decode error reported at read #i+1 means element i itself could not be
    # decoded, so the all-zero bytes were not accepted as that element's type.
    scan["point_like"] = [
        r["index"] for r in scan["rows"]
        if r["stage"] == "decode_error" and r["n_reads"] == r["index"] + 1
    ]
    # Zeroing a field element is legal, so the whole proof still decodes and the
    # rejection comes from a later check.
    scan["field_like"] = [
        r["index"] for r in scan["rows"]
        if r["stage"] == "verification_error" and r["fully_decoded"]
    ]
    # Rejected later than the element itself but before the end: the zero bytes
    # were accepted as a point encoding, so this probe cannot settle the type.
    scan["ambiguous"] = [
        r["index"] for r in scan["rows"]
        if r["stage"] == "decode_error" and r["n_reads"] > r["index"] + 1
    ] + [
        r["index"] for r in scan["rows"]
        if r["stage"] == "verification_error" and not r["fully_decoded"]
    ]
    report["cases"].append(scan)

    point_slot = scan["point_like"][0] if scan["point_like"] else 0
    other_point = scan["point_like"][1] if len(scan["point_like"]) > 1 else point_slot
    scalar_slot = scan["field_like"][0] if scan["field_like"] else None

    cases = []

    def add(name, what, src, mutate, proof_path_used, pvals_path_used):
        p, b, a = write_copy(name, src, mutate)
        cases.append({
            "case": name.replace(".bin", ""),
            "what": what,
            "file": p,
            "sha256_before": b,
            "sha256_after": a,
            "proof": proof_path_used(p),
            "pvals": pvals_path_used(p),
            "runs": [],
        })

    # ---- 2. public values, all three modes -------------------------------
    add(
        "neg_public_alt.bin",
        "public value 0 changed to another canonical field element (low bit flipped)",
        PVALS, lambda d: d.__setitem__(0, d[0] ^ 0x01),
        lambda p: PROOF, lambda p: p,
    )
    add(
        "neg_public_noncanonical.bin",
        "public value 0 replaced by 0xff*32, above the field modulus",
        PVALS, lambda d: d.__setitem__(slice(0, 32), b"\xff" * 32),
        lambda p: PROOF, lambda p: p,
    )

    # ---- 3. proof element that is a curve point, legal encoding ----------
    add(
        "neg_proof_point_slot_valid_wrong_point.bin",
        "element %d (a point) replaced by the bytes of element %d: still a valid "
        "point encoding, but not the committed point" % (point_slot, other_point),
        PROOF,
        lambda d, i=point_slot, j=other_point: d.__setitem__(
            slice(i * ELEM, (i + 1) * ELEM), bytes(d[j * ELEM:(j + 1) * ELEM])
        ),
        lambda p: p, lambda p: PVALS,
    )
    add(
        "neg_proof_point_slot_ff.bin",
        "element %d (a point) replaced by 0xff*32: not a valid point encoding"
        % point_slot,
        PROOF,
        lambda d, i=point_slot: d.__setitem__(slice(i * ELEM, (i + 1) * ELEM), b"\xff" * ELEM),
        lambda p: p, lambda p: PVALS,
    )

    # ---- 4. proof element that is a field element, legal encoding --------
    if scalar_slot is not None:
        add(
            "neg_proof_scalar_slot_lowbit.bin",
            "element %d (a field element) low bit flipped: another canonical "
            "field element" % scalar_slot,
            PROOF,
            lambda d, i=scalar_slot: d.__setitem__(i * ELEM, d[i * ELEM] ^ 0x01),
            lambda p: p, lambda p: PVALS,
        )

    # ---- 5. truncated proof ---------------------------------------------
    add(
        "neg_proof_truncated_32.bin",
        "last 32 bytes removed: the verifier runs out of proof",
        PROOF, lambda d: d.__delitem__(slice(len(d) - ELEM, len(d))),
        lambda p: p, lambda p: PVALS,
    )

    for c in cases:
        for m in MODES:
            c["runs"].append(run(m, c["proof"], c["pvals"], PARAMS_DIR, c["case"] + "-" + m))
        report["cases"].append(c)

    # ---- 6. broken trusted setup: fail fast, never regenerate -------------
    missing_dir = os.path.join(NEG, "srs_missing")
    os.makedirs(missing_dir, exist_ok=True)
    before_listing = sorted(os.listdir(missing_dir))
    r = run("memory", PROOF, PVALS, missing_dir, "srs_missing")
    r["dir_before"] = before_listing
    r["dir_after"] = sorted(os.listdir(missing_dir))
    r["created_anything"] = r["dir_after"] != before_listing
    report["cases"].append({
        "case": "srs_missing_fails_fast",
        "what": "params dir has no 15.params; must refuse, must not generate one",
        "runs": [r],
    })

    trunc_dir = os.path.join(NEG, "srs_truncated")
    os.makedirs(trunc_dir, exist_ok=True)
    tpath = os.path.join(trunc_dir, "15.params")
    with open(CANON_SRS, "rb") as f:
        head = f.read(4096)
    with open(tpath, "wb") as f:
        f.write(head)
    r = run("memory", PROOF, PVALS, trunc_dir, "srs_truncated")
    r["truncated_srs_sha256"] = sha(tpath)
    r["truncated_srs_bytes"] = os.path.getsize(tpath)
    r["canonical_srs_sha256_after"] = sha(CANON_SRS)
    report["cases"].append({
        "case": "srs_truncated_load_error",
        "what": "first 4096 bytes of the SRS only: load error, not a verification error",
        "runs": [r],
    })

    vk_dir = os.path.join(NEG, "vkey_truncated")
    os.makedirs(vk_dir, exist_ok=True)
    vpath = os.path.join(vk_dir, "vkey")
    with open(VKEY, "rb") as f:
        vk_head = f.read(1024)
    with open(vpath, "wb") as f:
        f.write(vk_head)
    r = run("memory", PROOF, PVALS, PARAMS_DIR, "vkey_truncated", vkey=vpath)
    r["truncated_vkey_sha256"] = sha(vpath)
    r["truncated_vkey_bytes"] = os.path.getsize(vpath)
    r["canonical_vkey_sha256_after"] = sha(VKEY)
    report["cases"].append({
        "case": "vkey_truncated_load_error",
        "what": "first 1024 bytes of the VK only: key-load error, distinct from a rejected proof",
        "runs": [r],
    })

    with open(RUNROOT + "/neg_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "n_elements": n_elems,
        "point_like_count": len(scan["point_like"]),
        "point_like": scan["point_like"],
        "field_like_count": len(scan["field_like"]),
        "field_like_first_last": [scan["field_like"][:3], scan["field_like"][-3:]],
        "ambiguous": scan["ambiguous"],
        "point_slot_used": point_slot,
        "other_point_used": other_point,
        "scalar_slot_used": scalar_slot,
    }, indent=2))
    for c in report["cases"]:
        if "runs" not in c:
            continue
        print("\n## " + c["case"])
        for r in c["runs"]:
            print(
                "  %-9s exit=%-3s n_reads=%-4s bytes=%-5s decoded=%-5s stage=%-18s result=%s"
                % (
                    r["mode"], r["exit_code"], r.get("n_reads"), r.get("bytes_consumed"),
                    r.get("fully_decoded"), r.get("stage"),
                    str(r.get("verify_result"))[:58],
                )
            )
        for k in ("created_anything", "dir_after", "truncated_srs_bytes"):
            if k in c["runs"][0]:
                print("  %s = %s" % (k, c["runs"][0][k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
