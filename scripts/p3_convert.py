#!/usr/bin/env python3
"""P3 step 4: PyTorch -> TFLite (LiteRT Torch) + numeric gate.

Reads the pre-registered tolerance from p3-prereg.json and never adjusts a
threshold; it only reports which tier the measured error falls into.

Layout: the repo's own TFLite artefacts are NHWC (`[1,28,28,1]`,
`[1,7,7,320]`), so the delivered model must expose NHWC I/O. LiteRT Torch keeps
the PyTorch NCHW I/O by default; attempt 1 records that default, attempt 2 uses
`to_channel_last_io` and is the one carried forward.

Also recorded on purpose:
  * the aten op histogram of the exported program, so the TFLite op list can be
    explained rather than just counted;
  * a layout negative control: the same bytes fed as a `reshape` instead of a
    `transpose` must produce an O(1) error. Without that, "max_abs is small"
    could come from a test that cannot fail.
"""
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torchvision

REPRO = "/root/autodl-tmp/zkml-repro"
ASSETS = os.path.join(REPRO, "assets")
os.environ["TORCH_HOME"] = os.path.join(ASSETS, "torch-hub")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
    }


def io_of(path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    i = it.get_input_details()[0]
    o = it.get_output_details()[0]
    return {
        "input": {"name": i["name"], "shape": [int(v) for v in i["shape"]], "dtype": str(i["dtype"])},
        "output": {"name": o["name"], "shape": [int(v) for v in o["shape"]], "dtype": str(o["dtype"])},
    }


def main():
    import litert_torch

    prereg = json.load(open(os.path.join(ASSETS, "p3-prereg.json")))
    thr = prereg["thresholds_logits"]
    inputs = prereg["inputs"]
    ref = prereg["reference_logits"]

    report = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": "torchvision resnet18 -> torch.export -> litert_torch.convert -> .tflite (FP32)",
        "prereg_file": os.path.join(ASSETS, "p3-prereg.json"),
        "prereg_thresholds": thr,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "litert_torch": litert_torch.__version__,
        "attempts": [],
    }

    ckpt = os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints", "resnet18-f37072fd.pth")
    ckpt_sha = sha256(ckpt)
    if ckpt_sha != prereg["model"]["checkpoint_sha256"]:
        raise SystemExit("checkpoint changed since pre-registration: " + ckpt_sha)
    report["checkpoint_sha256"] = ckpt_sha

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights).eval()
    sample_nchw = torch.from_numpy(np.load(inputs["real_dog"]["path"]))

    # ---- attempt 1: default (NCHW I/O) -----------------------------------
    p1 = os.path.join(ASSETS, "p3-model-nchw-io.tflite")
    t0 = time.time()
    c1 = litert_torch.convert(model, (sample_nchw,))
    d1 = time.time() - t0
    c1.export(p1)
    a1 = {
        "name": "default_nchw_io",
        "seconds": round(d1, 2),
        "path": p1,
        "bytes": os.path.getsize(p1),
        "sha256": sha256(p1),
        "io": io_of(p1),
        "accepted": False,
        "why_not_accepted": (
            "I/O is NCHW; the repo's own TFLite artefacts use NHWC, so this model "
            "would not be drop-in for the existing converter/circuit path"
        ),
    }
    report["attempts"].append(a1)
    print("attempt1 io:", a1["io"]["input"]["shape"], "->", a1["io"]["output"]["shape"])

    # ---- attempt 2: channel-last I/O -------------------------------------
    wrapped = litert_torch.to_channel_last_io(model, args=[0])
    sample_nhwc = torch.from_numpy(
        np.ascontiguousarray(np.transpose(sample_nchw.numpy(), (0, 2, 3, 1)))
    )
    p2 = os.path.join(ASSETS, "p3-model.tflite")
    t0 = time.time()
    c2 = litert_torch.convert(wrapped, (sample_nhwc,))
    d2 = time.time() - t0
    c2.export(p2)
    a2 = {
        "name": "channel_last_io",
        "seconds": round(d2, 2),
        "path": p2,
        "bytes": os.path.getsize(p2),
        "sha256": sha256(p2),
        "io": io_of(p2),
        "accepted": True,
    }
    report["attempts"].append(a2)
    print("attempt2 io:", a2["io"]["input"]["shape"], "->", a2["io"]["output"]["shape"])

    # ---- aten graph of the exported program ------------------------------
    ep = torch.export.export(wrapped, (sample_nhwc,))
    hist = Counter(str(n.target) for n in ep.graph.nodes if n.op == "call_function")
    report["aten_op_histogram"] = dict(sorted(hist.items()))

    # ---- numeric gate ----------------------------------------------------
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=p2)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outp = it.get_output_details()[0]

    per_input = {}
    for name in inputs:
        x_nchw = np.load(inputs[name]["path"])
        y_ref = np.load(ref[name]["path"])
        # NCHW -> NHWC is a transpose, never a reshape.
        x_nhwc = np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)))
        if list(x_nhwc.shape) != a2["io"]["input"]["shape"]:
            raise SystemExit("input shape mismatch: %s" % (x_nhwc.shape,))
        it.set_tensor(inp["index"], x_nhwc)
        it.invoke()
        y_tfl = it.get_tensor(outp["index"])
        m = metrics(y_ref, y_tfl)
        m["ref_top1"] = int(y_ref.argmax(axis=1)[0])
        m["tfl_top1"] = int(y_tfl.argmax(axis=1)[0])
        m["argmax_identical"] = m["ref_top1"] == m["tfl_top1"]
        per_input[name] = m

    report["per_input"] = per_input
    report["worst"] = max(per_input.items(), key=lambda kv: kv[1]["max_abs"])[0] + ": " + (
        "%.3e" % max(v["max_abs"] for v in per_input.values())
    )

    fails = [n for n, m in per_input.items() if not m["argmax_identical"]]
    over_fail = [n for n, m in per_input.items() if m["max_abs"] > thr["fail"]["max_abs_gt"]]
    over_pass = [n for n, m in per_input.items() if m["max_abs"] > thr["pass"]["max_abs_le"]]
    if fails or over_fail:
        verdict = "FAIL"
    elif over_pass:
        verdict = "PASS_WITH_NOTE"
    else:
        verdict = "PASS"
    report["verdict"] = verdict
    report["argmax_mismatch_inputs"] = fails
    report["inputs_above_pass_threshold"] = over_pass

    # ---- layout negative control ----------------------------------------
    x_nchw = np.load(inputs["real_dog"]["path"])
    y_ref = np.load(ref["real_dog"]["path"])
    x_bad = np.ascontiguousarray(x_nchw.reshape(1, 128, 128, 3))
    it.set_tensor(inp["index"], x_bad)
    it.invoke()
    y_bad = it.get_tensor(outp["index"])
    nb = metrics(y_ref, y_bad)
    nb["ref_top1"] = int(y_ref.argmax(axis=1)[0])
    nb["bad_top1"] = int(y_bad.argmax(axis=1)[0])
    nb["note"] = "reshape instead of transpose; expected to be far worse than the pass threshold"
    report["layout_negative_control"] = nb

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = os.path.join(ASSETS, "p3-convert-report.json")
    with open(p, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== per input (NHWC model, transposed input) ===")
    for n, m in per_input.items():
        print("  %-16s max_abs=%.3e rmse=%.3e top1 %d/%d %s"
              % (n, m["max_abs"], m["rmse"], m["ref_top1"], m["tfl_top1"],
                 "ok" if m["argmax_identical"] else "MISMATCH"))
    print("\n=== layout negative control (reshape instead of transpose) ===")
    print("  max_abs=%.3e top1 %d/%d" % (nb["max_abs"], nb["ref_top1"], nb["bad_top1"]))
    print("\nVERDICT:", verdict)
    print("report:", p)


if __name__ == "__main__":
    main()
