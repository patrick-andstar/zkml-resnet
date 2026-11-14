#!/usr/bin/env python3
"""P3 step 3: fix model identity, build test inputs, pre-register tolerance.

Order matters: this script runs BEFORE any conversion. The tolerance file it
writes is the record that the acceptance thresholds were fixed in advance, not
chosen after seeing the result.

Model identity (plan §P3 default branch: no original target artefact supplied):
  torchvision resnet18, 1000 classes, IMAGENET1K_V1 weights.
  This is an *independent baseline*, not proof of identity with any earlier
  route; the checkpoint digest is recorded so the claim stays checkable.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

REPRO = "/root/autodl-tmp/zkml-repro"
ASSETS = os.path.join(REPRO, "assets")
INPUTS = os.path.join(ASSETS, "p3-inputs")
os.makedirs(INPUTS, exist_ok=True)
os.environ["TORCH_HOME"] = os.path.join(ASSETS, "torch-hub")

H = 128
W = 128


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_inputs():
    """Five deterministic inputs. One is traceable to a real photo file."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    made = {}

    dog = os.path.join(ASSETS, "p3-dog.jpg")
    if not os.path.exists(dog):
        raise SystemExit("missing source image: " + dog)
    img = Image.open(dog).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(H),
        transforms.CenterCrop(H),
        transforms.ToTensor(),
    ])
    x = tf(img).numpy()  # [3,H,W] in [0,1]
    x = (x - mean[:, None, None]) / std[:, None, None]
    made["real_dog"] = x[None].astype(np.float32)

    made["zeros"] = np.zeros((1, 3, H, W), dtype=np.float32)
    made["neg_const"] = np.full((1, 3, H, W), -1.5, dtype=np.float32)

    rng = np.random.default_rng(1234)
    made["noise_seed1234"] = rng.uniform(-3.0, 3.0, size=(1, 3, H, W)).astype(np.float32)

    half = np.empty((1, 3, H, W), dtype=np.float32)
    half[..., : W // 2] = -2.0
    half[..., W // 2:] = 2.0
    made["half_split"] = half

    meta = {}
    for name, arr in made.items():
        p = os.path.join(INPUTS, name + ".npy")
        np.save(p, arr)
        meta[name] = {
            "path": p,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "bytes": os.path.getsize(p),
            "sha256": sha256(p),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return meta


def main():
    t0 = time.time()

    print("=== torchvision weights ===")
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights).eval()
    ckpt = os.path.join(
        os.environ["TORCH_HOME"], "hub", "checkpoints", "resnet18-f37072fd.pth"
    )
    if not os.path.exists(ckpt):
        raise SystemExit("weights were not cached where expected: " + ckpt)
    ckpt_sha = sha256(ckpt)
    print("checkpoint:", ckpt)
    print("sha256    :", ckpt_sha)
    print("filename hash prefix matches:", ckpt_sha.startswith("f37072fd"))

    n_params = sum(p.numel() for p in model.parameters())
    print("param count:", n_params)

    print("=== inputs ===")
    inputs = build_inputs()
    for k, v in inputs.items():
        print("  %-16s %s sha256=%s" % (k, v["shape"], v["sha256"][:16]))

    print("=== reference logits (PyTorch FP32) ===")
    ref = {}
    with torch.no_grad():
        for name, m in inputs.items():
            x = torch.from_numpy(np.load(m["path"]))
            y = model(x)
            arr = y.numpy().astype(np.float32)
            p = os.path.join(INPUTS, "ref_%s.npy" % name)
            np.save(p, arr)
            top1 = int(arr.argmax(axis=1)[0])
            ref[name] = {
                "path": p,
                "sha256": sha256(p),
                "shape": list(arr.shape),
                "top1": top1,
                "logit_max": float(arr.max()),
                "logit_min": float(arr.min()),
            }
            print("  %-16s top1=%4d  logits[%.3f, %.3f]" % (name, top1, arr.min(), arr.max()))

    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    with open(os.path.join(ASSETS, "p3-env-freeze.txt"), "w") as f:
        f.write(pip_freeze)

    prereg = {
        "kind": "pre_registered_acceptance_tolerance",
        "written_before_conversion": True,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "context": (
            "FP32 PyTorch -> FP32 TFLite. Differences come only from op reordering "
            "and fused arithmetic; any padding/transpose/layout mistake produces "
            "errors of order 1e-1..1e1 on logits whose own range is about [-10, 10]."
        ),
        "thresholds_logits": {
            "pass": {"max_abs_le": 1e-3, "argmax_identical": True},
            "pass_with_note": {"max_abs_le": 1e-2, "argmax_identical": True},
            "fail": {"max_abs_gt": 1e-2, "or": "argmax differs on any input"},
        },
        "report_only": ["rmse", "per_input_max_abs"],
        "rule": (
            "Thresholds are fixed now. If the result lands in pass_with_note the "
            "number is reported as-is with a note; the thresholds are not relaxed "
            "afterwards, and a fail is not retried with a looser bound."
        ),
        "model": {
            "family": "torchvision resnet18",
            "classes": 1000,
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "checkpoint_sha256": ckpt_sha,
            "param_count": n_params,
            "eval_mode": True,
            "input_shape": [1, 3, H, W],
        },
        "inputs": inputs,
        "reference_logits": ref,
    }
    outp = os.path.join(ASSETS, "p3-prereg.json")
    with open(outp, "w") as f:
        json.dump(prereg, f, indent=2)
    print("=== wrote", outp, "===")
    print("elapsed %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
