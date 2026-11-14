#!/usr/bin/env python3
"""P3 交叉校验：TFLite flatbuffer 与 converter 产出 msgpack 的逐层一致性。

只做结构/参数比对，不解包不推理。
  - 层数必须相等
  - 第 i 层的输出形状必须相等（这是最强的一条：能抓出 padding/stride 错位）
  - 每类算子的参数必须与 TFLite 选项一一对应（含 padding/activation 的枚举方向）
  - 顺手统计 Pad 造成的中间张量膨胀（P4 资源估算的输入，不是结论）
"""
import json
import os
import sys

REPRO = os.environ.get("REPRO", "/root/autodl-tmp/zkml-repro")
A = os.path.join(REPRO, "assets/p3-opaudit.json")
B = os.path.join(REPRO, "assets/p3-msgpack-summary.json")
OUT = os.path.join(REPRO, "assets/p3-crosscheck.json")

au = json.load(open(A))
ms = json.load(open(B))

ops = au["ops"]
layers = ms["layers"]
fails = []
rows = []

if len(ops) != len(layers):
    fails.append("层数不等: tflite=%d msgpack=%d" % (len(ops), len(layers)))

T2R = {"CONV_2D": "Conv2D", "MAX_POOL_2D": "MaxPool2D", "ADD": "Add",
       "MEAN": "Mean", "FULLY_CONNECTED": "FullyConnected",
       "PAD": "Pad", "PADV2": "Pad"}

for i, (o, L) in enumerate(zip(ops, layers)):
    r = {"i": i, "tflite_op": o["op"], "layer_type": L["type"],
         "out_tflite": o["outputs"][0]["shape"] if o["outputs"] else None,
         "out_msgpack": L["out_shapes"][0], "params": L["params"], "diff": []}

    if T2R.get(o["op"]) != L["type"]:
        r["diff"].append("类型映射不符: %s -> %s" % (o["op"], L["type"]))
    if r["out_tflite"] != L["out_shapes"][0]:
        r["diff"].append("输出形状不符: %s vs %s" % (r["out_tflite"], L["out_shapes"][0]))

    p = L["params"]
    if o["op"] in ("PAD", "PADV2"):
        if p != o["paddings"]:
            r["diff"].append("paddings 不符: %s vs %s" % (p, o["paddings"]))
        if o["op"] == "PADV2":
            r["pad_constant_dropped"] = o.get("constant")
    elif o["op"] == "CONV_2D":
        exp = [0, 0 if o["pad"] == "SAME" else 1,
               0 if o["act"] == "NONE" else 1, o["stride"][0], o["stride"][1]]
        if p != exp:
            r["diff"].append("conv params 不符: %s vs 期望 %s" % (p, exp))
    elif o["op"] == "MAX_POOL_2D":
        exp = [o["filter"][0], o["filter"][1], o["stride"][0], o["stride"][1]]
        if p != exp:
            r["diff"].append("maxpool params 不符: %s vs %s" % (p, exp))
        if o["pad"] != "VALID":
            r["diff"].append("maxpool 非 VALID，Rust 侧只实现 Valid")
    elif o["op"] == "ADD":
        exp = [0 if o["act"] == "NONE" else 1]
        if p != exp:
            r["diff"].append("add act 不符: %s vs %s" % (p, exp))
    elif o["op"] == "MEAN":
        if set(p) != set(o["axes"]):
            r["diff"].append("mean axes 不符: %s vs %s" % (p, o["axes"]))
    elif o["op"] == "FULLY_CONNECTED":
        if p != [0 if o["act"] == "NONE" else 1]:
            r["diff"].append("fc act 不符: %s" % p)

    if r["diff"]:
        fails.append("layer %d: %s" % (i, "; ".join(r["diff"])))
    rows.append(r)

# Pad 造成的中间张量膨胀（静态计数，仅作 P4 输入）
def numel(s):
    n = 1
    for d in s:
        n *= d
    return n

pad_cells = sum(numel(o["outputs"][0]["shape"]) for o in ops if o["op"] in ("PAD", "PADV2"))
tot_cells = sum(numel(r["out_tflite"]) for r in rows if r["out_tflite"])
inp_cells = sum(numel(o["inputs"][0]["shape"]) for o in ops)

rep = {
    "tflite_sha256": au["model_sha256"],
    "n_ops": len(ops), "n_layers": len(layers),
    "n_tensors_tflite": au["n_tensors"], "n_tensors_msgpack": ms.get("n_tensor_count"),
    "structure_match": not fails,
    "failures": fails,
    "pad_cell_inflation": {
        "pad_out_cells": pad_cells,
        "sum_of_all_layer_out_cells": tot_cells,
        "share_of_out_cells": round(pad_cells / tot_cells, 4) if tot_cells else None,
        "note": "Pad 在 Rust 侧会materialize 一个更大的中间张量；这是 P4 的 cell 计数输入，不是当前结论",
    },
    "layers": rows,
}
json.dump(rep, open(OUT, "w"), indent=1, ensure_ascii=False)

print("tflite sha256:", au["model_sha256"])
print("ops=%d layers=%d" % (len(ops), len(layers)))
print("structure_match:", rep["structure_match"])
if fails:
    print("--- 失败 ---")
    for f in fails:
        print("  " + f)
print()
print("Pad 膨胀: %d cells / 全部层输出 %d cells = %.2f%%"
      % (pad_cells, tot_cells, 100.0 * pad_cells / tot_cells if tot_cells else 0))
print()
print("  idx  tflite          -> rust              out                    params")
for r in rows:
    print("  %3d  %-15s -> %-16s %-22s %s%s" % (
        r["i"], r["tflite_op"], r["layer_type"], str(r["out_msgpack"]),
        str(r["params"])[:26], "   <<< " + "; ".join(r["diff"]) if r["diff"] else ""))
print()
print("report:", OUT)
sys.exit(0 if not fails else 1)
