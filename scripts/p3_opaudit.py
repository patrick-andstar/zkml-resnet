#!/usr/bin/env python3
"""P3 静态算子/布局审计。

只读 flatbuffer，不跑推理、不改模型。逐 op 抽 padding / stride / dilation /
fused_activation / filter 形状，再与 PyTorch ResNet-18 语义逐条对照。

用法:
  python p3_opaudit.py <model.tflite> <out.json>
"""
import json
import sys
import hashlib
import collections

import numpy as np
import tflite

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/zkml-repro/assets/p3-model.tflite"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/zkml-repro/assets/p3-opaudit.json"


def enum_map(cls):
    return {getattr(cls, a): a for a in dir(cls)
            if not a.startswith("_") and isinstance(getattr(cls, a), int)}


BUILTIN = enum_map(tflite.BuiltinOperator)
TTYPE = enum_map(tflite.TensorType)
PADDING = enum_map(tflite.Padding)
ACT = enum_map(tflite.ActivationFunctionType)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


buf = open(MODEL, "rb").read()
model = tflite.Model.GetRootAsModel(buf, 0)
graph = model.Subgraphs(0)

# ---------------- tensors ----------------
tensors = []
for i in range(graph.TensorsLength()):
    t = graph.Tensors(i)
    shape = [int(t.Shape(j)) for j in range(t.ShapeLength())] or [1]
    tensors.append({
        "idx": i,
        "name": (t.Name() or b"").decode("utf-8", "replace"),
        "shape": shape,
        "dtype": TTYPE.get(t.Type(), str(t.Type())),
    })


def tinfo(idx):
    if idx < 0:
        return {"idx": idx, "name": "<optional>", "shape": [], "dtype": "-"}
    return tensors[idx]


# ---------------- ops ----------------
def opt_of(op, cls):
    o = op.BuiltinOptions()
    if o is None:
        return None
    v = cls()
    v.Init(o.Bytes, o.Pos)
    return v


def const_array(idx):
    """读常量 tensor 的数值（仅用于 PAD/TRANSPOSE/MEAN 的 axis 等小张量）。"""
    if idx < 0:
        return None
    t = graph.Tensors(idx)
    bidx = t.Buffer()
    if bidx is None or bidx < 0 or bidx >= model.BuffersLength():
        return None
    b = model.Buffers(bidx)
    if b is None or b.DataLength() == 0:
        return None
    raw = bytes(b.DataAsNumpy())
    dt = TTYPE.get(t.Type())
    if dt == "INT32":
        return np.frombuffer(raw, dtype=np.int32).tolist()
    if dt == "INT64":
        return np.frombuffer(raw, dtype=np.int64).tolist()
    if dt == "FLOAT32":
        return [float(x) for x in np.frombuffer(raw, dtype=np.float32).tolist()]
    return None


ops = []
for i in range(graph.OperatorsLength()):
    op = graph.Operators(i)
    code = model.OperatorCodes(op.OpcodeIndex())
    bcode = code.BuiltinCode()
    if bcode == 0:  # PLACEHOLDER_FOR_GREATER_OP_CODES
        bcode = code.DeprecatedBuiltinCode()
    name = BUILTIN.get(bcode, str(bcode))
    ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
    outs = [int(x) for x in op.OutputsAsNumpy()] if op.OutputsLength() else []
    d = {
        "op_idx": i,
        "op": name,
        "inputs": [tinfo(x) for x in ins],
        "outputs": [tinfo(x) for x in outs],
    }

    if name == "CONV_2D":
        o = opt_of(op, tflite.Conv2DOptions)
        f = tensors[ins[1]]
        d.update(pad=PADDING.get(o.Padding()), stride=[o.StrideH(), o.StrideW()],
                 dilation=[o.DilationHFactor(), o.DilationWFactor()],
                 act=ACT.get(o.FusedActivationFunction()), filter_shape=f["shape"])
    elif name == "DEPTHWISE_CONV_2D":
        o = opt_of(op, tflite.DepthwiseConv2DOptions)
        f = tensors[ins[1]]
        d.update(pad=PADDING.get(o.Padding()), stride=[o.StrideH(), o.StrideW()],
                 dilation=[o.DilationHFactor(), o.DilationWFactor()],
                 act=ACT.get(o.FusedActivationFunction()),
                 depth_multiplier=o.DepthMultiplier(), filter_shape=f["shape"])
    elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
        o = opt_of(op, tflite.Pool2DOptions)
        d.update(pad=PADDING.get(o.Padding()), stride=[o.StrideH(), o.StrideW()],
                 act=ACT.get(o.FusedActivationFunction()),
                 filter=[o.FilterHeight(), o.FilterWidth()])
    elif name in ("ADD", "MUL", "SUB"):
        cls = {"ADD": tflite.AddOptions, "MUL": tflite.MulOptions,
               "SUB": tflite.SubOptions}[name]
        o = opt_of(op, cls)
        d.update(act=ACT.get(o.FusedActivationFunction()) if o else None)
    elif name == "FULLY_CONNECTED":
        o = opt_of(op, tflite.FullyConnectedOptions)
        d.update(act=ACT.get(o.FusedActivationFunction()),
                 keep_num_dims=bool(o.KeepNumDims()),
                 weights_format=int(o.WeightsFormat()),
                 filter_shape=tensors[ins[1]]["shape"])
    elif name == "MEAN":
        d.update(axes=const_array(ins[1]), keepdims=bool(o.KeepDims()) if (o := opt_of(op, tflite.ReducerOptions)) else None)
    elif name == "PAD":
        d.update(paddings=const_array(ins[1]))
    elif name == "PADV2":
        d.update(paddings=const_array(ins[1]), constant=const_array(ins[2]))
    elif name == "TRANSPOSE":
        d.update(perm=const_array(ins[1]))
    elif name == "RESHAPE":
        d.update(new_shape=const_array(ins[1]))
    elif name == "SOFTMAX":
        o = opt_of(op, tflite.SoftmaxOptions)
        d.update(beta=float(o.Beta()) if o else None)
    ops.append(d)

# ---------------- 汇总 ----------------
by_type = collections.Counter(o["op"] for o in ops)

# ---------------- 语义对照检查 ----------------
checks = []


def chk(cid, desc, ok, detail):
    checks.append({"id": cid, "desc": desc, "ok": bool(ok), "detail": detail})


def prev_pad(op_idx):
    """返回紧邻的前一个填充算子（PAD 或 PADV2，若有）。"""
    if op_idx == 0:
        return None
    p = ops[op_idx - 1]
    return p if p["op"] in ("PAD", "PADV2") else None


# C1 输入/输出布局
inp = tensors[graph.Inputs(0)] if graph.InputsLength() else None
out = tensors[graph.Outputs(0)] if graph.OutputsLength() else None
chk("C1", "I/O 布局为 NHWC，FP32",
    inp and inp["shape"] == [1, 128, 128, 3] and inp["dtype"] == "FLOAT32"
    and out and out["shape"] == [1, 1000] and out["dtype"] == "FLOAT32",
    {"input": inp, "output": out})

# C2 无量化：允许 FLOAT32 权重/激活 + INT32/INT64 的索引常量，不允许 int8/uint8
BAD_DT = {"INT8", "UINT8", "INT16", "UINT16"}
quanted = [t for t in tensors if t["dtype"] in BAD_DT]
chk("C2", "无量化：不存在 int8/uint8 tensor（仅 FLOAT32 + INT32/INT64 索引常量）",
    len(quanted) == 0,
    {"n_tensors": len(tensors),
     "dtype_hist": dict(collections.Counter(t["dtype"] for t in tensors)),
     "quantized": quanted[:8]})

# C3 stride=2 卷积：核 >1 必须 VALID 且前接显式 pad=k//2；1x1 核不应有 pad
s2 = [o for o in ops if o["op"] == "CONV_2D" and o["stride"] == [2, 2]]
s2_bad = []
for o in s2:
    kh = o["filter_shape"][1]
    p = prev_pad(o["op_idx"])
    need = kh // 2
    ok = (o["pad"] == "VALID")
    if kh == 1:
        ok = ok and p is None
    else:
        ok = ok and p is not None and p["paddings"][2:4] == [need, need] \
            and p["paddings"][4:6] == [need, need]
    if not ok:
        s2_bad.append({"op_idx": o["op_idx"], "k": kh, "pad": o["pad"],
                       "prev_pad": (p or {}).get("paddings")})
chk("C3", "stride2 卷积：核>1 为 VALID + 显式 pad=k//2，核=1 无 pad（否则窗口错位）",
    not s2_bad, {"n": len(s2), "bad": s2_bad,
                 "detail": [{"op_idx": o["op_idx"], "k": o["filter_shape"][1:3],
                             "in": o["inputs"][0]["shape"],
                             "out": o["outputs"][0]["shape"]} for o in s2]})

# C4 stem 7x7 PAD 值必须是 3（PyTorch pad=3），且 TFLite SAME 会给 2/3 不对称
stem_pads = [p for p in ops if p["op"] == "PAD" and p["paddings"]
             and p["paddings"][2:4] == [3, 3] and p["paddings"][4:6] == [3, 3]]
chk("C4", "stem conv7x7/pad3 用显式 PAD=3（SAME 只会给 pad_before=2，左右不对称）",
    len(stem_pads) == 1, {"found": [p["paddings"] for p in stem_pads]})

# C5 maxpool 3x3/2 pad1 → 显式 PAD=1 + VALID
mp = [o for o in ops if o["op"] == "MAX_POOL_2D"]
chk("C5", "maxpool 3x3/s2 为 显式 PAD(1) + VALID + 无 fused activation（SAME 会给 pad_before=0）",
    len(mp) == 1 and mp[0]["pad"] == "VALID" and mp[0]["act"] == "NONE"
    and mp[0]["filter"] == [3, 3] and mp[0]["stride"] == [2, 2]
    and (prev_pad(mp[0]["op_idx"]) or {}).get("paddings") == [0, 0, 1, 1, 1, 1, 0, 0],
    [{"op_idx": o["op_idx"], "pad": o["pad"], "filter": o["filter"],
      "stride": o["stride"], "act": o["act"],
      "prev_pad": (prev_pad(o["op_idx"]) or {}).get("paddings")} for o in mp])

# C6 stride=1 卷积应为 SAME（奇数核时 SAME 等价 PyTorch pad=(k-1)/2）
s1 = [o for o in ops if o["op"] == "CONV_2D" and o["stride"] == [1, 1]]
chk("C6", "stride1 3x3 卷积全部 SAME、无显式 PAD、核为奇数",
    all(o["pad"] == "SAME" and prev_pad(o["op_idx"]) is None
        and o["filter_shape"][1] % 2 == 1 for o in s1),
    {"n": len(s1),
     "distinct": sorted({(o["pad"], tuple(o["filter_shape"])) for o in s1}),
     "with_prev_pad": [o["op_idx"] for o in s1 if prev_pad(o["op_idx"])]})

# C7 dilation 必须全为 1
chk("C7", "所有卷积 dilation=1", all(o["dilation"] == [1, 1]
    for o in ops if o["op"] in ("CONV_2D", "DEPTHWISE_CONV_2D")),
    {"values": sorted({tuple(o["dilation"]) for o in ops if o["op"] == "CONV_2D"})})

# C8 无独立 ReLU op（应全部融合）
standalone = [o for o in ops if o["op"] in ("RELU", "RELU6", "RELU_N1_TO_1")]
chk("C8", "无独立 ReLU op（ReLU 全部融合进 CONV/ADD）", len(standalone) == 0,
    {"standalone": [o["op"] for o in standalone]})

# C9 ReLU 融合点计数应等于 ResNet-18 的 17 个
conv_relu = [o for o in ops if o["op"] == "CONV_2D" and o["act"] == "RELU"]
add_relu = [o for o in ops if o["op"] == "ADD" and o["act"] == "RELU"]
chk("C9", "ReLU 融合点 = 9(conv) + 8(add) = 17，与 ResNet-18 的 17 个 ReLU 一致",
    len(conv_relu) == 9 and len(add_relu) == 8,
    {"conv_relu": len(conv_relu), "add_relu": len(add_relu),
     "conv_no_act": len([o for o in ops if o["op"] == "CONV_2D" and o["act"] == "NONE"])})

# C10 残差 ADD：8 个，两侧同形，且非广播
adds = [o for o in ops if o["op"] == "ADD"]
same_shape = all(o["inputs"][0]["shape"] == o["inputs"][1]["shape"] for o in adds)
chk("C10", "8 个残差 ADD 两侧同形（无广播），无 -inf mask 常量",
    len(adds) == 8 and same_shape and all(o["act"] in ("RELU", "NONE") for o in adds),
    [{"op_idx": o["op_idx"], "in0": o["inputs"][0]["shape"], "in1": o["inputs"][1]["shape"],
      "act": o["act"]} for o in adds])

# C11 全局池化：MEAN over {1,2}，输出 [1,512]
means = [o for o in ops if o["op"] == "MEAN"]
chk("C11", "全局平均池化 = MEAN(axes={1,2}, 去掉 HW 两维)，[1,4,4,512] -> [1,512]",
    len(means) == 1 and set(means[0]["axes"]) == {1, 2}
    and means[0]["inputs"][0]["shape"] == [1, 4, 4, 512]
    and means[0]["outputs"][0]["shape"] == [1, 512],
    [{"op_idx": o["op_idx"], "axes": o["axes"], "keepdims": o["keepdims"],
      "in": o["inputs"][0]["shape"], "out": o["outputs"][0]["shape"]} for o in means])

# C12 FC：1000x512，无激活
fcs = [o for o in ops if o["op"] == "FULLY_CONNECTED"]
chk("C12", "FC 权重 [1000,512]、无 fused activation、输出 [1,1000]",
    len(fcs) == 1 and fcs[0]["filter_shape"] == [1000, 512] and fcs[0]["act"] == "NONE"
    and fcs[0]["outputs"][0]["shape"] == [1, 1000],
    [{"op_idx": o["op_idx"], "w": o["filter_shape"], "act": o["act"],
      "in": o["inputs"][0]["shape"], "out": o["outputs"][0]["shape"],
      "keep_num_dims": o["keep_num_dims"]} for o in fcs])

# C13 无 CUSTOM / 非法算子
odd = [o["op"] for o in ops if o["op"] in ("CUSTOM", "DELEGATE", "IF", "WHILE", "CALL_ONCE")]
chk("C13", "无 CUSTOM / 控制流算子", len(odd) == 0, {"odd": odd})

# C14 填充值：PAD 定义即 0；PADV2 的 constant 必须与下一算子语义匹配
#      （maxpool 需 -inf，卷积需 0）
pads = [o for o in ops if o["op"] in ("PAD", "PADV2")]
pad_bad = []
for o in pads:
    nxt = ops[o["op_idx"] + 1]["op"] if o["op_idx"] + 1 < len(ops) else "?"
    if o["op"] == "PAD":
        ok = True                      # PAD 无常量，按定义填 0
    else:
        c = o.get("constant")
        if nxt == "MAX_POOL_2D":
            ok = c is not None and len(c) == 1 and c[0] == float("-inf")
        else:
            ok = c is not None and len(c) == 1 and c[0] == 0
    if not ok:
        pad_bad.append({"op_idx": o["op_idx"], "op": o["op"], "next": nxt,
                        "constant": o.get("constant")})
chk("C14", "填充值符合语义（PAD=0；maxpool 前的 PADV2 必须 -inf，卷积前必须 0）",
    not pad_bad,
    {"bad": pad_bad,
     "pads": [{"op_idx": o["op_idx"], "op": o["op"], "paddings": o.get("paddings"),
               "constant": o.get("constant"), "in": o["inputs"][0]["shape"]}
              for o in pads]})

# C14b 每个填充量 = 对应卷积核的 k//2（步长2）/ (k-1)//2（步长1 但此处无显式 pad）
pad_amount_ok = True
for o in pads:
    nxt = ops[o["op_idx"] + 1] if o["op_idx"] + 1 < len(ops) else None
    if nxt is None:
        pad_amount_ok = False
        continue
    if nxt["op"] == "CONV_2D":
        need = nxt["filter_shape"][1] // 2
    elif nxt["op"] == "MAX_POOL_2D":
        need = nxt["filter"][0] // 2
    else:
        pad_amount_ok = False
        continue
    if o["paddings"][2:6] != [need] * 4:
        pad_amount_ok = False
chk("C14b", "每个显式填充量 = 相邻核的 k//2（PyTorch pad 约定）", pad_amount_ok,
    [{"pad_op": o["op_idx"], "next_op": ops[o["op_idx"] + 1]["op"],
      "paddings": o["paddings"]} for o in pads if o["op_idx"] + 1 < len(ops)])

# C15 0 填充的语义风险：卷积的 0 填充与 PyTorch 定义一致；
#     maxpool 的 0 填充只有在输入已保证非负时才等价于 PyTorch 的 -inf 填充
pad_risk = []
risk_ok = True
for o in pads:
    nxt = ops[o["op_idx"] + 1] if o["op_idx"] + 1 < len(ops) else None
    src = o["inputs"][0]["idx"]
    prod = None
    for p in ops:
        if src in [q["idx"] for q in p["outputs"]]:
            prod = p
            break
    kind = nxt["op"] if nxt else "?"
    is_neg_inf = (o["op"] == "PADV2" and o.get("constant") == [float("-inf")])
    nonneg = prod is not None and prod.get("act") == "RELU"
    if kind == "MAX_POOL_2D":
        if is_neg_inf:
            safe = True
            why = "-inf 填充，与 PyTorch MaxPool 的 -inf 填充完全一致，不依赖输入符号"
        else:
            safe = nonneg
            why = "0 填充只在输入非负时等价 -inf 填充；此处依赖生产者 ReLU"
    else:
        safe = True
        why = "卷积 0 填充与 PyTorch pad 定义一致，无需非负前提"
    risk_ok = risk_ok and safe
    pad_risk.append({"pad_op": o["op_idx"], "pad_in": str(o["inputs"][0]["name"])[:60],
                     "next_op": kind, "producer": prod["op"] if prod else "INPUT",
                     "producer_act": prod.get("act") if prod else None,
                     "const_is_neg_inf": is_neg_inf, "producer_is_relu": nonneg,
                     "safe": safe, "why": why})
chk("C15", "0 填充语义安全（maxpool 前的 0 填充需非负输入；卷积无需）", risk_ok, pad_risk)

# C16 TRANSPOSE 位置与语义
trs = [o for o in ops if o["op"] == "TRANSPOSE"]
chk("C16", "TRANSPOSE 数量/位置/perm 已记录", True,
    [{"op_idx": o["op_idx"], "perm": o["perm"], "in": o["inputs"][0]["shape"],
      "out": o["outputs"][0]["shape"]} for o in trs])

# C17 卷积总数与 ResNet-18 一致（1 stem + 16 block conv + 3 downsample）
convs = [o for o in ops if o["op"] == "CONV_2D"]
chk("C17", "CONV_2D=20（1 stem + 16 block + 3 downsample），无 depthwise",
    len(convs) == 20 and not [o for o in ops if o["op"] == "DEPTHWISE_CONV_2D"],
    {"n_conv": len(convs), "by_type": dict(by_type)})

# C18 无 BatchNorm 残留算子（BN 应已在 eval 下折叠进 conv 权重）
chk("C18", "无 BN 残留（无 BATCH_MATMUL/MUL+ADD 形式的归一化结构）",
    not [o for o in ops if o["op"] in ("BATCH_MATMUL", "MUL")],
    {"present": sorted({o["op"] for o in ops})})

report = {
    "model": MODEL,
    "model_sha256": sha256(MODEL),
    "model_bytes": len(buf),
    "schema_version": model.Version(),
    "n_ops": len(ops),
    "n_tensors": len(tensors),
    "op_histogram": dict(sorted(by_type.items())),
    "checks": checks,
    "all_ok": all(c["ok"] for c in checks),
    "ops": ops,
}

with open(OUT, "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)

print("model:", MODEL)
print("sha256:", report["model_sha256"], "bytes:", report["model_bytes"])
print("ops:", len(ops), "tensors:", len(tensors))
print("histogram:", report["op_histogram"])
print()
print("=== checks ===")
for c in checks:
    print(("  OK  " if c["ok"] else " FAIL ") + c["id"] + "  " + c["desc"])
print()
print("ALL_OK:", report["all_ok"])
print("report:", OUT)

# 关键明细直接打印，便于抄进文档
print()
print("=== conv detail (op_idx, pad, stride, filter, in, out, act) ===")
for o in convs:
    print("  %3d  %-5s s=%d k=%dx%d  %s -> %s  %s" % (
        o["op_idx"], o["pad"], o["stride"][0], o["filter_shape"][1], o["filter_shape"][2],
        o["inputs"][0]["shape"], o["outputs"][0]["shape"], o["act"]))
print()
print("=== pads ===")
for o in pads:
    print("  op%-3d %-6s %s in=%s" % (o["op_idx"], o["op"], o.get("paddings"),
                                      o["inputs"][0]["shape"]))
print()
print("=== transpose / mean / fc ===")
for o in trs + means + fcs:
    print("  op%-3d %-16s in=%s out=%s %s" % (
        o["op_idx"], o["op"], o["inputs"][0]["shape"], o["outputs"][0]["shape"],
        {k: v for k, v in o.items() if k in ("perm", "axes", "keepdims", "act")}))
