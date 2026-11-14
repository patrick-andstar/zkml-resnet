#!/usr/bin/env python3
"""P4-A v3（converter 侧）针对性自检。

不导入 tensorflow、不构造真实 tflite、不跑完整 converter。
做法：从 `patches/converter_p4a_v3.py` 用 AST **抽取真实函数源码**再 exec，
避免"测试里另写一份逻辑"导致测的不是候选。

覆盖 plan 要求的六类：
  1. 同 ID 不同位置（按位置过滤，数据位置保留）
  2. 动态元数据拒绝（有 producer 的 paddings/axes）
  3. Pad 被观察拒绝（整图输出 / 导出子图公开输出 / 多消费者）
  4. 半点及相邻浮点（0.5-2^-54 → 0；0.5 → 1）
  5. 正负对称 + 大整数不变（2^52+1 原样）
  6. CUSTOM 缩放反例（float32 提前乘 → 错一位）

只证明这些函数在给定输入下的行为，不证明完整 converter 正确。
"""
import ast
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "patches", "converter_p4a_v3.py")

FAILS = []


def check(name, cond, detail=""):
    print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILS.append(name)


def load_src():
    return io.open(CAND, encoding="utf-8").read()


def extract(src, func_name, kind="func"):
    """按名字取函数/方法的源码段（含缩进）。"""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            lines = src.splitlines(True)
            return "".join(lines[node.lineno - 1: node.end_lineno])
    raise KeyError("%s not found" % func_name)


def strip_indent(s):
    """去掉 def 行自身的缩进量（方法 2 空格、模块级函数 0 空格）。"""
    lines = s.splitlines(True)
    first = lines[0]
    indent = len(first) - len(first.lstrip(" "))
    if indent == 0:
        return s
    pref = " " * indent
    return "".join(l[len(pref):] if l.startswith(pref) else l for l in lines)


def build_module_level(src):
    """只取模块级常量，供 exec 使用。"""
    ns = {"np": np}
    for name in ("METADATA_INPUT_POS", "I64_MAX", "I32_MIN", "I32_MAX", "I32_POLICY"):
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<c>", "exec"), ns)
    return ns


class FakeGraph:
    def __init__(self, outs):
        self._outs = list(outs)
    def OutputsLength(self):
        return len(self._outs)
    def Outputs(self, i):
        return self._outs[i]


class FakeTensor:
    def __init__(self, buffer_idx):
        self._b = buffer_idx
    def Buffer(self):
        return self._b


class FakeBuffer:
    def __init__(self, n):
        self._n = n
    def DataLength(self):
        return self._n


class FakeModel:
    def __init__(self, bufs):
        self._bufs = bufs
    def Buffers(self, i):
        return self._bufs.get(i)


class FakeOp:
    def __init__(self, inputs):
        self._inp = list(inputs)
    def InputsLength(self):
        return len(self._inp)
    def Inputs(self, i):
        return self._inp[i] if 0 <= i < len(self._inp) else -1


def make_self(ns, src, attrs):
    """把抽取到的方法绑到一个假 self 上。"""
    cls = type("Fake", (), {})
    for fname in ("_metadata_input_positions", "_static_constant",
                  "_observation_tensors", "_validate_padv2_subgraph"):
        code = strip_indent(extract(src, fname))
        ns2 = dict(ns)
        exec(compile(code, "<%s>" % fname, "exec"), ns2)
        setattr(cls, fname, ns2[fname])
    obj = cls()
    for k, v in attrs.items():
        setattr(obj, k, v)
    # _op_name 用最简实现
    obj._op_name = lambda op_idx: attrs.get("_op_name_map", {}).get(op_idx, "UNKNOWN")
    return obj


def main():
    src = load_src()
    ns = build_module_level(src)

    # ---------- 4/5/6：舍入 ----------
    print("\n[1] 舍入：半点、相邻浮点、正负对称、大整数")
    rns = {"np": np}
    exec(compile(strip_indent(extract(src, "round_half_away_from_zero")),
                 "<round>", "exec"), rns)
    rnd = rns["round_half_away_from_zero"]

    below_half = 0.5 - 2.0 ** -54          # 0.5 前一个可表示浮点
    check("0.5-2^-54 -> 0（旧式会给 1）", float(rnd(np.float64(below_half))) == 0.0,
          "(%.20g -> %s)" % (below_half, float(rnd(np.float64(below_half)))))
    check("0.5 -> 1（half away）", float(rnd(np.float64(0.5))) == 1.0)
    check("-0.5 -> -1（正负对称）", float(rnd(np.float64(-0.5))) == -1.0)
    check("2.5 -> 3", float(rnd(np.float64(2.5))) == 3.0)
    check("-2.5 -> -3", float(rnd(np.float64(-2.5))) == -3.0)
    big = float((1 << 52) + 1)
    check("2^52+1 原样不变（旧式会给 +2）", float(rnd(np.float64(big))) == big,
          "(%.20g -> %.20g)" % (big, float(rnd(np.float64(big)))))
    asym = 0.49999999999999994           # 再下一个浮点，明确小于 0.5
    check("0.49999...994 -> 0", float(rnd(np.float64(asym))) == 0.0)

    # ---------- 6：CUSTOM 缩放顺序 ----------
    print("\n[2] CUSTOM：float32 提前乘 scale 会丢精度")
    qns = dict(ns)
    qns["np"] = np
    qns["I32_POLICY"] = False            # 允许跨过 i32 看真实数值
    exec(compile(strip_indent(extract(src, "_quantize_checked")),
                 "<quant>", "exec"), qns)
    quant = qns["_quantize_checked"]
    qns["round_half_away_from_zero"] = rnd
    exec(compile(strip_indent(extract(src, "_quantize_checked")),
                 "<quant>", "exec"), qns)
    quant = qns["_quantize_checked"]

    sf = (1 << 22) - 1
    w32 = np.float32(1.0 + 2.0 ** -23)
    # 候选的正确顺序：原始 float32 交给 helper，helper 先升 float64 再乘
    class QSelf:
        scale_factor = sf

    q_new = quant(QSelf, np.asarray([w32]), 999, already_scaled=False)
    # 旧顺序：先在 float32 里乘，再交给 helper
    q_old = quant(QSelf, np.asarray([w32 * np.float32(sf)], dtype=np.float64), 999,
                  already_scaled=True)
    exact = float(w32) * float(sf)
    check("旧顺序（float32 先乘）得到 4194304", int(q_old[0]) == 4194304, "(got %d)" % int(q_old[0]))
    check("新顺序（先升 float64）得到 4194303", int(q_new[0]) == 4194303, "(got %d)" % int(q_new[0]))
    check("两序结果确实不同（反例成立）", int(q_old[0]) != int(q_new[0]))
    check("精确乘积 < 4194303.5", exact < 4194303.5, "(%.10f)" % exact)

    # ---------- 5：元数据按位置 ----------
    print("\n[3] 元数据：同 ID 同时出现在数据位置与元数据位置")
    obj = make_self(ns, src, {
        "consumers": {},
        "shared_metadata": {},
        "_op_name_map": {},
        "producer": {},
    })
    pad_op = FakeOp([7, 7])              # pos0=data(7), pos1=paddings(同一个 7)
    pos = obj._metadata_input_positions(pad_op, "PAD")
    check("返回的是位置 [1] 而非 ID 集合", pos == [1], "(got %r)" % (pos,))

    raw = [int(x) for x in pad_op.InputsAsNumpy().tolist()] if hasattr(pad_op, "InputsAsNumpy") else [7, 7]
    pos_set = set(pos)
    filtered = [t for p, t in enumerate(raw) if p not in pos_set and t != -1]
    sim_line = "inp_idxes = [t for p, t in enumerate(_raw_inputs)"
    check("候选里存在该位置过滤表达式", sim_line in src)
    check("过滤后数据位置保留（[7] 而非 []）", filtered == [7], "(got %r)" % (filtered,))
    check("按旧逻辑（按 ID 删）会得到 []", [t for t in raw if t != 7] == [])

    # ---------- 5：动态元数据拒绝 ----------
    print("\n[4] 动态元数据：有 producer 的 paddings/axes 必须拒绝")
    class FakeInterp:
        def get_tensor(self, i):
            return np.zeros((2, 2), dtype=np.int32)
    obj2 = make_self(ns, src, {
        "consumers": {}, "shared_metadata": {}, "_op_name_map": {},
        "producer": {5: 3},                       # tensor 5 由 op3 产出
        "graph": type("G", (), {"Tensors": lambda self, i: FakeTensor(1)})(),
        "model": FakeModel({1: FakeBuffer(16)}),
        "interpreter": FakeInterp(),
    })
    try:
        obj2._static_constant(5, "PAD(op9).paddings")
        check("有 producer 的 paddings 被拒绝", False)
    except RuntimeError as e:
        check("有 producer 的 paddings 被拒绝", "运行时计算值" in str(e))
    # 无 buffer 的常量同样拒绝
    obj2.producer = {}
    obj2.graph = type("G", (), {"Tensors": lambda self, i: FakeTensor(0)})()
    try:
        obj2._static_constant(6, "MEAN(op8).axes")
        check("Buffer()==0 的 axes 被拒绝", False)
    except RuntimeError as e:
        check("Buffer()==0 的 axes 被拒绝", "没有常量 buffer" in str(e))
    # 真常量应通过
    obj2.graph = type("G", (), {"Tensors": lambda self, i: FakeTensor(1)})()
    try:
        v = obj2._static_constant(6, "MEAN(op8).axes")
        check("真常量可读取", v.shape == (2, 2))
    except Exception as e:
        check("真常量可读取", False, str(e))

    # ---------- 4b：Pad 被观察拒绝 ----------
    print("\n[5] PADV2：观察集合（整图输出 / 子图公开输出 / 多消费者）")
    def make_val(outs, expose, consumers, evidence, in_range, layer_out=None):
        layers = [{"out_idxes": layer_out if layer_out is not None else []}]
        return make_self(ns, src, {
            "graph": FakeGraph(outs),
            "expose_output": expose,
            "consumers": consumers,
            "shared_metadata": {},
            "_op_name_map": {},
            "producer": {},
            "_padv2_seen": len(evidence),
            "padv2_evidence": evidence,
            "_layer_op_idxes": in_range,
        }), layers

    ev = {"op_idx": 2, "pad_out_tensor": 49, "consumer_op_idx": 3}

    obj3, layers = make_val(outs=[49], expose=False, consumers={49: [(3, 0)]},
                            evidence=[dict(ev)], in_range={2, 3})
    try:
        obj3._validate_padv2_subgraph(layers, 0, 5)
        check("Pad 是整图输出 -> 拒绝", False)
    except RuntimeError as e:
        check("Pad 是整图输出 -> 拒绝", "会被外部观察" in str(e))

    obj4, layers = make_val(outs=[], expose=True, consumers={49: [(3, 0)]},
                            evidence=[dict(ev)], in_range={2, 3}, layer_out=[49])
    try:
        obj4._validate_padv2_subgraph(layers, 0, 5)
        check("Pad 是导出子图公开输出 -> 拒绝", False)
    except RuntimeError as e:
        check("Pad 是导出子图公开输出 -> 拒绝", "会被外部观察" in str(e))

    obj5, layers = make_val(outs=[], expose=False, consumers={49: [(3, 0), (9, 0)]},
                            evidence=[dict(ev)], in_range={2, 3, 9})
    try:
        obj5._validate_padv2_subgraph(layers, 0, 9)
        check("子图内 2 个消费者 -> 拒绝", False)
    except RuntimeError as e:
        check("子图内 2 个消费者 -> 拒绝", "要求恰好 1 个" in str(e))

    obj6, layers = make_val(outs=[], expose=False, consumers={49: [(3, 0)]},
                            evidence=[dict(ev)], in_range={2, 3})
    try:
        obj6._validate_padv2_subgraph(layers, 0, 5)
        check("未被观察且唯一消费者 -> 通过", True)
        check("证据写入 checked_against",
              obj6.padv2_evidence[0]["checked_against"] ==
              "graph_outputs+exported_subgraph+single_consumer")
    except RuntimeError as e:
        check("未被观察且唯一消费者 -> 通过", False, str(e))

    print("\nTOTAL: %s" % ("PASS" if not FAILS else "FAIL -> %s" % ", ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
