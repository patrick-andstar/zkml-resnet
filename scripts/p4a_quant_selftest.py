#!/usr/bin/env python3
"""P4-A 补正：converter 侧数值契约的**静态自测**（不导入 tensorflow、不跑转换）。

做法：从 patches/converter_p4a_v2.py 里按 AST 抽出 round_half_away_from_zero
与 Converter._quantize_checked 的源码，拼成一个最小命名空间执行，再用假 self
调用它。这样测的是**候选里真正那份代码**，不是这里另抄一份。

覆盖：
  1. 舍入规则：.5 远离 0（与 numpy half-to-even 明确不同）
  2. 非有限值（NaN / ±inf）必须拒绝，且错误信息点出 tensor id
  3. L1 硬边界：-2^31 通过、-2^31-1 拒绝；i64::MAX-2^31 通过、再 +1 拒绝
  4. L2 主动策略：i32 之外拒绝；I32_POLICY=False 时同一值通过
  5. already_scaled=True 不再二次乘 scale
  6. 超过 2^53 的值拒绝（float64 无法精确表示整数）
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "patches/converter_p4a_v2.py")

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MAX = (1 << 63) - 1
L1_HI = I64_MAX - (1 << 31)


def build_namespace(path):
    """抽出候选里的 helper 与 _quantize_checked，拼成可执行的命名空间。"""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "round_half_away_from_zero":
            chunks.append(ast.get_source_segment(src, node))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("I64_MAX", "I32_MIN", "I32_MAX", "I32_POLICY"):
                    chunks.append(ast.get_source_segment(src, node))
    assert len(chunks) >= 4, "没抽到 helper：%r" % chunks

    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Converter")
    quant = next((n for n in cls.body
                  if isinstance(n, ast.FunctionDef) and n.name == "_quantize_checked"), None)
    assert quant is not None, "候选里没有 _quantize_checked"

    # 方法体去掉一层缩进，变成模块级函数
    body = ast.get_source_segment(src, quant)
    lines = body.splitlines()
    dedented = [lines[0].replace("def _quantize_checked(self, ", "def _quantize_checked(self, ", 1)]
    dedented += [ln[2:] if ln.startswith("  ") else ln for ln in lines[1:]]
    chunks.append("\n".join(dedented))

    ns = {"np": np}
    exec(compile("\n\n".join(chunks), "<extracted>", "exec"), ns)
    return ns


class FakeSelf:
    def __init__(self, sf):
        self.scale_factor = sf


def main():
    ns = build_namespace(CAND)
    q = ns["_quantize_checked"]
    rh = ns["round_half_away_from_zero"]
    print("已从 %s 抽出 round_half_away_from_zero / _quantize_checked" % os.path.basename(CAND))
    print("I32_POLICY =", ns["I32_POLICY"], " scale_factor 用 2^16\n")

    fails = []
    passed_count = [0]

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s" % name)
            passed_count[0] += 1
        else:
            print("  FAIL  %s  %s" % (name, detail))
            fails.append(name)

    # 1 舍入规则
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    got = rh(x).tolist()
    numpy_got = np.round(x).tolist()
    print("1) 舍入规则")
    print("   输入      %s" % x.tolist())
    print("   half-away %s" % got)
    print("   numpy     %s  (half-to-even)" % numpy_got)
    check("0.5 -> 1", got[0] == 1.0, got)
    check("1.5 -> 2", got[1] == 2.0, got)
    check("2.5 -> 3（与 numpy 的 2 不同）", got[2] == 3.0, got)
    check("-0.5 -> -1", got[3] == -1.0, got)
    check("-2.5 -> -3（与 numpy 的 -2 不同）", got[5] == -3.0, got)
    check("与 numpy 至少一处不同", got != numpy_got, "两者相同说明规则没生效")

    # 2 非有限值
    print("\n2) 非有限值")
    for bad in (np.nan, np.inf, -np.inf):
        try:
            q(FakeSelf(1 << 16), np.array([0.0, bad]), 7)
            check("拒绝 %r" % bad, False, "没有抛错")
        except RuntimeError as e:
            check("拒绝 %r" % bad, "tensor 7" in str(e), str(e)[:60])

    # 3 L1 硬边界（scale=1 便于直接给整数）
    print("\n3) L1 编码硬边界（scale_factor=1）")
    for v, expect_ok in ((I32_MIN, True), (I32_MIN - 1, False),
                         (L1_HI, False), (I32_MAX, True)):
        # I32_POLICY 打开时 L1 高边界必然先被 i32 策略拦住，因此单独关掉策略测 L1
        ns["I32_POLICY"] = False
        try:
            out = q(FakeSelf(1), np.array([v]), 3)
            ok = True
            val = int(out[0])
        except RuntimeError as e:
            ok = False
            val = str(e)[:48]
        ns["I32_POLICY"] = True
        check("%d 期望 %s" % (v, "通过" if expect_ok else "拒绝"), ok == expect_ok, val)

    # 4 L2 主动策略
    print("\n4) L2 主动 i32 策略（scale_factor=1）")
    try:
        q(FakeSelf(1), np.array([I32_MAX + 1]), 4)
        check("I32_MAX+1 拒绝", False, "没有抛错")
    except RuntimeError as e:
        check("I32_MAX+1 拒绝", "i32" in str(e), str(e)[:60])
    ns["I32_POLICY"] = False
    try:
        out = q(FakeSelf(1), np.array([I32_MAX + 1]), 4)
        check("关掉策略后 I32_MAX+1 通过", int(out[0]) == I32_MAX + 1, out)
    except RuntimeError as e:
        check("关掉策略后 I32_MAX+1 通过", False, str(e)[:60])
    ns["I32_POLICY"] = True

    # 5 already_scaled
    print("\n5) already_scaled 不二次乘 scale")
    sf = 1 << 16
    a = np.array([2.0])
    out_false = q(FakeSelf(sf), a, 5)
    out_true = q(FakeSelf(sf), a * sf, 5, already_scaled=True)
    check("两条路径同值", np.array_equal(out_false, out_true),
          "%s vs %s" % (out_false.tolist(), out_true.tolist()))
    check("未缩放路径确实乘了 sf", int(out_false[0]) == int(2.0 * sf), out_false.tolist())

    # 6 超过 2^53
    print("\n6) float64 精确整数上限 2^53")
    try:
        q(FakeSelf(1), np.array([float(1 << 54)]), 6)
        check("2^54 拒绝", False, "没有抛错")
    except RuntimeError as e:
        check("2^54 拒绝", "2^53" in str(e), str(e)[:60])

    print()
    if fails:
        print("失败 %d 项：%s" % (len(fails), fails))
        return 1
    print("全部通过（%d 项，0 失败）" % passed_count[0])
    print("说明：这是对 converter 侧**数值契约代码**的自测，不是运行转换，"
          "也没有验证任何 Rust 侧行为。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
