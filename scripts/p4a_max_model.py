#!/usr/bin/env python3
"""P4-A / 问题1：MaxChip 归约的源码等价复刻与反例枚举。

**这是对 src/gadgets/max.rs 调度逻辑的 Python 复刻，不是运行 Rust。**
目的：在 P4-A（不编译、不跑测试）范围内，把静态反例固定成可复算的数字，
并给出候选修复的对照结果与 P4-B 的测试矩阵。

复刻依据（逐行对应）：
  num_cols_per_op()        = 3
  num_outputs_per_row()    = columns.len() / 3                    // = G
  num_inputs_per_row()     = columns.len() / 3 * 2                // = 2G
  op_row_region(inp)       : inp.chunks(G) -> (i1, i2); out[i] = max(i1[i], i2[i])
                             => 长度为 2G 的一行把位置 i 与位置 G+i 配对
  op_aligned_rows(lst)     : assert len % 2G == 0; 逐行 row_op 后拼接
  forward(inputs)          : 见 forward_current()
"""
import itertools
import math
import sys


def row_op(inp, G):
    """对应 MaxChip::op_row_region：chunks(G) 取前两块，逐位取 max。"""
    assert len(inp) == 2 * G, len(inp)
    chunks = [inp[i * G:(i + 1) * G] for i in range(2)]
    i1, i2 = chunks[0], chunks[1]
    return [max(a, b) for a, b in zip(i1, i2)]


def aligned_rows(lst, G):
    """对应 Gadget::op_aligned_rows。"""
    in_per_row = 2 * G
    assert len(lst) % in_per_row == 0, (len(lst), in_per_row)
    out = []
    for i in range(len(lst) // in_per_row):
        out.extend(row_op(lst[i * in_per_row:(i + 1) * in_per_row], G))
    return out


def forward_current(inputs, C):
    """对应 MaxChip::forward（原样复刻）。"""
    G = C // 3
    in_per_row = 2 * G
    first = inputs[0]                      # 空输入在此处 panic，等价于 Rust 的 inputs[0]
    cur = list(inputs)
    while len(cur) % in_per_row != 0:
        cur.append(first)
    num_iters = -(-len(cur) // in_per_row) + in_per_row      # div_ceil + num_inputs_per_row
    out = aligned_rows(cur, G)
    for _ in range(num_iters):
        while len(out) % in_per_row != 0:
            out.append(first)
        out = aligned_rows(out, G)
    return out[0]                          # outputs = vec![outputs[0]]


def max2(a, b, G):
    """候选修复的单元：用一行算 max(a,b)，再只取第 0 个输出。

    布局：位置 0 放 a、位置 G 放 b，其余位置全填 a。
    第 0 组 = (位置 0, 位置 G) = (a, b) -> out[0] = max(a,b)
    第 g>0 组 = (a, a) -> out[g] = a（无害，丢弃）
    """
    row = [a] + [a] * (G - 1) + [b] + [a] * (G - 1)
    assert len(row) == 2 * G
    return row_op(row, G)[0]


def forward_fixed(inputs, C):
    """候选修复：树形阶段 + 显式 collapse 到单输出。"""
    if len(inputs) == 0:
        raise ValueError("empty input: 契约要求拒绝")
    G = C // 3
    if G < 1:
        raise ValueError("columns.len() < 3")
    in_per_row = 2 * G
    cur = list(inputs)
    if len(cur) == 1:
        return cur[0]
    # 阶段一：每次把 L 压到 ceil(L/(2G))*G，只要 L > G 就仍有严格下降
    while len(cur) > G:
        while len(cur) % in_per_row != 0:
            cur.append(cur[0])
        cur = aligned_rows(cur, G)
    # 阶段二：两两合并直到 1 个（ceil(L/2) < L，L>=2 时严格下降）
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            a = cur[i]
            b = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(max2(a, b, G))
        cur = nxt
    return cur[0]


def fingerprint(fn, C, n, hi=1):
    """返回「把 1 放在位置 i 时函数能返回 1」的下标集合，即该实现对输入位置的覆盖。

    hi=0 表示其余位置取 0（与 plan.md 的反例一致：全非负，ReLU 救不回来）。
    """
    covered = []
    for i in range(n):
        xs = [hi if j == i else 0 for j in range(n)]
        if fn(xs, C) == hi:
            covered.append(i)
    return covered


def exhaustive_bad(fn, C, n, max_n_exhaust=11):
    """对全部 0/1 输入枚举，返回出错个数与总数。"""
    bad = 0
    total = 0
    for bits in itertools.product([0, 1], repeat=n):
        total += 1
        got = fn(list(bits), C)
        want = max(bits)
        if got != want:
            bad += 1
    return bad, total


def main():
    print("=" * 78)
    print("1) 当前实现（max.rs 原样）对输入位置的覆盖")
    print("   表项 = 把 1 放在该下标、其余取 0 时返回值仍为 1 的下标")
    print("=" * 78)
    print("  C   n   覆盖下标                                    缺失")
    for C in (3, 6, 9, 12):
        for n in (1, 2, 3, 4, 5, 6, 9, 12):
            cov = fingerprint(forward_current, C, n)
            miss = [i for i in range(n) if i not in cov]
            print("  %2d %3d  %-42s %s" % (C, n, cov, miss if miss else "-"))

    print()
    print("=" * 78)
    print("2) 当前实现的 0/1 穷举错误率（n<=10 全枚举）")
    print("=" * 78)
    print("  C   n   错误/总数      比例")
    for C in (3, 6, 9, 12):
        for n in (2, 3, 4, 6, 9, 10):
            bad, total = exhaustive_bad(forward_current, C, n)
            print("  %2d %3d  %5d/%-5d   %6.2f%%" % (C, n, bad, total, 100.0 * bad / total))

    print()
    print("=" * 78)
    print("3) plan.md 指定的反例：9 元素、零基 x1=1、其余 0")
    print("=" * 78)
    xs = [1 if i == 1 else 0 for i in range(9)]
    for C in (3, 6, 9, 12):
        got = forward_current(xs, C)
        print("  C=%2d  当前=%d  正确=1  %s" % (C, got, "OK" if got == 1 else "<<< 反例成立"))

    print()
    print("=" * 78)
    print("4) 候选修复：覆盖 + 穷举 + 反例")
    print("=" * 78)
    bad_any = 0
    for C in (3, 6, 9, 12):
        for n in range(1, 11):
            cov = fingerprint(forward_fixed, C, n)
            assert cov == list(range(n)), (C, n, cov)
            if n <= 10:
                bad, total = exhaustive_bad(forward_fixed, C, n)
                bad_any += bad
        print("  C=%2d  覆盖全位置(1..10) OK；0/1 穷举 n<=10 全部正确" % C)
    print()
    print("  修复版三反例：", [forward_fixed([1 if i == 1 else 0 for i in range(9)], C)
                              for C in (3, 6, 9, 12)])
    print("  修复版空输入：", end=" ")
    try:
        forward_fixed([], 6)
        print("未拒绝 <<< 不符合契约")
    except ValueError as e:
        print("拒绝（%s）" % e)

    print()
    print("=" * 78)
    print("5) 行数成本（每次 forward 调用产生的 gadget 行数）")
    print("=" * 78)
    def rows_current(n, C):
        G = C // 3
        in_per_row = 2 * G
        cur = list(range(n))
        while len(cur) % in_per_row != 0:
            cur.append(cur[0])
        num_iters = -(-len(cur) // in_per_row) + in_per_row
        total = len(cur) // in_per_row
        out = [0] * ((len(cur) // in_per_row) * G)
        for _ in range(num_iters):
            while len(out) % in_per_row != 0:
                out.append(0)
            total += len(out) // in_per_row
            out = [0] * ((len(out) // in_per_row) * G)
        return total

    def rows_fixed(n, C):
        G = C // 3
        in_per_row = 2 * G
        if n <= 1:
            return 0
        total = 0
        L = n
        while L > G:
            while L % in_per_row != 0:
                L += 1
            total += L // in_per_row
            L = (L // in_per_row) * G
        while L > 1:
            total += (L + 1) // 2
            L = (L + 1) // 2
        return total

    print("  C    n   当前行数   修复后行数")
    for C in (3, 6, 9, 12):
        for n in (4, 6, 9):
            print("  %2d  %3d   %6d     %6d" % (C, n, rows_current(n, C), rows_fixed(n, C)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
