#!/usr/bin/env python3
"""P4-A 机器可读记录生成器：manifests/p4a-contract-candidates.json

设计原则：
  - 所有 SHA-256 与穷举数字都从**实物重算**，不手抄；重跑即校验。
  - 记录里显式带 `scope` 字段，说明「本批没编译、没运行」，避免下游把
    静态结论误读成实测结论。
  - 反例/覆盖/行数成本直接调用 p4a_max_model 的纯函数，与本批正文同源。

用法：python scripts/p4a_manifest.py [--check]
      --check 只重算并与已有文件比对，不写盘（不一致则退出码 1）。
"""
import argparse
import hashlib
import json
import os
import sys
import platform
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import p4a_max_model as M  # noqa: E402

OUT = os.path.join(ROOT, "manifests", "p4a-contract-candidates.json")

COLS = (3, 6, 9, 12)                       # 本批考察的列数；3 是对照组（无缺陷）
NS = (1, 2, 3, 4, 5, 6, 9, 12)             # 覆盖指纹用的长度
# plan.md 指定的反例：9 元素、零基 x1=1、其余 0（全非负，ReLU 救不回来）
PLANE = [1 if i == 1 else 0 for i in range(9)]

# 目标源码（本批所有静态分析针对这些文件；身份必须可复核）
SOURCES = [
    ("zkml-main/src/gadgets/max.rs", "src/gadgets/max.rs", "MaxChip：本批问题 1 的缺陷所在"),
    ("zkml-main/src/layers/max_pool_2d.rs", "src/layers/max_pool_2d.rs", "MaxPool2D：窗口长度来源（4/6/9）"),
    ("zkml-main/src/layers/shape/pad.rs", "src/layers/shape/pad.rs", "PadChip：填充值恒为 0 的硬编码"),
    ("zkml-main/src/model.rs", "src/model.rs", "to_field / tensor_map_to_vec / ModelConfig"),
    ("zkml-main/python/converter.py", "python/converter.py", "PADV2 缺失分支、量化与元数据推导"),
]

ARTIFACTS = [
    ("patches/max_reduce_fix.rs", "patch", "Max 归约候选修复（替换目标文件）"),
    ("patches/max_reduce_fix.diff", "diff", "同上 unified diff"),
    ("patches/converter_p4a.py", "patch", "converter 契约补丁（替换目标文件）"),
    ("patches/converter_p4a.diff", "diff", "同上 unified diff"),
    ("patches/model_numeric_contract.rs", "patch", "数值边界 + 禁止填洞（替换目标文件）"),
    ("patches/model_numeric_contract.diff", "diff", "同上 unified diff"),
    ("patches/pad_contract.rs", "patch", "Pad 单输入断言 + 填充值契约（替换目标文件）"),
    ("patches/pad_contract.diff", "diff", "同上 unified diff"),
    ("scripts/p4a_max_model.py", "script", "Max 源码等价复刻 + 反例/穷举/行数成本"),
    ("scripts/p4a_make_converter_patch.py", "script", "converter 补丁生成器（幂等，锚点不唯一即失败）"),
    ("scripts/p4a_make_model_patch.py", "script", "model.rs 补丁生成器"),
    ("scripts/p4a_make_pad_patch.py", "script", "pad.rs 补丁生成器"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nlines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def collect_identity():
    out = []
    for rel, logical, role in SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            raise SystemExit("缺文件：%s" % rel)
        out.append({
            "path": logical,
            "local_path": rel,
            "lines": nlines(p),
            "bytes": os.path.getsize(p),
            "sha256": sha256(p),
            "role": role,
        })
    return out


def collect_artifacts():
    out = []
    for rel, kind, desc in ARTIFACTS:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            raise SystemExit("缺产物：%s" % rel)
        out.append({
            "path": rel,
            "kind": kind,
            "bytes": os.path.getsize(p),
            "sha256": sha256(p),
            "desc": desc,
        })
    return out


def collect_max_defect():
    """覆盖指纹 / 0/1 穷举 / 反例 / 行数成本，全部重算。"""
    coverage = []
    for C in COLS:
        for n in NS:
            cov = M.fingerprint(M.forward_current, C, n)
            coverage.append({
                "cols": C, "n": n,
                "covered": cov,
                "missing": [i for i in range(n) if i not in cov],
            })

    exhaustive = []
    for C in COLS:
        for n in (2, 3, 4, 6, 9, 10):
            bad, total = M.exhaustive_bad(M.forward_current, C, n)
            exhaustive.append({"cols": C, "n": n, "bad": bad, "total": total,
                               "rate": round(bad / total, 6)})

    # plan.md 指定的反例：9 元素、零基 x1=1、其余 0
    cex = [{"cols": C, "got": M.forward_current(PLANE, C), "want": 1} for C in COLS]

    # 最小反例：C=6, n=2, [0,1]
    minimal = {"cols": 6, "input": [0, 1], "got": M.forward_current([0, 1], 6), "want": 1}

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
        total, L = 0, n
        while L > G:
            while L % in_per_row != 0:
                L += 1
            total += L // in_per_row
            L = (L // in_per_row) * G
        while L > 1:
            total += (L + 1) // 2
            L = (L + 1) // 2
        return total

    row_cost = [{"cols": C, "n": n,
                 "current": rows_current(n, C), "fixed": rows_fixed(n, C)}
                for C in COLS for n in (4, 6, 9)]

    return {
        "location": "src/gadgets/max.rs :: MaxChip::forward / op_row_region / op_aligned_rows",
        "mechanism": {
            "G": "columns.len()/3 = num_outputs_per_row",
            "in_per_row": "2G = num_inputs_per_row",
            "pairing": "op_row_region 用 inp.chunks(G) 取前两块再 zip → 一行把位置 i 与位置 G+i 配对",
            "D1": "归约停在 G 个输出（补齐后成不动点），末尾只取 outputs[0] → 返回值退化为某个同余类的最大值",
            "D2": "num_iters = div_ceil + num_inputs_per_row 是经验常数，不是「直到剩 1 个」；多出的迭代全在不动点上空转",
            "independence": "调大 num_iters 无法消除 D1：outputs 长度仍是 G，outputs[0] 仍是同余类最大",
        },
        "minimal_counterexample": minimal,
        "plan_md_counterexample": cex,
        "coverage_fingerprint": {
            "definition": "把 1 放在下标 i、其余取 0 时函数返回值仍为 1 的下标集合",
            "rows": coverage,
        },
        "exhaustive_01": {
            "definition": "对全部 0/1 输入全枚举（非抽样），统计取值 != max(bits) 的例数",
            "rows": exhaustive,
        },
        "row_cost": {
            "definition": "单次 MaxChip::forward 产生的 gadget 行数（由调度逻辑计数，非实测）",
            "rows": row_cost,
        },
        "affects_pooled_windows": {
            "note": "max_pool_2d.rs 的 splat 只在越界时跳过，因此窗口长度不是常数",
            "lengths": [9, 6, 4],
            "kinds": ["内部 3x3", "边缘（一边越界）", "角（两边越界）"],
            "conclusion": "C=6 时 n=4/6/9 全部踩中 D1，不是只有内部窗口出错",
        },
    }


def collect_max_fix():
    """候选修复的静态论证与复刻验证结果（不是编译/单测结果）。"""
    return {
        "patch": "patches/max_reduce_fix.rs",
        "algorithm": {
            "stage1_tree": "while L > G { 补齐到 2G 的倍数（补 cur[0]）; 一次行归约 }  L -> ceil(L/(2G))*G",
            "stage2_collapse": "while L > 1 { 两两 max2 }  L -> ceil(L/2)",
            "max2_layout": "一行长 2G：位置 0 放 a、位置 G 放 b、其余填 a → 第 0 组=(a,b) → out[0]=max(a,b)",
            "why_not_naive": "[a,b] 直接补齐送行归约时 a、b 落在不同组的同一侧，永不被比较 —— 正是原缺陷机理",
        },
        "invariants": {
            "I1": "cur 每个 cell 都是原始输入某个非空子集的最大值 ⇒ max(cur) = 全窗口最大",
            "I2": "补齐只复制 cur[0]（集合内元素），既不引入更大值也不删元素",
            "I3": "一次行归约按 (i, G+i) 配对，覆盖全部 2G 输入且不重不漏",
            "I4": "max2(a,b) 精确等于 max(a,b)，长度减半",
            "termination": "阶段一 L>G 时 ceil(L/(2G))*G < L；阶段二 ceil(L/2) < L（L≥2）",
        },
        "domain": {
            "supported": "任意 n≥1；任意 3 的倍数且 ≥3 的列数",
            "verified_by_replica": "n ∈ 1..10，cols ∈ {3,6,9,12}：覆盖全位置且 0/1 穷举全过",
        },
        "rejects": {
            "columns_len_not_multiple_of_3": "构造期 assert_eq!（原实现用整除静默截断，例如 7 列被当 2 组、第 7 列永不使用）",
            "columns_len_lt_3": "构造期 assert!",
            "empty_input": "Err(Error::Synthesis)（原实现在 inputs[0] 处 panic）",
            "single_element": "直接返回，占 0 行",
        },
        "uncompiled_risk": "Error::Synthesis 变体名需在 P4-B 首次编译时确认；该 rev 的 plonk::Error 若无此 unit variant，换已知变体即可，不影响算法",
        "replica_verification": {
            "counterexample_plane": {
                "cols": list(COLS),
                "got": [M.forward_fixed(PLANE, C) for C in COLS],
                "want": [1] * len(COLS),
            },
            "empty_input": "ValueError（契约要求拒绝）",
            "note": "以上均为 Python 复刻结果，非 Rust 执行结果",
        },
    }


def collect_padv2():
    return {
        "gap_before": {
            "converter": "converter.py 完全没有 PADV2 分支（P3 实测停在 layer 2: 60, PADV2）",
            "p3_probe_was_not_a_contract": "P3 探针补丁只是「当 PAD 处理」，未做任何图检查 ⇒ -inf→0 降级无条件发生",
            "rust_side": "PadChip 硬编码 constants.get(&0)（填充值恒为 0）；dag.forward 在调层前把全部 inp_idxes 从 tensor_map clone 出来",
            "silent_fill": "tensor_map_to_vec 用 smallest_tensor 静默填补索引空洞 ⇒ 只从 tensors 删 idx1 不够，漏改的层会拿到形状可能恰好对得上的错误张量",
        },
        "graph_indexing": "按 tensor ID 建 producer[t] 与 consumers[t]=[(op_idx, input_pos)]，不依赖算子列表顺序或「相邻」",
        "conditions": [
            {"id": "P1", "rule": "常量是单元素，且为 -inf 或 0", "why": "其它常量（如 -3.0）无法用 0 填充等价"},
            {"id": "P2", "rule": "-inf 时数据输入的生产者 fused activation 必须是 RELU",
             "why": "0 填充与 -inf 填充等价 ⟺ max(x) ≥ 0；ReLU 给的就是这个保证"},
            {"id": "P3", "rule": "Pad 输出恰好 1 个消费者，且是 pad=VALID、无 fused activation 的 MAX_POOL_2D",
             "why": "Pad 输出若还被别的算子观察，降级就不只影响池化"},
            {"id": "P4", "rule": "Pad 输出不是图输出", "why": "公开张量不能被悄悄改值"},
            {"id": "P5", "rule": "填充在 H/W 上对称、每侧 < 核尺寸；batch/channel 轴无填充",
             "why": "对称 = PyTorch 语义；pad ≥ kernel 会出现整窗皆填充，此时 0 与 -inf 结果不同"},
        ],
        "reject_policy": "任一条件不满足即拒绝，不静默降级",
        "p5_check_this_graph": "本图 pad=1, kernel=3 → 满足（对称且 1 < 3）",
        "metadata_cleanup": {
            "registry": {"PAD": [1], "PADV2": [1, 2], "MEAN": [1]},
            "extra_guard": "该 tensor 的全部消费者都只在元数据位置引用它，才允许移除",
            "recompute": "按过滤后的 inp_idxes 重算 keep_tensors，而不是只从 tensors 删条目",
            "expected_removed_tensors": "PADV2 的 paddings([4,2]) 与 -inf 常量([1])、4 个 PAD 的 paddings（预期，本批未执行转换）",
            "expected_dep_change": "PADV2 的依赖 [48,45,1] → [48]；tensor ID 保留不变（预期，未验证）",
            "not_registered": "RESHAPE/TRANSPOSE/SLICE/SPLIT 等虽同为元数据驱动，但本图没有这些算子，登记了也无法验证 —— 无证据的扩大改动面不做",
        },
        "rust_side_assert": {
            "patch": "patches/pad_contract.rs",
            "change": "恢复被注释掉的 assert_eq!(tensors.len(), 1)",
            "why": "原注释 `// FIXME: the pad from tflite is actually two, but mine is one` 正是当年绕过点",
        },
    }


def collect_numeric():
    return {
        "encoding_interval": {
            "code": "model.rs:297-302  F::from((x + 2^31) as u64) - F::from(2^31)",
            "branches": [
                {"range": "[-2^31, 2^31-1]", "behavior": "x + 2^31 ∈ [0, 2^32-1]，as u64 精确，减回去等于 x", "ok": True},
                {"range": "x < -2^31", "behavior": "x + 2^31 为负 → as u64 补码回绕 → 编码成 x + 2^64（巨大正元素），静默改义", "ok": False},
                {"range": "x > 2^31-1", "behavior": "x + 2^31 可能溢出 i64（debug panic / release 回绕）", "ok": False},
            ],
            "consequence": "-inf → INT64_MIN = -2^63 落在第二条 ⇒ 若在 Rust 侧被赋值会变成 2^63 量级正元素。P3「对下游无影响」只在 TFLite/msgpack 层成立",
        },
        "derived_constraints": [
            {"name": "k ≤ 32", "derivation": "min_val = -(2^(k-1)) 必须 ≥ -2^31", "fail_at": "k=33"},
            {"name": "sf ≤ 2^22", "derivation": "shift_min_val = -(sf²·2^17) 不溢出 i64，即 sf²·2^18 ≤ 2^63-1",
             "pass_eg": "sf=2^22 → 2^62", "fail_eg": "sf=2^23 → 2^64"},
        ],
        "i64_overflow_note": "原代码 -(global_sf * global_sf * (1 << 17)) 是 i64 乘法，release 溢出静默；补丁改 i128::checked_mul",
        "maxchip_own_domain": {
            "requirement": "convert_to_u64(&(v + 2^49)) 要求 v + 2^49 ∈ [0, 2^64)（big_digits.len()==2 会 panic）",
            "general": "-2^49 ≤ v < 2^64 - sf²·2^17",
            "conclusion": "配合编码区间，编码区间才是绑定约束；写下来是为防止「以为 Max 能处理任意大数」",
        },
        "rounding": {
            "before": "converter.py 用 (x*sf).round() = numpy half-to-even（银行家舍入）",
            "after": "显式 round_half_away_from_zero",
            "why": "理由不是更准，而是规则必须写死并被两边知道；否则首个分歧会被误判成算子错误",
        },
        "checks": [
            {"id": "N1", "rule": "loader 逐元素检查落到 [-2^31, 2^31-1]，越界报 tensor id + 下标"},
            {"id": "N2", "rule": "to_field 入口 debug_assert"},
            {"id": "N3", "rule": "assert!(config.k <= 32)；sf²·2^18 用 i128::checked_mul"},
            {"id": "N4", "rule": "tensor_map 缺键即 panic 并打印缺失列表，禁止 smallest_tensor 填洞"},
            {"id": "N5", "rule": "smallest_tensor 改名 _smallest_tensor（防 deny(warnings)）"},
        ],
        "open_items": [
            "lookup 表的差值域到底覆盖 [0,2^k] 还是更窄：本批未逐表核对 input_lookup 填充方式（P4-B 第一项）",
            "各 gadget 自己的 min_val/max_val 派生域（ReLU/Div/BiasDiv 各不相同）：本批只给出框架与 Max 一条",
        ],
    }


def collect_matrix():
    return {
        "compile": [
            "cargo check --locked 只编 src，确认 4 个补丁的类型/借用/Error::Synthesis 变体",
            "先单独应用 max_reduce_fix 编译；再叠加另三个（分开才能定位失败补丁）",
        ],
        "max": [
            {"case": "n=2, [0,1]", "expect": "1（原实现 0，最小复现）"},
            {"case": "n=9, x1=1 其余 0", "expect": "1（原实现 0）"},
            {"case": "9 元素逐位置取最大（9 例）", "expect": "各自等于该位置值"},
            {"case": "并列最大 / 全 0 / 全同值 / 合法负值（全 -3）", "expect": "正确"},
            {"case": "全负窗口（neg_const 造出的池化窗口）", "expect": "正确 —— 必须单独覆盖"},
            {"case": "长度 4 / 6 / 9（角/边/内部）", "expect": "正确"},
            {"case": "非整行长度 5、7 及奇数长度", "expect": "正确"},
            {"case": "columns ∈ {3,6,9,12} 全部重跑", "expect": "正确（3 列是对照组）"},
            {"case": "空输入", "expect": "Err，不 panic"},
            {"case": "约束侧：每行 (in1-out)(in2-out)=0 与两条差值 lookup 同时成立", "expect": "MockProver 通过"},
        ],
        "pad_max": [
            {"case": "本图 conv1+ReLU → PADV2(-inf) → MaxPool", "expect": "转换通过，证据 JSON 落盘"},
            {"case": "生产者不是 ReLU", "expect": "拒绝"},
            {"case": "PADV2 常量改成 -3.0", "expect": "拒绝"},
            {"case": "Pad 输出多接一个消费者", "expect": "拒绝"},
            {"case": "Pad 输出设为图输出", "expect": "拒绝"},
            {"case": "pad=2, kernel=3（整窗可为填充）", "expect": "拒绝"},
            {"case": "删 pads 后漏改某层 inp_idxes", "expect": "N4 立即 panic 并打印缺失索引"},
            {"case": "MNIST 既有 msgpack 在 N4 下加载", "expect": "索引连续→通过；不连续→说明填洞在用，需回退 N4 或修数据"},
        ],
        "numeric": [
            {"case": "x = -2^31 / 2^31-1", "expect": "编码精确"},
            {"case": "x = -2^31-1 / 2^31", "expect": "拒绝并报 tensor id + 下标"},
            {"case": "INT64_MIN", "expect": "拒绝（正是 -inf 的产物）"},
            {"case": "NaN / ±Inf", "expect": "converter 侧 _quantize_checked 就拒绝"},
            {"case": "k=32 / k=33", "expect": "通过 / 拒绝"},
            {"case": "sf=2^22 / sf=2^23", "expect": "通过 / 拒绝"},
        ],
        "cost_caveat": "即使只测一个 gadget 的 MockProver 也走 configure，而 configure 按 2^k 建 HashMap 与 lookup；P4-B 必须显式指定小 k 并先算 configure 开销",
        "stop_condition": "任一用例与参考不符即停；宁可留失败原件，也不调参掩盖",
    }


def build():
    return {
        "schema": "zkml-repro/p4a-contract-candidates/1",
        "batch": "P4-A",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": {
            "script": "scripts/p4a_manifest.py",
            "sha256": sha256(os.path.join(HERE, "p4a_manifest.py")),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "recompute_hint": "python scripts/p4a_manifest.py --check   # 重算并与本文件比对",
        },
        "scope": {
            "allowed": ["固定源码身份", "候选修复", "静态审查", "P4-B 测试矩阵"],
            "forbidden_this_batch": [
                "编译任何 Rust（cargo check 也没跑）",
                "运行 converter / 模型 / configure / synthesize / MockProver",
                "生成 SRS / prove / verify",
            ],
            "evidence_class": {
                "source_identity": "实测（文件摘要）",
                "max_counterexample": "静态复刻（Python 等价调度，非 Rust 执行）",
                "numeric_bounds": "静态推导（由 to_field / shift_min_val 代码反推，非实测）",
                "patch_applicability": "patch --dry-run 通过 / py_compile 通过（未编译、未执行）",
            },
        },
        "source_identity": collect_identity(),
        "artifacts": collect_artifacts(),
        "problem_1_max": {"defect": collect_max_defect(), "candidate_fix": collect_max_fix()},
        "problem_2_3_padv2": collect_padv2(),
        "problem_4_numeric": collect_numeric(),
        "p4b_test_matrix": collect_matrix(),
        "not_concluded": [
            "「Max 已修复」——只做了静态论证与 Python 复刻，没有编译、没有约束检查",
            "「PADV2 补丁可直接采纳」——拒绝条件写好了，但没有在本图跑过一次转换",
            "「依赖清理已验证」——keep_tensors 重算正确性依赖 N4 一起上，两者都未执行",
            "「k ≤ 32 / sf ≤ 2^22 是实测结论」——是从代码反推的约束",
            "任何关于目标规模、prove 时间、1s 目标的判断",
        ],
        "reproduce": [
            "python scripts/p4a_max_model.py",
            "python scripts/p4a_make_converter_patch.py",
            "python scripts/p4a_make_model_patch.py",
            "python scripts/p4a_make_pad_patch.py",
            "python scripts/p4a_manifest.py --check",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只重算比对，不写盘")
    args = ap.parse_args()

    doc = build()
    text = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    if args.check:
        if not os.path.exists(OUT):
            print("缺少 %s，无法比对" % OUT)
            return 1
        with open(OUT, "r", encoding="utf-8") as f:
            old = json.load(f)
        # generated_at 每次都变，比对时剔除
        old.pop("generated_at", None)
        cur = json.loads(text)
        cur.pop("generated_at", None)
        if old == cur:
            print("OK  manifest 内容与实物一致（generated_at 已忽略）")
            return 0
        # 定位第一处差异，便于修
        for k in cur:
            if old.get(k) != cur[k]:
                print("DIFF  顶层键: %s" % k)
        for k in old:
            if k not in cur:
                print("DIFF  旧文件多余键: %s" % k)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("written: %s (%d bytes)" % (os.path.relpath(OUT, ROOT), len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
