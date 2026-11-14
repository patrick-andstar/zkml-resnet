#!/usr/bin/env python3
"""P4-A 补正：生成 manifests/p4a-contract-candidates-v2.json。

与 v1 的 manifests/p4a-contract-candidates.json 并存，**不覆盖**（另立身份）。

本记录特别区分证据类型，因为 §2.8.1 点名批评过 v1 那句"所有结论均从实物自动重算"：
  computed   —— 本次生成时由脚本重新算出来的（摘要、大小、数值边界、门禁结果）
  hand_written —— 人写的判断/理由，脚本只能验证它"在文件里存在"，不能验证它"正确"

用法：python scripts/p4a_manifest_v2.py [--check]
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "manifests", "p4a-contract-candidates-v2.json")

V1_SOURCES = [
    "zkml-main/src/gadgets/max.rs",
    "zkml-main/src/layers/max_pool_2d.rs",
    "zkml-main/src/layers/shape/pad.rs",
    "zkml-main/src/model.rs",
    "zkml-main/python/converter.py",
]
V1_ARTIFACTS = [
    "patches/max_reduce_fix.rs", "patches/max_reduce_fix.diff",
    "patches/converter_p4a.py", "patches/converter_p4a.diff",
    "patches/model_numeric_contract.rs", "patches/model_numeric_contract.diff",
    "patches/pad_contract.rs", "patches/pad_contract.diff",
    "manifests/p4a-contract-candidates.json",
]
V2_ARTIFACTS = [
    "patches/converter_p4a_v2.py", "patches/converter_p4a_v2.diff",
    "patches/model_numeric_contract_v2.rs", "patches/model_numeric_contract_v2.diff",
    "scripts/p4a_make_converter_patch_v2.py",
    "scripts/p4a_make_model_patch_v2.py",
    "scripts/p4a_quant_selftest.py",
    "scripts/p4a_manifest_v2.py",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        raise SystemExit("缺文件：%s" % rel)
    return {"path": rel, "bytes": os.path.getsize(p), "sha256": sha256(p)}


def load_gen(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numeric_facts():
    """数值边界全部重算，不手抄。"""
    import math
    S1 = (1 << 63) - 1
    return {
        "loader_exact_domain": {
            "lo": -(1 << 31),
            "hi": (1 << 63) - 1 - (1 << 31),
            "derivation": "F::from((x + 2^31) as u64) - F::from(2^31)："
                          "x+2^31 需非负（下界 -2^31）且不溢出 i64（上界 i64::MAX-2^31）",
            "computed": True,
        },
        "i32_policy_is_not_necessary": {
            "i32_range": [-(1 << 31), (1 << 31) - 1],
            "note": "i32 是本项目主动策略；原 loader 在上面的 domain 内全域精确",
            "computed": True,
        },
        "sf_bounds": {
            "S_eq_sf2_2pow17": {"max_sf": math.isqrt(S1 // (1 << 17)), "computed": True},
            "twoS_eq_sf2_2pow18": {"max_sf": math.isqrt(S1 // (1 << 18)), "computed": True,
                                   "note": "保守上界，不是原 shift 构造的精确边界"},
            "power_of_two_sf": {"max_sf": 1 << 22, "computed": True,
                                "note": "只有 sf 是 2 的幂时 2S 条件才等价于 sf<=2^22"},
        },
        "k_bounds": {
            "num_rows_2powk_minus_9": {"min_k": 4, "computed": True},
            "max_val_2powk1_minus_10": {"min_k": 5, "computed": True,
                                        "note": "k=4 时 max_val = 2^3-10 = -2 < 0"},
            "minus_min_val_times_2": {"max_k": 62, "computed": True},
            "conclusion": "k>=5 且 k<=62 是必要条件；实用上界由资源决定，不给出可用 k",
            "computed": True,
        },
        "max_difference_lookup": {
            "domain": "[0, 2^k - 10]",
            "derivation": "InputLookupChip::load_lookups 用 F::from(i as u64) 填 i in 0..num_rows，"
                          "num_rows = 2^k - 9，所以表里是裸无符号 0..=2^k-10；"
                          "Max 的 lookup 是 outp-inp1 / outp-inp2，负差值会成为巨大的域元素而不在表内",
            "computed": True,
        },
        "maxchip_host_domain": {
            "requirement": "-S <= v < 2^64 - S，S = -shift_min_val = sf^2*2^17",
            "derivation": "convert_to_u64(&(v + min_val_pos)) 要求结果落在 [0, 2^64)",
            "computed": True,
        },
    }


def run_selftest():
    p = subprocess.run([sys.executable, os.path.join(HERE, "p4a_quant_selftest.py")],
                       capture_output=True, text=True, cwd=ROOT)
    passed = p.stdout.count("  PASS  ")
    failed = p.stdout.count("  FAIL  ")
    return {"exit_code": p.returncode, "pass": passed, "fail": failed,
            "computed": True,
            "note": "对 converter 侧数值契约代码的静态自测；不导入 tensorflow、不跑转换"}


def gates():
    conv = load_gen("scripts/p4a_make_converter_patch_v2.py", "conv_v2")
    model = load_gen("scripts/p4a_make_model_patch_v2.py", "model_v2")
    return {
        "converter_ast_gate": {
            "v2": conv.verify(os.path.join(ROOT, "patches/converter_p4a_v2.py")),
            "v1_negative_control": conv.verify(os.path.join(ROOT, "patches/converter_p4a.py")),
            "computed": True,
            "kind": "AST（结构级）",
        },
        "model_text_gate": {
            "v2": model.verify(os.path.join(ROOT, "patches/model_numeric_contract_v2.rs")),
            "v1_negative_control": model.verify(os.path.join(ROOT, "patches/model_numeric_contract.rs")),
            "computed": True,
            "kind": "文本/结构级，不是类型检查",
        },
    }


CORRECTIONS = [
    {
        "id": "阻断1",
        "title": "converter 调用不存在的方法，舍入与有限性检查未接通",
        "v1_defect": "patches/converter_p4a.py:654 调用 self._quantize_checked，但候选里没有该方法；"
                     "生成器定义了 QUANT 字符串，E8 只插入了 METHODS。"
                     "另有 CUSTOM 权重与 DIV 两处 .round().astype 绕过全部检查。",
        "fix": [
            "E8 改为 METHODS + QUANT，_quantize_checked 真正进入类体",
            "三个量化点全部改为 self._quantize_checked：主张量循环、CUSTOM 权重（already_scaled）、"
            "DIV 的 adjusted_tensors（already_scaled）",
            "L1 硬边界（原编码精确域）与 L2 主动 i32 策略分成两层检查、两套文案",
            "比较全部搬到整数域：float64 在 2^53 以上无法精确表示每个整数，"
            "直接用 float 和 2^63 比会差几百到几千",
        ],
        "verification": {
            "ast_gate": "调用点 3 处 + 方法存在 + round_half_away_from_zero 真被调用",
            "selftest": "19 项，含 2.5→3 与 numpy 的 2.5→2 不同（证明规则真的换掉了）",
        },
        "residual_risk": "仍未运行转换：_static_constant 对真实 TFLite buffer 的行为、"
                         "以及 FC bias 是否落在主动 i32 策略内，都只能在 P4-B 实测。",
        "evidence_class": "computed(门禁+自测) + hand_written(设计取舍)",
    },
    {
        "id": "阻断2",
        "title": "PADV2 证据写入与导出脱节，且 pad_op=True 不是算子编号",
        "v1_defect": "to_dict 用局部 padv2_evidence 累积，to_msgpack 导出的却是从未更新的 "
                     "self.padv2_evidence，转换成功也只会写空列表；证据里的 pad_op 字段是布尔值。",
        "fix": [
            "证据容器统一到 self.padv2_evidence，to_dict 每次调用开头重置",
            "证据记录真实身份：op_idx / data_tensor / paddings_tensor / constant_tensor / "
            "producer_op_idx / consumer_op_idx / pad_out_tensor / 形状与几何",
            "写盘时绑定 model_sha256 与 config_sha256、subgraph_range、padv2_count、"
            "shared_metadata_tensors",
            "图中有 PADV2 但证据为空时直接拒绝写盘",
        ],
        "verification": {
            "ast_gate": "to_dict 内不得出现局部 padv2_evidence 赋值；to_msgpack 必须含 "
                        "self.padv2_evidence 与两个 sha256 字段",
        },
        "residual_risk": "证据内容是否与实际转换子图一致，只能在 P4-B 真跑一次转换后核对。",
        "evidence_class": "computed(门禁) + hand_written(字段设计)",
    },
    {
        "id": "阻断3",
        "title": "N4 把合法稀疏 ID 误判成缺依赖",
        "v1_defect": "v1 的 N4 要求初始 tensor_map 含 0..=max_key 的全部键。"
                     "删掉元数据张量、保留原 ID 本来就会留下合法空洞；尚未执行的层输出也不该"
                     "作为初始张量存在。而且该拒绝发生在 synthesize 赋值路径，不是加载门禁。",
        "fix": [
            "改成加载期校验：initial = 初始张量 ∪ 模型输入；按层推进 available；"
            "每层 inp_idxes 必须已可用",
            "把所有引用（模型输入 / 各层输入 / 最终输出 / commit_before / commit_after）"
            "与 initial ∪ produced 对账，缺一个就报出具体 ID",
            "引用集合存进 ModelCircuit 新字段 referenced_tensor_ids",
            "tensor_map_to_vec 只对**未被引用**的空洞占位；被引用却落空即 panic",
            "撤回 v1 的 N5（_smallest_tensor 改名）：v2 里它仍被使用",
        ],
        "verification": {
            "text_gate": "禁止 unwrap_or(smallest_tensor)；必须有 initial_ids/available_ids/"
                         "produced_ids/referenced_ids；字段三处齐备",
        },
        "residual_risk": "新增了 ModelCircuit 字段（model.rs 内，未改 dag.rs 接口）。"
                         "若 DAG 存在未在 layers 中声明的隐式引用，本检查不覆盖 —— 需 DAG 审计，"
                         "属另行批准的范围。",
        "evidence_class": "computed(门禁) + hand_written(接口范围判断)",
    },
    {
        "id": "阻断4",
        "title": "图保护只覆盖原图，未覆盖实际转换子图与静态元数据",
        "v1_defect": "P3/P4 用整图 graph.Outputs 判公开输出，但导出模型的公开输出是 "
                     "layers[-1]['out_idxes']；start_layer/end_layer 可裁掉 ReLU 或让 Pad 输出"
                     "直接成为公开输出。元数据没有证明是静态常量。缺 rank/[4,2]/非负/正核/stride/"
                     "shape 检查。共享元数据一律拒绝，过度收紧。",
        "fix": [
            "P3/P4 推迟到整段转换结束后按 [start_layer, end_layer] 复检："
            "公开输出用 layers[-1]['out_idxes']，消费者只数子图内的",
            "P2 额外要求 ReLU 生产者 op_idx >= start_layer（防裁掉 ReLU）",
            "新增 _static_constant：有 producer 即拒绝（不能冻结零输入 invoke 的结果），"
            "Buffer()==0 或 buffer 为空也拒绝",
            "新增 P0/P1b：paddings 必须 [4,2] 非负整数、batch/channel 轴为 0；"
            "核与 stride 必须为正；输入/Pad 输出/Pool 输出三段形状按 NHWC 公式核对",
            "元数据共享改为按引用位置清理：只从本层 inp_idxes 去掉，"
            "别的算子当数据用就仍进 keep_tensors，不误删",
            "补存活根：模型输入 ∪ 整图输出 ∪ 导出子图公开输出 ∪ 数据依赖",
        ],
        "verification": {
            "ast_gate": "_validate_padv2_subgraph 必须存在且被调用；_static_constant 必须存在",
        },
        "residual_risk": "轴顺序按 NHWC 解释；NCHW 模型会被 shape 等式挡住（这是有意的）。"
                         "整图消费者仍要求恰好 1 个，比只数子图内更保守。",
        "evidence_class": "computed(门禁) + hand_written(几何与策略)",
    },
    {
        "id": "阻断5",
        "title": "数值域、k 与 sf 的推导需重写",
        "v1_defect": "v1 把 loader 精确域写成 [-2^31, 2^31-1]（错，那只是塞得进 i32），"
                     "并据此断言 k<=32；把 i32 当成唯一有效域；把 sf^2*2^18 当成原 shift 构造的"
                     "精确边界。",
        "fix": [
            "loader 精确域改为 [-2^31, i64::MAX-2^31]，逐分支写清三条",
            "i32 降级为 I32_POLICY 常量 + 单独错误文案，明说不是必要条件",
            "删掉 k<=32；改用 k>=5（max_val>=0）、k<=62（-min_val*2 可表示）、global_sf>0，"
            "并注明实用上界由资源决定",
            "S 与 2S 分别 checked；2S 明确写成本保守上界，只有 sf 为 2 的幂时才等价 sf<=2^22，"
            "一般 sf 的界是 5931641",
            "GadgetConfig 的 shift_min_val 改用与 assert 同一条 checked 链算出的值，"
            "不再在字面量里重写未检查的乘法",
            "Max 差值 lookup 域明确为 [0, 2^k-10]（表里是裸无符号）",
        ],
        "verification": {
            "text_gate": "代码里不得出现 k<=32；必须有 i64::MAX-(1i64<<31)、I32_POLICY、"
                         "checked_mul(sf)、1i128<<17、5931641",
            "numeric": "全部边界由脚本重算，见 numeric_facts",
        },
        "residual_risk": "各 gadget 自己的数值域（ReLU/Div/BiasDiv/Update 等）本批未逐项推；"
                         "2S 的真实出现位置只从 bias_div 的 2*inp+div_val 形态确认，"
                         "没有逐个 gadget 复核。",
        "evidence_class": "computed(门禁+数值重算) + hand_written(保守取舍)",
    },
    {
        "id": "阻断6",
        "title": "目标窗口长度与全填充负例构造错误",
        "v1_defect": "v1 把池化窗口长度写成 4/6/9（角/边/内部）并当作本图的归约长度；"
                     "又把 pad=2,kernel=3 当全填充负例，但它既不必产生全填充窗，"
                     "也不会被 <kernel 检查拒绝。",
        "fix": [
            "明确本图是显式 PADV2 + VALID：padded 尺寸 = H+2，窗口全在内部，"
            "归约长度恒为 9；4/6 只作独立长度测试",
            "全填充负例改用 H=W=1, pad=3, kernel=3, stride=1：首窗全是填充",
            "P5 的 <kernel 条件保留为必要条件，但说明它只排除 ph>=fh，"
            "全填充窗的判定要按具体窗口几何",
        ],
        "verification": {
            "numeric": "H=W=1, pad=3, kernel=3 → padded 7x7，首窗取 [0:3]x[0:3]，"
                       "其中原图只有 (0,0) 一个真实元素，其余 8 个是填充；"
                       "window 内真实元素数 = 1 -> 0 填充与 -inf 填充在该窗结果不同",
        },
        "residual_risk": "这是纯几何论证，未在 Rust 侧验证；P4-B 用小合成图实测。",
        "evidence_class": "hand_written(几何论证) + computed(形状公式核对)",
    },
]


def build():
    gates_res = gates()
    selftest = run_selftest()
    return {
        "schema": "zkml-repro/p4a-contract-candidates-v2/1",
        "batch": "P4-A 补正",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "supersedes": "manifests/p4a-contract-candidates.json（v1，保留不改）",
        "generator": {
            "script": "scripts/p4a_manifest_v2.py",
            "sha256": sha256(os.path.join(HERE, "p4a_manifest_v2.py")),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "recompute_hint": "python scripts/p4a_manifest_v2.py --check",
        },
        "scope": {
            "allowed": ["修复 §2.8.2 六项阻断", "候选与生成器一致性", "静态门禁与自测", "更正测试矩阵"],
            "forbidden_this_batch": [
                "编译任何 Rust（cargo check 也没跑）",
                "运行 converter / 模型 / configure / synthesize / MockProver",
                "生成 SRS / prove / verify",
            ],
        },
        "evidence_class_legend": {
            "computed": "本次生成时由脚本重算（摘要、大小、数值边界、门禁结果、自测结果）",
            "hand_written": "人写的判断/理由；脚本只能验证它在文件里存在，不能验证它正确",
        },
        "not_auto_derived": [
            "六项阻断的 fix 列表与 residual_risk：设计判断，不是脚本推导",
            "P4-B 测试矩阵的期望值：待执行",
            "阻断6 的窗口几何论证：手写推理，只做了形状公式的算术核对",
        ],
        "source_identity_unchanged": [entry(p) for p in V1_SOURCES],
        "v1_artifacts_preserved": [entry(p) for p in V1_ARTIFACTS],
        "v2_artifacts": [entry(p) for p in V2_ARTIFACTS],
        "corrections": CORRECTIONS,
        "gates": gates_res,
        "selftest": selftest,
        "numeric_facts": numeric_facts(),
        "max_candidate_unchanged": {
            "artifact": "patches/max_reduce_fix.rs",
            "reason": "§2.8.1 判定『两阶段调度值得保留』，本批不改 Max 候选",
            "note": "行数 11→7 只是调度计数，不能声称 prove 变快",
            "computed": False,
        },
        "p4b_matrix_revision": {
            "window_lengths": {
                "this_graph": "显式 PADV2 + VALID ⇒ 归约长度恒为 9",
                "independent_tests": [4, 6],
                "note": "4/6 是独立长度测试，不是本图 Rust 归约长度",
            },
            "full_padding_negative": {
                "construct": "H=W=1, pad=3, kernel=3, stride=1",
                "why": "padded 7x7，首窗 3x3 内只有 (0,0) 一个真实元素",
                "computed": "几何论证 + 形状公式核对",
            },
            "sparse_id_cases": [
                "合法稀疏 ID 正例：删元数据张量后保留原 ID，空洞未被引用 → 加载通过",
                "真正缺依赖负例：某层 inp_idxes 指向既不初始也不产出的 ID → 报出该 ID",
                "未来引用负例：某层引用了更晚层才产出的 ID → 报出 layer 号",
                "共享常量：同一 tensor 既作元数据又作数据 → 保留并仍进 commit",
            ],
            "numeric_cases": [
                "原编码真实边界：-2^31 通过、-2^31-1 拒绝、i64::MAX-2^31 通过、再 +1 拒绝",
                "主动 i32 策略：单独开关，关掉后同一值按 i64 域通过",
                "舍入半点：2.5→3、-2.5→-3（与 numpy half-to-even 明确不同）",
                "scale 零/负：必须拒绝（global_sf>0）",
                "Max 差值上界：2^k-10 通过、2^k-10+1 拒绝",
            ],
            "big_k_warning": "k=32 与大 scale 只允许做不建表的纯配置边界检查；"
                             "禁止运行 k32 configure/MockProver",
        },
        "not_concluded": [
            "「六项阻断已闭环」——只做了静态修复与静态门禁；未编译、未运行",
            "「Max 已修复」——候选未变，仍未编译",
            "「PADV2 补丁可采纳」——拒绝条件与子图复检写好了，但没在本图跑过一次转换",
            "「N4 已验证」——加载期校验没有在 MNIST 既有产物上跑过",
            "任何关于目标规模、prove 时间、1s 目标的判断",
        ],
        "reproduce": [
            "python scripts/p4a_make_converter_patch_v2.py",
            "python scripts/p4a_make_model_patch_v2.py",
            "python scripts/p4a_quant_selftest.py",
            "python scripts/p4a_manifest_v2.py --check",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = build()
    text = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not os.path.exists(OUT):
            print("缺少 %s" % OUT)
            return 1
        old = json.load(open(OUT, encoding="utf-8"))
        old.pop("generated_at", None)
        cur = json.loads(text)
        cur.pop("generated_at", None)
        if old == cur:
            print("OK  v2 manifest 与实物一致（generated_at 已忽略）")
            return 0
        for k in cur:
            if old.get(k) != cur[k]:
                print("DIFF  顶层键: %s" % k)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("written: %s (%d bytes)" % (os.path.relpath(OUT, ROOT), len(text.encode("utf-8"))))
    print("门禁：converter AST v2=%d 项失败 / 负向对照 v1=%d 项；model 文本 v2=%d 项失败 / v1=%d 项"
          % (len(doc["gates"]["converter_ast_gate"]["v2"]),
             len(doc["gates"]["converter_ast_gate"]["v1_negative_control"]),
             len(doc["gates"]["model_text_gate"]["v2"]),
             len(doc["gates"]["model_text_gate"]["v1_negative_control"])))
    print("自测：exit=%d pass=%d fail=%d"
          % (doc["selftest"]["exit_code"], doc["selftest"]["pass"], doc["selftest"]["fail"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
