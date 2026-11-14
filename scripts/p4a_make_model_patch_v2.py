#!/usr/bin/env python3
"""P4-A 补正：生成 model.rs 的 v2 数值/依赖契约补丁（不改仓库原文件）。

针对 plan.md §2.8.2 的阻断 3、5 逐条修正；v1 产物（patches/model_numeric_contract.rs）
保留不动，本脚本写 _v2 新文件，另立身份。

v2 相对 v1 的差异：
  [阻断3] N4 不再要求 tensor_map 含 0..=max_key 的全部键（那个要求把"删元数据张量、
          保留原 ID"本来就会留下的合法空洞误判成缺依赖，而且尚未执行的层输出也不该
          作为初始张量存在）。改为：
            * 在 generate_from_msgpack 入口做**加载期**校验：初始可用 = 初始张量
              ∪ 模型输入；按层推进，每层输入必须已经可用；再把所有引用
              （模型输入 / 各层输入 / 最终输出 / commit_before / commit_after）
              与"初始 ∪ 已产出"对账。
            * 把引用集合存进 ModelCircuit（新增一个字段），tensor_map_to_vec 只在
              **未被引用的空洞**上占位；被引用的 ID 落空即硬失败。
  [阻断5] 数值契约按真实域重写：
            * loader 的精确域是 [-2^31, i64::MAX-2^31]，不是 [-2^31, 2^31-1]。
              上界 v1 写错了 —— 那只是"塞得进 i32"。
            * i32 降级为**主动保守策略**（常量开关 + 单独错误文案），明说不是必要条件。
            * 删掉 k<=32 这个错误结论：它由 v1 的错域反推而来。改用
              k>=5（max_val=2^(k-1)-10>=0）、k<=62（-min_val*2=2^k 可表示）、
              显式 global_sf>0，以及 S / 2S 的 checked 派生。
            * sf^2*2^18 明确写成"保守上界"，并注明只有 sf 是 2 的幂时才等价 sf<=2^22。
            * Max 差值 lookup 的真实域由 InputLookupChip 的填表方式决定：
              表里是裸无符号 0..=2^k-10，所以差值域就是 [0, 2^k-10]。
  [撤回] v1 的 N5（把 smallest_tensor 改名 _smallest_tensor）撤回：v2 里它仍被使用。

脚本自带文本门禁：生成后重新读候选，逐项断言 v1 的坑没有复发。
注意：本批**不编译 Rust**，因此这些门禁是文本级/结构级的，不是类型检查。
"""
import difflib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "zkml-main/src/model.rs")
OUT = os.path.join(ROOT, "patches/model_numeric_contract_v2.rs")
DIFF = os.path.join(ROOT, "patches/model_numeric_contract_v2.diff")


def main():
    src = open(SRC, encoding="utf-8").read()
    out = src
    applied = []

    def sub(old, new, tag):
        nonlocal out
        assert out.count(old) == 1, "锚点不唯一或缺失: %s" % tag
        out = out.replace(old, new)
        applied.append(tag)

    # ---------------------------------------------------------------- M1 to_field
    sub("""    let to_field = |x: i64| {
      let bias = 1 << 31;
      let x_pos = x + bias;
      F::from(x_pos as u64) - F::from(bias as u64)
    };
""",
        """    // P4-A v2 契约（M1）：原编码 F::from((x + 2^31) as u64) - F::from(2^31) 的
    // **精确域**是 [-2^31, i64::MAX - 2^31]，不是 [-2^31, 2^31-1]。逐分支：
    //   * x >= -2^31：x + 2^31 >= 0，`as u64` 精确；F::from(u64) 精确
    //     （任何 u64 都 < 标量域模数）；相减得到 x 本身（负值即 p + x，正是它的域编码）。
    //   * x <  -2^31：x + 2^31 < 0，`as u64` 走二进制补码回绕，
    //     静默编码成 x + 2^64（一个巨大的正域元素），语义被改而不报错。
    //   * x >  i64::MAX - 2^31：x + 2^31 溢出 i64（debug panic / release 回绕）。
    // v1 把上界写成 2^31-1 是错的：那只是"能塞进 i32"，不是"能精确编码"。
    const FIELD_ENC_LO: i64 = -(1i64 << 31);
    const FIELD_ENC_HI: i64 = i64::MAX - (1i64 << 31);

    let to_field = |x: i64| {
      debug_assert!(
        x >= FIELD_ENC_LO && x <= FIELD_ENC_HI,
        "to_field 输入 {} 超出可精确编码区间 [{}, {}]",
        x, FIELD_ENC_LO, FIELD_ENC_HI
      );
      let bias: i64 = 1 << 31;
      let x_pos = x + bias;
      F::from(x_pos as u64) - F::from(bias as u64)
    };
""",
        "M1/M2 to_field exact domain")

    # ------------------------------------------------------- M2 逐元素硬边界 + 策略
    sub("""    for flat in config.tensors {
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
""",
        """    // P4-A v2（M3）：**主动保守策略**，不是编码的必要条件。
    // 下游 bias/累加/shift 的宿主整数算术尚未逐项复核，所以本项目把定点值进一步
    // 限制在 i32 内。原 loader 在 M1 的真实域上是精确的 —— 关掉本策略不会让编码
    // 本身出错，只是放弃一层尚未量化的保护。
    const I32_POLICY: bool = true;

    for flat in config.tensors {
      // P4-A v2（M2）：在编码之前拒绝越界值。
      // 硬边界用 M1 的真实精确域；i32 只作为可关闭的主动策略单独报错。
      for (elem_idx, v) in flat.data.iter().enumerate() {
        if *v < FIELD_ENC_LO || *v > FIELD_ENC_HI {
          panic!(
            "tensor {} 第 {} 个元素 = {} 超出原编码的精确域 [{}, {}]；\\
             拒绝加载。上游 converter 必须先做有限性/范围检查，\\
             不能把非有限值或超界整数写进 msgpack。",
            flat.idx, elem_idx, v, FIELD_ENC_LO, FIELD_ENC_HI
          );
        }
        if I32_POLICY && (*v < -(1i64 << 31) || *v > (1i64 << 31) - 1) {
          panic!(
            "tensor {} 第 {} 个元素 = {} 超出主动 i32 策略 [-2^31, 2^31-1]；\\
             这是本项目的保守选择（下游宿主算术未逐项复核），\\
             不是原编码的必要条件 —— 确认下游范围后可把 I32_POLICY 置 false。",
            flat.idx, elem_idx, v
          );
        }
      }
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
""",
        "M2 element hard bound + i32 policy")

    # ------------------------------------------- M4 稀疏 ID 加载期校验（阻断3）
    sub("""    // The input lookup is always used
    used_gadgets.insert(GadgetType::InputLookup);
""",
        """    // P4-A v2 契约（M4）：稀疏 tensor ID + 逐层可用性（加载期校验）。
    // 原实现（以及 v1 的 N4）要求 tensor_map 含 0..=max_key 的全部键，这有两个问题：
    //   1) "删掉元数据张量、保留原 ID"本来就会留下合法空洞，不是缺依赖；
    //   2) 尚未执行到的层的输出本来就不该作为初始张量存在。
    // 因此改成显式的稀疏表示：初始可用 = 初始张量 ∪ 模型输入，然后按层推进。
    let initial_ids: BTreeSet<i64> = config
      .tensors
      .iter()
      .map(|t| t.idx)
      .chain(config.inp_idxes.iter().copied())
      .collect();
    let mut available_ids: BTreeSet<i64> = initial_ids.clone();
    let mut produced_ids: BTreeSet<i64> = BTreeSet::new();
    for (layer_idx, layer) in config.layers.iter().enumerate() {
      for idx in layer.inp_idxes.iter() {
        if !available_ids.contains(idx) {
          panic!(
            "layer {} 的输入 tensor {} 既不在初始张量、也没有更早的层产出；\\
             若这是删元数据张量导致的，说明还有层的 inp_idxes 没同步",
            layer_idx, idx
          );
        }
      }
      for idx in layer.out_idxes.iter() {
        available_ids.insert(*idx);
        produced_ids.insert(*idx);
      }
    }
    let mut referenced_ids: BTreeSet<i64> = BTreeSet::new();
    referenced_ids.extend(config.inp_idxes.iter().copied());
    for layer in config.layers.iter() {
      referenced_ids.extend(layer.inp_idxes.iter().copied());
    }
    referenced_ids.extend(config.out_idxes.iter().copied());
    if let Some(groups) = config.commit_before.as_ref() {
      for group in groups.iter() {
        referenced_ids.extend(group.iter().copied());
      }
    }
    if let Some(groups) = config.commit_after.as_ref() {
      for group in groups.iter() {
        referenced_ids.extend(group.iter().copied());
      }
    }
    for idx in referenced_ids.iter() {
      if !initial_ids.contains(idx) && !produced_ids.contains(idx) {
        panic!(
          "tensor {} 被引用，但既不在初始张量、也没有任何层产出",
          idx
        );
      }
    }
    let referenced_tensor_ids = referenced_ids;

    // The input lookup is always used
    used_gadgets.insert(GadgetType::InputLookup);
""",
        "M4 sparse id load gate")

    # ----------------------------------------------- M5 k/sf 派生边界（阻断5）
    sub("""    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
""",
        """    // P4-A v2 契约（M5）：k 与 global_sf 的派生边界。
    // **不再声称 k <= 32**：那条是 v1 从错误的 loader 域（上界 2^31-1）反推来的，
    // 而 loader 的真实上界是 i64::MAX-2^31，推不出 32。
    // 实际可用的既有：
    //   * num_rows = 2^k - 9 > 0            -> k >= 4
    //   * max_val  = 2^(k-1) - 10 >= 0      -> k >= 5（k=4 时 max_val = -2）
    //   * -min_val * 2 = 2^k 可用 i64 表示  -> k <= 62（assign_constants 里会算）
    // k 的**实用**上界由资源（num_rows 与 lookup 表规模）决定，不由上述几条决定；
    // 本函数只做"不静默溢出"的必要检查，不给出可用 k 的结论。
    assert!(config.global_sf > 0, "global_sf={} 必须为正", config.global_sf);
    assert!(
      config.k >= 5,
      "k={} 非法：max_val = 2^(k-1) - 10 在 k<5 时为负",
      config.k
    );
    assert!(
      config.k <= 62,
      "k={} 非法：assign_constants 里的 -min_val * 2 = 2^k 会溢出 i64",
      config.k
    );
    {
      let sf = config.global_sf as i128;
      // S = sf^2 * 2^17；先做 checked，避免 i128 也溢出
      let s = sf.checked_mul(sf).and_then(|v| v.checked_mul(1i128 << 17));
      let two_s = s.and_then(|v| v.checked_mul(2i128));
      assert!(
        matches!(s, Some(v) if v <= i64::MAX as i128),
        "S = sf^2*2^17（sf={}）超出 i64，shift_min_val = -S 无法构造",
        config.global_sf
      );
      // 注意措辞：这是**保守**上界，不是原 shift 构造的精确边界。
      // 原构造只需要 S 可表示；2S 是给"把两个已加 S 的量再相加"的 gadget 留的余量。
      // 只有当 sf 是 2 的幂时，2S <= i64::MAX 才等价于 sf <= 2^22；一般 sf 的界是
      // sf <= 5931641（= floor(sqrt((2^63-1)/2^18))）。
      assert!(
        matches!(two_s, Some(v) if v <= i64::MAX as i128),
        "2S = sf^2*2^18（sf={}）超出 i64；需要 2S 的 gadget 会溢出。\\
         这是保守上界，不是原 shift 构造的精确边界；\\
         仅当 sf 为 2 的幂时它才等价于 sf <= 2^22（一般 sf 的界是 5931641）",
        config.global_sf
      );
    }
    let min_val_checked: i64 = -(1i64 << (config.k - 1));
    let max_val_checked: i64 = (1i64 << (config.k - 1)) - 10;
    let num_rows_checked: i64 = (1i64 << config.k) - 9;
    // 用与上面 assert 完全同一条 checked 链算出 shift_min_val，避免在结构体字面量里
    // 重新写一遍未检查的乘法（那正是原实现的溢出点）。
    let shift_min_val_checked: i64 = (config.global_sf as i128)
      .checked_mul(config.global_sf as i128)
      .and_then(|v| v.checked_mul(1i128 << 17))
      .and_then(|v| v.checked_neg())
      .expect("S 已在上面的 assert 中确认可用 i64 表示")
      as i64;
    assert!(
      max_val_checked >= 0 && num_rows_checked > 0,
      "k={} 派生出的 max_val={} / num_rows={} 非法",
      config.k, max_val_checked, num_rows_checked
    );

    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
""",
        "M5 k/sf derived bounds")

    # ------------------------------- M6 用校验过的派生值填 GadgetConfig（阻断5）
    sub("""      scale_factor: config.global_sf as u64,
      shift_min_val: -(config.global_sf * config.global_sf * (1 << 17)),
      div_outp_min_val: -(1 << (config.k - 1)),
      min_val: -(1 << (config.k - 1)),
      max_val: (1 << (config.k - 1)) - 10,
      k: config.k as usize,
      num_rows: (1 << config.k) - 10 + 1,
""",
        """      // P4-A v2：全部改用上面已经 checked 过的派生值，避免这里再做未检查的
      // i64 乘法（原式 sf*sf*(1<<17) 在 release 下溢出是静默的）。
      scale_factor: config.global_sf as u64,
      shift_min_val: shift_min_val_checked,
      div_outp_min_val: min_val_checked,
      min_val: min_val_checked,
      max_val: max_val_checked,
      k: config.k as usize,
      num_rows: num_rows_checked as usize,
""",
        "M6 use checked derivations")

    # ------------------------------------- M7 tensor_map_to_vec 只填空未引用空洞
    sub("""    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap_or(smallest_tensor);
      tensors.push(tensor.clone());
    }
""",
        """    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      match tensor_map.get(&i) {
        Some(tensor) => tensors.push(tensor.clone()),
        None => {
          // P4-A v2 契约（M7）：这个 Vec 是按 tensor ID 索引的，所以空洞必须占位以
          // 保持长度；但**被引用**的 ID 落空说明依赖清理漏改，必须硬失败，不能
          // 静默给一个 smallest_tensor（形状可能恰好对得上，从而掩盖错误）。
          // "未被引用"的判定来自 M4 在加载期算出的 referenced_tensor_ids。
          if self.referenced_tensor_ids.contains(&i) {
            panic!(
              "tensor {} 被引用但未被赋值（tensor_map 里没有这个键）；\\
               禁止用 smallest_tensor 静默填充",
              i
            );
          }
          tensors.push(smallest_tensor.clone());
        }
      }
    }
""",
        "M7 hole filling only for unreferenced")

    # ------------------------------------------------- M8 ModelCircuit 新增字段
    sub("""  pub inp_idxes: Vec<i64>,
  pub num_random: i64,
}
""",
        """  pub inp_idxes: Vec<i64>,
  pub num_random: i64,
  /// P4-A v2：从 msgpack 算出的"被引用 tensor ID"集合，供 tensor_map_to_vec
  /// 判定空洞是否合法。只参与加载期检查，不参与布局，也不改变电路语义。
  pub referenced_tensor_ids: BTreeSet<i64>,
}
""",
        "M8 struct field")

    sub("""      inp_idxes: config.inp_idxes.clone(),
      commit_after: config.commit_after.unwrap_or(vec![]),
""",
        """      inp_idxes: config.inp_idxes.clone(),
      referenced_tensor_ids,
      commit_after: config.commit_after.unwrap_or(vec![]),
""",
        "M9 struct literal")

    open(OUT, "w", encoding="utf-8", newline="\n").write(out)
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="zkml-main/src/model.rs", tofile="patches/model_numeric_contract_v2.rs", n=3)
    open(DIFF, "w", encoding="utf-8", newline="\n").writelines(diff)

    failures = verify(OUT)
    print("已应用 %d 处编辑：" % len(applied))
    for a in applied:
        print("  -", a)
    print()
    print("文本门禁：%s" % ("全部通过" if not failures else "失败 %d 项" % len(failures)))
    for f in failures:
        print("  !", f)
    print()
    print("fixed ->", OUT, os.path.getsize(OUT), "bytes")
    print("diff  ->", DIFF, os.path.getsize(DIFF), "bytes")
    print("注意：本批不编译 Rust，以上是文本级检查，不是类型检查。")
    return 1 if failures else 0


def code_only(src):
    """去掉注释与字符串/字符字面量，只留代码。

    括号配平必须只看代码：注释里写 "1)" "2)"、错误信息里写 "[{}, {}]" 都会
    污染朴素计数（v2 第一次生成时就因此误报了 2 个多余的右括号）。
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == '"':
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "'":
            # 字符字面量/生命周期；只跳过形如 'x' 或 '\n' 的短片段
            if i + 2 < n and (src[i + 2] == "'" or (src[i + 1] == "\\" and i + 3 < n)):
                j = i + 1
                while j < n and src[j] != "'":
                    j += 1
                i = j + 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def verify(path):
    src = open(path, encoding="utf-8").read()
    code = code_only(src)
    fails = []

    # 阻断5：不得再声称 k<=32（只看代码，注释里"不再声称 k<=32"是说明文字）
    if re.search(r"k\s*<=\s*32", code):
        fails.append("[阻断5] 代码里仍出现 k <= 32（v1 的错误结论）")
    # 阻断5：真实上界必须出现在代码里
    if "i64::MAX - (1i64 << 31)" not in code:
        fails.append("[阻断5] 代码里没有真实编码上界 i64::MAX - 2^31")
    # 阻断5：i32 必须被明确标成策略（这里是看注释/文案，故用全文）
    if "I32_POLICY" not in code:
        fails.append("[阻断5] 没有 I32_POLICY 开关，i32 限制会被误当成必要条件")
    if "主动" not in src or "不是原编码的必要条件" not in src:
        fails.append("[阻断5] i32 策略缺少'非必要条件'的说明")
    # 阻断5：S / 2S 分开 checked
    if "checked_mul(sf)" not in code:
        fails.append("[阻断5] 缺少 checked_mul(sf)（S 的 checked 推导）")
    if "1i128 << 17" not in code:
        fails.append("[阻断5] 缺少 1i128 << 17")
    if "5931641" not in src:
        fails.append("[阻断5] 缺少一般 sf 上界 5931641 的说明")

    # 阻断3：禁止静默填洞；禁止全 ID 连续要求
    if "unwrap_or(smallest_tensor)" in code:
        fails.append("[阻断3] 仍有 unwrap_or(smallest_tensor) 静默填洞")
    if "0..max_tensor_key + 1" in code and "referenced_tensor_ids.contains" not in code:
        fails.append("[阻断3] 空洞循环没有引用集合判定")
    if re.search(r"missing\s*:\s*Vec<i64>", code):
        fails.append("[阻断3] 仍保留'缺键列表'式的全 ID 连续检查")

    # 阻断3：稀疏表示与逐层可用性
    for token in ("initial_ids", "available_ids", "produced_ids", "referenced_ids"):
        if token not in code:
            fails.append("[阻断3] 缺少 %s（稀疏 ID / 逐层可用性校验）" % token)

    # v1 的 N5 必须撤回
    if "_smallest_tensor" in code:
        fails.append("[撤回] 仍有 _smallest_tensor 改名（v2 里它仍被使用）")

    # 结构完整性：新增字段必须在 struct / 字面量 / 使用点都出现
    n_field = code.count("referenced_tensor_ids")
    if n_field < 4:
        fails.append("referenced_tensor_ids 出现 %d 次，少于 struct+字面量+使用点所需的 4 次"
                     % n_field)
    if "BTreeSet" not in code:
        fails.append("缺少 BTreeSet 引用")

    # 括号平衡（只看代码，排除注释与字符串）
    for op_, cl_ in (("{", "}"), ("(", ")"), ("[", "]")):
        if code.count(op_) != code.count(cl_):
            fails.append("代码括号不平衡: %s=%d %s=%d"
                         % (op_, code.count(op_), cl_, code.count(cl_)))
    return fails


if __name__ == "__main__":
    sys.exit(main())
