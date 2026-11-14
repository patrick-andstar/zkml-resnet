#!/usr/bin/env python3
"""P4-A v4：model 侧定向补正生成器。

针对 plan.md §2.9.2 的第 2/3/4 条（生命周期错位、config-only 回退、commit 判据）：
  * 把 v2/v3 混用的 `referenced_tensor_ids` 拆成三段生命周期：
      initial        = 初始张量 ∪ 模型输入
      produced       = 各层声明产出
      required_initial = 被引用 − produced      ← 只有它必须在初始 map 里
  * 引入 LoadMode：tensors 非空 = FullWitness（逐层核对可用性）；
    tensors 为空 = ConfigOnly（权重/输入不在请求里，只做结构检查）。
  * commit_before 只能引用初始张量；commit_after 可引用任意层产出。
  * tensor_map_to_vec 对空表直接返回空 Vec，不再 unwrap panic。

输入 patches/model_numeric_contract_v3.rs → 输出 patches/model_numeric_contract_v4.rs
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "patches", "model_numeric_contract_v3.rs")
DST = os.path.join(ROOT, "patches", "model_numeric_contract_v4.rs")


FIELD_OLD = '''  /// P4-A v2：从 msgpack 算出的"被引用 tensor ID"集合，供 tensor_map_to_vec
  /// 判定空洞是否合法。只参与加载期检查，不参与布局，也不改变电路语义。
  pub referenced_tensor_ids: BTreeSet<i64>,
'''

FIELD_NEW = '''  /// P4-A v4：初始 map 中**必须已有**的 tensor ID 集合，供 tensor_map_to_vec
  /// 判定空洞是否合法。等于「被引用 − 任意层产出」，因此未来层的输出不会
  /// 出现在这里，稀疏空洞合法。config-only 模式下为空集。
  /// 只参与加载期检查，不参与布局，也不改变电路语义。
  pub required_initial_ids: BTreeSet<i64>,
'''

EMPTY_OLD = '''    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
'''

EMPTY_NEW = '''    // P4-A v4：config-only 验证请求里 converter 会清空 config.tensors，
    // 落到这里就是空表。此时既没有 smallest_tensor 可退化、也没有真实张量可赋值，
    // 直接返回空 Vec，由调用方按"无 witness"处理。
    // 这是 v2 破坏瘦配置入口的直接原因之一：空表上 .unwrap() 必 panic。
    if tensor_map.is_empty() {
      return Ok(vec![]);
    }
    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
'''

HOLE_OLD = '''          // P4-A v2 契约（M7）：这个 Vec 是按 tensor ID 索引的，所以空洞必须占位以
          // 保持长度；但**被引用**的 ID 落空说明依赖清理漏改，必须硬失败，不能
          // 静默给一个 smallest_tensor（形状可能恰好对得上，从而掩盖错误）。
          // "未被引用"的判定来自 M4 在加载期算出的 referenced_tensor_ids。
          if self.referenced_tensor_ids.contains(&i) {
'''

HOLE_NEW = '''          // P4-A v4 契约（M7）：这个 Vec 是按 tensor ID 索引的，所以空洞必须占位以
          // 保持长度；但**初始必需**的 ID 落空说明依赖清理漏改，必须硬失败，不能
          // 静默给一个 smallest_tensor（形状可能恰好对得上，从而掩盖错误）。
          // "初始必需"= 被引用 − 任意层产出（M4）。未来层输出的空洞不在此列，
          // 它们本来就是合法空槽，会被执行到的层填上。
          if self.required_initial_ids.contains(&i) {
'''

CTOR_OLD = "      referenced_tensor_ids,\n"
CTOR_NEW = "      required_initial_ids,\n"

BLOCK_START = "    // P4-A v2 契约（M4）"
BLOCK_END = "    let referenced_tensor_ids = referenced_ids;\n"

BLOCK_NEW = '''    // P4-A v4 契约（M4）：三段生命周期 + 两种加载模式。
    //
    // v2/v3 把这些集合混成一个 `referenced_tensor_ids`，造成两个后果：
    //   1) 未来层的输出被要求"在 DAG 前已赋值"——初始真实 ID {0,2}、
    //      L0 输入 0 输出 1 的合法图，会在转 Vec 的空洞 1 处提前 panic；
    //   2) config-only 验证（converter 清空 tensors）被当成"权重缺失"拒绝。
    //
    // 本版明确三段：
    //   initial          = 初始张量 ∪ 模型输入        （DAG 前真实存在）
    //   produced         = 各层声明产出              （按层推进，含尚未执行的）
    //   required_initial = 被引用 − produced         （只有它必须在初始 map 里）
    // 模式：
    //   FullWitness = config.tensors 非空：再额外逐层核对"输入此刻可用"
    //   ConfigOnly  = config.tensors 为空：权重/输入都不在请求里，只做结构检查
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum LoadMode {
      FullWitness,
      ConfigOnly,
    }
    let load_mode = if config.tensors.is_empty() {
      LoadMode::ConfigOnly
    } else {
      LoadMode::FullWitness
    };

    let initial_ids: BTreeSet<i64> = config
      .tensors
      .iter()
      .map(|t| t.idx)
      .chain(config.inp_idxes.iter().copied())
      .collect();

    // producer 表：同一 ID 被两层产出是结构性错误，两种模式都可判。
    let mut producer: BTreeMap<i64, usize> = BTreeMap::new();
    for (layer_idx, layer) in config.layers.iter().enumerate() {
      for idx in layer.out_idxes.iter() {
        if let Some(prev) = producer.insert(*idx, layer_idx) {
          panic!(
            "tensor {} 被 layer {} 和 layer {} 重复产出；DAG 语义要求单一定义",
            idx, prev, layer_idx
          );
        }
      }
    }

    // 结构检查 1：不许引用"自身或更晚的层"才产出的 ID。
    // 这条只依赖拓扑，与权重是否在请求里无关，两种模式都执行。
    for (layer_idx, layer) in config.layers.iter().enumerate() {
      for idx in layer.inp_idxes.iter() {
        if let Some(p) = producer.get(idx) {
          if *p >= layer_idx {
            panic!(
              "layer {} 的输入 tensor {} 由 layer {} 产出；\\
               不允许引用自身或未来层的输出",
              layer_idx, idx, p
            );
          }
        }
      }
    }

    // 数据检查（仅 FullWitness）：逐层核对输入此刻确实可用。
    // config-only 下权重与输入本来就缺席，不能拿它当"缺依赖"。
    if load_mode == LoadMode::FullWitness {
      let mut available_ids: BTreeSet<i64> = initial_ids.clone();
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
        }
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

    // 结构检查 2：最终输出必须是模型输入或某层产出。
    for idx in config.out_idxes.iter() {
      if !initial_ids.contains(idx) && !producer.contains_key(idx) {
        panic!(
          "最终输出 tensor {} 既不是模型输入、也没有任何层产出",
          idx
        );
      }
    }

    // 结构检查 3：commit_before 在 DAG 前读取，只能绑定初始张量；
    //             commit_after 在 DAG 后读取，可引用任意层产出。
    if let Some(groups) = config.commit_before.as_ref() {
      for group in groups.iter() {
        for idx in group.iter() {
          if !initial_ids.contains(idx) {
            panic!(
              "commit_before 引用了 tensor {}，它在 DAG 前并不存在；\\
               commit_before 只能绑定初始张量或模型输入",
              idx
            );
          }
        }
      }
    }
    if let Some(groups) = config.commit_after.as_ref() {
      for group in groups.iter() {
        for idx in group.iter() {
          if !initial_ids.contains(idx) && !producer.contains_key(idx) {
            panic!(
              "commit_after 引用了 tensor {}，它既不是初始张量、也没有任何层产出",
              idx
            );
          }
        }
      }
    }

    // 只有"被引用且没有任何层产出"的 ID 才必须在初始 map 里。
    // 未来层的输出不在其中，所以稀疏空洞合法。
    // config-only：权重与输入都不在请求里，绝不要求它们出现在初始 map。
    let required_initial_ids: BTreeSet<i64> = if load_mode == LoadMode::FullWitness {
      referenced_ids
        .iter()
        .copied()
        .filter(|idx| !producer.contains_key(idx))
        .collect()
    } else {
      BTreeSet::new()
    };
'''


def code_only(text):
    """剥离 // 注释与字符串字面量，避免注释里的括号干扰配平检查。"""
    out = []
    i = 0
    n = len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    src = io.open(SRC, encoding="utf-8").read()

    for name, old in [
        ("FIELD", FIELD_OLD),
        ("EMPTY", EMPTY_OLD),
        ("HOLE", HOLE_OLD),
        ("CTOR", CTOR_OLD),
    ]:
        cnt = src.count(old)
        if cnt != 1:
            print("FAIL anchor %s count=%d" % (name, cnt))
            return 1

    out = src
    out = out.replace(FIELD_OLD, FIELD_NEW, 1)
    out = out.replace(EMPTY_OLD, EMPTY_NEW, 1)
    out = out.replace(HOLE_OLD, HOLE_NEW, 1)
    out = out.replace(CTOR_OLD, CTOR_NEW, 1)

    # 生命周期整块替换：按起止标记切片，避免逐字比对 50 行
    i = out.find(BLOCK_START)
    j = out.find(BLOCK_END)
    if i < 0 or j < 0 or j < i:
        print("FAIL block markers i=%d j=%d" % (i, j))
        return 1
    out = out[:i] + BLOCK_NEW + out[j + len(BLOCK_END):]

    io.open(DST, "w", encoding="utf-8", newline="\n").write(out)

    # ---- 门禁：结构断言 ----
    checks = [
        ("LoadMode 枚举存在", "enum LoadMode {" in out),
        ("ConfigOnly 分支", "LoadMode::ConfigOnly" in out),
        ("required_initial_ids 字段", "pub required_initial_ids: BTreeSet<i64>," in out),
        ("旧字段已消失", ("pub referenced_tensor_ids" not in out)
                            and ("self.referenced_tensor_ids" not in out)),
        ("空表早返回", "if tensor_map.is_empty() {" in out),
        ("producer 表", "let mut producer: BTreeMap<i64, usize>" in out),
        ("未来引用检查", "不允许引用自身或未来层的输出" in out),
        ("commit_before 限制", "commit_before 只能绑定初始张量或模型输入" in out),
        ("commit_after 限制", "commit_after 引用了 tensor" in out),
        ("required_initial 过滤", ".filter(|idx| !producer.contains_key(idx))" in out),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print("  %-24s %s" % (n, "OK" if ok else "FAIL"))

    code = code_only(out)
    bal = [code.count("(") - code.count(")"), code.count("{") - code.count("}"),
           code.count("[") - code.count("]")]
    print("  括号配平 ( / { / [ : %s" % bal)
    if any(bal):
        bad.append("括号不配平")

    if bad:
        print("GATE FAIL: %s" % ", ".join(bad))
        return 1
    print("GATE OK  %s -> %s (%d bytes)" % (os.path.basename(SRC), os.path.basename(DST), len(out.encode())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
