#!/usr/bin/env python3
"""P4-A v5：修 V4-G1（ConfigOnly 与 commit 不兼容）。

问题（用户复核指出）：
  ConfigOnly 下 config.tensors 被清空，initial_ids 只剩模型输入；
  而 commit_before 仍要求 ID ∈ initial_ids，commit_after 也要求
  ID ∈ initial_ids ∪ producer。于是完整模型里合法地在 DAG 前承诺
  一个权重（或 after 承诺一个初始常量）时，瘦配置仍被拒绝。

修法（不是"警告后放行"）：
  * 把"阶段合法性"与"存在性"分开判：
      - 阶段：before 绝不能引用任何层产出 —— 两种模式都硬拒绝；
      - 存在性：FullWitness 严格核 initial/producer；
                ConfigOnly 只能证明结构，没被证明的 ID 记进
                unresolved_commit_ids，**不被描述成"依赖已验证"**。
  * before 与 after 都改，不只补 before。
  * 不给 verifier 加回权重，也不静默删除承诺组。

输入 patches/model_numeric_contract_v4.rs -> 输出 _v5.rs
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "patches", "model_numeric_contract_v4.rs")
DST = os.path.join(ROOT, "patches", "model_numeric_contract_v5.rs")


FIELD_OLD = """  pub required_initial_ids: BTreeSet<i64>,
}
"""

FIELD_NEW = """  pub required_initial_ids: BTreeSet<i64>,
  /// P4-A v5：仅在 ConfigOnly 下可能非空 —— 出现在 commit_before/after 里、
  /// 既不在 initial 也没有层产出、因而**无法在瘦配置中核实**的 ID。
  /// 记录它们是为了不把"未解析"说成"依赖已验证"；FullWitness 下恒为空。
  pub unresolved_commit_ids: BTreeSet<i64>,
}
"""

BLOCK_START = "    // 结构检查 3：commit_before 在 DAG 前读取，只能绑定初始张量；"
BLOCK_END = "    // 只有\"被引用且没有任何层产出\"的 ID 才必须在初始 map 里。"

BLOCK_NEW = """    // 结构检查 3（v5）：把 commit 的"阶段合法性"与"存在性"分开判。
    //
    // 存在性在两种模式下可证的程度不同：
    //   FullWitness：initial 就是真实存在的张量，可以严格核对；
    //   ConfigOnly ：tensors 被清空，初始权重/常量根本没有实体，
    //                无法证明"存在于 initial"，只能证明"不是层产出"。
    // 所以 ConfigOnly 下不把未解析的 ID 当错误，而是记进 unresolved_commit_ids：
    // 它们**没有被验证过**，只是被允许通过。不能把它描述成"依赖已验证"。
    let mut unresolved_commit_ids: BTreeSet<i64> = BTreeSet::new();

    // before 在 DAG 之前求值：任何层产出都不可能被看到 —— 两种模式都硬拒绝。
    if let Some(groups) = config.commit_before.as_ref() {
      for group in groups.iter() {
        for idx in group.iter() {
          if producer.contains_key(idx) {
            panic!(
              "commit_before 引用了 tensor {}，它由某一层产出；\\
               before 在 DAG 之前求值，不可能看到层输出",
              idx
            );
          }
          match load_mode {
            LoadMode::FullWitness => {
              if !initial_ids.contains(idx) {
                panic!(
                  "commit_before 引用了 tensor {}，它既不是初始张量也不是模型输入",
                  idx
                );
              }
            }
            LoadMode::ConfigOnly => {
              if !initial_ids.contains(idx) {
                // 可能来自被清空的初始张量；瘦配置里无法核实，只记录。
                unresolved_commit_ids.insert(*idx);
              }
            }
          }
        }
      }
    }

    // after 在 DAG 之后求值，可引用层产出，也可引用初始张量。
    // 注意"after 直接承诺初始常量"这一情形：瘦配置里该常量同样没有实体，
    // 所以不能只修 before 就宣布兼容闭合 —— 这里同样记入 unresolved。
    if let Some(groups) = config.commit_after.as_ref() {
      for group in groups.iter() {
        for idx in group.iter() {
          match load_mode {
            LoadMode::FullWitness => {
              if !initial_ids.contains(idx) && !producer.contains_key(idx) {
                panic!(
                  "commit_after 引用了 tensor {}，它既不是初始张量、也没有任何层产出",
                  idx
                );
              }
            }
            LoadMode::ConfigOnly => {
              if !initial_ids.contains(idx) && !producer.contains_key(idx) {
                unresolved_commit_ids.insert(*idx);
              }
            }
          }
        }
      }
    }

"""

CTOR_OLD = "      required_initial_ids,\n"
CTOR_NEW = "      required_initial_ids,\n      unresolved_commit_ids,\n"


def code_only(text):
    out, i, n, in_str = [], 0, len(text), False
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

    if src.count(FIELD_OLD) != 1:
        print("FAIL anchor FIELD count=%d" % src.count(FIELD_OLD))
        return 1
    if src.count(CTOR_OLD) != 1:
        print("FAIL anchor CTOR count=%d" % src.count(CTOR_OLD))
        return 1

    out = src.replace(FIELD_OLD, FIELD_NEW, 1)
    out = out.replace(CTOR_OLD, CTOR_NEW, 1)

    i = out.find(BLOCK_START)
    j = out.find(BLOCK_END)
    if i < 0 or j < 0 or j <= i:
        print("FAIL block markers i=%d j=%d" % (i, j))
        return 1
    out = out[:i] + BLOCK_NEW + out[j:]

    io.open(DST, "w", encoding="utf-8", newline="\n").write(out)

    checks = [
        ("unresolved 字段", "pub unresolved_commit_ids: BTreeSet<i64>," in out),
        ("字段已初始化", "let mut unresolved_commit_ids: BTreeSet<i64> = BTreeSet::new();" in out),
        ("构造点已接字段", CTOR_NEW in out),
        ("before 阶段检查", "before 在 DAG 之前求值，不可能看到层输出" in out),
        ("before FullWitness 分支", "commit_before 引用了 tensor {}，它既不是初始张量也不是模型输入" in out),
        ("after FullWitness 分支", "commit_after 引用了 tensor {}，它既不是初始张量、也没有任何层产出" in out),
        ("ConfigOnly 分支存在 2 处", out.count("LoadMode::ConfigOnly => {") == 2),
        ("after 记入 unresolved", out.count("unresolved_commit_ids.insert(*idx);") == 2),
        ("旧文案已移除", "commit_before 只能绑定初始张量或模型输入" not in out),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print("  %-26s %s" % (n, "OK" if ok else "FAIL"))

    code = code_only(out)
    bal = [code.count("(") - code.count(")"), code.count("{") - code.count("}"),
           code.count("[") - code.count("]")]
    print("  括号配平 ( / { / [ : %s" % bal)
    if any(bal):
        bad.append("括号不配平")

    if bad:
        print("GATE FAIL: %s" % ", ".join(bad))
        return 1
    print("GATE OK  -> %s (%d bytes)" % (os.path.basename(DST), len(out.encode())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
