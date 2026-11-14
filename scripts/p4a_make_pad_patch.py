#!/usr/bin/env python3
"""P4-A：生成 pad.rs 的最小契约补丁（不改仓库原文件）。

P1 恢复 Pad 只接受 1 个数据输入的断言（元数据已由 converter 契约移除）；
P2 把「填充值恒为 0」写成显式契约，避免后来者误以为它跟随 TFLite 常量。
"""
import difflib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "zkml-main/src/layers/shape/pad.rs")
OUT = os.path.join(ROOT, "patches/pad_contract.rs")
DIFF = os.path.join(ROOT, "patches/pad_contract.diff")


def main():
    src = open(SRC, encoding="utf-8").read()
    out = src
    applied = []

    def sub(old, new, tag):
        nonlocal out
        assert out.count(old) == 1, "锚点不唯一或缺失: %s" % tag
        out = out.replace(old, new)
        applied.append(tag)

    sub("""    // FIXME: the pad from tflite is actually two, but mine is one
    // assert_eq!(tensors.len(), 1);
    let input = &tensors[0];

    let zero = constants.get(&0).unwrap().clone();
""",
        """    // P4-A 契约（P1）：PAD/PADV2 的 paddings 与常量已经编进 layer_params，
    // converter 契约把它们从运行时 inp_idxes 移除，因此这里必须只有 1 个数据输入。
    // 恢复断言可以在依赖清理漏改时立即失败，而不是默默用 tensors[0]。
    assert_eq!(
      tensors.len(),
      1,
      "PadChip 只接受 1 个数据输入；paddings/常量应在 layer_params 中，\\
       若这里 >1 说明 converter 没有清理元数据依赖"
    );
    let input = &tensors[0];

    // P4-A 契约（P2）：填充值**恒为 0**，不随 TFLite 常量变化。
    // 因此 PADV2 的 -inf 常量只能通过 converter 的图级保护降级：
    // 仅当被填充张量逐元素非负（生产者带 ReLU）且 Pad 输出唯一被 VALID MaxPool
    // 消费时，0 填充才与 -inf 填充等价。这里不做图检查，检查在 converter 侧。
    let zero = constants.get(&0).unwrap().clone();
""",
        "P1/P2 pad contract")

    open(OUT, "w", encoding="utf-8").write(out)
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="zkml-main/src/layers/shape/pad.rs", tofile="patches/pad_contract.rs", n=3)
    open(DIFF, "w", encoding="utf-8").writelines(diff)
    print("已应用：", applied)
    print("fixed ->", OUT, os.path.getsize(OUT))
    print("diff  ->", DIFF, os.path.getsize(DIFF))
    return 0


if __name__ == "__main__":
    sys.exit(main())
