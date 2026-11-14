#!/usr/bin/env python3
"""P4-A：生成 model.rs 的数值边界契约补丁（不改仓库原文件）。

生成 patches/model_numeric_contract.rs 与 .diff。

三处最小改动：
  N1 to_field 之前逐元素检查可精确编码区间 [-2^31, 2^31-1]，
     越界时报出 tensor id 与下标并拒绝加载（不静默编码成 x+2^64）
  N2 to_field 内部加 debug_assert，把契约写在调用点上
  N3 k 与 scale_factor 的派生边界：k<=32；sf^2*2^17*2 不溢出 i64
"""
import difflib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "zkml-main/src/model.rs")
OUT = os.path.join(ROOT, "patches/model_numeric_contract.rs")
DIFF = os.path.join(ROOT, "patches/model_numeric_contract.diff")


def main():
    src = open(SRC, encoding="utf-8").read()
    out = src
    applied = []

    def sub(old, new, tag):
        nonlocal out
        assert out.count(old) == 1, "锚点不唯一或缺失: %s" % tag
        out = out.replace(old, new)
        applied.append(tag)

    # N1 + N2
    sub("""    let to_field = |x: i64| {
      let bias = 1 << 31;
      let x_pos = x + bias;
      F::from(x_pos as u64) - F::from(bias as u64)
    };
""",
        """    // P4-A 契约（N1/N2）：当前编码只能精确表示 [-2^31, 2^31-1]。
    //   对 x >= -2^31：x + 2^31 落在 u64 的低 33 位，F::from 精确，减回去就是 x。
    //   对 x <  -2^31：x + 2^31 为负，(x + 2^31) as u64 发生二进制补码回绕，
    //                 实际编码成 x + 2^64（一个巨大的正域元素），静默改变语义。
    //   对 x >  2^31-1：x + 2^31 可能溢出 i64（debug 下 panic，release 下回绕）。
    // 因此把区间检查放在唯一入口，越界即拒绝，并报出 tensor id 与下标。
    const FIELD_ENC_LO: i64 = -(1i64 << 31);
    const FIELD_ENC_HI: i64 = (1i64 << 31) - 1;

    let to_field = |x: i64| {
      debug_assert!(
        x >= FIELD_ENC_LO && x <= FIELD_ENC_HI,
        "to_field 输入 {} 超出可精确编码区间 [{}, {}]",
        x, FIELD_ENC_LO, FIELD_ENC_HI
      );
      let bias = FIELD_ENC_HI + 1; // = 1 << 31
      let x_pos = x + bias;
      F::from(x_pos as u64) - F::from(bias as u64)
    };
""",
        "N1/N2 checked to_field")

    # N1 逐元素前置检查
    sub("""    for flat in config.tensors {
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
""",
        """    for flat in config.tensors {
      // P4-A：在编码之前拒绝越界值（原实现会把 -inf 之类的 INT64_MIN 静默编码）
      for (elem_idx, v) in flat.data.iter().enumerate() {
        if *v < FIELD_ENC_LO || *v > FIELD_ENC_HI {
          panic!(
            "tensor {} 第 {} 个元素 = {} 超出可精确编码区间 [{}, {}]；\\
             拒绝加载。上游 converter 必须先做有限性/范围检查，\\
             不能把非有限值或超界整数写进 msgpack。",
            flat.idx, elem_idx, v, FIELD_ENC_LO, FIELD_ENC_HI
          );
        }
      }
      let value_flat = flat.data.iter().map(|x| to_field(*x)).collect::<Vec<_>>();
""",
        "N1 per-element range check")

    # N3 k / scale 派生边界
    sub("""    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
""",
        """    // P4-A 契约（N3）：k 同时决定 min_val/max_val/lookup 范围/packing 位宽。
    //   min_val = -(2^(k-1)) 必须落在 FIELD_ENC 区间内 -> k <= 32。
    //   shift_min_val = -(sf^2 * 2^17) 必须不溢出 i64 -> sf^2 * 2^18 < 2^63。
    assert!(
      config.k <= 32,
      "k={} 会使 min_val=-(2^(k-1)) 小于 -2^31，超出可精确编码区间",
      config.k
    );
    assert!(config.k >= 1, "k={} 非法", config.k);
    {
      let sf = config.global_sf as i128;
      let need = sf
        .checked_mul(sf)
        .and_then(|v| v.checked_mul(1 << 18))
        .filter(|v| *v <= i64::MAX as i128);
      assert!(
        need.is_some(),
        "scale_factor={} 会让 shift_min_val = -(sf^2 * 2^17) 溢出 i64",
        config.global_sf
      );
    }

    let gadget = &GADGET_CONFIG;
    let cloned_gadget = gadget.lock().unwrap().clone();
""",
        "N3 k/scale derived bounds")

    # N4 禁止 smallest_tensor 静默填洞
    sub("""    let max_tensor_key = tensor_map
      .iter()
      .max_by_key(|(key, _)| *key)
      .unwrap()
      .0
      .clone();
    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap_or(smallest_tensor);
      tensors.push(tensor.clone());
    }
""",
        """    let max_tensor_key = tensor_map
      .iter()
      .max_by_key(|(key, _)| *key)
      .unwrap()
      .0
      .clone();
    // P4-A 契约（N4）：删除元数据张量后索引会变稀疏，而原实现在这里用
    // smallest_tensor 填空洞——任何漏改的 inp_idxes 都会静默拿到一个错误张量，
    // 形状恰好也可能对得上，从而掩盖依赖清理的缺陷。改为硬失败。
    // 注意：本改动会让「索引本来就不连续」的旧 msgpack 直接失败，
    // 因此 P4-B 必须先用 MNIST 既有产物确认其索引连续，再启用。
    let missing: Vec<i64> = (0..max_tensor_key + 1)
      .filter(|i| !tensor_map.contains_key(i))
      .collect();
    if !missing.is_empty() {
      panic!(
        "tensor 索引不连续，缺少 {:?}（0..={}）；禁止用 smallest_tensor 填充。\
         若这是删除元数据张量导致的，说明还有层的 inp_idxes 没同步。",
        missing, max_tensor_key
      );
    }
    let mut tensors = vec![];
    for i in 0..max_tensor_key + 1 {
      let tensor = tensor_map.get(&i).unwrap();
      tensors.push(tensor.clone());
    }
""",
        "N4 no silent hole filling")

    # N5 N4 之后 smallest_tensor 不再被使用，改名避免 unused 警告（deny(warnings) 会变错误）
    sub("""    let smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
""",
        """    // P4-A：N4 之后不再需要 smallest_tensor；保留绑定是为了让 diff 最小，
    // 但改名成 _ 前缀避免 unused_variables 警告（若 crate 开了 deny(warnings) 会变错误）。
    let _smallest_tensor = tensor_map
      .iter()
      .min_by_key(|(_, tensor)| tensor.len())
      .unwrap()
      .1;
""",
        "N5 rename unused binding")

    open(OUT, "w", encoding="utf-8").write(out)
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="zkml-main/src/model.rs", tofile="patches/model_numeric_contract.rs", n=3)
    open(DIFF, "w", encoding="utf-8").writelines(diff)

    print("已应用 %d 处编辑：" % len(applied))
    for a in applied:
        print("  -", a)
    print()
    print("fixed ->", OUT, os.path.getsize(OUT), "bytes")
    print("diff  ->", DIFF, os.path.getsize(DIFF), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
