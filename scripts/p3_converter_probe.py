#!/usr/bin/env python3
"""P3 诊断探针：仓库 converter.py 对 PADV2 无分支，会停在 layer 2。

本脚本不改仓库文件，而是：
  1) 读源码，在 `PAD` 分支之前插入一个 PADV2 分支，生成 patches/converter_p3_padv2.py；
  2) 打印插入的 diff 片段供复核；
  3) 用 ai-edge-litert 兼容层跑一遍，记录退出码与日志；
  4) 若成功，解析 msgpack 的层表，与 TFLite 侧逐层对形。

语义前提（必须写进报告）：Rust 的 PadChip 把填充值固定为常量 0
（src/layers/shape/pad.rs: `constants.get(&0)`），而本模型的 PADV2 常量是 -inf。
仅当被填充张量已由 ReLU 保证非负时，0 与 -inf 等价（max(real ∪ {0}) = max(real)）。
本模型的 PADV2 输入是 conv1+BN+ReLU 的输出，满足该前提，可由探针现场校验。
"""
import json
import os
import subprocess
import sys

REPRO = os.environ.get("REPRO", "/root/autodl-tmp/zkml-repro")
PY = os.path.join(REPRO, "env/py-export/bin/python")
SHIM = os.path.join(REPRO, "env/shim")
SRC = os.path.join(REPRO, "src/python/converter.py")
MODEL = os.path.join(REPRO, "assets/p3-model.tflite")
PATCHED = os.path.join(REPRO, "patches/converter_p3_padv2.py")
LOGDIR = os.path.join(REPRO, "logs/p3")
OUT_MODEL = os.path.join(LOGDIR, "patched-model.msgpack")
OUT_CFG = os.path.join(LOGDIR, "patched-config.msgpack")

ANCHOR = "      # Pad\n      elif op_code == tflite.BuiltinOperator.PAD:"
INSERT = '''      # Pad
      # --- P3 最小补丁：补 PADV2（= PAD + 常量） ---
      elif op_code == tflite.BuiltinOperator.PADV2:
        layer_type = 'Pad'
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().astype(np.int64)
        params = tensor.tolist()
        _c = float(interpreter.get_tensor(op.Inputs(2)).flatten()[0])
        print('PADV2 at op%d constant=%r paddings=%s' % (op_idx, _c, params))
      # --- 补丁结束 ---
      elif op_code == tflite.BuiltinOperator.PAD:'''


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(os.path.dirname(PATCHED), exist_ok=True)
    src = open(SRC).read()
    if "PADV2" in src:
        print("仓库源码已含 PADV2 分支，无需补丁")
        patched = src
    else:
        assert src.count(ANCHOR) == 1, "锚点不唯一，停止（不盲改）"
        patched = src.replace(ANCHOR, INSERT)
        print("=== 插入的 diff 片段 ===")
        for line in INSERT.splitlines():
            print("+" + line)
        print()
    open(PATCHED, "w").write(patched)
    print("patched ->", PATCHED, len(patched), "bytes")

    env = dict(os.environ, PYTHONPATH=SHIM)
    cmd = [PY, PATCHED, "--model", MODEL,
           "--model_output", OUT_MODEL, "--config_output", OUT_CFG]
    print("=== 运行 ===")
    print(" ".join(cmd))
    with open(os.path.join(LOGDIR, "converter-patched.log"), "w") as lf:
        p = subprocess.run(cmd, cwd=os.path.join(REPRO, "src"), env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    print("exit =", p.returncode)
    tail = open(os.path.join(LOGDIR, "converter-patched.log")).read().splitlines()
    print("--- 日志尾部 ---")
    for line in tail[-14:]:
        print("   " + line)
    if p.returncode != 0:
        return 1

    for f in (OUT_MODEL, OUT_CFG):
        print("%-60s %d bytes" % (os.path.basename(f), os.path.getsize(f)))

    import msgpack
    m = msgpack.unpackb(open(OUT_MODEL, "rb").read(), raw=False)
    c = msgpack.unpackb(open(OUT_CFG, "rb").read(), raw=False)
    print()
    print("=== 顶层键 ===")
    print(" model:", sorted(m.keys()))
    print(" config:", sorted(c.keys()))
    print(" global_sf=%s k=%s num_cols=%s num_random=%s use_selectors=%s"
          % (m["global_sf"], m["k"], m["num_cols"], m["num_random"], m["use_selectors"]))
    print(" n_layers=%d n_tensors=%d inp_idxes=%s out_idxes=%s"
          % (len(m["layers"]), len(m["tensors"]), m["inp_idxes"], m["out_idxes"]))
    print(" commit_before=%s commit_after=%s" % (m["commit_before"], m["commit_after"]))
    print()
    print("=== 层表（TFLite 侧对照）===")
    print("  idx  layer_type        params                         out_shapes")
    for i, L in enumerate(m["layers"]):
        print("  %3d  %-16s  %-30s %s" % (
            i, L["layer_type"], str(L["params"])[:30], L["out_shapes"]))

    json.dump({"n_layers": len(m["layers"]),
               "layers": [{"i": i, "type": L["layer_type"], "params": L["params"],
                           "inp_shapes": L["inp_shapes"], "out_shapes": L["out_shapes"],
                           "mask": L["mask"]} for i, L in enumerate(m["layers"])],
               "tensors": [{"idx": t["idx"], "shape": t["shape"]} for t in m["tensors"]],
               "config": {k: c[k] for k in c if k != "layers"}},
              open(os.path.join(REPRO, "assets/p3-msgpack-summary.json"), "w"), indent=1)
    print()
    print("summary:", os.path.join(REPRO, "assets/p3-msgpack-summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
