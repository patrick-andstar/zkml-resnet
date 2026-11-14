#!/usr/bin/env python3
"""P4-A v3（converter 侧）：4b / 5 / 6 三条补正。

4b  观察集合：先形成「整图输出 ∪ 导出子图公开输出」，再判断 PADV2 的 -inf
    能否降为 0；不再只看最后一层输出和消费者数。
5   元数据按**输入位置**过滤（不再按 tensor ID），并同步 inp_idxes；
    普通 PAD 的 paddings / MEAN 的 axes 必须走 _static_constant，禁止冻结
    一次 invoke 的结果当常量。
6   舍入去掉 abs(x)+0.5 的二次舍入；CUSTOM 先升 float64 再乘 scale；
    复核主循环 / CUSTOM / DIV 三条路径，确认不漏检也不重复缩放。

输入 patches/converter_p4a_v2.py -> 输出 patches/converter_p4a_v3.py
"""
import io
import os
import py_compile
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "patches", "converter_p4a_v2.py")
DST = os.path.join(ROOT, "patches", "converter_p4a_v3.py")


ROUND_OLD = '''def round_half_away_from_zero(x):
  """显式舍入规则：0.5 远离 0。numpy 的 .round() 是 half-to-even，不能混用。"""
  x = np.asarray(x, dtype=np.float64)
  return np.sign(x) * np.floor(np.abs(x) + 0.5)
'''

ROUND_NEW = '''def round_half_away_from_zero(x):
  """显式舍入：0.5 远离 0，且不引入二次舍入。

  [阻断6] `sign(x)*floor(abs(x)+0.5)` 在 binary64 上不是全域正确：
    * x = 0.5 - 2^-54（0.5 前一个可表示浮点数）应舍为 0，
      但 abs(x)+0.5 先在 float64 里被舍成 1.0，floor 得到 1；
    * x = 2^52 + 1 本身是整数，加 0.5 得 2^52 + 1.5；该量级 binary64
      的间隔是 1，ties-to-even 会得到 2^52 + 2。
  改为"取整 + 判余数"：对 |x| < 2^53，floor 与 x - floor(x) 都是精确的，
  余数比较不产生新的舍入；|x| >= 2^53 时余数恒为 0，原样返回。
  """
  x = np.asarray(x, dtype=np.float64)
  a = np.abs(x)
  fl = np.floor(a)
  frac = a - fl
  return np.sign(x) * (fl + (frac >= 0.5).astype(np.float64))
'''

META_START = "  def _metadata_tensor_idxes(self, op, op_name):"
META_END = "  def _convert_padv2(self, op, op_idx):"

META_NEW = '''  def _metadata_input_positions(self, op, op_name):
    """该算子哪些**输入位置**是元数据（只喂 params），可从本层运行时依赖里去掉。

    [阻断5] 返回的是**位置**而不是 tensor ID。按 ID 过滤会把"同一个 tensor
    同时出现在数据位置和元数据位置"的引用一起删掉：例如 INT32 零常量 t
    形状 [2,2]，`PAD([t,t])` 本是无填充恒等，按 ID 删就会变成零输入。
    按位置过滤只去掉元数据那几处，数据位置原样保留。
    """
    positions = METADATA_INPUT_POS.get(op_name)
    if not positions:
      return []
    out = []
    for pos in positions:
      if pos >= op.InputsLength():
        continue
      t = op.Inputs(pos)
      if t is None or t < 0:
        continue
      out.append(int(pos))
      shared = [(c_op, c_pos) for (c_op, c_pos) in self.consumers.get(int(t), [])
                if c_pos not in METADATA_INPUT_POS.get(self._op_name(c_op), [])]
      if shared:
        # 记录而非拒绝：这些引用会让该 tensor 留在 keep_tensors 里
        self.shared_metadata.setdefault(int(t), []).extend(
          ['%s@%d' % (self._op_name(c_op), c_pos) for (c_op, c_pos) in shared])
    return out

'''

CALLER_OLD = '''      inp_idxes = get_inputs(op)
      # P4-A：元数据张量只从**本层**运行时依赖里去掉；若别处当数据用，
      # 它仍会经那个算子的 inp_idxes 进入 keep_tensors（不误删共享常量）
      _meta = self._metadata_tensor_idxes(op, self._op_name(op_idx))
      if _meta:
        inp_idxes = [i for i in inp_idxes if i not in _meta]
        print('dropping metadata inputs %s from layer %d (%s)'
              % (sorted(_meta), op_idx, self._op_name(op_idx)))
'''

CALLER_NEW = '''      # P4-A v3（阻断5）：按**输入位置**剔除元数据，不按 tensor ID。
      # get_inputs 会先过滤 -1（可选输入），位置会错位，所以这里直接读原始数组。
      _raw_inputs = [int(x) for x in handle_numpy_or_literal(op.InputsAsNumpy()).tolist()]
      _meta_pos = self._metadata_input_positions(op, self._op_name(op_idx))
      if _meta_pos:
        _pos_set = set(_meta_pos)
        _dropped = [_raw_inputs[p] for p in _meta_pos if 0 <= p < len(_raw_inputs)]
        inp_idxes = [t for p, t in enumerate(_raw_inputs)
                     if p not in _pos_set and t != -1]
        print('dropping metadata inputs %s at positions %s from layer %d (%s)'
              % (sorted(_dropped), sorted(_meta_pos), op_idx, self._op_name(op_idx)))
      else:
        inp_idxes = [t for t in _raw_inputs if t != -1]
'''

PAD_OLD = '''      elif op_code == tflite.BuiltinOperator.PAD:
        layer_type = 'Pad'
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().astype(np.int64)
        params = tensor.tolist()
'''

PAD_NEW = '''      elif op_code == tflite.BuiltinOperator.PAD:
        layer_type = 'Pad'
        tensor_idx = op.Inputs(1)
        # [阻断5] paddings 必须是编译期常量。有 producer 的 tensor 是运行时值，
        # 冻结一次 invoke 的结果会把一条图依赖伪装成常量。
        tensor = self._static_constant(tensor_idx, 'PAD(op%d).paddings' % op_idx)
        tensor = tensor.flatten().astype(np.int64)
        params = tensor.tolist()
'''

MEAN_OLD = '''        inp_shape = interpreter.get_tensor(op.Inputs(0)).shape
        mean_idxes = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64)
'''

MEAN_NEW = '''        inp_shape = interpreter.get_tensor(op.Inputs(0)).shape
        # [阻断5] axes 同样必须是编译期常量，理由同 PAD。
        mean_idxes = self._static_constant(op.Inputs(1), 'MEAN(op%d).axes' % op_idx)
        mean_idxes = mean_idxes.flatten().astype(np.int64)
'''

CUSTOM_OLD = '''        weights = self._quantize_checked(weights * self.scale_factor, op.Inputs(1),
                                        already_scaled=True)
'''

CUSTOM_NEW = '''        # [阻断6] 不能先乘 scale 再交给 helper：weights 通常是 float32，
        # float32 * scale 会先在 float32 里舍一次，之后的有限性与范围检查
        # 已经看不到丢掉的精度。传原始权重，由 helper 先升 float64 再乘。
        weights = self._quantize_checked(weights, op.Inputs(1), already_scaled=False)
'''

DIV_OLD = '''        adjusted_tensors[op.Inputs(1)] = self._quantize_checked(
          np.asarray([self.scale_factor / div_val]), op.Inputs(1), already_scaled=True)
'''

DIV_NEW = '''        # [阻断6] scale/div 必须在 float64 里算：div_val 是 float32，
        # 直接相除会先在 float32 里舍入，helper 的检查看不到这一步。
        _ratio = np.asarray([float(self.scale_factor) / float(div_val)], dtype=np.float64)
        adjusted_tensors[op.Inputs(1)] = self._quantize_checked(
          _ratio, op.Inputs(1), already_scaled=True)
'''

VALIDATE_START = "  def _validate_padv2_subgraph(self, layers, start_layer, end_layer):"
VALIDATE_END = "  def _quantize_checked(self, arr, tensor_idx, already_scaled=False):"

VALIDATE_NEW = '''  def _observation_tensors(self, layers):
    """[阻断4b] 先形成"会被外部观察"的 tensor 集合，再判定 -inf 能否降级。

    来源必须算全，缺一个就可能把 Pad 的 -inf 静默换成 0：
      1) 整图输出 graph.Outputs()：即使没有 expose_output，这也是模型的真实
         输出；commit_after 取的正是 output_details，与之等价。
      2) 导出子图公开输出 layers[-1]['out_idxes']（仅 expose_output 时）。
      3) 额外消费者：由 _validate_padv2_subgraph 用 consumers 单独检查。
    commit_before 不会含 Pad 输出：它由 kept 常量与模型输入构成，而 Pad 输出
    是层产物；若它真被当常量收进 keep_tensors，说明它同时是图输出，已由 (1) 覆盖。
    """
    obs = set()
    for i in range(self.graph.OutputsLength()):
      obs.add(int(self.graph.Outputs(i)))
    if self.expose_output and layers:
      obs.update(int(x) for x in layers[-1]['out_idxes'])
    return obs

  def _validate_padv2_subgraph(self, layers, start_layer, end_layer):
    """[阻断4b] -inf -> 0 只对**实际导出子图**且**未被观察**时成立。

    layers 只包含 [start_layer, end_layer] 内的算子。观察集合 =
    整图输出 ∪ 导出子图公开输出（见 _observation_tensors），
    不再是"只看最后一层输出"。消费者数按子图重算，要求恰好 1 个且是记录的 MaxPool。
    """
    if self._padv2_seen != len(self.padv2_evidence):
      raise RuntimeError('PADV2 计数不一致：见到 %d 个，证据 %d 条'
                         % (self._padv2_seen, len(self.padv2_evidence)))
    if self._padv2_seen and not self.padv2_evidence:
      raise RuntimeError('图中有 %d 个 PADV2，但证据为空' % self._padv2_seen)
    if not self.padv2_evidence:
      return
    in_range = set(self._layer_op_idxes)
    observed = self._observation_tensors(layers)
    for ev in self.padv2_evidence:
      out_t = ev['pad_out_tensor']
      if out_t in observed:
        raise RuntimeError(
          'PADV2(op%d) 的输出 tensor %d 会被外部观察（整图输出或导出子图公开输出），'
          '拒绝把 -inf 降级为 0' % (ev['op_idx'], out_t))
      cons = [c for c in self.consumers.get(out_t, []) if c[0] in in_range]
      if len(cons) != 1:
        raise RuntimeError(
          'PADV2(op%d) 的输出 tensor %d 在子图 [%d,%d] 内有 %d 个消费者，要求恰好 1 个 MaxPool'
          % (ev['op_idx'], out_t, start_layer, end_layer, len(cons)))
      c_op, _c_pos = cons[0]
      if c_op != ev['consumer_op_idx']:
        raise RuntimeError('PADV2(op%d) 子图内消费者 op%d 与转换时记录的 op%d 不一致'
                           % (ev['op_idx'], c_op, ev['consumer_op_idx']))
      ev['n_consumers_in_subgraph'] = 1
      ev['consumer_op_in_subgraph'] = int(c_op)
      ev['is_subgraph_public_output'] = False
      ev['observed_tensor_ids'] = sorted(observed)
      ev['checked_against'] = 'graph_outputs+exported_subgraph+single_consumer'

'''


def main():
    src = io.open(SRC, encoding="utf-8").read()

    pairs = [
        ("ROUND", ROUND_OLD, ROUND_NEW),
        ("CALLER", CALLER_OLD, CALLER_NEW),
        ("PAD", PAD_OLD, PAD_NEW),
        ("MEAN", MEAN_OLD, MEAN_NEW),
        ("CUSTOM", CUSTOM_OLD, CUSTOM_NEW),
        ("DIV", DIV_OLD, DIV_NEW),
    ]
    for name, old, _new in pairs:
        cnt = src.count(old)
        if cnt != 1:
            print("FAIL anchor %s count=%d" % (name, cnt))
            return 1

    out = src
    for _name, old, new in pairs:
        out = out.replace(old, new, 1)

    i = out.find(META_START)
    j = out.find(META_END)
    if i < 0 or j < 0 or j <= i:
        print("FAIL metadata slice i=%d j=%d" % (i, j))
        return 1
    out = out[:i] + META_NEW + out[j:]

    i = out.find(VALIDATE_START)
    j = out.find(VALIDATE_END)
    if i < 0 or j < 0 or j <= i:
        print("FAIL validate slice i=%d j=%d" % (i, j))
        return 1
    out = out[:i] + VALIDATE_NEW + out[j:]

    io.open(DST, "w", encoding="utf-8", newline="\n").write(out)

    checks = [
        ("舍入改判余数", "frac = a - fl" in out and "frac >= 0.5" in out),
        ("舍入旧式已消失", "np.abs(x) + 0.5" not in out),
        ("位置过滤函数", "def _metadata_input_positions" in out),
        ("旧按ID函数已消失", "_metadata_tensor_idxes" not in out),
        ("调用点用位置", "_pos_set = set(_meta_pos)" in out and "_raw_inputs" in out),
        ("PAD 走静态常量", "self._static_constant(tensor_idx, 'PAD(op%d).paddings'" in out),
        ("MEAN 走静态常量", "self._static_constant(op.Inputs(1), 'MEAN(op%d).axes'" in out),
        ("CUSTOM 先升精度", "self._quantize_checked(weights, op.Inputs(1), already_scaled=False)" in out),
        ("CUSTOM 旧式已消失", "weights * self.scale_factor" not in out),
        ("DIV 用 float64", "float(self.scale_factor) / float(div_val)" in out),
        ("观察集合函数", "def _observation_tensors" in out),
        ("含整图输出", "self.graph.OutputsLength()" in out),
        ("消费链已更新", "graph_outputs+exported_subgraph+single_consumer" in out),
        ("旧 checked_against 已更新", "'exported_subgraph'" not in out),
    ]
    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print("  %-24s %s" % (n, "OK" if ok else "FAIL"))

    # 语法检查：py_compile 只证明能解析，不证明语义
    try:
        tmp = tempfile.mkdtemp()
        py_compile.compile(DST, cfile=os.path.join(tmp, "x.pyc"), doraise=True)
        print("  %-24s OK" % "py_compile 语法")
    except Exception as e:
        print("  %-24s FAIL %s" % ("py_compile 语法", e))
        bad.append("py_compile")

    if bad:
        print("GATE FAIL: %s" % ", ".join(bad))
        return 1
    print("GATE OK  -> %s (%d bytes)" % (os.path.basename(DST), len(out.encode())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
