#!/usr/bin/env python3
"""P4-A：生成 converter.py 的契约补丁（不改仓库原文件）。

生成物：
  patches/converter_p4a.py      —— 打过补丁的完整文件（供审阅/后续 P4-B 使用）
  patches/converter_p4a.diff    —— unified diff（供核对改动面）

补丁做四件事：
  E1 建立 producer / consumers 图索引（按 tensor ID，不靠列表相邻）
  E2 元数据张量从运行时依赖移除：只对**本图出现的** PAD / PADV2 / MEAN 生效，
     并且要求该张量的全部消费者都只在元数据位置引用它；
     移除后按数据依赖重算 keep_tensors（而不是只从 tensors 里删条目）
  E3 PADV2 图级保护：常量必须是 -inf 或 0；-inf 时要求生产者带 ReLU；
     输出必须唯一被 VALID 的 MAX_POOL_2D 消费、不是图输出、
     填充量对称且小于核（不存在"整窗都是填充"的窗口）
  E4 数值边界：写 msgpack 前检查有限性与 i32 范围，舍入规则显式写成
     round-half-away-from-zero（numpy 的 round 是 half-to-even）
"""
import difflib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "zkml-main/python/converter.py")
OUT_PY = os.path.join(ROOT, "patches/converter_p4a.py")
OUT_DIFF = os.path.join(ROOT, "patches/converter_p4a.diff")

# ---------------------------------------------------------------- E1 helpers
HELPERS = '''
def round_half_away_from_zero(x):
  """显式舍入规则：0.5 远离 0。numpy 的 .round() 是 half-to-even，不能混用。"""
  x = np.asarray(x, dtype=np.float64)
  return np.sign(x) * np.floor(np.abs(x) + 0.5)


# 哪些算子把某些输入位置当作**元数据**（只用于生成 params，Rust 层运行时只读数据输入）。
# 只登记本图实际出现的算子；新增算子必须先逐一核对对应 Layer::forward 读了哪几个 tensor。
METADATA_INPUT_POS = {
  'PAD':    [1],      # paddings
  'PADV2':  [1, 2],   # paddings, constant
  'MEAN':   [1],      # axes
}

# 带 fused activation 的算子 -> 取激活的读取器
FUSED_ACT_OPS = {
  'CONV_2D': ('Conv2DOptions', 'FusedActivationFunction'),
  'DEPTHWISE_CONV_2D': ('DepthwiseConv2DOptions', 'FusedActivationFunction'),
  'ADD': ('AddOptions', 'FusedActivationFunction'),
  'SUB': ('SubOptions', 'FusedActivationFunction'),
  'MUL': ('MulOptions', 'FusedActivationFunction'),
  'FULLY_CONNECTED': ('FullyConnectedOptions', 'FusedActivationFunction'),
}
'''

# ---------------------------------------------------------------- E7 methods
METHODS = '''
  # ------------------------------------------------------------------ P4-A
  def _build_graph_maps(self):
    """按 tensor ID 建 producer / consumers 索引。不依赖算子列表顺序。"""
    graph = self.graph
    self.producer = {}
    self.consumers = {}
    self.op_codes = []
    for op_idx in range(graph.OperatorsLength()):
      op = graph.Operators(op_idx)
      code = self.model.OperatorCodes(op.OpcodeIndex())
      bcode = code.BuiltinCode()
      if bcode == 0:
        bcode = code.DeprecatedBuiltinCode()
      self.op_codes.append(bcode)
      for i in range(op.InputsLength()):
        t = op.Inputs(i)
        if t is None or t < 0:
          continue
        self.consumers.setdefault(t, []).append((op_idx, i))
      for i in range(op.OutputsLength()):
        t = op.Outputs(i)
        if t is None or t < 0:
          continue
        self.producer[t] = op_idx

  def _op_name(self, op_idx):
    code = self.op_codes[op_idx]
    for attr in dir(tflite.BuiltinOperator):
      if not attr.startswith('__') and getattr(tflite.BuiltinOperator, attr) == code:
        return attr
    return 'OP_%d' % code

  def _option(self, op, cls):
    o = op.BuiltinOptions()
    if o is None:
      return None
    v = cls()
    v.Init(o.Bytes, o.Pos)
    return v

  def _fused_activation(self, op_idx):
    """返回该算子的 fused activation 名；无此概念返回 None。"""
    name = self._op_name(op_idx)
    if name not in FUSED_ACT_OPS:
      return None
    cls_name, getter = FUSED_ACT_OPS[name]
    op = self.graph.Operators(op_idx)
    o = self._option(op, getattr(tflite, cls_name))
    if o is None:
      return None
    val = getattr(o, getter)()
    for attr in dir(tflite.ActivationFunctionType):
      if not attr.startswith('__') and getattr(tflite.ActivationFunctionType, attr) == val:
        return attr
    return 'ACT_%d' % val

  def _metadata_tensor_idxes(self, op, op_name):
    """该算子的哪些输入 tensor 是元数据（只喂 params），并且可以安全地不参与运行时依赖。"""
    positions = METADATA_INPUT_POS.get(op_name)
    if not positions:
      return set()
    out = set()
    for pos in positions:
      if pos >= op.InputsLength():
        continue
      t = op.Inputs(pos)
      if t is None or t < 0:
        continue
      # 只有在「全部消费者都把它当元数据」时才允许移除
      ok = True
      for (c_op, c_pos) in self.consumers.get(t, []):
        c_name = self._op_name(c_op)
        if c_pos not in METADATA_INPUT_POS.get(c_name, []):
          ok = False
          break
      if not ok:
        raise RuntimeError(
          'tensor %d 仍被数据路径引用（消费者 op%d 的输入位置 %d），'
          '不能作为元数据移除' % (t, c_op, c_pos))
      out.add(t)
    return out

  def _convert_padv2(self, op, inp_idxes, evidence):
    """PADV2 -> Pad 的图级保护。

    前提（任一不满足即拒绝，不静默降级）：
      P1 常量是单元素，且为 -inf 或 0；
      P2 若常量为 -inf（PyTorch MaxPool 语义），则数据输入的生产者必须带 ReLU，
         从而逐元素非负 —— 这时 Rust PadChip 的 0 填充才与 -inf 等价；
         若常量为 0，则该条自然成立，无需非负前提；
      P3 Pad 输出只有一个消费者，且是 pad=VALID 的 MAX_POOL_2D；
      P4 Pad 输出不是图输出；
      P5 填充量对称，且每侧 < 核尺寸（保证不存在整窗都是填充的窗口）。
    """
    data_t = op.Inputs(0)
    pad_t = op.Inputs(1)
    const_t = op.Inputs(2)

    paddings = self.interpreter.get_tensor(pad_t).flatten().astype(np.int64).tolist()
    const = np.asarray(self.interpreter.get_tensor(const_t)).flatten()
    if const.size != 1:
      raise RuntimeError('PADV2 常量不是单元素: shape=%s' % (const.shape,))
    c = float(const[0])
    if not (np.isneginf(c) or c == 0.0):
      raise RuntimeError('PADV2 常量只支持 -inf 或 0，实际 %r' % (c,))
    ev = {'pad_op': None, 'constant_is_neg_inf': bool(np.isneginf(c)),
          'paddings': paddings, 'data_tensor': int(data_t)}

    # P2 非负性
    prod = self.producer.get(int(data_t))
    if prod is None:
      if np.isneginf(c):
        raise RuntimeError('PADV2 数据输入 %d 没有生产者（是模型输入），'
                           '-inf 填充无法降级为 0' % data_t)
      ev['producer'] = 'MODEL_INPUT'
    else:
      act = self._fused_activation(prod)
      ev['producer'] = '%s(act=%s)' % (self._op_name(prod), act)
      if np.isneginf(c) and act != 'RELU':
        raise RuntimeError(
          'PADV2 常量为 -inf 但生产者 op%d(%s) 的 fused activation 是 %r 而非 RELU；'
          'Rust PadChip 只能填 0，此处会改变语义，拒绝转换'
          % (prod, self._op_name(prod), act))
      ev['producer_act_is_relu'] = (act == 'RELU')

    # P3 / P4 输出唯一被 VALID maxpool 消费且不是图输出
    out_t = op.Outputs(0)
    graph_outputs = {int(self.graph.Outputs(i)) for i in range(self.graph.OutputsLength())}
    if int(out_t) in graph_outputs:
      raise RuntimeError('PADV2 输出 %d 是图输出，属于公开可观察张量，拒绝降级' % out_t)
    cons = self.consumers.get(int(out_t), [])
    ev['pad_out_tensor'] = int(out_t)
    ev['is_graph_output'] = False
    ev['n_consumers'] = len(cons)
    if len(cons) != 1:
      raise RuntimeError('PADV2 输出 %d 有 %d 个消费者，要求恰好 1 个 MaxPool'
                         % (out_t, len(cons)))
    c_op, _c_pos = cons[0]
    c_name = self._op_name(c_op)
    ev['consumer'] = c_name
    if c_name != 'MAX_POOL_2D':
      raise RuntimeError('PADV2 输出 %d 的消费者是 %s，不是 MAX_POOL_2D' % (out_t, c_name))
    pool = self.graph.Operators(c_op)
    pool_opt = self._option(pool, tflite.Pool2DOptions)
    if pool_opt is None:
      raise RuntimeError('MaxPool options is None')
    if pool_opt.Padding() != tflite.Padding.VALID:
      raise RuntimeError('MaxPool 是 SAME，本契约只覆盖显式 PAD + VALID')
    if pool_opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
      raise RuntimeError('MaxPool 带 fused activation，未覆盖')

    # P5 窗口几何
    fh, fw = pool_opt.FilterHeight(), pool_opt.FilterWidth()
    ph0, ph1 = paddings[2], paddings[3]
    pw0, pw1 = paddings[4], paddings[5]
    ev.update({'filter': [fh, fw], 'pad_h': [ph0, ph1], 'pad_w': [pw0, pw1],
               'stride': [pool_opt.StrideH(), pool_opt.StrideW()]})
    if ph0 != ph1 or pw0 != pw1:
      raise RuntimeError('PADV2 填充不对称，PyTorch pad 是对称的')
    if ph0 >= fh or pw0 >= fw:
      raise RuntimeError('填充量 >= 核尺寸，存在整窗都是填充的池化窗口')
    if paddings[0] or paddings[1] or paddings[6] or paddings[7]:
      raise RuntimeError('PADV2 在 batch/channel 轴上有填充，未覆盖')

    ev['pad_op'] = True
    evidence.append(ev)
    return ('Pad', paddings)

'''

# ---------------------------------------------------------------- 数值检查
QUANT = '''  def _quantize_checked(self, arr, tensor_idx):
    """写 msgpack 前的数值边界检查 + 显式舍入。"""
    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
      bad = int(np.size(a) - np.count_nonzero(np.isfinite(a)))
      raise RuntimeError('tensor %d 含 %d 个非有限值，拒绝写入 msgpack'
                         '(原实现会把它变成 INT64_MIN)' % (tensor_idx, bad))
    scaled = a * float(self.scale_factor)
    lim = float((1 << 31) - 1)
    mx = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if mx > lim:
      raise RuntimeError('tensor %d 乘 scale 后绝对值 %.1f 超过 i32 上限 %.0f'
                         % (tensor_idx, mx, lim))
    return round_half_away_from_zero(scaled).astype(np.int64)

'''


def main():
    src = open(SRC, encoding="utf-8").read()
    out = src
    applied = []

    def sub(old, new, tag):
        nonlocal out
        assert out.count(old) == 1, "锚点不唯一或缺失: %s" % tag
        out = out.replace(old, new)
        applied.append(tag)

    # E1 全局 helper
    sub("""def get_inputs(op: tflite.Operator):
  idxes = handle_numpy_or_literal(op.InputsAsNumpy())
  idxes = idxes.tolist()
  idxes = list(filter(lambda x: x != -1, idxes))
  return idxes
""",
        """def get_inputs(op: tflite.Operator):
  idxes = handle_numpy_or_literal(op.InputsAsNumpy())
  idxes = idxes.tolist()
  idxes = list(filter(lambda x: x != -1, idxes))
  return idxes
""" + HELPERS,
        "E1 helpers")

    # E2 图索引在 __init__ 里建立
    sub("""    self.graph = self.model.Subgraphs(0)
""",
        """    self.graph = self.model.Subgraphs(0)
    self._build_graph_maps()
""",
        "E2 build graph maps")

    # E3 去掉"先无条件保留所有输入"的写法，改为按数据依赖保留
    sub("""      # Keep the input tensors
      for input in handle_numpy_or_literal(op.InputsAsNumpy()):
        keep_tensors.add(input)
""",
        """      # P4-A：不再无条件保留全部输入。元数据张量（paddings / 常量 / axes）
      # 在算出 inp_idxes 之后再按数据依赖决定是否保留，见下面的 E4。
""",
        "E3 drop unconditional keep")

    # E4 用过滤后的 inp_idxes 重算 keep_tensors
    sub("""      inp_idxes = get_inputs(op)
""",
        """      inp_idxes = get_inputs(op)
      # P4-A：元数据张量不进运行时依赖，也不进 keep_tensors
      _meta = self._metadata_tensor_idxes(op, self._op_name(op_idx))
      if _meta:
        inp_idxes = [i for i in inp_idxes if i not in _meta]
        print('dropping metadata inputs %s from layer %d (%s)'
              % (sorted(_meta), op_idx, self._op_name(op_idx)))
      for _i in inp_idxes:
        keep_tensors.add(_i)
""",
        "E4 recompute keep_tensors")

    # E5 PADV2 分支（带图级保护）
    sub("""      # Pad
      elif op_code == tflite.BuiltinOperator.PAD:
""",
        """      # Pad
      elif op_code == tflite.BuiltinOperator.PADV2:
        layer_type, params = self._convert_padv2(op, get_inputs(op), padv2_evidence)
      elif op_code == tflite.BuiltinOperator.PAD:
""",
        "E5 padv2 branch")

    # E6 证据容器
    sub("""    layers = []
    keep_tensors = set()
""",
        """    layers = []
    keep_tensors = set()
    padv2_evidence = []
""",
        "E6 evidence list")

    # E7 数值检查替换
    sub("""      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = (tensor_data * self.scale_factor).round().astype(np.int64)
""",
        """      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = self._quantize_checked(tensor_data, tensor_idx)
""",
        "E7 checked quantize")

    # E8 方法体
    sub("""  def to_dict(self, start_layer, end_layer):
""",
        METHODS + """  def to_dict(self, start_layer, end_layer):
""",
        "E8 methods")

    # E11 命令行暴露证据旁路参数
    sub("""  parser.add_argument('--num_randoms', type=int, default=20001)
""",
        """  parser.add_argument('--num_randoms', type=int, default=20001)
  parser.add_argument('--padv2_evidence', type=str, default=None,
                      help='P4-A: 把 PADV2 图级保护的检查证据写成 JSON')
""",
        "E11 argparse evidence")

    # E12 把参数接到实例上
    sub("""  model_packed, config_packed = converter.to_msgpack(
""",
        """  converter.padv2_evidence_out = args.padv2_evidence
  model_packed, config_packed = converter.to_msgpack(
""",
        "E12 wire evidence path")

    # E9 PADV2 证据落到旁路文件（不改 msgpack schema）
    sub("""  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
""",
        """  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
    if getattr(self, 'padv2_evidence_out', None):
      import json as _json
      with open(self.padv2_evidence_out, 'w') as _f:
        _json.dump(getattr(self, 'padv2_evidence', []), _f, indent=1)
""",
        "E9 evidence sidecar")

    # E10 __init__ 里挂证据字段
    sub("""    self.expose_output = expose_output
""",
        """    self.expose_output = expose_output
    self.padv2_evidence = []
    self.padv2_evidence_out = None
""",
        "E10 evidence field")

    open(OUT_PY, "w", encoding="utf-8").write(out)
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="zkml-main/python/converter.py", tofile="patches/converter_p4a.py",
        n=3)
    open(OUT_DIFF, "w", encoding="utf-8").writelines(diff)

    print("已应用 %d 处编辑：" % len(applied))
    for a in applied:
        print("  -", a)
    print()
    print("fixed  ->", OUT_PY, os.path.getsize(OUT_PY), "bytes")
    print("diff   ->", OUT_DIFF, os.path.getsize(OUT_DIFF), "bytes")
    # 语法检查（只解析，不执行转换）
    import py_compile
    try:
        py_compile.compile(OUT_PY, doraise=True, cfile=os.devnull + ".pyc")
        print("语法检查：通过（py_compile，未执行任何转换）")
    except Exception as e:
        print("语法检查：失败 ->", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
