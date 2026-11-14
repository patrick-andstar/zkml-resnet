#!/usr/bin/env python3
"""P4-A 补正：生成 converter.py 的 v2 契约补丁（不改仓库原文件）。

针对 plan.md §2.8.2 的阻断 1、2、4 逐条修正；v1 产物
（patches/converter_p4a.py）保留不动，本脚本写 _v2 新文件，另立身份。

v2 相对 v1 的差异（对照 §2.8.2 编号）：
  [阻断1] 真正插入 _quantize_checked（v1 的生成器定义了 QUANT 却只插了 METHODS），
          并把三处 .round().astype 全部接到它上面：主张量循环、CUSTOM 权重、
          DIV 的 adjusted_tensors。DIV 与 CUSTOM 权重已经乘过 sf，用
          already_scaled=True 避免二次缩放。
  [阻断2] 证据容器统一到 self.padv2_evidence（v1 的 to_dict 用了局部列表，
          to_msgpack 导出的是永远为空的实例属性）；每次 to_dict 重置；
          证据记录真实 op/tensor 身份，并在写盘时绑定 model/config 摘要。
  [阻断4] 图保护改为对**实际转换子图**成立：
          P3/P4 推迟到整段转换结束后按 [start_layer, end_layer] 复检，
          公开输出用 layers[-1]['out_idxes'] 而不是整图 graph.Outputs；
          新增 P0/P1b（静态常量、[4,2]、非负、正核/stride、shape 一致性）；
          P2 额外要求生产者 op 本身在子图内（防 start_layer 裁掉 ReLU）。
          元数据共享改为"按引用位置清理"而非整体拒绝。

脚本自带 AST 门禁：生成后重新解析候选，逐项断言 v1 的坑没有复发。
"""
import ast
import difflib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "zkml-main/python/converter.py")
OUT_PY = os.path.join(ROOT, "patches/converter_p4a_v2.py")
OUT_DIFF = os.path.join(ROOT, "patches/converter_p4a_v2.diff")

# ------------------------------------------------------------------ E1 helpers
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

# 允许 i64 表示的上界（数值契约用，与 model.rs 的 FIELD_ENC_HI 分开）
I64_MAX = (1 << 63) - 1
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1
'''

# ------------------------------------------------------------------ E8 methods
METHODS = '''
  # ------------------------------------------------------------------ P4-A v2
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

  def _static_constant(self, tensor_idx, what):
    """确认该 tensor 是编译期常量并返回其值。

    [阻断4] 不能把运行时算出来的值（零输入 invoke 的结果）冻结进 params：
    有 producer 的 tensor 是数据流上的中间结果，它的值取决于输入，把它当
    paddings/constant 写进 params 会把一条图依赖伪装成常量。
    """
    tid = int(tensor_idx)
    prod = self.producer.get(tid)
    if prod is not None:
      raise RuntimeError(
        '%s 的 tensor %d 由 op%d 产生，是运行时计算值，不能冻结进 params'
        % (what, tid, prod))
    tensor = self.graph.Tensors(tid)
    if tensor is None:
      raise RuntimeError('%s 的 tensor %d 不存在' % (what, tid))
    bidx = tensor.Buffer()
    if bidx == 0:
      raise RuntimeError('%s 的 tensor %d 没有常量 buffer（Buffer()==0）' % (what, tid))
    buf = self.model.Buffers(bidx)
    if buf is None or buf.DataLength() == 0:
      raise RuntimeError('%s 的 tensor %d 的常量 buffer 为空' % (what, tid))
    return np.asarray(self.interpreter.get_tensor(tid))

  def _metadata_tensor_idxes(self, op, op_name):
    """该算子的哪些输入位置是元数据（只喂 params），可从**本层**运行时依赖里去掉。

    [阻断4] 按引用位置清理：只从这一层的 inp_idxes 里去掉，不要求"全部消费者
    都只把它当元数据"。若别的算子把它当数据用，它会因为那个算子的 inp_idxes
    仍然进入 keep_tensors，从而继续被保留、继续被 commit —— 不会误删共享常量。
    """
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
      out.add(int(t))
      shared = [(c_op, c_pos) for (c_op, c_pos) in self.consumers.get(int(t), [])
                if c_pos not in METADATA_INPUT_POS.get(self._op_name(c_op), [])]
      if shared:
        # 记录而非拒绝：这些引用会让该 tensor 留在 keep_tensors 里
        self.shared_metadata.setdefault(int(t), []).extend(
          ['%s@%d' % (self._op_name(c_op), c_pos) for (c_op, c_pos) in shared])
    return out

  def _convert_padv2(self, op, op_idx):
    """PADV2 -> Pad 的图级保护（P0/P1/P1b/P2/P5）。

    P3/P4 依赖"这段转换实际覆盖哪些层"，只能在整个 [start_layer, end_layer]
    走完之后判定，见 _validate_padv2_subgraph。
    """
    if op.InputsLength() != 3:
      raise RuntimeError('PADV2(op%d) 期望 3 个输入，实际 %d' % (op_idx, op.InputsLength()))
    data_t, pad_t, const_t = op.Inputs(0), op.Inputs(1), op.Inputs(2)
    for nm, t in (('data', data_t), ('paddings', pad_t), ('constant', const_t)):
      if t is None or t < 0:
        raise RuntimeError('PADV2(op%d) 的 %s 输入缺失（%r）' % (op_idx, nm, t))

    # P1b paddings 必须是静态常量、rank=2 且形状 [4,2]、非负整数、batch/channel 轴为 0
    paddings_arr = self._static_constant(pad_t, 'PADV2(op%d).paddings' % op_idx)
    if paddings_arr.ndim != 2 or paddings_arr.shape != (4, 2):
      raise RuntimeError('PADV2(op%d) paddings 必须是 [4,2]，实际 %s'
                         % (op_idx, paddings_arr.shape))
    if not np.issubdtype(paddings_arr.dtype, np.integer):
      raise RuntimeError('PADV2(op%d) paddings 必须是整数，实际 dtype=%s'
                         % (op_idx, paddings_arr.dtype))
    paddings = [int(v) for v in paddings_arr.reshape(-1)]
    if any(v < 0 for v in paddings):
      raise RuntimeError('PADV2(op%d) paddings 含负值 %r（负填充=裁剪，未覆盖）'
                         % (op_idx, paddings))
    if paddings[0] or paddings[1] or paddings[6] or paddings[7]:
      raise RuntimeError('PADV2(op%d) 在 batch/channel 轴上有填充 %r，未覆盖'
                         % (op_idx, paddings))

    # P1 常量必须是静态常量、单元素、且为 -inf 或 0
    const_arr = self._static_constant(const_t, 'PADV2(op%d).constant' % op_idx)
    if const_arr.size != 1:
      raise RuntimeError('PADV2(op%d) 常量不是单元素: shape=%s'
                         % (op_idx, const_arr.shape))
    c = float(const_arr.reshape(-1)[0])
    if not (np.isneginf(c) or c == 0.0):
      raise RuntimeError('PADV2(op%d) 常量只支持 -inf 或 0，实际 %r' % (op_idx, c))
    is_neginf = bool(np.isneginf(c))

    ev = {
      'op_idx': int(op_idx),
      'op_name': 'PADV2',
      'data_tensor': int(data_t),
      'paddings_tensor': int(pad_t),
      'constant_tensor': int(const_t),
      'constant': ('-inf' if is_neginf else 0.0),
      'constant_is_neg_inf': is_neginf,
      'paddings': paddings,
      'subgraph_range': list(self._range),
    }

    # P2 -inf 时要求生产者带 ReLU，且生产者本身必须落在子图内
    prod = self.producer.get(int(data_t))
    if prod is None:
      if is_neginf:
        raise RuntimeError('PADV2(op%d) 数据输入 %d 没有生产者（是模型输入），'
                           '-inf 填充无法降级为 0' % (op_idx, data_t))
      ev['producer_op_idx'] = None
      ev['producer'] = 'MODEL_INPUT'
      ev['producer_act'] = None
    else:
      act = self._fused_activation(prod)
      ev['producer_op_idx'] = int(prod)
      ev['producer'] = self._op_name(prod)
      ev['producer_act'] = act
      if is_neginf and act != 'RELU':
        raise RuntimeError(
          'PADV2(op%d) 常量为 -inf 但生产者 op%d(%s) 的 fused activation 是 %r 而非 RELU；'
          'Rust PadChip 只能填 0，此处会改变语义，拒绝转换'
          % (op_idx, prod, self._op_name(prod), act))
      if is_neginf and prod < self._range[0]:
        raise RuntimeError(
          'PADV2(op%d) 常量为 -inf，但其 ReLU 生产者 op%d 在 start_layer=%d 之外；'
          '导出的子图没有这段非负保证，拒绝降级'
          % (op_idx, prod, self._range[0]))

    # P5 窗口几何：正核/stride、对称填充、每侧 < 核、shape 链条自洽
    out_t = op.Outputs(0)
    cons = self.consumers.get(int(out_t), [])
    if len(cons) != 1:
      raise RuntimeError('PADV2(op%d) 输出 %d 有 %d 个消费者，要求恰好 1 个 MaxPool'
                         % (op_idx, out_t, len(cons)))
    c_op, _c_pos = cons[0]
    c_name = self._op_name(c_op)
    if c_name != 'MAX_POOL_2D':
      raise RuntimeError('PADV2(op%d) 输出的消费者是 %s，不是 MAX_POOL_2D'
                         % (op_idx, c_name))
    pool = self.graph.Operators(c_op)
    pool_opt = self._option(pool, tflite.Pool2DOptions)
    if pool_opt is None:
      raise RuntimeError('MaxPool(op%d) options is None' % c_op)
    if pool_opt.Padding() != tflite.Padding.VALID:
      raise RuntimeError('MaxPool(op%d) 是 SAME，本契约只覆盖显式 PAD + VALID' % c_op)
    if pool_opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
      raise RuntimeError('MaxPool(op%d) 带 fused activation，未覆盖' % c_op)

    fh, fw = pool_opt.FilterHeight(), pool_opt.FilterWidth()
    sh, sw = pool_opt.StrideH(), pool_opt.StrideW()
    if fh <= 0 or fw <= 0 or sh <= 0 or sw <= 0:
      raise RuntimeError('MaxPool(op%d) 的核/stride 非正: kernel=(%d,%d) stride=(%d,%d)'
                         % (c_op, fh, fw, sh, sw))

    ph0, ph1 = paddings[2], paddings[3]
    pw0, pw1 = paddings[4], paddings[5]
    if ph0 != ph1 or pw0 != pw1:
      raise RuntimeError('PADV2(op%d) 填充不对称 %r，PyTorch pad 是对称的'
                         % (op_idx, paddings))
    if ph0 >= fh or pw0 >= fw:
      raise RuntimeError('PADV2(op%d) 填充量 (%d,%d) >= 核 (%d,%d)，存在整窗都是填充的窗口'
                         % (op_idx, ph0, pw0, fh, fw))

    in_shape = get_shape(self.interpreter, data_t)
    pad_shape = get_shape(self.interpreter, out_t)
    pool_out_shape = get_shape(self.interpreter, pool.Outputs(0))
    if len(in_shape) != 4 or len(pad_shape) != 4 or len(pool_out_shape) != 4:
      raise RuntimeError('PADV2(op%d) 只覆盖 rank-4 张量: in=%s pad=%s pool=%s'
                         % (op_idx, in_shape, pad_shape, pool_out_shape))
    # 轴顺序按 NHWC 解释：paddings 的 4 组依次是 batch/H/W/C。
    # 轴顺序若错，下面的 shape 等式会对不上，因此这一条同时兜底轴顺序。
    expect_pad = [in_shape[0] + paddings[0] + paddings[1],
                  in_shape[1] + ph0 + ph1,
                  in_shape[2] + pw0 + pw1,
                  in_shape[3] + paddings[6] + paddings[7]]
    if list(pad_shape) != expect_pad:
      raise RuntimeError('PADV2(op%d) 输出形状 %s 与 in=%s + paddings=%r 不符（轴顺序应为 NHWC）'
                         % (op_idx, pad_shape, in_shape, paddings))
    expect_pool = [pad_shape[0],
                   (pad_shape[1] - fh) // sh + 1,
                   (pad_shape[2] - fw) // sw + 1,
                   pad_shape[3]]
    if list(pool_out_shape) != expect_pool:
      raise RuntimeError('MaxPool(op%d) 输出形状 %s 与 VALID 公式 %s 不符'
                         % (c_op, pool_out_shape, expect_pool))
    ev['in_shape'] = list(in_shape)
    ev['pad_shape'] = list(pad_shape)
    ev['pool_out_shape'] = list(pool_out_shape)
    ev['filter'] = [fh, fw]
    ev['stride'] = [sh, sw]
    ev['pad_h'] = [ph0, ph1]
    ev['pad_w'] = [pw0, pw1]
    ev['pad_out_tensor'] = int(out_t)
    ev['consumer_op_idx'] = int(c_op)
    ev['consumer'] = c_name

    self._padv2_seen += 1
    self.padv2_evidence.append(ev)
    return ('Pad', paddings)

  def _validate_padv2_subgraph(self, layers, start_layer, end_layer):
    """[阻断4] P3/P4 只能对**实际导出子图**判定。

    layers 只包含 [start_layer, end_layer] 内的算子；导出模型的公开输出是
    layers[-1]['out_idxes']（expose_output 时），不是整图的 graph.Outputs。
    因此"Pad 输出被别的东西观察"和"Pad 输出成为公开输出"都必须按子图重算。
    """
    if self._padv2_seen != len(self.padv2_evidence):
      raise RuntimeError('PADV2 计数不一致：见到 %d 个，证据 %d 条'
                         % (self._padv2_seen, len(self.padv2_evidence)))
    if self._padv2_seen and not self.padv2_evidence:
      raise RuntimeError('图中有 %d 个 PADV2，但证据为空' % self._padv2_seen)
    if not self.padv2_evidence:
      return
    in_range = set(self._layer_op_idxes)
    public = set()
    if self.expose_output and layers:
      public = {int(x) for x in layers[-1]['out_idxes']}
    for ev in self.padv2_evidence:
      out_t = ev['pad_out_tensor']
      if out_t in public:
        raise RuntimeError(
          'PADV2(op%d) 的输出 tensor %d 是导出子图的公开输出（end_layer=%d），拒绝降级'
          % (ev['op_idx'], out_t, end_layer))
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
      ev['checked_against'] = 'exported_subgraph'
'''

# ------------------------------------------------------------------ E7 quantize
QUANT = '''  def _quantize_checked(self, arr, tensor_idx, already_scaled=False):
    """写 msgpack 前的数值契约检查 + 显式舍入。

    两层检查分开写，避免把"主动保守策略"说成"原编码的必要条件"：
      L1 硬边界：值必须能用原 to_field 的偏置编码精确表示，即
         x in [-2^31, i64::MAX - 2^31]；x < -2^31 会被 as u64 回绕成 x+2^64，
         x > i64::MAX-2^31 则在 x + 2^31 处溢出 i64。
      L2 主动策略：进一步限制到 i32（I32_POLICY=True），因为下游 bias/累加/
         shift 的宿主算术没有逐项复核。这是**本项目的保守选择**，
         不是原 loader 的唯一有效域（原 loader 在 L1 全域精确）。

    比较一律在**整数域**做：float64 在 2^53 以上已不能精确表示每个整数，
    直接拿 float 跟 2^63 比会差几百到几千，边界判断就不可靠了。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
      return np.zeros(a.shape, dtype=np.int64)
    if not np.all(np.isfinite(a)):
      bad = int(np.size(a) - np.count_nonzero(np.isfinite(a)))
      raise RuntimeError('tensor %d 含 %d 个非有限值，拒绝写入 msgpack'
                         '(原实现会把它变成 INT64_MIN)' % (tensor_idx, bad))
    scaled = a if already_scaled else a * float(self.scale_factor)
    if not np.all(np.isfinite(scaled)):
      raise RuntimeError('tensor %d 乘 scale 后溢出 float64，拒绝' % tensor_idx)
    rounded = round_half_away_from_zero(scaled)
    # float64 只能精确表示到 2^53；超过就没有"整数"可谈，先拒绝
    if float(np.max(np.abs(rounded))) > float(1 << 53):
      raise RuntimeError('tensor %d 乘 scale 后绝对值超过 2^53，float64 已不能精确表示整数'
                         % tensor_idx)
    q = rounded.astype(np.int64)
    mx, mn = int(q.max()), int(q.min())
    if mn < I32_MIN or mx > (I64_MAX - (1 << 31)):
      raise RuntimeError(
        'tensor %d 乘 scale 后范围 [%d, %d] 超出原编码精确域 [2^31 负下界, i64::MAX-2^31]'
        % (tensor_idx, mn, mx))
    if I32_POLICY and (mn < I32_MIN or mx > I32_MAX):
      raise RuntimeError(
        'tensor %d 乘 scale 后范围 [%d, %d] 超出主动 i32 策略 [%d, %d]；'
        '这是保守选择，可通过 I32_POLICY=False 关闭后按 i64 全域处理'
        % (tensor_idx, mn, mx, I32_MIN, I32_MAX))
    return q

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

    # E1 全局 helper（含舍入与常量表）
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
""" + HELPERS + """
# 主动保守策略开关：把定点值限制在 i32 内。原 loader 的编码在
# [-2^31, i64::MAX-2^31] 上都是精确的，所以这不是必要条件，是策略选择。
I32_POLICY = True
""",
        "E1 helpers + I32_POLICY")

    # E2 图索引 + 证据/状态容器
    sub("""    self.graph = self.model.Subgraphs(0)
""",
        """    self.graph = self.model.Subgraphs(0)
    self._build_graph_maps()
    # P4-A v2：证据与子图状态统一挂在实例上（v1 的 to_dict 用了局部列表，
    # 导致 to_msgpack 导出的 self.padv2_evidence 永远为空）
    self.padv2_evidence = []
    self.padv2_evidence_out = None
    self.shared_metadata = {}
    self._layer_op_idxes = []
    self._padv2_seen = 0
    self._range = (0, 1 << 30)
""",
        "E2 graph maps + evidence state")

    # E3 不再无条件保留全部输入
    sub("""      # Keep the input tensors
      for input in handle_numpy_or_literal(op.InputsAsNumpy()):
        keep_tensors.add(input)
""",
        """      # P4-A：不再无条件保留全部输入。元数据张量（paddings / 常量 / axes）
      # 在算出 inp_idxes 之后再按数据依赖决定是否保留，见下面的 E4。
""",
        "E3 drop unconditional keep")

    # E4 按引用位置清理，并按过滤后的 inp_idxes 重算 keep_tensors
    sub("""      inp_idxes = get_inputs(op)
""",
        """      inp_idxes = get_inputs(op)
      # P4-A：元数据张量只从**本层**运行时依赖里去掉；若别处当数据用，
      # 它仍会经那个算子的 inp_idxes 进入 keep_tensors（不误删共享常量）
      _meta = self._metadata_tensor_idxes(op, self._op_name(op_idx))
      if _meta:
        inp_idxes = [i for i in inp_idxes if i not in _meta]
        print('dropping metadata inputs %s from layer %d (%s)'
              % (sorted(_meta), op_idx, self._op_name(op_idx)))
      for _i in inp_idxes:
        keep_tensors.add(_i)
""",
        "E4 recompute keep_tensors")

    # E5 PADV2 分支（证据写实例容器）
    sub("""      # Pad
      elif op_code == tflite.BuiltinOperator.PAD:
""",
        """      # Pad
      elif op_code == tflite.BuiltinOperator.PADV2:
        layer_type, params = self._convert_padv2(op, op_idx)
      elif op_code == tflite.BuiltinOperator.PAD:
""",
        "E5 padv2 branch")

    # E5b 记录每个 layer 对应的算子编号（只存内存，不进 msgpack，避免改 schema）
    sub("""      layers.append({
        'layer_type': layer_type,
""",
        """      self._layer_op_idxes.append(op_idx)
      layers.append({
        'layer_type': layer_type,
""",
        "E5b record layer op idx")

    # E6 每次 to_dict 重置子图状态
    sub("""    layers = []
    keep_tensors = set()
""",
        """    layers = []
    keep_tensors = set()
    # P4-A v2：每次调用重置，避免复用实例时证据串场
    self.padv2_evidence = []
    self.shared_metadata = {}
    self._layer_op_idxes = []
    self._padv2_seen = 0
    self._range = (start_layer, end_layer)
""",
        "E6 reset per call")

    # E7 主张量循环的数值检查
    sub("""      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = (tensor_data * self.scale_factor).round().astype(np.int64)
""",
        """      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = self._quantize_checked(tensor_data, tensor_idx)
""",
        "E7 checked quantize (main loop)")

    # E7b CUSTOM 权重走同一检查（已乘过 sf，用 already_scaled）
    sub("""        weights = np.transpose(weights, (3, 0, 1, 2))
        weights = (weights * self.scale_factor).round().astype(np.int64)
""",
        """        weights = np.transpose(weights, (3, 0, 1, 2))
        weights = self._quantize_checked(weights * self.scale_factor, op.Inputs(1),
                                        already_scaled=True)
""",
        "E7b checked quantize (custom weights)")

    # E7c DIV 的 adjusted_tensors 也走同一检查（值已含 sf）
    sub("""        adjusted_tensors[op.Inputs(1)] = np.array([(self.scale_factor / div_val).round().astype(np.int64)])
""",
        """        adjusted_tensors[op.Inputs(1)] = self._quantize_checked(
          np.asarray([self.scale_factor / div_val]), op.Inputs(1), already_scaled=True)
""",
        "E7c checked quantize (div adjusted)")

    # E8 方法体（METHODS + QUANT 都插进去；v1 漏了 QUANT）
    sub("""  def to_dict(self, start_layer, end_layer):
""",
        METHODS + QUANT + """  def to_dict(self, start_layer, end_layer):
""",
        "E8 methods + quantize")

    # E9 存活根 + 子图复检
    sub("""    # Get tensors
    print('keep tensors:', keep_tensors)
""",
        """    # P4-A：存活根 = 数据依赖 ∪ 模型输入 ∪ 整图输出 ∪ 导出子图公开输出
    for _inp in input_details:
      keep_tensors.add(_inp['index'])
    for _i in range(graph.OutputsLength()):
      keep_tensors.add(int(graph.Outputs(_i)))
    if self.expose_output and layers:
      for _o in layers[-1]['out_idxes']:
        keep_tensors.add(int(_o))

    # P4-A：[阻断4] P3/P4 对实际导出子图复检，必须在算完 out_idxes 之前
    self._validate_padv2_subgraph(layers, start_layer, end_layer)

    # Get tensors
    print('keep tensors:', keep_tensors)
""",
        "E9 liveness roots + subgraph validation")

    # E10 证据写盘：在打包之后，绑定 model/config 摘要
    sub("""  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
    model_packed = msgpack.packb(d, use_bin_type=True)
    d['tensors'] = []
    config_packed = msgpack.packb(d, use_bin_type=True)
    return model_packed, config_packed
""",
        """  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
    model_packed = msgpack.packb(d, use_bin_type=True)
    d['tensors'] = []
    config_packed = msgpack.packb(d, use_bin_type=True)
    # P4-A：[阻断2] 证据与这次转换的实际产物绑定：摘要 + 子图范围 + 条目数。
    # 证据来自 self.padv2_evidence（to_dict 内累积），不是另开一个局部列表。
    if getattr(self, 'padv2_evidence_out', None):
      import hashlib as _hashlib
      import json as _json
      if self._padv2_seen and not self.padv2_evidence:
        raise RuntimeError('图中有 %d 个 PADV2，但证据列表为空，拒绝写空证据'
                           % self._padv2_seen)
      payload = {
        'subgraph_range': list(self._range),
        'padv2_count': self._padv2_seen,
        'model_sha256': _hashlib.sha256(model_packed).hexdigest(),
        'config_sha256': _hashlib.sha256(config_packed).hexdigest(),
        'shared_metadata_tensors': {str(k): v for k, v in self.shared_metadata.items()},
        'evidence': self.padv2_evidence,
      }
      with open(self.padv2_evidence_out, 'w') as _f:
        _json.dump(payload, _f, indent=1, ensure_ascii=False)
    return model_packed, config_packed
""",
        "E10 evidence bound to artefacts")

    # E11 命令行暴露证据旁路参数
    sub("""  parser.add_argument('--num_randoms', type=int, default=20001)
""",
        """  parser.add_argument('--num_randoms', type=int, default=20001)
  parser.add_argument('--padv2_evidence', type=str, default=None,
                      help='P4-A: 把 PADV2 图级保护的检查证据写成 JSON')
""",
        "E11 argparse evidence")

    # E12 参数接到实例上
    sub("""  model_packed, config_packed = converter.to_msgpack(
""",
        """  converter.padv2_evidence_out = args.padv2_evidence
  model_packed, config_packed = converter.to_msgpack(
""",
        "E12 wire evidence arg")

    for stale in ("(tensor_data * self.scale_factor).round().astype(np.int64)",
                  "(weights * self.scale_factor).round().astype(np.int64)",
                  "(self.scale_factor / div_val).round().astype(np.int64)"):
        assert stale not in out, "残留旧量化路径: %s" % stale

    open(OUT_PY, "w", encoding="utf-8", newline="\n").write(out)
    diff = difflib.unified_diff(
        src.splitlines(keepends=True), out.splitlines(keepends=True),
        fromfile="zkml-main/python/converter.py", tofile="patches/converter_p4a_v2.py", n=3)
    open(OUT_DIFF, "w", encoding="utf-8", newline="\n").writelines(diff)

    # ---------------------------------------------------------- AST 门禁
    failures = verify(OUT_PY)
    print("已应用 %d 处编辑：" % len(applied))
    for a in applied:
        print("  -", a)
    print()
    print("AST 门禁：%s" % ("全部通过" if not failures else "失败 %d 项" % len(failures)))
    for f in failures:
        print("  !", f)
    print()
    print("fixed ->", OUT_PY, os.path.getsize(OUT_PY), "bytes")
    print("diff  ->", OUT_DIFF, os.path.getsize(OUT_DIFF), "bytes")
    return 1 if failures else 0


def verify(path):
    """把 v1 踩过的坑变成可自动重跑的门禁。"""
    fails = []
    tree = ast.parse(open(path, encoding="utf-8").read())

    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Converter"), None)
    if cls is None:
        return ["找不到 class Converter"]
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}

    # [阻断1] 被调用的方法必须真的存在
    called = set()
    for node in ast.walk(cls):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            called.add(node.func.attr)
    missing = sorted(called - methods - set(dir(object)))
    if missing:
        fails.append("[阻断1] self 上调用了不存在的方法: %s" % missing)

    # [阻断1] 三处量化路径都必须走 _quantize_checked
    if "_quantize_checked" not in methods:
        fails.append("[阻断1] 未定义 _quantize_checked")
    qc_calls = [n for n in ast.walk(cls)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_quantize_checked"]
    if len(qc_calls) != 3:
        fails.append("[阻断1] _quantize_checked 调用点应为 3 处（主循环/CUSTOM 权重/DIV），实际 %d"
                     % len(qc_calls))

    # [阻断1] 舍入必须真正被调用（不是只定义）
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "round_half_away_from_zero" for n in ast.walk(tree)):
        fails.append("[阻断1] round_half_away_from_zero 定义了但没有被调用")

    # [阻断2] to_dict 内不得再出现局部 padv2_evidence 赋值
    to_dict = next((n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "to_dict"), None)
    if to_dict is None:
        fails.append("[阻断2] 找不到 to_dict")
    else:
        for node in ast.walk(to_dict):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "padv2_evidence":
                        fails.append("[阻断2] to_dict 里仍有局部 padv2_evidence 赋值")
    # [阻断2] 证据必须来自实例属性，且带产物摘要
    to_msgpack = next((n for n in cls.body
                       if isinstance(n, ast.FunctionDef) and n.name == "to_msgpack"), None)
    if to_msgpack is None:
        fails.append("[阻断2] 找不到 to_msgpack")
    else:
        src_m = ast.get_source_segment(open(path, encoding="utf-8").read(), to_msgpack) or ""
        for token in ("self.padv2_evidence", "model_sha256", "config_sha256"):
            if token not in src_m:
                fails.append("[阻断2] to_msgpack 缺 %s" % token)

    # [阻断4] 子图级复检必须存在并在 to_dict 里被调用
    if "_validate_padv2_subgraph" not in methods:
        fails.append("[阻断4] 未定义 _validate_padv2_subgraph")
    else:
        if "_validate_padv2_subgraph" not in called:
            fails.append("[阻断4] _validate_padv2_subgraph 定义了但没被调用")
        if "_static_constant" not in methods:
            fails.append("[阻断4] 未定义 _static_constant（无法证明元数据是静态常量）")
    return fails


if __name__ == "__main__":
    sys.exit(main())
