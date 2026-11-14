import argparse
import ast
from typing import Literal, Union
import tensorflow as tf
import numpy as np
import tflite
import msgpack

def get_shape(interpreter: tf.lite.Interpreter, tensor_idx):
  if tensor_idx == -1:
    return []
  tensor = interpreter.get_tensor(tensor_idx)
  return list(tensor.shape)

def handle_numpy_or_literal(inp: Union[np.ndarray, Literal[0]]):
  if isinstance(inp, int):
    return np.array([inp])
  return inp

def get_inputs(op: tflite.Operator):
  idxes = handle_numpy_or_literal(op.InputsAsNumpy())
  idxes = idxes.tolist()
  idxes = list(filter(lambda x: x != -1, idxes))
  return idxes

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

class Converter:
  def __init__(
      self, model_path, scale_factor, k, num_cols, num_randoms, use_selectors, commit,
      expose_output
    ):
    self.model_path = model_path
    self.scale_factor = scale_factor
    self.k = k
    self.num_cols = num_cols
    self.num_randoms = num_randoms
    self.use_selectors = use_selectors
    self.commit = commit
    self.expose_output = expose_output
    self.padv2_evidence = []
    self.padv2_evidence_out = None

    self.interpreter = tf.lite.Interpreter(
      model_path=self.model_path,
      experimental_preserve_all_tensors=True
    )
    self.interpreter.allocate_tensors()

    with open(self.model_path, 'rb') as f:
      buf = f.read()
      self.model = tflite.Model.GetRootAsModel(buf, 0)
    self.graph = self.model.Subgraphs(0)
    self._build_graph_maps()


  def valid_activations(self):
    return [
      tflite.ActivationFunctionType.NONE,
      tflite.ActivationFunctionType.RELU,
      tflite.ActivationFunctionType.RELU6,
    ]

  def _convert_add(self, op: tflite.Operator, generated_tensors: set):
    # Get params
    op_opt = op.BuiltinOptions()
    if op_opt is None:
      raise RuntimeError('Add options is None')
    opt = tflite.AddOptions()
    opt.Init(op_opt.Bytes, op_opt.Pos)
    params = [opt.FusedActivationFunction()]

    # Get inputs
    inputs = get_inputs(op)
    print(generated_tensors)
    print('Add inputs: ', inputs)
    if len(inputs) != 2:
      raise RuntimeError('Add must have 2 inputs')

    # If both tensors are generated, do nothing
    print(inputs[0] in generated_tensors, inputs[1] in generated_tensors)
    if (inputs[0] in generated_tensors) and (inputs[1] in generated_tensors):
      return ('Add', params)

    nb_generated = (inputs[0] in generated_tensors) + (inputs[1] in generated_tensors)
    if nb_generated != 1:
      raise RuntimeError('Add must have 1 generated tensor')

    # Check if there are any negative infinities
    const_tensor = self.interpreter.get_tensor(inputs[0]) if inputs[0] not in generated_tensors else self.interpreter.get_tensor(inputs[1])
    if np.any(const_tensor == -np.inf):
      # Ensure that the constant tensor is all -inf and 0
      if not np.all(np.logical_or(np.isneginf(const_tensor), const_tensor == 0)):
        raise RuntimeError('Add constant tensor must be -inf and 0 only')
      mask = (const_tensor == -np.inf).astype(np.int64)
      params = [len(mask.shape)] + list(mask.shape)
      params += mask.flatten().tolist()
      return ('MaskNegInf', params)
    else:
      return ('Add', params)



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

  def to_dict(self, start_layer, end_layer):
    interpreter = self.interpreter
    model = self.model
    graph = self.graph
    if graph is None:
      raise RuntimeError('Graph is None')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp_detail in input_details:
      inp = np.zeros(inp_detail['shape'], dtype=inp_detail['dtype'])
      interpreter.set_tensor(inp_detail['index'], inp)
    # for i, inp in enumerate(inps):
    #   interpreter.set_tensor(input_details[i]['index'], inp)
    interpreter.invoke()

    # Get layers
    generated_tensor_idxes = set()
    for inp in input_details:
      generated_tensor_idxes.add(inp['index'])

    layers = []
    keep_tensors = set()
    padv2_evidence = []
    adjusted_tensors = {}
    for op_idx in range(graph.OperatorsLength()):
      op = graph.Operators(op_idx)
      if op is None:
        raise RuntimeError('Operator is None')
      model_opcode = model.OperatorCodes(op.OpcodeIndex())
      if model_opcode is None:
        raise RuntimeError('Operator code is None')
      op_code = model_opcode.BuiltinCode()

      # Skip generated tensors
      for output in handle_numpy_or_literal(op.OutputsAsNumpy()):
        generated_tensor_idxes.add(output)

      if op_idx < start_layer:
        continue
      if op_idx > end_layer:
        break

      # P4-A：不再无条件保留全部输入。元数据张量（paddings / 常量 / axes）
      # 在算出 inp_idxes 之后再按数据依赖决定是否保留，见下面的 E4。

      # AvgPool2D
      if op_code == tflite.BuiltinOperator.AVERAGE_POOL_2D:
        layer_type = 'AveragePool2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('AvgPool2D options is None')
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      elif op_code == tflite.BuiltinOperator.MAX_POOL_2D:
        layer_type = 'MaxPool2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('MaxPool2D options is None')
        opt = tflite.Pool2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.Padding() == tflite.Padding.SAME:
          raise NotImplementedError('SAME padding is not supported')
        if opt.FusedActivationFunction() != tflite.ActivationFunctionType.NONE:
          raise NotImplementedError('Fused activation is not supported')
        params = [opt.FilterHeight(), opt.FilterWidth(), opt.StrideH(), opt.StrideW()]
      # FIXME: hack for Keras... not sure why this isn't being converted properly
      elif op_code == tflite.BuiltinOperator.CUSTOM:
        layer_type = 'Conv2D'
        activation = 0
        weights = self.interpreter.get_tensor(op.Inputs(1))
        weights = np.transpose(weights, (3, 0, 1, 2))
        weights = (weights * self.scale_factor).round().astype(np.int64)
        adjusted_tensors[op.Inputs(1)] = weights
        params = [0, 1, activation, 1, 1]
      # Conv2D
      elif op_code == tflite.BuiltinOperator.CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Conv2D options is None')
        opt = tflite.Conv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError('Unsupported activation function at layer {op_idx}')
        # 0 is Conv2D
        params = \
          [0] + \
          [opt.Padding()] + \
          [opt.FusedActivationFunction()] + \
          [opt.StrideH(), opt.StrideW()]
      # DepthwiseConv2D
      elif op_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
        layer_type = 'Conv2D'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('DepthwiseConv2D options is None')
        opt = tflite.DepthwiseConv2DOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
          raise NotImplementedError('Dilation is not supported')
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError('Unsupported activation function at layer {op_idx}')
        # 1 is DepthwiseConv2D
        params = \
          [1] + \
          [opt.Padding()] + \
          [opt.FusedActivationFunction()] + \
          [opt.StrideH(), opt.StrideW()]
      # Fully connected
      elif op_code == tflite.BuiltinOperator.FULLY_CONNECTED:
        layer_type = 'FullyConnected'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Fully connected options is None')
        opt = tflite.FullyConnectedOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.FusedActivationFunction() not in self.valid_activations():
          raise NotImplementedError(f'Unsupported activation function at layer {op_idx}')
        params = [opt.FusedActivationFunction()]
      elif op_code == tflite.BuiltinOperator.BATCH_MATMUL:
        layer_type = 'BatchMatMul'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('BatchMatMul options is None')
        opt = tflite.BatchMatMulOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.AdjX() is True: raise NotImplementedError('AdjX is not supported')
        params = [int(opt.AdjX()), int(opt.AdjY())]

      ## Arithmetic
      # Add
      elif op_code == tflite.BuiltinOperator.ADD:
        layer_type, params = self._convert_add(op, generated_tensor_idxes)
      # Mul
      elif op_code == tflite.BuiltinOperator.MUL:
        layer_type = 'Mul'
        params = []
      # Sub
      elif op_code == tflite.BuiltinOperator.SUB:
        sub_val = interpreter.get_tensor(op.Inputs(1))
        # TODO: this is a bit of a hack
        if np.any(np.isin(sub_val, 10000)):
          layer_type = 'MaskNegInf'
          mask = (sub_val == 10000).astype(np.int64)
          params = [len(mask.shape)] + list(mask.shape)
          params += mask.flatten().tolist()
        else:
          layer_type = 'Sub'
          params = []
      # Div
      elif op_code == tflite.BuiltinOperator.DIV:
        # Implement division as multiplication by the inverse
        layer_type = 'Mul'
        div_val = interpreter.get_tensor(op.Inputs(1))
        if type(div_val) != np.float32: raise NotImplementedError('Only support one divisor')
        adjusted_tensors[op.Inputs(1)] = np.array([(self.scale_factor / div_val).round().astype(np.int64)])
        params = []
      # Pad
      elif op_code == tflite.BuiltinOperator.PADV2:
        layer_type, params = self._convert_padv2(op, get_inputs(op), padv2_evidence)
      elif op_code == tflite.BuiltinOperator.PAD:
        layer_type = 'Pad'
        tensor_idx = op.Inputs(1)
        tensor = interpreter.get_tensor(tensor_idx).flatten().astype(np.int64)
        params = tensor.tolist()
      # Softmax
      elif op_code == tflite.BuiltinOperator.SOFTMAX:
        layer_type = 'Softmax'
        # TODO: conditionally determine whether or not to subtract the max
        # It should depend on the input to the softmax
        if layers[-1]['layer_type'] == 'MaskNegInf':
          params = layers[-1]['params']
        elif layers[-2]['layer_type'] == 'MaskNegInf':
          params = layers[-2]['params']
          params = [params[0] - 1] + params[2:]
        else:
          params = []
      # Mean
      elif op_code == tflite.BuiltinOperator.MEAN:
        layer_type = 'Mean'
        inp_shape = interpreter.get_tensor(op.Inputs(0)).shape
        mean_idxes = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64)
        if len(mean_idxes) + 2 != len(inp_shape):
          raise NotImplementedError(f'Only mean over all but one axis is supported: {op_idx}')
        params = mean_idxes.tolist()
      elif op_code == tflite.BuiltinOperator.SQUARE:
        layer_type = 'Square'
        params = []
      # Squared difference
      elif op_code == tflite.BuiltinOperator.SQUARED_DIFFERENCE:
        layer_type = 'SquaredDifference'
        params = []

      # Pointwise
      elif op_code == tflite.BuiltinOperator.RSQRT:
        layer_type = 'Rsqrt'
        params = []
      elif op_code == tflite.BuiltinOperator.LOGISTIC:
        layer_type = 'Logistic'
        params = []
      elif op_code == tflite.BuiltinOperator.TANH:
        layer_type = 'Tanh'
        params = []
      elif op_code == tflite.BuiltinOperator.POW:
        layer_type = 'Pow'
        power = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.float32)
        if power != 3.: raise NotImplementedError(f'Only support power 3')
        power = power.round().astype(np.int64)
        if len(power) != 1: raise NotImplementedError(f'Only scalar power is supported: {op_idx}')
        params = power.tolist()

      # The following are no-ops in the sense that they don't change the tensor
      # However, we need to pass along the right tensors
      # The param says which input to pass along
      elif op_code == tflite.BuiltinOperator.SHAPE:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.GATHER:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.REDUCE_PROD:
        # TODO: not sure if this is in general a no-op
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.STRIDED_SLICE:
        # FIXME: this is not in general a no-op
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.BROADCAST_ARGS:
        layer_type = 'Noop'
        params = [0]
      elif op_code == tflite.BuiltinOperator.BROADCAST_TO:
        layer_type = 'Noop'
        params = [0]

      ## Shape
      elif op_code == tflite.BuiltinOperator.RESHAPE:
        layer_type = 'Reshape'
        params = []
      elif op_code == tflite.BuiltinOperator.TRANSPOSE:
        layer_type = 'Transpose'
        params = get_shape(interpreter, op.Inputs(0)) + interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
      elif op_code == tflite.BuiltinOperator.CONCATENATION:
        # FIXME: This is not in general a no-op
        layer_type = 'Concatenation'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Concatenation options is None')
        opt = tflite.ConcatenationOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.Axis()]
      elif op_code == tflite.BuiltinOperator.PACK:
        layer_type = 'Pack'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Pack options is None')
        opt = tflite.PackOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        params = [opt.Axis()]
        if params[0] > 1: raise NotImplementedError(f'Only axis=0,1 supported at layer {op_idx}')
      elif op_code == tflite.BuiltinOperator.SPLIT:
        layer_type = 'Split'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('Split options is None')
        opt = tflite.SplitOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        axis = interpreter.get_tensor(op.Inputs(0)).flatten().astype(np.int64)[0]
        num_splits = opt.NumSplits()
        inp = interpreter.get_tensor(op.Inputs(1))
        if inp.shape[axis] % num_splits != 0:
          raise NotImplementedError(f'Only equal splits supported at layer {op_idx}')
        params = [int(axis), num_splits]
      elif op_code == tflite.BuiltinOperator.SLICE:
        layer_type = 'Slice'
        begin = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
        size = interpreter.get_tensor(op.Inputs(2)).flatten().astype(np.int64).tolist()
        params = begin + size
      elif op_code == tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR:
        layer_type = 'ResizeNearestNeighbor'
        op_opt = op.BuiltinOptions()
        if op_opt is None:
          raise RuntimeError('ResizeNearestNeighbor options is None')
        opt = tflite.ResizeNearestNeighborOptions()
        opt.Init(op_opt.Bytes, op_opt.Pos)
        if opt.AlignCorners():
          raise NotImplementedError(f'Align corners not supported at layer {op_idx}')
        if not opt.HalfPixelCenters():
          raise NotImplementedError(f'Half pixel centers not supported at layer {op_idx}')
        # Can take the out shape directly from the tensor
        params = [int(opt.AlignCorners()), int(opt.HalfPixelCenters())]

      # Not implemented
      else:
        op_name = None
        for attr in dir(tflite.BuiltinOperator):
          if not attr.startswith('__'):
            if getattr(tflite.BuiltinOperator, attr) == op_code:
              op_name = attr
        raise NotImplementedError('Unsupported operator at layer {}: {}, {}'.format(op_idx, op_code, op_name))

      inp_idxes = get_inputs(op)
      # P4-A：元数据张量不进运行时依赖，也不进 keep_tensors
      _meta = self._metadata_tensor_idxes(op, self._op_name(op_idx))
      if _meta:
        inp_idxes = [i for i in inp_idxes if i not in _meta]
        print('dropping metadata inputs %s from layer %d (%s)'
              % (sorted(_meta), op_idx, self._op_name(op_idx)))
      for _i in inp_idxes:
        keep_tensors.add(_i)
      # FIXME: hack for testing
      rsqrt_overflows = [99, 158, 194, 253, 289, 348]
      if op_idx in rsqrt_overflows:
        if op_code == tflite.BuiltinOperator.RSQRT:
          mask = [0, 1]
        else:
          mask = []
      else:
        mask = []
      layers.append({
        'layer_type': layer_type,
        'inp_idxes': inp_idxes,
        'inp_shapes': [get_shape(interpreter, inp_idx) for inp_idx in inp_idxes],
        'out_idxes': [op.Outputs(i) for i in range(op.OutputsLength())],
        'out_shapes': [get_shape(interpreter, op.Outputs(i)) for i in range(op.OutputsLength())],
        'params': params,
        'mask': mask,
      })
    print(layers)
    print()


    # Get tensors
    print('keep tensors:', keep_tensors)
    tensors = []
    for tensor_idx in range(graph.TensorsLength()):
      if tensor_idx not in keep_tensors:
        continue

      tensor = graph.Tensors(tensor_idx)
      if tensor is None:
        raise NotImplementedError('Tensor is None')

      if tensor_idx in generated_tensor_idxes:
        print(f'skipping generated tensor: {format(tensor_idx)}, {tensor.Name()}')
        continue

      shape = []
      for i in range(tensor.ShapeLength()):
        shape.append(int(tensor.Shape(i)))
      if shape == []:
        shape = [1]

      tensor_data = interpreter.get_tensor(tensor_idx)
      if tensor.Type() == tflite.TensorType.FLOAT32:
        tensor_data = self._quantize_checked(tensor_data, tensor_idx)
      elif tensor.Type() == tflite.TensorType.INT32:
        tensor_data = tensor_data.astype(np.int64)
      elif tensor.Type() == tflite.TensorType.INT64:
        continue
      else:
        raise NotImplementedError('Unsupported tensor type: {}'.format(tensor.Type()))

      if tensor_idx in adjusted_tensors:
        tensor_data = adjusted_tensors[tensor_idx]
        shape = tensor_data.shape

      tensors.append({
        'idx': tensor_idx,
        'shape': shape,
        'data': tensor_data.flatten().tolist(),
      })
      # print(tensor_idx, tensor.Type(), tensor.Name(), tensors[-1]['shape'])
      # print(np.abs(tensor_data).max())

    commit_before = []
    commit_after = []
    if self.commit:
      input_tensors = [inp['index'] for inp in input_details]
      weight_tensors = [tensor['idx'] for tensor in tensors if tensor['idx'] not in input_tensors]
      commit_before = [weight_tensors, input_tensors]

      output_tensors = [out['index'] for out in output_details]
      commit_after = [output_tensors]

    out_idxes = layers[-1]['out_idxes'] if self.expose_output else []
    d = {
      'global_sf': self.scale_factor,
      'k': self.k,
      'num_cols': self.num_cols,
      'num_random': self.num_randoms,
      'inp_idxes': [inp['index'] for inp in input_details],
      # 'out_idxes': [out['index'] for out in output_details],
      'out_idxes': out_idxes,
      'layers': layers,
      'tensors': tensors,
      'use_selectors': self.use_selectors,
      'commit_before': commit_before,
      'commit_after': commit_after,
    }
    print()
    print(d['layers'][-1])
    # d['out_idxes'] = [14]
    print(d.keys())
    print(d['out_idxes'])
    return d

  def to_msgpack(self, start_layer, end_layer, use_selectors=True):
    d = self.to_dict(start_layer, end_layer)
    if getattr(self, 'padv2_evidence_out', None):
      import json as _json
      with open(self.padv2_evidence_out, 'w') as _f:
        _json.dump(getattr(self, 'padv2_evidence', []), _f, indent=1)
    model_packed = msgpack.packb(d, use_bin_type=True)
    d['tensors'] = []
    config_packed = msgpack.packb(d, use_bin_type=True)
    return model_packed, config_packed


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument('--model_output', type=str, required=True)
  parser.add_argument('--config_output', type=str, required=True)
  parser.add_argument('--scale_factor', type=int, default=2**16)
  parser.add_argument('--k', type=int, default=19)
  parser.add_argument('--eta', type=float, default=0.001)
  parser.add_argument('--num_cols', type=int, default=6)
  parser.add_argument('--use_selectors', action=argparse.BooleanOptionalAction, required=False, default=True)
  parser.add_argument('--commit', action=argparse.BooleanOptionalAction, required=False, default=False)
  parser.add_argument('--expose_output', action=argparse.BooleanOptionalAction, required=False, default=True)
  parser.add_argument('--start_layer', type=int, default=0)
  parser.add_argument('--end_layer', type=int, default=10000)
  parser.add_argument('--num_randoms', type=int, default=20001)
  parser.add_argument('--padv2_evidence', type=str, default=None,
                      help='P4-A: 把 PADV2 图级保护的检查证据写成 JSON')
  args = parser.parse_args()

  converter = Converter(
    args.model,
    args.scale_factor,
    args.k,
    args.num_cols,
    args.num_randoms,
    args.use_selectors,
    args.commit,
    args.expose_output,
  )

  converter.padv2_evidence_out = args.padv2_evidence
  model_packed, config_packed = converter.to_msgpack(
    start_layer=args.start_layer,
    end_layer=args.end_layer,
  )
  if model_packed is None:
    raise Exception('Failed to convert model')

  with open(args.model_output, 'wb') as f:
    f.write(model_packed)
  with open(args.config_output, 'wb') as f:
    f.write(config_packed)

if __name__ == '__main__':
  main()
