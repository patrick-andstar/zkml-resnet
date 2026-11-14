# 04 · P3：标准 ResNet-18 的转换与算子/布局门禁

> 本批只做「固定模型与输入身份 + 隔离导出环境 + 浮点导出 + 静态算子/布局/数值检查」。
> 不预设目标 k / 列数 / scale，不生成 SRS，不跑 MockProver 或 prove，不改标准结构。
> 因此本文任何结果都**不能**被读成「ResNet 证明成功」。

---

## 30 秒速览

固定 torchvision 标准 1000 类 ResNet-18（`IMAGENET1K_V1`，checkpoint `sha256=f37072fd…`，11,689,512 参数），在隔离 conda 环境里用 `torch.export → litert_torch` 导出 **NHWC FP32 TFLite**（46,789,576 B，0.593 GMAC）。5 个输入上对 PyTorch 的最大绝对误差 **3.43e-05**、argmax 全一致；把 transpose 故意写成 reshape 的负向对照误差 **22.28**、top1 从 258 掉到 825，说明这道门禁确实能抓到布局错误。36 个算子逐条审计 19 项全过。

一个原先担心的阻塞点被证伪：**maxpool 用的不是 SAME**，而是「显式 `PADV2(-inf)` + `VALID`」，所以仓库 `converter.py` 里那条 `SAME padding is not supported` 根本不会触发。真正挡住它的是 **`PADV2` 这个算子没有分支**——原样运行停在 `layer 2: 60, PADV2`。补一个最小分支后 36 层全部产出，与 TFLite 侧逐层形状/参数**完全一致（36/36）**。

但补丁把 **-inf 填充降级成了 Rust `PadChip` 硬编码的 0 填充**，只在「该张量已被 ReLU 保证非负」时才等价——本模型满足（它的生产者是 conv1+BN+ReLU），这条等价条件必须写进报告、不能被默认掉。定点误差、k/scale、SRS、prove 全部未做。

---

## 1 本批做了什么、没做什么

| | 内容 | 状态 |
|---|---|---|
| 做 | 模型身份固定（权重文件 + SHA-256 + 参数量 + 类别数） | ✅ |
| 做 | 隔离 Python 导出环境，不与 Rust 构建链混用 | ✅ |
| 做 | PyTorch → TFLite 浮点导出，NHWC I/O | ✅ |
| 做 | 预登记容差 + 5 输入数值门禁 + 布局负向对照 | ✅ |
| 做 | 36 个算子逐条 padding/stride/dilation/fused activation 审计 | ✅ |
| 做 | 与仓库 `converter.py` 的兼容性实测（原样跑 + 最小补丁探针） | ✅ |
| 做 | TFLite ↔ converter msgpack 的逐层交叉校验 | ✅ |
| 不做 | 定点（scale）误差、Rust 侧数值对齐 | ❌ 属 P4 |
| 不做 | k / 列数 / scale 的确定 | ❌ 属 P4 |
| 不做 | SRS 生成、MockProver、`create_proof`、`verify_proof` | ❌ 属 P4/P5 |
| 不做 | 数据集精度评估（因此不宣称「精度无损」） | ❌ 无数据集验证 |

---

## 2 环境（隔离，不与 Rust 链混用）

| 组件 | 版本 |
|---|---|
| Python | 3.11.18（conda，`env/py-export`） |
| torch | 2.14.1+cpu |
| torchvision | 0.29.1+cpu |
| litert-torch | 0.10.0 |
| ai-edge-litert | 2.3.0 |
| tflite（flatbuffer schema） | 2.18.0 |
| 磁盘 | `/root/autodl-tmp` 50G，导出后余 46G |

**没有装 TensorFlow。** 理由有两条：一是 plan.md 要求「不同时盲装多条链」；二是 TensorFlow 只为 `converter.py` 顶部那一个 `tf.lite.Interpreter` 符号，装整套约 +600 MB，对「读图 + 取张量」这件事没有额外信息量。改用 `ai-edge-litert` 的 `Interpreter`（它同样支持 `experimental_preserve_all_tensors`）做一层**只转发符号、不改逻辑**的兼容层：

```python
# env/shim/tensorflow.py（全文 12 行）
from ai_edge_litert.interpreter import Interpreter
class _Lite:
    Interpreter = Interpreter
lite = _Lite()
```

这一层必须披露：`converter.py` 用到的接口只有 `allocate_tensors / get_input_details / get_output_details / set_tensor / invoke / get_tensor`，转发是忠实的；但**它不是官方 TensorFlow**，若日后出现与 TF 行为不一致的疑点，要回到真 TF 复核，不能拿这层当等价证明。

---

## 3 模型与输入身份

| 项 | 值 |
|---|---|
| 族 | `torchvision.models.resnet18`，标准 18 层、1000 类 |
| 权重 | `ResNet18_Weights.IMAGENET1K_V1` |
| checkpoint `sha256` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`（661,378 B 之外另有权重文件，按文件名哈希前缀 `f37072fd` 核对一致） |
| 参数量 | 11,689,512 |
| 模式 | `eval()`（BN 用 running stats） |
| 输入 | NCHW `[1,3,128,128]` FP32 |
| 输出 | `[1,1000]` logits（无 softmax） |

**身份边界（必须与旧路线区分）**：这是**独立基线**——标准 torchvision 权重 + 明确指定的预处理。没有访问被禁止的旧项目目录，因此**不能**声称与 ZKTorch 路线「完全同模型同输入」；将来要做严格横向比较，需要用户把相同权重与输入的副本/摘要放进本项目。

### 5 个测试输入（≥3 个，含可追溯图像与边界）

| 名称 | 定义 | 取值范围 | `sha256`（前 12） | 参考 top1 |
|---|---|---|---|---|
| `real_dog` | 真实 JPEG 缩放至 128×128（`p3-dog.jpg`，`sha256=f3f87bb8…`） | [-2.084, 2.640] | `b03cdd3b91b8` | 258（Samoyed） |
| `zeros` | 全 0 | 0 | `f48685eec604` | 892 |
| `neg_const` | 全 -1.5（**负值边界**） | -1.5 | `afd535a518de` | 644 |
| `noise_seed1234` | 高斯噪声 `seed=1234`，σ=1 | [-3.000, 3.000] | `671e5d5ddcfe` | 971 |
| `half_split` | 左半 +2 / 右半 -2（**空间边界**） | [-2, 2] | `2e1807c318b8` | 916 |

`neg_const` 是特意加的：它会一路产生负的被卷积量，用来检验「零填充 / -inf 填充」这类分歧。参考 logits 全部落盘（`assets/p3-inputs/ref_*.npy`），可独立复算。

---

## 4 导出路线（两次尝试，只采纳一次）

```
torchvision resnet18(IMAGENET1K_V1) → torch.export → litert_torch.convert → .tflite (FP32)
```

| 尝试 | 做法 | 输入形状 | 大小 | 结果 |
|---|---|---|---|---|
| 1 | `litert_torch.convert(module, args)`，默认 | `[1,3,128,128]` NCHW | 46,776,576 B | **不采纳**：I/O 是 NCHW，仓库自带的 TFLite 产物是 NHWC，不是 drop-in |
| 2 | 加 `litert_torch.to_channel_last_io(module, args=[0])` | `[1,128,128,3]` NHWC | 46,789,576 B | **采纳** |

采纳版本 `sha256=4014a8b4cde23d61e328114cf67a2247a766e812b2195d820a49fc13dd4cb4a1`。

两个必须记录的现象：

1. **`Your model is converted in training mode` 警告**：即使先 `model.eval()`，`to_channel_last_io` 的 wrapper 会丢掉 eval 状态，转换器因此按 training 模式提示。它不是「已确保 eval」的证据，只是一个提示；BN 到底折没折、折得对不对，靠第 5 节的数值门禁回答（若 BN 用了 batch 统计量，在 batch=1 下 logits 会整体崩掉，误差量级会是 1e0 而非 1e-5）。
2. **`aten.permute` 出现在图里，但 TFLite 里没有 `TRANSPOSE` 算子**：说明 channel-last 改写把 permute 消解掉了。这件事不能只看算子表就断言「对」，仍由数值门禁背书（第 5 节的布局负向对照正是为此设计）。

---

## 5 数值门禁（阈值预登记，事后不放宽）

预登记文件 `assets/p3-prereg.json`，写入时间 `2026-11-14T12:40:26Z`，**早于**转换（`12:42:08Z` 开始）。规则原文：阈值现在固定，落在 `pass_with_note` 就如实标注，事后不放宽、失败不重试更松的界。

| 档 | 判据 |
|---|---|
| PASS | `max_abs ≤ 1e-3` 且各输入 argmax 一致 |
| PASS_WITH_NOTE | `max_abs ≤ 1e-2` 且各输入 argmax 一致 |
| FAIL | `max_abs > 1e-2`，或任一输入 argmax 不同 |

### 结果（TFLite 侧输入由 NCHW 转置为 NHWC 后送入）

| 输入 | max_abs | RMSE | top1（PyTorch / TFLite） | 判 |
|---|---|---|---|---|
| real_dog | **3.433e-05** | 6.265e-06 | 258 / 258 | ok |
| zeros | 1.168e-05 | 3.942e-06 | 892 / 892 | ok |
| neg_const | 6.437e-06 | 2.087e-06 | 644 / 644 | ok |
| noise_seed1234 | 6.676e-06 | 1.964e-06 | 971 / 971 | ok |
| half_split | 1.395e-05 | 3.952e-06 | 916 / 916 | ok |

**VERDICT: PASS**（最差 real_dog 3.433e-05，比 1e-3 还低一个半数量级；没有任何输入落在 `pass_with_note`）。

### 布局负向对照（证明门禁不是走过场）

把「NCHW→NHWC」故意写成 `reshape` 而不是 `transpose`：

| | max_abs | top1 |
|---|---|---|
| 正确（transpose） | 3.433e-05 | 258 |
| 错误（reshape） | **2.228e+01** | **825** |

错误做法的误差比阈值大 2200 倍，top1 从 Samoyed 掉到 825。**这道门禁对布局错误是灵敏的**，PASS 才有意义。

### 不能由本节推出的结论

- 不能推「精度无损」：只测了 5 个输入，没有数据集精度。
- 不能推「定点也对」：这里全是 FP32，定点（scale）误差是另一件事，属 P4。
- 不能推「Rust 侧算得对」：本节只覆盖 PyTorch ↔ TFLite。

---

## 6 算子清单与逐条语义审计

### 6.1 算子直方图（36 个）

```
CONV_2D 20 · ADD 8 · PAD 4 · PADV2 1 · MAX_POOL_2D 1 · MEAN 1 · FULLY_CONNECTED 1
```

对照 `torch.export` 的 aten 直方图：

| aten | 次数 | 在 TFLite 里的去向 |
|---|---|---|
| `conv2d` | 20 | `CONV_2D` 20 |
| `batch_norm` | 20 | **消失**（eval 下折叠进 conv 权重/偏置） |
| `relu_` | **17** | **全部融合**（9 进 `CONV_2D.fused_activation`，8 进 `ADD.fused_activation`） |
| `add_.Tensor` | 8 | `ADD` 8 |
| `max_pool2d` | 1 | `PADV2 + MAX_POOL_2D` |
| `adaptive_avg_pool2d` | 1 | `MEAN` 1 |
| `linear` | 1 | `FULLY_CONNECTED` 1 |
| `permute` | 1 | **消失**（channel-last 改写消解） |
| `flatten` | 1 | 形状变化，无独立算子 |

17 个 ReLU 拆分校验：9（conv 融合）+ 8（add 融合）= 17，与 ResNet-18 的「1 个 stem + 4 层 × 2 block × 2」完全对齐。这是「融合没有多吃也没有漏掉」的硬计数。

### 6.2 19 项检查（脚本 `scripts/p3_opaudit.py`，结果 `assets/p3-opaudit.json`）

| # | 检查 | 结果 |
|---|---|---|
| C1 | I/O 为 NHWC、FP32、`[1,128,128,3] → [1,1000]` | ✅ |
| C2 | 无量化：无 int8/uint8（83 个 tensor = 80 FLOAT32 + 3 INT32 索引常量） | ✅ |
| C3 | stride2 卷积：核>1 一律 VALID + 显式 `pad=k//2`；核=1 无 pad | ✅ |
| C4 | stem 7×7 用显式 `PAD=3`（SAME 只会给 pad_before=2，左右不对称） | ✅ |
| C5 | maxpool 3×3/s2 = 显式 PAD(1) + VALID + 无 fused activation | ✅ |
| C6 | stride1 的 3×3 卷积全部 SAME、无显式 pad、核为奇数 | ✅ |
| C7 | 所有卷积 dilation=1 | ✅ |
| C8 | 无独立 RELU/RELU6 算子 | ✅ |
| C9 | ReLU 融合点 = 9 + 8 = 17 | ✅ |
| C10 | 8 个残差 ADD 两侧同形（无广播），无 -inf mask 常量 | ✅ |
| C11 | 全局池化 = `MEAN(axes={1,2})`，`[1,4,4,512] → [1,512]` | ✅ |
| C12 | FC 权重 `[1000,512]`、无激活、输出 `[1,1000]` | ✅ |
| C13 | 无 CUSTOM / 控制流算子 | ✅ |
| C14 | 填充值符合语义：PAD=0；**maxpool 前的 PADV2 = -inf** | ✅ |
| C14b | 每个显式填充量 = 相邻核的 `k//2` | ✅ |
| C15 | 0 填充的语义安全性（见 6.4） | ✅ |
| C16 | TRANSPOSE 记录项：**0 个** | 记录项 |
| C17 | `CONV_2D=20`（1 stem + 16 block + 3 downsample），无 depthwise | ✅ |
| C18 | 无 BN 残留（无 `BATCH_MATMUL`/`MUL` 形式的归一化） | ✅ |

**C16 是记录项不是判据**（TRANSPOSE 数为 0，检查恒真）；布局正确性由 C1 + 第 5 节负向对照承担，不由 C16 承担。C15 对卷积部分也是「定义上成立」而非「测出来成立」，它真正的判别力在 maxpool 那一条。

### 6.3 完整算子表（36 行，`pad` 取 TFLite 枚举，`act` 为融合激活）

| op | 类型 | pad | stride | 核 | 输入 → 输出 | act |
|---|---|---|---|---|---|---|
| 0 | PAD | – | – | – | `[1,128,128,3] → [1,134,134,3]`，paddings `[0,0,3,3,3,3,0,0]` | – |
| 1 | CONV_2D | VALID | 2 | 7×7 | `[1,134,134,3] → [1,64,64,64]` | RELU |
| 2 | PADV2 | – | – | – | `[1,64,64,64] → [1,66,66,64]`，paddings `[0,0,1,1,1,1,0,0]`，**constant=-inf** | – |
| 3 | MAX_POOL_2D | VALID | 2 | 3×3 | `[1,66,66,64] → [1,32,32,64]` | NONE |
| 4 | CONV_2D | SAME | 1 | 3×3 | `[1,32,32,64] → [1,32,32,64]` | RELU |
| 5 | CONV_2D | SAME | 1 | 3×3 | `[1,32,32,64] → [1,32,32,64]` | NONE |
| 6 | ADD | – | – | – | 同形相加 | RELU |
| 7 | CONV_2D | SAME | 1 | 3×3 | `[1,32,32,64] → [1,32,32,64]` | RELU |
| 8 | CONV_2D | SAME | 1 | 3×3 | `[1,32,32,64] → [1,32,32,64]` | NONE |
| 9 | ADD | – | – | – | 同形相加 | RELU |
| 10 | PAD | – | – | – | `[1,32,32,64] → [1,34,34,64]`，paddings `[0,0,1,1,1,1,0,0]` | – |
| 11 | CONV_2D | VALID | 2 | 3×3 | `[1,34,34,64] → [1,16,16,128]` | RELU |
| 12 | CONV_2D | SAME | 1 | 3×3 | `[1,16,16,128] → [1,16,16,128]` | NONE |
| 13 | CONV_2D | VALID | 2 | 1×1 | `[1,32,32,64] → [1,16,16,128]` | NONE |
| 14 | ADD | – | – | – | downsample 相加 | RELU |
| 15 | CONV_2D | SAME | 1 | 3×3 | `[1,16,16,128]` 保持 | RELU |
| 16 | CONV_2D | SAME | 1 | 3×3 | `[1,16,16,128]` 保持 | NONE |
| 17 | ADD | – | – | – | 同形相加 | RELU |
| 18 | PAD | – | – | – | `[1,16,16,128] → [1,18,18,128]` | – |
| 19 | CONV_2D | VALID | 2 | 3×3 | `[1,18,18,128] → [1,8,8,256]` | RELU |
| 20 | CONV_2D | SAME | 1 | 3×3 | `[1,8,8,256]` 保持 | NONE |
| 21 | CONV_2D | VALID | 2 | 1×1 | `[1,16,16,128] → [1,8,8,256]` | NONE |
| 22 | ADD | – | – | – | downsample 相加 | RELU |
| 23 | CONV_2D | SAME | 1 | 3×3 | `[1,8,8,256]` 保持 | RELU |
| 24 | CONV_2D | SAME | 1 | 3×3 | `[1,8,8,256]` 保持 | NONE |
| 25 | ADD | – | – | – | 同形相加 | RELU |
| 26 | PAD | – | – | – | `[1,8,8,256] → [1,10,10,256]` | – |
| 27 | CONV_2D | VALID | 2 | 3×3 | `[1,10,10,256] → [1,4,4,512]` | RELU |
| 28 | CONV_2D | SAME | 1 | 3×3 | `[1,4,4,512]` 保持 | NONE |
| 29 | CONV_2D | VALID | 2 | 1×1 | `[1,8,8,256] → [1,4,4,512]` | NONE |
| 30 | ADD | – | – | – | downsample 相加 | RELU |
| 31 | CONV_2D | SAME | 1 | 3×3 | `[1,4,4,512]` 保持 | RELU |
| 32 | CONV_2D | SAME | 1 | 3×3 | `[1,4,4,512]` 保持 | NONE |
| 33 | ADD | – | – | – | 同形相加 | RELU |
| 34 | MEAN | – | – | – | `[1,4,4,512] → [1,512]`，axes `[2,1]`，keep_dims=false | – |
| 35 | FULLY_CONNECTED | – | – | 1000×512 | `[1,512] → [1,1000]` | NONE |

四个 stage 的结构完全同构：`[PAD(仅下采样时) → CONV 3×3/s2 → CONV 3×3/s1 → ADD] × 2`，downsample 支路是 `CONV 1×1/s2`。上表 36 行与 `assets/p3-opaudit.json` 的 `ops` 数组一一对应，无折叠。

### 6.4 窗口位置：为什么「显式 PAD + VALID」不是障眼法

plan.md 要求「不能仅因输出尺寸同为 64 就换成 SAME」。这里逐条给几何对应，而不是只比尺寸：

**stem（7×7/s2/pad3）**：显式 pad 3 后 `128→134`，`floor((134-7)/2)+1 = 64` ✅。窗口对齐：padded 下标 = 原下标 + 3，输出第 j 个位置在 padded 上覆盖 `[2j, 2j+6]`，即原图 `[2j-3, 2j+3]`——正是 PyTorch `pad=3` 的窗口。若改用 TFLite SAME，`pad_total = (64-1)*2+7-128 = 5`，前 2 后 3，**左边界少 1 像素**，窗口整体错位。

**stride2 3×3（pad1）**：`32→34`，`floor((34-3)/2)+1 = 16` ✅；padded 覆盖 `[2j, 2j+2]` = 原图 `[2j-1, 2j+1]`。SAME 会给 `pad_total = (16-1)*2+3-32 = 1`，前 0 后 1，同样错位。

**1×1 stride2**：不需要 pad，直接 VALID，`32→16` ✅（C3 专门断言「1×1 不该有 pad」，否则会白拿一圈 0 并改变尺寸）。

**maxpool 3×3/s2/pad1**：pad 1 后 `64→66`，`floor((66-3)/2)+1 = 32` ✅；窗口 `[2j, 2j+2]` = 原图 `[2j-1, 2j+1]`。**关键在于填充值**：TFLite 这里给的是 `PADV2` 且常量 = **-inf**，与 PyTorch `MaxPool2d` 的 `-inf` 填充完全一致，所以「负值输入」这条风险在 TFLite 层面根本不存在（`neg_const` 输入也验证了这一点，误差 6.4e-06）。

> 结论：显式 PAD + VALID 是**保语义**的等价替换，不是尺寸凑数。SAME 在这里反而是错的那个。

---

## 7 与仓库 `converter.py` 的兼容性（本批最重要的实测）

### 7.1 原样运行：卡在 PADV2，不卡在 SAME

```
$ python python/converter.py --model assets/p3-model.tflite ...
NotImplementedError: Unsupported operator at layer 2: 60, PADV2
```

（完整日志 `logs/p3/converter-smoke.log`；退出码 1，未产出任何 msgpack。）

三点澄清，都来自这一次实测而不是读码猜测：

1. **`MAX_POOL_2D` 的 `SAME` 拒绝逻辑没有被触发**。`converter.py:161-162` 那条 `raise NotImplementedError('SAME padding is not supported')` 只在 `opt.Padding() == SAME` 时触发，而本模型是 VALID（`padding=1`），直接跳过。**先前担心的「maxpool SAME 阻塞」被证伪。**
2. **`CONV_2D` 的 VALID 也被正常接受**，op0/op1 顺利通过，说明「显式 PAD 出更大张量再 VALID 卷积」这条形态是 converter 认的。
3. 真正缺的是 `PADV2`（TFLite `BuiltinOperator` 60）——`converter.py` 只写了 `PAD` 分支。

### 7.2 最小补丁探针（诊断用，未提交为仓库改动）

`scripts/p3_converter_probe.py` 不改仓库文件，而是读源码、在 `PAD` 分支前插入 8 行、写出 `patches/converter_p3_padv2.py`，再把插入片段打印出来给复核。插入内容：

```python
elif op_code == tflite.BuiltinOperator.PADV2:
    layer_type = 'Pad'
    params = interpreter.get_tensor(op.Inputs(1)).flatten().astype(np.int64).tolist()
    _c = float(interpreter.get_tensor(op.Inputs(2)).flatten()[0])   # 打印用
```

补丁后：**exit=0**，36 层全部产出。

| 产物 | 大小 | `sha256`（前 12） |
|---|---|---|
| `logs/p3/patched-model.msgpack` | 33,441,970 B | `74bc3f77f772` |
| `logs/p3/patched-config.msgpack` | 3,958 B | `d23276b52ecc` |

烟测参数是 `converter.py` 的**默认值**（`global_sf=2^16`、`k=19`、`num_cols=6`、`num_random=20001`、`--no-commit`）。这些**不是** P4 的选型结论，只是让流程跑通的占位值。

### 7.3 补丁带来的语义降级（必须写进报告，不能默认）

Rust 侧 `src/layers/shape/pad.rs` 把填充值**硬编码为常量 0**：

```rust
let zero = constants.get(&0).unwrap().clone();
let padded = pad(input, padding.padding, &zero);
```

而本模型的 `PADV2` 常量是 **-inf**。也就是说补丁把一个 `-inf` 填充**降级成 0 填充**。等价性条件：

- 设被填充张量为 `x`，填充后窗口取 max。`max(x ∪ {0}) = max(x)` **当且仅当** `max(x) ≥ 0`。
- 本模型的 `PADV2`（op2）输入是 `conv1 + BN + ReLU` 的输出，**逐元素 ≥ 0**（生产者 `CONV_2D` 的 `fused_activation = RELU`，审计 C15 已记录该证据）。
- 因此本模型的 0 填充与 -inf 填充**等价**；但这是**这个图的性质**，不是 `PadChip` 的通用性质。换一个 maxpool 前没有 ReLU 的模型，同样的补丁就会算错。

**所以「最小补丁」的完整形态应当是三条，而不是一条：**

1. 补 `PADV2` 分支，只取 paddings（把 `PADV2` 当 `PAD` 处理）；
2. **不要把 PADV2 的常量张量放进 `keep_tensors`**（理由见 7.4）；
3. 加断言：若该 `PADV2` 常量是 `-inf` 且其输入不是「ReLU 之后的张量」，直接报错退出，不静默降级。

第 3 条是本批新增的、必须由用户决定的点：它会让 converter 对某些模型变「更挑剔」，但能防止一类静默错误。

### 7.4 顺手抓到一个死常量：-inf 被转成 INT64_MIN

`converter.py` 对所有 FLOAT32 常量做 `(x * scale_factor).round().astype(np.int64)`。PADV2 的常量 `-inf` 因此变成 `INT64_MIN`，并以一个**无人引用的死常量**身份留在 msgpack 的 `tensors` 里：

```
tensor idx 1, shape [1], size 1, value -9223372036854775808
```

`Pad` 层只吃 paddings，不读它，所以不影响本批结论；但它是个陷阱：任何「按张量做定点范围检查/分配 cell」的下游步骤都可能被这个 `INT64_MIN` 干扰。这正是 7.3 第 2 条要把它剔除的原因。

---

## 8 TFLite ↔ converter msgpack 逐层交叉校验

脚本 `scripts/p3_crosscheck.py`，结果 `assets/p3-crosscheck.json`：

```
ops=36  layers=36  structure_match: True
```

逐层比对的是三件事，缺一不可：

1. **算子类型映射**（`CONV_2D→Conv2D`、`PADV2→Pad`、`MAX_POOL_2D→MaxPool2D`…）；
2. **输出形状完全相等**（36/36）——这条最强：padding 或 stride 错一个像素，形状链就会断；
3. **参数逐项对应**：`Conv2D = [0, pad, act, sH, sW]`、`MaxPool2D = [kh, kw, sH, sW]`、`Add = [act]`、`Mean = axes`、`FullyConnected = [act]`、`Pad = paddings`。

其中一条是**方向性核对**，值得单独写出来：`Conv2D` 的 padding 参数在 converter 里直接取 TFLite 枚举值（`SAME=0, VALID=1`），Rust 侧 `PaddingEnum` 恰好也是 `Same=0, Valid=1`（`layers/conv2d.rs:62-64`）。**方向一致**。若这两者顺序相反，整个网络每一层的 padding 都会被静默翻转，而形状链仍然「看起来对」——这类错误只能靠逐层参数比对抓。

其余与 Rust 实现相关的实测/读码结论（供 P4 预判，不是本批结论）：

| 侧 | 事实 | 对本模型的影响 |
|---|---|---|
| `MaxPool2DChip` | 形状计算固定用 `PaddingEnum::Valid`（`layers/max_pool_2d.rs:31`） | 本模型 VALID，兼容 |
| `MeanChip::get_keep_axis` | 要求「去掉 mean 轴后只剩 1 个轴」（`layers/mean.rs`） | axes={1,2}、rank=4 → 留 axis 3，兼容；`[2,1]` 的顺序无影响 |
| `PadChip` | 填充值固定为常量 0 | 见 7.3 的等价条件 |
| `FullyConnected` | 参数只有 activation | `[0]`，兼容 |

---

## 9 对 P4 的输入（静态计数，不是资源结论）

| 项 | 值 | 说明 |
|---|---|---|
| 计算量 | **0.593 GMAC**（= 1.185 G ops） | 与 litert 自报 1.187 G 差 0.15%（取整口径） |
| 最大单层 | stem conv 38.5 MMAC（6.5%） | 其余 16 个 3×3 各 37.7 MMAC |
| 权重标量 | 11,684,731（msgpack 全张量合计） | 含 padding 索引常量 |
| msgpack 大小 | 33.4 MB（model）/ 3.9 KB（config） | — |
| **Pad 中间张量** | 473,708 cells / 全部层输出 1,597,524 cells = **29.65%** | 显式 PAD 会在 Rust 侧 materialize 一个更大的张量，这部分 cell 是「纯填充」 |

最后一行是本批唯一带资源含义的观察，但它**只是静态计数**：真正的 cell 数还取决于 gadget 布局、lookup、packing 与 selector，必须在 P4 用受控探针测。不要拿 29.65% 去线性推 prove 时间。

---

## 10 结论、限制与待决

### 能写的

1. 标准 1000 类 ResNet-18（权重身份可核）在 128×128、NCHW 输入下，可用 `torch.export → litert_torch` 导出为 **NHWC FP32 TFLite**，导出的 I/O 布局、算子清单、padding/stride 语义逐条可解释。
2. 导出数值与 PyTorch 一致到 `3.4e-05` 量级，5 个输入 argmax 全同；布局负向对照能把错误放大到 22.28，门禁有效。
3. 结构未被改动：20 个卷积、8 个残差、17 个 ReLU（融合后计数一致）、1 次全局池化、1000 类输出，BN 在 eval 下折叠且无数值残留。
4. 仓库 `converter.py` 在本模型上的**真实阻塞点是 `PADV2`**，不是 `SAME`；补一个最小分支即可产出 36 层 msgpack，且与 TFLite 逐层一致。

### 不能写的

- ❌「ResNet 证明成功」——没有 circuit、没有 SRS、没有 proof、没有 verify。
- ❌「精度无损」——只测 5 个输入，没有数据集精度。
- ❌「定点等价」——FP32 门禁与定点误差是两件事，scale 都还没定。
- ❌「与旧路线同模型」——这是独立基线，权重来自 torchvision 官方。
- ❌「补丁可以直接采纳」——它把 -inf 降级成 0，只在可验证的非负前提下等价。

### 待用户决定

1. **是否采纳 PADV2 补丁**，以及是否接受 7.3 的第 3 条（对「-inf 填充 + 非 ReLU 输入」直接拒绝）。
2. **是否先补 §2.6 的三处 P2A 收紧**（末元素 104 的代数分类依据、加载异常三模式逐一覆盖、105 个扫描副本独立归档），再进 P4。
3. **P4 的目标规模**：`k` 与列数由预算门禁决定，本批没有给任何建议值；`converter.py` 的默认 `k=19` 只是烟测占位。

---

## 11 复现命令与产物

```bash
# 1) 环境（只做一次）
bash scripts/p3_setup_env.sh          # conda env + torch/torchvision (cpu wheel)
bash scripts/p3_setup_litert.sh       # litert-torch / ai-edge-litert / tflite
bash scripts/p3_fetch_weights.sh      # checkpoint（curl -C -，含 sha256 校验）

# 2) 身份与输入（写 prereg，早于转换）
python scripts/p3_prep.py

# 3) 导出 + 数值门禁
python scripts/p3_convert.py

# 4) 静态算子审计
python scripts/p3_opaudit.py assets/p3-model.tflite assets/p3-opaudit.json

# 5) converter 兼容性（原样 → 必然停在 PADV2）
bash scripts/p3_run_converter.sh

# 6) 最小补丁探针 + 逐层交叉校验
python scripts/p3_converter_probe.py
python scripts/p3_crosscheck.py
```

产物（远端 `/root/autodl-tmp/zkml-repro/`）：

| 路径 | 大小 | `sha256` |
|---|---|---|
| `assets/p3-model.tflite` | 46,789,576 B | `4014a8b4cde23d61e328114cf67a2247a766e812b2195d820a49fc13dd4cb4a1` |
| `assets/p3-model-nchw-io.tflite`（不采纳） | 46,776,576 B | `bb609776b55a0dd080b72825536f4c970e83bf1bada6e21c4066d25ba0b30634` |
| `assets/p3-prereg.json` | 4,750 B | `2a898d0c1a7e1d55c6c37c1970661373e7cdb12ee10c4c804713b59a499e9972` |
| `assets/p3-convert-report.json` | 4,152 B | `c4f71363fa8d8f519b18c92983187e392b6592537fbba8437fd7057ac88332fe` |
| `assets/p3-opaudit.json` | 102,033 B | `85a11cf3558522008ffb40bd9221cbe2a6c6630a24257f266971e3f3aa295c83` |
| `assets/p3-crosscheck.json` | — | `f5e7884f763191c0317f1705d29dc13346515dda96c217c66ef4fa72a71c749b` |
| `assets/p3-msgpack-summary.json` | — | `6d28e58ced53786c87b0469b72ab9a5ef3a209518e3f9434e58410b0ea8b5775` |
| `assets/p3-env-freeze.txt` | — | `e91fe0b937480f16f1c6d8b8d23915cc98c1bf6c296302acbbbe86ee05ffad2f` |
| `logs/p3/patched-model.msgpack` | 33,441,970 B | `74bc3f77f772c4e9883a1d6545521b92d1fe564af24f8e4ff635b18b1833f6a5` |
| `logs/p3/patched-config.msgpack` | 3,958 B | `d23276b52ecc5479226d95a13e788c70c17d6500f3ae9fba4c62397cc9fa917d` |
| `patches/converter_p3_padv2.py` | 21,529 B | `c6bb8e8d19f15e749e89386ab211062698cbf87a03accdaa9f7d72411054439b` |
| `scripts/p3_opaudit.py` | — | `d1cca4c5d2935de2dc4c22d9440a2cb1e2d297ad9d073d135a20ef7cf2e52c99` |
| `scripts/p3_crosscheck.py` | — | `eacc6628adeb9fe333a6a2a57c0fd87a414d251c4384a409a9abd5beed4d8d02` |
| `scripts/p3_converter_probe.py` | — | `e163679ff3eae9ba0255dab1fc9682678ccd6dd86af7eda85b80b9591d64b390` |
| `scripts/p3_prep.py` | — | `6d2025b945d88067b6ebc6ebe77652bee50289a440cc4891f307edeb72479c06` |
| `scripts/p3_convert.py` | — | `aad4a8d266d3f31ae36a7a8477dd6138feaa5ab0bf4c7bb07d1575731f856436` |

磁盘占用：`assets` 136 M、`env` 4.0 G、`logs/p3` 32 M、`target` 172 M、`runs` 115 M、`srs` 4.1 M。无 SRS 大文件生成，未触及 120 GiB / 16 CPU 门禁。

**本地归档**：`logs/p3/` 下已同步 7 个小体积证据（`p3-opaudit.json` 102,033 B、`p3-crosscheck.json`、`p3-msgpack-summary.json`、`p3-convert-report.json`、`p3-prereg.json`、`converter-smoke.log`、`converter-patched.log`）。TFLite 模型、msgpack 与 conda 环境留在远端，不下载。
