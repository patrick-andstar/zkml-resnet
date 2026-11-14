# P2 过程记录：MNIST 闭环与端到端计时校准

执行时间：2026-11-14 18:12–18:14（UTC+8）

判据：**MNIST 只是工具链闭环与量具校准，不是目标结论。** 目标仍是 ResNet-18 / 128×128。

> ⚠️ **本文 §4–§8 的性能数字与归因已被 P2A 推翻。** 原文保留不改；纠正见文末 **§勘误（P2A）**。一句话：SRS 慢在未缓冲 `File` 的 52 万次小 `read_exact`，加 1 MiB 缓冲后 187.8 → 6.18 ms；外层 222.9 ms 含 20 ms 轮询误差，校准后为 213.4 ms。

## 1. 附带模型参数（先核实，不照抄 README）

| 文件 | k | num_cols | scale | 备注 |
|---|---|---|---|---|
| `examples/mnist/model.msgpack` | 15 | 7 | 512 | 含 20 个 tensor 权重 |
| `examples/mnist/config.msgpack` | 15 | 7 | 512 | `tensors` 为空（瘦配置） |
| `examples/mnist/inp.msgpack` | — | — | — | `[1,28,28,1]`，784 个整数 |

与交接材料的 `k=17` 不符：**README 的 k=17/10 列是"重新转换示例"，不是附带模型的实际配置。**

## 2. 约束测试（先正确性，后证明）

```bash
cd runs/mnist-baseline
RAYON_NUM_THREADS=16 python scripts/measure.py ... -- \
  target/release/test_circuit src/examples/mnist/model.msgpack src/examples/mnist/inp.msgpack
```

| 指标 | 值 |
|---|---|
| 退出码 | 0 |
| 全进程 wall | **0.647 s** |
| 峰值 RSS | 117 MB |

最终输出（定点整数 → 除以 scale=512）：

```text
out[5] = 6614  -> 12.91796875   ← argmax
out[0] = -5312 -> -10.375
...
final out idxes: [37]
```

`out.msgpack` 31 字节。**约束 assert 通过**（`test_circuit.rs:22` 的 `assert_eq!(prover.verify(), Ok(()))` 未 panic）。

## 3. 首次证明（冷 SRS）

SRS 使用**全新命名空间** `srs/mnist-initial/`，运行目录的 `params_kzg` 是指向它的符号链接，确保不会误用未知参数。

```bash
RAYON_NUM_THREADS=16 time_circuit model.msgpack inp.msgpack kzg
```

| 阶段（上游日志原样） | 值 |
|---|---|
| params construction（含 setup + 序列化 + 重读） | **1.404 s** |
| generating vkey | 684 ms |
| generating pkey | 446 ms |
| filling circuit | 164 ms |
| **Proving time** | **3.357 s** |
| Verifying time（进程内，仅核心） | 6.76 ms |
| 全进程 wall | **6.115 s** |
| 峰值 RSS | 326 MB |

产物：

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `15.params` | 4,194,564 | `1e9ef6f46a33d2c60f6e4275f16ae936475b52b4dfbf056235daffa86306c95d` |
| `vkey` | 25,608 | `eb095671d3365d615f5cdf7000bad91ba606dc14b02c5d4b09d2e6f5cca5aafa` |
| `pkey` | 113,272,044 | `47aea2dbc09fb0276fd246ffb2322da4c79feb7cde0347a829e479f69362cbbc` |
| `proof` | 3,360 | `d700fd154a6e23783cb8da09e11b94a9efcd942861127f8a38abb3b310e35623` |
| `public_vals` | 320 | `37987fed3d53d98d07a4da9dd4ad9e9691b063673a532b8594d2fab4200a9b22` |

输入文件摘要：`config.msgpack` `dcf61f7d…`、`model.msgpack` `9d00a1a3…`、`inp.msgpack` `9eb8082a…`

**注意**：`time_circuit` 每次运行都重新 `keygen_vk` + `keygen_pk`，所以第二次运行"更快"只是 SRS 命中缓存，**不存在"热 PK prove"**。

### 顺带确认的 SRS 结构

`ParamsKZG` 存两组点：`g`（系数基）和 `g_lagrange`（拉格朗日基），每组 `2^k` 个。文件大小可精确对上：

```text
2 × 2^15 × 64 B(未压缩点) + 4 B(k) + 128 B(g2) + 128 B(s_g2) = 4,194,564 B
```

这一点很重要：**验证只需要 `g2`、`s_g2` 和 proof 里的承诺，根本不需要这 2×2^k 个 G1 点，却每次都被完整读入解码。**

## 4. 独立验证（原版 `verify_circuit`，单 CPU）

```bash
RAYON_NUM_THREADS=1 taskset -c 0 verify_circuit config.msgpack vkey proof public_vals kzg
```

```text
Loaded configuration
Loaded the parameters
Loaded vkey
Verifying time: 7.162547ms
Proof verified!
```

10 次重复（每次都是新进程、绑 CPU 0）：

| 指标 | 值 |
|---|---|
| 全进程 wall | median **222.9 ms**（min 222.7 / max 224.0） |
| 核心验证 | 8.8 ms（8.79–8.95） |
| 峰值 RSS | 11.5 MB |

## 5. 负向验证（两类，都必须被拒绝）

| 用例 | 改动 | 退出码 | 结果 |
|---|---|---|---|
| proof 比特翻转 | 第 1680 字节 `0x4e → 0x4f` | **101** | `panicked ... proof did not verify` |
| public value 替换 | 第 0 个元素改为合法域元素 `2` | **101** | `panicked ... proof did not verify` |

第二条特意改成**合法的规范域元素**（不是让解码器失败的乱码），所以拒绝来自密码学校验，而不是解析错误。

## 6. 分段测量：新写 `verify_profile`

补丁文件：`patches/verify_profile.rs`（另存 `src/src/bin/verify_profile.rs`，SHA-256 `3e84f8e2…`）。

设计约束（与原版行为兼容）：
- 复用 `load_config_msgpack` / `ModelCircuit::generate_from_msgpack` / `verify_kzg`，**不改协议、不改密钥与证明格式**；
- 参数目录**显式传入**，文件不存在直接 panic，**绝不 fallback 到随机 setup**；
- 分段计时 + 显式 `drop` 统计清理 + 读 `/proc/self/status:VmHWM`。

缺参保护实测：

```text
thread 'main' panicked at src/bin/verify_profile.rs:80:
trusted setup not found at /tmp/empty-params/15.params - refusing to generate a fresh SRS
```

### 10 次分段结果（单 CPU，warm page cache）

| run | 外层 wall | 内部合计 | SRS | VK | 核心 | 配置 | 清理 |
|---|---|---|---|---|---|---|---|
| 1 | 223.0 | 207.9 | 190.9 | 7.4 | 8.85 | 0.09 | 0.39 |
| 2 | 223.0 | 209.2 | 192.5 | 7.3 | 8.82 | 0.07 | 0.34 |
| 3 | 222.9 | 209.9 | 193.2 | 7.2 | 8.83 | 0.08 | 0.33 |
| 4 | 223.0 | 204.2 | 187.0 | 7.3 | 9.22 | 0.07 | 0.33 |
| 5 | 223.1 | 202.7 | 185.9 | 7.5 | 8.83 | 0.03 | 0.36 |
| 6 | 222.7 | 199.7 | 183.3 | 7.1 | 8.81 | 0.04 | 0.31 |
| 7 | 222.7 | 198.7 | 182.3 | 7.1 | 8.83 | 0.03 | 0.30 |
| 8 | 223.1 | 201.6 | 185.0 | 7.2 | 8.82 | 0.08 | 0.30 |
| 9 | 222.8 | 199.8 | 183.4 | 7.1 | 8.77 | 0.08 | 0.31 |
| 10 | 222.9 | 198.7 | 182.4 | 7.0 | 8.78 | 0.04 | 0.30 |
| **median** | **222.9** | 202.2 | **185.5** | 7.2 | **8.8** | 0.05 | 0.33 |

（单位 ms）

完整请求分解：

```text
222.9 ms 外层 wall
├─  0.05  config 解码
├─  0.05  电路配置构造
├─ 185.5  SRS 读取（ParamsKZG::read）
├─   7.2  VK 读取/解码/configure
├─   0.02  proof 读取
├─   0.01  public values 读取/解码
├─   0.00  strategy/transcript 初始化
├─   8.8  核心 verify
├─   0.33 清理解构
└─ ~20.7  进程启动 + main 返回后的静态对象析构 + 测量开销（未单独拆分）
```

## 7. SRS 加载为什么这么慢：不是磁盘，是逐点解码

| 测试 | 结果 |
|---|---|
| Python 直接读 4 MB 文件到内存（page cache 命中） | **2.7–3.2 ms** |
| `ParamsKZG::read` 同一文件 | **185.5 ms** |

差 60 倍 ⇒ 瓶颈在**逐点解码校验**，不在 I/O。已用 16 线程对照实测：`RAYON_NUM_THREADS=16`、不绑核，外层仍是 222.9 / 223.2 / 222.9 ms，SRS 段 184.9 / 184.5 / 184.5 ms，与单线程 185.5 ms 无差别（核心 8.8 → 8.0 ms，属噪声量级）。原因：`ParamsKZG::read` 的 `RawBytes` 分支是 `(0..n).map(read)` 串行解码，Rayon 管不到。

### 对目标规模的推算（**估算，非实测**）

点数 = 2×2^k，读取成本按点数线性：

| k | 参数文件 | 仅 SRS 加载（按 k=15 外推） |
|---|---|---|
| 15（实测） | 4.2 MB | 0.186 s |
| 20 | 134 MB | ~5.9 s |
| 24 | 2.1 GB | ~95 s |
| 28 | 34.4 GB | **~25 min** |

即使 k=20（134 MB / 约 6 s）也已经是 1000 ms 预算的 6 倍。**这意味着：不改变验证方加载参数的方式，目标在架构上就达不到，与证明算得多快无关。**

## 8. 本阶段结论

1. 工具链、编译、证明、独立验证、负向拒绝，**全链路真实跑通**（不是静态检查）。
2. 原版 `verify_circuit` 的"完整请求 / 核心验证"= **222.9 / 8.8 ≈ 25×**，验证了计划里"不能拿核心当端到端"的判断。
3. 主要开销是 `ParamsKZG::read`（83%），而它对验证**在数学上并不需要**。
4. 下一阶段若继续，必须先决定：是否允许做"验证方专用轻量参数加载"（只读 `g2`/`s_g2`）作为**独立优化实验**，还是先按原实现硬扛到更大 k。这是 `plan.md` §6.1 明确要求"另立实验"的决策点。

## 9. 未做的事

- 未跑 `time_circuit` 第二次（会重复 3.4 s 证明并覆盖产物，无新增信息）。
- 未做 128×128 相关的一切工作（转换、大 k、SRS 扩容）。
- 16 线程验证对照已做（见 §7），结果与单线程一致；未做多 NUMA 绑定等更细的调度实验。

---

## 勘误（P2A 实测，2026-11-14 20:15）

依据：`03-P2A-量具校准与读取对照.md`、`runs/p2a-calib1/`。§3 的证明段与产物摘要**仍然有效**（SRS/vkey/proof/public_vals 摘要已复核未变）；出问题的是 §4、§5、§6、§7 的解释。

### 本文件受影响的具体条目

| 位置 | 原文 | 实测纠正 |
|---|---|---|
| §4 表 | 全进程 wall median 222.9 ms | **213.4 ms**（同一原版 binary，新量具，10 次中位） |
| §6 表 | 外层 222.9 / 内部 202.2 / SRS 185.5 | 外层 **211.7**、内部 **204.9**、SRS **187.8**（`--read-mode file`，10 次中位） |
| §6 分解图 | `~20.7 进程启动 + 析构 + 测量开销` | 校准后外-内差 **6.9 ms**；旧值里约 14 ms 是 20 ms 轮询的检测滞后 |
| §7 标题 | "SRS 加载为什么这么慢：不是磁盘，是逐点解码" | **归因错**。换成 1 MiB `BufReader` 后同一 `ParamsKZG::read` 从 187.8 → **6.18 ms**；慢的是 524,288 次 8 B `read_exact`，不是解码 |
| §7 表 | 4 MB 读 2.7–3.2 ms vs `ParamsKZG::read` 185.5 ms，"差 60 倍⇒瓶颈在逐点解码" | 这条推理不成立：差的 60 倍几乎全是系统调用次数，缓冲后两者同量级 |
| §7 外推表 | k=20 ≈ 5.9 s / k=24 ≈ 95 s / k=28 ≈ 25 min | **基数错误，作废**。按 6.18 ms 重算（仍为估算）：k=20 ≈ 198 ms、k=24 ≈ 3.2 s、k=28 ≈ 51 s，且 k≥24 时几十 GB 的文件读取成为新下限 |
| §5 表 | proof 比特翻转退出 101，"拒绝来自密码学校验" | 当时**未记录底层错误类型**。P2A 补做了可分类证据：合法编码篡改 → 读满 105 次/3360 B 后被 `ConstraintSystemFailure` 拒绝；截断/非法编码 → 停在解码阶段 |
| §8 结论 2 | 完整请求/核心 = 222.9/8.8 ≈ 25× | 校准后 **213.4/8.85 ≈ 24×**（原版）；开启缓冲后 **28.5/8.85 ≈ 3.2×** |

### 不变的部分

- §1 的模型参数（k=15 / 7 列 / scale 512）不变。
- §2 的约束测试结论不变。
- §3 的证明阶段耗时与全部产物 SHA-256 不变，且 P2A 已复核四个产物摘要与本文一致（**未被 P2A 改动**）。
- §3 末尾"验证只需要 `g2`、`s_g2`，却每次读入 2×2^k 个 G1 点"这一**结构性观察不变**；变的只是"读这些点的代价"——从 185 ms 降到 6.18 ms。这也意味着"轻量参数"优化的收益上限随之从 185 ms 降到 ≤6.18 ms。

