# zkml-resnet

标准 ResNet-18（NCHW `1×3×128×128`）在 [ZKML](https://github.com/ddkang/zkml-public)（EuroSys'24）的
Halo2 / KZG 后端上做复现的过程仓库。目标指标：**独立新进程、单 CPU 的完整验证请求 < 1000 ms**
（含 SRS / VK / config 加载与进程清理退出）。

本仓库按**阶段断点**分次提交，每个提交对应一次可验收的产出，不合并成一坨。

## 提交结构

| 提交 | 阶段 | 内容 |
|---|---|---|
| 1 | P1 | 上游源码快照、`Cargo.lock`、`rust-toolchain.toml`、构建日志 |
| 2 | P2 | MNIST 真实证明 + 独立验证闭环；SRS、proof、VK、public values、负向测试 |
| 3 | P2A | 量具校准（`wait4`）与 SRS 读取四组对照（原版 / 未缓冲 / BufReader / 预读） |
| 4 | P3 | 标准 ResNet-18 浮点导出（TFLite）、算子审计、converter 兼容性探针 |
| 5 | P4-A | 转换契约与候选修复（v1 与补正 v2）、生成器一致性门禁 |
| 6 | P4-B | 分层 `cargo check`、Max 归约最小用例 |

## 当前状态（务必先读）

**这不是一个"已复现成功"的仓库。** 截至最后一个提交：

- 1 s 目标**未验证**——从未在目标规模上生成过 SRS 或 proof。
- P4-A 的候选（v2）**静态契约验收未通过**，六条阻断中本仓库只解决了第 1 条（一处 Rust 所有权错误）。
- 上游 `converter.py` 没有 `PADV2` 分支，原样运行会在 `layer 2` 报错；能跑通的是带诊断补丁的探针。
- `cargo check` 通过**不等于**电路正确：它只做类型与借用检查。

这些边界在 `docs/` 与 `plan.md` 里逐条写明，未做美化。

## 目录

```
src/        上游 ZKML 源码（提交 1 为原始快照；提交 6 应用了 P4-B 候选）
scripts/    驱动、量具、补丁生成器、converter 探针
patches/    补丁与候选源码（含 .diff）
manifests/  摘要与结论清单（JSON）
runs/       运行产物：vkey / proof / public_vals / 日志（pkey 见下）
srs/        MNIST 用的 KZG 公共参数（k=15）
assets/     P3 导出产物：TFLite、输入、参考输出、审计报告
logs/       构建、转换、P4-B 编译与用例日志
docs/       过程文档（00 总览 + 01~07 各阶段）
plan.md     主计划与验收门禁
```

## 没有放进来的东西

| 排除项 | 体积 | 原因 |
|---|---|---|
| `runs/mnist-baseline/pkey` | 108 MiB | 超过 GitHub 单文件 100 MiB 上限；可由同一 SRS 重新 keygen |
| `assets/torch-hub/`（`resnet18-f37072fd.pth`） | 46 MiB | torchvision 公开权重，SHA-256 `f37072fd…` 已在文档登记 |
| `assets/p3-model-nchw-io.tflite` | 46 MiB | 未被采纳的对照版本；采纳版为 `assets/p3-model.tflite`（SHA-256 `4014a8b4…`） |
| `runs/p2a-calib1/orig-srs/` | 4 MiB | 与 `srs/mnist-initial/15.params` 同一份，去重 |
| `target/`、`env/` | 1.8 G / 4.0 G | 构建产物与隔离工具链，可重建 |
| `/tmp` 与系统盘内容 | — | 按约定不上传；仅取 AutoDL 数据盘 `/root/autodl-tmp` 的项目目录 |

## 环境

Ubuntu 22.04 / Xeon Gold 6430（cgroup 16 CPU、120 GiB 内存）/
`rustc 1.101.0-nightly (f9ceb4a4d 2026-10-31)` /
Halo2 `17e9765c199670534c0299c96128d0464a188d0b` / halo2curves `0.3.2`。
