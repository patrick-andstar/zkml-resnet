# 11 · B.3.1：model 加载期契约回归

> 依据：用户 2026-11-15 00:46 的路线第 2 步 —— model 加载回归（合法稀疏图、缺依赖、未来引用、重复产出、双模式、before/after、unresolved_commit_ids）。
>
> **只调用 `generate_from_msgpack`**：不 configure、不 MockProver、不合成、不转真模型。`generate` 只解析与校验，大表在 configure/synthesize 才建，所以这一批不会拉起重任务。

## 1. 做法

- 新增 `src/bin/p4b_model_load_test.rs`（11 例带断言）与 `src/bin/p4b_model_load_ctl.rs`（控制驱动，不引用 v5 专有字段）。
- 配置：`k=6`（num_rows=55、max_val=22、shift=64 均合法）、`num_cols=6`、`global_sf=1`、层类型用 `Noop`（`used_gadgets` 为空，不牵入任何重 gadget）。
- 模型直接用 Rust 结构体构造，不经 msgpack 文件。

## 2. 结果：v5 11/11 PASS

| # | 用例 | v5 | v3（对照） | 有区分力 |
|---|---|---|---|---|
| 1 | 稀疏图 `{0,2}→L0产1` | Ok | Ok | **否** |
| 2 | 缺依赖（FullWitness） | Panic | Panic 同文本 | 否 |
| 3 | 未来引用 | Panic（未来引用文案） | Panic（可用性文案） | 是（文本） |
| 4 | 重复产出 | Panic | **Ok** | 是 |
| 5 | config-only 结构合法 | **Ok** | Panic | 是 |
| 6 | FullWitness 权重缺席 | Panic | Panic 同文本 | 否 |
| 7 | config-only before 引用产出 | Panic（阶段文案） | Panic（权重 99 处先崩） | 是（理由） |
| 8 | config-only after 引用产出 | **Ok** | Panic | 是 |
| 9 | config-only after 未解析常量 | **Ok** + unresolved 含 77 | Panic | 是 |
| 10 | config-only before 未解析常量 | **Ok** + unresolved 含 77 | Panic | 是 |
| 11 | FullWitness before 初始张量 | Ok + unresolved 空 | Ok | 否 |

**6 例在 v3 下翻转或理由不同**，说明这组用例不是空转。

## 3. 必须说清的两点

1. **c1 没有区分力。** v2/v3 真正的空洞 panic 发生在 `tensor_map_to_vec`（synthesize 期），本驱动不调用它，所以 c1 在 v3 下也是 Ok。要覆盖它必须走 MockProver——那是另一批。
2. **c7 在 v3 下"也 panic"，但理由不对**：v3 在权重 `99` 的可用性检查处就崩了，根本没走到 commit 判据。所以「两边都 panic」不能算一致，要看理由。

## 4. 证据

- `logs/p4b/b11-model-load-v5.log`（2,270 B，`132fe284…`）—— v5 主驱动，rc=0
- `logs/p4b/b10-model-load-NEGATIVE-v3.log`（2,737 B，`8d3396e0…`）—— v3 控制驱动
- 负向对照后已把 `model.rs` 还原为 v5 并核对 `1b0d3232…`
- 摘要曾写反（两行 `cut -c1-16` 对调），已逐文件 `sha256sum` 核对改正

## 5. 不覆盖

synthesize / configure / MockProver / 约束；真实 TFLite 与 converter；Pad→Max 组合；目标 k。控制驱动的对比是文本与 ok/panic 的人工对照，不是自动断言。
