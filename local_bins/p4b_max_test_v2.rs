//! P4-B.2 扩展矩阵（v2）：Max 独立 / 数值边界 / 拒绝契约。
//!
//! 身份：与 `p4b_max_test.rs` 并存，不覆盖。旧测试的 27/27 结论不回改。
//! 预登记：`manifests/p4b-b2-prereg.json`、`过程文档/10-P4B2-扩展执行清单（预登记）.md`。
//! **本文件尚未编译、尚未运行。**
//!
//! 三个子矩阵：
//!   A  Max 独立：21 向量 × {3,6,9,12} 列 × 正负两臂 = 168 次
//!   B  数值边界：shift=1024 的独立配置，4 次（区分 lookup 拒绝 / 宿主 panic）
//!   C  拒绝契约：空输入 4 次 + 非法列数 5 次 = 9 次
//! 合计 181 次 MockProver::run。

use std::{
  collections::BTreeSet,
  panic::{catch_unwind, AssertUnwindSafe},
  rc::Rc,
  sync::Arc,
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
  dev::MockProver,
  halo2curves::{bn256::Fr, ff::PrimeField},
  plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Instance},
};

use zkml::gadgets::{
  gadget::{Gadget, GadgetConfig, GadgetType},
  input_lookup::InputLookupChip,
  max::MaxChip,
};

const K: u32 = 12;
const NUM_ROWS: usize = 1000;

/// 与 model.rs `to_field` 同一编码。
fn to_field(x: i64) -> Fr {
  Fr::from((x + (1i64 << 31)) as u64) - Fr::from(1u64 << 31)
}

#[derive(Clone, Debug)]
struct TestConfig {
  gadget: GadgetConfig,
  input_col: Column<Advice>,
  instance: Column<Instance>,
}

/// 统一的 configure：num_cols 由调用方给，MaxChip 自己会校验 3 的倍数。
fn build_config(meta: &mut ConstraintSystem<Fr>, num_cols: usize, shift: i64) -> TestConfig {
  let mut gc = GadgetConfig::default();
  let cols: Vec<Column<Advice>> = (0..num_cols).map(|_| meta.advice_column()).collect();
  for c in cols.iter() {
    meta.enable_equality(*c);
  }
  gc.columns = cols;
  gc.num_cols = num_cols;
  gc.num_rows = NUM_ROWS;
  gc.k = K as usize;
  gc.use_selectors = true;
  gc.shift_min_val = shift;
  gc.used_gadgets = Arc::new(
    [GadgetType::Max, GadgetType::InputLookup]
      .into_iter()
      .collect::<BTreeSet<_>>(),
  );

  gc = InputLookupChip::<Fr>::configure(meta, gc);
  // 非法列数会在这里 panic —— 这正是子矩阵 C2 要打到的候选检查路径。
  gc = MaxChip::<Fr>::configure(meta, gc);

  let input_col = meta.advice_column();
  meta.enable_equality(input_col);
  let instance = meta.instance_column();
  meta.enable_equality(instance);

  TestConfig {
    gadget: gc,
    input_col,
    instance,
  }
}

/// 子矩阵 A：advice 列数 = 3G，shift = 64。
struct MaxTest<const G: usize> {
  input: Vec<i64>,
  expected: i64,
}

impl<const G: usize> Circuit<Fr> for MaxTest<G> {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self {
      input: vec![0],
      expected: 0,
    }
  }

  fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
    build_config(meta, 3 * G, 64)
  }

  fn synthesize(
    &self,
    config: Self::Config,
    mut layouter: impl Layouter<Fr>,
  ) -> Result<(), Error> {
    synthesize_max(&self.input, &config, &mut layouter)
  }
}

/// 子矩阵 B：固定 6 列、shift = 1024，专门用来把差值推到查表边界。
struct MaxTestBoundary {
  input: Vec<i64>,
  expected: i64,
}

impl Circuit<Fr> for MaxTestBoundary {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self {
      input: vec![0],
      expected: 0,
    }
  }

  fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
    build_config(meta, 6, 1024)
  }

  fn synthesize(
    &self,
    config: Self::Config,
    mut layouter: impl Layouter<Fr>,
  ) -> Result<(), Error> {
    synthesize_max(&self.input, &config, &mut layouter)
  }
}

/// 子矩阵 C2：给定 N 列，**不做 3 的倍数预检**，让候选的 assert 真正触发。
struct MaxTestRaw<const N: usize> {
  input: Vec<i64>,
  expected: i64,
}

impl<const N: usize> Circuit<Fr> for MaxTestRaw<N> {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self {
      input: vec![0],
      expected: 0,
    }
  }

  fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
    build_config(meta, N, 64)
  }

  fn synthesize(
    &self,
    config: Self::Config,
    mut layouter: impl Layouter<Fr>,
  ) -> Result<(), Error> {
    synthesize_max(&self.input, &config, &mut layouter)
  }
}

fn synthesize_max(
  input: &[i64],
  config: &TestConfig,
  layouter: &mut impl Layouter<Fr>,
) -> Result<(), Error> {
  let cells: Vec<AssignedCell<Fr, Fr>> = layouter.assign_region(
    || "p4b v2 inputs",
    |mut region| {
      let mut v = Vec::with_capacity(input.len());
      for (i, x) in input.iter().enumerate() {
        v.push(region.assign_advice(
          || "in",
          config.input_col,
          i,
          || Value::known(to_field(*x)),
        )?);
      }
      Ok(v)
    },
  )?;

  let lookup_chip = InputLookupChip::<Fr>::construct(Rc::new(config.gadget.clone()));
  lookup_chip.load_lookups(layouter.namespace(|| "input lookup"))?;

  let chip = MaxChip::<Fr>::construct(Rc::new(config.gadget.clone()));
  let refs: Vec<&AssignedCell<Fr, Fr>> = cells.iter().collect();
  let outs = chip.forward(layouter.namespace(|| "max forward"), &vec![refs], &vec![])?;

  if outs.len() != 1 {
    return Err(Error::Synthesis);
  }
  layouter.constrain_instance(outs[0].cell(), config.instance, 0)?;
  Ok(())
}

// ---------------- 结果记录 ----------------

#[derive(Debug, PartialEq)]
enum Stage {
  /// 全部约束满足
  Ok,
  /// verify() 返回 Err（lookup / 门约束失败）
  VerifyErr,
  /// MockProver::run 返回 Err（synthesize 阶段主动拒绝）
  RunErr,
  /// 进程内 panic。**不区分 configure/synthesize**——MockProver::run 把两段包在一次调用里。
  /// 靠 panic 文本区分来源：C2 的 assert 带断言文本，B3 的 convert_to_u64 是空文本。
  Panic,
}

#[derive(Debug)]
struct Outcome {
  stage: Stage,
  detail: String,
}

fn panic_text(e: Box<dyn std::any::Any + Send>) -> String {
  if let Some(s) = e.downcast_ref::<&str>() {
    return (*s).to_string();
  }
  if let Some(s) = e.downcast_ref::<String>() {
    return s.clone();
  }
  "<non-string panic>".to_string()
}

fn run_classified<C: Circuit<Fr>>(circuit: C, instance: Fr) -> Outcome {
  let res = catch_unwind(AssertUnwindSafe(|| {
    MockProver::run(K, &circuit, vec![vec![instance]])
  }));
  match res {
    Err(p) => Outcome {
      stage: Stage::Panic,
      detail: panic_text(p),
    },
    Ok(Err(e)) => Outcome {
      stage: Stage::RunErr,
      detail: format!("run Err: {:?}", e),
    },
    Ok(Ok(prover)) => match prover.verify() {
      Ok(()) => Outcome {
        stage: Stage::Ok,
        detail: "verify Ok".to_string(),
      },
      Err(errs) => Outcome {
        stage: Stage::VerifyErr,
        detail: format!("verify Err: {} 项", errs.len()),
      },
    },
  }
}

fn main() {
  let vectors: Vec<(&str, Vec<i64>)> = vec![
    ("L1", vec![0]),
    ("L2", vec![5, 1]),
    ("L3", vec![1, 2, 3]),
    ("L4", vec![3, 1, 2, 3]),
    ("L5", vec![-5, -1, -3, -16, 0]),
    ("L6", vec![1, 5, 2, 4, 3, 0]),
    ("L7", vec![7, 1, 2, 3, 4, 5, 6]),
    ("L8", vec![0, 1, 2, 3, 4, 5, 6, 7]),
    ("P0", vec![9, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("P1", vec![0, 9, 0, 0, 0, 0, 0, 0, 0]),
    ("P2", vec![0, 0, 9, 0, 0, 0, 0, 0, 0]),
    ("P3", vec![0, 0, 0, 9, 0, 0, 0, 0, 0]),
    ("P4", vec![0, 0, 0, 0, 9, 0, 0, 0, 0]),
    ("P5", vec![0, 0, 0, 0, 0, 9, 0, 0, 0]),
    ("P6", vec![0, 0, 0, 0, 0, 0, 9, 0, 0]),
    ("P7", vec![0, 0, 0, 0, 0, 0, 0, 9, 0]),
    ("P8", vec![0, 0, 0, 0, 0, 0, 0, 0, 9]),
    ("T1", vec![3, 3, 3, 3, 3, 3, 3, 3, 3]),
    ("T2", vec![0; 9]),
    ("T3", vec![-1, -2, -3, -4, -5, -6, -7, -8, -9]),
    ("T4", vec![1, 9, 0, 0, 9, 0, 0, 0, 0]),
  ];

  let mut runs = 0usize;
  let mut fails: Vec<String> = Vec::new();

  println!("device  = MockProver(k={})", K);
  println!("prereg  = manifests/p4b-b2-prereg.json");
  println!();

  // ---------- A：Max 独立 ----------
  println!("=== A: Max 独立（21 向量 x 4 列数 x 正负两臂）===");
  for cols in [3usize, 6, 9, 12] {
    let g = cols / 3;
    println!("-- num_cols={} (G={}) --", cols, g);
    for (name, v) in vectors.iter() {
      let expect = *v.iter().max().unwrap();
      let pos = dispatch_a(g, v, expect);
      let neg = dispatch_a(g, v, expect + 1);
      runs += 2;
      let ok = pos.stage == Stage::Ok && neg.stage != Stage::Ok;
      if !ok {
        fails.push(format!(
          "A/{}/{}: pos={:?} neg={:?}",
          cols, name, pos, neg
        ));
      }
      println!(
        "  {:<4} n={:<2} max={:<4} pos={:?} neg={:?} {}",
        name,
        v.len(),
        expect,
        pos.stage,
        neg.stage,
        if ok { "PASS" } else { "FAIL" }
      );
    }
  }

  // ---------- B：数值边界 ----------
  println!();
  println!("=== B: 数值边界（shift=1024, num_rows=1000, 6 列）===");
  // B1 [0,999] 差 999 -> 通过
  let b1 = run_classified(
    MaxTestBoundary {
      input: vec![0, 999],
      expected: 999,
    },
    to_field(999),
  );
  runs += 1;
  report("B1 [0,999] 期望 PASS", b1.stage == Stage::Ok, &b1, &mut fails);
  // B2 [0,1000] 差 1000 -> lookup 拒绝（约束失败，不是 panic）
  let b2 = run_classified(
    MaxTestBoundary {
      input: vec![0, 1000],
      expected: 1000,
    },
    to_field(1000),
  );
  runs += 1;
  report(
    "B2 [0,1000] 期望 lookup 拒绝",
    b2.stage == Stage::VerifyErr,
    &b2,
    &mut fails,
  );
  // B3 [0,1024] -> 宿主溢出 panic
  let b3 = run_classified(
    MaxTestBoundary {
      input: vec![0, 1024],
      expected: 1024,
    },
    to_field(1024),
  );
  runs += 1;
  report(
    "B3 [0,1024] 期望宿主 panic",
    b3.stage == Stage::Panic && b3.detail.is_empty(),
    &b3,
    &mut fails,
  );
  // B4 [1000,1000] 差 0 -> 通过
  let b4 = run_classified(
    MaxTestBoundary {
      input: vec![1000, 1000],
      expected: 1000,
    },
    to_field(1000),
  );
  runs += 1;
  report("B4 [1000,1000] 期望 PASS", b4.stage == Stage::Ok, &b4, &mut fails);

  // ---------- C1：空输入 ----------
  println!();
  println!("=== C1: 空输入（4 列数）===");
  for cols in [3usize, 6, 9, 12] {
    let g = cols / 3;
    let out = dispatch_a(g, &[][..], 0);
    runs += 1;
    report(
      &format!("C1/{}/空输入 期望 synthesize 拒绝", cols),
      out.stage == Stage::RunErr,
      &out,
      &mut fails,
    );
  }

  // ---------- C2：非法列数 ----------
  println!();
  println!("=== C2: 非法列数（必须打到 MaxChip::configure）===");
  for n in [4usize, 5, 7, 2, 0] {
    let out = dispatch_raw(n);
    runs += 1;
    let hit = out.stage == Stage::Panic
      && (out.detail.contains("multiple of 3") || out.detail.contains("at least 3"));
    report(&format!("C2/{} 列 期望 configure panic", n), hit, &out, &mut fails);
  }

  println!();
  println!("total runs = {} (预登记 181)", runs);
  if runs != 181 {
    println!("WARN: 次数与预登记不一致，结论不得沿用");
    fails.push(format!("runs {} != 181", runs));
  }
  println!("TOTAL: {}", if fails.is_empty() { "PASS" } else { "FAIL" });
  for f in fails.iter() {
    println!("  FAIL {}", f);
  }
  if !fails.is_empty() {
    std::process::exit(1);
  }
}

fn report(name: &str, ok: bool, out: &Outcome, fails: &mut Vec<String>) {
  println!(
    "  {:<34} {:?}  {}",
    name,
    out.stage,
    if ok { "PASS" } else { "FAIL" }
  );
  if !ok {
    fails.push(format!("{} -> {:?} ({})", name, out.stage, out.detail));
  }
}

fn dispatch_a(g: usize, v: &[i64], expected: i64) -> Outcome {
  match g {
    1 => run_classified(
      MaxTest::<1> {
        input: v.to_vec(),
        expected,
      },
      to_field(expected),
    ),
    2 => run_classified(
      MaxTest::<2> {
        input: v.to_vec(),
        expected,
      },
      to_field(expected),
    ),
    3 => run_classified(
      MaxTest::<3> {
        input: v.to_vec(),
        expected,
      },
      to_field(expected),
    ),
    4 => run_classified(
      MaxTest::<4> {
        input: v.to_vec(),
        expected,
      },
      to_field(expected),
    ),
    _ => panic!("dispatch_a: unsupported G={}", g),
  }
}

fn dispatch_raw(n: usize) -> Outcome {
  match n {
    0 => run_classified(
      MaxTestRaw::<0> {
        input: vec![0, 1],
        expected: 1,
      },
      to_field(1),
    ),
    2 => run_classified(
      MaxTestRaw::<2> {
        input: vec![0, 1],
        expected: 1,
      },
      to_field(1),
    ),
    4 => run_classified(
      MaxTestRaw::<4> {
        input: vec![0, 1],
        expected: 1,
      },
      to_field(1),
    ),
    5 => run_classified(
      MaxTestRaw::<5> {
        input: vec![0, 1],
        expected: 1,
      },
      to_field(1),
    ),
    7 => run_classified(
      MaxTestRaw::<7> {
        input: vec![0, 1],
        expected: 1,
      },
      to_field(1),
    ),
    _ => panic!("dispatch_raw: unsupported N={}", n),
  }
}
