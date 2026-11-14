//! P4-B.2 扩展矩阵（v2 驱动，修订版 rev2）。
//!
//! 与 `p4b_max_test.rs` 并存，不覆盖；旧测试的 27/27 结论不回改。
//! 预登记：`manifests/p4b-b2-prereg.json`、`过程文档/10`。
//!
//! 相对初版骨架的修订（按用户 2026-11-15 00:46 的六点）：
//!   1. 正臂必须 `Ok`；错误公共输出负臂必须 `VerifyErr`（不是"非 Ok"）；
//!      正臂失败立即停，不再跑负臂。
//!   2. 任一意外结果立即非零退出（fail-fast），删除"累计失败>3"。
//!   3. panic 来源用 **hook 记录的 file:line** 判定，不用空载荷/文本猜。
//!   4. B2 保存完整 verify 失败并核对确有 `Lookup` 型失败，不只看条数。
//!   5. C1 只认 `Error::Synthesis`，其它 RunErr 记 FAIL。
//!   6. 运行时用 `$CARGO_TARGET_DIR` 下的绝对路径，核对二进制摘要。

use std::{
  collections::BTreeSet,
  panic::{catch_unwind, AssertUnwindSafe},
  rc::Rc,
  sync::{Arc, Mutex, Once},
};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
  dev::{MockProver, VerifyFailure},
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
const EXPECTED_RUNS: usize = 181;

/// 与 model.rs `to_field` 同一编码。
fn to_field(x: i64) -> Fr {
  Fr::from((x + (1i64 << 31)) as u64) - Fr::from(1u64 << 31)
}

// ---------------- panic 来源捕获 ----------------

#[derive(Clone, Debug)]
struct PanicInfo {
  file: String,
  line: u32,
  col: u32,
  msg: String,
}

static PANIC_SLOT: Mutex<Option<PanicInfo>> = Mutex::new(None);
static HOOK_ONCE: Once = Once::new();

fn install_hook() {
  HOOK_ONCE.call_once(|| {
    std::panic::set_hook(Box::new(|info| {
      let (file, line, col) = match info.location() {
        Some(l) => (l.file().to_string(), l.line(), l.column()),
        None => ("<unknown>".to_string(), 0, 0),
      };
      let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
      } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
      } else {
        "<non-string>".to_string()
      };
      *PANIC_SLOT.lock().unwrap() = Some(PanicInfo {
        file,
        line,
        col,
        msg,
      });
    }));
  });
}

fn take_panic() -> PanicInfo {
  PANIC_SLOT.lock().unwrap().take().unwrap_or(PanicInfo {
    file: "<none>".to_string(),
    line: 0,
    col: 0,
    msg: "<none>".to_string(),
  })
}

// ---------------- 电路 ----------------

#[derive(Clone, Debug)]
struct TestConfig {
  gadget: GadgetConfig,
  input_col: Column<Advice>,
  instance: Column<Instance>,
}

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
  // 非法列数会在这里 panic —— C2 靠 hook 记录的 file 证明打到了这里。
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

struct MaxTest<const G: usize> {
  input: Vec<i64>,
}

impl<const G: usize> Circuit<Fr> for MaxTest<G> {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self { input: vec![0] }
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

struct MaxTestBoundary {
  input: Vec<i64>,
}

impl Circuit<Fr> for MaxTestBoundary {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self { input: vec![0] }
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

struct MaxTestRaw<const N: usize> {
  input: Vec<i64>,
}

impl<const N: usize> Circuit<Fr> for MaxTestRaw<N> {
  type Config = TestConfig;
  type FloorPlanner = SimpleFloorPlanner;
  type Params = ();

  fn without_witnesses(&self) -> Self {
    Self { input: vec![0] }
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

// ---------------- 结局分类 ----------------

#[derive(Debug, PartialEq)]
enum Kind {
  Ok,
  /// verify() 返回 Err
  VerifyErr,
  /// MockProver::run 返回 Err(Error::Synthesis)
  RunErrSynthesis,
  /// MockProver::run 返回其它 Err
  RunErrOther,
  /// 进程内 panic
  Panic,
}

struct Outcome {
  kind: Kind,
  /// verify 失败详情（逐条 Debug）
  failures: Vec<String>,
  /// 其中 Lookup 型失败条数
  lookup_failures: usize,
  run_err: String,
  panic: Option<PanicInfo>,
}

fn run_classified<C: Circuit<Fr>>(circuit: C, instance: Fr) -> Outcome {
  *PANIC_SLOT.lock().unwrap() = None;
  let res = catch_unwind(AssertUnwindSafe(|| {
    MockProver::run(K, &circuit, vec![vec![instance]])
  }));
  match res {
    Err(_) => Outcome {
      kind: Kind::Panic,
      failures: vec![],
      lookup_failures: 0,
      run_err: String::new(),
      panic: Some(take_panic()),
    },
    Ok(Err(Error::Synthesis)) => Outcome {
      kind: Kind::RunErrSynthesis,
      failures: vec![],
      lookup_failures: 0,
      run_err: "Error::Synthesis".to_string(),
      panic: None,
    },
    Ok(Err(e)) => Outcome {
      kind: Kind::RunErrOther,
      failures: vec![],
      lookup_failures: 0,
      run_err: format!("{:?}", e),
      panic: None,
    },
    Ok(Ok(prover)) => match prover.verify() {
      Ok(()) => Outcome {
        kind: Kind::Ok,
        failures: vec![],
        lookup_failures: 0,
        run_err: String::new(),
        panic: None,
      },
      Err(errs) => {
        let mut lookup = 0usize;
        let mut details = Vec::with_capacity(errs.len());
        for f in errs.iter() {
          if let VerifyFailure::Lookup { .. } = f {
            lookup += 1;
          }
          details.push(format!("{:?}", f));
        }
        Outcome {
          kind: Kind::VerifyErr,
          failures: details,
          lookup_failures: lookup,
          run_err: String::new(),
          panic: None,
        }
      }
    },
  }
}

// ---------------- 判定与 fail-fast ----------------

struct Ctx {
  runs: usize,
}

impl Ctx {
  fn die(&self, what: &str, out: &Outcome) -> ! {
    println!("  FAIL {} -> kind={:?}", what, out.kind);
    if let Some(p) = out.panic.as_ref() {
      println!(
        "       panic at {}:{}:{} msg={:?}",
        p.file, p.line, p.col, p.msg
      );
    }
    if !out.run_err.is_empty() {
      println!("       run_err={}", out.run_err);
    }
    for (i, f) in out.failures.iter().enumerate().take(5) {
      println!("       verify[{}]={}", i, f);
    }
    println!();
    println!("STOP: 预登记要求立即停止，不继续后续用例。已完成 runs={}", self.runs);
    std::process::exit(1);
  }
}

fn expect_ok(ctx: &mut Ctx, what: &str, out: Outcome) {
  ctx.runs += 1;
  if out.kind != Kind::Ok {
    ctx.die(what, &out);
  }
  println!("  {:<40} OK", what);
}

fn expect_verify_err(ctx: &mut Ctx, what: &str, out: Outcome, need_lookup: bool) {
  ctx.runs += 1;
  if out.kind != Kind::VerifyErr {
    ctx.die(what, &out);
  }
  if need_lookup && out.lookup_failures == 0 {
    ctx.die(&format!("{} [要求至少 1 条 Lookup 失败]", what), &out);
  }
  println!(
    "  {:<40} VerifyErr(总 {} 条, Lookup {} 条)",
    what,
    out.failures.len(),
    out.lookup_failures
  );
}

fn expect_panic_in_file(ctx: &mut Ctx, what: &str, out: Outcome, file_suffix: &str) {
  ctx.runs += 1;
  if out.kind != Kind::Panic {
    ctx.die(what, &out);
  }
  let p = out.panic.as_ref().unwrap();
  if !p.file.ends_with(file_suffix) {
    ctx.die(
      &format!("{} [panic 位置 {} 不以 {} 结尾]", what, p.file, file_suffix),
      &out,
    );
  }
  println!(
    "  {:<40} Panic@{}:{} msg={:?}",
    what, p.file, p.line, p.msg
  );
}

fn dispatch_a(g: usize, v: &[i64]) -> Outcome {
  dispatch_a_with(g, v, 0)
}

fn dispatch_a_with(g: usize, v: &[i64], instance: i64) -> Outcome {
  match g {
    1 => run_classified(
      MaxTest::<1> {
        input: v.to_vec(),
      },
      to_field(instance),
    ),
    2 => run_classified(
      MaxTest::<2> {
        input: v.to_vec(),
      },
      to_field(instance),
    ),
    3 => run_classified(
      MaxTest::<3> {
        input: v.to_vec(),
      },
      to_field(instance),
    ),
    4 => run_classified(
      MaxTest::<4> {
        input: v.to_vec(),
      },
      to_field(instance),
    ),
    _ => unreachable!(),
  }
}

fn dispatch_raw(n: usize) -> Outcome {
  match n {
    0 => run_classified(
      MaxTestRaw::<0> {
        input: vec![0, 1],
      },
      to_field(1),
    ),
    2 => run_classified(
      MaxTestRaw::<2> {
        input: vec![0, 1],
      },
      to_field(1),
    ),
    4 => run_classified(
      MaxTestRaw::<4> {
        input: vec![0, 1],
      },
      to_field(1),
    ),
    5 => run_classified(
      MaxTestRaw::<5> {
        input: vec![0, 1],
      },
      to_field(1),
    ),
    7 => run_classified(
      MaxTestRaw::<7> {
        input: vec![0, 1],
      },
      to_field(1),
    ),
    _ => unreachable!(),
  }
}

fn main() {
  install_hook();

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

  let mut ctx = Ctx { runs: 0 };

  println!("driver  = p4b_max_test_v2 (rev2)");
  println!("device  = MockProver(k={})", K);
  println!("prereg  = manifests/p4b-b2-prereg.json");
  println!();

  println!("=== A: Max 独立（21 向量 x 4 列数 x 正负两臂）===");
  for cols in [3usize, 6, 9, 12] {
    let g = cols / 3;
    println!("-- num_cols={} (G={}) --", cols, g);
    for (name, v) in vectors.iter() {
      let expect = *v.iter().max().unwrap();
      // 正臂：必须 Ok。失败立即停，不再跑负臂。
      let pos = dispatch_a_with(g, v, expect);
      expect_ok(
        &mut ctx,
        &format!("A/{}/{} 正臂 max={}", cols, name, expect),
        pos,
      );
      // 负臂：错误公共输出，必须 VerifyErr（不是"非 Ok"）。
      let neg = dispatch_a_with(g, v, expect + 1);
      expect_verify_err(&mut ctx, &format!("A/{}/{} 负臂 max+1", cols, name), neg, false);
    }
  }

  println!();
  println!("=== B: 数值边界（shift=1024, num_rows=1000, 6 列）===");
  expect_ok(
    &mut ctx,
    "B1 [0,999] 差 999 应通过",
    run_classified(
      MaxTestBoundary {
        input: vec![0, 999],
      },
      to_field(999),
    ),
  );
  expect_verify_err(
    &mut ctx,
    "B2 [0,1000] 差 1000 应 Lookup 拒绝",
    run_classified(
      MaxTestBoundary {
        input: vec![0, 1000],
      },
      to_field(1000),
    ),
    true,
  );
  expect_panic_in_file(
    &mut ctx,
    "B3 [0,1024] 宿主溢出 panic",
    run_classified(
      MaxTestBoundary {
        input: vec![0, 1024],
      },
      to_field(1024),
    ),
    "gadgets/gadget.rs",
  );
  expect_ok(
    &mut ctx,
    "B4 [1000,1000] 差 0 应通过",
    run_classified(
      MaxTestBoundary {
        input: vec![1000, 1000],
      },
      to_field(1000),
    ),
  );

  println!();
  println!("=== C1: 空输入（必须 Error::Synthesis）===");
  for cols in [3usize, 6, 9, 12] {
    let g = cols / 3;
    let out = dispatch_a(g, &[][..]);
    ctx.runs += 1;
    if out.kind != Kind::RunErrSynthesis {
      ctx.die(&format!("C1/{}/空输入 期望 Error::Synthesis", cols), &out);
    }
    println!("  {:<40} RunErr(Synthesis)", format!("C1/{}/空输入", cols));
  }

  println!();
  println!("=== C2: 非法列数（必须打到 MaxChip::configure）===");
  for n in [4usize, 5, 7, 2, 0] {
    let out = dispatch_raw(n);
    expect_panic_in_file(&mut ctx, &format!("C2/{} 列", n), out, "gadgets/max.rs");
  }

  println!();
  println!("total runs = {} (预登记 {})", ctx.runs, EXPECTED_RUNS);
  if ctx.runs != EXPECTED_RUNS {
    println!("STOP: 次数与预登记不一致，结论不得沿用");
    std::process::exit(1);
  }
  println!("TOTAL: PASS（所有用例符合预登记）");
}
