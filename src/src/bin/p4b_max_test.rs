//! P4-B 最小用例：MaxChip 归约的数值与约束双验收。
//!
//! 只覆盖 MaxChip 本身，不构造完整模型：
//!   * 输入按 model.rs 的 `to_field` 同一编码写入 advice（含负值）；
//!   * 输出用 `constrain_instance` 钉住宿主参考 `max`，算错必然让 MockProver 失败，
//!     不依赖人工读值；
//!   * 每个用例再跑一次「期望值故意 +1」的负向对照，证明约束确实在起作用
//!     （否则「通过」可能只是因为约束根本没生效）。
//!
//! 覆盖 plan §P4-B.2 要求的：[0,1] 最小、9 元素逐位置最大、并列最大、全零、
//! 合法负值、非整行/奇数长度、单元素。

use std::{collections::BTreeSet, rc::Rc, sync::Arc};

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
const SHIFT: i64 = 64;

/// 与 model.rs `to_field` 同一编码：F::from((x + 2^31) as u64) - F::from(2^31)。
fn to_field(x: i64) -> Fr {
  Fr::from((x + (1i64 << 31)) as u64) - Fr::from(1u64 << 31)
}

#[derive(Clone, Debug)]
struct TestConfig {
  gadget: GadgetConfig,
  input_col: Column<Advice>,
  instance: Column<Instance>,
}

/// G = 每行输出个数；advice 列数 = 3G。
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
    let num_cols = 3 * G;
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
    gc.shift_min_val = SHIFT;
    gc.used_gadgets = Arc::new(
      [GadgetType::Max, GadgetType::InputLookup]
        .into_iter()
        .collect::<BTreeSet<_>>(),
    );

    gc = InputLookupChip::<Fr>::configure(meta, gc);
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

  fn synthesize(
    &self,
    config: Self::Config,
    mut layouter: impl Layouter<Fr>,
  ) -> Result<(), Error> {
    let cells: Vec<AssignedCell<Fr, Fr>> = layouter.assign_region(
      || "p4b inputs",
      |mut region| {
        let mut v = Vec::with_capacity(self.input.len());
        for (i, x) in self.input.iter().enumerate() {
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
}

fn run<const G: usize>(input: &[i64], expected: i64) -> (bool, usize) {
  let circuit = MaxTest::<G> {
    input: input.to_vec(),
    expected,
  };
  let prover = MockProver::run(K, &circuit, vec![vec![to_field(expected)]])
    .expect("MockProver::run 失败（布局层面）");
  match prover.verify() {
    Ok(()) => (true, 0),
    Err(errs) => (false, errs.len()),
  }
}

fn run_at(num_cols: usize, input: &[i64], expected: i64) -> (bool, usize) {
  match num_cols {
    3 => run::<1>(input, expected),
    6 => run::<2>(input, expected),
    12 => run::<4>(input, expected),
    _ => panic!("unsupported num_cols {}", num_cols),
  }
}

fn main() {
  let vectors: Vec<(&str, Vec<i64>)> = vec![
    ("最小 [0,1]", vec![0, 1]),
    ("9元素-最大在首", vec![9, 0, 0, 0, 0, 0, 0, 0, 0]),
    ("9元素-最大在尾", vec![0, 0, 0, 0, 0, 0, 0, 0, 9]),
    ("反例 x1=1 其余0", vec![0, 1, 0, 0, 0, 0, 0, 0, 0]),
    ("并列最大", vec![3, 3, 3, 3, 3, 3, 3, 3, 3]),
    ("全零", vec![0; 9]),
    ("合法负值", vec![-5, -1, -3, -16, 0]),
    ("奇数长度", vec![1, 2, 3]),
    ("单元素", vec![7]),
  ];

  println!("device   = MockProver(k={})", K);
  println!("encoding = to_field, shift_min_val={}, num_rows={}", SHIFT, NUM_ROWS);
  println!();

  let mut all_ok = true;
  for num_cols in [3usize, 6, 12] {
    println!("=== num_cols={} (G={}) ===", num_cols, num_cols / 3);
    for (name, v) in vectors.iter() {
      let expected = *v.iter().max().unwrap();
      let (ok, nerr) = run_at(num_cols, v, expected);
      let (ok_bad, _) = run_at(num_cols, v, expected + 1);
      let pass = ok && !ok_bad;
      if !pass {
        all_ok = false;
      }
      println!(
        "  {:<18} n={:<2} max={:<4} {:>4}  (错误行数={}, 错期望被拒={})",
        name,
        v.len(),
        expected,
        if pass { "PASS" } else { "FAIL" },
        nerr,
        !ok_bad
      );
    }
    println!();
  }

  println!("TOTAL: {}", if all_ok { "PASS" } else { "FAIL" });
  if !all_ok {
    std::process::exit(1);
  }
}
