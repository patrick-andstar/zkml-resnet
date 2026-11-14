// ============================================================================
// P4-A 候选修复：MaxChip 归约。**尚未编译、尚未运行**，仅静态审查用。
//
// 原实现（src/gadgets/max.rs）的两个独立缺陷：
//   D1 覆盖不全：一行把位置 i 与位置 G+i 配对（G = columns.len()/3），
//      树形归约结束后剩 G 个输出就停止，最后 `outputs[0]` 只保留第 0 组。
//      因此返回值 = 每 G 个位置取一个的「同余类」最大，不是全窗口最大。
//      G=1（3 列）时同余类只有一个，恰好正确；G>=2（>=6 列）必错。
//      G=2、n=9 时返回值恒等于 max(x0,x2,x4,x6,x8)，奇数下标全丢。
//   D2 终止条件错误：`num_iters = div_ceil + num_inputs_per_row` 是经验上界，
//      每轮用 `first` 补齐后输出个数在 G 处成为不动点，永远到不了 1；
//      末尾直接取第 0 个而不是「合并到最后 1 个」。
//
// 修复策略：
//   阶段一（树形）：只要剩余个数 L > G 就做一次行归约，L -> ceil(L/(2G))*G，
//                  严格下降；补齐值取集合内已有元素，max 不变。
//   阶段二（collapse）：两两合并，每次用一行把 (a,b) 放在同一组内，只取第 0 组
//                  输出，直到剩 1 个。L -> ceil(L/2)，严格下降。
//   单元素直接返回，不占行；空输入按契约拒绝，不再 `inputs[0]` panic。
//
// 正确性论证（见 过程文档/05-P4A-契约与候选修复.md §3）：
//   I1 不变式：cur 的每个 cell 都是「原始输入某个非空子集的最大值」，
//      故 cur 的最大值 <= 全窗口最大，且全集最大一定被某个 cell 取到。
//   I2 补齐只复制 cur[0]（集合内元素），既不引入更大的值，也不删除任何原元素。
//   I3 一行归约把 2G 个位置按 (i, G+i) 两两配对，覆盖全部 2G 个输入且不重不漏，
//      输出仍是「子集最大」，故集合最大不变。
//   I4 阶段二每步 max2(a,b) 精确等于 max(a,b)，故集合最大不变，长度减半。
//   由 I1~I4，返回的唯一 cell 等于全窗口最大。
// ============================================================================

use std::{marker::PhantomData, rc::Rc};

use halo2_proofs::{
  circuit::{AssignedCell, Layouter, Region},
  halo2curves::ff::PrimeField,
  plonk::{ConstraintSystem, Error},
  poly::Rotation,
};

use crate::gadgets::gadget::convert_to_u64;

use super::gadget::{Gadget, GadgetConfig, GadgetType};

pub struct MaxChip<F: PrimeField> {
  config: Rc<GadgetConfig>,
  _marker: PhantomData<F>,
}

impl<F: PrimeField> MaxChip<F> {
  pub fn construct(config: Rc<GadgetConfig>) -> Self {
    Self {
      config,
      _marker: PhantomData,
    }
  }

  pub fn num_cols_per_op() -> usize {
    3
  }

  pub fn configure(meta: &mut ConstraintSystem<F>, gadget_config: GadgetConfig) -> GadgetConfig {
    let columns = gadget_config.columns;
    let tables = gadget_config.tables;

    // 契约：列数必须是 3 的倍数且至少 3。原实现用 `columns.len() / 3` 静默截断，
    // 例如 7 列会被当成 2 组、第 7 列永不使用且不报错。这里改为构造期硬失败。
    assert!(
      columns.len() >= Self::num_cols_per_op(),
      "MaxChip requires at least {} advice columns, got {}",
      Self::num_cols_per_op(),
      columns.len()
    );
    assert_eq!(
      columns.len() % Self::num_cols_per_op(),
      0,
      "MaxChip requires columns.len() to be a multiple of {}, got {}",
      Self::num_cols_per_op(),
      columns.len()
    );

    let selector = meta.complex_selector();

    let inp_lookup = tables.get(&GadgetType::InputLookup).unwrap()[0];

    meta.create_gate("max arithmetic", |meta| {
      let s = meta.query_selector(selector);
      let mut constraints = vec![];
      for i in 0..columns.len() / Self::num_cols_per_op() {
        let offset = i * Self::num_cols_per_op();
        let inp1 = meta.query_advice(columns[offset], Rotation::cur());
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        constraints.push(s.clone() * (inp1 - outp.clone()) * (inp2 - outp))
      }
      constraints
    });

    for idx in 0..columns.len() / Self::num_cols_per_op() {
      meta.lookup("max inp1", |meta| {
        let s = meta.query_selector(selector);
        let offset = idx * Self::num_cols_per_op();
        let inp1 = meta.query_advice(columns[offset], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s * (outp - inp1), inp_lookup)]
      });
      meta.lookup("max inp2", |meta| {
        let s = meta.query_selector(selector);
        let offset = idx * Self::num_cols_per_op();
        let inp2 = meta.query_advice(columns[offset + 1], Rotation::cur());
        let outp = meta.query_advice(columns[offset + 2], Rotation::cur());

        vec![(s * (outp - inp2), inp_lookup)]
      });
    }

    let mut selectors = gadget_config.selectors;
    selectors.insert(GadgetType::Max, vec![selector]);

    GadgetConfig {
      columns,
      selectors,
      tables,
      ..gadget_config
    }
  }

  /// 用一行精确计算 max(a, b)，只取第 0 组输出。
  ///
  /// 布局（长度 2G）：位置 0 放 a、位置 G 放 b、其余位置全部填 a。
  /// 第 0 组 = (位置 0, 位置 G) = (a, b) -> out[0] = max(a, b)
  /// 第 g>0 组 = (a, a) -> out[g] = a（无害，丢弃）
  ///
  /// 注意：不能简单地把 [a, b] 补齐后送进行归约——那样 a、b 会落在**不同组**的
  /// 同一侧，永远不被比较（这正是原实现在 G>=2 时丢一半输入的机理）。
  fn max_of_two(
    &self,
    mut layouter: impl Layouter<F>,
    a: &AssignedCell<F, F>,
    b: &AssignedCell<F, F>,
  ) -> Result<AssignedCell<F, F>, Error> {
    let g = self.num_outputs_per_row();
    let mut row: Vec<&AssignedCell<F, F>> = Vec::with_capacity(2 * g);
    row.push(a);
    for _ in 1..g {
      row.push(a);
    }
    row.push(b);
    for _ in 1..g {
      row.push(a);
    }
    debug_assert_eq!(row.len(), self.num_inputs_per_row());

    let outp = self.op_aligned_rows(
      layouter.namespace(|| "max collapse row"),
      &vec![row],
      &vec![],
    )?;
    outp
      .into_iter()
      .next()
      .ok_or(Error::Synthesis)
  }
}

impl<F: PrimeField> Gadget<F> for MaxChip<F> {
  fn name(&self) -> String {
    "max".to_string()
  }

  fn num_cols_per_op(&self) -> usize {
    3
  }

  fn num_inputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op() * 2
  }

  fn num_outputs_per_row(&self) -> usize {
    self.config.columns.len() / self.num_cols_per_op()
  }

  fn op_row_region(
    &self,
    region: &mut Region<F>,
    row_offset: usize,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    _single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    assert_eq!(vec_inputs.len(), 1);
    let inp = &vec_inputs[0];
    assert_eq!(inp.len(), self.num_inputs_per_row());

    if self.config.use_selectors {
      let selector = self.config.selectors.get(&GadgetType::Max).unwrap()[0];
      selector.enable(region, row_offset)?;
    }

    let min_val_pos = F::from((-self.config.shift_min_val) as u64);

    let mut outp = vec![];

    let g = self.num_outputs_per_row();
    let (i1, i2) = inp.split_at(g);
    for (idx, (inp1, inp2)) in i1.iter().zip(i2.iter()).enumerate() {
      let offset = idx * self.num_cols_per_op();
      inp1
        .copy_advice(|| "", region, self.config.columns[offset], row_offset)
        .unwrap();
      inp2
        .copy_advice(|| "", region, self.config.columns[offset + 1], row_offset)
        .unwrap();

      let max = inp1.value().zip(inp2.value()).map(|(a, b)| {
        let a = convert_to_u64(&(*a + min_val_pos));
        let b = convert_to_u64(&(*b + min_val_pos));
        let max = a.max(b);
        let max = F::from(max) - min_val_pos;
        max
      });

      let res = region
        .assign_advice(|| "", self.config.columns[offset + 2], row_offset, || max)
        .unwrap();
      outp.push(res);
    }

    Ok(outp)
  }

  fn forward(
    &self,
    mut layouter: impl Layouter<F>,
    vec_inputs: &Vec<Vec<&AssignedCell<F, F>>>,
    single_inputs: &Vec<&AssignedCell<F, F>>,
  ) -> Result<Vec<AssignedCell<F, F>>, Error> {
    // ---- 前置契约（违反即拒绝，不再 panic 或静默截断） ----
    if vec_inputs.len() != 1 {
      return Err(Error::Synthesis);
    }
    let g = self.num_outputs_per_row();
    if g == 0 || self.num_inputs_per_row() != 2 * g {
      return Err(Error::Synthesis);
    }
    let inputs = &vec_inputs[0];
    if inputs.is_empty() {
      return Err(Error::Synthesis);
    }
    // 单元素窗口：直接返回，不占任何 gadget 行
    if inputs.len() == 1 {
      return Ok(vec![inputs[0].clone()]);
    }

    let in_per_row = self.num_inputs_per_row();
    let mut cur: Vec<AssignedCell<F, F>> = inputs.iter().map(|c| (*c).clone()).collect();

    // ---- 阶段一：树形归约，L > G 时 L -> ceil(L/(2G))*G 严格下降 ----
    while cur.len() > g {
      while cur.len() % in_per_row != 0 {
        let pad = cur[0].clone(); // 补齐值取集合内元素，max 不变
        cur.push(pad);
      }
      let refs: Vec<&AssignedCell<F, F>> = cur.iter().collect();
      cur = self.op_aligned_rows(
        layouter.namespace(|| "max tree"),
        &vec![refs],
        single_inputs,
      )?;
    }

    // ---- 阶段二：collapse 到唯一输出，L -> ceil(L/2) 严格下降 ----
    while cur.len() > 1 {
      let mut nxt: Vec<AssignedCell<F, F>> = Vec::with_capacity((cur.len() + 1) / 2);
      let mut i = 0;
      while i < cur.len() {
        let b = if i + 1 < cur.len() {
          &cur[i + 1]
        } else {
          &cur[i] // 奇数个时最后一个与自己合并，等价于原样保留
        };
        let merged = self.max_of_two(layouter.namespace(|| "max collapse"), &cur[i], b)?;
        nxt.push(merged);
        i += 2;
      }
      cur = nxt;
    }

    Ok(vec![cur.remove(0)])
  }
}
