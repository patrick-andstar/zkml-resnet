//! B.3.1：model 加载期契约回归（不跑 MockProver、不 configure）。
//!
//! 只调用 `ModelCircuit::generate_from_msgpack`，验证 v4/v5 改的三段生命周期、
//! 双模式与 commit 判据。**不建大表**：generate 只解析与校验，configure 才建表。
//!
//! 覆盖 11 例：
//!   1 合法稀疏图（初始 {0,2} → L0 产 1）        期望 Ok
//!   2 缺依赖（FullWitness）                      期望 panic
//!   3 未来引用（L0 消费 L1 的产出）              期望 panic
//!   4 重复产出（两层产同一 ID）                  期望 panic
//!   5 config-only 结构合法                       期望 Ok
//!   6 FullWitness 下权重缺席                     期望 panic
//!   7 config-only：before 引用层产出             期望 panic（阶段检查）
//!   8 config-only：after 引用层产出              期望 Ok
//!   9 config-only：after 承诺未解析常量          期望 Ok + unresolved 含该 ID
//!  10 config-only：before 承诺未解析常量         期望 Ok + unresolved 含该 ID
//!  11 FullWitness：before 承诺初始张量           期望 Ok + unresolved 为空

use std::{
  collections::BTreeSet,
  panic::{catch_unwind, AssertUnwindSafe},
  sync::{Mutex, Once},
};

use halo2_proofs::halo2curves::bn256::Fr;

use zkml::{
  model::ModelCircuit,
  utils::loader::{LayerMsgpack, ModelMsgpack, TensorMsgpack},
};

static SLOT: Mutex<Option<String>> = Mutex::new(None);
static ONCE: Once = Once::new();

fn install_hook() {
  ONCE.call_once(|| {
    std::panic::set_hook(Box::new(|info| {
      let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
        (*s).to_string()
      } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
      } else {
        "<non-string>".to_string()
      };
      *SLOT.lock().unwrap() = Some(msg);
    }));
  });
}

fn take() -> String {
  SLOT.lock().unwrap().take().unwrap_or_default()
}

fn tensor(idx: i64, data: Vec<i64>) -> TensorMsgpack {
  TensorMsgpack {
    idx,
    shape: vec![data.len() as i64],
    data,
  }
}

fn layer(kind: &str, inp: Vec<i64>, out: Vec<i64>) -> LayerMsgpack {
  LayerMsgpack {
    layer_type: kind.to_string(),
    params: vec![],
    inp_idxes: inp,
    inp_shapes: vec![],
    out_idxes: out,
    out_shapes: vec![],
    mask: vec![],
  }
}

fn msg(
  tensors: Vec<TensorMsgpack>,
  layers: Vec<LayerMsgpack>,
  inp: Vec<i64>,
  out: Vec<i64>,
  cb: Option<Vec<Vec<i64>>>,
  ca: Option<Vec<Vec<i64>>>,
) -> ModelMsgpack {
  ModelMsgpack {
    global_sf: 1,
    k: 6, // num_rows = 55, max_val = 22, shift = 64，均合法
    num_cols: 6,
    inp_idxes: inp,
    out_idxes: out,
    tensors,
    layers,
    use_selectors: Some(true),
    commit_before: cb,
    commit_after: ca,
    bits_per_elem: None,
    num_random: None,
  }
}

fn run(cfg: ModelMsgpack) -> (bool, String, BTreeSet<i64>) {
  *SLOT.lock().unwrap() = None;
  let r = catch_unwind(AssertUnwindSafe(|| {
    ModelCircuit::<Fr>::generate_from_msgpack(cfg, false)
  }));
  match r {
    Ok(c) => (true, String::new(), c.unresolved_commit_ids.clone()),
    Err(_) => (false, take(), BTreeSet::new()),
  }
}

fn main() {
  install_hook();
  let mut fails = 0usize;
  let mut n = 0usize;

  let mut case = |name: &str, cfg: ModelMsgpack, expect_ok: bool, expect_msg: Option<&str>,
                  expect_unresolved: Option<Vec<i64>>| {
    n += 1;
    let (ok, pmsg, unres) = run(cfg);
    let mut good = ok == expect_ok;
    let mut why = String::new();
    if good {
      if let Some(m) = expect_msg {
        if ok {
          good = false;
          why = format!("期望 panic 但通过");
        } else if !pmsg.contains(m) {
          good = false;
          why = format!("panic 文本不含 {:?}: {:?}", m, pmsg);
        }
      }
      if let Some(ids) = expect_unresolved {
        if !ok {
          good = false;
        } else if !ids.iter().all(|i| unres.contains(i)) {
          good = false;
          why = format!("unresolved={:?}，缺 {:?}", unres, ids);
        }
      }
    } else {
      why = format!("期望 ok={} 实际 ok={} msg={:?}", expect_ok, ok, pmsg);
    }
    if !good {
      fails += 1;
    }
    println!(
      "  {:<44} {} {}",
      name,
      if good { "PASS" } else { "FAIL" },
      why
    );
    if !good {
      println!("STOP: 预登记要求立即停止。已完成 {}/11", n);
      std::process::exit(1);
    }
  };

  println!("B.3.1 model 加载期契约回归（k=6, cols=6, sf=1, Noop 层）");
  println!();

  // 1 合法稀疏图：初始 {0,2}，L0 输入 0 产出 1，最终输出 1
  case(
    "1 合法稀疏图（初始{0,2}→L0产1）",
    msg(
      vec![tensor(0, vec![1]), tensor(2, vec![2])],
      vec![layer("Noop", vec![0], vec![1])],
      vec![0],
      vec![1],
      None,
      None,
    ),
    true,
    None,
    Some(vec![]),
  );

  // 2 缺依赖：L0 消费 5，5 既非初始也无产出
  case(
    "2 缺依赖（FullWitness）",
    msg(
      vec![tensor(0, vec![1])],
      vec![layer("Noop", vec![5], vec![1])],
      vec![0],
      vec![1],
      None,
      None,
    ),
    false,
    Some("既不在初始张量、也没有更早的层产出"),
    None,
  );

  // 3 未来引用：L0 消费 7，7 由 L1 产出
  case(
    "3 未来引用（L0 消费 L1 产出）",
    msg(
      vec![tensor(0, vec![1])],
      vec![
        layer("Noop", vec![7], vec![1]),
        layer("Noop", vec![1], vec![7]),
      ],
      vec![0],
      vec![7],
      None,
      None,
    ),
    false,
    Some("不允许引用自身或未来层的输出"),
    None,
  );

  // 4 重复产出
  case(
    "4 重复产出（两层产同一 ID）",
    msg(
      vec![tensor(0, vec![1])],
      vec![
        layer("Noop", vec![0], vec![1]),
        layer("Noop", vec![0], vec![1]),
      ],
      vec![0],
      vec![1],
      None,
      None,
    ),
    false,
    Some("重复产出"),
    None,
  );

  // 5 config-only 结构合法：tensors 为空，L0 消费未知 99（当权重看）
  case(
    "5 config-only 结构合法",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![1])],
      vec![],
      vec![1],
      None,
      None,
    ),
    true,
    None,
    Some(vec![]),
  );

  // 6 FullWitness 下权重缺席（tensors 非空 => FullWitness）
  case(
    "6 FullWitness 下权重缺席",
    msg(
      vec![tensor(0, vec![1])],
      vec![layer("Noop", vec![99], vec![1])],
      vec![],
      vec![1],
      None,
      None,
    ),
    false,
    Some("既不在初始张量、也没有更早的层产出"),
    None,
  );

  // 7 config-only：before 引用层产出 -> 阶段检查必须拒
  case(
    "7 config-only：before 引用层产出",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      Some(vec![vec![5]]),
      None,
    ),
    false,
    Some("before 在 DAG 之前求值，不可能看到层输出"),
    None,
  );

  // 8 config-only：after 引用层产出 -> 允许
  case(
    "8 config-only：after 引用层产出",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      None,
      Some(vec![vec![5]]),
    ),
    true,
    None,
    Some(vec![]),
  );

  // 9 config-only：after 承诺未解析常量 -> 允许但记入 unresolved
  case(
    "9 config-only：after 承诺未解析常量",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      None,
      Some(vec![vec![77]]),
    ),
    true,
    None,
    Some(vec![77]),
  );

  // 10 config-only：before 承诺未解析常量 -> 允许但记入 unresolved
  case(
    "10 config-only：before 承诺未解析常量",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      Some(vec![vec![77]]),
      None,
    ),
    true,
    None,
    Some(vec![77]),
  );

  // 11 FullWitness：before 承诺真实初始张量 -> Ok，unresolved 必须为空
  case(
    "11 FullWitness：before 承诺初始张量",
    msg(
      vec![tensor(0, vec![3])],
      vec![layer("Noop", vec![0], vec![1])],
      vec![0],
      vec![1],
      Some(vec![vec![0]]),
      None,
    ),
    true,
    None,
    Some(vec![]),
  );

  println!();
  println!("total cases = {} (预登记 11)", n);
  if n != 11 {
    std::process::exit(1);
  }
  println!("TOTAL: PASS");
}
