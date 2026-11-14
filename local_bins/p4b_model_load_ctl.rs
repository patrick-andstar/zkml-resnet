//! B.3.1 负向对照驱动：不引用 v5 专有字段（unresolved_commit_ids），
//! 因此可以在 v3 / v4 / v5 上分别编译，用来检验主驱动的 11 个用例是否真有区分力。
//!
//! 只打印每例的 ok/panic 与 panic 文本，不做断言；断言在报告里人工对照。

use std::{
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
    k: 6,
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

fn probe(name: &str, cfg: ModelMsgpack) {
  *SLOT.lock().unwrap() = None;
  let r = catch_unwind(AssertUnwindSafe(|| {
    ModelCircuit::<Fr>::generate_from_msgpack(cfg, false)
  }));
  match r {
    Ok(_) => println!("  {:<34} Ok", name),
    Err(_) => {
      let m = take();
      let short: String = m.chars().take(46).collect();
      println!("  {:<34} Panic  {}", name, short);
    }
  }
}

fn main() {
  install_hook();
  println!("control driver（不引用 unresolved_commit_ids）");

  probe(
    "c1 稀疏图 {0,2}->L0产1",
    msg(
      vec![tensor(0, vec![1]), tensor(2, vec![2])],
      vec![layer("Noop", vec![0], vec![1])],
      vec![0],
      vec![1],
      None,
      None,
    ),
  );
  probe(
    "c2 缺依赖",
    msg(
      vec![tensor(0, vec![1])],
      vec![layer("Noop", vec![5], vec![1])],
      vec![0],
      vec![1],
      None,
      None,
    ),
  );
  probe(
    "c3 未来引用",
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
  );
  probe(
    "c4 重复产出",
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
  );
  probe(
    "c5 config-only 结构合法",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![1])],
      vec![],
      vec![1],
      None,
      None,
    ),
  );
  probe(
    "c6 FullWitness 权重缺席",
    msg(
      vec![tensor(0, vec![1])],
      vec![layer("Noop", vec![99], vec![1])],
      vec![],
      vec![1],
      None,
      None,
    ),
  );
  probe(
    "c7 config-only before 引用产出",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      Some(vec![vec![5]]),
      None,
    ),
  );
  probe(
    "c8 config-only after 引用产出",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      None,
      Some(vec![vec![5]]),
    ),
  );
  probe(
    "c9 config-only after 未解析常量",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      None,
      Some(vec![vec![77]]),
    ),
  );
  probe(
    "c10 config-only before 未解析常量",
    msg(
      vec![],
      vec![layer("Noop", vec![99], vec![5])],
      vec![],
      vec![5],
      Some(vec![vec![77]]),
      None,
    ),
  );
  probe(
    "c11 FullWitness before 初始张量",
    msg(
      vec![tensor(0, vec![3])],
      vec![layer("Noop", vec![0], vec![1])],
      vec![0],
      vec![1],
      Some(vec![vec![0]]),
      None,
    ),
  );
}
