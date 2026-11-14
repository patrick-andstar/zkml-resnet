//! Profiling entry point for standalone KZG verification.
//!
//! Drop-in replacement for `verify_circuit` that
//!   * never generates a trusted setup (missing/invalid params is a hard error),
//!   * takes an explicit params directory,
//!   * reports per-phase wall clock and process peak RSS.
//!
//! Verification logic, key/proof formats and the assert in `verify_kzg` are
//! unchanged: the same proof that `verify_circuit` accepts is accepted here.
//!
//! Usage:
//!   verify_profile <config> <vkey> <proof> <public_vals> kzg [params_dir]

use std::{fs::File, io::BufReader, time::Instant};

use halo2_proofs::{
  halo2curves::bn256::{Bn256, Fr},
  plonk::VerifyingKey,
  poly::{
    commitment::Params,
    kzg::{commitment::ParamsKZG, strategy::SingleStrategy},
  },
  transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
  SerdeFormat,
};

use zkml::{
  model::ModelCircuit,
  utils::{loader::load_config_msgpack, proving_kzg::verify_kzg},
};

fn vm_hwm_bytes() -> u64 {
  let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
  for line in s.lines() {
    if let Some(rest) = line.strip_prefix("VmHWM:") {
      let kb: u64 = rest
        .trim()
        .trim_end_matches(" kB")
        .trim()
        .parse()
        .unwrap_or(0);
      return kb * 1024;
    }
  }
  0
}

fn ms(d: std::time::Duration) -> f64 {
  d.as_secs_f64() * 1000.0
}

fn main() {
  let t_proc = Instant::now();

  let config_fname = std::env::args().nth(1).expect("config file path");
  let vkey_fname = std::env::args().nth(2).expect("verification key file path");
  let proof_fname = std::env::args().nth(3).expect("proof file path");
  let public_vals_fname = std::env::args().nth(4).expect("public values file path");
  let backend = std::env::args().nth(5).expect("kzg or ipa");
  let params_dir = std::env::args()
    .nth(6)
    .unwrap_or_else(|| "./params_kzg".to_string());

  if backend != "kzg" {
    panic!("verify_profile currently supports kzg only");
  }

  let t_config = Instant::now();
  let config = load_config_msgpack(&config_fname);
  let d_config = t_config.elapsed();
  let k = config.k as u32;
  let num_cols = config.num_cols;

  let t_circuit = Instant::now();
  let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);
  let d_circuit = t_circuit.elapsed();

  let params_path = format!("{}/{}.params", params_dir, k);
  if !std::path::Path::new(&params_path).exists() {
    panic!(
      "trusted setup not found at {} - refusing to generate a fresh SRS",
      params_path
    );
  }

  let t_params = Instant::now();
  let mut params_fs = File::open(&params_path).expect("couldn't open params");
  let params = ParamsKZG::<Bn256>::read(&mut params_fs).expect("Failed to read params");
  let d_params = t_params.elapsed();

  let t_vk = Instant::now();
  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open(&vkey_fname).unwrap()),
    SerdeFormat::RawBytes,
    (),
  )
  .unwrap();
  let d_vk = t_vk.elapsed();

  let t_proof = Instant::now();
  let proof = std::fs::read(&proof_fname).expect("couldn't read proof");
  let d_proof = t_proof.elapsed();

  let t_pub = Instant::now();
  let public_vals_u8 = std::fs::read(&public_vals_fname).expect("couldn't read public values");
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|chunk| {
      Fr::from_bytes(chunk.try_into().expect("conversion failed"))
        .expect("public value is not a canonical field element")
    })
    .collect();
  let d_pub = t_pub.elapsed();

  let t_init = Instant::now();
  let strategy = SingleStrategy::new(&params);
  let transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
  let d_init = t_init.elapsed();

  let t_core = Instant::now();
  verify_kzg(&params, &vk, strategy, &public_vals, transcript);
  let d_core = t_core.elapsed();

  let t_clean = Instant::now();
  drop(vk);
  drop(params);
  drop(circuit);
  let d_clean = t_clean.elapsed();

  let total = t_proc.elapsed();
  let accounted = d_config + d_circuit + d_params + d_vk + d_proof + d_pub + d_init + d_core + d_clean;

  println!(
    "{{\"k\":{},\"num_cols\":{},\"params_path\":\"{}\",\"proof_bytes\":{},\"vk_bytes\":{},\
\"public_vals\":{},\"config_ms\":{:.3},\"circuit_ms\":{:.3},\"params_ms\":{:.3},\
\"vk_ms\":{:.3},\"proof_ms\":{:.3},\"public_ms\":{:.3},\"init_ms\":{:.3},\"core_ms\":{:.3},\
\"cleanup_ms\":{:.3},\"accounted_ms\":{:.3},\"total_ms\":{:.3},\"residual_ms\":{:.3},\
\"peak_rss_bytes\":{}}}",
    k,
    num_cols,
    params_path,
    proof.len(),
    std::fs::metadata(&vkey_fname).map(|m| m.len()).unwrap_or(0),
    public_vals.len(),
    ms(d_config),
    ms(d_circuit),
    ms(d_params),
    ms(d_vk),
    ms(d_proof),
    ms(d_pub),
    ms(d_init),
    ms(d_core),
    ms(d_clean),
    ms(accounted),
    ms(total),
    ms(total.saturating_sub(accounted)),
    vm_hwm_bytes()
  );
  println!("Proof verified!");
}
