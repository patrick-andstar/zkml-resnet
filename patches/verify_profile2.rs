//! P2A measurement entry point: three parameter-read modes, optional raw-error
//! diagnosis, and an optional read-trace that identifies proof element offsets.
//!
//! Everything about the proof system is unchanged: same `ParamsKZG::read`,
//! same `SerdeFormat::RawBytes`, same VK/proof/public-values decoding, same
//! `verify_proof` call and the same assert as upstream `verify_kzg`.
//! The only differences are instrumentation and how the parameters file
//! reaches `ParamsKZG::read`:
//!
//!   file      - `ParamsKZG::read(&mut File)`                 (upstream style)
//!   bufreader - `ParamsKZG::read(&mut BufReader<File>)`      (1 MiB buffer)
//!   memory    - `std::fs::read` then `ParamsKZG::read(&mut Cursor)`
//!
//! `--diag` prints the real `Result` from `verify_proof` instead of only
//! asserting, so negative tests can distinguish "decode/parse failed" from
//! "decoded fine but the proof does not verify". It does not alter any check.
//!
//! `--trace-reads` interposes a pass-through `Read` adapter that records the
//! offset and length of every `read_exact` the transcript performs. The adapter
//! only copies bytes through; the decoding path and every check stay the same.
//! Its purpose is evidence: it shows how many elements were consumed and at
//! which offsets, and whether a tampered proof was still fully decoded before
//! being rejected. It deliberately does **not** label an element as a point or
//! a scalar — in this dependency set a compressed point encoding and a field
//! element encoding are both 32 bytes, so length alone cannot decide the kind.
//! The proof is verified exactly once.
//!
//! Usage:
//!   verify_profile2 [--read-mode file|bufreader|memory] [--diag] [--trace-reads] \
//!     <config> <vkey> <proof> <public_vals> kzg [params_dir]

use std::{
  fs::File,
  io::{BufReader, Cursor, Read},
  time::Instant,
};

use halo2_proofs::{
  halo2curves::bn256::{Bn256, Fr, G1Affine},
  plonk::{verify_proof, Error as PlonkError, VerifyingKey},
  poly::{
    commitment::Params,
    kzg::{
      commitment::{KZGCommitmentScheme, ParamsKZG},
      multiopen::VerifierSHPLONK,
      strategy::SingleStrategy,
    },
  },
  transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
  SerdeFormat,
};

use zkml::{model::ModelCircuit, utils::loader::load_config_msgpack};

const BUF_CAPACITY: usize = 1 << 20;

/// Pass-through reader that records every `read_exact` (offset, length).
/// It never inspects or modifies the bytes it forwards.
struct TraceReader<'a> {
  buf: &'a [u8],
  pos: usize,
  log: Vec<(usize, usize)>,
}

impl<'a> TraceReader<'a> {
  fn new(buf: &'a [u8]) -> Self {
    Self {
      buf,
      pos: 0,
      log: Vec::new(),
    }
  }
}

impl<'a> Read for TraceReader<'a> {
  fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
    let n = std::cmp::min(out.len(), self.buf.len() - self.pos);
    out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
    self.pos += n;
    Ok(n)
  }

  fn read_exact(&mut self, out: &mut [u8]) -> std::io::Result<()> {
    self.log.push((self.pos, out.len()));
    if self.buf.len() - self.pos < out.len() {
      return Err(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "trace reader hit end of proof",
      ));
    }
    out.copy_from_slice(&self.buf[self.pos..self.pos + out.len()]);
    self.pos += out.len();
    Ok(())
  }
}

fn vm_hwm_bytes() -> u64 {
  let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
  for line in s.lines() {
    if let Some(rest) = line.strip_prefix("VmHWM:") {
      return rest
        .trim()
        .trim_end_matches(" kB")
        .trim()
        .parse::<u64>()
        .unwrap_or(0)
        * 1024;
    }
  }
  0
}

fn cpus_allowed() -> String {
  let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
  for line in s.lines() {
    if let Some(rest) = line.strip_prefix("Cpus_allowed_list:") {
      return rest.trim().to_string();
    }
  }
  "unknown".to_string()
}

fn threads_now() -> usize {
  std::fs::read_dir("/proc/self/task")
    .map(|d| d.count())
    .unwrap_or(0)
}

fn ms(d: std::time::Duration) -> f64 {
  d.as_secs_f64() * 1000.0
}

/// The unchanged verification call, generic only over how the proof bytes are
/// read. Returns the raw result plus the init/core durations.
fn do_verify<R: Read>(
  params: &ParamsKZG<Bn256>,
  vk: &VerifyingKey<G1Affine>,
  public_vals: &[Fr],
  reader: R,
) -> (Result<(), PlonkError>, std::time::Duration, std::time::Duration) {
  let t0 = Instant::now();
  let strategy = SingleStrategy::new(params);
  let mut transcript = Blake2bRead::<R, G1Affine, Challenge255<G1Affine>>::init(reader);
  let d_init = t0.elapsed();

  let t0 = Instant::now();
  let result = verify_proof::<
    KZGCommitmentScheme<Bn256>,
    VerifierSHPLONK<'_, Bn256>,
    Challenge255<G1Affine>,
    Blake2bRead<R, G1Affine, Challenge255<G1Affine>>,
    SingleStrategy<'_, Bn256>,
  >(params, vk, strategy, &[&[public_vals]], &mut transcript);
  let d_core = t0.elapsed();
  (result, d_init, d_core)
}

fn main() {
  let t_proc = Instant::now();
  let mut args: Vec<String> = std::env::args().skip(1).collect();
  let mut read_mode = "file".to_string();
  let mut diag = false;
  let mut trace_reads = false;
  while let Some(a) = args.first().cloned() {
    if a == "--diag" {
      diag = true;
      args.remove(0);
    } else if a == "--trace-reads" {
      trace_reads = true;
      args.remove(0);
    } else if a == "--read-mode" {
      args.remove(0);
      read_mode = args.remove(0);
    } else {
      break;
    }
  }
  if args.len() < 5 {
    panic!("need <config> <vkey> <proof> <public_vals> kzg [params_dir]");
  }
  let config_fname = &args[0];
  let vkey_fname = &args[1];
  let proof_fname = &args[2];
  let public_vals_fname = &args[3];
  let backend = &args[4];
  let params_dir = args
    .get(5)
    .cloned()
    .unwrap_or_else(|| "./params_kzg".to_string());
  if backend != "kzg" {
    panic!("verify_profile2 supports kzg only");
  }
  let affinity_start = cpus_allowed();

  let t0 = Instant::now();
  let config = load_config_msgpack(config_fname);
  let d_config = t0.elapsed();
  let k = config.k as u32;
  let num_cols = config.num_cols;

  let t0 = Instant::now();
  let circuit = ModelCircuit::<Fr>::generate_from_msgpack(config, false);
  let d_circuit = t0.elapsed();

  let params_path = format!("{}/{}.params", params_dir, k);
  if !std::path::Path::new(&params_path).exists() {
    panic!(
      "trusted setup not found at {} - refusing to generate a fresh SRS",
      params_path
    );
  }

  let t0 = Instant::now();
  let mut params_file = File::open(&params_path).expect("couldn't open params");
  let d_params_open = t0.elapsed();

  // Memory mode: bulk read is timed separately from parsing, and the buffer
  // stays alive until the cleanup window so its release is inside the request.
  let mut preloaded: Option<Vec<u8>> = None;
  let d_preload = if read_mode == "memory" {
    let t = Instant::now();
    let mut buf = Vec::new();
    params_file
      .read_to_end(&mut buf)
      .expect("Failed to read params bytes");
    preloaded = Some(buf);
    t.elapsed()
  } else {
    std::time::Duration::ZERO
  };

  let t0 = Instant::now();
  let params = match read_mode.as_str() {
    "file" => ParamsKZG::<Bn256>::read(&mut params_file).expect("Failed to read params"),
    "bufreader" => {
      let mut r = BufReader::with_capacity(BUF_CAPACITY, params_file);
      ParamsKZG::<Bn256>::read(&mut r).expect("Failed to read params")
    }
    "memory" => {
      let mut c = Cursor::new(preloaded.as_ref().unwrap().as_slice());
      ParamsKZG::<Bn256>::read(&mut c).expect("Failed to read params")
    }
    other => panic!("unknown read mode: {}", other),
  };
  let d_params_parse = t0.elapsed();

  let t0 = Instant::now();
  let vk = VerifyingKey::read::<BufReader<File>, ModelCircuit<Fr>>(
    &mut BufReader::new(File::open(vkey_fname).unwrap()),
    SerdeFormat::RawBytes,
    (),
  )
  .unwrap();
  let d_vk = t0.elapsed();

  let t0 = Instant::now();
  let proof = std::fs::read(proof_fname).expect("couldn't read proof");
  let d_proof = t0.elapsed();

  let t0 = Instant::now();
  let public_vals_u8 = std::fs::read(public_vals_fname).expect("couldn't read public values");
  let public_vals: Vec<Fr> = public_vals_u8
    .chunks(32)
    .map(|c| {
      Fr::from_bytes(c.try_into().expect("public value chunk is not 32 bytes"))
        .expect("public value is not a canonical field element")
    })
    .collect();
  let d_public = t0.elapsed();

  let proof_len = proof.len();
  let (result, d_init, d_core) = if trace_reads {
    // `&mut TraceReader` is itself a `Read`, so the log survives the call.
    let mut tr = TraceReader::new(&proof);
    let (r, di, dc) = do_verify(&params, &vk, &public_vals, &mut tr);
    let entries: Vec<String> = tr
      .log
      .iter()
      .map(|(o, l)| format!("[{},{}]", o, l))
      .collect();
    println!(
      "TRACE {{\"n_reads\":{},\"bytes_consumed\":{},\"proof_bytes\":{},\"reads\":[{}]}}",
      tr.log.len(),
      tr.pos,
      proof_len,
      entries.join(",")
    );
    (r, di, dc)
  } else {
    do_verify(&params, &vk, &public_vals, &proof[..])
  };

  let threads = threads_now();
  let affinity_end = cpus_allowed();
  let hwm = vm_hwm_bytes();

  let t0 = Instant::now();
  drop(preloaded);
  drop(vk);
  drop(params);
  drop(circuit);
  let d_cleanup = t0.elapsed();

  let total = t_proc.elapsed();
  let accounted = d_config
    + d_circuit
    + d_params_open
    + d_preload
    + d_params_parse
    + d_vk
    + d_proof
    + d_public
    + d_init
    + d_core
    + d_cleanup;

  let verdict = match &result {
    Ok(()) => "Ok".to_string(),
    Err(e) => format!("{:?}", e).replace('"', "'"),
  };

  println!(
    "{{\"read_mode\":\"{}\",\"diag\":{},\"trace_reads\":{},\"k\":{},\"num_cols\":{},\
\"proof_bytes\":{},\"public_vals\":{},\"config_ms\":{:.3},\"circuit_ms\":{:.3},\
\"params_open_ms\":{:.3},\"params_parse_ms\":{:.3},\"params_preload_ms\":{:.3},\
\"params_ms\":{:.3},\"vk_ms\":{:.3},\"proof_ms\":{:.3},\"public_ms\":{:.3},\
\"init_ms\":{:.3},\"core_ms\":{:.3},\"cleanup_ms\":{:.3},\"accounted_ms\":{:.3},\
\"total_ms\":{:.3},\"residual_ms\":{:.3},\"peak_rss_bytes\":{},\"threads_at_end\":{},\
\"affinity_start\":\"{}\",\"affinity_end\":\"{}\",\"verify_result\":\"{}\"}}",
    read_mode,
    diag,
    trace_reads,
    k,
    num_cols,
    proof_len,
    public_vals.len(),
    ms(d_config),
    ms(d_circuit),
    ms(d_params_open),
    ms(d_params_parse),
    ms(d_preload),
    ms(d_params_open + d_preload + d_params_parse),
    ms(d_vk),
    ms(d_proof),
    ms(d_public),
    ms(d_init),
    ms(d_core),
    ms(d_cleanup),
    ms(accounted),
    ms(total),
    ms(total.saturating_sub(accounted)),
    hwm,
    threads,
    affinity_start,
    affinity_end,
    verdict
  );

  if result.is_err() {
    if diag {
      println!("verify_result_error: {}", verdict);
      std::process::exit(2);
    }
    panic!("proof did not verify");
  }
  println!("Proof verified!");
}
