//! Host for the modified-SP1 (TC_DOT chip) bare guest: backend `sp1 tc-dot`, label "modified SP1 (TC_DOT chip)".
//!
//! Same statement, identity string, public values and JSON events as the stock host's `bare-*` subcommands
//! (`../../host/src/bare_cmd.rs`); what differs is the guest input.  The host lays each VU out as 96 in-place
//! operand blocks for the `TC_DOT_BF16` precompile (with `stream-operands`, for the witness-operands fork: as 96
//! input buffers the precompile takes from the input stream) and computes the routing hint by running the fork's own
//! contract check (`try_tc_dot_bf16_witness`) along the reference chain (`tc::tc_dot`); steps outside the chip's
//! contract run in software in the guest.  Every chip step's kernel result is compared with `tc::tc_dot`, and the
//! public values the guest must commit come from the stock layout through `bare::decide`, independently.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sp1_core_executor::tc_dot::{try_tc_dot_bf16_witness, TcBf16Violation};
use sp1_sdk::prelude::*;
use sp1_sdk::{ProverClient, SP1Proof};
use veritor_zk_common::bare::{self, Format};
use veritor_zk_common::{sha256, tc};

const ELF: Elf = include_elf!("verity-tcdot-guest");
const BACKEND: &str = "sp1 tc-dot";
const LABEL: &str = "modified SP1 (TC_DOT chip)";
const SP1_VERSION: &str = "6.4.0";
const STATEMENT: &str = "relation-bare/v2";
/// The fork's HEAD, set by the pod build script; empty when built outside it.
const FORK_HEAD: &str = match option_env!("VERITY_TCDOT_FORK_HEAD") {
    Some(head) => head,
    None => "",
};
/// What `vector_run.py --backend sp1-bare` reads from `info` / `setup` to accept a host.
const GUEST_FEATURES: [&str; 1] = ["relation-bare"];
/// The one format with a chip: the A100 row's `tc-ampere-bf16`.
const FORMAT: Format = Format::Bf16Ampere;
const K: usize = bare::K;
const STEPS: usize = K / 16;
/// One VU's routing mask: two u64le words, so the guest reads it with two aligned loads (bits 96.. stay clear unless
/// the whole VU runs in software).
const ROUTE_BYTES: usize = 16;
/// Bytes per step block: 16 x words, then 16 W words.
const BLOCK_BYTES: usize = 64;
/// Where the fork's `TC_DOT_BF16` takes its operands: from the input stream (`stream-operands`, the witness-operands
/// fork, FORK_HEAD_WIT) or from guest memory.
const OPERANDS: &str = if cfg!(feature = "stream-operands") { "stream" } else { "memory" };

#[derive(Parser)]
#[command(name = "verity-tcdot-host")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// A contiguous id range `[lo, hi)` of a frozen VU tier (as `bare_cmd::BareSet`).
#[derive(clap::Args, Clone, Debug)]
struct BareSet {
    /// Directory holding `<tier>.x.u16`, `<tier>.w.u16`, `<tier>.y.u16`.
    #[arg(long)]
    instances: PathBuf,
    #[arg(long, default_value = "vu-k1536")]
    tier: String,
    #[arg(long, default_value = "bench-instances/v1")]
    dataset: String,
    /// sha256 of the dataset's manifest.json (part of the identity string).
    #[arg(long)]
    manifest_sha256: String,
    #[arg(long, default_value_t = 0)]
    lo: usize,
    #[arg(long)]
    hi: usize,
    /// VUs per `read_vec` chunk.
    #[arg(long, default_value_t = 64)]
    vus_per_read: usize,
    /// A VU with at least this many steps outside the chip's contract runs whole through `bare::check_one`
    /// (97 = never, unless no step is inside).
    #[arg(long, env = "VERITY_TCDOT_VU_SOFTWARE_THRESHOLD", default_value_t = 8)]
    vu_software_threshold: u32,
}

#[derive(Subcommand)]
enum Command {
    /// ELF digest, verifying-key hash and the fork pin.
    Info,
    /// Route every step of the range natively; compare each chip step with `tc::tc_dot`.
    Crosscheck {
        #[command(flatten)]
        set: BareSet,
    },
    /// Execute the guest (no proof); the public values must equal `bare::decide`.
    BareExecute {
        #[command(flatten)]
        set: BareSet,
        /// Claim a wrong word (low bit flipped) for VU `lo + I`; the verdict must be 0.
        #[arg(long)]
        flip_y: Option<usize>,
    },
    /// Write the statement bytes a verifier hashes.
    BareStatement {
        #[command(flatten)]
        set: BareSet,
        #[arg(long)]
        out: PathBuf,
    },
    /// Set up once, prove a warm-up range, prove the range `--reps` times (core), save and verify each proof.
    BareProve {
        #[command(flatten)]
        set: BareSet,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 1)]
        reps: usize,
        #[arg(long, default_value_t = 0)]
        warmup_vus: usize,
        /// Only `core`: the fork's recursion vk map does not know the TC_DOT chips.
        #[arg(long, default_value = "core")]
        mode: String,
        /// Skip the executor pass before proving (the routing and `bare::decide` checks still run).
        #[arg(long)]
        skip_execute: bool,
    },
    /// Verify a saved proof with the fork's verifier under the guest's vk; check the statement digest and verdict.
    Verify {
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        statement: PathBuf,
    },
    /// Every committed negative of `vu-k1536-neg` through the executor.
    BareNegatives {
        #[arg(long)]
        instances: PathBuf,
        #[arg(long)]
        manifest_sha256: String,
        #[arg(long, default_value = "bench-instances/v1")]
        dataset: String,
        /// As `bare-prove`'s (0 = every VU through `bare::check_one`, 97 = per-step routing wherever a step is inside).
        #[arg(long, env = "VERITY_TCDOT_VU_SOFTWARE_THRESHOLD", default_value_t = 8)]
        vu_software_threshold: u32,
    },
}

fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 2 == 0, "{}: odd length", path.display());
    Ok(bytes)
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}

#[derive(Default, Serialize, Clone, Debug)]
struct RouteStats {
    /// `TC_DOT_BF16` calls the guest makes.
    chip_steps: u64,
    /// Single steps the guest runs through `tc::tc_dot`.
    software_steps: u64,
    /// VUs the guest runs whole through `bare::check_one` (stock layout).
    software_vus: u64,
    /// Steps outside the chip's contract, by violation (whatever the VU's route).
    contract_violations: BTreeMap<String, u64>,
    /// VUs by their number of steps outside the chip's contract.
    vus_by_violations: BTreeMap<u32, u64>,
    /// VUs whose reference chain leaves the relation's domain (the guest rejects them).
    rejected_vus: u64,
    /// In-contract steps whose kernel result differs from `tc::tc_dot` (must be 0).
    kernel_mismatches: u64,
}

fn reason(violation: &TcBf16Violation) -> String {
    match violation {
        TcBf16Violation::NonFinite => "non_finite".into(),
        TcBf16Violation::Subnormal(stage) => format!("subnormal_g{}", stage + 1),
        TcBf16Violation::Overflow(stage) => format!("overflow_g{}", stage + 1),
    }
}

/// One VU: route each step along the reference chain (bit t set = step t outside the chip's contract, or after the
/// chain rejected), then append the VU to `out` -- as 96 operand blocks `x[16t..16t+16] || W[16t..16t+16]` (u16le)
/// with that mask, or, when at least `threshold` steps are outside the contract or none is inside, in the stock
/// layout with the all-ones mask.  Returns the mask and the reference output word (`None` where the chain rejects).
fn route_vu(x: &[u16], w: &[u16], threshold: u32, out: &mut Vec<u8>, stats: &mut RouteStats) -> ([u8; ROUTE_BYTES], Option<u16>) {
    let mut mask = [0u8; ROUTE_BYTES];
    let mut acc: Option<u32> = Some(0);
    let mut violations = 0u32;
    for t in 0..STEPS {
        let a: [u16; 16] = x[16 * t..16 * t + 16].try_into().expect("16 words");
        let b: [u16; 16] = w[16 * t..16 * t + 16].try_into().expect("16 words");
        let Some(c) = acc else {
            mask[t / 8] |= 1 << (t % 8);
            continue;
        };
        let reference = tc::tc_dot(&tc::AMPERE_BF16_M16N8K16, u64::from(c), &a.map(u64::from), &b.map(u64::from));
        match try_tc_dot_bf16_witness(&a, &b, c) {
            Ok(witness) => {
                if reference != Some(u64::from(witness.d_bits)) {
                    stats.kernel_mismatches += 1;
                }
                acc = Some(witness.d_bits);
            }
            Err(violation) => {
                violations += 1;
                *stats.contract_violations.entry(reason(&violation)).or_default() += 1;
                mask[t / 8] |= 1 << (t % 8);
                acc = reference.map(|bits| bits as u32);
            }
        }
    }
    if acc.is_none() {
        stats.rejected_vus += 1;
    }
    *stats.vus_by_violations.entry(violations).or_default() += 1;
    let software_bits: u32 = mask.iter().map(|byte| byte.count_ones()).sum();
    if violations >= threshold || software_bits == STEPS as u32 {
        mask = [0xFF; ROUTE_BYTES];
        stats.software_vus += 1;
        out.extend(x.iter().chain(w.iter()).flat_map(|v| v.to_le_bytes()));
    } else {
        stats.chip_steps += u64::from(STEPS as u32 - software_bits);
        stats.software_steps += u64::from(software_bits);
        for t in 0..STEPS {
            for word in x[16 * t..16 * t + 16].iter().chain(&w[16 * t..16 * t + 16]) {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    (mask, acc.map(|bits| tc::f32_to_bf16(u64::from(bits)) as u16))
}

/// A range's guest input with what the guest must commit.
struct Prepared {
    stdin: SP1Stdin,
    statement: Vec<u8>,
    identity: String,
    /// `bare::decide` on the stock layout of the same range and claim.
    expected: [u8; bare::PUBLIC_VALUES_LEN],
    stats: RouteStats,
    reference_outputs: Vec<Option<u16>>,
    input_bytes: usize,
    chunks: usize,
}

impl BareSet {
    fn path(&self, what: &str) -> PathBuf {
        self.instances.join(format!("{}.{what}.u16", self.tier))
    }

    fn identity(&self) -> String {
        bare::identity_string(&self.dataset, &self.tier, self.lo as u64, self.hi as u64, &self.manifest_sha256)
    }

    fn claimed(&self) -> Result<Vec<u32>> {
        let y = u16s(&read_bytes(&self.path("y"))?);
        ensure!(self.lo < self.hi && self.hi <= y.len(), "range [{}, {}) is not inside the {} instances of {}", self.lo, self.hi, y.len(), self.tier);
        Ok(y[self.lo..self.hi].iter().map(|&word| u32::from(word)).collect())
    }

    fn statement(&self, claimed: &[u32]) -> Result<Vec<u8>> {
        let y_le = bare::encode_y(FORMAT, claimed).ok_or_else(|| anyhow!("a claimed word does not fit {}", FORMAT.name()))?;
        Ok(bare::statement_bytes(FORMAT, &bare::identity_digest(&self.identity()), K as u32, &y_le))
    }

    fn prepare(&self, claimed: Option<Vec<u32>>) -> Result<Prepared> {
        let claimed = match claimed {
            Some(words) => words,
            None => self.claimed()?,
        };
        ensure!(claimed.len() == self.hi - self.lo && self.vus_per_read > 0, "bad claim / vus_per_read");
        let (xb, wb) = (read_bytes(&self.path("x"))?, read_bytes(&self.path("w"))?);
        let rb = FORMAT.row_bytes();
        ensure!(xb.len() >= self.hi * rb && wb.len() >= self.hi * rb, "{}: fewer than {} rows", self.tier, self.hi);
        let (x, w) = (u16s(&xb), u16s(&wb));
        let identity = self.identity();
        let id = bare::identity_digest(&identity);
        let y_le = bare::encode_y(FORMAT, &claimed).ok_or_else(|| anyhow!("a claimed word does not fit {}", FORMAT.name()))?;
        let header = bare::encode_header(FORMAT, K as u32, self.vus_per_read as u32, &id, &y_le);
        let mut stats = RouteStats::default();
        let mut routes = Vec::with_capacity((self.hi - self.lo) * ROUTE_BYTES);
        let mut reference_outputs = Vec::with_capacity(self.hi - self.lo);
        let mut chunks = Vec::new();
        let mut stock = Vec::new();
        for lo in (self.lo..self.hi).step_by(self.vus_per_read) {
            let hi = (lo + self.vus_per_read).min(self.hi);
            let mut blocks = Vec::with_capacity((hi - lo) * 2 * rb);
            for i in lo..hi {
                let (xi, wi) = (&x[K * i..K * (i + 1)], &w[K * i..K * (i + 1)]);
                let (mask, out) = route_vu(xi, wi, self.vu_software_threshold, &mut blocks, &mut stats);
                routes.extend_from_slice(&mask);
                reference_outputs.push(out);
                // Streamed operands: the VU as its own buffers, whole in the stock layout or one per step block.
                if cfg!(feature = "stream-operands") {
                    if mask == [0xFF; ROUTE_BYTES] {
                        chunks.push(std::mem::take(&mut blocks));
                    } else {
                        chunks.extend(blocks.chunks(BLOCK_BYTES).map(<[u8]>::to_vec));
                        blocks.clear();
                    }
                }
            }
            if !cfg!(feature = "stream-operands") {
                chunks.push(blocks);
            }
            stock.push(bare::encode_chunk(FORMAT, &xb, &wb, lo, hi));
        }
        let expected = bare::decide(&header, &stock);
        let n_chunks = chunks.len();
        let input_bytes = header.len() + routes.len() + chunks.iter().map(Vec::len).sum::<usize>();
        let mut stdin = SP1Stdin::new();
        stdin.write_vec(header);
        stdin.write_vec(routes);
        for chunk in chunks {
            stdin.write_vec(chunk);
        }
        Ok(Prepared {
            stdin,
            statement: bare::statement_bytes(FORMAT, &id, K as u32, &y_le),
            identity,
            expected,
            stats,
            reference_outputs,
            input_bytes,
            chunks: n_chunks,
        })
    }
}

#[derive(Serialize)]
struct Outcome {
    public_values: String,
    statement_digest: String,
    verdict: bool,
    unsound: bool,
}

fn outcome(bytes: &[u8]) -> Result<Outcome> {
    ensure!(bytes.len() == bare::PUBLIC_VALUES_LEN, "guest committed {} public bytes, expected 33", bytes.len());
    ensure!(bytes[32] <= 1, "unknown verdict byte {}", bytes[32]);
    Ok(Outcome {
        public_values: hex::encode(bytes),
        statement_digest: hex::encode(&bytes[..32]),
        verdict: bytes[32] == bare::VERDICT_ACCEPT,
        unsound: false,
    })
}

#[derive(Serialize)]
struct ExecuteOutput {
    backend: &'static str,
    label: &'static str,
    format: &'static str,
    #[serde(flatten)]
    outcome: Outcome,
    total_cycles: u64,
    cycle_tracker: BTreeMap<String, u64>,
    gas: Option<u64>,
    syscalls: BTreeMap<String, u64>,
    execute_seconds: f64,
    statement_bytes: usize,
    witness_bytes: usize,
    plan_bytes: usize,
    identity: String,
    lo: usize,
    hi: usize,
    vus_per_read: usize,
    chunks: usize,
    input_bytes: usize,
    expected_public_values: String,
    matches_expected: bool,
    statement_sha256: String,
    routing: RouteStats,
    /// `--flip-y I`: the VU (relative to `lo`) whose claimed word is wrong.
    flipped: Option<usize>,
}

async fn execute_set<P: Prover>(client: &P, set: &BareSet, claimed: Option<Vec<u32>>) -> Result<(ExecuteOutput, Prepared)> {
    let prepared = set.prepare(claimed)?;
    let started = Instant::now();
    let (public_values, report) = client
        .execute(ELF, prepared.stdin.clone())
        .await
        .map_err(|error| anyhow!("execution failed: {error}"))?;
    let execute_seconds = started.elapsed().as_secs_f64();
    ensure!(report.exit_code == 0, "guest exited with status {}", report.exit_code);
    let output = ExecuteOutput {
        backend: BACKEND,
        label: LABEL,
        format: FORMAT.name(),
        outcome: outcome(public_values.as_slice())?,
        total_cycles: report.total_instruction_count(),
        cycle_tracker: report.cycle_tracker.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        gas: report.gas(),
        syscalls: report
            .syscall_counts
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(code, count)| (format!("{code:?}"), *count))
            .collect(),
        execute_seconds,
        statement_bytes: prepared.statement.len(),
        witness_bytes: prepared.input_bytes,
        plan_bytes: 0,
        identity: prepared.identity.clone(),
        lo: set.lo,
        hi: set.hi,
        vus_per_read: set.vus_per_read,
        chunks: prepared.chunks,
        input_bytes: prepared.input_bytes,
        expected_public_values: hex::encode(prepared.expected),
        matches_expected: public_values.as_slice() == prepared.expected.as_slice(),
        statement_sha256: hex::encode(sha256(&prepared.statement)),
        routing: prepared.stats.clone(),
        flipped: None,
    };
    Ok((output, prepared))
}

fn save_proof(proof: &SP1ProofWithPublicValues, out: &Path) -> Result<()> {
    let file = fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    bincode::serialize_into(&mut writer, proof).context("serialising proof")?;
    writer.flush()?;
    Ok(())
}

fn load_proof(path: &Path) -> Result<SP1ProofWithPublicValues> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    bincode::deserialize_from(BufReader::with_capacity(8 << 20, file)).with_context(|| format!("loading {}", path.display()))
}

fn emit<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    std::io::stdout().flush()?;
    Ok(())
}

fn shards(proof: &SP1ProofWithPublicValues) -> usize {
    match &proof.proof {
        SP1Proof::Core(shards) => shards.len(),
        _ => 1,
    }
}

async fn cmd_prove(set: BareSet, out_dir: PathBuf, reps: usize, warmup_vus: usize, skip_execute: bool) -> Result<()> {
    ensure!(reps >= 1, "--reps must be at least 1");
    let started = Instant::now();
    let client = ProverClient::from_env().await;
    let client_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let prepared = if skip_execute {
        set.prepare(None)?
    } else {
        let (output, prepared) = execute_set(&client, &set, None).await?;
        ensure!(output.matches_expected, "guest committed {}, bare::decide {}", output.outcome.public_values, output.expected_public_values);
        emit(&serde_json::json!({"event": "execute", "report": output}))?;
        prepared
    };
    let prepare_seconds = started.elapsed().as_secs_f64();
    ensure!(prepared.stats.kernel_mismatches == 0, "fork kernel disagrees with tc::tc_dot on {} steps", prepared.stats.kernel_mismatches);
    ensure!(prepared.expected[32] == bare::VERDICT_ACCEPT, "the range does not satisfy the statement: nothing to prove");
    let started = Instant::now();
    let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
    let setup_seconds = started.elapsed().as_secs_f64();
    emit(&serde_json::json!({"event": "setup", "backend": BACKEND, "label": LABEL, "sp1_version": SP1_VERSION,
        "fork_head": FORK_HEAD, "operands": OPERANDS, "elf_sha256": hex::encode(sha256(&ELF)), "vk_hash": pk.verifying_key().bytes32(),
        "guest_features": GUEST_FEATURES, "statement": STATEMENT, "format": FORMAT.name(),
        "client_seconds": client_seconds, "prepare_seconds": prepare_seconds, "setup_seconds": setup_seconds,
        "identity": prepared.identity, "statement_sha256": hex::encode(sha256(&prepared.statement)),
        "input_bytes": prepared.input_bytes, "chunks": prepared.chunks,
        "vu_software_threshold": set.vu_software_threshold, "routing": prepared.stats}))?;
    fs::create_dir_all(&out_dir)?;
    fs::write(out_dir.join("statement.bin"), &prepared.statement)?;

    if warmup_vus > 0 {
        let warm = BareSet { hi: set.lo + warmup_vus.min(set.hi - set.lo), ..set.clone() };
        let warm_input = warm.prepare(None)?;
        let started = Instant::now();
        let proof = client.prove(&pk, warm_input.stdin).core().await.map_err(|error| anyhow!("warm-up prove failed: {error}"))?;
        let seconds = started.elapsed().as_secs_f64();
        let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
        emit(&serde_json::json!({"event": "warmup", "vus": warm.hi - warm.lo, "prove_seconds": seconds, "accepted": ok,
            "public_values": hex::encode(proof.public_values.as_slice())}))?;
    }

    for rep in 0..reps {
        let started = Instant::now();
        let proof = client.prove(&pk, prepared.stdin.clone()).core().await.map_err(|error| anyhow!("prove failed: {error}"))?;
        let prove_seconds = started.elapsed().as_secs_f64();
        let path = out_dir.join(format!("proof-rep{rep}.bin"));
        let started = Instant::now();
        save_proof(&proof, &path)?;
        let save_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
        let verify_seconds = started.elapsed().as_secs_f64();
        let public_values = proof.public_values.as_slice().to_vec();
        let accepted = ok && public_values == prepared.expected;
        let n_shards = shards(&proof);
        drop(proof);
        let bytes = fs::read(&path)?;
        emit(&serde_json::json!({"event": "rep", "rep": rep, "mode": "core", "prove_seconds": prove_seconds,
            "save_seconds": save_seconds, "verify_seconds": verify_seconds, "proof_path": path.display().to_string(),
            "proof_bytes": bytes.len(), "proof_sha256": hex::encode(sha256(&bytes)), "shards": n_shards,
            "public_values": hex::encode(&public_values), "accepted": accepted}))?;
        ensure!(accepted, "rep {rep}: proof not accepted (verify ok = {ok})");
    }
    Ok(())
}

async fn cmd_verify(proof_path: PathBuf, statement_path: PathBuf) -> Result<()> {
    let statement = fs::read(&statement_path).with_context(|| format!("reading {}", statement_path.display()))?;
    let started = Instant::now();
    let proof = load_proof(&proof_path)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let client = ProverClient::builder().cpu().build().await;
    let started = Instant::now();
    let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let checked = client.verify(&proof, pk.verifying_key(), None);
    let verify_seconds = started.elapsed().as_secs_f64();
    let public_values = proof.public_values.as_slice();
    let statement_match = public_values.len() == bare::PUBLIC_VALUES_LEN && public_values[..32] == sha256(&statement);
    let verdict = public_values.get(32) == Some(&bare::VERDICT_ACCEPT);
    emit(&serde_json::json!({"backend": BACKEND, "label": LABEL, "ok": checked.is_ok(), "unsound": false,
        "error": checked.as_ref().err().map(|e| e.to_string()), "statement_match": statement_match, "verdict": verdict,
        "public_values": hex::encode(public_values), "vk_hash": pk.verifying_key().bytes32(),
        "elf_sha256": hex::encode(sha256(&ELF)), "fork_head": FORK_HEAD, "operands": OPERANDS, "shards": shards(&proof),
        "load_seconds": load_seconds, "setup_seconds": setup_seconds, "verify_seconds": verify_seconds}))?;
    ensure!(checked.is_ok() && statement_match && verdict, "proof rejected");
    Ok(())
}

async fn cmd_crosscheck(set: BareSet) -> Result<()> {
    let started = Instant::now();
    let prepared = set.prepare(None)?;
    let claimed = set.claimed()?;
    let differing = prepared.reference_outputs.iter().zip(&claimed).filter(|(out, y)| out.map(u32::from) != Some(**y)).count();
    emit(&serde_json::json!({"backend": BACKEND, "vus": set.hi - set.lo, "routing": prepared.stats,
        "reference_outputs_differing_from_y": differing, "decide_verdict": prepared.expected[32],
        "seconds": started.elapsed().as_secs_f64()}))?;
    ensure!(prepared.stats.kernel_mismatches == 0, "fork kernel disagrees with tc::tc_dot");
    Ok(())
}

async fn cmd_negatives(instances: PathBuf, manifest_sha256: String, dataset: String, vu_software_threshold: u32) -> Result<()> {
    let tier = "vu-k1536-neg";
    let index = instances.join(format!("{tier}.index.json"));
    let doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?)?;
    let rows = doc["rows"].as_array().ok_or_else(|| anyhow!("{}: no rows", index.display()))?;
    let client = ProverClient::builder().mock().build().await;
    let (mut rejected, mut correct_accepted, mut in_domain) = (0usize, 0usize, 0usize);
    for row in rows {
        let i = row[0].as_u64().ok_or_else(|| anyhow!("bad row {row}"))? as usize;
        let set = BareSet {
            instances: instances.clone(),
            tier: tier.to_string(),
            dataset: dataset.clone(),
            manifest_sha256: manifest_sha256.clone(),
            lo: i,
            hi: i + 1,
            vus_per_read: 1,
            vu_software_threshold,
        };
        let (claimed, _) = execute_set(&client, &set, None).await?;
        let claimed_verdict = claimed.outcome.verdict;
        if !claimed_verdict && claimed.matches_expected {
            rejected += 1;
        }
        let correct = match row[4].as_u64() {
            Some(word) => {
                in_domain += 1;
                let (out, _) = execute_set(&client, &set, Some(vec![word as u32])).await?;
                if out.outcome.verdict && out.matches_expected {
                    correct_accepted += 1;
                }
                Some(out.outcome.verdict)
            }
            None => None,
        };
        emit(&serde_json::json!({"event": "negative", "id": i, "mistake": row[1], "verdict_expected": row[5],
            "claimed_y": row[3], "correct_y": row[4], "claimed_verdict": claimed_verdict,
            "claimed_matches_native": claimed.matches_expected, "correct_verdict": correct,
            "cycles": claimed.total_cycles, "routing": claimed.routing}))?;
    }
    let ok = rejected == rows.len() && correct_accepted == in_domain;
    emit(&serde_json::json!({"event": "summary", "backend": BACKEND, "negatives": rows.len(), "claimed_rejected": rejected,
        "in_domain": in_domain, "correct_accepted": correct_accepted, "ok": ok}))?;
    if !ok {
        bail!("negatives: {rejected}/{} claimed words rejected, {correct_accepted}/{in_domain} correct words accepted", rows.len());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Info => {
            let client = ProverClient::builder().cpu().build().await;
            let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
            emit(&serde_json::json!({"backend": BACKEND, "label": LABEL, "sp1_version": SP1_VERSION,
                "fork_head": FORK_HEAD, "operands": OPERANDS, "elf_sha256": hex::encode(sha256(&ELF)),
                "vk_hash": pk.verifying_key().bytes32(),
                "public_values_len": bare::PUBLIC_VALUES_LEN, "guest_features": GUEST_FEATURES, "statement": STATEMENT,
                "formats": [FORMAT.name()]}))
        }
        Command::Crosscheck { set } => cmd_crosscheck(set).await,
        Command::BareExecute { set, flip_y } => {
            let client = ProverClient::builder().mock().build().await;
            let claimed = match flip_y {
                None => None,
                Some(j) => {
                    ensure!(j < set.hi - set.lo, "--flip-y {j} is outside the range's {} VUs", set.hi - set.lo);
                    let mut words = set.claimed()?;
                    words[j] ^= 1;
                    Some(words)
                }
            };
            let (mut output, _) = execute_set(&client, &set, claimed).await?;
            output.flipped = flip_y;
            ensure!(output.matches_expected, "guest committed {}, bare::decide {}", output.outcome.public_values, output.expected_public_values);
            ensure!(flip_y.is_none() || !output.outcome.verdict, "a wrong claimed word was accepted (verdict 1)");
            emit(&output)
        }
        Command::BareStatement { set, out } => {
            let statement = set.statement(&set.claimed()?)?;
            fs::write(&out, &statement).with_context(|| format!("writing {}", out.display()))?;
            emit(&serde_json::json!({"backend": BACKEND, "format": FORMAT.name(), "identity": set.identity(), "statement": out.display().to_string(),
                "statement_bytes": statement.len(), "statement_sha256": hex::encode(sha256(&statement)),
                "accepting_public_values": hex::encode(bare::public_values(&sha256(&statement), true))}))
        }
        Command::BareProve { set, out_dir, reps, warmup_vus, mode, skip_execute } => {
            ensure!(mode == "core", "--mode {mode}: only core proofs (the fork's recursion vk map lacks the TC_DOT chips)");
            cmd_prove(set, out_dir, reps, warmup_vus, skip_execute).await
        }
        Command::Verify { proof, statement } => cmd_verify(proof, statement).await,
        Command::BareNegatives { instances, manifest_sha256, dataset, vu_software_threshold } => {
            cmd_negatives(instances, manifest_sha256, dataset, vu_software_threshold).await
        }
    }
}
