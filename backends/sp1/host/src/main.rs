//! Host driver for the veritor SP1 checker guest.  Every subcommand prints one
//! JSON object to stdout (SP1's own logging and the guest's prints go to
//! stderr), which is the subprocess protocol `veritor.protocol.proofs.sp1`
//! speaks.
//!
//!   veritor-zk-host info
//!       -> {"backend", "sp1_version", "elf_sha256", "vk_hash", "guest_features"}
//!   veritor-zk-host execute --statement S --witness W [--plan P] [--typed-object-digest H]
//!       -> {"public_values", "statement_digest", ["typed_object_digest",] "verdict", "unsound",
//!           "total_cycles", "cycle_tracker": {...}, "gas", "syscalls": {...}}
//!   veritor-zk-host prove --statement S --witness W --out P [--mode core|compressed|groth16|plonk]
//!                         [--plan P] [--typed-object-digest H]
//!       -> execute's fields plus {"proof_bytes", "shards", "setup_seconds",
//!           "prove_seconds", "verify_seconds", "vk_hash", "mode"}
//!   veritor-zk-host verify --proof P [--statement S]
//!       -> {"ok", "public_values", "statement_digest", "verdict", "unsound", "vk_hash",
//!           "statement_match"}
//!   veritor-zk-host serve
//!       -> one long-lived prover: the GPU server and the proving key are set up ONCE
//!          ({"event":"ready", "setup_seconds", ...} on stdout), then every line of stdin is a
//!          job {"id", "statement", "witness", ["plan",] ["typed_object_digest",] "mode",
//!          ["out",] ["execute": bool,] ["verify": bool]} answered by one
//!          {"event":"job", "ok", "prove_seconds", "verify_seconds", "proof_bytes", "shards",
//!           "job_wall_seconds", "since_ready_seconds", ["execute": {...},] ...} line;
//!          EOF ends the process.  This is the steady-state measurement of the proving-overhead
//!          benchmark (`zk/bench/pb_prover_daemon.py`): `prove_seconds` is the recurring cost,
//!          `setup_seconds` the cold start, reported apart.
//!
//! `execute` never proves: it is the exact cycle meter used by the Python
//! tests and by the cost table.  The witness may be in any format of
//! `veritor_zk_common::codec`; the host passes its bytes through unread.
//! `--typed-object-digest` (32 bytes hex) is carried through the guest's
//! public values unchecked, giving `(h_O, h_L, verdict)`; without it the
//! public values are the 33-byte `(h_L, verdict)` as before.
//!
//! `unsound` is `true` when the guest committed one of the DIAGNOSTIC verdict
//! bytes (`0xEE` / `0xEF`): the embedded guest was built with an `unsound-*`
//! feature (listed under `guest_features` by `info`) and its verdict measures
//! cycles, not correctness.  Nothing downstream may treat such a run as a result.
//!
//! RELATION-ONLY build (`--features unsound-relation-only`): the embedded guest runs only the
//! `GemmCoordinate<K>` gate chain on a **compact input** `K || x || W`
//! (`veritor_zk_common::relation_only`).  The `--witness` file IS that compact input; `--statement`,
//! `--plan` and `--typed-object-digest` are accepted for CLI compatibility and ignored, and the
//! guest gets one stdin buffer.  Its public values are `sha256(input) || sha256(output word) || 0xEE`,
//! so in this build the reported `typed_object_digest` is the input digest and `statement_digest`
//! the output-word digest; `verify --statement` is meaningless (`statement_match` compares the
//! output-word digest).  The CLI surface is otherwise unchanged.
//!
//! RELATION-BARE build (`--features relation-bare`): the embedded guest proves the Table 2 bare
//! statement of `veritor_zk_common::bare` (public: the format, the instance-set identity and the
//! output words `y`; private: every VU's rows).  `bare-execute`, `bare-prove`, `bare-statement` and
//! `bare-negatives` (`bare_cmd`) take a range of a frozen set (a BF16 fixture tier or a
//! `verity_sp1.format_instances` file); `verify --proof P --statement S`, with `S` from
//! `bare-statement`, accepts exactly when `ok`, `statement_match` and `verdict`.

use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sp1_sdk::prelude::*;
use sp1_sdk::{ProverClient, SP1Proof};
use veritor_zk_common::{decode_verdict, sha256, split_public_values, PUBLIC_VALUES_LEN};

#[cfg(feature = "relation-bare")]
mod bare_cmd;

const ELF: Elf = include_elf!("veritor-zk-guest");
const BACKEND: &str = "sp1";
/// The pinned `sp1-sdk` (and guest `sp1-zkvm`) version; keep in step with Cargo.toml.
const SP1_VERSION: &str = "6.4.0";
/// The features build.rs forwarded to the embedded guest (comma separated; empty for a sound guest).
const GUEST_FEATURES: &str = env!("VERITOR_GUEST_FEATURES");

#[derive(Parser)]
#[command(name = "veritor-zk-host")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// STARK shard proofs; size grows with cycles.  Not zero-knowledge.
    Core,
    /// Recursively compressed STARK; constant size.  Not zero-knowledge.
    Compressed,
    /// Compressed proof wrapped in Groth16 over BN254: the ZK stage.
    Groth16,
    /// Compressed proof wrapped in PLONK over BN254: the ZK stage (universal setup).
    Plonk,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Core => "core",
            Mode::Compressed => "compressed",
            Mode::Groth16 => "groth16",
            Mode::Plonk => "plonk",
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Identify the guest: ELF digest and verifying-key hash.
    Info,
    /// Run the guest in the executor (no proof) and report exact cycles.
    Execute {
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        /// Untrusted plan bytes (the statement's tree table); absent = reference pass.
        #[arg(long)]
        plan: Option<PathBuf>,
        /// 32-byte hex typed-obligation/v0 object digest to carry through the public
        /// values unchecked (`h_O || h_L || verdict`); absent = the 33-byte layout.
        #[arg(long)]
        typed_object_digest: Option<String>,
    },
    /// Produce one proof of one batch and verify it locally.
    Prove {
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        /// Untrusted plan bytes (the statement's tree table); absent = reference pass.
        #[arg(long)]
        plan: Option<PathBuf>,
        /// 32-byte hex typed-obligation/v0 object digest to carry through the public
        /// values unchecked (`h_O || h_L || verdict`); absent = the 33-byte layout.
        #[arg(long)]
        typed_object_digest: Option<String>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = Mode::Core)]
        mode: Mode,
    },
    /// Verify a saved proof against the guest's verifying key.
    Verify {
        #[arg(long)]
        proof: PathBuf,
        /// If given, also check the public statement digest against these bytes.
        #[arg(long)]
        statement: Option<PathBuf>,
    },
    /// Long-lived prover: set up once, then prove one batch per JSON job line on stdin.
    Serve,
    /// Long-lived verifier: derive the verifying key ONCE, then verify one saved proof per JSON
    /// job line {"id", "proof", ["statement"]} on stdin; `{"event":"ready", "setup_seconds", ...}`
    /// first, one `{"event":"verify", ...}` line per job, EOF ends the process.
    VerifyMany,
    /// relation-bare: the guest in the executor on a range of a frozen set (exact cycles, public values).
    #[cfg(feature = "relation-bare")]
    BareExecute {
        #[command(flatten)]
        set: bare_cmd::BareSet,
        /// Claim a wrong word (bit 0 flipped) for VU `lo + I` and require verdict 0: a negative for any format.
        #[arg(long)]
        flip_y: Option<usize>,
    },
    /// relation-bare: set up once, prove a warm-up range, then prove the range `--reps` times (save and verify each).
    #[cfg(feature = "relation-bare")]
    BareProve {
        #[command(flatten)]
        set: bare_cmd::BareSet,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = Mode::Core)]
        mode: Mode,
        #[arg(long, default_value_t = 1)]
        reps: usize,
        /// VUs of the warm-up proof (0: none).
        #[arg(long, default_value_t = 0)]
        warmup_vus: usize,
        /// Skip the executor pass (cycles) before proving.
        #[arg(long)]
        skip_execute: bool,
    },
    /// relation-bare: write the statement bytes of a range (what `verify --statement` hashes).
    #[cfg(feature = "relation-bare")]
    BareStatement {
        #[command(flatten)]
        set: bare_cmd::BareSet,
        #[arg(long)]
        out: PathBuf,
    },
    /// relation-bare: every committed negative of `vu-k1536-neg` through the executor.
    #[cfg(feature = "relation-bare")]
    BareNegatives {
        /// Directory holding `vu-k1536-neg.{x,w,y}.u16` and `vu-k1536-neg.index.json`.
        #[arg(long)]
        instances: PathBuf,
        #[arg(long)]
        manifest_sha256: String,
        #[arg(long, default_value = "bench-instances/v1")]
        dataset: String,
        #[arg(long)]
        index: Option<PathBuf>,
        /// The guest input layout each negative is run in.
        #[arg(long, value_enum, default_value_t = bare_cmd::Layout::Chunks)]
        layout: bare_cmd::Layout,
    },
}

/// One `verify-many` job (a JSON line on stdin).
#[derive(Deserialize)]
struct VerifyJob {
    #[serde(default)]
    id: Option<String>,
    proof: PathBuf,
    #[serde(default)]
    statement: Option<PathBuf>, // if given, also check the public statement digest against these bytes
}

#[derive(Serialize)]
struct VerifyManyResult {
    event: &'static str,
    id: Option<String>,
    job_index: usize,
    #[serde(flatten)]
    verify: VerifyOutput,
    /// `load_proof` (read + deserialise) and the statement hashing, each apart from the check itself.
    load_seconds: f64,
    statement_hash_seconds: f64,
    job_wall_seconds: f64,
}

/// One `verify-many` job after setup: load, verify against the derived vk, optional statement match.
fn verify_job<P: Prover>(client: &P, pk: &P::ProvingKey, job: &VerifyJob, job_index: usize) -> Result<VerifyManyResult> {
    let (job_started, mut started) = (Instant::now(), Instant::now());
    let proof = load_proof(&job.proof)?;
    let load_seconds = started.elapsed().as_secs_f64();
    started = Instant::now();
    let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
    let verify_seconds = started.elapsed().as_secs_f64();
    let outcome = PublicOutcome::from_bytes(proof.public_values.as_slice())?;
    started = Instant::now();
    let statement_match = match &job.statement {
        None => None,
        Some(path) => {
            let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            Some(outcome.statement_digest == hex::encode(sha256(&bytes)))
        }
    };
    Ok(VerifyManyResult {
        event: "verify",
        id: job.id.clone(),
        job_index,
        verify: VerifyOutput { backend: BACKEND, ok, outcome, vk_hash: pk.verifying_key().bytes32(), statement_match, verify_seconds },
        load_seconds,
        statement_hash_seconds: started.elapsed().as_secs_f64(),
        job_wall_seconds: job_started.elapsed().as_secs_f64(),
    })
}

/// One `serve` job (a JSON line on stdin).
#[derive(Deserialize)]
struct ServeJob {
    #[serde(default)]
    id: Option<String>,
    statement: PathBuf,
    witness: PathBuf,
    /// Untrusted plan bytes; absent = reference pass.
    #[serde(default)]
    plan: Option<PathBuf>,
    #[serde(default)]
    typed_object_digest: Option<String>,
    #[serde(default = "default_mode")]
    mode: Mode,
    /// Where to save the proof; absent = the proof is sized in memory and dropped.
    #[serde(default)]
    out: Option<PathBuf>,
    /// Also run the executor first (exact cycles, syscall counts; CPU work, timed apart).
    #[serde(default)]
    execute: bool,
    /// Verify the proof locally after proving (default true).
    #[serde(default = "default_true")]
    verify: bool,
}

fn default_mode() -> Mode {
    Mode::Core
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct ServeReady {
    event: &'static str,
    backend: &'static str,
    sp1_version: &'static str,
    elf_sha256: String,
    vk_hash: String,
    guest_features: Vec<&'static str>,
    /// GPU prover server start + proving-key generation: the cold start, paid once per process.
    setup_seconds: f64,
    pid: u32,
}

#[derive(Serialize)]
struct ServeResult {
    event: &'static str,
    id: Option<String>,
    job_index: usize,
    ok: bool,
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    execute: Option<ExecuteOutput>,
    #[serde(flatten)]
    outcome: PublicOutcome,
    proof_path: Option<String>,
    proof_bytes: u64,
    shards: usize,
    vk_hash: String,
    /// The prover timer for this job (the GPU-seconds charged to it); no setup inside.
    prove_seconds: f64,
    verify_seconds: Option<f64>,
    /// Whole job: read inputs (+ execute) + prove (+ verify) (+ save).
    job_wall_seconds: f64,
    since_ready_seconds: f64,
    statement_bytes: usize,
    witness_bytes: usize,
    plan_bytes: usize,
}

#[derive(Serialize)]
struct ServeFailure {
    event: &'static str,
    id: Option<String>,
    job_index: usize,
    ok: bool,
    error: String,
    job_wall_seconds: f64,
    since_ready_seconds: f64,
}

/// One `serve` job after setup: (execute,) prove, (verify,) (save).
async fn serve_job<P: Prover>(
    client: &P,
    pk: &P::ProvingKey,
    job: &ServeJob,
    job_index: usize,
    ready: Instant,
) -> Result<ServeResult> {
    let job_started = Instant::now();
    let inputs = stdin_for(
        &job.statement,
        &job.witness,
        job.plan.as_ref(),
        job.typed_object_digest.as_deref(),
    )?;
    let stdin = inputs.stdin;
    let execute_output = if job.execute {
        let output = execute(client, stdin.clone(), inputs.sizes).await?;
        if !output.outcome.commits_input(&inputs.statement_bytes) {
            bail!("guest committed a digest of some other statement");
        }
        Some(output)
    } else {
        None
    };

    let started = Instant::now();
    let request = client.prove(pk, stdin);
    let proof = match job.mode {
        Mode::Core => request.core().await,
        Mode::Compressed => request.compressed().await,
        Mode::Groth16 => request.groth16().await,
        Mode::Plonk => request.plonk().await,
    }
    .map_err(|error| anyhow!("prove failed: {error}"))?;
    let prove_seconds = started.elapsed().as_secs_f64();

    let verify_seconds = if job.verify {
        let started = Instant::now();
        client
            .verify(&proof, pk.verifying_key(), None)
            .map_err(|error| anyhow!("verify failed: {error}"))?;
        Some(started.elapsed().as_secs_f64())
    } else {
        None
    };
    let outcome = PublicOutcome::from_bytes(proof.public_values.as_slice())?;
    if !outcome.commits_input(&inputs.statement_bytes) {
        bail!("proof commits a digest of some other statement");
    }
    let shards = match &proof.proof {
        SP1Proof::Core(shards) => shards.len(),
        _ => 1,
    };
    let proof_bytes = match &job.out {
        Some(out) => {
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            save_proof(&proof, out)?;
            fs::metadata(out)?.len()
        }
        None => bincode::serialized_size(&proof).context("sizing proof")?,
    };
    Ok(ServeResult {
        event: "job",
        id: job.id.clone(),
        job_index,
        ok: true,
        mode: job.mode.name(),
        execute: execute_output,
        outcome,
        proof_path: job.out.as_ref().map(|p| p.display().to_string()),
        proof_bytes,
        shards,
        vk_hash: pk.verifying_key().bytes32(),
        prove_seconds,
        verify_seconds,
        job_wall_seconds: job_started.elapsed().as_secs_f64(),
        since_ready_seconds: ready.elapsed().as_secs_f64(),
        statement_bytes: inputs.sizes.0,
        witness_bytes: inputs.sizes.1,
        plan_bytes: inputs.sizes.2,
    })
}

#[derive(Serialize)]
struct PublicOutcome {
    public_values: String,
    statement_digest: String,
    /// The typed-obligation/v0 object digest carried through the public values
    /// (65-byte layout), unchecked by the guest; absent in the 33-byte layout.
    #[serde(skip_serializing_if = "Option::is_none")]
    typed_object_digest: Option<String>,
    verdict: bool,
    /// The verdict byte was a DIAGNOSTIC one: the guest is an `unsound-*` build.
    unsound: bool,
}

impl PublicOutcome {
    /// The guest committed the digest of what the host sent it: `h_L = sha256(statement)` -- or, in
    /// the relation-only build, the first 32 bytes are `sha256(compact input)`.
    fn commits_input(&self, input: &[u8]) -> bool {
        let digest = hex::encode(sha256(input));
        if cfg!(feature = "unsound-relation-only") {
            self.typed_object_digest.as_deref() == Some(digest.as_str())
        } else {
            self.statement_digest == digest
        }
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (object_digest, statement_digest, verdict_byte) = split_public_values(bytes)
            .ok_or_else(|| {
                anyhow!(
                    "guest committed {} public bytes, expected {PUBLIC_VALUES_LEN} or {}",
                    bytes.len(),
                    veritor_zk_common::PUBLIC_VALUES_WITH_OBJECT_LEN
                )
            })?;
        let (verdict, unsound) = decode_verdict(verdict_byte)
            .ok_or_else(|| anyhow!("guest committed an unknown verdict byte {verdict_byte}"))?;
        Ok(PublicOutcome {
            public_values: hex::encode(bytes),
            statement_digest: hex::encode(statement_digest),
            typed_object_digest: object_digest.map(hex::encode),
            verdict,
            unsound,
        })
    }
}

#[derive(Serialize)]
struct ExecuteOutput {
    backend: &'static str,
    #[serde(flatten)]
    outcome: PublicOutcome,
    total_cycles: u64,
    cycle_tracker: BTreeMap<String, u64>,
    gas: Option<u64>,
    syscalls: BTreeMap<String, u64>,
    /// Executed RISC-V instructions by opcode (non-zero only): with SP1's per-chip trace widths, the proving cost.
    opcodes: BTreeMap<String, u64>,
    execute_seconds: f64,
    statement_bytes: usize,
    witness_bytes: usize,
    /// Bytes of the plan handed to the guest (0 when none: the reference pass).
    plan_bytes: usize,
}

#[derive(Serialize)]
struct ProveOutput {
    #[serde(flatten)]
    execute: ExecuteOutput,
    mode: &'static str,
    proof_path: String,
    proof_bytes: u64,
    shards: usize,
    vk_hash: String,
    setup_seconds: f64,
    prove_seconds: f64,
    verify_seconds: f64,
}

#[derive(Serialize)]
struct VerifyOutput {
    backend: &'static str,
    ok: bool,
    #[serde(flatten)]
    outcome: PublicOutcome,
    vk_hash: String,
    statement_match: Option<bool>,
    verify_seconds: f64,
}

#[derive(Serialize)]
struct InfoOutput {
    backend: &'static str,
    sp1_version: &'static str,
    elf_sha256: String,
    vk_hash: String,
    public_values_len: usize,
    /// Cargo features of the embedded guest; any `unsound-*` entry marks a DIAGNOSTIC build.
    guest_features: Vec<&'static str>,
}

/// The guest's three inputs: statement, witness and the (possibly empty) plan.  (Relation-only
/// build: the one compact input, held in `statement_bytes` so the digest checks read as usual.)
struct Inputs {
    stdin: SP1Stdin,
    statement_bytes: Vec<u8>,
    sizes: (usize, usize, usize),
}

/// RELATION-ONLY build: the `--witness` file is the compact input and the guest's only buffer;
/// the statement, plan and typed object digest are accepted and ignored (module docs).
#[cfg(feature = "unsound-relation-only")]
fn stdin_for(
    _statement: &PathBuf,
    witness: &PathBuf,
    _plan: Option<&PathBuf>,
    _typed_object_digest: Option<&str>,
) -> Result<Inputs> {
    let compact = fs::read(witness).with_context(|| format!("reading {}", witness.display()))?;
    let mut stdin = SP1Stdin::new();
    stdin.write_vec(compact.clone());
    Ok(Inputs {
        stdin,
        sizes: (0, compact.len(), 0),
        statement_bytes: compact,
    })
}

#[cfg(not(feature = "unsound-relation-only"))]
fn stdin_for(
    statement: &PathBuf,
    witness: &PathBuf,
    plan: Option<&PathBuf>,
    typed_object_digest: Option<&str>,
) -> Result<Inputs> {
    let statement_bytes =
        fs::read(statement).with_context(|| format!("reading {}", statement.display()))?;
    let witness_bytes =
        fs::read(witness).with_context(|| format!("reading {}", witness.display()))?;
    let plan_bytes = match plan {
        Some(path) => fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        None => Vec::new(),
    };
    let object_digest_bytes = match typed_object_digest {
        Some(text) => {
            let bytes = hex::decode(text).context("--typed-object-digest is not hex")?;
            if bytes.len() != 32 {
                bail!("--typed-object-digest must be 32 bytes, got {}", bytes.len());
            }
            bytes
        }
        None => Vec::new(),
    };
    let sizes = (statement_bytes.len(), witness_bytes.len(), plan_bytes.len());
    let mut stdin = SP1Stdin::new();
    stdin.write_vec(statement_bytes.clone());
    stdin.write_vec(witness_bytes);
    stdin.write_vec(plan_bytes);
    stdin.write_vec(object_digest_bytes);
    Ok(Inputs {
        stdin,
        statement_bytes,
        sizes,
    })
}

async fn execute<P: Prover>(
    client: &P,
    stdin: SP1Stdin,
    sizes: (usize, usize, usize),
) -> Result<ExecuteOutput> {
    let started = Instant::now();
    let (public_values, report) = client
        .execute(ELF, stdin)
        .await
        .map_err(|error| anyhow!("execution failed: {error}"))?;
    let execute_seconds = started.elapsed().as_secs_f64();
    if report.exit_code != 0 {
        bail!("guest exited with status {}", report.exit_code);
    }
    let outcome = PublicOutcome::from_bytes(public_values.as_slice())?;
    let cycle_tracker = report
        .cycle_tracker
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    let syscalls = report
        .syscall_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(code, count)| (format!("{code:?}"), *count))
        .collect();
    let opcodes = report
        .opcode_counts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(op, count)| (format!("{op:?}"), *count))
        .collect();
    Ok(ExecuteOutput {
        backend: BACKEND,
        outcome,
        total_cycles: report.total_instruction_count(),
        cycle_tracker,
        gas: report.gas(),
        syscalls,
        opcodes,
        execute_seconds,
        statement_bytes: sizes.0,
        witness_bytes: sizes.1,
        plan_bytes: sizes.2,
    })
}

/// Write the proof with a buffered writer.  The SDK's `SP1ProofWithPublicValues::save`
/// hands bincode an unbuffered `File`, which issues one `write(2)` per field element:
/// a 1 GB core proof took ~30 min of syscalls on the RunPod campaign (275 M writes for
/// 1.1 GB), inflating `prove_wall_s` far beyond the prover timer.  Same bytes as `save`.
fn save_proof(proof: &SP1ProofWithPublicValues, out: &Path) -> Result<()> {
    let file = fs::File::create(out).with_context(|| format!("creating {}", out.display()))?;
    let mut writer = BufWriter::with_capacity(8 << 20, file);
    bincode::serialize_into(&mut writer, proof).context("serialising proof")?;
    writer.flush()?;
    Ok(())
}

/// Read a proof written by `save_proof` (or the SDK's `save`) through a buffered reader.
fn load_proof(path: &Path) -> Result<SP1ProofWithPublicValues> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::with_capacity(8 << 20, file);
    bincode::deserialize_from(reader).with_context(|| format!("loading {}", path.display()))
}

fn emit<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info => {
            let client = ProverClient::from_env().await;
            let pk = client
                .setup(ELF)
                .await
                .map_err(|error| anyhow!("setup failed: {error}"))?;
            emit(&InfoOutput {
                backend: BACKEND,
                sp1_version: SP1_VERSION,
                elf_sha256: hex::encode(sha256(&ELF)),
                vk_hash: pk.verifying_key().bytes32(),
                public_values_len: PUBLIC_VALUES_LEN,
                guest_features: GUEST_FEATURES
                    .split(',')
                    .filter(|item| !item.is_empty())
                    .collect(),
            })
        }
        Command::Execute {
            statement,
            witness,
            plan,
            typed_object_digest,
        } => {
            // The mock prover is the light node: an executor without the proving
            // machinery, so `execute` starts in well under a second.
            let client = ProverClient::builder().mock().build().await;
            let inputs = stdin_for(&statement, &witness, plan.as_ref(), typed_object_digest.as_deref())?;
            let output = execute(&client, inputs.stdin, inputs.sizes).await?;
            if !output.outcome.commits_input(&inputs.statement_bytes) {
                bail!("guest committed a digest of some other statement");
            }
            emit(&output)
        }
        Command::Prove {
            statement,
            witness,
            plan,
            typed_object_digest,
            out,
            mode,
        } => {
            let client = ProverClient::from_env().await;
            let inputs = stdin_for(&statement, &witness, plan.as_ref(), typed_object_digest.as_deref())?;
            let stdin = inputs.stdin;
            let execute_output = execute(&client, stdin.clone(), inputs.sizes).await?;
            if !execute_output.outcome.commits_input(&inputs.statement_bytes) {
                bail!("guest committed a digest of some other statement");
            }

            let started = Instant::now();
            let pk = client
                .setup(ELF)
                .await
                .map_err(|error| anyhow!("setup failed: {error}"))?;
            let setup_seconds = started.elapsed().as_secs_f64();

            let started = Instant::now();
            let request = client.prove(&pk, stdin);
            let proof = match mode {
                Mode::Core => request.core().await,
                Mode::Compressed => request.compressed().await,
                Mode::Groth16 => request.groth16().await,
                Mode::Plonk => request.plonk().await,
            }
            .map_err(|error| anyhow!("prove failed: {error}"))?;
            let prove_seconds = started.elapsed().as_secs_f64();

            let started = Instant::now();
            client
                .verify(&proof, pk.verifying_key(), None)
                .map_err(|error| anyhow!("verify failed: {error}"))?;
            let verify_seconds = started.elapsed().as_secs_f64();

            let shards = match &proof.proof {
                SP1Proof::Core(shards) => shards.len(),
                _ => 1,
            };
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            save_proof(&proof, &out)?;
            let proof_bytes = fs::metadata(&out)?.len();
            emit(&ProveOutput {
                execute: execute_output,
                mode: mode.name(),
                proof_path: out.display().to_string(),
                proof_bytes,
                shards,
                vk_hash: pk.verifying_key().bytes32(),
                setup_seconds,
                prove_seconds,
                verify_seconds,
            })
        }
        Command::Verify { proof, statement } => {
            let client = ProverClient::from_env().await;
            let pk = client
                .setup(ELF)
                .await
                .map_err(|error| anyhow!("setup failed: {error}"))?;
            let proof = load_proof(&proof)?;
            let started = Instant::now();
            let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
            let verify_seconds = started.elapsed().as_secs_f64();
            let outcome = PublicOutcome::from_bytes(proof.public_values.as_slice())?;
            let statement_match = match statement {
                None => None,
                Some(path) => {
                    let bytes =
                        fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    Some(outcome.statement_digest == hex::encode(sha256(&bytes)))
                }
            };
            emit(&VerifyOutput {
                backend: BACKEND,
                ok,
                outcome,
                vk_hash: pk.verifying_key().bytes32(),
                statement_match,
                verify_seconds,
            })
        }
        Command::Serve => {
            let started = Instant::now();
            let client = ProverClient::from_env().await;
            let pk = client
                .setup(ELF)
                .await
                .map_err(|error| anyhow!("setup failed: {error}"))?;
            let setup_seconds = started.elapsed().as_secs_f64();
            emit(&ServeReady {
                event: "ready",
                backend: BACKEND,
                sp1_version: SP1_VERSION,
                elf_sha256: hex::encode(sha256(&ELF)),
                vk_hash: pk.verifying_key().bytes32(),
                guest_features: GUEST_FEATURES
                    .split(',')
                    .filter(|item| !item.is_empty())
                    .collect(),
                setup_seconds,
                pid: std::process::id(),
            })?;
            let ready = Instant::now();
            let stdin = std::io::stdin();
            let mut job_index = 0usize;
            for line in stdin.lock().lines() {
                let line = line.context("reading a job line")?;
                if line.trim().is_empty() {
                    continue;
                }
                let job_started = Instant::now();
                let job: ServeJob = match serde_json::from_str(&line) {
                    Ok(job) => job,
                    Err(error) => {
                        emit(&ServeFailure {
                            event: "job",
                            id: None,
                            job_index,
                            ok: false,
                            error: format!("malformed job line: {error}"),
                            job_wall_seconds: job_started.elapsed().as_secs_f64(),
                            since_ready_seconds: ready.elapsed().as_secs_f64(),
                        })?;
                        job_index += 1;
                        continue;
                    }
                };
                match serve_job(&client, &pk, &job, job_index, ready).await {
                    Ok(result) => emit(&result)?,
                    Err(error) => emit(&ServeFailure {
                        event: "job",
                        id: job.id.clone(),
                        job_index,
                        ok: false,
                        error: format!("{error:#}"),
                        job_wall_seconds: job_started.elapsed().as_secs_f64(),
                        since_ready_seconds: ready.elapsed().as_secs_f64(),
                    })?,
                }
                job_index += 1;
            }
            Ok(())
        }
        Command::VerifyMany => {
            let started = Instant::now();
            let client = ProverClient::from_env().await;
            let client_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
            let setup_seconds = started.elapsed().as_secs_f64();
            emit(&serde_json::json!({"event": "ready", "backend": BACKEND, "sp1_version": SP1_VERSION,
                "elf_sha256": hex::encode(sha256(&ELF)), "vk_hash": pk.verifying_key().bytes32(),
                "client_seconds": client_seconds, "setup_seconds": setup_seconds, "pid": std::process::id()}))?;
            let ready = Instant::now();
            for (job_index, line) in std::io::stdin().lock().lines().enumerate() {
                let line = line.context("reading a job line")?;
                if line.trim().is_empty() {
                    continue;
                }
                let job_started = Instant::now();
                let failure = |id: Option<String>, error: String| ServeFailure {
                    event: "verify", id, job_index, ok: false, error,
                    job_wall_seconds: job_started.elapsed().as_secs_f64(), since_ready_seconds: ready.elapsed().as_secs_f64(),
                };
                match serde_json::from_str::<VerifyJob>(&line) {
                    Err(error) => emit(&failure(None, format!("malformed job line: {error}")))?,
                    Ok(job) => match verify_job(&client, &pk, &job, job_index) {
                        Ok(result) => emit(&result)?,
                        Err(error) => emit(&failure(job.id.clone(), format!("{error:#}")))?,
                    },
                }
            }
            Ok(())
        }
        #[cfg(feature = "relation-bare")]
        Command::BareExecute { set, flip_y } => bare_cmd::cmd_execute(set, flip_y).await,
        #[cfg(feature = "relation-bare")]
        Command::BareProve { set, out_dir, mode, reps, warmup_vus, skip_execute } => {
            bare_cmd::cmd_prove(set, out_dir, mode, reps, warmup_vus, skip_execute).await
        }
        #[cfg(feature = "relation-bare")]
        Command::BareStatement { set, out } => bare_cmd::cmd_statement(set, out).await,
        #[cfg(feature = "relation-bare")]
        Command::BareNegatives { instances, manifest_sha256, dataset, index, layout } => {
            bare_cmd::cmd_negatives(instances, manifest_sha256, dataset, index, layout).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_job_line_parses_with_optional_fields() {
        let job: VerifyJob = serde_json::from_str(r#"{"proof": "p.bin"}"#).unwrap();
        assert!(job.id.is_none() && job.statement.is_none() && job.proof == PathBuf::from("p.bin"));
        let job: VerifyJob = serde_json::from_str(r#"{"id": "chunk00", "proof": "p.bin", "statement": "s.bin"}"#).unwrap();
        assert_eq!((job.id.as_deref(), job.statement), (Some("chunk00"), Some(PathBuf::from("s.bin"))));
        assert!(serde_json::from_str::<VerifyJob>(r#"{"id": "no proof"}"#).is_err());
    }
}
