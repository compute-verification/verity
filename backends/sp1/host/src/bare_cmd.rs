//! The `relation-bare` subcommands (feature `relation-bare`): the Table 2 bare statement of
//! `veritor_zk_common::bare` on a contiguous id range of a frozen VU set -- a BF16 tier of
//! `fixtures/bench-instances/v1` (`--instances DIR --tier T`, format `bf16-ampere`) or a
//! `verity_sp1.format_instances` file (`--batch FILE`, `sp1-format-instances/v1`: the format and the frozen ref come
//! from its header, every block is checked against its sha256).
//!
//! `bare-execute` runs the guest in the executor (exact cycles, the public values checked against the host's own
//! evaluation of the kernel; `--flip-y I` claims a wrong word for VU `lo + I` and requires verdict 0);
//! `bare-prove` sets up once, proves a warm-up range, then proves the range `--reps` times, saving and verifying
//! every proof; `bare-statement` writes the statement bytes a verifier hashes (`verify --proof P --statement S` then
//! checks `ok`, `statement_match` and `verdict`); `bare-negatives` runs every committed negative of `vu-k1536-neg`
//! through the executor: its claimed word must give verdict 0, and the correct word of an in-domain one verdict 1.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use sp1_sdk::prelude::*;
use sp1_sdk::{ProverClient, SP1Proof};
use veritor_zk_common::bare::{self, Format};
use veritor_zk_common::sha256;

use super::{emit, execute, save_proof, ExecuteOutput, Mode, BACKEND, ELF, GUEST_FEATURES, SP1_VERSION};

/// The `verity_sp1.format_instances` file magic and schema.
const BATCH_MAGIC: &[u8; 8] = b"VYSP1FI\x01";
const BATCH_SCHEMA: &str = "sp1-format-instances/v1";

/// A contiguous id range `[lo, hi)` of a frozen VU set.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct BareSet {
    /// Directory holding `<tier>.x.u16`, `<tier>.w.u16`, `<tier>.y.u16` (a BF16 tier of `bench-instances/v1`).
    #[arg(long, conflicts_with = "batch", required_unless_present = "batch")]
    pub(crate) instances: Option<PathBuf>,
    #[arg(long, default_value = "vu-k1536")]
    pub(crate) tier: String,
    #[arg(long, default_value = "bench-instances/v1")]
    pub(crate) dataset: String,
    /// sha256 of the dataset's manifest.json (part of the identity string; required with --instances).
    #[arg(long, required_unless_present = "batch")]
    pub(crate) manifest_sha256: Option<String>,
    /// A `sp1-format-instances/v1` file (`verity_sp1.format_instances`): format, frozen ref and blocks from the file.
    #[arg(long)]
    pub(crate) batch: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub(crate) lo: usize,
    /// End of the range (default: every VU of the set).
    #[arg(long)]
    pub(crate) hi: Option<usize>,
    /// VUs per `read_vec` chunk (chunked layout).
    #[arg(long, default_value_t = 64)]
    pub(crate) vus_per_read: usize,
    /// The guest input layout: `chunks` (version 2: every VU's rows in order) or `indexed` (version 3: a table of
    /// distinct rows, read once, and a row pair per VU).  The statement is the same.
    #[arg(long, value_enum, default_value_t = Layout::Chunks)]
    pub(crate) layout: Layout,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    Chunks,
    Indexed,
}

impl Layout {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Layout::Chunks => "chunks",
            Layout::Indexed => "indexed",
        }
    }
}

/// A loaded set: VU-major operand blocks, the committed words, the identity's parts.
pub(crate) struct Loaded {
    pub format: Format,
    pub(crate) dataset: String,
    pub(crate) tier: String,
    pub(crate) manifest_sha256: String,
    /// The tier id of the set's first VU (the frozen ref's range start).
    pub(crate) base: usize,
    pub(crate) x: Vec<u8>,
    pub(crate) w: Vec<u8>,
    pub(crate) y: Vec<u32>,
}

impl Loaded {
    pub(crate) fn n(&self) -> usize {
        self.y.len()
    }
}

fn load_fixture(dir: &std::path::Path, tier: &str, dataset: &str, manifest_sha256: &str) -> Result<Loaded> {
    let read = |what: &str| {
        let path = dir.join(format!("{tier}.{what}.u16"));
        fs::read(&path).with_context(|| format!("reading {}", path.display()))
    };
    let (x, w, y) = (read("x")?, read("w")?, read("y")?);
    let format = Format::Bf16Ampere;
    let n = y.len() / 2;
    if y.len() % 2 != 0 || x.len() != n * format.row_bytes() || w.len() != n * format.row_bytes() {
        bail!("{tier}: {} y bytes, {} x bytes, {} w bytes are not {n} VUs of {} row bytes", y.len(), x.len(), w.len(), format.row_bytes());
    }
    let y = y.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u32).collect();
    Ok(Loaded { format, dataset: dataset.into(), tier: tier.into(), manifest_sha256: manifest_sha256.into(), base: 0, x, w, y })
}

fn load_batch(path: &std::path::Path) -> Result<Loaded> {
    let data = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if data.len() < 12 || &data[..8] != BATCH_MAGIC {
        bail!("{}: not a {BATCH_SCHEMA} file", path.display());
    }
    let hl = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let header: serde_json::Value = serde_json::from_slice(data.get(12..12 + hl).ok_or_else(|| anyhow!("{}: truncated header", path.display()))?)?;
    let field = |key: &str| header.get(key).ok_or_else(|| anyhow!("{}: header has no {key}", path.display()));
    let text = |v: &serde_json::Value, key: &str| -> Result<String> {
        v.get(key).and_then(|s| s.as_str()).map(str::to_string).ok_or_else(|| anyhow!("{}: header field {key} is not a string", path.display()))
    };
    let int = |v: &serde_json::Value| v.as_u64().map(|n| n as usize).ok_or_else(|| anyhow!("{}: {v} is not an integer", path.display()));
    if field("schema")?.as_str() != Some(BATCH_SCHEMA) {
        bail!("{}: schema {} is not {BATCH_SCHEMA}", path.display(), field("schema")?);
    }
    let name = text(&header, "format")?;
    let format = Format::from_name(&name).ok_or_else(|| anyhow!("{}: unknown format {name}", path.display()))?;
    let (n, k, rb, yb) = (int(field("n")?)?, int(field("K")?)?, int(field("row_bytes")?)?, int(field("y_bytes")?)?);
    if k != bare::K || rb != format.row_bytes() || yb != format.y_bytes() {
        bail!("{}: K {k}, row_bytes {rb}, y_bytes {yb} are not {name}'s {}, {}, {}", path.display(), bare::K, format.row_bytes(), format.y_bytes());
    }
    let o = 12 + hl;
    if data.len() != o + n * (2 * rb + yb) {
        bail!("{}: {} bytes, the header describes {}", path.display(), data.len(), o + n * (2 * rb + yb));
    }
    let (x, rest) = data[o..].split_at(n * rb);
    let (w, y) = rest.split_at(n * rb);
    for (block, key) in [(x, "x_sha256"), (w, "w_sha256"), (y, "y_sha256")] {
        if hex::encode(sha256(block)) != text(&header, key)? {
            bail!("{}: the {key} block does not match its header digest", path.display());
        }
    }
    let inst = field("instances")?;
    let range = inst.get("range").and_then(|r| r.as_array()).ok_or_else(|| anyhow!("{}: instances.range missing", path.display()))?;
    let base = int(range.first().ok_or_else(|| anyhow!("{}: empty instances.range", path.display()))?)?;
    let y = y
        .chunks_exact(yb)
        .map(|b| {
            let mut word = [0u8; 4];
            word[..yb].copy_from_slice(b);
            u32::from_le_bytes(word)
        })
        .collect();
    Ok(Loaded {
        format,
        dataset: text(inst, "dataset")?,
        tier: text(inst, "tier")?,
        manifest_sha256: text(inst, "manifest_sha256")?,
        base,
        x: x.to_vec(),
        w: w.to_vec(),
        y,
    })
}

/// A range's guest input with what the guest must commit.
pub(crate) struct Prepared {
    pub stdin: SP1Stdin,
    pub format: Format,
    pub statement: Vec<u8>,
    pub identity: String,
    /// `bare::decide` on the same buffers: the kernel run natively.
    pub expected: [u8; bare::PUBLIC_VALUES_LEN],
    pub input_bytes: usize,
    /// `read_vec` buffers after the header.
    pub chunks: usize,
    /// Distinct rows in the table (indexed layout).
    pub rows: Option<u32>,
}

impl BareSet {
    pub(crate) fn load(&self) -> Result<Loaded> {
        match (&self.batch, &self.instances) {
            (Some(path), _) => load_batch(path),
            (None, Some(dir)) => {
                let manifest = self.manifest_sha256.as_deref().ok_or_else(|| anyhow!("--manifest-sha256 is required with --instances"))?;
                load_fixture(dir, &self.tier, &self.dataset, manifest)
            }
            (None, None) => bail!("give --instances DIR or --batch FILE"),
        }
    }

    /// `[lo, hi)` within the loaded set.
    pub(crate) fn range(&self, set: &Loaded) -> Result<(usize, usize)> {
        let hi = self.hi.unwrap_or(set.n());
        if self.lo >= hi || hi > set.n() {
            bail!("range [{}, {hi}) is not inside the {} VUs of {}", self.lo, set.n(), set.tier);
        }
        Ok((self.lo, hi))
    }

    pub(crate) fn identity(&self, set: &Loaded) -> Result<String> {
        let (lo, hi) = self.range(set)?;
        Ok(bare::identity_string(&set.dataset, &set.tier, (set.base + lo) as u64, (set.base + hi) as u64, &set.manifest_sha256))
    }

    fn statement(&self, set: &Loaded, claimed: &[u32]) -> Result<Vec<u8>> {
        let y_le = bare::encode_y(set.format, claimed).ok_or_else(|| anyhow!("a claimed word does not fit {}", set.format.name()))?;
        Ok(bare::statement_bytes(set.format, &bare::identity_digest(&self.identity(set)?), bare::K as u32, &y_le))
    }

    /// The guest input for the range, claiming `claimed` (default: the committed words).
    fn prepare(&self, set: &Loaded, claimed: Option<Vec<u32>>) -> Result<Prepared> {
        let (lo, hi) = self.range(set)?;
        let claimed = claimed.unwrap_or_else(|| set.y[lo..hi].to_vec());
        if claimed.len() != hi - lo || self.vus_per_read == 0 {
            bail!("{} claimed words for a range of {}, vus_per_read {}", claimed.len(), hi - lo, self.vus_per_read);
        }
        let identity = self.identity(set)?;
        let id = bare::identity_digest(&identity);
        let y_le = bare::encode_y(set.format, &claimed).ok_or_else(|| anyhow!("a claimed word does not fit {}", set.format.name()))?;
        let (header, buffers, rows) = match self.layout {
            Layout::Chunks => {
                let header = bare::encode_header(set.format, bare::K as u32, self.vus_per_read as u32, &id, &y_le);
                let chunks: Vec<Vec<u8>> = (lo..hi)
                    .step_by(self.vus_per_read)
                    .map(|at| bare::encode_chunk(set.format, &set.x, &set.w, at, (at + self.vus_per_read).min(hi)))
                    .collect();
                (header, chunks, None)
            }
            Layout::Indexed => {
                let (index, table, rows) = bare::index_rows(set.format, &set.x, &set.w, lo, hi);
                (bare::encode_header_indexed(set.format, bare::K as u32, rows, &id, &y_le), vec![index, table], Some(rows))
            }
        };
        let expected = bare::decide(&header, &buffers);
        let mut stdin = SP1Stdin::new();
        let input_bytes = header.len() + buffers.iter().map(Vec::len).sum::<usize>();
        let n_chunks = buffers.len();
        stdin.write_vec(header);
        for buffer in buffers {
            stdin.write_vec(buffer);
        }
        let statement = bare::statement_bytes(set.format, &id, bare::K as u32, &y_le);
        Ok(Prepared { stdin, format: set.format, statement, identity, expected, input_bytes, chunks: n_chunks, rows })
    }
}

#[derive(Serialize)]
struct BareExecuteOutput {
    #[serde(flatten)]
    execute: ExecuteOutput,
    format: &'static str,
    identity: String,
    lo: usize,
    hi: usize,
    layout: &'static str,
    vus_per_read: usize,
    chunks: usize,
    rows: Option<u32>,
    input_bytes: usize,
    expected_public_values: String,
    matches_expected: bool,
    statement_sha256: String,
    /// `--flip-y I`: the VU (relative to `lo`) whose claimed word is wrong.
    flipped: Option<usize>,
}

async fn execute_set<P: Prover>(client: &P, bare_set: &BareSet, set: &Loaded, claimed: Option<Vec<u32>>) -> Result<(BareExecuteOutput, Prepared)> {
    let prepared = bare_set.prepare(set, claimed)?;
    let (lo, hi) = bare_set.range(set)?;
    let output = execute(client, prepared.stdin.clone(), (prepared.statement.len(), prepared.input_bytes, 0)).await?;
    let matches_expected = output.outcome.public_values == hex::encode(prepared.expected);
    Ok((
        BareExecuteOutput {
            execute: output,
            format: set.format.name(),
            identity: prepared.identity.clone(),
            lo,
            hi,
            layout: bare_set.layout.name(),
            vus_per_read: bare_set.vus_per_read,
            chunks: prepared.chunks,
            rows: prepared.rows,
            input_bytes: prepared.input_bytes,
            expected_public_values: hex::encode(prepared.expected),
            matches_expected,
            statement_sha256: hex::encode(sha256(&prepared.statement)),
            flipped: None,
        },
        prepared,
    ))
}

pub(crate) async fn cmd_execute(bare_set: BareSet, flip_y: Option<usize>) -> Result<()> {
    let client = ProverClient::builder().mock().build().await;
    let set = bare_set.load()?;
    let claimed = match flip_y {
        None => None,
        Some(j) => {
            let (lo, hi) = bare_set.range(&set)?;
            if j >= hi - lo {
                bail!("--flip-y {j} is outside the range's {} VUs", hi - lo);
            }
            let mut words = set.y[lo..hi].to_vec();
            words[j] ^= 1;
            Some(words)
        }
    };
    let (mut output, _) = execute_set(&client, &bare_set, &set, claimed).await?;
    output.flipped = flip_y;
    if !output.matches_expected {
        bail!("guest committed {}, the native kernel {}", output.execute.outcome.public_values, output.expected_public_values);
    }
    if flip_y.is_some() && output.execute.outcome.verdict {
        bail!("a wrong claimed word was accepted (verdict 1)");
    }
    emit(&output)
}

pub(crate) async fn cmd_statement(bare_set: BareSet, out: PathBuf) -> Result<()> {
    let set = bare_set.load()?;
    let (lo, hi) = bare_set.range(&set)?;
    let statement = bare_set.statement(&set, &set.y[lo..hi])?;
    fs::write(&out, &statement).with_context(|| format!("writing {}", out.display()))?;
    emit(&serde_json::json!({"backend": BACKEND, "format": set.format.name(), "identity": bare_set.identity(&set)?,
        "statement": out.display().to_string(), "statement_bytes": statement.len(), "statement_sha256": hex::encode(sha256(&statement)),
        "accepting_public_values": hex::encode(bare::public_values(&sha256(&statement), true))}))
}

#[derive(Serialize)]
struct RepOutput {
    event: &'static str,
    rep: usize,
    mode: &'static str,
    prove_seconds: f64,
    save_seconds: f64,
    verify_seconds: f64,
    proof_path: String,
    proof_bytes: u64,
    proof_sha256: String,
    shards: usize,
    public_values: String,
    accepted: bool,
}

pub(crate) async fn cmd_prove(bare_set: BareSet, out_dir: PathBuf, mode: Mode, reps: usize, warmup_vus: usize, skip_execute: bool) -> Result<()> {
    let started = Instant::now();
    let client = ProverClient::from_env().await;
    let client_seconds = started.elapsed().as_secs_f64();
    let set = bare_set.load()?;
    let prepared = if skip_execute {
        bare_set.prepare(&set, None)?
    } else {
        let (output, prepared) = execute_set(&client, &bare_set, &set, None).await?;
        if !output.matches_expected {
            bail!("guest committed {}, the native kernel {}", output.execute.outcome.public_values, output.expected_public_values);
        }
        emit(&serde_json::json!({"event": "execute", "report": output}))?;
        prepared
    };
    if prepared.expected[32] != bare::VERDICT_ACCEPT {
        bail!("the range does not satisfy the statement (native verdict 0): nothing to prove");
    }
    let started = Instant::now();
    let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
    let setup_seconds = started.elapsed().as_secs_f64();
    emit(&serde_json::json!({"event": "setup", "backend": BACKEND, "sp1_version": SP1_VERSION,
        "elf_sha256": hex::encode(sha256(&ELF)), "vk_hash": pk.verifying_key().bytes32(),
        "guest_features": GUEST_FEATURES, "client_seconds": client_seconds, "setup_seconds": setup_seconds,
        "format": prepared.format.name(), "identity": prepared.identity, "statement_sha256": hex::encode(sha256(&prepared.statement)),
        "layout": bare_set.layout.name(), "input_bytes": prepared.input_bytes, "chunks": prepared.chunks, "rows": prepared.rows}))?;
    fs::create_dir_all(&out_dir)?;
    fs::write(out_dir.join("statement.bin"), &prepared.statement)?;

    let prove = |stdin: SP1Stdin| {
        let request = client.prove(&pk, stdin);
        async move {
            match mode {
                Mode::Core => request.core().await,
                Mode::Compressed => request.compressed().await,
                Mode::Groth16 => request.groth16().await,
                Mode::Plonk => request.plonk().await,
            }
            .map_err(|error| anyhow!("prove failed: {error}"))
        }
    };

    if warmup_vus > 0 {
        let (lo, hi) = bare_set.range(&set)?;
        let warm = BareSet { hi: Some(lo + warmup_vus.min(hi - lo)), ..bare_set.clone() };
        let warm_input = warm.prepare(&set, None)?;
        let started = Instant::now();
        let proof = prove(warm_input.stdin).await?;
        let seconds = started.elapsed().as_secs_f64();
        let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
        emit(&serde_json::json!({"event": "warmup", "vus": warmup_vus.min(hi - lo), "prove_seconds": seconds, "accepted": ok,
            "public_values": hex::encode(proof.public_values.as_slice())}))?;
    }

    for rep in 0..reps {
        let started = Instant::now();
        let proof = prove(prepared.stdin.clone()).await?;
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
        let shards = match &proof.proof {
            SP1Proof::Core(shards) => shards.len(),
            _ => 1,
        };
        drop(proof);
        let bytes = fs::read(&path)?;
        emit(&RepOutput {
            event: "rep",
            rep,
            mode: mode.name(),
            prove_seconds,
            save_seconds,
            verify_seconds,
            proof_path: path.display().to_string(),
            proof_bytes: bytes.len() as u64,
            proof_sha256: hex::encode(sha256(&bytes)),
            shards,
            public_values: hex::encode(&public_values),
            accepted,
        })?;
        if !accepted {
            bail!("rep {rep}: proof not accepted (verify ok = {ok})");
        }
    }
    Ok(())
}

pub(crate) async fn cmd_negatives(instances: PathBuf, manifest_sha256: String, dataset: String, index: Option<PathBuf>, layout: Layout) -> Result<()> {
    let tier = "vu-k1536-neg";
    let index = index.unwrap_or_else(|| instances.join(format!("{tier}.index.json")));
    let doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(&index).with_context(|| format!("reading {}", index.display()))?)?;
    let rows = doc["rows"].as_array().ok_or_else(|| anyhow!("{}: no rows", index.display()))?;
    let client = ProverClient::builder().mock().build().await;
    let set = load_fixture(&instances, tier, &dataset, &manifest_sha256)?;
    let (mut rejected, mut correct_accepted, mut in_domain) = (0usize, 0usize, 0usize);
    for row in rows {
        let i = row[0].as_u64().ok_or_else(|| anyhow!("bad row {row}"))? as usize;
        let bare_set = BareSet {
            instances: Some(instances.clone()),
            tier: tier.to_string(),
            dataset: dataset.clone(),
            manifest_sha256: Some(manifest_sha256.clone()),
            batch: None,
            lo: i,
            hi: Some(i + 1),
            vus_per_read: 1,
            layout,
        };
        let (claimed, _) = execute_set(&client, &bare_set, &set, None).await?;
        let claimed_verdict = claimed.execute.outcome.verdict;
        if !claimed_verdict && claimed.matches_expected {
            rejected += 1;
        }
        let correct = match row[4].as_u64() {
            Some(word) => {
                in_domain += 1;
                let (out, _) = execute_set(&client, &bare_set, &set, Some(vec![word as u32])).await?;
                if out.execute.outcome.verdict && out.matches_expected {
                    correct_accepted += 1;
                }
                Some(out.execute.outcome.verdict)
            }
            None => None,
        };
        emit(&serde_json::json!({"event": "negative", "id": i, "mistake": row[1], "verdict_expected": row[5],
            "claimed_y": row[3], "correct_y": row[4], "claimed_verdict": claimed_verdict,
            "claimed_matches_native": claimed.matches_expected, "correct_verdict": correct,
            "cycles": claimed.execute.total_cycles}))?;
    }
    let ok = rejected == rows.len() && correct_accepted == in_domain;
    emit(&serde_json::json!({"event": "summary", "negatives": rows.len(), "claimed_rejected": rejected,
        "in_domain": in_domain, "correct_accepted": correct_accepted, "ok": ok}))?;
    if !ok {
        bail!("negatives: {rejected}/{} claimed words rejected, {correct_accepted}/{in_domain} correct words accepted", rows.len());
    }
    Ok(())
}
