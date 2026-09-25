//! The `relation-committed` subcommands (feature `relation-committed`): `veritor_zk_common::committed` on a contiguous
//! id range of a frozen VU set (the same `--instances` / `--batch` sets as `bare_cmd`, chunked layout only).
//!
//! `committed-commit` commits the range natively -- the `sha256/row/v1` digest of every x row and W column, then the
//! frame-v3 trees `a`, `b`, `y` -- and writes the statement (`sp1-committed-statement/v1`: instance range, identity,
//! the three tree references); `committed-execute` runs the guest in the executor and checks its public values against
//! `committed::decide` and the statement's roots (`--flip-y I`: a wrong claimed word must give verdict 0;
//! `--tamper-x I`: a wrong operand word of VU `lo + I` with the honest y must be rejected, by the tree check when the
//! relation still holds; `--wrong-root T`: a statement whose root of tree `T` is wrong must be rejected);
//! `committed-prove` sets up once, proves a warm-up range, then `--reps` times commits the range and proves it, saving
//! and verifying every proof (SP1 and the tree check); `committed-verify` checks a saved proof against a statement
//! under the pinned verifying key.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use sp1_sdk::prelude::*;
use sp1_sdk::{ProverClient, SP1Proof};
use veritor_zk_common::bare::{self, Format};
use veritor_zk_common::committed::{self, TreeRef, TREE_NAMES};
use veritor_zk_common::sha256;

use super::bare_cmd::{BareSet, Layout, Loaded};
use super::{emit, load_proof, save_proof, Mode, BACKEND, ELF, GUEST_FEATURES, SP1_VERSION};

pub(crate) const STATEMENT_SCHEMA: &str = "sp1-committed-statement/v1";

/// A committed range: what the verifier is given, and the private digests the guest must reproduce.
struct Commitment {
    format: Format,
    identity: String,
    id: [u8; 32],
    lo: u64,
    hi: u64,
    trees: [TreeRef; 3],
    rows_seconds: f64,
    trees_seconds: f64,
}

/// The `sha256/row/v1` digests of rows `[lo, hi)` of a VU-major block, over `threads` threads.
fn digest_rows(format: Format, block: &[u8], lo: usize, hi: usize, role: u8, threads: usize) -> Vec<u8> {
    let rb = format.row_bytes();
    let hasher = committed::RowHasher::new(format);
    let rows = &block[lo * rb..hi * rb];
    let per = (hi - lo).div_ceil(threads.max(1)).max(1);
    let mut out = vec![0u8; 32 * (hi - lo)];
    std::thread::scope(|scope| {
        for (src, dst) in rows.chunks(per * rb).zip(out.chunks_mut(32 * per)) {
            let hasher = &hasher;
            scope.spawn(move || {
                for (row, d) in src.chunks_exact(rb).zip(dst.chunks_exact_mut(32)) {
                    d.copy_from_slice(&if role == committed::ROLE_X { hasher.x(row) } else { hasher.w(row) });
                }
            });
        }
    });
    out
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Commit `[lo, hi)` of `set` (its rows and the committed words) to the three frame-v3 trees.
fn commit(bare_set: &BareSet, set: &Loaded) -> Result<Commitment> {
    let (lo, hi) = bare_set.range(set)?;
    let identity = bare_set.identity(set)?;
    let (abs_lo, abs_hi) = ((set.base + lo) as u64, (set.base + hi) as u64);
    let format = set.format;
    let bindings = committed::bindings(format, &set.dataset, &set.tier, &set.manifest_sha256, abs_lo, abs_hi)
        .ok_or_else(|| anyhow!("dataset / tier / manifest need a JSON escape; not a frozen set's names"))?;
    let started = Instant::now();
    let n = threads();
    let dx = digest_rows(format, &set.x, lo, hi, committed::ROLE_X, n);
    let dw = digest_rows(format, &set.w, lo, hi, committed::ROLE_W, n);
    let rows_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let y_le = bare::encode_y(format, &set.y[lo..hi]).ok_or_else(|| anyhow!("a committed word does not fit {}", format.name()))?;
    let trees = committed::roots(format, &bindings, &y_le, &dx, &dw);
    let trees_seconds = started.elapsed().as_secs_f64();
    Ok(Commitment { format, id: bare::identity_digest(&identity), identity, lo: abs_lo, hi: abs_hi, trees, rows_seconds, trees_seconds })
}

fn statement_json(c: &Commitment, set: &Loaded) -> Value {
    json!({
        "schema": STATEMENT_SCHEMA, "scheme": "frame-v3", "row_schema": committed::ROW_SCHEMA, "binding_tag": committed::BINDING_TAG,
        "format": c.format.name(), "K": bare::K, "B": c.hi - c.lo, "identity": c.identity, "id": hex::encode(c.id),
        "instances": {"dataset": set.dataset, "tier": set.tier, "manifest_sha256": set.manifest_sha256, "lo": c.lo, "hi": c.hi},
        "trees": (0..3).map(|i| json!({"name": TREE_NAMES[i], "schema": committed::tree_schema(c.format, TREE_NAMES[i]),
            "binding": hex::encode(c.trees[i].binding), "owner": c.trees[i].owner, "count": c.trees[i].count,
            "root": hex::encode(c.trees[i].root)})).collect::<Vec<_>>(),
    })
}

/// A statement as the verifier takes it: the format, the id and the tree references, every binding and the id
/// re-derived from the instance range (the statement's own copies must agree).
fn parse_statement(doc: &Value) -> Result<(Format, [u8; 32], [TreeRef; 3])> {
    let text = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).map(str::to_string).ok_or_else(|| anyhow!("statement: {key} missing"));
    let int = |v: &Value, key: &str| v.get(key).and_then(Value::as_u64).ok_or_else(|| anyhow!("statement: {key} missing"));
    if doc.get("schema").and_then(Value::as_str) != Some(STATEMENT_SCHEMA)
        || doc.get("scheme").and_then(Value::as_str) != Some("frame-v3")
        || doc.get("row_schema").and_then(Value::as_str) != Some(committed::ROW_SCHEMA)
    {
        bail!("not a {STATEMENT_SCHEMA} frame-v3 {} statement", committed::ROW_SCHEMA);
    }
    let format = Format::from_name(&text(doc, "format")?).ok_or_else(|| anyhow!("statement: unknown format"))?;
    let inst = doc.get("instances").ok_or_else(|| anyhow!("statement: instances missing"))?;
    let (dataset, tier, manifest) = (text(inst, "dataset")?, text(inst, "tier")?, text(inst, "manifest_sha256")?);
    let (lo, hi) = (int(inst, "lo")?, int(inst, "hi")?);
    if lo >= hi {
        bail!("statement: empty range");
    }
    let id = bare::identity_digest(&bare::identity_string(&dataset, &tier, lo, hi, &manifest));
    if text(doc, "id")? != hex::encode(id) {
        bail!("statement: id is not sha256 of the instance range's identity");
    }
    let bindings = committed::bindings(format, &dataset, &tier, &manifest, lo, hi).ok_or_else(|| anyhow!("statement: unescapable names"))?;
    let trees = doc.get("trees").and_then(Value::as_array).ok_or_else(|| anyhow!("statement: trees missing"))?;
    if trees.len() != 3 {
        bail!("statement: {} trees, not a, b, y", trees.len());
    }
    let mut out = [TreeRef { binding: [0; 32], owner: 0, count: 0, root: [0; 32] }; 3];
    for (i, t) in trees.iter().enumerate() {
        let root: [u8; 32] = hex::decode(text(t, "root")?)?.try_into().map_err(|_| anyhow!("statement: a root is not 32 bytes"))?;
        let owner = t.get("owner").and_then(Value::as_i64).ok_or_else(|| anyhow!("statement: owner missing"))?;
        if text(t, "name")? != TREE_NAMES[i]
            || text(t, "schema")? != committed::tree_schema(format, TREE_NAMES[i])
            || text(t, "binding")? != hex::encode(bindings[i])
            || owner != committed::OWNERS[i]
            || int(t, "count")? != hi - lo
        {
            bail!("statement: tree {} does not match the instance range's binding / owner / count / schema", TREE_NAMES[i]);
        }
        out[i] = TreeRef { binding: bindings[i], owner, count: hi - lo, root };
    }
    Ok((format, id, out))
}

/// The guest input of `[lo, hi)` claiming `claimed`, and what the guest must commit (`committed::decide`).
fn input(bare_set: &BareSet, set: &Loaded, id: &[u8; 32], claimed: &[u32]) -> Result<(SP1Stdin, Vec<u8>, usize)> {
    if bare_set.layout != Layout::Chunks || bare_set.vus_per_read == 0 {
        bail!("relation-committed reads the chunked layout (--layout chunks, --vus-per-read > 0)");
    }
    let (lo, hi) = bare_set.range(set)?;
    let y_le = bare::encode_y(set.format, claimed).ok_or_else(|| anyhow!("a claimed word does not fit {}", set.format.name()))?;
    let header = bare::encode_header(set.format, bare::K as u32, bare_set.vus_per_read as u32, id, &y_le);
    let chunks: Vec<Vec<u8>> = (lo..hi)
        .step_by(bare_set.vus_per_read)
        .map(|at| bare::encode_chunk(set.format, &set.x, &set.w, at, (at + bare_set.vus_per_read).min(hi)))
        .collect();
    let expected = committed::decide(&header, &chunks);
    let bytes = header.len() + chunks.iter().map(Vec::len).sum::<usize>();
    let mut stdin = SP1Stdin::new();
    stdin.write_vec(header);
    for c in chunks {
        stdin.write_vec(c);
    }
    Ok((stdin, expected, bytes))
}

/// Change one operand word of the x row of VU `vu` (absolute index into the set).  Prefers the sign of an exact zero
/// word (+0 -> -0: every format's product is then a signed zero and the accumulation keeps its value, so the relation
/// still holds and only the commitment can reject); else flips the low bit of the first word.  Returns the byte offset
/// in the row and whether it was a zero's sign.
fn tamper_x(set: &mut Loaded, vu: usize) -> (usize, bool) {
    let rb = set.format.row_bytes();
    let row = &mut set.x[vu * rb..(vu + 1) * rb];
    let zero = match set.format {
        Format::Bf16Ampere | Format::Bf16Hopper => {
            (0..rb / 2).find(|&j| row[2 * j] == 0 && row[2 * j + 1] == 0).map(|j| (2 * j + 1, 0x80u8))
        }
        Format::Fp8Hopper | Format::Fp8Ada => (0..rb).find(|&j| row[j] == 0).map(|j| (j, 0x80)),
        Format::Fp4Nvf4 => (0..rb).filter(|j| j % 68 < 64).find(|&j| row[j] == 0).map(|j| (j, 0x08)),
    };
    match zero {
        Some((at, sign)) => {
            row[at] |= sign;
            (at, true)
        }
        None => {
            row[0] ^= 1;
            (0, false)
        }
    }
}

fn wrong_root(trees: &mut [TreeRef; 3], name: &str) -> Result<()> {
    let i = TREE_NAMES.iter().position(|t| *t == name).ok_or_else(|| anyhow!("--wrong-root takes a, b or y"))?;
    trees[i].root[0] ^= 1;
    Ok(())
}

async fn run_executor<P: Prover>(client: &P, stdin: SP1Stdin) -> Result<(Vec<u8>, Value)> {
    let started = Instant::now();
    let (public_values, report) = client.execute(ELF, stdin).await.map_err(|error| anyhow!("execution failed: {error}"))?;
    let seconds = started.elapsed().as_secs_f64();
    if report.exit_code != 0 {
        bail!("guest exited with status {}", report.exit_code);
    }
    let syscalls: serde_json::Map<String, Value> = report
        .syscall_counts
        .iter()
        .filter(|(_, n)| **n > 0)
        .map(|(code, n)| (format!("{code:?}"), json!(*n)))
        .collect();
    let tracker: serde_json::Map<String, Value> = report.cycle_tracker.iter().map(|(k, v)| (k.clone(), json!(*v))).collect();
    Ok((public_values.as_slice().to_vec(), json!({"total_cycles": report.total_instruction_count(), "gas": report.gas(),
        "cycle_tracker": tracker, "syscalls": syscalls, "execute_seconds": seconds})))
}

pub(crate) async fn cmd_commit(bare_set: BareSet, out: PathBuf) -> Result<()> {
    let set = bare_set.load()?;
    let started = Instant::now();
    let c = commit(&bare_set, &set)?;
    let seconds = started.elapsed().as_secs_f64();
    let doc = statement_json(&c, &set);
    fs::write(&out, serde_json::to_vec_pretty(&doc)?).with_context(|| format!("writing {}", out.display()))?;
    emit(&json!({"backend": BACKEND, "event": "commit", "statement": out.display().to_string(), "commit_seconds": seconds,
        "rows_seconds": c.rows_seconds, "trees_seconds": c.trees_seconds, "threads": threads(), "format": c.format.name(),
        "identity": c.identity, "roots": doc["trees"].as_array().unwrap().iter().map(|t| t["root"].clone()).collect::<Vec<_>>()}))
}

pub(crate) async fn cmd_execute(bare_set: BareSet, flip_y: Option<usize>, tamper: Option<usize>, wrong: Option<String>) -> Result<()> {
    let client = ProverClient::builder().mock().build().await;
    let mut set = bare_set.load()?;
    let c = commit(&bare_set, &set)?;
    let mut trees = c.trees;
    if let Some(name) = &wrong {
        wrong_root(&mut trees, name)?;
    }
    let (lo, hi) = bare_set.range(&set)?;
    let mut claimed = set.y[lo..hi].to_vec();
    if let Some(j) = flip_y {
        if j >= hi - lo {
            bail!("--flip-y {j} is outside the range's {} VUs", hi - lo);
        }
        claimed[j] ^= 1;
    }
    let tampered = match tamper {
        Some(j) if j >= hi - lo => bail!("--tamper-x {j} is outside the range's {} VUs", hi - lo),
        Some(j) => Some(tamper_x(&mut set, lo + j)),
        None => None,
    };
    let (stdin, expected, input_bytes) = input(&bare_set, &set, &c.id, &claimed)?;
    let (public, report) = run_executor(&client, stdin).await?;
    let parsed = committed::parse_public(&public).ok_or_else(|| anyhow!("the guest committed no relation-committed/v1 values"))?;
    let reason = committed::check(&public, c.format, &c.id, &trees);
    let negative = flip_y.is_some() || tamper.is_some() || wrong.is_some();
    emit(&json!({"backend": BACKEND, "event": "committed-execute", "format": c.format.name(), "identity": c.identity,
        "lo": lo, "hi": hi, "vus_per_read": bare_set.vus_per_read, "input_bytes": input_bytes, "report": report,
        "public_values_bytes": public.len(), "public_values_sha256": hex::encode(sha256(&public)),
        "matches_expected": public == expected, "verdict": parsed.verdict, "tree_check": reason,
        "accepted": reason.is_none(), "flipped": flip_y, "tampered_vu": tamper,
        "tamper": tampered.map(|(at, zero)| json!({"byte": at, "sign_of_zero": zero})), "wrong_root": wrong,
        "commit_seconds": c.rows_seconds + c.trees_seconds}))?;
    if public != expected {
        bail!("the guest's public values differ from committed::decide");
    }
    match (negative, reason) {
        (false, Some(r)) => bail!("the honest range was rejected: {r}"),
        (true, None) => bail!("a negative was accepted"),
        (true, Some(r)) if tampered.is_some_and(|(_, zero)| zero) && !r.contains("tree a") => {
            bail!("a relation-preserving x change should be caught by tree a, got: {r}")
        }
        _ => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_prove(
    bare_set: BareSet,
    out_dir: PathBuf,
    mode: Mode,
    reps: usize,
    warmup_vus: usize,
    skip_execute: bool,
    tamper: Option<usize>,
) -> Result<()> {
    let started = Instant::now();
    let client = ProverClient::from_env().await;
    let client_seconds = started.elapsed().as_secs_f64();
    let mut set = bare_set.load()?;
    let (lo, hi) = bare_set.range(&set)?;
    let started = Instant::now();
    let cold = commit(&bare_set, &set)?;
    let cold_seconds = started.elapsed().as_secs_f64();
    fs::create_dir_all(&out_dir)?;
    let statement_path = out_dir.join("statement.json");
    fs::write(&statement_path, serde_json::to_vec_pretty(&statement_json(&cold, &set))?)?;
    let claimed = set.y[lo..hi].to_vec();
    let tampered = tamper.map(|j| tamper_x(&mut set, lo + j.min(hi - lo - 1)));
    let (stdin, expected, input_bytes) = input(&bare_set, &set, &cold.id, &claimed)?;
    let want_accept = tampered.is_none();
    if committed::parse_public(&expected).map(|p| p.verdict) != Some(bare::VERDICT_ACCEPT) {
        bail!("the range does not satisfy the relation natively (verdict 0): nothing to prove");
    }
    if !skip_execute {
        let (public, report) = run_executor(&client, stdin.clone()).await?;
        if public != expected {
            bail!("the guest's public values differ from committed::decide");
        }
        emit(&json!({"event": "execute", "report": report, "tree_check": committed::check(&public, cold.format, &cold.id, &cold.trees)}))?;
    }
    let started = Instant::now();
    let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
    let setup_seconds = started.elapsed().as_secs_f64();
    emit(&json!({"event": "setup", "backend": BACKEND, "sp1_version": SP1_VERSION, "elf_sha256": hex::encode(sha256(&ELF)),
        "vk_hash": pk.verifying_key().bytes32(), "guest_features": GUEST_FEATURES, "client_seconds": client_seconds,
        "setup_seconds": setup_seconds, "format": cold.format.name(), "identity": cold.identity, "statement": statement_path,
        "input_bytes": input_bytes, "commit_cold_seconds": cold_seconds, "commit_threads": threads(),
        "tamper": tampered.map(|(at, zero)| json!({"vu": tamper, "byte": at, "sign_of_zero": zero}))}))?;

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
        let warm = BareSet { hi: Some(lo + warmup_vus.min(hi - lo)), ..bare_set.clone() };
        let wc = commit(&warm, &set)?;
        let (wlo, whi) = warm.range(&set)?;
        let (warm_stdin, _, _) = input(&warm, &set, &wc.id, &set.y[wlo..whi])?;
        let started = Instant::now();
        let proof = prove(warm_stdin).await?;
        let seconds = started.elapsed().as_secs_f64();
        let ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
        emit(&json!({"event": "warmup", "vus": whi - wlo, "prove_seconds": seconds, "accepted": ok}))?;
    }

    for rep in 0..reps {
        let started = Instant::now();
        let c = commit(&bare_set, &set)?;
        let commit_seconds = started.elapsed().as_secs_f64();
        if tampered.is_none() && c.trees != cold.trees {
            bail!("rep {rep}: the commitment is not deterministic");
        }
        let started = Instant::now();
        let proof = prove(stdin.clone()).await?;
        let prove_seconds = started.elapsed().as_secs_f64();
        let path = out_dir.join(format!("proof-rep{rep}.bin"));
        let started = Instant::now();
        save_proof(&proof, &path)?;
        let save_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let sp1_ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
        let verify_seconds = started.elapsed().as_secs_f64();
        let public = proof.public_values.as_slice().to_vec();
        let started = Instant::now();
        let reason = committed::check(&public, cold.format, &cold.id, &cold.trees);
        let check_seconds = started.elapsed().as_secs_f64();
        let accepted = sp1_ok && reason.is_none();
        let shards = match &proof.proof {
            SP1Proof::Core(shards) => shards.len(),
            _ => 1,
        };
        drop(proof);
        let bytes = fs::read(&path)?;
        emit(&json!({"event": "rep", "rep": rep, "mode": mode.name(), "commit_seconds": commit_seconds,
            "commit_rows_seconds": c.rows_seconds, "commit_trees_seconds": c.trees_seconds, "prove_seconds": prove_seconds,
            "save_seconds": save_seconds, "verify_seconds": verify_seconds, "tree_check_seconds": check_seconds,
            "proof_path": path.display().to_string(), "proof_bytes": bytes.len(), "proof_sha256": hex::encode(sha256(&bytes)),
            "shards": shards, "public_values_bytes": public.len(), "public_values_sha256": hex::encode(sha256(&public)),
            "matches_expected": public == expected, "sp1_ok": sp1_ok, "tree_check": reason, "accepted": accepted}))?;
        if accepted != want_accept {
            bail!("rep {rep}: accepted = {accepted}, expected {want_accept} (sp1 {sp1_ok}, tree check {reason:?})");
        }
    }
    Ok(())
}

pub(crate) async fn cmd_verify(proof: PathBuf, statement: PathBuf, expect_vk: Option<String>, wrong: Option<String>) -> Result<()> {
    let doc: Value = serde_json::from_slice(&fs::read(&statement).with_context(|| format!("reading {}", statement.display()))?)?;
    let parsed = parse_statement(&doc);
    let started = Instant::now();
    let client = ProverClient::from_env().await;
    let pk = client.setup(ELF).await.map_err(|error| anyhow!("setup failed: {error}"))?;
    let setup_seconds = started.elapsed().as_secs_f64();
    let vk = pk.verifying_key().bytes32();
    let vk_pinned = expect_vk.as_ref().map(|want| want.eq_ignore_ascii_case(&vk));
    let proof = load_proof(Path::new(&proof))?;
    let started = Instant::now();
    let sp1_ok = client.verify(&proof, pk.verifying_key(), None).is_ok();
    let verify_seconds = started.elapsed().as_secs_f64();
    let public = proof.public_values.as_slice();
    let started = Instant::now();
    let reason = match parsed {
        Err(error) => Some(format!("{error:#}")),
        Ok((format, id, mut trees)) => match &wrong {
            Some(name) => wrong_root(&mut trees, name).map_or_else(|e| Some(e.to_string()), |_| committed::check(public, format, &id, &trees)),
            None => committed::check(public, format, &id, &trees),
        },
    };
    let check_seconds = started.elapsed().as_secs_f64();
    let verdict = committed::parse_public(public).map(|p| p.verdict);
    let ok = sp1_ok && vk_pinned != Some(false) && reason.is_none();
    emit(&json!({"backend": BACKEND, "event": "committed-verify", "ok": ok, "sp1_ok": sp1_ok, "vk_hash": vk,
        "expected_vk": expect_vk, "vk_pinned": vk_pinned, "verdict": verdict, "tree_check": reason, "wrong_root": wrong,
        "public_values_sha256": hex::encode(sha256(public)), "setup_seconds": setup_seconds, "verify_seconds": verify_seconds,
        "tree_check_seconds": check_seconds}))?;
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
