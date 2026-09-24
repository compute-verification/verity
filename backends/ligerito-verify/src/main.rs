//! ligerito-verify: the independent verifier for the Ligerito backend.
//!
//! ~~~text
//! ligerito-verify verify --proof X.proof [--coins X.coins] [--label L] [--json out.json] [--reps R] [--target-bits B] [--n-proofs N]
//!     # one PCS proof: LGRTOPV0 (statement embedded) or proto/pcs.py bytes (point in X.z); --coins = the
//!     # verifier's coins of a live session (u32 LE length + bytes each; interactive replay; default: Fiat-Shamir
//!     # diagnostic with transcript label L, default "ligerito-proto/v1").  The proof's own soundness figure (paper
//!     # (17) recomputed from its parameters) must be <= 2^-(B + log2 N), default B = 128, N = 1 (N = the number of
//!     # proofs the job was split into: the batch union bound); --target-bits 0 waives it (toys)
//! ligerito-verify batch --dir DIR [--jobs N] [--json out.json] [--label L] [--target-bits B] [--n-proofs N]
//!     # --n-proofs (else DIR/manifest.json "n_proofs"): the job's proof count; exactly N distinct statements accepted
//!     # every DIR/*.proof (with DIR/<name>.z / <name>.coins when present) must verify and every DIR/*.proof.neg must
//!     # be rejected -- a JSON *.proof / *.proof.neg is a ref.py (ligerito-ref/v2) fixture list and the rule applies
//!     # to every entry; every entry of DIR/*.ref.json (ref.py fixtures; tampers in DIR/<name>.ref.neg.json) must get
//!     # ref.py's verdict; with DIR/manifest.json (a list, or {"proofs": [...]}, of {"proof": name, "accepted": bool,
//!     # "reason": str}) every listed verdict must agree with ours; the union of the accepted proofs' own soundness
//!     # bounds must be <= 2^-B (default 128; --target-bits 0 waives it).  Exit 0 iff accepted.
//!     # LGTO0001 (ligerito-relation's end-to-end batch proofs, run.py's dump layout): DIR/*.lgto must verify and
//!     # DIR/*.lgto.neg be rejected, each against <stem>.stmt (LGTS0001) and DIR/key.bin (LGVK0001), coins as the
//!     # proof declares them (live: <stem>.coins; local: manifest "coins_seed"), manifest {"files": [{"proof",
//!     # "python_verdict": {"accepted"}}]} verdicts must agree.  The key must be a pinned one (--allow-any-key: any key
//!     # the statement names, flagged); proofs without the red-team-2 V1 zero claims are rejected (--allow-legacy reads
//!     # them, UNSOUND: no soundness figure, and batch lists them as a problem); local-coin proofs are verdicts only (no soundness figure; --no-local rejects them); proofs of
//!     # one statement count once in the union.  `verify --proof X.lgto [--stmt S] [--key K] [--coins C]` likewise.
//!     # --session PATH (repeatable; a live.ChallengeServerSession record dir or a dir of them): live proofs are
//!     # authenticated against the issuing verifier's record instead (`session`: slots from the record's coins, labels
//!     # and statement, never the dump's .coins; every round's MSG = the record's; session accepted) and claim the
//!     # interactive union x the recorded batches on the statement (red-team-3 R3-7 / R3-10); unauthenticated: rejected.
//! ligerito-verify ref-fixtures --file F.json [--neg N.json] [--json out.json] [--reps R]
//!     # ligerito-design's ref.py fixtures (`python -m backends.direct.ligerito.ref`, format ligerito-ref/v2): every
//!     # statement + proof must get ref.py's verdict; --neg = declarative tampers (fixtures/ref/negatives.json) replayed
//!     # on them, same rule.  CONFORMANCE ONLY: no soundness target is applied here (the fixtures are 8-query toys), so
//!     # an accept is "same verdict as ref.py", never a 2^-128 statement; use `batch` for that.
//! ligerito-verify pcs-sparse --file X.json [--json out.json]
//!     # a sparse-claim PCS document (tools/pcs_sparse_fixture.py, proto commit-first openings with sparse claims, t_pad
//!     # 0 / 32): every case's honest terms verify, every listed negative is rejected.  CONFORMANCE ONLY (toy sizes).
//! ligerito-verify sumcheck-fixture --fixture F.json [--seed fixture]   (LGSC0002 or LGSC0003, by magic)
//!     # ligerito-sumcheck's verifier fixture (layout, constraints, public rows, LGSC0002/3 proof, claims, recorded
//!     # transcript): verify with the recorded transcript and with LocalCoins(seed); claims must match; the fixture's
//!     # "negatives" must be rejected in the stage Python names; lists the virtual-row zero claims (red-team-2 V1)
//!     # the PCS must add -- the sumcheck layer alone accepts a proof with non-zero virtual rows by construction
//! ligerito-verify selftest [--n N] [--kprime a,b,c] [--queries s,s,s] [--rate-log2 R|r1,r2,r3] [--emit DIR]
//!     # reference prover -> verifier round trip (+ negatives); --emit writes the proof / coins / negatives to DIR
//! ~~~

use ligerito_verify::format::{looks_like_proto, read_pcs_v0, read_point, read_proto, write_pcs_v0};
use ligerito_verify::hash::{hex, sha256};
use ligerito_verify::lgto;
use ligerito_verify::params::Params;
use ligerito_verify::pcs::{verify, Report};
use ligerito_verify::refpcs;
use ligerito_verify::refprover::{prove, random_point, random_vector, tamper};
use ligerito_verify::session;
use ligerito_verify::transcript::{Coins, FiatShamir, Live, Local};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

struct Verdict {
    name: String,
    accepted: bool,
    reason: String,
    report: Option<Report>,
    params: Option<Params>,
    bytes: usize,
    mode: &'static str,
    seconds: f64,
    /// an LGTO0001 proof's end-to-end report (statement, relation sumchecks, zero claims, PCS)
    lgto: Option<lgto::Report>,
}

impl Verdict {
    /// the proof's own soundness figure: the interactive union its parameters give (the parameter target applies to it,
    /// whatever the coins; what the run establishes is `lgto.claimed_log2`)
    fn soundness_log2(&self) -> Option<f64> {
        match (&self.report, &self.lgto) {
            (Some(r), _) => Some(r.soundness_log2),
            // a pre-V1 proof has none: any claimed output verifies (red-team-2 V1)
            (None, Some(r)) if r.legacy => None,
            (None, Some(r)) => Some(r.soundness_log2),
            _ => None,
        }
    }
}

fn verdict_json(v: &Verdict) -> String {
    let rep = match &v.report {
        Some(r) => format!(
            "{{\"coins\":{},\"soundness_log2\":{:.3},\"opened_rows\":{},\"terms_final\":{}}}",
            r.coins, r.soundness_log2, r.opened_rows, r.terms_final
        ),
        None => "null".into(),
    };
    let params = match &v.params {
        Some(p) => format!(
            "{{\"n\":{},\"kprime\":{:?},\"queries\":{:?},\"rate_log2\":{:?},\"domain_shift\":{}}}",
            p.n, p.kprime, p.queries, p.rate_log2, p.domain_shift
        ),
        None => "null".into(),
    };
    let lg = match &v.lgto {
        Some(r) => format!(
            "{{\"relation\":{},\"coins\":{},\"coin_rounds\":{},\"soundness_log2\":{},\"claimed_log2\":{},\"claim_basis\":{},\"sumcheck\":\"{}\",\"zk\":{},\"zk_mode\":\"{}\",\"vus\":{},\"sub_batches\":{},\"n\":{},\"n_claims\":{},\"zero_blocks\":{},\"opened_rows\":{},\"legacy_no_zero_claims\":{},\"key_pinned\":{},\"key_derived_from\":{},\"key_sha256\":\"{}\",\"stmt_sha256\":\"{}\",\"live_session\":{}}}",
            json_str(&r.relation),
            json_str(&r.coins_kind),
            r.coins,
            if r.legacy { "null".into() } else { format!("{:.3}", r.soundness_log2) },
            r.claimed_log2.map(|c| format!("{c:.3}")).unwrap_or_else(|| "null".into()),
            json_str(&r.claim_basis),
            r.sumcheck,
            r.zk,
            r.zk_mode,
            r.vus,
            r.sub_batches,
            r.n,
            r.n_claims,
            r.zero_blocks,
            r.opened_rows,
            r.legacy,
            r.key_pinned,
            r.key_derived_from.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
            r.key_sha256,
            r.stmt_sha256,
            r.live_session.as_ref().map(|(s, b)| format!("{{\"session\":{},\"batch\":{b}}}", json_str(s))).unwrap_or_else(|| "null".into())
        ),
        None => "null".into(),
    };
    format!(
        "{{\"name\":{},\"accepted\":{},\"reason\":{},\"mode\":\"{}\",\"proof_bytes\":{},\"seconds\":{:.6},\"report\":{},\"params\":{},\"lgto\":{}}}",
        json_str(&v.name),
        v.accepted,
        json_str(&v.reason),
        v.mode,
        v.bytes,
        v.seconds,
        rep,
        params,
        lg
    )
}

/// How LGTO0001 proofs are verified (the command-line switches).
#[derive(Clone)]
struct LgtoOpts {
    opt: lgto::Options,
    /// local coins (`LocalCoins(seed)`, seed public): verdict only, never a soundness statement; `--no-local` rejects them
    allow_local: bool,
    stmt: Option<PathBuf>,
    key: Option<PathBuf>,
    coins: Option<PathBuf>,
    /// `--session`: the issuing verifiers' records; live proofs are then authenticated against them (slots derived from
    /// the record, never from `.coins`) and rejected when no recorded batch authenticates them
    sessions: Option<Vec<session::Batch>>,
    /// recorded batches (session id, batch) that already authenticate another proof of the run
    used: Vec<(String, usize)>,
}

/// `--session PATH` (repeatable): a `live.ChallengeServerSession` record dir (`session.json` + `verdict.json`) or a dir
/// of them.
fn load_sessions(paths: &[PathBuf]) -> Result<Vec<session::Batch>, String> {
    let mut dirs = Vec::new();
    for p in paths {
        if p.join("session.json").exists() {
            dirs.push(p.clone());
        } else {
            let mut sub: Vec<PathBuf> = std::fs::read_dir(p)
                .map_err(|e| format!("--session {}: {e}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|d| d.join("session.json").exists())
                .collect();
            if sub.is_empty() {
                return Err(format!("--session {}: no session.json in it or its subdirectories", p.display()));
            }
            sub.sort();
            dirs.extend(sub);
        }
    }
    let mut out = Vec::new();
    for d in dirs {
        let read = |f: &str| std::fs::read_to_string(d.join(f)).map_err(|e| format!("{}: {e}", d.join(f).display()));
        let name = d.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        out.extend(session::read_record(&read("session.json")?, &read("verdict.json")?, &name)?);
    }
    Ok(out)
}

/// run.py's manifest entry for one LGTO0001 file (`{"files": [{"proof", "stmt", "coins", "coins_seed", "coins_file"}]}`).
#[derive(Clone, Debug, Default)]
struct LgtoEntry {
    stmt: Option<String>,
    coins: Option<String>,
    seed: u64,
    coins_file: Option<String>,
}

fn manifest_lgto_entries(dir: &Path) -> Vec<(String, LgtoEntry)> {
    let Ok(text) = std::fs::read_to_string(dir.join("manifest.json")) else { return Vec::new() };
    let Ok(doc) = ligerito_verify::json::parse(&text) else { return Vec::new() };
    let list = doc.get("files").and_then(|f| f.as_arr()).unwrap_or(&[]);
    list.iter()
        .filter_map(|e| {
            let s = |k: &str| e.get(k).and_then(|v| v.as_str()).map(str::to_string);
            let seed = e.get("coins_seed").and_then(|v| v.as_usize()).unwrap_or(1) as u64;
            Some((s("proof")?, LgtoEntry { stmt: s("stmt"), coins: s("coins"), seed, coins_file: s("coins_file") }))
        })
        .collect()
}

fn is_lgto(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 8];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut buf)).is_ok() && &buf == lgto::MAGIC_PROOF
}

/// One LGTO0001 proof: statement `<stem>.stmt` (or the manifest's), key `DIR/key.bin`, coins as the proof declares them
/// (live: `<stem>.coins` / the manifest's `coins_file`; local: the manifest's `coins_seed`, default 1).
fn verify_lgto_file(proof_path: &Path, entry: Option<&LgtoEntry>, lo: &LgtoOpts, reps: usize) -> Verdict {
    let name = proof_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let dir = proof_path.parent().unwrap_or(Path::new("."));
    let fail = |reason: String, bytes: usize| Verdict { name: name.clone(), accepted: false, reason, report: None, params: None, bytes, mode: "lgto", seconds: 0.0, lgto: None };
    let proof = match std::fs::read(proof_path) {
        Ok(b) => b,
        Err(e) => return fail(format!("read: {e}"), 0),
    };
    let stmt_path = lo.stmt.clone().or_else(|| entry.and_then(|e| e.stmt.as_ref()).map(|s| dir.join(s))).or_else(|| sidecar(proof_path, "stmt"));
    let Some(stmt_path) = stmt_path else { return fail("no statement (<stem>.stmt)".into(), proof.len()) };
    let key_path = lo.key.clone().unwrap_or_else(|| dir.join("key.bin"));
    let (stmt, key) = match (std::fs::read(&stmt_path), std::fs::read(&key_path)) {
        (Ok(s), Ok(k)) => (s, k),
        (Err(e), _) => return fail(format!("statement {}: {e}", stmt_path.display()), proof.len()),
        (_, Err(e)) => return fail(format!("key {}: {e}", key_path.display()), proof.len()),
    };
    let declared = entry.and_then(|e| e.coins.clone());
    if let Some(recs) = &lo.sessions {
        if lgto::read_proof(&proof).is_ok_and(|p| p.params.coins == "live") {
            if declared.as_deref().is_some_and(|d| d != "live") {
                return fail(format!("the proof declares live coins, the manifest {}", declared.unwrap()), proof.len());
            }
            let t0 = Instant::now();
            let mut res = Err(String::new());
            for _ in 0..reps.max(1) {
                res = session::authenticate_unused(&stmt, &key, &proof, recs, &lo.used, &lo.opt);
            }
            let seconds = t0.elapsed().as_secs_f64() / reps.max(1) as f64;
            return match res {
                Ok(a) => Verdict {
                    name,
                    accepted: true,
                    reason: format!("ok (session {} batch {}, {} rounds)", a.session, a.batch, a.rounds),
                    report: None,
                    params: None,
                    bytes: proof.len(),
                    mode: "lgto live-session",
                    seconds,
                    lgto: Some(a.report),
                },
                Err(e) => Verdict { name, accepted: false, reason: format!("live session: {e}"), report: None, params: None, bytes: proof.len(), mode: "lgto live-session", seconds, lgto: None },
            };
        }
    }
    let coins_path = lo.coins.clone().or_else(|| entry.and_then(|e| e.coins_file.as_ref()).map(|s| dir.join(s))).or_else(|| sidecar(proof_path, "coins"));
    let seed = entry.map(|e| e.seed).unwrap_or(1);
    let mut coins_for = |kind: &str| -> Result<Box<dyn Coins>, String> {
        if let Some(d) = &declared {
            if d != kind {
                return Err(format!("the proof declares {kind} coins, the manifest {d}"));
            }
        }
        match kind {
            "fiat-shamir" => Ok(Box::new(FiatShamir::new(FiatShamir::LABEL))),
            "local" if lo.allow_local => Ok(Box::new(Local::new(seed))),
            "local" => Err("local coins (public seed): no soundness; --no-local rejects them".into()),
            "live" => {
                let p = coins_path.as_ref().ok_or("live coins: no .coins file")?;
                lgto::read_coins(&std::fs::read(p).map_err(|e| format!("coins {}: {e}", p.display()))?).map(|c| Box::new(c) as Box<dyn Coins>)
            }
            _ => Err(format!("coins {kind:?}: not a kind this verifier replays")),
        }
    };
    let t0 = Instant::now();
    let mut res = Err(String::new());
    for _ in 0..reps.max(1) {
        res = lgto::verify(&stmt, &key, &proof, &mut coins_for, &lo.opt);
    }
    let seconds = t0.elapsed().as_secs_f64() / reps.max(1) as f64;
    let mode = match &res {
        Ok(r) if r.coins_kind == "live" => "lgto live-replay",
        Ok(r) if r.coins_kind == "local" => "lgto local (diagnostic)",
        Ok(_) => "lgto fiat-shamir",
        Err(_) => "lgto",
    };
    match res {
        Ok(r) => Verdict { name, accepted: true, reason: "ok".into(), report: None, params: None, bytes: proof.len(), mode, seconds, lgto: Some(r) },
        Err(e) => Verdict { name, accepted: false, reason: e, report: None, params: None, bytes: proof.len(), mode, seconds, lgto: None },
    }
}

/// Verify one proof file, `reps` times (timing), with replayed coins when a coins file is given.
/// `X.proof` / `X.proof.neg` -> `X.<ext>` when it exists.
fn sidecar(proof_path: &Path, ext: &str) -> Option<PathBuf> {
    let name = proof_path.file_name()?.to_string_lossy().to_string();
    let stem = [".proof.neg", ".proof", ".lgto.neg", ".lgto"].iter().find_map(|x| name.strip_suffix(x))?;
    let p = proof_path.with_file_name(format!("{stem}.{ext}"));
    if p.exists() { Some(p) } else { None }
}

/// DIR/manifest.json: a JSON list (or an object with a "proofs" list) of {"proof": name, "accepted": bool, "reason": str}
/// -- the Python verifier's verdicts, matched by file name.  Missing or unparsable manifest: no entries.
fn manifest_verdicts(dir: &Path) -> Vec<(String, bool, String)> {
    let text = match std::fs::read_to_string(dir.join("manifest.json")) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let doc = match ligerito_verify::json::parse(&text) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };
    // run.py (LGTO0001 dumps): {"files": [{"proof", "python_verdict": {"accepted", "reason"}}]}
    let list = doc.as_arr().or_else(|| doc.get("proofs").and_then(|p| p.as_arr())).or_else(|| doc.get("files").and_then(|p| p.as_arr())).unwrap_or(&[]);
    list.iter()
        .filter_map(|e| {
            let name = e.get("proof")?.as_str()?.to_string();
            let v = e.get("python_verdict").unwrap_or(e);
            let acc = v.get("accepted")?.as_bool()?;
            let why = v.get("reason").and_then(|r| r.as_str()).unwrap_or("").to_string();
            Some((name, acc, why))
        })
        .collect()
}

/// DIR/manifest.json's `"n_proofs"` (an object manifest), the number of proofs the job was split into.
fn manifest_n_proofs(dir: &Path) -> Option<usize> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
    ligerito_verify::json::parse(&text).ok()?.get("n_proofs")?.as_usize()
}

fn verify_file(proof_path: &Path, coins_path: Option<&Path>, label: &[u8], reps: usize) -> Verdict {
    let name = proof_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let bytes = match std::fs::read(proof_path) {
        Ok(b) => b,
        Err(e) => {
            return Verdict { name, accepted: false, reason: format!("read: {e}"), report: None, params: None, bytes: 0, mode: "-", seconds: 0.0, lgto: None }
        }
    };
    let coins_raw = match coins_path {
        Some(p) => match std::fs::read(p) {
            Ok(b) => Some(b),
            Err(e) => {
                return Verdict { name, accepted: false, reason: format!("coins: {e}"), report: None, params: None, bytes: bytes.len(), mode: "-", seconds: 0.0, lgto: None }
            }
        },
        None => None,
    };
    let mode = if coins_raw.is_some() { "interactive-replay" } else { "fiat-shamir" };
    let t0 = Instant::now();
    let mut result: Result<Report, String> = Err("no reps".into());
    let mut params = None;
    for _ in 0..reps.max(1) {
        result = (|| {
            let (stmt, proof) = if looks_like_proto(&bytes) {
                let zp = sidecar(proof_path, "z").unwrap_or_else(|| proof_path.with_extension("z"));
                let zraw = std::fs::read(&zp).map_err(|e| format!("proto proof needs its point {}: {e}", zp.display()))?;
                read_proto(&bytes, read_point(&zraw)?)?
            } else {
                read_pcs_v0(&bytes)?
            };
            params = Some(proof.params.clone());
            let rep = match &coins_raw {
                Some(raw) => {
                    let mut lv = Live::parse(raw)?;
                    let rep = verify(&stmt, &proof, &mut lv)?;
                    if !lv.exhausted() {
                        return Err("coins: unused verifier coins after the protocol ended".into());
                    }
                    rep
                }
                None => verify(&stmt, &proof, &mut FiatShamir::new(label))?,
            };
            Ok(rep)
        })();
    }
    let seconds = t0.elapsed().as_secs_f64() / reps.max(1) as f64;
    match result {
        Ok(rep) => Verdict { name, accepted: true, reason: "ok".into(), report: Some(rep), params, bytes: bytes.len(), mode, seconds, lgto: None },
        Err(e) => Verdict { name, accepted: false, reason: e, report: None, params, bytes: bytes.len(), mode, seconds, lgto: None },
    }
}

/// ref.py fixtures (+ optional tampers): one verdict per statement, `accepted` = ours, plus whether it agrees with
/// ref.py's recorded verdict (a disagreement is a problem for the caller).
fn ref_fixture_verdicts(fixtures_path: &Path, neg_path: Option<&Path>, reps: usize) -> Vec<(Verdict, bool)> {
    let file_name = fixtures_path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let text = match std::fs::read_to_string(fixtures_path) {
        Ok(t) => t,
        Err(e) => return vec![(Verdict { name: file_name, accepted: false, reason: format!("read: {e}"), report: None, params: None, bytes: 0, mode: "ref.py", seconds: 0.0, lgto: None }, false)],
    };
    let fxs = match refpcs::load_fixtures(&text) {
        Ok(f) => f,
        Err(e) => return vec![(Verdict { name: file_name, accepted: false, reason: format!("parse: {e}"), report: None, params: None, bytes: text.len(), mode: "ref.py", seconds: 0.0, lgto: None }, false)],
    };
    let mut out = Vec::new();
    let run = |name: String, fx: &refpcs::Fixture, doc: &ligerito_verify::json::Json, expect_ok: bool, expect_reason: &str| -> (Verdict, bool) {
        let t0 = Instant::now();
        let mut res = Err(String::new());
        for _ in 0..reps.max(1) {
            res = refpcs::verify_fixture(fx, doc);
        }
        let seconds = t0.elapsed().as_secs_f64() / reps.max(1) as f64;
        let (accepted, reason, report) = match res {
            Ok(r) => (true, "ok".to_string(), Some(r)),
            Err(e) => (false, e, None),
        };
        let agree = accepted == expect_ok;
        let reason = if agree { reason } else { format!("{reason} -- ref.py said {} ({expect_reason})", if expect_ok { "accept" } else { "reject" }) };
        let bytes = doc.dump().len();
        (Verdict { name, accepted, reason, report, params: None, bytes, mode: "ref.py", seconds, lgto: None }, agree)
    };
    for fx in &fxs {
        out.push(run(format!("{file_name}#{}", fx.name), fx, &fx.proof, fx.expect_ok, &fx.expect_reason));
    }
    if let Some(np) = neg_path {
        match std::fs::read_to_string(np).map_err(|e| e.to_string()).and_then(|t| refpcs::load_tampers(&t)) {
            Ok(tampers) => {
                for t in &tampers {
                    let name = format!("{file_name}#{}", t.name);
                    let Some(fx) = fxs.get(t.base) else {
                        out.push((Verdict { name, accepted: false, reason: format!("tamper base {} out of range", t.base), report: None, params: None, bytes: 0, mode: "ref.py", seconds: 0.0, lgto: None }, false));
                        continue;
                    };
                    let mut doc = fx.proof.clone();
                    if let Err(e) = refpcs::apply_ops(&mut doc, &t.ops) {
                        out.push((Verdict { name, accepted: false, reason: format!("tamper: {e}"), report: None, params: None, bytes: 0, mode: "ref.py", seconds: 0.0, lgto: None }, false));
                        continue;
                    }
                    out.push(run(name, fx, &doc, t.expect_ok, &t.expect_reason));
                }
            }
            Err(e) => out.push((Verdict { name: np.display().to_string(), accepted: false, reason: format!("tampers: {e}"), report: None, params: None, bytes: 0, mode: "ref.py", seconds: 0.0, lgto: None }, false)),
        }
    }
    out
}

/// A `*.proof` / `*.proof.neg` whose content is JSON is a ref.py (`ligerito-ref/v2`) fixture list (or one fixture):
/// every entry must be accepted (`.proof`) or rejected (`.proof.neg`), whatever its recorded `expect`.
fn is_json_file(path: &Path) -> bool {
    use std::io::Read;
    let mut buf = [0u8; 64];
    let n = std::fs::File::open(path).and_then(|mut f| f.read(&mut buf)).unwrap_or(0);
    matches!(buf[..n].iter().find(|b| !b.is_ascii_whitespace()), Some(b'[') | Some(b'{'))
}

fn ref_file_verdicts(path: &Path, must_accept: bool, reps: usize) -> Vec<(Verdict, bool)> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let wrapped;
    let path_used = if text.trim_start().starts_with('{') {
        // a single fixture object: wrap it in a list next to the original
        wrapped = std::env::temp_dir().join(format!("ligerito-verify-{}-{}.json", std::process::id(), path.file_name().unwrap().to_string_lossy()));
        std::fs::write(&wrapped, format!("[{text}]")).ok();
        wrapped.as_path()
    } else {
        path
    };
    let mut out: Vec<(Verdict, bool)> = ref_fixture_verdicts(path_used, None, reps)
        .into_iter()
        .map(|(mut v, _)| {
            let agree = v.accepted == must_accept;
            if let Some(base) = path.file_name() {
                v.name = format!("{}#{}", base.to_string_lossy(), v.name.split_once('#').map(|x| x.1).unwrap_or(&v.name));
            }
            if !agree {
                v.reason = format!("{} but the file suffix says {}", if v.accepted { "accepted" } else { v.reason.as_str() }, if must_accept { "accept" } else { "reject" });
            }
            (v, agree)
        })
        .collect();
    if path_used != path {
        std::fs::remove_file(path_used).ok();
    }
    if out.is_empty() {
        out.push((Verdict { name: path.display().to_string(), accepted: false, reason: "empty fixture list".into(), report: None, params: None, bytes: 0, mode: "ref.py", seconds: 0.0, lgto: None }, !must_accept));
    }
    out
}

fn write_json(path: Option<String>, text: &str) {
    if let Some(p) = path {
        std::fs::write(&p, text).unwrap_or_else(|e| panic!("--json {p}: {e}"));
    }
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse().expect("list of integers")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let label = arg(&args, "--label").unwrap_or_else(|| String::from_utf8_lossy(FiatShamir::LABEL).to_string());
    let reps: usize = arg(&args, "--reps").map(|s| s.parse().expect("--reps")).unwrap_or(1);
    // soundness target enforced by `verify` (per proof) and `batch` (union bound); 0 waives it (toy fixtures, conformance runs)
    let target_bits: f64 = arg(&args, "--target-bits").map(|s| s.parse().expect("--target-bits")).unwrap_or(128.0);
    // red-team F5: a job split into N proofs (Ligerito proofs carry no n_proofs) -- `verify` then needs 2^-(B + log2 N)
    // per proof, `batch` needs exactly N distinct accepted statements and their union <= 2^-B
    let n_proofs_arg: Option<usize> = arg(&args, "--n-proofs").map(|s| s.parse().expect("--n-proofs"));
    assert!(n_proofs_arg != Some(0), "--n-proofs must be positive");
    let flag = |f: &str| args.iter().any(|a| a == f);
    let session_paths: Vec<PathBuf> = args.windows(2).filter(|w| w[0] == "--session").map(|w| PathBuf::from(&w[1])).collect();
    let sessions = if session_paths.is_empty() { None } else { Some(load_sessions(&session_paths).unwrap_or_else(|e| panic!("{e}"))) };
    let lgto_opts = |n_proofs: usize| LgtoOpts {
        opt: lgto::Options {
            allow_any_key: flag("--allow-any-key"),
            allow_legacy: flag("--allow-legacy"),
            target_log2: if target_bits > 0.0 { Some(-(target_bits + (n_proofs as f64).log2())) } else { None },
        },
        allow_local: !flag("--no-local"),
        stmt: arg(&args, "--stmt").map(PathBuf::from),
        key: arg(&args, "--key").map(PathBuf::from),
        coins: arg(&args, "--coins").map(PathBuf::from),
        sessions: sessions.clone(),
        used: Vec::new(),
    };
    match cmd {
        "verify" => {
            let proof = PathBuf::from(arg(&args, "--proof").expect("--proof FILE"));
            let coins = arg(&args, "--coins").map(PathBuf::from);
            let mut v = if is_lgto(&proof) {
                let dir = proof.parent().unwrap_or(Path::new(".")).to_path_buf();
                let name = proof.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let entry = manifest_lgto_entries(&dir).into_iter().find(|(n, _)| *n == name).map(|x| x.1);
                verify_lgto_file(&proof, entry.as_ref(), &lgto_opts(n_proofs_arg.unwrap_or(1)), reps)
            } else {
                verify_file(&proof, coins.as_deref(), label.as_bytes(), reps)
            };
            // the proof's own soundness figure (recomputed from its parameters, never prover-supplied) must meet the target,
            // union-bounded over the job's --n-proofs
            let per_proof_bits = target_bits + (n_proofs_arg.unwrap_or(1) as f64).log2();
            if let (true, Some(r)) = (v.accepted, v.report.as_ref()) {
                if target_bits > 0.0 && r.soundness_log2 > -per_proof_bits {
                    v.accepted = false;
                    v.reason = format!(
                        "verifies, but its soundness 2^{:.2} misses the target 2^-{per_proof_bits:.3} (2^-{target_bits} over {} proof(s); --target-bits 0 to waive)",
                        r.soundness_log2,
                        n_proofs_arg.unwrap_or(1)
                    );
                }
            }
            let js = verdict_json(&v);
            println!("{js}");
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if v.accepted { 0 } else { 1 });
        }
        "batch" => {
            let dir = PathBuf::from(arg(&args, "--dir").expect("--dir DIR"));
            let jobs: usize = arg(&args, "--jobs")
                .map(|s| s.parse().expect("--jobs"))
                .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
            // *.proof must verify; *.proof.neg must be rejected (JSON ones are ref.py fixture lists, handled below)
            let mut files: Vec<(PathBuf, bool)> = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("--dir {}: {e}", dir.display()))
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter_map(|p| {
                    let name = p.file_name()?.to_string_lossy().to_string();
                    if name.ends_with(".proof") || name.ends_with(".lgto") {
                        Some((p, true))
                    } else if name.ends_with(".proof.neg") || name.ends_with(".lgto.neg") {
                        Some((p, false))
                    } else {
                        None
                    }
                })
                .collect();
            files.sort();
            let (json_files, bin_files): (Vec<_>, Vec<_>) = files.into_iter().partition(|(p, _)| is_json_file(p));
            let files = bin_files;
            let manifest = manifest_verdicts(&dir);
            let entries = manifest_lgto_entries(&dir);
            let lo = lgto_opts(n_proofs_arg.or_else(|| manifest_n_proofs(&dir)).unwrap_or(1));
            let next = AtomicUsize::new(0);
            let results: Vec<std::sync::Mutex<Option<Verdict>>> = files.iter().map(|_| std::sync::Mutex::new(None)).collect();
            let t0 = Instant::now();
            std::thread::scope(|sc| {
                for _ in 0..jobs.max(1).min(files.len().max(1)) {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= files.len() {
                            break;
                        }
                        let p = &files[i].0;
                        let v = if is_lgto(p) {
                            let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                            verify_lgto_file(p, entries.iter().find(|(n, _)| *n == name).map(|x| &x.1), &lo, reps)
                        } else {
                            let coins = sidecar(p, "coins");
                            verify_file(p, coins.as_deref(), label.as_bytes(), reps)
                        };
                        *results[i].lock().unwrap() = Some(v);
                    });
                }
            });
            let mut verdicts: Vec<Verdict> = results.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
            // one recorded verifier batch authenticates one proof: in file order, a proof whose batch is taken is
            // re-authenticated against the batches still unused
            let mut used: Vec<(String, usize)> = Vec::new();
            for (i, (p, _)) in files.iter().enumerate() {
                let Some(sb) = verdicts[i].lgto.as_ref().and_then(|r| r.live_session.clone()).filter(|_| verdicts[i].accepted) else { continue };
                if used.contains(&sb) {
                    let lo2 = LgtoOpts { used: used.clone(), ..lo.clone() };
                    let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                    verdicts[i] = verify_lgto_file(p, entries.iter().find(|(n, _)| *n == name).map(|x| &x.1), &lo2, reps);
                }
                if let Some(sb) = verdicts[i].lgto.as_ref().and_then(|r| r.live_session.clone()).filter(|_| verdicts[i].accepted) {
                    used.push(sb);
                }
            }
            let mut problems: Vec<String> = Vec::new();
            // ref.py fixture lists: DIR/X.ref.json (+ DIR/X.ref.neg.json tampers), each entry must get ref.py's verdict
            let mut ref_files: Vec<PathBuf> = std::fs::read_dir(&dir)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().map(|n| n.to_string_lossy().ends_with(".ref.json")).unwrap_or(false))
                .collect();
            ref_files.sort();
            let mut n_ref = 0usize;
            let (mut n_json_pos, mut n_json_neg) = (0usize, 0usize);
            for (jf, must_accept) in &json_files {
                for (v, agree) in ref_file_verdicts(jf, *must_accept, reps) {
                    if *must_accept {
                        n_json_pos += 1;
                    } else {
                        n_json_neg += 1;
                    }
                    if !agree {
                        problems.push(format!("{}: {}", v.name, v.reason));
                    }
                    verdicts.push(v);
                }
            }
            for rf in &ref_files {
                let neg = rf.with_file_name(rf.file_name().unwrap().to_string_lossy().replace(".ref.json", ".ref.neg.json"));
                for (v, agree) in ref_fixture_verdicts(rf, if neg.exists() { Some(neg.as_path()) } else { None }, reps) {
                    n_ref += 1;
                    if !agree {
                        problems.push(format!("{}: {}", v.name, v.reason));
                    }
                    verdicts.push(v);
                }
            }
            if verdicts.is_empty() {
                problems.push("no *.proof or *.ref.json in the directory".into());
            }
            for (v, (_, expect)) in verdicts.iter().zip(&files) {
                // (ref.py entries come after `files` and were checked above)
                if v.accepted != *expect {
                    problems.push(format!("{}: {} but expected {}", v.name, if v.accepted { "accepted" } else { "rejected" }, if *expect { "acceptance" } else { "rejection" }));
                }
            }
            let mut manifest_checked = 0usize;
            for (name, py_accepted, py_reason) in &manifest {
                if let Some(v) = verdicts.iter().find(|v| &v.name == name) {
                    manifest_checked += 1;
                    if v.accepted != *py_accepted {
                        problems.push(format!("{name}: python_verdict {py_accepted} ({py_reason}) but ours is {} ({})", v.accepted, v.reason));
                    }
                }
            }
            // union bound over the accepted proofs' own (interactive) soundness figures; LGTO0001 proofs of the same statement
            // (one batch proved under several coin kinds) count once, at their worst
            let union_log2 = {
                let mut by_stmt: Vec<(String, f64)> = Vec::new();
                let mut s = 0.0f64;
                for v in verdicts.iter().filter(|v| v.accepted) {
                    let Some(e) = v.soundness_log2() else { continue };
                    match v.lgto.as_ref().map(|r| r.stmt_sha256.clone()) {
                        Some(d) => match by_stmt.iter_mut().find(|(x, _)| *x == d) {
                            Some(slot) => slot.1 = slot.1.max(e),
                            None => by_stmt.push((d, e)),
                        },
                        None => s += e.exp2(),
                    }
                }
                s += by_stmt.iter().map(|(_, e)| e.exp2()).sum::<f64>();
                if s > 0.0 { s.log2() } else { f64::NEG_INFINITY }
            };
            let n_local = verdicts.iter().filter(|v| v.accepted && v.lgto.as_ref().is_some_and(|r| r.coins_kind == "local")).count();
            // what the accepted LGTO0001 proofs establish (red-team F12), union over distinct statements; null if any has no claim
            let claimed_union: Option<f64> = {
                let acc: Vec<&lgto::Report> = verdicts.iter().filter(|v| v.accepted).filter_map(|v| v.lgto.as_ref()).collect();
                if acc.is_empty() || acc.iter().any(|r| r.claimed_log2.is_none()) {
                    None
                } else {
                    let mut by_stmt: Vec<(&str, f64)> = Vec::new();
                    for r in &acc {
                        let c = r.claimed_log2.unwrap();
                        match by_stmt.iter_mut().find(|(d, _)| *d == r.stmt_sha256) {
                            Some(slot) => slot.1 = slot.1.max(c),
                            None => by_stmt.push((&r.stmt_sha256, c)),
                        }
                    }
                    Some(by_stmt.iter().map(|(_, c)| c.exp2()).sum::<f64>().log2())
                }
            };
            let n_unpinned = verdicts.iter().filter(|v| v.accepted && v.lgto.as_ref().is_some_and(|r| !r.key_pinned)).count();
            let n_legacy = verdicts.iter().filter(|v| v.accepted && v.lgto.as_ref().is_some_and(|r| r.legacy)).count();
            if n_legacy > 0 {
                problems.push(format!("{n_legacy} pre-V1 proof(s) accepted under --allow-legacy: no soundness (no zero claims on the virtual rows)"));
            }
            if target_bits > 0.0 && union_log2 > -target_bits {
                problems.push(format!("union soundness 2^{union_log2:.2} misses the target 2^-{target_bits} (--target-bits 0 to waive)"));
            }
            // the job's proof count (--n-proofs, else DIR/manifest.json "n_proofs"): exactly that many distinct statements
            // accepted (LGTO0001 proofs of one statement under several coin kinds are one proof of the job)
            let n_expected = n_proofs_arg.or_else(|| manifest_n_proofs(&dir));
            let n_acc = {
                let mut stmts: Vec<&str> = Vec::new();
                let mut other = 0usize;
                for v in verdicts.iter().filter(|v| v.accepted) {
                    match v.lgto.as_ref() {
                        Some(r) if !stmts.contains(&r.stmt_sha256.as_str()) => stmts.push(&r.stmt_sha256),
                        Some(_) => {}
                        None => other += 1,
                    }
                }
                stmts.len() + other
            };
            if let Some(n) = n_expected {
                if n_acc != n {
                    problems.push(format!("batch: n_proofs mismatch ({n_acc} distinct statements accepted, the job has n_proofs = {n})"));
                }
            }
            let all = problems.is_empty();
            let js = format!(
                "{{\"dir\":{},\"accepted\":{},\"target_bits\":{target_bits},\"n_proofs_expected\":{},\"n_proofs\":{},\"n_negatives\":{},\"n_ref_fixtures\":{},\"n_accepted\":{},\"manifest_checked\":{},\"union_soundness_log2\":{},\"accepted_local_coins\":{n_local},\"claimed_union_log2\":{},\"accepted_unpinned_key\":{n_unpinned},\"accepted_legacy\":{n_legacy},\"problems\":[{}],\"wall_seconds\":{:.6},\"proofs\":[{}]}}",
                json_str(&dir.display().to_string()),
                all,
                n_expected.map(|n| n.to_string()).unwrap_or_else(|| "null".into()),
                files.iter().filter(|f| f.1).count() + n_json_pos,
                files.iter().filter(|f| !f.1).count() + n_json_neg,
                n_ref,
                verdicts.iter().filter(|v| v.accepted).count(),
                manifest_checked,
                if union_log2.is_finite() { format!("{union_log2:.3}") } else { "null".into() },
                claimed_union.map(|c| format!("{c:.3}")).unwrap_or_else(|| "null".into()),
                problems.iter().map(|p| json_str(p)).collect::<Vec<_>>().join(","),
                t0.elapsed().as_secs_f64(),
                verdicts.iter().map(verdict_json).collect::<Vec<_>>().join(",")
            );
            println!("{js}");
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if all { 0 } else { 1 });
        }
        "ref-fixtures" => {
            let file = PathBuf::from(arg(&args, "--file").expect("--file F.json"));
            let neg = arg(&args, "--neg").map(PathBuf::from);
            let t0 = Instant::now();
            let vs = ref_fixture_verdicts(&file, neg.as_deref(), reps);
            let all = vs.iter().all(|(_, agree)| *agree);
            for (v, agree) in &vs {
                println!("{:60} {} {:8.3} ms  {}", v.name, if v.accepted { "accept" } else { "reject" }, v.seconds * 1e3, if *agree { "" } else { "DISAGREES WITH ref.py" });
            }
            let js = format!(
                "{{\"file\":{},\"accepted\":{},\"n\":{},\"n_accepted\":{},\"n_rejected\":{},\"wall_seconds\":{:.6},\"verdicts\":[{}]}}",
                json_str(&file.display().to_string()),
                all,
                vs.len(),
                vs.iter().filter(|(v, _)| v.accepted).count(),
                vs.iter().filter(|(v, _)| !v.accepted).count(),
                t0.elapsed().as_secs_f64(),
                vs.iter().map(|(v, _)| verdict_json(v)).collect::<Vec<_>>().join(",")
            );
            eprintln!("{} fixtures/tampers: {} accepted, {} rejected, {}", vs.len(), vs.iter().filter(|(v, _)| v.accepted).count(), vs.iter().filter(|(v, _)| !v.accepted).count(), if all { "all agree with ref.py" } else { "DISAGREEMENTS" });
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if all { 0 } else { 1 });
        }
        "zk-key" => {
            // the LGSC0004 key (`layout_for(zk=True)` + `constraints()`) of an S x l batch, derived from an LGSC0003 key;
            // `--out F` writes it; the verifier pins exactly these bytes when the LGSC0003 key is pinned
            let path = PathBuf::from(arg(&args, "--key").expect("--key LGSC0003-key.bin"));
            let l: usize = arg(&args, "--l").expect("--l L").parse().expect("--l");
            let s: usize = arg(&args, "--s").expect("--s S (sub-batches, a power of two)").parse().expect("--s");
            assert!(l.is_power_of_two() && s.is_power_of_two(), "--l and --s must be powers of two");
            let kb = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let key3 = lgto::read_key(&kb).unwrap_or_else(|e| panic!("{e}"));
            let zk = lgto::zk_key_of(&key3, s.trailing_zeros() as usize, l.trailing_zeros() as usize).unwrap_or_else(|e| panic!("{e}"));
            let zb = lgto::write_key(&zk);
            let (sid, vk3) = (hex(&key3.sys_id), hex(&sha256(&kb)));
            let pinned = lgto::relation_spec(&key3.relation).is_some_and(|sp| sp.pins.iter().any(|(a, b)| *a == sid && *b == vk3));
            if let Some(out) = arg(&args, "--out") {
                std::fs::write(&out, &zb).unwrap_or_else(|e| panic!("{out}: {e}"));
            }
            println!(
                "{{\"relation\":{},\"l\":{l},\"S\":{s},\"lgsc3_key_sha256\":\"{vk3}\",\"lgsc3_key_pinned\":{pinned},\"zk_key_sha256\":\"{}\",\"m\":{},\"n_i\":{},\"n_k\":{},\"K\":{},\"virt\":{},\"bytes\":{}}}",
                json_str(&key3.relation),
                hex(&sha256(&zb)),
                zk.m,
                zk.n_i,
                zk.n_k,
                zk.k,
                zk.virt.len(),
                zb.len()
            );
        }
        "pcs-sparse" => {
            // a sparse-claim PCS document (tools/pcs_sparse_fixture.py on a GPU, or lgto's pcs_sparse_proto_bytes): every
            // case's honest terms must verify and every negative must be rejected
            let path = PathBuf::from(arg(&args, "--file").expect("--file X.json"));
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let (all, body) = match lgto::verify_sparse_doc(&text) {
                Ok(cases) => {
                    let all = cases.iter().all(|c| c.accepted == c.expect_accept);
                    let body = cases
                        .iter()
                        .map(|c| format!("{{\"name\":{},\"expect_accept\":{},\"accepted\":{},\"reason\":{}}}", json_str(&c.name), c.expect_accept, c.accepted, json_str(&c.reason)))
                        .collect::<Vec<_>>()
                        .join(",");
                    (all, format!("\"cases\":[{body}]"))
                }
                Err(e) => (false, format!("\"error\":{}", json_str(&e))),
            };
            let js = format!("{{\"file\":{},\"accepted\":{all},{body}}}", json_str(&path.display().to_string()));
            println!("{js}");
            write_json(arg(&args, "--json"), &js);
            std::process::exit(if all { 0 } else { 1 });
        }
        "sumcheck-fixture" => {
            // ligerito-sumcheck's `python -m backends.direct.ligerito.sumcheck --fixture F.json` (optionally gzip-free):
            // verify the LGSC0002 proof against the layout / constraints / public rows twice -- with the recorded transcript
            // (every absorb byte-checked) and with the lane's LocalCoins(seed) -- and compare the claims.
            let path = PathBuf::from(arg(&args, "--fixture").expect("--fixture F.json"));
            let seed = arg(&args, "--seed").unwrap_or_else(|| "fixture".into());
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let t0 = Instant::now();
            let fx = ligerito_verify::relation::load_fixture(&text).unwrap_or_else(|e| panic!("{e}"));
            let t_load = t0.elapsed().as_secs_f64();
            use ligerito_verify::{lgsc3, relation};
            let run = |bytes: &[u8], coins: &mut dyn ligerito_verify::transcript::Coins| -> Result<Vec<relation::EvalClaim>, String> {
                if bytes.starts_with(lgsc3::MAGIC) {
                    lgsc3::verify(&fx.layout, &fx.constraints, &fx.pub_rows, &lgsc3::parse(bytes)?, coins)
                } else {
                    relation::verify(&fx.layout, &fx.constraints, &fx.pub_rows, &relation::parse_lgsc(bytes)?, coins)
                }
            };
            let format = String::from_utf8_lossy(&fx.proof[..fx.proof.len().min(8)]).to_string();
            let mut out = vec![format!("\"format\":{}", json_str(&format))];
            let t1 = Instant::now();
            let scripted = if fx.transcript.is_empty() {
                "absent".to_string()
            } else {
                let mut sc = ligerito_verify::transcript::Scripted::new(fx.transcript.clone());
                match run(&fx.proof, &mut sc) {
                    Ok(cl) => match sc.finish() {
                        Ok(()) if cl == fx.claims => "accept, claims match".to_string(),
                        Ok(()) => "accept but the claims differ".to_string(),
                        Err(e) => format!("accept but {e}"),
                    },
                    Err(e) => format!("reject: {e}"),
                }
            };
            let t_scripted = t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let mut lc = ligerito_verify::transcript::SumcheckLocal::new(seed.as_bytes());
            let local = match run(&fx.proof, &mut lc) {
                Ok(cl) if cl == fx.claims => "accept, claims match".to_string(),
                Ok(_) => "accept but the claims differ".to_string(),
                Err(e) => format!("reject: {e}"),
            };
            let t_local = t2.elapsed().as_secs_f64();
            // the fixture's recorded must-reject proofs: each rejected in the stage the Python verifier names
            let negs = lgsc3::fixture_negatives(&text).unwrap_or_else(|e| panic!("{e}"));
            let neg_ok = negs
                .iter()
                .filter(|(b, why)| {
                    let got = run(b, &mut ligerito_verify::transcript::SumcheckLocal::new(seed.as_bytes()));
                    got.is_err_and(|e| e.starts_with(why.split(':').next().unwrap_or("")))
                })
                .count();
            out.push(format!("\"scripted\":{}", json_str(&scripted)));
            out.push(format!("\"local_coins\":{}", json_str(&local)));
            out.push(format!("\"negatives\":{},\"negatives_rejected_same_stage\":{}", negs.len(), neg_ok));
            // red-team-2 V1: the PCS batch must also open these (value 0); the sumcheck layer alone cannot see them
            let zero = fx.claims.first().map(|c| relation::virtual_zero_claims(&fx.layout, c)).unwrap_or_default();
            let names: Vec<String> = zero.iter().map(|z| json_str(&z.name)).collect();
            out.push(format!("\"virtual_zero_claims\":[{}]", names.join(",")));
            // a red-team V1 forgery records its change of the committed w: -delta at (row_index, col) of a virtual row, on
            // an honest w that is zero there.  Each zero claim's true value on the forged w is then -delta eq(point, (row,
            // col)); a non-zero one is a claim the PCS opening cannot meet, i.e. the full verifier rejects.
            if let Some(rt) = ligerito_verify::json::parse(&text).ok().and_then(|d| d.get("redteam").cloned()) {
                let get = |k: &str| rt.get(k).and_then(|v| v.as_usize());
                if let (Some(row), Some(col), Some(delta)) = (get("row_index"), get("col"), get("delta")) {
                    use ligerito_verify::field::{Ext, P};
                    let n_i = fx.layout.n_i;
                    let bit = |t: usize| if t < n_i { (row >> t) & 1 } else { (col >> (t - n_i)) & 1 };
                    let caught: Vec<String> = zero
                        .iter()
                        .filter(|z| {
                            let eq = z.point.iter().enumerate().fold(Ext::from_base(1), |acc, (t, x)| acc.mul(&if bit(t) == 1 { *x } else { Ext::from_base(1).sub(x) }));
                            !eq.mul_base((P as u64 - delta as u64 % P as u64) as u32 % P).is_zero()
                        })
                        .map(|z| json_str(&z.name))
                        .collect();
                    out.push(format!("\"redteam_delta\":{{\"row\":{row},\"col\":{col},\"v1_caught\":{},\"nonzero_zero_claims\":[{}]}}", !caught.is_empty(), caught.join(",")));
                }
            }
            let ok = scripted.starts_with("accept, claims match") || (scripted == "absent" && local.starts_with("accept, claims match"));
            let ok = ok && local.starts_with("accept, claims match") && neg_ok == negs.len();
            println!(
                "{{\"fixture\":{},\"accepted\":{},{},\"proof_bytes\":{},\"n_i\":{},\"n_k\":{},\"n_c\":{},\"n_links\":{},\"nnz\":{},\"claims\":{},\"seconds_load\":{:.4},\"seconds_verify_scripted\":{:.6},\"seconds_verify_local\":{:.6}}}",
                json_str(&path.display().to_string()),
                ok,
                out.join(","),
                fx.proof.len(),
                fx.layout.n_i,
                fx.layout.n_k,
                fx.layout.n_c(),
                fx.layout.n_links(),
                fx.constraints.a.len() + fx.constraints.b.len() + fx.constraints.c.len(),
                fx.claims.len(),
                t_load,
                t_scripted,
                t_local
            );
            std::process::exit(if ok { 0 } else { 1 });
        }
        "selftest" => {
            let n: usize = arg(&args, "--n").map(|s| s.parse().expect("--n")).unwrap_or(12);
            let kprime = arg(&args, "--kprime").map(|s| parse_list(&s)).unwrap_or_else(|| vec![4, 3, 2]);
            let queries = arg(&args, "--queries").map(|s| parse_list(&s)).unwrap_or_else(|| vec![16; kprime.len()]);
            // --rate-log2 R or R1,R2,... (per committed round)
            let rates: Vec<usize> = arg(&args, "--rate-log2").map(|s| parse_list(&s)).unwrap_or_else(|| vec![2]);
            let rates = if rates.len() == 1 { vec![rates[0]; kprime.len()] } else { rates };
            let p = Params::with_rates(n, kprime, queries, rates);
            p.validate().unwrap_or_else(|e| panic!("{e}"));
            let f = random_vector(n, 1);
            let z = random_point(n, 2);
            let t0 = Instant::now();
            let (stmt, proof) = prove(&p, &f, &z, &mut FiatShamir::new(label.as_bytes()));
            let t_prove = t0.elapsed().as_secs_f64();
            let n_coins = p.kprime.iter().sum::<usize>() + 2 * p.rounds() - 1;
            let coins = Live::random_coins(n_coins, 12345);
            let (stmt_live, proof_live) = prove(&p, &f, &z, &mut Live::new(coins.clone()));
            verify(&stmt_live, &proof_live, &mut Live::new(coins.clone())).unwrap_or_else(|e| panic!("selftest live: {e}"));
            let bytes = write_pcs_v0(&stmt, &proof);
            let t1 = Instant::now();
            let rep = verify(&stmt, &proof, &mut FiatShamir::new(label.as_bytes())).unwrap_or_else(|e| panic!("selftest: {e}"));
            let t_verify = t1.elapsed().as_secs_f64();
            let negs = tamper(&stmt, &proof);
            let mut rejected = 0;
            for (what, s, pf) in &negs {
                match verify(s, pf, &mut FiatShamir::new(label.as_bytes())) {
                    Ok(_) => eprintln!("NEGATIVE ACCEPTED: {what}"),
                    Err(_) => rejected += 1,
                }
            }
            let (snd, terms) = p.soundness_log2();
            println!(
                "{{\"params\":{{\"n\":{},\"kprime\":{:?},\"queries\":{:?},\"rate_log2\":{:?}}},\"proof_bytes\":{},\"coins\":{},\"t_prove\":{:.4},\"t_verify\":{:.6},\"negatives\":{},\"rejected\":{},\"soundness_log2\":{:.2},\"terms\":{{{}}}}}",
                p.n, p.kprime, p.queries, p.rate_log2, bytes.len(), rep.coins, t_prove, t_verify, negs.len(), rejected, snd,
                terms.iter().map(|(k, v)| format!("\"{k}\":{v:.2}")).collect::<Vec<_>>().join(",")
            );
            if let Some(dir) = arg(&args, "--emit") {
                let dir = PathBuf::from(dir);
                std::fs::create_dir_all(&dir).expect("--emit dir");
                std::fs::write(dir.join("honest.proof"), &bytes).unwrap();
                std::fs::write(dir.join("live.proof"), write_pcs_v0(&stmt_live, &proof_live)).unwrap();
                std::fs::write(dir.join("live.coins"), Live::serialize(&coins)).unwrap();
                std::fs::write(dir.join("proto.proof"), ligerito_verify::format::write_proto(&stmt, &proof, rep.coins)).unwrap();
                std::fs::write(dir.join("proto.z"), ligerito_verify::transcript::ext_bytes(&stmt.z)).unwrap();
                let mut written = 0;
                for (j, (what, s, pf)) in negs.iter().enumerate() {
                    let b = write_pcs_v0(s, pf);
                    if b == bytes {
                        continue; // an in-memory tamper the container cannot express (e.g. an ext entry in a base row)
                    }
                    std::fs::write(dir.join(format!("neg{j:02}.proof.neg")), &b).unwrap();
                    std::fs::write(dir.join(format!("neg{j:02}.what")), what).unwrap();
                    written += 1;
                }
                eprintln!("wrote {} (honest.proof, live.proof/.coins, proto.proof/.z + {written} negatives as *.proof.neg)", dir.display());
            }
            std::process::exit(if rejected == negs.len() { 0 } else { 1 });
        }
        _ => {
            eprintln!("usage: ligerito-verify verify --proof X.proof [--coins X.coins] | batch --dir DIR [--jobs N] | selftest [--n N --kprime a,b --queries s,s --rate-log2 R --emit DIR]");
            std::process::exit(2);
        }
    }
}
