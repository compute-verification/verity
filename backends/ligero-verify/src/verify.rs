//! The verifier, written from `backends/direct/ligero/PROTOCOL.md` (sections 3, 4, 7b, 8b, 8c) and the transcript
//! rules of `redteam_vu.md` F1-F3: every size and every challenge input comes from the verifier's own statement,
//! the proof is parsed against those bounds, and the checks run reject-first.

use crate::field::{self as f, add, mul, mul_acc, neg, sub, Ntt, P};
use crate::format::{Expr, Proof, Statement, System};
use crate::hash::{blake2b_256, blake3, shake256};
use crate::relation::{self, Relation};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const STATEMENT_TAG: &[u8] = b"ligero-b/statement|v2";
pub const COIN_TAG: &[u8] = b"ligero-b/coin|v1|";
/// Fiat-Shamir mode tags (challenges from the transcript prefix) and interactive-mode tags (`|v2`: challenges from
/// the verifier's coins alone, PROTOCOL.md 8c after b-zk-fix).
pub const CHALLENGE1_TAG_FS: &[u8] = b"ligero-b/c1|";
pub const CHALLENGE2_TAG_FS: &[u8] = b"ligero-b/c2|";
pub const CHALLENGE1_TAG_IA: &[u8] = b"ligero-b/c1|v2|";
pub const CHALLENGE2_TAG_IA: &[u8] = b"ligero-b/c2|v2|";
/// The campaign's Fiat-Shamir rule (ZK_CONSTRUCTION.md 3.3): eps_FS <= 2^60 x eps_round on every statistical term.
pub const FS_QUERIES_LOG2: f64 = 60.0;

// The pinned constraint systems (one per relation: `sys_id` = the Python `system_id`, plus this verifier's own digest
// of the exported tables) and everything else a relation fixes outside `system.bin` live in relation.rs.  A system
// file whose digests are not the statement's relation's pinned pair is refused unless `--allow-any-system`.

#[derive(Default, Debug, Clone)]
pub struct Timing {
    /// System parse + pinned-digest check + domain tables when they are built for this one verification (`verify`);
    /// 0 when they come from a `Shared` built once for a batch.
    pub shared: f64,
    pub parse: f64,
    pub soundness: f64,
    pub statement_hash: f64,
    pub challenge_hash: f64,
    pub merkle: f64,
    pub ntt: f64,
    pub dfold: f64,
    pub quadratic: f64,
    pub chain: f64,
    /// authentication = included: leaf recomputation + multiproof folds (0 for an excluded statement)
    pub auth: f64,
    pub total: f64,
}

#[derive(Clone)]
pub struct Options {
    pub threads: usize,
    /// `verify` (one proof on its own): reject unless the proof's parameters give at most `2^-bits` union-bounded over
    /// its OWN `n_proofs`.  `None` in `batch`: there the gate is `batch_bound` -- the union over the sub-batches actually
    /// presented, at `--target-bits` (PROTOCOL.md 8c, "Batch verification").
    pub soundness_bits: Option<f64>,
    pub allow_any_system: bool,
    /// Interactive mode only: the verifier's OWN step-0 randomness `r1 || s1 || r2 || s2` (128 bytes).  Without it a
    /// file verifier can only REPLAY the transcript's openings, and an interactive-mode accept is conditional on those
    /// coins being the verifier's (PROTOCOL.md 8c: the challenges are functions of the coins alone, so a prover that
    /// chose them knows `r` and `cols` before committing -- the file alone is not evidence).
    pub own_coins: Option<[u8; 128]>,
}

pub struct Soundness {
    pub per_proof_log2: f64,
    pub union_log2: f64,
    pub terms: Vec<(String, f64)>,
}

fn log2_pow(base_num: f64, base_den: f64, e: usize) -> f64 {
    e as f64 * (base_num.log2() - base_den.log2())
}

fn log2_sum(xs: &[f64]) -> f64 {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + xs.iter().map(|x| (x - m).exp2()).sum::<f64>().log2()
}

/// Appendix C of Ligero as `security.accounting.ligero_error` applies it (PROTOCOL.md section 6, 8b, 8c): with
/// `k = l + t_pad` (ZK) or `l + 1` (the accounting's stand-in for k = l), `e = floor((n - k) / 2)`, the challenge
/// field BabyBear^D, plus the step-0 coin-commitment hiding term when the commitment is on; union over `n_proofs`.
pub fn soundness(st: &Statement) -> Result<Soundness, String> {
    let (n, l, t, d) = (st.n as f64, st.l, st.t, st.d);
    let k = if st.zk { st.l + st.t_pad } else { st.l + 1 };
    if !(0 < st.l && st.l < k && k < st.n) {
        return Err("accounting: need 0 < l < k < n".into());
    }
    if st.n <= 2 * k {
        return Err(format!("accounting: the quadratic test needs n > 2k (n = {}, k = {k})", st.n));
    }
    if st.zk && !(k > st.l + t) {
        return Err(format!("accounting: ZK needs k > l + t (Lemma 4.15); k = {k}, l + t = {}", st.l + t));
    }
    let e = (st.n - k) / 2;
    let log_fs = d as f64 * (P as f64).log2();
    // Fiat-Shamir: every statistical term x 2^60 (the q x eps rule); interactive: no shift, plus the coin-hiding term.
    let shift = if st.interactive { 0.0 } else { FS_QUERIES_LOG2 };
    let mut terms = vec![
        ("irs_query".to_string(), log2_pow((st.n - e) as f64, n, t) + shift),
        ("linear_query".to_string(), log2_pow((k + l) as f64, n, t) + shift),
        ("quadratic_query".to_string(), log2_pow((2 * k) as f64, n, t) + shift),
        ("irs_field".to_string(), ((st.n + 1) as f64).log2() - log_fs + shift),
        ("linear_field".to_string(), -log_fs + shift),
        ("quadratic_field".to_string(), -log_fs + shift),
    ];
    if st.interactive {
        terms.push(("coin_commitment_hiding".to_string(), 1.0 + 64.0 - 256.0));
    }
    let per = log2_sum(&terms.iter().map(|x| x.1).collect::<Vec<_>>());
    Ok(Soundness { per_proof_log2: per, union_log2: per + (st.n_proofs as f64).log2(), terms })
}

/// The batch verifier's accounting verdict (PROTOCOL.md 8c, "Batch verification").
pub struct BatchBound {
    pub accepted: bool,
    pub reason: String,
    /// `log2 sum_i 2^b_i` over the sub-batches presented; the batch's bits are its negative.
    pub batch_log2: f64,
}

/// `128` for an integral target, else two decimals -- `protocol._bits_str`, so the reason strings match byte for byte.
fn bits_str(bits: f64) -> String {
    if bits.fract() == 0.0 {
        format!("{}", bits as i64)
    } else {
        format!("{bits:.2}")
    }
}

/// The batch rule on `N = per_proof_log2.len()` sub-batch proofs that each verified on their own (`protocol.batch_bound`,
/// same reason strings, same precedence): `per_proof_log2[i]` is THIS verifier's bound of proof `i` (`soundness()` from
/// the proof's own statement, never a prover-supplied number) and `n_proofs_claimed[i]` the count its `t` was sized for.
/// Accept iff every claimed count equals `N` and `sum_i 2^b_i <= 2^-target_bits`.  Each per-proof bound is a statement
/// about ONE commitment; the union bound is what makes "the batch is sound to 2^-target" true, and `n_proofs = N` is
/// what makes a sub-batch's `t` a commitment to this batch rather than a coincidence.
pub fn batch_bound(per_proof_log2: &[f64], n_proofs_claimed: &[usize], target_bits: f64, names: &[String]) -> BatchBound {
    let n = per_proof_log2.len();
    assert!(n > 0 && n_proofs_claimed.len() == n && names.len() == n, "batch_bound: one bound, one n_proofs, one name per sub-batch");
    let total = log2_sum(per_proof_log2);
    for (name, &k) in names.iter().zip(n_proofs_claimed) {
        if k != n {
            return BatchBound {
                accepted: false,
                reason: format!("batch: n_proofs mismatch ({name}: n_proofs = {k}, {n} sub-batches presented)"),
                batch_log2: total,
            };
        }
    }
    if total > -target_bits {
        return BatchBound {
            accepted: false,
            reason: format!("batch: union bound 2^{total:.2} over {n} sub-batches exceeds the target 2^-{}", bits_str(target_bits)),
            batch_log2: total,
        };
    }
    BatchBound {
        accepted: true,
        reason: format!("accept: all {n} sub-batches accepted; union bound 2^{total:.2} <= 2^-{}", bits_str(target_bits)),
        batch_log2: total,
    }
}

// ------------------------------------------------------------------------------------------- public statement ----

/// The words as `protocol._u_bytes(x, dtype)` writes them: `numpy.astype("<uN").tobytes()`, N = the relation's width
/// (the statement's width equals it, checked before this is called; a wider word would have been truncated by numpy,
/// which is exactly why the widths are pinned per relation instead of trusted from the file).
fn absorb_words(h: &mut Vec<u8>, words: &[u32], width: u8) {
    for &x in words {
        h.extend_from_slice(&x.to_le_bytes()[..width as usize]);
    }
}

fn statement_digest(st: &Statement, rel: &Relation, sys_id: &[u8; 32]) -> Result<Vec<u8>, String> {
    let k = st.k();
    let mut h = Vec::new();
    h.extend_from_slice(STATEMENT_TAG);
    // protocol.statement_digest: the relation tag right after the statement tag (b"" for the BF16 relations)
    h.extend_from_slice(rel.tag);
    // Python: f"|p={P}|hash=blake3|l={l}|n={n}|D={D}|t={t}|target={target!r}" -- repr of a float
    h.extend_from_slice(
        format!("|p={}|hash=blake3|l={}|n={}|D={}|t={}|target={}", P, st.l, st.n, st.d, st.t, py_float_repr(st.target_log2)?)
            .as_bytes(),
    );
    h.extend_from_slice(format!("|zk={}|k={}|mode={}", st.zk as u8, k, st.mode_name()).as_bytes());
    h.extend_from_slice(b"|sys=");
    h.extend_from_slice(sys_id);
    h.extend_from_slice(format!("|mode=chain|l={}|n_vus={}|steps={}", st.l, st.n_vus, st.steps).as_bytes());
    h.extend_from_slice(b"|a=");
    absorb_words(&mut h, &st.a, rel.word_bytes);
    h.extend_from_slice(b"|b=");
    absorb_words(&mut h, &st.b, rel.word_bytes);
    h.extend_from_slice(b"|y16=");
    absorb_words(&mut h, &st.y16, rel.y_bytes);
    // authentication = included: `auth.digest_bytes` (version, schema, VU indices, the three tree references) --
    // the challenges are bound to "the words at these indices under these roots"
    if let Some(ab) = &st.auth {
        h.extend_from_slice(&crate::auth::digest_bytes(ab));
    }
    Ok(blake2b_256(&h).to_vec())
}

/// Python's `repr(float)`: shortest round-trip digits, always with a fractional part or an exponent.  Rust's `{:?}`
/// agrees for the values a target takes (e.g. -128.0, -80.0, -133.02); documented in DISCREPANCIES.md (D2).
fn py_float_repr(x: f64) -> Result<String, String> {
    let s = format!("{:?}", x);
    if !x.is_finite() || s.contains('e') {
        // Rust: 1e-5 / 1e16; Python: 1e-05 / 1e+16 -- targets never reach these; refuse rather than guess
        return Err(format!("statement: target_log2 {x} has an exponent repr; refusing to guess Python's repr"));
    }
    Ok(s)
}

/// The pin name `"<a|b>[<i>].<s|m|e>"` -> (operand array, index, component).
fn pin_name(name: &str) -> Result<(char, usize, &str), String> {
    let (arr, rest) = name.split_at(1);
    let arr = match arr {
        "a" => 'a',
        "b" => 'b',
        _ => return Err(format!("pin {name}: not an operand pin (unit-mode systems are not supported)")),
    };
    let close = rest.find(']').ok_or_else(|| format!("pin {name}: bad name"))?;
    if !rest.starts_with('[') {
        return Err(format!("pin {name}: bad name"));
    }
    let i: usize = rest[1..close].parse().map_err(|_| format!("pin {name}: bad index"))?;
    Ok((arr, i, &rest[close + 1..]))
}

/// The operands per unit the SYSTEM was compiled for: `1 + max i` over its `a[i]` / `b[i]` pins (16 for the BF16
/// units, 32 for E4M3).  The statement's K must equal it (`public_vectors*`: `a.shape[1] != k` is a rejection).
fn system_k(sys: &System) -> Result<usize, String> {
    let mut k = 0usize;
    for (name, _) in &sys.pins {
        let (_, i, _) = pin_name(name)?;
        k = k.max(i + 1);
    }
    if k == 0 {
        return Err("system has no operand pins".into());
    }
    Ok(k)
}

/// The pinned decodes of the public operands (`unit.public_vectors` / `public_vectors_fp8`, chain mode): per pin name
/// `a[i].s|m|e` / `b[i].s|m|e` the (l,) vector mod p, decoded per the relation's operand format; rejects a word off
/// the finite domain and a statement whose K is not the system's.
fn public_pins(st: &Statement, sys: &System, rel: &Relation) -> Result<Vec<Vec<u32>>, String> {
    let l = st.l;
    let k_sys = system_k(sys)?;
    if st.k_ops != k_sys {
        return Err(format!("statement has K = {} operands per unit, the system pins {k_sys}", st.k_ops));
    }
    let mut out = Vec::with_capacity(sys.pins.len());
    for (name, _row) in &sys.pins {
        let (arr, i, comp) = pin_name(name)?;
        let src = if arr == 'a' { &st.a } else { &st.b };
        if i >= st.k_ops {
            return Err(format!("pin {name}: operand index >= K"));
        }
        let mut v = Vec::with_capacity(l);
        for j in 0..l {
            let (s, m, e) = rel.operand.decode(src[j * st.k_ops + i])?;
            v.push(match comp {
                ".s" => s,
                ".m" => m,
                ".e" => e.rem_euclid(P as i64) as u32,
                _ => return Err(format!("pin {name}: bad component")),
            });
        }
        out.push(v);
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------- challenges ----

fn expand(seed: &[u8], count: usize) -> Vec<u32> {
    let raw = shake256(seed, 4 * count);
    (0..count).map(|i| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()) % P).collect()
}

fn coin_prefix(st: &Statement, pf: &Proof) -> Vec<u8> {
    let mut v = Vec::new();
    if st.interactive {
        let c = pf.coins.as_ref().unwrap();
        v.extend_from_slice(b"|coins=");
        v.extend_from_slice(&c.c1);
        v.extend_from_slice(&c.c2);
    }
    v
}

fn coin_open(st: &Statement, r: &[u8; 32]) -> Vec<u8> {
    let mut v = Vec::new();
    if st.interactive {
        v.extend_from_slice(b"|r=");
        v.extend_from_slice(r);
    }
    v
}

fn u32s_le(xs: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 * xs.len());
    for x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

/// `_challenge2`: t distinct column indices by rejection from SHAKE-256(seed || ctr), ctr a u32 LE, 2t words per call.
fn column_challenge(seed: &[u8], t: usize, n: usize) -> Vec<u32> {
    let mut cols = Vec::with_capacity(t);
    let mut seen = vec![false; n];
    let mut ctr: u32 = 0;
    while cols.len() < t {
        let mut s = seed.to_vec();
        s.extend_from_slice(&ctr.to_le_bytes());
        let raw = shake256(&s, 4 * 2 * t);
        for i in 0..2 * t {
            let u = u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            let c = (u as usize) % n;
            if !seen[c] {
                seen[c] = true;
                cols.push(c as u32);
                if cols.len() == t {
                    break;
                }
            }
        }
        ctr = ctr.checked_add(1).expect("column challenge counter overflow");
    }
    cols
}

// ---------------------------------------------------------------------------------------------------- helpers ----

pub(crate) fn par_for<T: Send>(items: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.max(1).min(items.max(1));
    let mut out: Vec<Option<T>> = (0..items).map(|_| None).collect();
    std::thread::scope(|s| {
        let chunks = out.chunks_mut((items + threads - 1) / threads.max(1)).enumerate();
        let f = &f;
        for (ci, chunk) in chunks {
            let base = ci * ((items + threads - 1) / threads.max(1));
            s.spawn(move || {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(f(base + k));
                }
            });
        }
    });
    out.into_iter().map(|x| x.unwrap()).collect()
}

/// Evaluate an affine expression at column c of the opened matrix.
#[inline]
fn eval_expr(e: &Expr, uo: &[u32], t: usize, c: usize) -> u32 {
    let mut acc = e.c as u64;
    for (row, coef) in &e.lin {
        acc = mul_acc(acc, *coef, uo[*row as usize * t + c]);
    }
    acc as u32
}

/// The read-only domain tables of one `(l, n)`: NTT twiddles for H and for the coset, the coset scaling `g^i` and the
/// coset points `eta_c`.  Pure functions of `(l, n)`; built once per `Shared` and read by every verification.
pub struct Tables {
    pub ntt_l: Ntt,
    pub ntt_n: Ntt,
    pub scale: Vec<u32>,
    pub pts: Vec<u32>,
}

impl Tables {
    fn new(l: usize, n: usize) -> Tables {
        Tables { ntt_l: Ntt::new(l), ntt_n: Ntt::new(n), scale: f::coset_scale(n), pts: f::coset_points(n) }
    }
}

/// What every verification of a batch shares and none of them changes: the parsed system file with its two digests
/// (computed once; the pinned-digest verdict itself is per statement, since the statement names the relation), and
/// the domain tables per `(l, n)` -- one instance, built by the first verification that needs it.  Everything else
/// (statement, proof, challenges, every check) is per verification, so sub-batches stay independent.
pub struct Shared {
    sys: Result<System, String>,
    /// The system file's digests: the Python `sys_id` (trailing 32 bytes) and this verifier's table digest, hex.
    pub sys_id_hex: String,
    pub table_digest_hex: String,
    /// The relation whose pinned pair the system file carries, if any.
    pub pinned_relation: Option<&'static Relation>,
    tables: Mutex<HashMap<(usize, usize), Arc<Tables>>>,
}

impl Shared {
    pub fn new(sys_bytes: &[u8], _opt: &Options) -> Shared {
        let sys = System::parse(sys_bytes).map_err(|e| format!("system file: {e}"));
        let (sys_id_hex, table_digest_hex) = match &sys {
            Ok(s) => (crate::hash::hex(&s.sys_id), crate::hash::hex(&System::table_digest(sys_bytes))),
            Err(_) => (String::new(), String::new()),
        };
        let pinned_relation = relation::by_digests(&sys_id_hex, &table_digest_hex);
        Shared { sys, sys_id_hex, table_digest_hex, pinned_relation, tables: Mutex::new(HashMap::new()) }
    }

    /// The pinned-system gate for a statement of relation `rel`: `Err` unless the system file carries the relation's
    /// pinned digests (skipped under `--allow-any-system`, which the verdict then reports as `system_pinned = false`).
    fn gate(&self, rel: &Relation) -> Result<(), String> {
        if self.sys_id_hex == rel.sys_id && self.table_digest_hex == rel.table_digest {
            return Ok(());
        }
        Err(match self.pinned_relation {
            Some(other) => format!(
                "system file is the pinned {} system, the statement is a {} statement (sys_id {}..., table digest {}...)",
                other.name, rel.name, &self.sys_id_hex[..16], &self.table_digest_hex[..16]
            ),
            None => format!(
                "system file is not the pinned {} system (sys_id {}..., table digest {}...; pinned {}... / {}...); pass --allow-any-system to verify against it unpinned",
                rel.name, &self.sys_id_hex[..16], &self.table_digest_hex[..16], &rel.sys_id[..16], &rel.table_digest[..16]
            ),
        })
    }

    /// The tables of `(l, n)`, built on first use.  The lock is held while building so a second verification of the
    /// same shape waits for the one copy instead of building its own; the content is a function of `(l, n)` alone.
    fn tables(&self, l: usize, n: usize) -> Arc<Tables> {
        let mut map = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        map.entry((l, n)).or_insert_with(|| Arc::new(Tables::new(l, n))).clone()
    }
}

struct Ctx<'a> {
    st: &'a Statement,
    sys: &'a System,
    tb: &'a Tables,
    eta: Vec<u32>, // eta_c for the opened columns
    zh: Vec<u32>,  // Z_H(eta_c)
    xk: Vec<u32>,  // eta_c^k
    threads: usize,
}

impl<'a> Ctx<'a> {
    /// eval_on_coset(coefs)[cols] for D coefficient rows (row length `len`), in parallel over the rows.
    fn eval_rows_at_cols(&self, coefs: &[u32], len: usize, rows: usize, cols: &[u32]) -> Vec<Vec<u32>> {
        par_for(rows, self.threads, |d| {
            let ev = f::eval_on_coset(&coefs[d * len..(d + 1) * len], &self.tb.ntt_n, &self.tb.scale);
            cols.iter().map(|&c| ev[c as usize]).collect()
        })
    }
    /// encode(rows on H)[cols] for `rows` message rows of length l.
    fn encode_rows_at_cols(&self, msgs: &[u32], rows: usize, cols: &[u32]) -> Vec<Vec<u32>> {
        let l = self.st.l;
        par_for(rows, self.threads, |d| {
            let cw = f::encode(&msgs[d * l..(d + 1) * l], &self.tb.ntt_l, &self.tb.ntt_n, &self.tb.scale);
            cols.iter().map(|&c| cw[c as usize]).collect()
        })
    }
}

/// sum_i coef[i] * uo[i][c] over the rows `rows`, for every opened column.
fn combine(coef: &[u32], uo: &[u32], t: usize, rows: std::ops::Range<usize>) -> Vec<u32> {
    let mut out = vec![0u64; t];
    for i in rows {
        let k = coef[i];
        if k == 0 {
            continue;
        }
        let row = &uo[i * t..(i + 1) * t];
        for c in 0..t {
            out[c] = mul_acc(out[c], k, row[c]);
        }
    }
    out.into_iter().map(|x| x as u32).collect()
}

/// What the verifier read from the statement file (reported so two verifications can be compared field by field):
/// the configuration, the chain layout, and the BLAKE2b statement digest once it is computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatementInfo {
    pub l: usize,
    pub n: usize,
    pub d: usize,
    pub t: usize,
    pub zk: bool,
    pub interactive: bool,
    pub n_proofs: usize,
    pub n_vus: usize,
    pub steps: usize,
    pub k_ops: usize,
    /// The file format (2, 3, 4) and the relation the statement is checked as (the canonical name of relation.rs; a
    /// v2 / v3 file is `bf16-ampere`).
    pub format_version: u8,
    pub relation: String,
    pub digest: Option<String>,
}

impl StatementInfo {
    fn of(st: &Statement, rel: &Relation) -> StatementInfo {
        StatementInfo {
            l: st.l,
            n: st.n,
            d: st.d,
            t: st.t,
            zk: st.zk,
            interactive: st.interactive,
            n_proofs: st.n_proofs,
            n_vus: st.n_vus,
            steps: st.steps,
            k_ops: st.k_ops,
            format_version: st.version,
            relation: rel.name.to_string(),
            digest: None,
        }
    }
}

pub struct Verdict {
    pub accepted: bool,
    pub reason: String,
    pub timing: Timing,
    pub soundness: Option<Soundness>,
    /// `None` when the statement file did not parse (or named no relation this build pins).
    pub statement: Option<StatementInfo>,
    /// The system file was required to carry the relation's pinned digests (false under `--allow-any-system`: the
    /// verdict then says nothing about WHICH relation was proved).
    pub system_pinned: bool,
}

fn reject(reason: &str, timing: Timing, snd: Option<Soundness>, st: Option<StatementInfo>) -> Verdict {
    Verdict { accepted: false, reason: reason.to_string(), timing, soundness: snd, statement: st, system_pinned: false }
}

/// Verify one (system, statement, proof) triple on its own: the system is parsed and the tables built for this one
/// verification (`timing.shared`), inside `timing.total`.
pub fn verify(sys_bytes: &[u8], st_bytes: &[u8], pf_bytes: &[u8], opt: &Options) -> Verdict {
    let t_start = Instant::now();
    let shared = Shared::new(sys_bytes, opt);
    let t_shared = t_start.elapsed().as_secs_f64();
    let mut v = verify_inner(&shared, st_bytes, pf_bytes, opt);
    v.timing.shared = t_shared + v.timing.shared;
    v.timing.total = t_start.elapsed().as_secs_f64();
    v.system_pinned = !opt.allow_any_system;
    v
}

/// Verify one (statement, proof) pair against a `Shared` built once for the batch.  `timing.total` covers this
/// verification only; the checks, their order and their reasons are those of `verify`.
pub fn verify_with(shared: &Shared, st_bytes: &[u8], pf_bytes: &[u8], opt: &Options) -> Verdict {
    let t_start = Instant::now();
    let mut v = verify_inner(shared, st_bytes, pf_bytes, opt);
    v.timing.total = t_start.elapsed().as_secs_f64();
    v.system_pinned = !opt.allow_any_system;
    v
}

fn verify_inner(shared: &Shared, st_bytes: &[u8], pf_bytes: &[u8], opt: &Options) -> Verdict {
    let mut tm = Timing::default();
    let lap = |since: &mut Instant| -> f64 {
        let now = Instant::now();
        let d = now.duration_since(*since).as_secs_f64();
        *since = now;
        d
    };
    let mut clk = Instant::now();

    // ---- parse (statement-bounded) ----
    let sys = match &shared.sys {
        Ok(s) => s,
        Err(e) => return reject(e, tm, None, None),
    };
    let chain = match &sys.chain {
        Some(c) => c,
        None => return reject("system has no chain rows (unit-mode systems are not supported)", tm, None, None),
    };
    let st = match Statement::parse(st_bytes) {
        Ok(s) => s,
        Err(e) => return reject(&format!("statement file: {e}"), tm, None, None),
    };
    // ---- the relation: named by the statement (v4) or the Ampere BF16 unit (v2 / v3); its pinned system ----
    let rel = match relation::for_statement(st.relation.as_deref()) {
        Ok(r) => r,
        Err(e) => return reject(&format!("statement file: {e}"), tm, None, None),
    };
    let mut info = StatementInfo::of(&st, rel);
    if (st.word_bytes, st.y_bytes) != (rel.word_bytes, rel.y_bytes) {
        let r = format!(
            "statement file: word widths {}/{} bytes are not the {} relation's {}/{}",
            st.word_bytes, st.y_bytes, rel.name, rel.word_bytes, rel.y_bytes
        );
        return reject(&r, tm, None, Some(info));
    }
    if !opt.allow_any_system {
        if let Err(gate) = shared.gate(rel) {
            return reject(&gate, tm, None, Some(info));
        }
    }
    let m = sys.m;
    let pf = match Proof::parse(pf_bytes, &st, m) {
        Ok(p) => p,
        Err(e) => return reject(&format!("proof file: {e}"), tm, None, Some(info)),
    };
    tm.parse = lap(&mut clk);

    // ---- soundness of the parameters the statement fixes ----
    let snd = match soundness(&st) {
        Ok(s) => s,
        Err(e) => return reject(&e, tm, None, Some(info)),
    };
    // `verify`: the proof's own accounting, union-bounded over the n_proofs IT claims.  `batch` (soundness_bits = None)
    // gates on the union over the sub-batches actually presented instead (batch_bound, in main.rs after every verdict).
    if let Some(bits) = opt.soundness_bits {
        if snd.union_log2 > -bits {
            let r = format!(
                "parameters give 2^{:.2} over {} sub-batches (per proof 2^{:.2}); requested 2^-{}",
                snd.union_log2, st.n_proofs, snd.per_proof_log2, bits
            );
            return reject(&r, tm, Some(snd), Some(info));
        }
    }
    tm.soundness = lap(&mut clk);

    // ---- step 0: coin commitments c_i = BLAKE2b(tag || r_i || s_i) ----
    if let (true, Some(c)) = (st.interactive, &pf.coins) {
        let com = |r: &[u8; 32], s: &[u8; 32]| {
            let mut v = COIN_TAG.to_vec();
            v.extend_from_slice(r);
            v.extend_from_slice(s);
            blake2b_256(&v)
        };
        if com(&c.r1, &c.s1) != c.c1 {
            return reject("coin slot 1: opening does not match its commitment", tm, Some(snd), Some(info));
        }
        if com(&c.r2, &c.s2) != c.c2 {
            return reject("coin slot 2: opening does not match its commitment", tm, Some(snd), Some(info));
        }
        if let Some(own) = &opt.own_coins {
            let mut mine = Vec::with_capacity(128);
            for part in [&c.r1, &c.s1, &c.r2, &c.s2] {
                mine.extend_from_slice(part);
            }
            if mine[..] != own[..] {
                return reject("coins are not this verifier's step-0 coins", tm, Some(snd), Some(info));
            }
        }
    }

    // ---- statement: public decodes and digest ----
    let pins = match public_pins(&st, sys, rel) {
        Ok(p) => p,
        Err(e) => return reject(&format!("public inputs: {e}"), tm, Some(snd), Some(info)),
    };
    // the chain-end public words are in [0, 2^y_bits) (protocol.verify: `y16_pub >= 2 ** rel.y16_bits` rejects)
    if let Some(j) = st.y16.iter().position(|&y| y >> rel.y_bits != 0) {
        let r = format!("public inputs: chain-end word {j} = {} is not a {}-bit word", st.y16[j], rel.y_bits);
        return reject(&r, tm, Some(snd), Some(info));
    }
    let stmt = match statement_digest(&st, rel, &sys.sys_id) {
        Ok(d) => d,
        Err(e) => return reject(&e, tm, Some(snd), Some(info)),
    };
    info.digest = Some(crate::hash::hex(&stmt));
    tm.statement_hash = lap(&mut clk);

    // ---- column set: distinct, then the challenge ----
    {
        let mut seen = vec![false; st.n];
        for &c in &pf.cols {
            if seen[c as usize] {
                return reject("bad column set: repeated column index", tm, Some(snd), Some(info));
            }
            seen[c as usize] = true;
        }
    }
    let (l, n, d, t, k) = (st.l, st.n, st.d, st.t, st.k());
    let m_rows = st.committed_rows(m);
    let (nl, nq) = (sys.linear.len(), sys.quadratic.len());
    let zero = [0u8; 32];
    let (r1, r2) = match &pf.coins {
        Some(c) => (&c.r1, &c.r2),
        None => (&zero, &zero),
    };
    // challenge 1.  Interactive (8c, b-zk-fix F2): BLAKE3(tag_v2 || stmt || c1 c2 || r1) -- the verifier's coin alone,
    // NO root.  Fiat-Shamir: BLAKE3(tag || stmt || root).
    let mut seed1 = if st.interactive { CHALLENGE1_TAG_IA.to_vec() } else { CHALLENGE1_TAG_FS.to_vec() };
    seed1.extend_from_slice(&stmt);
    if st.interactive {
        seed1.extend_from_slice(&coin_prefix(&st, &pf));
        seed1.extend_from_slice(&coin_open(&st, r1));
    } else {
        seed1.extend_from_slice(&pf.root);
    }
    let seed1 = blake3(&seed1);
    let per_d = m_rows + nl + nq + 7 * l;
    let x = expand(&seed1, d * per_d);
    // challenge 2.  Interactive: BLAKE3(tag_v2 || stmt || c1 c2 || r2) -- none of root, w, h, q, v.
    // Fiat-Shamir: BLAKE3(tag || stmt || root || w || h || q [|| "|v=" v]).
    let mut seed2 = if st.interactive { CHALLENGE2_TAG_IA.to_vec() } else { CHALLENGE2_TAG_FS.to_vec() };
    seed2.extend_from_slice(&stmt);
    if st.interactive {
        seed2.extend_from_slice(&coin_prefix(&st, &pf));
        seed2.extend_from_slice(&coin_open(&st, r2));
    } else {
        seed2.extend_from_slice(&pf.root);
        seed2.extend_from_slice(&u32s_le(&pf.w));
        seed2.extend_from_slice(&u32s_le(&pf.h));
        seed2.extend_from_slice(&u32s_le(&pf.q));
        if st.zk {
            seed2.extend_from_slice(b"|v=");
            seed2.extend_from_slice(&u32s_le(&pf.v));
        }
    }
    let seed2 = blake3(&seed2);
    if column_challenge(&seed2, t, n) != pf.cols {
        return reject("column challenge mismatch", tm, Some(snd), Some(info));
    }
    tm.challenge_hash = lap(&mut clk);

    // ---- (iv) Merkle paths: leaf = BLAKE3(column as M u32 LE), node = BLAKE3(left || right) ----
    let depth = st.depth();
    let uo = &pf.opened;
    let bad = par_for(t, opt.threads, |j| {
        let mut leaf = Vec::with_capacity(4 * m_rows);
        for i in 0..m_rows {
            leaf.extend_from_slice(&uo[i * t + j].to_le_bytes());
        }
        let mut h = blake3(&leaf);
        let mut idx = pf.cols[j] as usize;
        for lvl in 0..depth {
            let sib = &pf.paths[j * depth + lvl];
            let mut buf = [0u8; 64];
            if idx % 2 == 0 {
                buf[..32].copy_from_slice(&h);
                buf[32..].copy_from_slice(sib);
            } else {
                buf[..32].copy_from_slice(sib);
                buf[32..].copy_from_slice(&h);
            }
            h = blake3(&buf);
            idx >>= 1;
        }
        h != pf.root
    });
    if let Some(j) = bad.iter().position(|b| *b) {
        return reject(&format!("merkle path {j} invalid"), tm, Some(snd), Some(info));
    }
    tm.merkle = lap(&mut clk);

    // ---- domain tables (shared across the batch: built once per (l, n)) ----
    let t_tables = Instant::now();
    let tables = shared.tables(l, n);
    let tables_secs = t_tables.elapsed().as_secs_f64(); // ~0 once built; booked under `shared`, not `ntt`
    tm.shared += tables_secs;
    let eta: Vec<u32> = pf.cols.iter().map(|&c| tables.pts[c as usize]).collect();
    let zh: Vec<u32> = eta.iter().map(|&e| sub(f::pow(e, l as u64), 1)).collect();
    let xk: Vec<u32> = eta.iter().map(|&e| f::pow(e, k as u64)).collect();
    let ctx = Ctx { st: &st, sys, tb: &tables, eta, zh, xk, threads: opt.threads };
    let _ = &ctx.eta;
    let _ = ctx.sys;
    // mask rows (PROTOCOL.md 8b, b-zk-fix): prox, lin, qa, qb, chain_a, chain_b -- D rows each after the m witness rows
    let row_prox = |dd: usize| m + dd;
    let row_lin = |dd: usize| m + d + dd;
    let row_qa = |dd: usize| m + 2 * d + dd;
    let row_qb = |dd: usize| m + 3 * d + dd;
    let row_chain_a = |dd: usize| m + 4 * d + dd;
    let row_chain_b = |dd: usize| m + 5 * d + dd;
    tm.ntt += lap(&mut clk) - tables_secs;

    // ---- (i) proximity ----
    let w_len = st.w_len();
    let w_at_cols = if st.zk {
        ctx.eval_rows_at_cols(&pf.w, w_len, d, &pf.cols)
    } else {
        ctx.encode_rows_at_cols(&pf.w, d, &pf.cols)
    };
    tm.ntt += lap(&mut clk);
    for dd in 0..d {
        let mut r = x[dd * per_d..dd * per_d + m_rows].to_vec();
        if st.zk {
            for j in 0..d {
                r[row_prox(j)] = 0; // the proximity mask row enters with coefficient 1
            }
        }
        let mut rhs = combine(&r, uo, t, 0..m_rows);
        if st.zk {
            for c in 0..t {
                rhs[c] = add(rhs[c], uo[row_prox(dd) * t + c]);
            }
        }
        if rhs != w_at_cols[dd] {
            tm.dfold += lap(&mut clk);
            return reject("proximity test failed", tm, Some(snd), Some(info));
        }
    }
    tm.dfold += lap(&mut clk);

    // ---- (ii) linear constraints: alpha = sum rho a_q, beta = sum rho (pub_q - c_q) ----
    let mut alpha = vec![0u32; d * m];
    let mut beta = vec![0u32; d * l];
    for dd in 0..d {
        let rho = &x[dd * per_d + m_rows..dd * per_d + m_rows + nl];
        let mut konst = 0u64;
        for (q, lc) in sys.linear.iter().enumerate() {
            let rq = rho[q];
            if rq == 0 {
                continue;
            }
            for (row, coef) in &lc.e.lin {
                let a = &mut alpha[dd * m + *row as usize];
                *a = mul_acc(*a as u64, rq, *coef) as u32;
            }
            konst = mul_acc(konst, rq, neg(lc.e.c));
            if let Some(pi) = lc.pub_pin {
                let pv = &pins[pi];
                for j in 0..l {
                    let b = &mut beta[dd * l + j];
                    *b = mul_acc(*b as u64, rq, pv[j]) as u32;
                }
            }
        }
        for j in 0..l {
            beta[dd * l + j] = add(beta[dd * l + j], konst as u32);
        }
    }
    tm.dfold += lap(&mut clk);
    if st.zk {
        // v|_H = beta (fold mod x^l - 1, NTT_l) and v(eta_c) = alpha . U[cols] + Z_H s_lin
        for dd in 0..d {
            let v = &pf.v[dd * k..(dd + 1) * k];
            // x^j = x^(j mod l) on H; with t_pad > l (FS-ZK at l = 256: k = l + 512) a coefficient wraps more than once
            let mut fold = v[..l].to_vec();
            for j in l..k {
                fold[j % l] = add(fold[j % l], v[j]);
            }
            ctx.tb.ntt_l.transform(&mut fold, false);
            if fold != beta[dd * l..(dd + 1) * l] {
                return reject("linear constraints failed (public values)", tm, Some(snd), Some(info));
            }
        }
        let v_at = ctx.eval_rows_at_cols(&pf.v, k, d, &pf.cols);
        tm.ntt += lap(&mut clk);
        for dd in 0..d {
            let mut expect = combine(&alpha[dd * m..(dd + 1) * m], uo, t, 0..m);
            for c in 0..t {
                expect[c] = add(expect[c], mul(ctx.zh[c], uo[row_lin(dd) * t + c]));
            }
            if expect != v_at[dd] {
                return reject("linear constraints failed", tm, Some(snd), Some(info));
            }
        }
    } else {
        let beta_at = ctx.encode_rows_at_cols(&beta, d, &pf.cols);
        tm.ntt += lap(&mut clk);
        for dd in 0..d {
            let expect = combine(&alpha[dd * m..(dd + 1) * m], uo, t, 0..m);
            if expect != beta_at[dd] {
                return reject("linear constraints failed", tm, Some(snd), Some(info));
            }
        }
    }
    tm.dfold += lap(&mut clk);

    // ---- (iii) quadratic constraints: sum rho (A B - C)(U[cols]) (+ Z_H (s_qa + x^k s_qb)) = h(eta) Z_H(eta) ----
    let h_at = ctx.eval_rows_at_cols(&pf.h, st.h_len(), d, &pf.cols);
    tm.ntt += lap(&mut clk);
    let rho_q: Vec<&[u32]> = (0..d).map(|dd| &x[dd * per_d + m_rows + nl..dd * per_d + m_rows + nl + nq]).collect();
    let lhs_cols = par_for(t, opt.threads, |c| {
        let mut acc = vec![0u64; d];
        for (q, [a, b, cc]) in sys.quadratic.iter().enumerate() {
            let av = eval_expr(a, uo, t, c);
            let bv = eval_expr(b, uo, t, c);
            let cv = eval_expr(cc, uo, t, c);
            let tv = sub(mul(av, bv), cv);
            if tv == 0 {
                continue;
            }
            for dd in 0..d {
                acc[dd] = mul_acc(acc[dd], rho_q[dd][q], tv);
            }
        }
        acc.into_iter().map(|v| v as u32).collect::<Vec<u32>>()
    });
    for dd in 0..d {
        for c in 0..t {
            let mut lhs = lhs_cols[c][dd];
            if st.zk {
                let mask = add(uo[row_qa(dd) * t + c], mul(ctx.xk[c], uo[row_qb(dd) * t + c]));
                lhs = add(lhs, mul(ctx.zh[c], mask));
            }
            let rhs = mul(h_at[dd][c], ctx.zh[c]);
            if lhs != rhs {
                tm.quadratic += lap(&mut clk);
                return reject("quadratic constraints failed", tm, Some(snd), Some(info));
            }
        }
    }
    tm.quadratic += lap(&mut clk);

    // ---- (v) chain: the general linear test (PROTOCOL.md 7b, chain.py) ----
    let steps = st.steps;
    let in_vu = |j: usize| j < steps * st.n_vus;
    let is_start = |j: usize| !in_vu(j) || j % steps == 0;
    let is_end = |j: usize| in_vu(j) && j % steps == steps - 1;
    let is_link = |j: usize| in_vu(j) && j % steps != steps - 1;
    let mut involved: Vec<u32> = Vec::new();
    let pos = |row: u32, inv: &mut Vec<u32>| -> usize {
        match inv.iter().position(|r| *r == row) {
            Some(p) => p,
            None => {
                inv.push(row);
                inv.len() - 1
            }
        }
    };
    for &cr in &chain.c_rows {
        pos(cr, &mut involved);
    }
    for e in chain.y.iter().chain(std::iter::once(&chain.y16)) {
        for (row, _) in &e.lin {
            pos(*row, &mut involved);
        }
    }
    let nr = involved.len();
    let mut coef = vec![0u32; d * nr * l]; // (D, R, l) evaluations on H
    let mut rhs = vec![0u32; d];
    let y16_c = chain.y16.c;
    for dd in 0..d {
        let rc = |fam: usize, j: usize| x[dd * per_d + m_rows + nl + nq + fam * l + j];
        let cf = |r: usize| dd * nr * l + r * l;
        for xx in 0..3 {
            let ci = involved.iter().position(|r| *r == chain.c_rows[xx]).unwrap();
            let yx = &chain.y[xx];
            let mut sum_link = 0u32;
            for j in 0..l {
                let rl = if is_link(j) { rc(xx, j) } else { 0 };
                let rs = if is_start(j) { rc(3 + xx, j) } else { 0 };
                if rl != 0 {
                    // link_j: + W[c][j+1] - sum kappa_i W[i][j] = Y.c
                    if j + 1 < l {
                        let a = &mut coef[cf(ci) + j + 1];
                        *a = add(*a, rl);
                    }
                    for (row, kappa) in &yx.lin {
                        let ri = involved.iter().position(|r| r == row).unwrap();
                        let a = &mut coef[cf(ri) + j];
                        *a = add(*a, mul(rl, neg(*kappa)));
                    }
                    sum_link = add(sum_link, rl);
                }
                if rs != 0 {
                    let a = &mut coef[cf(ci) + j];
                    *a = add(*a, rs);
                }
            }
            rhs[dd] = add(rhs[dd], mul(sum_link, yx.c));
        }
        for j in 0..l {
            if !is_end(j) {
                continue;
            }
            let re = rc(6, j);
            for (row, kappa) in &chain.y16.lin {
                let ri = involved.iter().position(|r| r == row).unwrap();
                let a = &mut coef[cf(ri) + j];
                *a = add(*a, mul(re, *kappa));
            }
            rhs[dd] = add(rhs[dd], mul(re, sub(st.y16[j] % P, y16_c)));
        }
    }
    // sum over H: l * sum_{l | j} q_j = r^T b
    let q_len = st.q_len();
    for dd in 0..d {
        let qd = &pf.q[dd * q_len..(dd + 1) * q_len];
        let mut s = 0u32;
        let mut j = 0;
        while j < q_len {
            s = add(s, qd[j]);
            j += l;
        }
        if mul(l as u32 % P, s) != rhs[dd] {
            tm.chain += lap(&mut clk);
            return reject("chain constraints failed (sum over H)", tm, Some(snd), Some(info));
        }
    }
    tm.chain += lap(&mut clk);
    let r_at = ctx.encode_rows_at_cols(&coef, d * nr, &pf.cols); // (D*R) x t
    let q_at = ctx.eval_rows_at_cols(&pf.q, q_len, d, &pf.cols);
    tm.ntt += lap(&mut clk);
    for dd in 0..d {
        let mut expect = vec![0u64; t];
        for (ri, &row) in involved.iter().enumerate() {
            let rr = &r_at[dd * nr + ri];
            for c in 0..t {
                expect[c] = mul_acc(expect[c], rr[c], uo[row as usize * t + c]);
            }
        }
        for c in 0..t {
            let mut e = expect[c] as u32;
            if st.zk {
                // q_d = Q_d + u_ca + x^k u_cb (8b after b-zk-fix F1): the mask is reconstructed from the two opened
                // chain mask rows, u_ca(eta_c) + eta_c^k u_cb(eta_c); the pre-fix Z_H s_chain vanished on H and leaked q|_H.
                e = add(e, uo[row_chain_a(dd) * t + c]);
                e = add(e, mul(ctx.xk[c], uo[row_chain_b(dd) * t + c]));
            }
            if e != q_at[dd][c] {
                tm.chain += lap(&mut clk);
                return reject("chain constraints failed (opened columns)", tm, Some(snd), Some(info));
            }
        }
    }
    tm.chain += lap(&mut clk);
    // authentication = included: every published word is a leaf at its claimed index under the statement's roots
    // (auth.rs).  Runs after the Ligero tests so a rejection here means "valid proof, unauthenticated words".
    let mut auth_note = String::new();
    if let Some(ab) = &st.auth {
        let mps = match &pf.auth {
            Some(m) => m,
            None => return reject("auth: transcript carries no multiproofs", tm, Some(snd), Some(info)),
        };
        // the leaf schema is u16: the parser admits an authentication block with 2-byte words only
        let (a16, b16, y16) = (Statement::words_u16(&st.a), Statement::words_u16(&st.b), Statement::words_u16(&st.y16));
        if let Err(e) = crate::auth::check(ab, mps, &a16, &b16, &y16, st.n_vus, st.steps, st.k_ops, opt.threads) {
            tm.auth += lap(&mut clk);
            return reject(&e, tm, Some(snd), Some(info));
        }
        tm.auth += lap(&mut clk);
        auth_note = format!(" (authentication: {} VUs, {} + {} + {} leaves under the a/b/y roots)", ab.vu_index.len(),
                            st.n_vus * st.steps * st.k_ops, st.n_vus * st.steps * st.k_ops, st.n_vus);
    }
    // an accept under --allow-any-system says the proof is valid for the system FILE, not that the file is the
    // relation's pinned system: say so in the reason as well as in `system_pinned`
    let pin_note = if opt.allow_any_system {
        format!(" [system NOT pinned: --allow-any-system; sys_id {}...]", &shared.sys_id_hex[..16])
    } else {
        String::new()
    };
    let reason = if !st.interactive {
        format!("accept{auth_note}{pin_note}")
    } else if opt.own_coins.is_some() {
        format!("accept (interactive: coins are this verifier's step-0 coins){auth_note}{pin_note}")
    } else {
        format!("accept (interactive: coins replayed from the transcript; sound only if they were the verifier's step-0 coins){auth_note}{pin_note}")
    };
    Verdict { accepted: true, reason, timing: tm, soundness: Some(snd), statement: Some(info), system_pinned: !opt.allow_any_system }
}
