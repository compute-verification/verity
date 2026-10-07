//! Uniform copies in Flock's lincheck: N copies of one 2^13-row template (the sm_120 BF16 k16 step, exported by
//! `arch_proto/export_template.py`) chained into one 2^k block, k = 13 + log2 N, n_log = 0.
//!
//! One block matrix `A_0 = I_N ⊗ T_A + Δ_A`, `B_0 = I_N ⊗ T_B + Δ_B` (Δ: each copy's accumulator from the previous copy's
//! outputs, its 672 operand bits from pseudo-random columns, its constant from the pin). The copies are the soundness
//! package's `Rows.stack` (`soundness/FlockSoundness/ExecRows.lean`) placed by `stackPlace` with `base u = u·2^13`: copy u's
//! column c at `base u + c`, its constant at `base u + nIn + nComp`, the constant's row reading the pin (`placement_stack`'s
//! `hconstA`/`hconstB`), the pin at 2^k − 1. Described five ways:
//!
//! - `flat`: one sparse 2^k matrix (upstream's default `SparseMatrixCircuit`), k ≤ `--flat-max`;
//! - `cflock-unit`: C-Flock's `BlockCircuit` fold today (one type folded once, ratio-filled over every slot, then Δ) with the
//!   type a 2^u unit holding 2^(u−13) flattened copies (u = 20 ≈ a K=2048 coordinate's 128 steps);
//! - `cflock-step`: the same code with the 2^13 step as the type (a layout change only);
//! - `tmpl-step` / `tmpl-unit`: a structured verifier that never builds a 2^k vector: after replaying the rounds it computes
//!   `comb_partial[i] = Σ_c [c mod 64 = i] eq(r, c >> 6) comb[c]` as (slot factor) × (one template fold) + Δ + pin.
//!
//! The first three run upstream's `lincheck::verify`; all five must return the same claim on upstream's proof.
//!
//! Build and run from a flock b684b12 checkout, with this file as `crates/flock-core/examples/uniform_copies.rs`:
//!
//!   cargo build --release -p flock-core --example uniform_copies
//!   RAYON_NUM_THREADS=1 target/release/examples/uniform_copies TEMPLATE kmin kmax flat_max unit_log [timing]
//!
//! TEMPLATE is `export_template.py`'s output. The default mode proves with upstream's `lincheck::prove` (both descriptions
//! must give byte-identical proofs), checks every verifier's claim against the prover's and that a tampered proof or Δ is
//! refused, and prints JSON lines. `timing` skips the prover (k = 26 does not fit beside it in 16 GB): each verifier on a
//! random well-shaped proof, three times, plus `cflock-unit-scattered`, the unit type under a random relabeling.
use std::{env, fs, time::Instant};

use flock_core::{
    challenger::{Challenger, FsChallenger},
    field::F128,
    lincheck::{
        LincheckCircuit, LincheckClaim, LincheckProof, QuirkyPoint, SkipPoint, SparseMatrixCircuit, build_eq_table,
        build_quirky_eq_table_from_weights, pack_z_lincheck, prove, verify,
    },
    r1cs::SparseBinaryMatrix,
    test_rng::Rng,
};
use rayon::prelude::*;

const SL: usize = 13;
const K_SKIP: usize = 6;

struct Tmpl {
    a: Vec<Vec<u32>>,
    b: Vec<Vec<u32>>,
    acc: Vec<u32>,
    ops: Vec<u32>,
    outs: Vec<u32>,
    /// The copy's constant column, `nIn + nComp` (`Rows.stack`'s order): the last used row.
    konst: u32,
}

fn load(path: &str) -> Tmpl {
    let bytes = fs::read(path).expect("template file");
    let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(w[0], 0x54504C31);
    let rows = w[1] as usize;
    assert_eq!(rows, 1 << SL);
    let mut p = 3;
    let (mut a, mut b) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let la = w[p] as usize;
        a.push(w[p + 1..p + 1 + la].to_vec());
        p += 1 + la;
        let lb = w[p] as usize;
        b.push(w[p + 1..p + 1 + lb].to_vec());
        p += 1 + lb;
    }
    let acc = w[p..p + 32].to_vec();
    let ops = w[p + 32..p + 32 + 672].to_vec();
    let outs = w[p + 704..p + 736].to_vec();
    assert_eq!(p + 736, w.len());
    Tmpl { a, b, acc, ops, outs, konst: w[2] - 1 }
}

/// The block: N = 2^(k−13) copies, the pin, and Δ split by whether an entry crosses a `tl`-slot boundary.
struct Block {
    k: usize,
    pin: usize,
    /// (row, col) entries of Δ_A, Δ_B over the whole block, and per entry whether it is a chain edge inside one unit of
    /// `per_unit` copies (folded into the unit type, not Δ, in the `-unit` forms).
    da: Vec<(u32, u32, bool)>,
    db: Vec<(u32, u32)>,
}

fn block(t: &Tmpl, k: usize, per_unit: usize, rng: &mut Rng) -> Block {
    let n = 1usize << (k - SL);
    let pin = (1usize << k) - 1;
    let mut da = Vec::with_capacity(n * 705);
    let mut db = Vec::with_capacity(n);
    for s in 0..n {
        let base = s << SL;
        let konst = base + t.konst as usize;
        da.push((konst as u32, pin as u32, false));
        db.push((konst as u32, pin as u32));
        let prev = (s + n - 1) % n;
        let inside = s % per_unit != 0;
        for j in 0..32 {
            da.push(((base + t.acc[j] as usize) as u32, ((prev << SL) + t.outs[j] as usize) as u32, inside));
        }
        for &o in &t.ops {
            da.push(((base + o as usize) as u32, rng.below(1 << k) as u32, false));
        }
    }
    Block { k, pin, da, db }
}

/// `per` flattened copies of T as one sparse matrix of 2^(13 + log per) rows, with the chain edges inside it.
fn unit_matrices(t: &Tmpl, per: usize) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let size = per << SL;
    let mut a: Vec<Vec<usize>> = Vec::with_capacity(size);
    let mut b: Vec<Vec<usize>> = Vec::with_capacity(size);
    for s in 0..per {
        let base = s << SL;
        for x in 0..1 << SL {
            a.push(t.a[x].iter().map(|&c| base + c as usize).collect());
            b.push(t.b[x].iter().map(|&c| base + c as usize).collect());
        }
        if s > 0 {
            for j in 0..32 {
                a[base + t.acc[j] as usize].push(((s - 1) << SL) + t.outs[j] as usize);
            }
        }
    }
    (SparseBinaryMatrix::new(size, size, a), SparseBinaryMatrix::new(size, size, b))
}

/// `m` with rows and columns relabeled by `p`: the same nnz and fold work, with long-range accesses (an upper bound for a real
/// unit's wiring, whose layers reach across the unit).
fn scattered(m: &SparseBinaryMatrix, p: &[usize]) -> SparseBinaryMatrix {
    let mut rows = vec![Vec::new(); p.len()];
    for (r, row) in m.rows.iter().enumerate() {
        rows[p[r]] = row.iter().map(|&c| p[c]).collect();
    }
    SparseBinaryMatrix::new(p.len(), p.len(), rows)
}

/// The whole block as one flat matrix (template copies + every Δ entry).
fn flat_matrices(t: &Tmpl, blk: &Block) -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let n = 1usize << blk.k;
    let mut a: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut b: Vec<Vec<usize>> = Vec::with_capacity(n);
    for s in 0..n >> SL {
        let base = s << SL;
        for x in 0..1 << SL {
            a.push(t.a[x].iter().map(|&c| base + c as usize).collect());
            b.push(t.b[x].iter().map(|&c| base + c as usize).collect());
        }
    }
    for &(r, c, _) in &blk.da {
        a[r as usize].push(c as usize);
    }
    for &(r, c) in &blk.db {
        b[r as usize].push(c as usize);
    }
    (SparseBinaryMatrix::new(n, n, a), SparseBinaryMatrix::new(n, n, b))
}

/// C-Flock's `BlockCircuit::fold_alpha_batched` (verity/ml/flock/live/src/circuit.rs) with one slot type covering the block.
struct BlockLike<'a> {
    k: usize,
    tl: usize,
    ty: SparseMatrixCircuit<'a>,
    da: Vec<(u32, u32)>,
    db: &'a [(u32, u32)],
    pin: usize,
}

impl LincheckCircuit for BlockLike<'_> {
    fn n_cols(&self) -> usize {
        1 << self.k
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        let lo = 1usize << self.tl;
        let c0 = eq[..lo].iter().position(|e| *e != F128::ZERO).unwrap();
        let inv = eq[c0].inv();
        let base = self.ty.fold_alpha_batched(alpha, &eq[..lo]);
        let mut out = vec![F128::ZERO; 1 << self.k];
        out.par_chunks_mut(lo).enumerate().for_each(|(s, o)| {
            let ratio = eq[(s << self.tl) | c0] * inv;
            for (x, y) in o.iter_mut().zip(&base) {
                *x = ratio * *y;
            }
        });
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq[i as usize];
        }
        for &(i, c) in self.db {
            out[c as usize] += eq[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.pin)
    }
}

/// The structured verifier: upstream `lincheck::verify`'s transcript, step for step, with `comb_partial` computed from the
/// structure after the rounds. `tl` is the type's log size (13: the step; u: a flattened unit); `ty` is its matrices.
#[allow(clippy::too_many_arguments)]
fn verify_structured(
    k: usize,
    tl: usize,
    ty: (&SparseBinaryMatrix, &SparseBinaryMatrix),
    da: &[(u32, u32)],
    db: &[(u32, u32)],
    pin: usize,
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    ch: &mut FsChallenger,
) -> Result<LincheckClaim, &'static str> {
    if proof.rounds.len() != k - K_SKIP || proof.z_partial.len() != 1 << K_SKIP || !proof.grinding_nonces.is_empty() {
        return Err("shape");
    }
    ch.observe_label(b"flock-lincheck-v0");
    let alpha = ch.sample_f128();
    let beta = ch.sample_f128();
    let mut running = alpha * v_a + v_b + beta;
    let mut r_rounds = Vec::with_capacity(k - K_SKIP);
    for &(e1, einf) in &proof.rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let r = ch.sample_f128();
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        r_rounds.push(r);
    }
    ch.observe_f128_slice(&proof.z_partial);
    let mut rr = r_rounds;
    rr.reverse();

    let ws = x_ab.z_skip.weights(K_SKIP);
    let xr = &x_ab.x_inner_rest;
    let lo = tl - K_SKIP;
    let ex_lo = build_eq_table(&xr[..lo]);
    let ex_hi = build_eq_table(&xr[lo..]);
    let er_lo = build_eq_table(&rr[..lo]);
    let er_hi = build_eq_table(&rr[lo..]);
    let slot = xr[lo..].iter().zip(&rr[lo..]).fold(F128::ONE, |a, (x, r)| a * (F128::ONE + *x + *r));
    let eq_t = build_quirky_eq_table_from_weights(&ws, &xr[..lo]);
    let base = SparseMatrixCircuit::new(ty.0, ty.1).fold_alpha_batched(alpha, &eq_t);
    let mut cp = vec![F128::ZERO; 1 << K_SKIP];
    for (y, v) in base.iter().enumerate() {
        cp[y & 63] += *v * er_lo[y >> K_SKIP];
    }
    for v in &mut cp {
        *v *= slot;
    }
    let mask = (1usize << tl) - 1;
    let eq_row = |r: usize| ws[r & 63] * ex_lo[(r & mask) >> K_SKIP] * ex_hi[r >> tl];
    let eq_col = |c: usize| er_lo[(c & mask) >> K_SKIP] * er_hi[c >> tl];
    for &(r, c) in da {
        cp[c as usize & 63] += alpha * eq_row(r as usize) * eq_col(c as usize);
    }
    for &(r, c) in db {
        cp[c as usize & 63] += eq_row(r as usize) * eq_col(c as usize);
    }
    cp[pin & 63] += beta * eq_col(pin);

    let final_sum = cp.iter().zip(&proof.z_partial).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    if final_sum != running {
        return Err("sumcheck-final");
    }
    let r_inner_skip = x_ab.z_skip.sample_fresh(ch);
    let lambda = r_inner_skip.weights(K_SKIP);
    let w = lambda.iter().zip(&proof.z_partial).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    Ok(LincheckClaim { r_inner_skip, r_inner_rest: rr, w })
}

/// `a = A_0 z` over the block from its structure.
fn apply(t_rows: &[Vec<u32>], k: usize, delta: &[(u32, u32)], z: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 1 << k];
    out.par_chunks_mut(1 << SL).enumerate().for_each(|(s, o)| {
        let base = s << SL;
        for (x, row) in t_rows.iter().enumerate() {
            o[x] = row.iter().fold(false, |acc, &c| acc ^ z[base + c as usize]);
        }
    });
    for &(r, c) in delta {
        out[r as usize] ^= z[c as usize];
    }
    out
}

/// One block's inner quirky MLE `Σ_x λ(z_skip)[x mod 64] eq(x_inner_rest, x >> 6) f[x]`.
fn quirky_eval(f: &[bool], x: &QuirkyPoint) -> F128 {
    let eq = build_quirky_eq_table_from_weights(&x.z_skip.weights(K_SKIP), &x.x_inner_rest);
    eq.par_iter().zip(f.par_iter()).filter(|(_, b)| **b).map(|(e, _)| *e).reduce(|| F128::ZERO, |a, b| a + b)
}

/// Blocks per statement: `pack_z_lincheck` needs n_outer ≥ 8.
const N_LOG: usize = 3;

/// Each verifier's time on a well-shaped random proof, `REPS` times (min and median): every verifier does all its work
/// before the final consistency check, so an invalid proof times the same as an honest one, without the prover's memory.
#[allow(clippy::too_many_arguments)]
fn timing_only(t: &Tmpl, ta: &SparseBinaryMatrix, tb: &SparseBinaryMatrix, t_nnz: usize, kmin: usize, kmax: usize,
               unit_log: usize, threads: usize) {
    const REPS: usize = 3;
    for k in kmin..=kmax {
        let ul = unit_log.min(k);
        let per_unit = 1usize << (ul - SL);
        let mut rng = Rng::new(20260930 + k as u64);
        let blk = block(t, k, per_unit, &mut rng);
        let da_all: Vec<(u32, u32)> = blk.da.iter().map(|&(r, c, _)| (r, c)).collect();
        let da_unit: Vec<(u32, u32)> = blk.da.iter().filter(|e| !e.2).map(|&(r, c, _)| (r, c)).collect();
        let (ua, ub) = unit_matrices(t, per_unit);
        let unit_nnz: usize = ua.rows.iter().chain(ub.rows.iter()).map(|r| r.len()).sum();
        let m = k + N_LOG;
        let x_ab = QuirkyPoint { z_skip: SkipPoint::Phi8(rng.f128()), x_inner_rest: rng.f128_vec(k - K_SKIP),
                                 x_outer: rng.f128_vec(N_LOG) };
        let proof = LincheckProof { rounds: (0..k - K_SKIP).map(|_| (rng.f128(), rng.f128())).collect(),
                                    z_partial: rng.f128_vec(1 << K_SKIP), matrix_evals: vec![], grinding_nonces: vec![],
                                    comb_partial: vec![] };
        let (v_a, v_b) = (rng.f128(), rng.f128());
        let dom = b"verity-uniform-copies-v0";
        let step = BlockLike { k, tl: SL, ty: SparseMatrixCircuit::new(ta, tb), da: da_all.clone(), db: &blk.db, pin: blk.pin };
        let unit = BlockLike { k, tl: ul, ty: SparseMatrixCircuit::new(&ua, &ub), da: da_unit.clone(), db: &blk.db, pin: blk.pin };
        let perm = rng.permutation(1 << ul);
        let (sa, sb) = (scattered(&ua, &perm), scattered(&ub, &perm));
        let unit_sc = BlockLike { k, tl: ul, ty: SparseMatrixCircuit::new(&sa, &sb), da: da_unit.clone(), db: &blk.db, pin: blk.pin };
        let stats = |f: &dyn Fn() -> f64| -> serde_json::Value {
            let mut v: Vec<f64> = (0..REPS).map(|_| f()).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            serde_json::json!({"min": v[0], "median": v[REPS / 2]})
        };
        let up = |c: &dyn LincheckCircuit| timed(|| verify(m, k, K_SKIP, c, &x_ab, v_a, v_b, &proof, &mut FsChallenger::new(dom))).1;
        let st = |tl: usize, ty: (&SparseBinaryMatrix, &SparseBinaryMatrix), da: &[(u32, u32)]| {
            timed(|| verify_structured(k, tl, ty, da, &blk.db, blk.pin, &x_ab, v_a, v_b, &proof, &mut FsChallenger::new(dom))).1
        };
        let rec = serde_json::json!({
            "mode": "timing", "k_log": k, "m": m, "copies": 1usize << (k - SL), "threads": threads, "template_nnz": t_nnz,
            "unit_log": ul, "unit_copies": per_unit, "unit_nnz": unit_nnz, "delta_step": da_all.len() + blk.db.len(),
            "delta_unit": da_unit.len() + blk.db.len(),
            "verify_cflock_unit_s": stats(&|| up(&unit)), "verify_cflock_unit_scattered_s": stats(&|| up(&unit_sc)),
            "verify_cflock_step_s": stats(&|| up(&step)),
            "verify_tmpl_step_s": stats(&|| st(SL, (ta, tb), &da_all)), "verify_tmpl_unit_s": stats(&|| st(ul, (&ua, &ub), &da_unit)),
        });
        println!("{rec}");
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("usage: uniform_copies TEMPLATE [kmin kmax flat_max unit_log]");
    let kmin: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(13);
    let kmax: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(22);
    let flat_max: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(21);
    let unit_log: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(20);
    let t = load(path);
    let t_nnz: usize = t.a.iter().chain(&t.b).map(|r| r.len()).sum();
    let (ta, tb) = {
        let conv = |m: &[Vec<u32>]| m.iter().map(|r| r.iter().map(|&c| c as usize).collect()).collect::<Vec<Vec<usize>>>();
        (SparseBinaryMatrix::new(1 << SL, 1 << SL, conv(&t.a)), SparseBinaryMatrix::new(1 << SL, 1 << SL, conv(&t.b)))
    };
    let threads = rayon::current_num_threads();
    if args.get(6).map(String::as_str) == Some("timing") {
        timing_only(&t, &ta, &tb, t_nnz, kmin, kmax, unit_log, threads);
        return;
    }
    for k in kmin..=kmax {
        let n = 1usize << (k - SL);
        let ul = unit_log.min(k);
        let per_unit = 1usize << (ul - SL);
        let mut rng = Rng::new(20260930 + k as u64);
        let blk = block(&t, k, per_unit, &mut rng);
        let da_all: Vec<(u32, u32)> = blk.da.iter().map(|&(r, c, _)| (r, c)).collect();
        let da_unit: Vec<(u32, u32)> = blk.da.iter().filter(|e| !e.2).map(|&(r, c, _)| (r, c)).collect();
        let (ua, ub) = unit_matrices(&t, per_unit);
        let unit_nnz: usize = ua.rows.iter().chain(ub.rows.iter()).map(|r| r.len()).sum();

        let m = k + N_LOG;
        let mut z = rng.bits(1 << m);
        for i in 0..1usize << N_LOG {
            z[(i << k) + blk.pin] = true;
        }
        let x_ab = QuirkyPoint { z_skip: SkipPoint::Phi8(rng.f128()), x_inner_rest: rng.f128_vec(k - K_SKIP),
                                 x_outer: rng.f128_vec(N_LOG) };
        let eo = build_eq_table(&x_ab.x_outer);
        let mle_of = |rows: &[Vec<u32>], d: &[(u32, u32)]| -> F128 {
            z.chunks(1 << k).zip(&eo).fold(F128::ZERO, |acc, (zb, o)| acc + *o * quirky_eval(&apply(rows, k, d, zb), &x_ab))
        };
        let (v_a, v_b) = (mle_of(&t.a, &da_all), mle_of(&t.b, &blk.db));
        let z_packed = pack_z_lincheck(&z, m, k);
        drop(z);

        let step = BlockLike { k, tl: SL, ty: SparseMatrixCircuit::new(&ta, &tb), da: da_all.clone(), db: &blk.db, pin: blk.pin };
        let unit = BlockLike { k, tl: ul, ty: SparseMatrixCircuit::new(&ua, &ub), da: da_unit.clone(), db: &blk.db, pin: blk.pin };
        let dom = b"verity-uniform-copies-v0";
        let ((proof, claim_p), prove_s) = timed(|| prove(&z_packed, m, k, K_SKIP, &step, &x_ab, &mut FsChallenger::new(dom)));
        let prove_unit_s = (k <= 23).then(|| {
            let ((proof_u, _), s) = timed(|| prove(&z_packed, m, k, K_SKIP, &unit, &x_ab, &mut FsChallenger::new(dom)));
            assert_eq!(proof, proof_u, "proofs differ between the step and unit descriptions at k={k}");
            s
        });

        let mut rec = serde_json::json!({
            "k_log": k, "m": m, "copies": n, "threads": threads, "template_nnz": t_nnz, "unit_log": ul, "unit_copies": per_unit,
            "unit_nnz": unit_nnz, "delta_step": da_all.len() + blk.db.len(), "delta_unit": da_unit.len() + blk.db.len(),
            "flat_nnz": n * t_nnz + da_all.len() + blk.db.len(),
            "proof_bytes": proof.rounds.len() * 32 + proof.z_partial.len() * 16 + proof.grinding_nonces.len() * 8,
            "prove_s": prove_s, "prove_unit_s": prove_unit_s,
        });

        let up = |c: &dyn LincheckCircuit| {
            let (r, s) = timed(|| verify(m, k, K_SKIP, c, &x_ab, v_a, v_b, &proof, &mut FsChallenger::new(dom)));
            assert_eq!(r.expect("upstream verify rejected the honest proof"), claim_p);
            s
        };
        rec["verify_cflock_unit_s"] = up(&unit).into();
        rec["verify_cflock_step_s"] = up(&step).into();
        if k <= flat_max {
            let (fa, fb) = flat_matrices(&t, &blk);
            let flat = SparseMatrixCircuit::new(&fa, &fb).with_const_pin(Some(blk.pin));
            let ((pf, _), _) = timed(|| prove(&z_packed, m, k, K_SKIP, &flat, &x_ab, &mut FsChallenger::new(dom)));
            assert_eq!(pf, proof, "flat proof differs at k={k}");
            rec["verify_flat_s"] = up(&flat).into();
        }
        let st = |tl: usize, ty: (&SparseBinaryMatrix, &SparseBinaryMatrix), da: &[(u32, u32)], pf: &LincheckProof| {
            timed(|| verify_structured(k, tl, ty, da, &blk.db, blk.pin, &x_ab, v_a, v_b, pf, &mut FsChallenger::new(dom)))
        };
        let (r, s) = st(SL, (&ta, &tb), &da_all, &proof);
        assert_eq!(r.expect("structured verifier rejected the honest proof"), claim_p);
        rec["verify_tmpl_step_s"] = s.into();
        let (r, s) = st(ul, (&ua, &ub), &da_unit, &proof);
        assert_eq!(r.expect("structured verifier (unit) rejected the honest proof"), claim_p);
        rec["verify_tmpl_unit_s"] = s.into();

        let mut bad = proof.clone();
        bad.z_partial[3] += F128::ONE;
        let rej_up = verify(m, k, K_SKIP, &step, &x_ab, v_a, v_b, &bad, &mut FsChallenger::new(dom)).is_err();
        let rej_st = st(SL, (&ta, &tb), &da_all, &bad).0.is_err();
        let mut bad_v = proof.clone();
        bad_v.rounds[0].0 += F128::ONE;
        let rej_round = st(SL, (&ta, &tb), &da_all, &bad_v).0.is_err();
        let mut da_bad = da_all.clone();
        da_bad[7].1 ^= 1;
        let rej_struct = st(SL, (&ta, &tb), &da_bad, &proof).0.is_err();
        assert!(rej_up && rej_st && rej_round && rej_struct, "a tampered proof or structure was accepted at k={k}");
        rec["tamper_rejected"] = true.into();

        // the verifier's phases on today's form: the 2^k eq table, the type fold + ratio fill + Δ, the k−6 binds
        let (eq, eq_s) = timed(|| build_quirky_eq_table_from_weights(&x_ab.z_skip.weights(K_SKIP), &x_ab.x_inner_rest));
        let (mut comb, fold_s) = timed(|| unit.fold_alpha_batched(F128::ONE, &eq));
        let ((), base_s) = timed(|| {
            let _ = unit.ty.fold_alpha_batched(F128::ONE, &eq[..1 << ul]);
        });
        let mut rch = FsChallenger::new(b"bind");
        let ((), bind_s) = timed(|| {
            while comb.len() > 1 << K_SKIP {
                let r = rch.sample_f128();
                let half = comb.len() / 2;
                let (lo, hi) = comb.split_at_mut(half);
                lo.par_iter_mut().zip(hi.par_iter()).for_each(|(l, h)| *l = *l + r * (*h + *l));
                comb.truncate(half);
            }
        });
        rec["phases_cflock_unit"] = serde_json::json!({"eq_table_s": eq_s, "fold_s": fold_s, "of_which_type_fold_s": base_s,
                                                        "binds_s": bind_s});
        println!("{rec}");
    }
}
