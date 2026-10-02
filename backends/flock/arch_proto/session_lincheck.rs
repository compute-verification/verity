//! The session verifier's lincheck (`flock_live::session_verify::lincheck`) against upstream's on `uniform_copies.rs`'s block,
//! k = kmin..kmax: N = 2^(k−13) copies of the 2^13-row sm_120 BF16 step template chained into one 2^k block, n_log = 3.
//!
//! The block is described as C-Flock's `BlockCircuit` three ways, all the same matrices:
//! - `unit`: one slot type, a 2^u unit of 2^(u−13) flattened copies with their chain edges, over the whole block;
//! - `step`: one slot type, the 2^13 step, every chain edge in Δ;
//! - `mixed`: the unit type over all units but the last, the step type over the last unit in two ranges split at an odd
//!   position (several types, ranges that are not dyadic blocks).
//!
//! Upstream's `lincheck::prove` proves it once. Upstream's `lincheck::verify` and `session_verify::lincheck` in each mode
//! (`Flat`, `Partial`, `Both`) verify it under every description: each must return the prover's claim and leave the transcript
//! where upstream's leaves it (the next challenge equal), and reject the proof with `z_partial` or a round altered, and the
//! honest proof against an altered Δ. Prints one JSON line per k, with each verifier's time.
//!
//! Build in a flock b684b12 checkout with the pods' patches (60-circuit.sh's, flock-vtime included) and `crates/flock-live`,
//! as the bin of a crate depending on flock-core, flock-live (features sha512), flock-field, rayon and serde_json:
//!
//!   RAYON_NUM_THREADS=4 session_lincheck TEMPLATE kmin kmax unit_log
//!
//! TEMPLATE is `export_template.py`'s output.
use std::{env, fs, time::Instant};

use flock_core::{
    challenger::{Challenger, FsChallenger},
    lincheck::{
        CscCircuit, LincheckGrinding, QuirkyPoint, SkipPoint, build_quirky_eq_table_from_weights,
        pack_z_lincheck, prove, verify,
    },
    r1cs::SparseBinaryMatrix,
    test_rng::Rng,
};
use flock_field::F128;
use flock_live::{
    circuit::BlockCircuit,
    session_verify::{Lincheck, Phases, lincheck},
};
use rayon::prelude::*;
use serde_json::json;

const SL: usize = 13;
const K_SKIP: usize = 6;
const N_LOG: usize = 3;

struct Tmpl {
    a: Vec<Vec<u32>>,
    b: Vec<Vec<u32>>,
    acc: Vec<u32>,
    ops: Vec<u32>,
    outs: Vec<u32>,
    konst: u32,
}

fn load(path: &str) -> Tmpl {
    let bytes = fs::read(path).expect("template file");
    let w: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!((w[0], w[1] as usize), (0x54504C31, 1 << SL));
    let mut p = 3;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..1 << SL {
        let la = w[p] as usize;
        a.push(w[p + 1..p + 1 + la].to_vec());
        p += 1 + la;
        let lb = w[p] as usize;
        b.push(w[p + 1..p + 1 + lb].to_vec());
        p += 1 + lb;
    }
    assert_eq!(p + 736, w.len());
    Tmpl { a, b, acc: w[p..p + 32].to_vec(), ops: w[p + 32..p + 704].to_vec(), outs: w[p + 704..p + 736].to_vec(), konst: w[2] - 1 }
}

/// `uniform_copies.rs`'s Δ: per copy its constant from the pin, its accumulator from the previous copy's outputs (with that
/// copy's index), its operands from pseudo-random columns.
fn block(t: &Tmpl, k: usize, rng: &mut Rng) -> (usize, Vec<(u32, u32, Option<usize>)>, Vec<(u32, u32)>) {
    let n = 1usize << (k - SL);
    let pin = (1usize << k) - 1;
    let (mut da, mut db) = (Vec::new(), Vec::new());
    for s in 0..n {
        let base = s << SL;
        let konst = base + t.konst as usize;
        da.push((konst as u32, pin as u32, None));
        db.push((konst as u32, pin as u32));
        let prev = (s + n - 1) % n;
        for j in 0..32 {
            da.push(((base + t.acc[j] as usize) as u32, ((prev << SL) + t.outs[j] as usize) as u32, Some(s)));
        }
        for &o in &t.ops {
            da.push(((base + o as usize) as u32, rng.below(1 << k) as u32, None));
        }
    }
    (pin, da, db)
}

/// `per` flattened copies of T with the chain edges between them, column-major as C-Flock's slot types are.
fn unit_matrices(t: &Tmpl, per: usize) -> CscCircuit {
    let size = per << SL;
    let (mut a, mut b): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (Vec::with_capacity(size), Vec::with_capacity(size));
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
    CscCircuit::from_matrices(&SparseBinaryMatrix::new(size, size, a), &SparseBinaryMatrix::new(size, size, b))
}

fn apply(t_rows: &[Vec<u32>], k: usize, delta: &[(u32, u32)], z: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 1 << k];
    out.par_chunks_mut(1 << SL).enumerate().for_each(|(s, o)| {
        for (x, row) in t_rows.iter().enumerate() {
            o[x] = row.iter().fold(false, |acc, &c| acc ^ z[(s << SL) + c as usize]);
        }
    });
    for &(r, c) in delta {
        out[r as usize] ^= z[c as usize];
    }
    out
}

fn quirky_eval(f: &[bool], x: &QuirkyPoint) -> F128 {
    let eq = build_quirky_eq_table_from_weights(&x.z_skip.weights(K_SKIP), &x.x_inner_rest);
    eq.par_iter().zip(f.par_iter()).filter(|(_, b)| **b).map(|(e, _)| *e).reduce(|| F128::ZERO, |a, b| a + b)
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let v = f();
    (v, t.elapsed().as_secs_f64())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let t = load(args.get(1).expect("usage: session_lincheck TEMPLATE kmin kmax unit_log"));
    let num = |i: usize, d: usize| args.get(i).map(|s| s.parse().unwrap()).unwrap_or(d);
    let (kmin, kmax, unit_log) = (num(2, 13), num(3, 22), num(4, 20));
    let step_ty = unit_matrices(&t, 1);
    for k in kmin..=kmax {
        let ul = unit_log.min(k);
        let per = 1usize << (ul - SL);
        let units = 1usize << (k - ul);
        let mut rng = Rng::new(20260930 + k as u64);
        let (pin, da3, db) = block(&t, k, &mut rng);
        let all: Vec<(u32, u32)> = da3.iter().map(|&(r, c, _)| (r, c)).collect();
        let outside = |inside: &dyn Fn(usize) -> bool| -> Vec<(u32, u32)> {
            da3.iter().filter(|e| !e.2.is_some_and(inside)).map(|&(r, c, _)| (r, c)).collect()
        };
        let unit_ty = unit_matrices(&t, per);
        let bc = |types: Vec<(usize, CscCircuit)>, ranges: Vec<(usize, usize, usize)>, da: Vec<(u32, u32)>| {
            let tables = vec![None; types.len()];
            BlockCircuit { k_log: k, pin, types, tables, ranges, da, db: db.clone() }
        };
        let step = bc(vec![(SL, step_ty.clone())], vec![(0, 0, 1 << (k - SL))], all.clone());
        let unit = bc(vec![(ul, unit_ty.clone())], vec![(0, 0, units)], outside(&|s| s % per != 0));
        // the last unit's copies as steps, in two ranges split at an odd copy
        let (first, last) = ((units - 1) * per, units * per);
        let split = (first + per / 2 + 1).min(last);
        let mixed = (units >= 2).then(|| bc(vec![(ul, unit_ty.clone()), (SL, step_ty.clone())],
                                            vec![(0, 0, units - 1), (1, first, split - first), (1, split, last - split)],
                                            outside(&|s| s % per != 0 && s < first)));

        let m = k + N_LOG;
        let mut z = rng.bits(1 << m);
        for i in 0..1usize << N_LOG {
            z[(i << k) + pin] = true;
        }
        let x_ab = QuirkyPoint { z_skip: SkipPoint::Phi8(rng.f128()), x_inner_rest: rng.f128_vec(k - K_SKIP), x_outer: rng.f128_vec(N_LOG) };
        let eo = flock_core::lincheck::build_eq_table(&x_ab.x_outer);
        let mle = |rows: &[Vec<u32>], d: &[(u32, u32)]| -> F128 {
            z.chunks(1 << k).zip(&eo).fold(F128::ZERO, |acc, (zb, o)| acc + *o * quirky_eval(&apply(rows, k, d, zb), &x_ab))
        };
        let (v_a, v_b) = (mle(&t.a, &all), mle(&t.b, &db));
        let z_packed = pack_z_lincheck(&z, m, k);
        drop(z);
        let dom = b"verity-session-lincheck-v0";
        let ((proof, claim), prove_s) = timed(|| prove(&z_packed, m, k, K_SKIP, &step, &x_ab, &mut FsChallenger::new(dom)));
        drop(z_packed);

        let mut descs: Vec<(&str, &BlockCircuit)> = vec![("unit", &unit), ("step", &step)];
        if let Some(mx) = &mixed {
            descs.push(("mixed", mx));
        }
        let mut bad_z = proof.clone();
        bad_z.z_partial[3] += F128::ONE;
        let mut bad_r = proof.clone();
        bad_r.rounds[0].0 += F128::ONE;
        let mut rec = json!({"k_log": k, "m": m, "copies": 1usize << (k - SL), "unit_log": ul, "units": units, "prove_s": prove_s,
                             "threads": rayon::current_num_threads()});
        let mut ok = true;
        for (name, c) in &descs {
            let mut ch = FsChallenger::new(dom);
            let (up, up_s) = timed(|| verify(m, k, K_SKIP, *c, &x_ab, v_a, v_b, &proof, &mut ch));
            let next = ch.sample_f128();
            ok &= up.as_ref() == Ok(&claim);
            rec[format!("upstream_{name}_s")] = json!(up_s);
            for mode in [Lincheck::Flat, Lincheck::Partial, Lincheck::Both] {
                let run = |p: &flock_core::lincheck::LincheckProof, c: &BlockCircuit| {
                    let (mut ch, mut ph) = (FsChallenger::new(dom), Phases::default());
                    let (r, s) = timed(|| lincheck(m, k, K_SKIP, c, &x_ab, v_a, v_b, p, LincheckGrinding::disabled(), mode, &mut ch, &mut ph));
                    (r, ch.sample_f128(), s, ph)
                };
                let (r, nx, s, ph) = run(&proof, c);
                let same = r.as_ref() == Ok(&claim) && nx == next;
                let rejects = run(&bad_z, c).0.is_err() && run(&bad_r, c).0.is_err();
                let mut wrong = BlockCircuit { k_log: k, pin, types: c.types.clone(),
                                               tables: c.tables.clone(), ranges: c.ranges.clone(), da: c.da.clone(), db: c.db.clone() };
                wrong.da[7].1 ^= 1;
                let rejects_delta = run(&proof, &wrong).0.is_err();
                ok &= same && rejects && rejects_delta;
                rec[format!("{name}_{mode:?}")] = json!({"same_claim_and_transcript": same, "tampers_rejected": rejects && rejects_delta,
                                                          "s": s, "comb_mode": ph.comb_mode,
                                                          "comb": ph.comb.iter().map(|(p, v)| (p.to_string(), json!(v))).collect::<serde_json::Map<_, _>>()});
            }
        }
        rec["all_agree"] = json!(ok);
        println!("{rec}");
        assert!(ok, "a verifier disagreed with upstream at k={k}");
    }
}
