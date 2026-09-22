//! logUp with the fractional-sum GKR (PROTOCOL.md 4.2, Hab22 / PH23): for table T with queries w_j and
//! multiplicities m_v,  Σ_j 1/(z - w_j) = Σ_v m_v/(z - t_v).  Leaves (p,q): witness (1, z - w_j), table
//! (-m_v, z - t_v), padding (0, 1); layer k combines pairs by (p0 q1 + p1 q0, q0 q1); the root numerator must be 0.
//! No division anywhere.  Vector-valued rows are compressed with powers of β.
//!
//! The witness side of one table may come from several data-parallel circuits ("segments": the unit circuit and
//! the epilogue circuit of a VU batch, PROTOCOL.md 12); their queries are laid out one segment after another.

use crate::circuit::{eval_lin, Query, Table};
use crate::field::{eq_table, eval_mle, fold_in_place, interpolate, Ext, Fp};
use crate::gkr::{eq_point, threads, Reader};
use crate::transcript::Transcript;
use std::collections::HashMap;

pub struct TableChallenges { pub z: Ext, pub beta: Ext }

fn log2ceil(n: usize) -> usize { let mut s = 0; while (1usize << s) < n { s += 1; } s }

/// One segment's queries into a table: the query forms over its columns, and its committed rows.
pub struct SegQueries<'a> { pub queries: Vec<&'a Query>, pub cols: &'a [Fp], pub ncols: usize, pub units: usize }

impl<'a> SegQueries<'a> { pub fn n_wit(&self) -> usize { self.units * self.queries.len() } }

pub fn hypercube_dim(segs: &[SegQueries], table: &Table) -> usize { log2ceil(segs.iter().map(|s| s.n_wit()).sum::<usize>() + table.rows.len()) }

pub fn betas(table: &Table, ch: &TableChallenges) -> Vec<Ext> { let mut v = vec![Ext::ONE]; for k in 1..table.cols { v.push(v[k - 1] * ch.beta); } v }

/// Leaves for one table over every segment (prover; and the clear-witness diagnostic verifier).  Returns
/// (p, q, n, multiplicities); with `z = 0, beta = 1` only the multiplicities are meaningful.
pub fn leaves(table: &Table, segs: &[SegQueries], ch: &TableChallenges) -> (Vec<Ext>, Vec<Ext>, usize, Vec<u64>) { fill(table, segs, ch, true) }

/// Only the multiplicities (challenge-independent; counted before the commitment).
pub fn multiplicities(table: &Table, segs: &[SegQueries]) -> Vec<u64> { fill(table, segs, &TableChallenges { z: Ext::ZERO, beta: Ext::ONE }, false).3 }

fn fill(table: &Table, segs: &[SegQueries], ch: &TableChallenges, want_pq: bool) -> (Vec<Ext>, Vec<Ext>, usize, Vec<u64>) {
    let n = hypercube_dim(segs, table);
    let size = if want_pq { 1usize << n } else { 0 };
    let mut p = vec![Ext::ZERO; size];
    let mut q = vec![Ext::ONE; size];
    let mut index: HashMap<Fp, usize> = HashMap::with_capacity(table.rows.len());
    for (i, r) in table.rows.iter().enumerate() { index.insert(r[0], i); }
    let mut mult = vec![0u64; table.rows.len()];
    let bs = betas(table, ch);
    let nthreads = threads();
    let mut base = 0usize;
    for seg in segs {
        let q_t = seg.queries.len();
        if q_t == 0 { continue; }
        let n_wit = seg.n_wit();
        let (pw, qw): (&mut [Ext], &mut [Ext]) = if want_pq { (&mut p[base..base + n_wit], &mut q[base..base + n_wit]) } else { (&mut [], &mut []) };
        let chunk = seg.units.div_ceil(nthreads).max(1);
        let n_chunks = seg.units.div_ceil(chunk);
        let partial: Vec<Vec<u64>> = std::thread::scope(|s| {
            let mut hs = Vec::new();
            let mut pcs: Vec<&mut [Ext]> = if want_pq { pw.chunks_mut(chunk * q_t).collect() } else { (0..n_chunks).map(|_| &mut [][..]).collect() };
            let mut qcs: Vec<&mut [Ext]> = if want_pq { qw.chunks_mut(chunk * q_t).collect() } else { (0..n_chunks).map(|_| &mut [][..]).collect() };
            for ci in 0..n_chunks {
                let (pc, qc) = (std::mem::take(&mut pcs[ci]), std::mem::take(&mut qcs[ci]));
                let (index, bs, table, queries) = (&index, &bs, table, &seg.queries);
                let cols = seg.cols;
                let ncols = seg.ncols;
                let units = seg.units;
                hs.push(s.spawn(move || {
                    let mut m = vec![0u64; table.rows.len()];
                    let mut vals: Vec<Fp> = Vec::with_capacity(table.cols);
                    let lo = ci * chunk;
                    let hi = ((ci + 1) * chunk).min(units);
                    for u in lo..hi {
                        let k = u - lo;
                        let row = &cols[u * ncols..(u + 1) * ncols];
                        for (j, qu) in queries.iter().enumerate() {
                            vals.clear();
                            vals.extend(qu.cols.iter().map(|l| eval_lin(l, row)));
                            if want_pq {
                                let mut comb = Ext::ZERO;
                                for (kk, v) in vals.iter().enumerate() { comb += bs[kk].scale(*v); }
                                pc[k * q_t + j] = Ext::ONE;
                                qc[k * q_t + j] = ch.z - comb;
                            }
                            if let Some(&ri) = index.get(&vals[0]) {
                                if table.rows[ri][..] == vals[..] { m[ri] += 1; }
                            }
                        }
                    }
                    m
                }));
            }
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for m in partial { for (a, b) in mult.iter_mut().zip(m) { *a += b; } }
        base += n_wit;
    }
    if want_pq {
        for (i, r) in table.rows.iter().enumerate() {
            let mut comb = Ext::ZERO;
            for (k, v) in r.iter().enumerate() { comb += bs[k].scale(*v); }
            p[base + i] = -Ext::from_u64(mult[i]);
            q[base + i] = ch.z - comb;
        }
    }
    (p, q, n, mult)
}

pub struct LookupStats { pub n: usize, pub leaves: usize }

fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let nthreads = threads().min(n.max(1));
    let chunk = n.div_ceil(nthreads).max(1);
    let f = &f;
    let parts: Vec<Vec<T>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads).map(|t| s.spawn(move || { let lo = t * chunk; let hi = ((t + 1) * chunk).min(n); (lo..hi).map(f).collect::<Vec<T>>() })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.into_iter().flatten().collect()
}

/// Σ over x < half of the degree-3 round polynomial at t = 0..3 (parallel over x).
fn round_poly(p0: &[Ext], p1: &[Ext], q0: &[Ext], q1: &[Ext], e: &[Ext], half: usize, lambda: Ext) -> [Ext; 4] {
    let nthreads = threads().min(half.max(1));
    let chunk = half.div_ceil(nthreads).max(1);
    let parts: Vec<[Ext; 4]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads).map(|t| s.spawn(move || {
            let mut h = [Ext::ZERO; 4];
            for x in t * chunk..((t + 1) * chunk).min(half) {
                let d = |v: &[Ext]| (v[x], v[x + half] - v[x]);
                let (a0, da) = d(p0); let (b0, db) = d(p1); let (c0, dc) = d(q0); let (d0, dd) = d(q1); let (e0, de) = d(e);
                for tt in 0..4u64 {
                    let t = Ext::from_u64(tt);
                    let (pa, pb, qa, qb) = (a0 + t * da, b0 + t * db, c0 + t * dc, d0 + t * dd);
                    h[tt as usize] += (e0 + t * de) * (pa * qb + pb * qa + lambda * (qa * qb));
                }
            }
            h
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut h = [Ext::ZERO; 4];
    for p in parts { for t in 0..4 { h[t] += p[t]; } }
    h
}

fn par_fold(v: &mut Vec<Ext>, r: Ext) {
    let half = v.len() / 2;
    if half < 1 << 14 { fold_in_place(v, r); return; }
    let (lo, hi) = v.split_at_mut(half);
    let nthreads = threads();
    let chunk = half.div_ceil(nthreads).max(1);
    std::thread::scope(|s| {
        for (lc, hc) in lo.chunks_mut(chunk).zip(hi.chunks(chunk)) {
            s.spawn(move || { for (a, b) in lc.iter_mut().zip(hc) { *a = *a + (*b - *a) * r; } });
        }
    });
    v.truncate(half);
}

pub fn prove_table(p_leaf: Vec<Ext>, q_leaf: Vec<Ext>, n: usize, tr: &mut Transcript, out: &mut Vec<Ext>) -> (LookupStats, LeafClaim) {
    // tree levels: level n = leaves ... level 0 = root
    let mut levels: Vec<(Vec<Ext>, Vec<Ext>)> = Vec::with_capacity(n + 1);
    levels.push((p_leaf, q_leaf));
    for _ in 0..n {
        let (p, q) = levels.last().unwrap();
        let half = p.len() / 2;
        let pairs: Vec<(Ext, Ext)> = par_map(half, |x| {
            let (p0, q0, p1, q1) = (p[2 * x], q[2 * x], p[2 * x + 1], q[2 * x + 1]);
            (p0 * q1 + p1 * q0, q0 * q1)
        });
        let np: Vec<Ext> = pairs.iter().map(|x| x.0).collect();
        let nq: Vec<Ext> = pairs.iter().map(|x| x.1).collect();
        levels.push((np, nq));
    }
    levels.reverse(); // levels[k] has 2^k entries
    let q_root = levels[0].1[0];
    debug_assert!(levels[0].0[0].is_zero(), "honest root numerator is zero");
    out.push(q_root);
    tr.label("logup: root denominator q_root");
    tr.absorb_ext(q_root);
    tr.label("logup: lambda");
    tr.round_trip();
    let mut lambda = tr.challenge();
    let mut rho: Vec<Ext> = vec![];
    let mut claim = lambda * q_root;
    for k in 0..n {
        let (pc, qc) = &levels[k + 1];
        let size = 1usize << k;
        let mut p0: Vec<Ext> = par_map(size, |x| pc[2 * x]);
        let mut p1: Vec<Ext> = par_map(size, |x| pc[2 * x + 1]);
        let mut q0: Vec<Ext> = par_map(size, |x| qc[2 * x]);
        let mut q1: Vec<Ext> = par_map(size, |x| qc[2 * x + 1]);
        levels[k + 1] = (vec![], vec![]); // this level is no longer needed
        let mut e = eq_table(&rho);
        let mut x_star = Vec::with_capacity(k);
        let mut len = size;
        tr.label(&format!("logup: level {k}: {k} rounds of degree-3 polynomials h(0..3), then (p0,p1,q0,q1), then mu, lambda"));
        for _ in 0..k {
            let half = len / 2;
            let h = round_poly(&p0, &p1, &q0, &q1, &e, half, lambda);
            out.extend_from_slice(&h);
            tr.absorb_exts(&h);
            tr.round_trip();
            let r = tr.challenge();
            x_star.push(r);
            for v in [&mut p0, &mut p1, &mut q0, &mut q1, &mut e] { par_fold(v, r); }
            len = half;
        }
        let vals = [p0[0], p1[0], q0[0], q1[0]];
        out.extend_from_slice(&vals);
        tr.absorb_exts(&vals);
        tr.round_trip(); // mu and lambda are two coins of ONE verifier message (no prover message between them)
        let mu = tr.challenge();
        lambda = tr.challenge();
        rho = x_star;
        rho.push(mu);
        claim = (vals[0] + mu * (vals[1] - vals[0])) + lambda * (vals[2] + mu * (vals[3] - vals[2]));
    }
    (LookupStats { n, leaves: 1 << n }, LeafClaim { rho, lambda, value: claim })
}

/// The verifier's residual claim after the fractional GKR: p̃(ρ) + λ q̃(ρ) = value on the leaf vectors.
pub struct LeafClaim { pub rho: Vec<Ext>, pub lambda: Ext, pub value: Ext }

/// Verifier of the fractional GKR; returns the leaf claim, which the caller checks against the clear leaf
/// vectors (`check_leaves`) or turns into a linear functional on the committed columns (PCS mode).
pub fn verify_table(n: usize, tr: &mut Transcript, rd: &mut Reader, name: &str) -> Result<LeafClaim, String> {
    let q_root = rd.next()?;
    if q_root.is_zero() { return Err(format!("{name}: zero root denominator")); }
    tr.absorb_ext(q_root);
    let mut lambda = tr.challenge();
    let mut rho: Vec<Ext> = vec![];
    let mut claim = lambda * q_root; // p_root = 0 asserted
    for k in 0..n {
        let mut running = claim;
        let mut x_star = Vec::with_capacity(k);
        for round in 0..k {
            let h = rd.take(4)?;
            if h[0] + h[1] != running { return Err(format!("{name}: level {k} round {round} sum mismatch")); }
            tr.absorb_exts(&h);
            let r = tr.challenge();
            running = interpolate(&h, r);
            x_star.push(r);
        }
        let v = rd.take(4)?;
        let (p0, p1, q0, q1) = (v[0], v[1], v[2], v[3]);
        let e_star = eq_point(&rho, &x_star);
        if e_star * (p0 * q1 + p1 * q0 + lambda * (q0 * q1)) != running { return Err(format!("{name}: level {k} final identity")); }
        tr.absorb_exts(&v);
        let mu = tr.challenge();
        lambda = tr.challenge();
        rho = x_star;
        rho.push(mu);
        let p_next = p0 + mu * (p1 - p0);
        let q_next = q0 + mu * (q1 - q0);
        claim = p_next + lambda * q_next;
    }
    Ok(LeafClaim { rho, lambda, value: claim })
}

pub fn check_leaves(p_leaf: &[Ext], q_leaf: &[Ext], c: &LeafClaim, name: &str) -> Result<(), String> {
    let pv = eval_mle(p_leaf, &c.rho);
    let qv = eval_mle(q_leaf, &c.rho);
    if pv + c.lambda * qv != c.value { return Err(format!("{name}: leaf evaluation mismatch (multiset differs)")); }
    Ok(())
}

pub fn log2ceil_pub(n: usize) -> usize { log2ceil(n) }
