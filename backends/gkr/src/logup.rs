//! logUp with the fractional-sum GKR (PROTOCOL.md 4.2, Hab22 / PH23): for table T with queries w_j and
//! multiplicities m_v,  Σ_j 1/(z - w_j) = Σ_v m_v/(z - t_v).  Leaves (p,q): witness (1, z - w_j), table
//! (-m_v, z - t_v), padding (0, 1); layer k combines pairs by (p0 q1 + p1 q0, q0 q1); the root numerator must be 0.
//! No division anywhere.  Vector-valued rows are compressed with powers of β.

use crate::circuit::{eval_lin, Circuit, Query, Table};
use crate::field::{eq_table, eval_mle, fold_in_place, interpolate, Ext, Fp};
use crate::gkr::{eq_point, Reader};
use crate::transcript::Transcript;
use std::collections::HashMap;

pub struct TableChallenges { pub z: Ext, pub beta: Ext }

fn log2ceil(n: usize) -> usize { let mut s = 0; while (1usize << s) < n { s += 1; } s }

/// Leaves for one table over the whole batch (used by prover and, in the clear-witness diagnostic, verifier).
pub fn leaves(circ: &Circuit, table: &Table, queries: &[&Query], cols: &[Fp], ncols: usize, units: usize, ch: &TableChallenges) -> (Vec<Ext>, Vec<Ext>, usize) {
    let n_wit = units * queries.len();
    let n = log2ceil(n_wit + table.rows.len());
    let size = 1usize << n;
    let mut p = vec![Ext::ZERO; size];
    let mut q = vec![Ext::ONE; size];
    // key -> row index (col 0 is the key for every table; R16's key is the value itself)
    let mut index: HashMap<u64, usize> = HashMap::with_capacity(table.rows.len());
    for (i, r) in table.rows.iter().enumerate() { index.insert(r[0].0, i); }
    let mut mult = vec![0u64; table.rows.len()];
    let betas: Vec<Ext> = { let mut v = vec![Ext::ONE]; for k in 1..table.cols { v.push(v[k - 1] * ch.beta); } v };
    for u in 0..units {
        let row = &cols[u * ncols..(u + 1) * ncols];
        for (j, qu) in queries.iter().enumerate() {
            let vals: Vec<Fp> = qu.cols.iter().map(|l| eval_lin(l, row)).collect();
            let mut comb = Ext::ZERO;
            for (k, v) in vals.iter().enumerate() { comb += betas[k].scale(*v); }
            let idx = u * queries.len() + j;
            p[idx] = Ext::ONE;
            q[idx] = ch.z - comb;
            if let Some(&ri) = index.get(&vals[0].0) {
                if table.rows[ri][..] == vals[..] { mult[ri] += 1; }
            }
        }
    }
    for (i, r) in table.rows.iter().enumerate() {
        let mut comb = Ext::ZERO;
        for (k, v) in r.iter().enumerate() { comb += betas[k].scale(*v); }
        p[n_wit + i] = -Ext::from_u64(mult[i]);
        q[n_wit + i] = ch.z - comb;
    }
    let _ = circ;
    (p, q, n)
}

pub struct LookupStats { pub n: usize, pub leaves: usize }

pub fn prove_table(p_leaf: Vec<Ext>, q_leaf: Vec<Ext>, n: usize, tr: &mut Transcript, out: &mut Vec<Ext>) -> LookupStats {
    // tree levels: level n = leaves ... level 0 = root
    let mut levels: Vec<(Vec<Ext>, Vec<Ext>)> = Vec::with_capacity(n + 1);
    levels.push((p_leaf, q_leaf));
    for _ in 0..n {
        let (p, q) = levels.last().unwrap();
        let half = p.len() / 2;
        let mut np = Vec::with_capacity(half);
        let mut nq = Vec::with_capacity(half);
        for x in 0..half {
            let (p0, q0, p1, q1) = (p[2 * x], q[2 * x], p[2 * x + 1], q[2 * x + 1]);
            np.push(p0 * q1 + p1 * q0);
            nq.push(q0 * q1);
        }
        levels.push((np, nq));
    }
    levels.reverse(); // levels[k] has 2^k entries
    let q_root = levels[0].1[0];
    debug_assert!(levels[0].0[0].is_zero(), "honest root numerator is zero");
    out.push(q_root);
    tr.absorb_ext(q_root);
    let mut lambda = tr.challenge();
    let mut rho: Vec<Ext> = vec![];
    for k in 0..n {
        let (pc, qc) = &levels[k + 1];
        let size = 1usize << k;
        let mut p0: Vec<Ext> = (0..size).map(|x| pc[2 * x]).collect();
        let mut p1: Vec<Ext> = (0..size).map(|x| pc[2 * x + 1]).collect();
        let mut q0: Vec<Ext> = (0..size).map(|x| qc[2 * x]).collect();
        let mut q1: Vec<Ext> = (0..size).map(|x| qc[2 * x + 1]).collect();
        let mut e = eq_table(&rho);
        let mut x_star = Vec::with_capacity(k);
        let mut len = size;
        for _ in 0..k {
            let half = len / 2;
            let mut h = [Ext::ZERO; 4];
            for x in 0..half {
                let d = |v: &Vec<Ext>| (v[x], v[x + half] - v[x]);
                let (a0, da) = d(&p0); let (b0, db) = d(&p1); let (c0, dc) = d(&q0); let (d0, dd) = d(&q1); let (e0, de) = d(&e);
                for t in 0..4u64 {
                    let tt = Ext::from_u64(t);
                    let (pa, pb, qa, qb) = (a0 + tt * da, b0 + tt * db, c0 + tt * dc, d0 + tt * dd);
                    h[t as usize] += (e0 + tt * de) * (pa * qb + pb * qa + lambda * (qa * qb));
                }
            }
            out.extend_from_slice(&h);
            tr.absorb_exts(&h);
            let r = tr.challenge();
            x_star.push(r);
            for v in [&mut p0, &mut p1, &mut q0, &mut q1, &mut e] { fold_in_place(v, r); }
            len = half;
        }
        let vals = [p0[0], p1[0], q0[0], q1[0]];
        out.extend_from_slice(&vals);
        tr.absorb_exts(&vals);
        let mu = tr.challenge();
        lambda = tr.challenge();
        rho = x_star;
        rho.push(mu);
    }
    LookupStats { n, leaves: 1 << n }
}

/// Verifier; `p_leaf`/`q_leaf` are the leaf vectors the verifier reconstructs from the clear witness (with a
/// PCS they would be opened as linear functionals of the committed columns).  Returns Ok on acceptance.
pub fn verify_table(p_leaf: &[Ext], q_leaf: &[Ext], n: usize, tr: &mut Transcript, rd: &mut Reader, name: &str) -> Result<(), String> {
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
    let pv = eval_mle(p_leaf, &rho);
    let qv = eval_mle(q_leaf, &rho);
    if pv + lambda * qv != claim { return Err(format!("{name}: leaf evaluation mismatch (multiset differs)")); }
    Ok(())
}
