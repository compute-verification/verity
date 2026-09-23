//! The Ligerito PCS verifier (paper section 6.2, verifier algorithm, with the tensor-claim bookkeeping of 6.6).
//!
//! Statement: a base-field coefficient vector f of N = 2^n entries, committed in round 1, and the multilinear
//! evaluation claim  sum_x f[x] eq(x, z) = alpha  for z in F^n (bit j of x <-> z_j).
//!
//! Round i = 1..L (L committed rounds = the paper's l - 1):
//!   receive root_i;
//!   if i >= 2: coin S_{i-1} (|S_{i-1}| distinct rows of codeword i-1), receive the claimed symbols v_{i-1} and the
//!     opened rows with their Merkle multi-opening; check row_s . r-bar_{i-1} = v_{i-1}[s] and the opening; coin
//!     alpha_i and batch the |S| geometric claims  < eta_s^x , vec(M_i) > = v[s]  with weights alpha_i^(j+1) into
//!     the running claim;
//!   partial sumcheck over the k'_i column variables, highest first: receive g_j (values at 0, 1, 2), check
//!     g_j(0) + g_j(1) = claim, coin r_j, claim <- g_j(r_j); then restrict every term at r_i.
//! Final: receive y (2^{k_L} entries); coin S_L; check row_s . r-bar_L = sum_x eta_s^x y[x] and the opening;
//!   check claim = sum_t coef_t < tensor_t , y >.
//!
//! Transcript order (Fiat-Shamir tags): "z", "alpha", then per round "root", ["v"], "g" x k'_i, finally "y".

use crate::claims::{eq_vector, interpolate_deg2, powers, Term};
use crate::field::{dot_ext, dot_ext_base, Ext};
use crate::merkle::{leaf_digest, verify_multi, Digest};
use crate::params::Params;
use crate::transcript::{ext_bytes, Coins};

#[derive(Clone, Debug, PartialEq)]
pub struct Opening {
    /// Claimed symbols v_{i-1}[s] (absent for the final opening, where the verifier computes them from y).
    pub v: Vec<Ext>,
    /// Opened rows in S draw order, `cols` entries each (base-field entries for round 1).
    pub rows: Vec<Vec<Ext>>,
    /// Merkle siblings in `merkle::plan` order.
    pub siblings: Vec<Digest>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub params: Params,
    pub roots: Vec<Digest>,
    /// openings[i-2] opens round i-1 during round i (i = 2..L).
    pub openings: Vec<Opening>,
    /// sumcheck[i-1]: k'_i messages of round i.
    pub sumcheck: Vec<Vec<[Ext; 3]>>,
    pub y_final: Vec<Ext>,
    pub final_opening: Opening,
}

#[derive(Clone, Debug)]
pub struct Statement {
    pub z: Vec<Ext>,
    pub alpha: Ext,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub coins: usize,
    pub soundness_log2: f64,
    pub opened_rows: usize,
    pub terms_final: usize,
}

/// Leaf words of an opened row: 1 u32 per entry in a base round, 6 per entry otherwise.
pub fn row_words(row: &[Ext], base: bool) -> Vec<u32> {
    let mut w = Vec::with_capacity(if base { row.len() } else { 6 * row.len() });
    for e in row {
        if base {
            w.push(e.0[0]);
        } else {
            w.extend_from_slice(&e.0);
        }
    }
    w
}

/// Verify `proof` for `stmt`, drawing every challenge from `coins`.  Ok(report) = accepted.
pub fn verify(stmt: &Statement, proof: &Proof, coins: &mut dyn Coins) -> Result<Report, String> {
    let p = &proof.params;
    p.validate()?;
    let l = p.rounds();
    if stmt.z.len() != p.n {
        return Err(format!("statement: z has {} coordinates, n = {}", stmt.z.len(), p.n));
    }
    if proof.roots.len() != l {
        return Err(format!("proof: {} roots for {l} rounds", proof.roots.len()));
    }
    if proof.openings.len() != l - 1 {
        return Err(format!("proof: {} openings for {l} rounds", proof.openings.len()));
    }
    if proof.sumcheck.len() != l {
        return Err(format!("proof: {} sumcheck rounds for {l} rounds", proof.sumcheck.len()));
    }
    if proof.y_final.len() != p.final_len() {
        return Err(format!("proof: final vector has {} entries, expected {}", proof.y_final.len(), p.final_len()));
    }

    coins.absorb(b"z", &ext_bytes(&stmt.z));
    coins.absorb(b"alpha", &stmt.alpha.to_bytes());

    let mut terms: Vec<Term> = vec![Term::eq(Ext::ONE, stmt.z.clone())];
    let mut claim = stmt.alpha;
    let mut r_prev: Vec<Ext> = Vec::new(); // challenges of the previous round, variable order
    let mut opened_rows = 0usize;

    for i in 1..=l {
        coins.absorb(b"root", &proof.roots[i - 1]);
        if i >= 2 {
            let prev = i - 1;
            let op = &proof.openings[i - 2];
            let s_set = coins.squeeze_indices(p.m(prev), p.queries[prev - 1])?;
            check_opening(p, prev, op, &s_set, &proof.roots[prev - 1], Some(&r_prev), None)?;
            opened_rows += s_set.len();
            coins.absorb(b"v", &ext_bytes(&op.v));
            // batch the |S| geometric claims on vec(M_i) with alpha^(j+1)
            let alpha = coins.squeeze_ext(1)?[0];
            let mut w = alpha;
            for (j, &s) in s_set.iter().enumerate() {
                terms.push(Term::geom(w, p.domain_point(prev, s), p.k(prev)));
                claim = claim.add(&w.mul(&op.v[j]));
                w = w.mul(&alpha);
            }
        }
        // partial sumcheck over the k'_i column variables, highest bit first
        let msgs = &proof.sumcheck[i - 1];
        if msgs.len() != p.kprime[i - 1] {
            return Err(format!("round {i}: {} sumcheck messages, k' = {}", msgs.len(), p.kprime[i - 1]));
        }
        let mut draws = Vec::with_capacity(msgs.len());
        for (j, g) in msgs.iter().enumerate() {
            if g[0].add(&g[1]) != claim {
                return Err(format!("round {i}: sumcheck message {j} does not sum to the running claim"));
            }
            let mut gb = Vec::with_capacity(72);
            for e in g {
                gb.extend_from_slice(&e.to_bytes());
            }
            coins.absorb(b"g", &gb);
            let r = coins.squeeze_ext(1)?[0];
            claim = interpolate_deg2(g, &r);
            draws.push(r);
        }
        draws.reverse(); // r[j] <-> variable k_i + j
        let k = p.k(i);
        for t in terms.iter_mut() {
            *t = t.restrict(k, &draws);
        }
        r_prev = draws;
    }

    // final round: y in the clear
    coins.absorb(b"y", &ext_bytes(&proof.y_final));
    let s_set = coins.squeeze_indices(p.m(l), p.queries[l - 1])?;
    check_opening(p, l, &proof.final_opening, &s_set, &proof.roots[l - 1], Some(&r_prev), Some(&proof.y_final))?;
    opened_rows += s_set.len();
    let mut expect = Ext::ZERO;
    for t in &terms {
        expect = expect.add(&t.dot(&proof.y_final));
    }
    if expect != claim {
        return Err("final check: the running claim does not match w~_l^T y".into());
    }
    Ok(Report { coins: coins.coins(), soundness_log2: p.soundness_log2().0, opened_rows, terms_final: terms.len() })
}

/// Rows of round `i`'s codeword opened at `s_set`: shape, canonical base entries in round 1, the Merkle
/// multi-opening, and row_s . r-bar = v[s] (or = sum_x eta_s^x y[x] when `y` is given, the final round).
fn check_opening(
    p: &Params,
    i: usize,
    op: &Opening,
    s_set: &[usize],
    root: &Digest,
    r: Option<&[Ext]>,
    y: Option<&[Ext]>,
) -> Result<(), String> {
    let cols = p.cols(i);
    let base = i == 1;
    if op.rows.len() != s_set.len() {
        return Err(format!("round {i} opening: {} rows for {} queries", op.rows.len(), s_set.len()));
    }
    if y.is_none() && op.v.len() != s_set.len() {
        return Err(format!("round {i} opening: {} symbols for {} queries", op.v.len(), s_set.len()));
    }
    if y.is_some() && !op.v.is_empty() {
        return Err(format!("round {i} opening: the final opening carries symbols"));
    }
    let rbar = eq_vector(r.expect("challenges"));
    debug_assert_eq!(rbar.len(), cols);
    let mut leaves = Vec::with_capacity(s_set.len());
    for (j, (&s, row)) in s_set.iter().zip(op.rows.iter()).enumerate() {
        if row.len() != cols {
            return Err(format!("round {i} opening: row {j} has {} entries, expected {cols}", row.len()));
        }
        if base && !row.iter().all(|e| e.is_base()) {
            return Err(format!("round {i} opening: row {j} of the base-field round has extension entries"));
        }
        let got = dot_ext(row, &rbar);
        let want = match y {
            None => op.v[j],
            Some(y) => dot_ext_base(y, &powers(p.domain_point(i, s), y.len())),
        };
        if got != want {
            return Err(format!("round {i} opening: row {s} . r-bar does not match the claimed symbol"));
        }
        leaves.push((s, leaf_digest(&row_words(row, base))));
    }
    verify_multi(root, &leaves, &op.siblings, p.depth(i)).map_err(|e| format!("round {i} {e}"))
}
