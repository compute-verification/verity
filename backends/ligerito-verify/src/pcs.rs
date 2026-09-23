//! The Ligerito PCS verifier (paper section 6.2, verifier algorithm, with the tensor-claim bookkeeping of 6.6), in
//! the concrete protocol of `backends/direct/ligerito/proto/pcs.py` (lane ligerito-proto) and its Coins protocol.
//!
//! Statement: a base-field coefficient vector f of N = 2^n entries, committed in round 1, and the multilinear
//! evaluation claim  sum_x f[x] eq(x, z) = v  for z in F^n (bit j of x <-> z_j; the column bits of the column-major
//! matrix are the HIGH variables).
//!
//! Coins protocol (every `challenge` / `indices` is one verifier coin):
//!   absorb header (params JSON), absorb z | absorb root_1, absorb v | sumcheck round 1: k'_1 x [absorb g_j, r_j <-
//!   challenge("r")] | for i = 2..L: absorb root_i, S_{i-1} <- indices("S", m_{i-1}), absorb the opened rows of
//!   U_{i-1} at S_{i-1} (u32 words) and the multi-path siblings, alpha <- challenge("alpha", 1 + |S|), sumcheck
//!   round i | absorb y_L, S_L <- indices("S", m_L); the last opening is not absorbed (no coin follows it).
//!
//! Checks: every opened row against its root (BLAKE3 multi-path); g_j(0) + g_j(1) = running claim and
//! claim <- g_j(r_j); alpha-batching  claim <- alpha_0 claim + sum_j alpha_{1+j} < U_{i-1}[s_j, :], r-bar_{i-1} >
//! with the |S| geometric terms < eta_s^x , vec(M_i) > (nvars = k_{i-1}) joining the term list (existing terms
//! scaled by alpha_0); finally  P_y(eta_s) = < U_L[s, :], r-bar_L >  for s in S_L and  sum_t coef_t < tensor_t , y >
//! = running claim.  The claimed symbols v_{i-1} of the paper are not sent: they ARE the row inner products.

use crate::claims::{eq_vector, interpolate_deg2, powers, Term};
use crate::field::{dot_ext, dot_ext_base, Ext, EXT_DEG};
use crate::merkle::{leaf_digest, verify_multi, Digest};
use crate::params::Params;
use crate::transcript::{ext_bytes, Coins};

#[derive(Clone, Debug, PartialEq)]
pub struct Opening {
    /// Opened rows in S draw order, `cols` entries each (base-field entries for round 1).
    pub rows: Vec<Vec<Ext>>,
    /// Merkle siblings in `merkle::plan` order.
    pub siblings: Vec<Digest>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Proof {
    pub params: Params,
    pub roots: Vec<Digest>,
    /// sumcheck[i-1]: k'_i messages (values at 0, 1, 2) of round i.
    pub sumcheck: Vec<Vec<[Ext; 3]>>,
    /// openings[i-1]: the rows of round i's codeword opened at S_i (i = 1..L; the last during the final check).
    pub openings: Vec<Opening>,
    pub y_final: Vec<Ext>,
}

#[derive(Clone, Debug)]
pub struct Statement {
    pub z: Vec<Ext>,
    pub v: Ext,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub coins: usize,
    pub soundness_log2: f64,
    pub opened_rows: usize,
    pub terms_final: usize,
}

/// Leaf words of an opened row (what the prover hashes and absorbs): 1 u32 per entry in a base round; in an
/// extension round the 6 coordinate PLANES one after the other (word a C + c = coordinate a of entry c).
pub fn row_words(row: &[Ext], base: bool) -> Vec<u32> {
    if base {
        return row.iter().map(|e| e.0[0]).collect();
    }
    let mut w = Vec::with_capacity(EXT_DEG * row.len());
    for a in 0..EXT_DEG {
        w.extend(row.iter().map(|e| e.0[a]));
    }
    w
}

/// Inverse of `row_words`: `None` when the word count does not fit.
pub fn row_from_words(words: &[u32], cols: usize, base: bool) -> Option<Vec<Ext>> {
    if base {
        if words.len() != cols {
            return None;
        }
        return Some(words.iter().map(|&w| Ext::from_base(w)).collect());
    }
    if words.len() != EXT_DEG * cols {
        return None;
    }
    Some((0..cols).map(|c| Ext(std::array::from_fn(|a| words[a * cols + c]))).collect())
}

fn words_bytes(words: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 * words.len());
    for w in words {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b
}

/// Verify `proof` for `stmt`, drawing every challenge from `coins`.  Ok(report) = accepted.
pub fn verify(stmt: &Statement, proof: &Proof, coins: &mut dyn Coins) -> Result<Report, String> {
    let p = &proof.params;
    p.validate()?;
    let l = p.rounds();
    if stmt.z.len() != p.n {
        return Err(format!("statement: z has {} coordinates, n = {}", stmt.z.len(), p.n));
    }
    if proof.roots.len() != l || proof.openings.len() != l || proof.sumcheck.len() != l {
        return Err(format!(
            "proof: {} roots / {} openings / {} sumcheck rounds for {l} rounds",
            proof.roots.len(),
            proof.openings.len(),
            proof.sumcheck.len()
        ));
    }
    if proof.y_final.len() != p.final_len() {
        return Err(format!("proof: final vector has {} entries, expected {}", proof.y_final.len(), p.final_len()));
    }

    coins.absorb(b"header", p.header_json().as_bytes());
    coins.absorb(b"z", &ext_bytes(&stmt.z));

    let mut terms: Vec<Term> = vec![Term::eq(Ext::ONE, stmt.z.clone())];
    let mut claim = stmt.v;
    let mut r_prev: Vec<Ext> = Vec::new(); // challenges of the previous round, variable order
    let mut opened_rows = 0usize;

    for i in 1..=l {
        coins.absorb(b"root", &proof.roots[i - 1]);
        if i == 1 {
            coins.absorb(b"v", &stmt.v.to_bytes());
        } else {
            let prev = i - 1;
            let op = &proof.openings[prev - 1];
            let s_set = coins.indices(b"S", p.m(prev), p.queries[prev - 1])?;
            let symbols = check_opening(p, prev, op, &s_set, &proof.roots[prev - 1], &r_prev, Some(coins))?;
            opened_rows += s_set.len();
            let alpha = coins.challenge(b"alpha", 1 + s_set.len())?;
            for t in terms.iter_mut() {
                t.coef = t.coef.mul(&alpha[0]);
            }
            claim = claim.mul(&alpha[0]);
            for (j, &s) in s_set.iter().enumerate() {
                terms.push(Term::geom(alpha[1 + j], p.domain_point(prev, s), p.k(prev)));
                claim = claim.add(&alpha[1 + j].mul(&symbols[j]));
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
            coins.absorb(b"g", &ext_bytes(g));
            let r = coins.challenge(b"r", 1)?[0];
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

    // final: y in the clear
    coins.absorb(b"y", &ext_bytes(&proof.y_final));
    let s_set = coins.indices(b"S", p.m(l), p.queries[l - 1])?;
    let symbols = check_opening(p, l, &proof.openings[l - 1], &s_set, &proof.roots[l - 1], &r_prev, None)?;
    opened_rows += s_set.len();
    for (j, &s) in s_set.iter().enumerate() {
        let want = dot_ext_base(&proof.y_final, &powers(p.domain_point(l, s), proof.y_final.len()));
        if want != symbols[j] {
            return Err("final check: the final vector is inconsistent with an opened row".into());
        }
    }
    let mut expect = Ext::ZERO;
    for t in &terms {
        expect = expect.add(&t.dot(&proof.y_final));
    }
    if expect != claim {
        return Err("final check: the running claim does not match w~_l^T y".into());
    }
    Ok(Report { coins: coins.rounds(), soundness_log2: p.soundness_log2().0, opened_rows, terms_final: terms.len() })
}

/// Rows of round `i`'s codeword opened at `s_set`: shape, canonical base entries in round 1, the Merkle
/// multi-opening; absorbs the rows and siblings when `coins` is given; returns the row inner products with r-bar.
pub(crate) fn check_opening(
    p: &Params,
    i: usize,
    op: &Opening,
    s_set: &[usize],
    root: &Digest,
    r: &[Ext],
    coins: Option<&mut dyn Coins>,
) -> Result<Vec<Ext>, String> {
    let cols = p.cols(i);
    let base = i == 1;
    if op.rows.len() != s_set.len() {
        return Err(format!("round {i} opening: {} rows for {} queries", op.rows.len(), s_set.len()));
    }
    let rbar = eq_vector(r);
    debug_assert_eq!(rbar.len(), cols);
    let mut leaves = Vec::with_capacity(s_set.len());
    let mut symbols = Vec::with_capacity(s_set.len());
    let mut all_words = Vec::with_capacity(s_set.len() * cols * if base { 1 } else { EXT_DEG });
    for (j, (&s, row)) in s_set.iter().zip(op.rows.iter()).enumerate() {
        if row.len() != cols {
            return Err(format!("round {i} opening: row {j} has {} entries, expected {cols}", row.len()));
        }
        if base && !row.iter().all(|e| e.is_base()) {
            return Err(format!("round {i} opening: row {j} of the base-field round has extension entries"));
        }
        symbols.push(dot_ext(row, &rbar));
        let words = row_words(row, base);
        leaves.push((s, leaf_digest(&words)));
        all_words.extend_from_slice(&words);
    }
    if let Some(c) = coins {
        c.absorb(b"rows", &words_bytes(&all_words));
        let mut sib = Vec::with_capacity(32 * op.siblings.len());
        for d in &op.siblings {
            sib.extend_from_slice(d);
        }
        c.absorb(b"sib", &sib);
    }
    verify_multi(root, &leaves, &op.siblings, p.depth(i)).map_err(|e| format!("round {i} {e}"))?;
    Ok(symbols)
}
