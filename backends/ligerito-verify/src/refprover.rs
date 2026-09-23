//! A small reference PROVER for the PCS, written straight from the paper (section 6.2, prover algorithm) so the
//! verifier has an oracle before the Python spec (`ref.py`) lands.  Toy sizes only (dense O(N) work, host NTTs);
//! it also generates the negatives (`tamper`) the tests reject.  Not part of the verifier's trust base.

use crate::claims::{eq_vector, Term};
use crate::field::{add, mul, Ext, Ntt};
use crate::merkle::{leaf_digest, Tree};
use crate::params::Params;
use crate::pcs::{row_words, Opening, Proof, Statement};
use crate::transcript::{ext_bytes, Coins};

/// Codeword of one column (coefficient basis): evaluations of sum_x c[x] X^x on shift . <w_m>.
fn encode_column(coefs: &[Ext], m: usize, shift: u32, ntt: &Ntt) -> Vec<Ext> {
    let mut planes = vec![vec![0u32; m]; 6];
    let mut sc = 1u32;
    for (x, c) in coefs.iter().enumerate() {
        for d in 0..6 {
            planes[d][x] = mul(c.0[d], sc);
        }
        sc = mul(sc, shift);
    }
    for pl in planes.iter_mut() {
        ntt.transform(pl, false);
    }
    (0..m).map(|s| Ext([planes[0][s], planes[1][s], planes[2][s], planes[3][s], planes[4][s], planes[5][s]])).collect()
}

struct Committed {
    /// codeword rows: m x cols
    u: Vec<Vec<Ext>>,
    tree: Tree,
}

fn commit(p: &Params, i: usize, mat: &[Ext]) -> Committed {
    let (rows, cols, m) = (p.rows(i), p.cols(i), p.m(i));
    assert_eq!(mat.len(), rows * cols);
    let ntt = Ntt::new(m);
    let mut u = vec![vec![Ext::ZERO; cols]; m];
    for c in 0..cols {
        let cw = encode_column(&mat[c * rows..(c + 1) * rows], m, p.domain_shift, &ntt);
        for s in 0..m {
            u[s][c] = cw[s];
        }
    }
    let leaves = u.iter().map(|row| leaf_digest(&row_words(row, i == 1))).collect();
    Committed { u, tree: Tree::build(leaves) }
}

/// Per-term sumcheck tables: A_t[c] = sum_row M[row, c] rowpart_t[row], B_t[c] = colpart_t[c].
struct Table {
    coef: Ext,
    a: Vec<Ext>,
    b: Vec<Ext>,
}

fn tables(terms: &[Term], mat: &[Ext], rows: usize, cols: usize, k: usize, kp: usize) -> Vec<Table> {
    terms
        .iter()
        .map(|t| {
            let rowpart = t.expand(0, k);
            let colpart = t.expand(k, k + kp);
            debug_assert_eq!(rowpart.len(), rows);
            debug_assert_eq!(colpart.len(), cols);
            let a = (0..cols).map(|c| crate::field::dot_ext(&mat[c * rows..(c + 1) * rows], &rowpart)).collect();
            Table { coef: t.coef, a, b: colpart }
        })
        .collect()
}

fn round_message(tabs: &[Table]) -> [Ext; 3] {
    let mut g = [Ext::ZERO; 3];
    for t in tabs {
        let h = t.a.len() / 2;
        let mut s = [Ext::ZERO; 3];
        for c in 0..h {
            let (alo, ahi, blo, bhi) = (&t.a[c], &t.a[c + h], &t.b[c], &t.b[c + h]);
            s[0] = s[0].add(&alo.mul(blo));
            s[1] = s[1].add(&ahi.mul(bhi));
            let a2 = ahi.mul_base(2).sub(alo);
            let b2 = bhi.mul_base(2).sub(blo);
            s[2] = s[2].add(&a2.mul(&b2));
        }
        for j in 0..3 {
            g[j] = g[j].add(&t.coef.mul(&s[j]));
        }
    }
    g
}

fn fold(tabs: &mut [Table], r: &Ext) {
    let one_minus = Ext::ONE.sub(r);
    for t in tabs.iter_mut() {
        let h = t.a.len() / 2;
        t.a = (0..h).map(|c| t.a[c].mul(&one_minus).add(&t.a[c + h].mul(r))).collect();
        t.b = (0..h).map(|c| t.b[c].mul(&one_minus).add(&t.b[c + h].mul(r))).collect();
    }
}

/// Multilinear evaluation of the base vector f at z (bit j of x <-> z_j).
pub fn evaluate(f: &[u32], z: &[Ext]) -> Ext {
    assert_eq!(f.len(), 1 << z.len());
    crate::field::dot_ext_base(&eq_vector(z), f)
}

/// Prove  sum_x f[x] eq(x, z) = alpha  (alpha computed here and returned in the statement).
pub fn prove(p: &Params, f: &[u32], z: &[Ext], coins: &mut dyn Coins) -> (Statement, Proof) {
    p.validate().unwrap();
    assert_eq!(f.len(), 1 << p.n);
    assert_eq!(z.len(), p.n);
    let alpha = evaluate(f, z);
    let stmt = Statement { z: z.to_vec(), alpha };
    coins.absorb(b"z", &ext_bytes(z));
    coins.absorb(b"alpha", &alpha.to_bytes());

    let l = p.rounds();
    let mut mat: Vec<Ext> = f.iter().map(|&x| Ext::from_base(x)).collect();
    let mut terms: Vec<Term> = vec![Term::eq(Ext::ONE, z.to_vec())];
    let mut roots = Vec::new();
    let mut openings = Vec::new();
    let mut sumcheck = Vec::new();
    let mut prev: Option<Committed> = None;
    let mut r_prev: Vec<Ext> = Vec::new();

    for i in 1..=l {
        let com = commit(p, i, &mat);
        roots.push(com.tree.root());
        coins.absorb(b"root", &com.tree.root());
        if i >= 2 {
            let pc = prev.take().unwrap();
            let s_set = coins.squeeze_indices(p.m(i - 1), p.queries[i - 2]).unwrap();
            let rbar = eq_vector(&r_prev);
            let v: Vec<Ext> = s_set.iter().map(|&s| crate::field::dot_ext(&pc.u[s], &rbar)).collect();
            let rows: Vec<Vec<Ext>> = s_set.iter().map(|&s| pc.u[s].clone()).collect();
            let siblings = pc.tree.open_multi(&s_set);
            coins.absorb(b"v", &ext_bytes(&v));
            openings.push(Opening { v, rows, siblings });
            let alpha_i = coins.squeeze_ext(1).unwrap()[0];
            let mut w = alpha_i;
            for &s in &s_set {
                terms.push(Term::geom(w, p.domain_point(i - 1, s), p.k(i - 1)));
                w = w.mul(&alpha_i);
            }
        }
        let (rows, cols, k, kp) = (p.rows(i), p.cols(i), p.k(i), p.kprime[i - 1]);
        let mut tabs = tables(&terms, &mat, rows, cols, k, kp);
        let mut msgs = Vec::with_capacity(kp);
        let mut draws = Vec::with_capacity(kp);
        for _ in 0..kp {
            let g = round_message(&tabs);
            let mut gb = Vec::with_capacity(72);
            for e in &g {
                gb.extend_from_slice(&e.to_bytes());
            }
            coins.absorb(b"g", &gb);
            let r = coins.squeeze_ext(1).unwrap()[0];
            fold(&mut tabs, &r);
            msgs.push(g);
            draws.push(r);
        }
        sumcheck.push(msgs);
        draws.reverse();
        for t in terms.iter_mut() {
            *t = t.restrict(k, &draws);
        }
        // y = M r-bar
        let rbar = eq_vector(&draws);
        let mut y = vec![Ext::ZERO; rows];
        for c in 0..cols {
            for row in 0..rows {
                y[row] = y[row].add(&mat[c * rows + row].mul(&rbar[c]));
            }
        }
        mat = y;
        r_prev = draws;
        prev = Some(com);
    }
    let pc = prev.take().unwrap();
    coins.absorb(b"y", &ext_bytes(&mat));
    let s_set = coins.squeeze_indices(p.m(l), p.queries[l - 1]).unwrap();
    let rows: Vec<Vec<Ext>> = s_set.iter().map(|&s| pc.u[s].clone()).collect();
    let siblings = pc.tree.open_multi(&s_set);
    let proof = Proof {
        params: p.clone(),
        roots,
        openings,
        sumcheck,
        y_final: mat,
        final_opening: Opening { v: Vec::new(), rows, siblings },
    };
    (stmt, proof)
}

/// A deterministic pseudo-random base vector.
pub fn random_vector(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..1usize << n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % crate::field::P as u64) as u32
        })
        .collect()
}

pub fn random_point(n: usize, seed: u64) -> Vec<Ext> {
    let mut x = seed ^ 0xD1B5_4A32_D192_ED03;
    (0..n)
        .map(|_| {
            let mut c = [0u32; 6];
            for d in c.iter_mut() {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                *d = (x % crate::field::P as u64) as u32;
            }
            Ext(c)
        })
        .collect()
}

/// The negatives.  Each returns a description and a mutated (statement, proof).
pub fn tamper(stmt: &Statement, proof: &Proof) -> Vec<(&'static str, Statement, Proof)> {
    let mut out = Vec::new();
    let bump = |e: &Ext| -> Ext { Ext([add(e.0[0], 1), e.0[1], e.0[2], e.0[3], e.0[4], e.0[5]]) };
    // wrong evaluation
    let mut s = stmt.clone();
    s.alpha = bump(&s.alpha);
    out.push(("wrong evaluation alpha", s, proof.clone()));
    // tampered opened row entry (final round)
    let mut pf = proof.clone();
    pf.final_opening.rows[0][0] = bump(&pf.final_opening.rows[0][0]);
    out.push(("tampered opened row (final)", stmt.clone(), pf));
    // tampered final vector
    let mut pf = proof.clone();
    pf.y_final[1] = bump(&pf.y_final[1]);
    out.push(("tampered final vector y", stmt.clone(), pf));
    // tampered sumcheck message (keeps g0 + g1, breaks g2)
    let mut pf = proof.clone();
    pf.sumcheck[0][0][2] = bump(&pf.sumcheck[0][0][2]);
    out.push(("tampered sumcheck g(2)", stmt.clone(), pf));
    // tampered sumcheck message breaking the sum
    let mut pf = proof.clone();
    pf.sumcheck[0][0][0] = bump(&pf.sumcheck[0][0][0]);
    out.push(("tampered sumcheck g(0)", stmt.clone(), pf));
    // tampered root
    let mut pf = proof.clone();
    pf.roots[0][3] ^= 0x80;
    out.push(("tampered root", stmt.clone(), pf));
    if !proof.openings.is_empty() {
        // wrong claimed symbol
        let mut pf = proof.clone();
        pf.openings[0].v[0] = bump(&pf.openings[0].v[0]);
        out.push(("wrong claimed symbol v", stmt.clone(), pf));
        // tampered opened row in round 1 (base) -- row . r-bar changes
        let mut pf = proof.clone();
        pf.openings[0].rows[0][0] = bump(&pf.openings[0].rows[0][0]);
        out.push(("tampered opened row (round 1)", stmt.clone(), pf));
        // a row with an extension entry in the base round
        let mut pf = proof.clone();
        pf.openings[0].rows[0][0].0[1] = 1;
        out.push(("extension entry in a base-field row", stmt.clone(), pf));
        // a sibling digest flipped
        let mut pf = proof.clone();
        if !pf.openings[0].siblings.is_empty() {
            pf.openings[0].siblings[0][0] ^= 1;
            out.push(("tampered merkle sibling", stmt.clone(), pf));
        }
        // consistent row + symbol tamper: the row's dot product and v agree, but the Merkle leaf changes
        let mut pf = proof.clone();
        let cols = pf.openings[0].rows[0].len();
        // adding delta to row entry c changes row . r-bar by delta * rbar[c]; we cannot know rbar here, so pick a
        // whole-row swap between two opened rows instead (leaf mismatch at both indices)
        if pf.openings[0].rows.len() >= 2 && cols > 0 {
            pf.openings[0].rows.swap(0, 1);
            pf.openings[0].v.swap(0, 1);
            out.push(("swapped two opened rows with their symbols", stmt.clone(), pf));
        }
    }
    // wrong evaluation point
    let mut s = stmt.clone();
    s.z[0] = bump(&s.z[0]);
    out.push(("wrong evaluation point z", s, proof.clone()));
    // truncated final vector
    let mut pf = proof.clone();
    pf.y_final.pop();
    out.push(("truncated final vector", stmt.clone(), pf));
    // missing a sumcheck message
    let mut pf = proof.clone();
    pf.sumcheck[0].pop();
    out.push(("missing sumcheck message", stmt.clone(), pf));
    out
}
