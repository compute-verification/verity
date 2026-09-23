//! A small reference PROVER for the PCS, written from the paper (section 6.2, prover algorithm) in the concrete
//! protocol of `proto/pcs.py`, so the verifier has an oracle before the Python fixtures land and for the negatives
//! (`tamper`).  Toy sizes only (dense O(N) host work).  Not part of the verifier's trust base.

use crate::claims::{eq_vector, Term};
use crate::field::{add, mul, Ext, Ntt};
use crate::merkle::{leaf_digest, Tree};
use crate::params::Params;
use crate::pcs::{row_words, Opening, Proof, Statement};
use crate::transcript::{ext_bytes, Coins};

/// Codeword of one column (coefficient basis): evaluations of sum_x c[x] X^x at eta_s for every position s,
/// laid out in the encoder's position order (`Params::eta_index`).
fn encode_column(p: &Params, i: usize, coefs: &[Ext], ntt: &Ntt) -> Vec<Ext> {
    let m = p.m(i);
    let mut planes = vec![vec![0u32; m]; 6];
    let mut sc = 1u32;
    for (x, c) in coefs.iter().enumerate() {
        for d in 0..6 {
            planes[d][x] = mul(c.0[d], sc);
        }
        sc = mul(sc, p.domain_shift);
    }
    for pl in planes.iter_mut() {
        ntt.transform(pl, false); // natural order: index e holds the evaluation at w^e
    }
    (0..m)
        .map(|s| {
            let e = p.eta_index(i, s);
            Ext([planes[0][e], planes[1][e], planes[2][e], planes[3][e], planes[4][e], planes[5][e]])
        })
        .collect()
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
        let cw = encode_column(p, i, &mat[c * rows..(c + 1) * rows], &ntt);
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

fn open(p: &Params, i: usize, com: &Committed, s_set: &[usize]) -> Opening {
    let _ = (p, i);
    Opening { rows: s_set.iter().map(|&s| com.u[s].clone()).collect(), siblings: com.tree.open_multi(s_set) }
}

fn absorb_opening(coins: &mut dyn Coins, op: &Opening, base: bool) {
    let mut words = Vec::new();
    for row in &op.rows {
        for w in row_words(row, base) {
            words.extend_from_slice(&w.to_le_bytes());
        }
    }
    coins.absorb(b"rows", &words);
    let mut sib = Vec::with_capacity(32 * op.siblings.len());
    for d in &op.siblings {
        sib.extend_from_slice(d);
    }
    coins.absorb(b"sib", &sib);
}

/// Prove  sum_x f[x] eq(x, z) = v  (v computed here and returned in the statement).
pub fn prove(p: &Params, f: &[u32], z: &[Ext], coins: &mut dyn Coins) -> (Statement, Proof) {
    p.validate().unwrap();
    assert_eq!(f.len(), 1 << p.n);
    assert_eq!(z.len(), p.n);
    let v = evaluate(f, z);
    let stmt = Statement { z: z.to_vec(), v };
    coins.absorb(b"header", p.header_json().as_bytes());
    coins.absorb(b"z", &ext_bytes(z));

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
        if i == 1 {
            coins.absorb(b"v", &v.to_bytes());
        } else {
            let pc = prev.take().unwrap();
            let s_set = coins.indices(b"S", p.m(i - 1), p.queries[i - 2]).unwrap();
            let op = open(p, i - 1, &pc, &s_set);
            absorb_opening(coins, &op, i - 1 == 1);
            let rbar = eq_vector(&r_prev);
            let symbols: Vec<Ext> = op.rows.iter().map(|row| crate::field::dot_ext(row, &rbar)).collect();
            openings.push(op);
            let alpha = coins.challenge(b"alpha", 1 + s_set.len()).unwrap();
            for t in terms.iter_mut() {
                t.coef = t.coef.mul(&alpha[0]);
            }
            for (j, &s) in s_set.iter().enumerate() {
                terms.push(Term::geom(alpha[1 + j], p.domain_point(i - 1, s), p.k(i - 1)));
            }
            let _ = symbols; // the prover's running claim is recomputed from the tables below
        }
        let (rows, cols, k, kp) = (p.rows(i), p.cols(i), p.k(i), p.kprime[i - 1]);
        let mut tabs = tables(&terms, &mat, rows, cols, k, kp);
        let mut msgs = Vec::with_capacity(kp);
        let mut draws = Vec::with_capacity(kp);
        for _ in 0..kp {
            let g = round_message(&tabs);
            coins.absorb(b"g", &ext_bytes(&g));
            let r = coins.challenge(b"r", 1).unwrap()[0];
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
    let s_set = coins.indices(b"S", p.m(l), p.queries[l - 1]).unwrap();
    openings.push(open(p, l, &pc, &s_set));
    let proof = Proof { params: p.clone(), roots, sumcheck, openings, y_final: mat };
    (stmt, proof)
}

/// ligerito-proto's commit-first `commit` + `open(com, zs, values, coins, sparse=)` without ZK padding (the transcript
/// `lgto::pcs_verify` replays): `bind_root` gets root_1 first (the caller's relation transcript), then the opening absorbs
/// the points and the sparse terms and draws `m + K` batching coefficients.  A sparse term `sum_t w_t f[x_t]` enters each
/// round's sumcheck as `_Sparse.contract` does: one product per column `g`, rows weighted by the group's `w_t`, column
/// part one-hot at `g`.  Returns the values `f~(z_j)`, the sparse values and the proof.
pub fn prove_commit_first(
    p: &Params,
    f: &[u32],
    points: &[Vec<Ext>],
    sparse: &[(Vec<usize>, Vec<Ext>)],
    coins: &mut dyn Coins,
    bind_root: &mut dyn FnMut(&mut dyn Coins, &crate::merkle::Digest),
) -> (Vec<Ext>, Vec<Ext>, crate::lgto::PcsProof) {
    p.validate().unwrap();
    assert_eq!(f.len(), 1 << p.n);
    let l = p.rounds();
    let m = points.len();
    let values: Vec<Ext> = points.iter().map(|z| evaluate(f, z)).collect();
    let svals: Vec<Ext> = sparse.iter().map(|(c, w)| c.iter().zip(w).fold(Ext::ZERO, |a, (&x, w)| a.add(&w.mul_base(f[x])))).collect();
    let mut mat: Vec<Ext> = f.iter().map(|&x| Ext::from_base(x)).collect();
    let (mut roots, mut openings, mut sumcheck) = (Vec::new(), Vec::new(), Vec::new());
    let mut terms: Vec<Term> = Vec::new();
    let (mut sp_idx, mut sp_w): (Vec<usize>, Vec<Ext>) = (Vec::new(), Vec::new());
    let mut prev: Option<Committed> = None;
    for i in 1..=l {
        let com = commit(p, i, &mat);
        let root = com.tree.root();
        roots.push(root);
        if i == 1 {
            bind_root(coins, &root);
            let zb: Vec<u8> = points.iter().flat_map(|z| ext_bytes(z)).collect();
            coins.absorb(b"z", &zb);
            for (c, w) in sparse {
                let mut b: Vec<u8> = c.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect();
                b.extend(ext_bytes(w));
                coins.absorb(b"sparse", &b);
            }
            let beta = coins.challenge(b"beta", m + sparse.len()).unwrap();
            terms = points.iter().enumerate().map(|(j, z)| Term::eq(beta[j], z.clone())).collect();
            for (k, (c, w)) in sparse.iter().enumerate() {
                sp_idx.extend_from_slice(c);
                sp_w.extend(w.iter().map(|x| x.mul(&beta[m + k])));
            }
        } else {
            coins.absorb(b"root", &root);
            let pc = prev.take().unwrap();
            let s_set = coins.indices(b"S", p.m(i - 1), p.queries[i - 2]).unwrap();
            let op = open(p, i - 1, &pc, &s_set);
            absorb_opening(coins, &op, i - 1 == 1);
            openings.push(op);
            let alpha = coins.challenge(b"alpha", 1 + s_set.len()).unwrap();
            for t in terms.iter_mut() {
                t.coef = t.coef.mul(&alpha[0]);
            }
            for w in sp_w.iter_mut() {
                *w = w.mul(&alpha[0]);
            }
            for (j, &s) in s_set.iter().enumerate() {
                terms.push(Term::geom(alpha[1 + j], p.domain_point(i - 1, s), p.k(i - 1)));
            }
        }
        let (rows, cols, k, kp) = (p.rows(i), p.cols(i), p.k(i), p.kprime[i - 1]);
        let mut tabs = tables(&terms, &mat, rows, cols, k, kp);
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (t, &x) in sp_idx.iter().enumerate() {
            groups.entry(x >> k).or_default().push(t);
        }
        for (g, ts) in groups {
            let a = (0..cols).map(|c| ts.iter().fold(Ext::ZERO, |acc, &t| acc.add(&sp_w[t].mul(&mat[c * rows + (sp_idx[t] & (rows - 1))])))).collect();
            let mut b = vec![Ext::ZERO; cols];
            b[g] = Ext::ONE;
            tabs.push(Table { coef: Ext::ONE, a, b });
        }
        let mut msgs = Vec::with_capacity(kp);
        let mut draws = Vec::with_capacity(kp);
        for _ in 0..kp {
            let g = round_message(&tabs);
            coins.absorb(b"g", &ext_bytes(&g));
            let r = coins.challenge(b"r", 1).unwrap()[0];
            fold(&mut tabs, &r);
            msgs.push(g);
            draws.push(r);
        }
        sumcheck.push(msgs);
        draws.reverse();
        for t in terms.iter_mut() {
            *t = t.restrict(k, &draws);
        }
        let rbar = eq_vector(&draws);
        for (x, w) in sp_idx.iter_mut().zip(sp_w.iter_mut()) {
            *w = w.mul(&rbar[*x >> k]);
            *x &= rows - 1;
        }
        let mut y = vec![Ext::ZERO; rows];
        for c in 0..cols {
            for row in 0..rows {
                y[row] = y[row].add(&mat[c * rows + row].mul(&rbar[c]));
            }
        }
        mat = y;
        prev = Some(com);
    }
    let pc = prev.take().unwrap();
    coins.absorb(b"y", &ext_bytes(&mat));
    let s_set = coins.indices(b"S", p.m(l), p.queries[l - 1]).unwrap();
    openings.push(open(p, l, &pc, &s_set));
    (values, svals, crate::lgto::PcsProof { roots, sumcheck, openings, final_y: mat, ybar: Vec::new() })
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
    let last = proof.openings.len() - 1;
    let mut s = stmt.clone();
    s.v = bump(&s.v);
    out.push(("wrong evaluation v", s, proof.clone()));
    let mut s = stmt.clone();
    s.z[0] = bump(&s.z[0]);
    out.push(("wrong evaluation point z", s, proof.clone()));
    let mut pf = proof.clone();
    pf.openings[last].rows[0][0] = bump(&pf.openings[last].rows[0][0]);
    out.push(("tampered opened row (final)", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.y_final[1] = bump(&pf.y_final[1]);
    out.push(("tampered final vector y", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.sumcheck[0][0][2] = bump(&pf.sumcheck[0][0][2]);
    out.push(("tampered sumcheck g(2)", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.sumcheck[0][0][0] = bump(&pf.sumcheck[0][0][0]);
    out.push(("tampered sumcheck g(0)", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.roots[0][3] ^= 0x80;
    out.push(("tampered root", stmt.clone(), pf));
    if proof.openings.len() >= 2 {
        let mut pf = proof.clone();
        pf.openings[0].rows[0][0] = bump(&pf.openings[0].rows[0][0]);
        out.push(("tampered opened row (round 1)", stmt.clone(), pf));
        let mut pf = proof.clone();
        pf.openings[0].rows[0][0].0[1] = 1;
        out.push(("extension entry in a base-field row", stmt.clone(), pf));
        let mut pf = proof.clone();
        if !pf.openings[0].siblings.is_empty() {
            pf.openings[0].siblings[0][0] ^= 1;
            out.push(("tampered merkle sibling", stmt.clone(), pf));
        }
        let mut pf = proof.clone();
        if pf.openings[0].rows.len() >= 2 {
            pf.openings[0].rows.swap(0, 1);
            out.push(("swapped two opened rows", stmt.clone(), pf));
        }
        let mut pf = proof.clone();
        pf.openings[0].rows.pop();
        out.push(("missing opened row", stmt.clone(), pf));
        let mut pf = proof.clone();
        pf.openings[0].siblings.pop();
        out.push(("missing merkle sibling", stmt.clone(), pf));
    }
    let mut pf = proof.clone();
    pf.y_final.pop();
    out.push(("truncated final vector", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.sumcheck[0].pop();
    out.push(("missing sumcheck message", stmt.clone(), pf));
    let mut pf = proof.clone();
    pf.params.queries[0] = pf.params.queries[0].saturating_sub(1).max(1);
    out.push(("fewer queries than declared", stmt.clone(), pf));
    out
}
