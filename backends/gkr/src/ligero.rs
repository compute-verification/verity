//! Hash-based Ligero-style commitment to the witness vector (PROTOCOL.md 3, without the ZK rows / HM96 leaves):
//! X is laid out as R rows of k symbols (evaluations on the subgroup H, |H| = k), each row Reed-Solomon encoded to
//! n = 4k points (H ⊂ D), columns Merkle-hashed with SHA-256.  Two tests on t opened columns:
//!   interleaved proximity: w = Σ_i r_i X_i sent as a message row; Enc(w)_j = Σ_i r_i U_ij at opened j;
//!   linear functional ⟨a, X⟩ = b: q(x) = Σ_i a_i(x) P_i(x) (deg < 2k-1) sent by coefficients; Σ_{ζ∈H} q(ζ) = b,
//!   q(η_j) = Σ_i a_i(η_j) U_ij at opened j, a_i(η_j) recomputed by the verifier (a is public).
//! No masking: the opened columns reveal witness symbols → NON_ZK_PROOF_DIAGNOSTIC.

use crate::field::{Ext, Fp};
use crate::gkr::Reader;
use crate::transcript::{Sha256, Transcript};

pub struct Params { pub k: usize, pub n: usize, pub t: usize }

fn root_of_unity(order: usize) -> Fp {
    // 7 generates F_p^*; p - 1 = 2^32 (2^32 - 1)
    let w32 = Fp::new(7).pow((1u64 << 32) - 1);
    let mut w = w32;
    let mut o = 1u64 << 32;
    while o > order as u64 { w = w * w; o /= 2; }
    w
}

fn bitrev(v: &mut [Fp]) {
    let n = v.len();
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if (j as usize) > i { v.swap(i, j as usize); }
    }
}

/// In-place NTT (evaluations of the polynomial with coefficients `v` on ⟨ω_n⟩), or inverse.
pub fn ntt(v: &mut [Fp], inverse: bool) {
    let n = v.len();
    if n == 1 { return; }
    bitrev(v);
    let mut w_n = root_of_unity(n);
    if inverse { w_n = w_n.inv(); }
    let mut len = 2;
    while len <= n {
        let w_len = w_n.pow((n / len) as u64);
        for start in (0..n).step_by(len) {
            let mut w = Fp::ONE;
            for j in 0..len / 2 {
                let u = v[start + j];
                let t = v[start + j + len / 2] * w;
                v[start + j] = u + t;
                v[start + j + len / 2] = u - t;
                w = w * w_len;
            }
        }
        len *= 2;
    }
    if inverse {
        let ninv = Fp::new(n as u64).inv();
        for x in v.iter_mut() { *x = *x * ninv; }
    }
}

pub fn ntt_ext(v: &mut [Ext], inverse: bool) {
    for c in 0..3 {
        let mut col: Vec<Fp> = v.iter().map(|x| x.0[c]).collect();
        ntt(&mut col, inverse);
        for (x, y) in v.iter_mut().zip(col) { x.0[c] = y; }
    }
}

/// k evaluations on H -> n evaluations on D.
pub fn encode(row: &[Fp], n: usize) -> Vec<Fp> {
    let mut c = row.to_vec();
    ntt(&mut c, true);
    c.resize(n, Fp::ZERO);
    ntt(&mut c, false);
    c
}

pub fn encode_ext(row: &[Ext], n: usize) -> Vec<Ext> {
    let mut c = row.to_vec();
    ntt_ext(&mut c, true);
    c.resize(n, Ext::ZERO);
    ntt_ext(&mut c, false);
    c
}

pub struct Committed { pub rows: usize, pub codewords: Vec<Fp>, pub tree: Vec<Vec<[u8; 32]>> }

fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] { let mut s = Sha256::new(); s.update(a); s.update(b); s.finalize() }

pub fn commit(x: &[Fp], p: &Params) -> Committed {
    let rows = x.len().div_ceil(p.k);
    let mut codewords = vec![Fp::ZERO; rows * p.n];
    let nthreads = crate::gkr::threads();
    let chunk = rows.div_ceil(nthreads).max(1);
    std::thread::scope(|s| {
        for (ci, cw) in codewords.chunks_mut(chunk * p.n).enumerate() {
            s.spawn(move || {
                for (ri, out) in cw.chunks_mut(p.n).enumerate() {
                    let i = ci * chunk + ri;
                    let mut row = vec![Fp::ZERO; p.k];
                    let lo = i * p.k;
                    let hi = ((i + 1) * p.k).min(x.len());
                    if lo < hi { row[..hi - lo].copy_from_slice(&x[lo..hi]); }
                    out.copy_from_slice(&encode(&row, p.n));
                }
            });
        }
    });
    let mut leaves = vec![[0u8; 32]; p.n];
    let lchunk = p.n.div_ceil(nthreads).max(1);
    std::thread::scope(|s| {
        for (ci, lv) in leaves.chunks_mut(lchunk).enumerate() {
            let codewords = &codewords;
            s.spawn(move || {
                for (jj, leaf) in lv.iter_mut().enumerate() {
                    let j = ci * lchunk + jj;
                    let mut h = Sha256::new();
                    for i in 0..rows { h.update(&codewords[i * p.n + j].to_bytes()); }
                    *leaf = h.finalize();
                }
            });
        }
    });
    let mut tree = vec![leaves];
    while tree.last().unwrap().len() > 1 {
        let prev = tree.last().unwrap();
        let next: Vec<[u8; 32]> = (0..prev.len() / 2).map(|i| hash_pair(&prev[2 * i], &prev[2 * i + 1])).collect();
        tree.push(next);
    }
    Committed { rows, codewords, tree }
}

impl Committed {
    pub fn root(&self) -> [u8; 32] { self.tree.last().unwrap()[0] }
    pub fn column(&self, j: usize, n: usize) -> Vec<Fp> { (0..self.rows).map(|i| self.codewords[i * n + j]).collect() }
    pub fn path(&self, j: usize) -> Vec<[u8; 32]> {
        let mut out = Vec::new();
        let mut idx = j;
        for level in &self.tree[..self.tree.len() - 1] { out.push(level[idx ^ 1]); idx /= 2; }
        out
    }
}

fn verify_path(root: &[u8; 32], j: usize, column: &[Fp], path: &[[u8; 32]]) -> bool {
    let mut h = Sha256::new();
    for v in column { h.update(&v.to_bytes()); }
    let mut cur = h.finalize();
    let mut idx = j;
    for sib in path {
        cur = if idx & 1 == 0 { hash_pair(&cur, sib) } else { hash_pair(sib, &cur) };
        idx /= 2;
    }
    &cur == root
}

fn open_set(tr: &mut Transcript, n: usize, t: usize) -> Vec<usize> {
    let mut set = Vec::with_capacity(t);
    while set.len() < t {
        let c = tr.challenge();
        let j = (c.0[0].0 % n as u64) as usize;
        if !set.contains(&j) { set.push(j); }
    }
    set
}

fn row_coeffs(tr: &mut Transcript, rows: usize) -> Vec<Ext> {
    let r = tr.challenge();
    let mut v = vec![Ext::ONE];
    for i in 1..rows { v.push(v[i - 1] * r); }
    v
}

/// Prove ⟨a, X⟩ = b (a is public, length rows·k, zero-padded) and proximity; appends to `out` and to the byte
/// stream `bytes` (opened columns and paths).  Message order: w (k), q coefficients (2k-1), then columns+paths.
pub fn prove_open(cm: &Committed, mut a: Vec<Ext>, p: &Params, tr: &mut Transcript, out: &mut Vec<Ext>, bytes: &mut Vec<u8>) {
    let rows = cm.rows;
    a.resize(rows * p.k, Ext::ZERO);
    let r = row_coeffs(tr, rows);
    // w = Σ r_i X_i  (message form: X_i = codeword restricted to H = positions j·(n/k)); q on D: Σ_i Enc(a_i)_j U_ij.
    // Both are sums over rows: every thread accumulates its rows, the partials are added.
    let step = p.n / p.k;
    let nthreads = crate::gkr::threads().min(rows.max(1));
    let chunk = rows.div_ceil(nthreads).max(1);
    let parts: Vec<(Vec<Ext>, Vec<Ext>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nthreads).map(|t| {
            let (r, a) = (&r, &a);
            s.spawn(move || {
                let mut w = vec![Ext::ZERO; p.k];
                let mut q = vec![Ext::ZERO; p.n];
                for i in t * chunk..((t + 1) * chunk).min(rows) {
                    let cw = &cm.codewords[i * p.n..(i + 1) * p.n];
                    for m in 0..p.k { w[m] += r[i].scale(cw[m * step]); }
                    let arow = &a[i * p.k..(i + 1) * p.k];
                    if arow.iter().all(|x| x.is_zero()) { continue; }
                    let ae = encode_ext(arow, p.n);
                    for j in 0..p.n { q[j] += ae[j].scale(cw[j]); }
                }
                (w, q)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut w = vec![Ext::ZERO; p.k];
    let mut q = vec![Ext::ZERO; p.n];
    for (pw, pq) in parts {
        for (x, y) in w.iter_mut().zip(pw) { *x += y; }
        for (x, y) in q.iter_mut().zip(pq) { *x += y; }
    }
    ntt_ext(&mut q, true);
    debug_assert!(q[2 * p.k - 1..].iter().all(|x| x.is_zero()), "q degree exceeds 2k-2");
    let qc = q[..2 * p.k - 1].to_vec();
    out.extend_from_slice(&w);
    out.extend_from_slice(&qc);
    tr.absorb_exts(&w);
    tr.absorb_exts(&qc);
    let set = open_set(tr, p.n, p.t);
    for &j in &set {
        for v in cm.column(j, p.n) { bytes.extend_from_slice(&v.to_bytes()); }
        for h in cm.path(j) { bytes.extend_from_slice(&h); }
    }
}

pub fn verify_open(root: &[u8; 32], rows: usize, mut a: Vec<Ext>, b: Ext, p: &Params, tr: &mut Transcript, rd: &mut Reader, bytes: &[u8]) -> Result<(), String> {
    a.resize(rows * p.k, Ext::ZERO);
    let r = row_coeffs(tr, rows);
    let w = rd.take(p.k)?;
    let qc = rd.take(2 * p.k - 1)?;
    tr.absorb_exts(&w);
    tr.absorb_exts(&qc);
    // Σ_{ζ∈H} q(ζ) = k (c_0 + c_k)
    let sum_h = (qc[0] + qc[p.k]).scale(Fp::new(p.k as u64));
    if sum_h != b { return Err("ligero: linear functional value mismatch".into()); }
    let set = open_set(tr, p.n, p.t);
    let w_enc = encode_ext(&w, p.n);
    let omega = root_of_unity(p.n);
    // a_i(η_j) for the opened columns only: encode every nonzero row of a once (public work, parallel over rows),
    // keep the t opened positions (rows × t values, not rows × n)
    let nthreads = crate::gkr::threads().min(rows.max(1));
    let chunk = rows.div_ceil(nthreads).max(1);
    let a_enc: Vec<Option<Vec<Ext>>> = {
        let parts: Vec<Vec<Option<Vec<Ext>>>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nthreads).map(|t| {
                let (a, set) = (&a, &set);
                s.spawn(move || {
                    let mut v = Vec::new();
                    for i in t * chunk..((t + 1) * chunk).min(rows) {
                        let arow = &a[i * p.k..(i + 1) * p.k];
                        if arow.iter().all(|x| x.is_zero()) { v.push(None); } else {
                            let ae = encode_ext(arow, p.n);
                            v.push(Some(set.iter().map(|&j| ae[j]).collect()));
                        }
                    }
                    v
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        parts.into_iter().flatten().collect()
    };
    let col_bytes = rows * 8;
    let depth = p.n.trailing_zeros() as usize;
    let per = col_bytes + 32 * depth;
    if bytes.len() != per * set.len() { return Err("ligero: opening size".into()); }
    for (oi, &j) in set.iter().enumerate() {
        let blob = &bytes[oi * per..(oi + 1) * per];
        let column: Vec<Fp> = (0..rows).map(|i| Fp::new(u64::from_le_bytes(blob[8 * i..8 * i + 8].try_into().unwrap()))).collect();
        let path: Vec<[u8; 32]> = (0..depth).map(|d| blob[col_bytes + 32 * d..col_bytes + 32 * d + 32].try_into().unwrap()).collect();
        if !verify_path(root, j, &column, &path) { return Err(format!("ligero: merkle path {j}")); }
        let mut lhs = Ext::ZERO;
        for i in 0..rows { lhs += r[i].scale(column[i]); }
        if lhs != w_enc[j] { return Err(format!("ligero: proximity test at column {j}")); }
        let eta = Ext::from_base(omega.pow(j as u64));
        let mut qv = Ext::ZERO;
        for c in qc.iter().rev() { qv = qv * eta + *c; }
        let mut rhs = Ext::ZERO;
        for i in 0..rows { if let Some(ae) = &a_enc[i] { rhs += ae[oi].scale(column[i]); } }
        if qv != rhs { return Err(format!("ligero: linear test at column {j}")); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ntt_roundtrip_and_encode_consistency() {
        let mut v: Vec<Fp> = (0..16u64).map(|i| Fp::new(i * i + 3)).collect();
        let orig = v.clone();
        ntt(&mut v, false);
        ntt(&mut v, true);
        assert_eq!(v, orig);
        // H ⊂ D: message symbol m sits at codeword position m * (n/k)
        let enc = encode(&orig, 64);
        for m in 0..16 { assert_eq!(enc[m * 4], orig[m]); }
    }
}
