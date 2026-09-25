//! The bit link's prime side in sigma form (PROTOCOL.md §17, `gpu/link.py`; the red team's S1, NON_ZK_PROOF only),
//! re-derived here: after the GKR segments, with the Ligero root and the binary proof's root `root_b` absorbed,
//!
//!   r in GF(2^256)^m     `absorb(TAG || root_b)`, then m = log2ceil(n_pos) whole-digest coins (`challenge_bytes`)
//!   y = z^(r)            the binary side's claim; here a STAND-IN computed from `x.bin` / `w.bin`, which must hash to
//!                        the statement's public row digests (`sha256/row/v1`), until the binary proof is wired
//!   sigma_t, t < 256     prover messages `(sigma_t, 0, 0, 0, 0, 0)`: 0 <= sigma_t <= n_cells, sigma_t = bit_t(y) mod 2
//!   rho                  one extension challenge; `a[cell q] += sum_t rho^t bit_t(eq(r, pos(q)))`,
//!                        `b += sum_t rho^t sigma_t`
//!
//! GF(2^256) = GF(2)[x] / (x^256 + x^10 + x^5 + x^2 + 1), bit t = the coefficient of x^t, little-endian bytes.
//! `eq(r, i) = prod_j (i_j ? r_j : 1 + r_j)` with bit j of i against r_j, evaluated as eq_hi(i >> LO) eq_lo(i & mask).
//! Lambda_F (`derive`) comes from the unit circuit's column names, booleanity and recomposition constraints and the
//! checker-v2 layout, never from the statement, and is checked to be a bijection onto [0, n_pos).

use crate::circuit::{log2ceil, Circuit, Wire};
use crate::commitments::{from_limbs, sha256_row_digest, Spec as CommitSpec, LIMBS, ROLE_W, ROLE_X};
use crate::field::{Ext, DEG, P, P64};
use crate::sha256::Sha256;
use crate::transcript::Transcript;
use std::collections::{HashMap, HashSet};

pub const TAG: &[u8] = b"verity-gkr/link/sigma-gf2_256/v1";
pub const PLANES: usize = 256;
const LO: usize = 14;

// ---- GF(2^256) --------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct G(pub [u64; 4]);

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline(always)]
fn clmul(a: u64, b: u64) -> (u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let r = _mm_clmulepi64_si128(_mm_set_epi64x(0, a as i64), _mm_set_epi64x(0, b as i64), 0);
        (_mm_cvtsi128_si64(r) as u64, _mm_cvtsi128_si64(_mm_unpackhi_epi64(r, r)) as u64)
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
#[inline(always)]
fn clmul(a: u64, b: u64) -> (u64, u64) {
    let (mut lo, mut hi) = (0u64, 0u64);
    for i in 0..64 {
        if (b >> i) & 1 == 1 {
            lo ^= a << i;
            if i > 0 { hi ^= a >> (64 - i); }
        }
    }
    (lo, hi)
}

impl G {
    pub const ONE: G = G([1, 0, 0, 0]);
    pub fn from_bytes(b: &[u8; 32]) -> G { G(std::array::from_fn(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))) }
    pub fn to_bytes(self) -> [u8; 32] {
        let mut o = [0u8; 32];
        for i in 0..4 { o[8 * i..8 * i + 8].copy_from_slice(&self.0[i].to_le_bytes()); }
        o
    }
    #[inline(always)]
    pub fn xor(self, o: G) -> G { G(std::array::from_fn(|i| self.0[i] ^ o.0[i])) }
    #[inline(always)]
    pub fn bit(&self, t: usize) -> u64 { (self.0[t >> 6] >> (t & 63)) & 1 }
    #[inline(always)]
    pub fn mul(self, o: G) -> G {
        let mut r = [0u64; 8];
        for i in 0..4 {
            for j in 0..4 {
                let (lo, hi) = clmul(self.0[i], o.0[j]);
                r[i + j] ^= lo;
                r[i + j + 1] ^= hi;
            }
        }
        // x^256 = x^10 + x^5 + x^2 + 1: fold the high half H in as H (1 + x^2 + x^5 + x^10); the < 2^10 overflow of
        // the shifted copies folds once more into the low limb
        let h = [r[4], r[5], r[6], r[7]];
        let mut o = [r[0] ^ h[0], r[1] ^ h[1], r[2] ^ h[2], r[3] ^ h[3]];
        let mut ov = 0u64;
        for s in [2u32, 5, 10] {
            o[0] ^= h[0] << s;
            for i in 1..4 { o[i] ^= (h[i] << s) | (h[i - 1] >> (64 - s)); }
            ov ^= h[3] >> (64 - s);
        }
        o[0] ^= ov ^ (ov << 2) ^ (ov << 5) ^ (ov << 10);
        G(o)
    }
}

/// `eq(r, i)` for every `i < 2^len(r)`, bit j of i against `r[j]` (`gpu/link.py::eq_table`).
pub fn eq_table(r: &[G]) -> Vec<G> {
    let mut t = vec![G::default(); 1 << r.len()];
    t[0] = G::ONE;
    for (j, rj) in r.iter().enumerate() {
        let h = 1 << j;
        for i in 0..h {
            let v = t[i].mul(*rj);
            t[i + h] = v;
            t[i] = t[i].xor(v);
        }
    }
    t
}

// ---- Lambda_F -----------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout { pub vus: usize, pub steps: usize, pub kk: usize, pub k: usize, pub bits: usize, pub col0: usize, pub ncols: usize }

impl Layout {
    pub fn per_unit(&self) -> usize { 2 * self.k * self.bits }
    pub fn n_cells(&self) -> usize { self.vus * self.steps * self.per_unit() }
    pub fn n_pos(&self) -> usize { 2 * self.vus * self.kk * self.bits }
    pub fn m(&self) -> usize { log2ceil(self.n_pos()) }
    /// The position of link cell `jb` of unit `u`: link column `col0 + j bits + i` is bit `i` of operand column
    /// `5 + j`, i.e. of x row `v` word `k s + j` (`j < k`) or of W column `v` word `k s + j - k`.
    #[inline(always)]
    pub fn pos(&self, u: usize, jb: usize) -> usize {
        let (v, s) = (u / self.steps, u % self.steps);
        let (j, i) = (jb / self.bits, jb % self.bits);
        let (port, jj) = (j / self.k, j % self.k);
        ((port * self.vus + v) * self.kk + self.k * s + jj) * self.bits + i
    }
}

/// Lambda_F of a checker-v2 unit circuit extended by `tools/link_stub.extend_unit` (`gpu/link.py::derive`).
pub fn derive(circ: &Circuit, vus: usize, steps: usize, kk: usize, bits: usize) -> Result<Layout, String> {
    let names = &circ.col_names;
    let ys = names.iter().position(|n| n == "y.s").ok_or("link: not a checker-v2 unit (no y.s column)")?;
    let k = ys.saturating_sub(5) / 2;
    if k == 0 || steps * k != kk || !(bits == 8 || bits == 16) { return Err(format!("link: layout k = {k}, steps = {steps}, K = {kk}, bits = {bits}")); }
    let nb = 2 * k * bits;
    if circ.ncols < nb + 5 + 2 * k { return Err("link: too few columns".into()); }
    let col0 = circ.ncols - nb;
    let want = (0..2 * k).flat_map(move |j| (0..bits).map(move |i| format!("link.b{}.{i}", 5 + j)));
    if !names[col0..].iter().cloned().eq(want) || names[..col0].iter().any(|n| n.starts_with("link.")) {
        return Err("link: the link columns are not the last 2 k bits columns, in operand-column order".into());
    }
    let mut colwire: HashMap<u32, u32> = HashMap::new();
    let mut sq: HashMap<u32, Vec<u32>> = HashMap::new();
    for (wi, w) in circ.wires.iter().enumerate() {
        match w {
            Wire::Input(c) => { colwire.insert(*c, wi as u32); }
            Wire::Prod { a, b, .. } => {
                if a.terms.len() == 1 && a.terms[0].coef == 1 && a.konst == 0 {
                    let x = a.terms[0].idx;
                    if b.terms.len() == 1 && b.terms[0].idx == x && b.terms[0].coef == 1 && b.konst == P - 1 { sq.entry(x).or_default().push(wi as u32); }
                }
            }
        }
    }
    let zero: HashSet<(Vec<(u32, u32)>, u32)> = circ.asserts.iter().map(|(_, l)| {
        let mut t: Vec<(u32, u32)> = l.terms.iter().map(|t| (t.idx, t.coef)).collect();
        t.sort_unstable();
        (t, l.konst)
    }).collect();
    let cw = |c: usize| colwire.get(&(c as u32)).copied().ok_or_else(|| format!("link: column {c} is not an input wire"));
    for j in 0..2 * k {
        let mut rec = vec![(cw(5 + j)?, 1u32)];
        for i in 0..bits {
            let bw = cw(col0 + j * bits + i)?;
            let boolean = sq.get(&bw).map(|ws| ws.iter().any(|&pw| zero.contains(&(vec![(pw, 1)], 0)))).unwrap_or(false);
            if !boolean { return Err(format!("link: no booleanity constraint on link.b{}.{i}", 5 + j)); }
            rec.push((bw, ((P64 - (1u64 << i) % P64) % P64) as u32));
        }
        rec.sort_unstable();
        if !zero.contains(&(rec, 0)) { return Err(format!("link: no recomposition constraint for operand column {}", 5 + j)); }
    }
    let lay = Layout { vus, steps, kk, k, bits, col0, ncols: circ.ncols };
    if lay.n_cells() != lay.n_pos() || lay.n_cells() as u64 > P64 - 1 {
        return Err(format!("link: {} cells for {} positions (p - 1 = {})", lay.n_cells(), lay.n_pos(), P - 1));
    }
    let n = lay.n_pos();
    let mut seen = vec![0u64; n.div_ceil(64)];
    let mut hit = 0usize;
    for u in 0..vus * steps {
        for jb in 0..lay.per_unit() {
            let p = lay.pos(u, jb);
            if p >= n { return Err("link: Lambda_F leaves the linked positions".into()); }
            let (w, b) = (p >> 6, 1u64 << (p & 63));
            if seen[w] & b == 0 { seen[w] |= b; hit += 1; }
        }
    }
    // n_cells == n_pos cells on `hit` distinct positions
    if hit != lay.n_cells() { return Err("link: Lambda_F is not injective".into()); }
    Ok(lay)
}

// ---- the statement's link ------------------------------------------------------------------------------------------

/// `link.txt` (`link sigma-gf2_256/v1`, `root_b <hex>`) with the derived layout and the stand-in operand words
/// (x rows, then W columns, `vus x K` each, in position order).
pub struct Spec { pub layout: Layout, pub root_b: [u8; 32], pub words: Vec<u16>, pub t_derive: f64 }

/// The STAND-IN binary root until the binary proof is wired (`gpu/link.py::standin_root_b`).
pub fn standin_root_b(commitment_text: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TAG);
    h.update(b"/root-b-standin/");
    h.update(commitment_text);
    h.finalize()
}

/// `link.txt`, `x.bin`, `w.bin` in `d` against the unit circuit, `commitment.txt` and the public words (per VU: the
/// `y` words, then the x row and W column digest limbs).
pub fn load(d: &std::path::Path, text: &str, circ: &Circuit, steps: usize, vus: usize, commitment: &[u8], public: &[u32]) -> Result<Spec, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next() != Some("link sigma-gf2_256/v1") { return Err("link.txt: not a sigma-gf2_256/v1 link".into()); }
    let rb = lines.next().and_then(|l| l.strip_prefix("root_b ")).and_then(crate::commitments::unhex32).ok_or("link.txt: no root_b line")?;
    if lines.next().is_some() { return Err("link.txt: trailing lines".into()); }
    if rb != standin_root_b(commitment) { return Err("link.txt: root_b is not the stand-in root of commitment.txt".into()); }
    let cs = CommitSpec::parse(std::str::from_utf8(commitment).map_err(|_| "commitment.txt: not UTF-8")?)?;
    let (kk, bits) = (cs.k as usize, cs.x_bits as usize);
    if cs.w_bits as usize != bits { return Err("link: x and W word widths differ".into()); }
    let t0 = std::time::Instant::now();
    let layout = derive(circ, vus, steps, kk, bits)?;
    let t_derive = t0.elapsed().as_secs_f64();
    let per = cs.n_y + 2 * LIMBS;
    if public.len() != vus * per { return Err(format!("link: {} public words, {} per VU expected", public.len(), per)); }
    let mut words = Vec::with_capacity(2 * vus * kk);
    for (a, name, role) in [(0usize, "x.bin", ROLE_X), (1, "w.bin", ROLE_W)] {
        let bytes = std::fs::read(d.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if bytes.len() != 2 * vus * kk { return Err(format!("{name}: {} bytes, {} expected", bytes.len(), 2 * vus * kk)); }
        let ws: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if ws.iter().any(|&w| (w as usize) >> bits != 0) { return Err(format!("{name}: a word exceeds {bits} bits")); }
        for v in 0..vus {
            let row = &ws[v * kk..(v + 1) * kk];
            let raw: Vec<u8> = if bits == 16 { row.iter().flat_map(|w| w.to_le_bytes()).collect() } else { row.iter().map(|&w| w as u8).collect() };
            let got = sha256_row_digest(role, bits as u8, kk as u32, &raw);
            let lo = v * per + cs.n_y + a * LIMBS;
            let want = from_limbs(&public[lo..lo + LIMBS]).ok_or("link: a public digest limb is not below 2^16")?;
            if got != want { return Err(format!("link: {name} row {v} does not hash to the statement's digest")); }
        }
        words.extend_from_slice(&ws);
    }
    Ok(Spec { layout, root_b: rb, words, t_derive })
}

// ---- the transcript part and the functional term -------------------------------------------------------------------

/// What the functional needs: the split eq tables, the byte tables of `rho^t`, and the value side `b`.
pub struct Term { pub lay: Layout, hi: Vec<G>, lo: Vec<G>, tab: Vec<[u32; DEG]>, pub b: Ext }

fn stand_in_y(spec: &Spec, hi: &[G], lo: &[G], threads: usize) -> G {
    let lay = &spec.layout;
    let (n, bits) = (lay.n_pos(), lay.bits);
    let nlo = lo.len();
    let nh = n.div_ceil(nlo);
    let th = threads.max(1);
    let chunk = nh.div_ceil(th);
    let parts: Vec<G> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..th).map(|t| s.spawn(move || {
            let mut acc = G::default();
            for h in t * chunk..((t + 1) * chunk).min(nh) {
                let mut inner = G::default();
                for l in 0..nlo.min(n - h * nlo) {
                    let p = h * nlo + l;
                    if (spec.words[p / bits] >> (p % bits)) & 1 == 1 { inner = inner.xor(lo[l]); }
                }
                acc = acc.xor(hi[h].mul(inner));
            }
            acc
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    parts.into_iter().fold(G::default(), G::xor)
}

/// Points, the stand-in y, the sigma messages and their checks, rho (`gpu/link.py::verify`), then the term of `a`.
pub fn verify(spec: &Spec, tr: &mut Transcript, sigma: &[Ext], threads: usize) -> Result<Term, String> {
    let lay = spec.layout;
    let mut buf = TAG.to_vec();
    buf.extend_from_slice(&spec.root_b);
    tr.absorb(&buf);
    let r: Vec<G> = (0..lay.m()).map(|_| G::from_bytes(&tr.challenge_bytes())).collect();
    let lo_bits = LO.min(r.len());
    let lo = eq_table(&r[..lo_bits]);
    let hi = eq_table(&r[lo_bits..]);
    let y = stand_in_y(spec, &hi, &lo, threads);
    let mut yb = b"link/y".to_vec();
    yb.extend_from_slice(&y.to_bytes());
    tr.absorb(&yb);
    if sigma.len() != PLANES { return Err("link: sigma count".into()); }
    for (t, s) in sigma.iter().enumerate() {
        if s.0[1..].iter().any(|&c| c != 0) || s.0[0] as usize > lay.n_cells() {
            return Err(format!("link: sigma_{t} is not a canonical integer in [0, {}]", lay.n_cells()));
        }
        if (s.0[0] as u64 & 1) != y.bit(t) { return Err(format!("link: sigma_{t} parity disagrees with the binary side's y")); }
    }
    tr.absorb_exts(sigma);
    let rho = tr.challenge();
    let mut pw = Vec::with_capacity(PLANES);
    let mut g = Ext::ONE;
    for _ in 0..PLANES { pw.push(g); g = g * rho; }
    let mut b = Ext::ZERO;
    for t in 0..PLANES { b += pw[t].scale(sigma[t].0[0]); }
    let mut tab = vec![[0u32; DEG]; 32 * 256];
    for byte in 0..32 {
        for v in 1..256usize {
            let low = v & v.wrapping_neg();
            let e = tab[byte * 256 + (v ^ low)];
            let add = pw[8 * byte + low.trailing_zeros() as usize];
            tab[byte * 256 + v] = (Ext(e) + add).0;
        }
    }
    Ok(Term { lay, hi, lo, tab, b })
}

impl Term {
    /// `out[col0 + jb] += sum_t rho^t bit_t(eq(r, pos(u, jb)))` for unit `u` of segment 0.
    #[inline]
    pub fn add_row(&self, u: usize, out: &mut [Ext]) {
        let lay = &self.lay;
        let mask = self.lo.len() - 1;
        let lo_bits = self.lo.len().trailing_zeros();
        for jb in 0..lay.per_unit() {
            let p = lay.pos(u, jb);
            let e = self.hi[p >> lo_bits].mul(self.lo[p & mask]).to_bytes();
            let mut acc = [0u64; DEG];
            for (byte, &v) in e.iter().enumerate() {
                let t = &self.tab[byte * 256 + v as usize];
                for d in 0..DEG { acc[d] += t[d] as u64; }
            }
            out[lay.col0 + jb] += Ext(std::array::from_fn(|d| (acc[d] % P64) as u32));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gmul_ref(a: &[u8; 32], b: &[u8; 32]) -> G {
        // bit-serial shift-and-add, the reference `gpu/link.py::gmul`
        let (mut x, y) = (G::from_bytes(a), G::from_bytes(b));
        let mut r = G::default();
        for t in 0..256 {
            if y.bit(t) == 1 { r = r.xor(x); }
            let top = x.0[3] >> 63;
            x = G([x.0[0] << 1, (x.0[1] << 1) | (x.0[0] >> 63), (x.0[2] << 1) | (x.0[1] >> 63), (x.0[3] << 1) | (x.0[2] >> 63)]);
            if top == 1 { x.0[0] ^= 0x425; }
        }
        r
    }

    #[test]
    fn gf_mul_matches_shift_and_add() {
        let mut s = [7u8; 32];
        for it in 0..200 {
            let a = Sha256::digest(&s);
            s[0] = it as u8;
            let b = Sha256::digest(&a);
            s = b;
            assert_eq!(G::from_bytes(&a).mul(G::from_bytes(&b)), gmul_ref(&a, &b));
        }
        let top = { let mut t = [0u8; 32]; t[31] = 0xff; t };
        assert_eq!(G::from_bytes(&top).mul(G::from_bytes(&top)), gmul_ref(&top, &top));
    }

    #[test]
    fn eq_table_sums_to_one_and_splits() {
        let r: Vec<G> = (0..6u8).map(|i| G::from_bytes(&Sha256::digest(&[i]))).collect();
        let t = eq_table(&r);
        assert_eq!(t.iter().fold(G::default(), |a, &x| a.xor(x)), G::ONE);
        let (lo, hi) = (eq_table(&r[..2]), eq_table(&r[2..]));
        for i in 0..64 { assert_eq!(t[i], hi[i >> 2].mul(lo[i & 3])); }
    }
}
