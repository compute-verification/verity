//! The bit link's prime side in sigma form (PROTOCOL.md §17, `gpu/link.py`; the red team's S1, NON_ZK_PROOF only),
//! against the live Flock session of route (a) (flock-link, `verity/ml/flock/live`): after the GKR segments,
//!
//!   root_F               SHA-256(ROOT_F_TAG || transcript state || ctr): the prime proof's commitment (Ligero root and
//!                        every GKR message) that the session's `Commit` carries; the record holds its SHA-256
//!   r_0, r_1 in GF(2^128)^m, y_0, y_1   the session's two link points (the verifier's OS coins, drawn after root_F and
//!                        root_B) and the binary side's claims y_k = z^(r_k), read from the session record (`link.txt`),
//!                        absorbed as TAG || Σ || SHA-256(root_B) || points || y
//!   sigma_t, t < 256     prover messages `(sigma_t, 0, 0, 0, 0, 0)`: plane t < 128 of point 0, t - 128 of point 1;
//!                        0 <= sigma_t <= n_cells, sigma_t = bit_t(y_0 || y_1) mod 2
//!   rho                  one extension challenge; `a[cell q] += sum_t rho^t bit_t(eq(r, pos(q)))`,
//!                        `b += sum_t rho^t sigma_t`
//!
//! GF(2^128) is Flock's: GF(2)[x] / (x^128 + x^7 + x^2 + x + 1), bit t = the coefficient of x^t. A [`G`] holds the
//! pair (point 0, point 1) as four limbs, so `eq(r, i) = prod_j (i_j ? r_j : 1 + r_j)` is taken limb pair by limb
//! pair and its 256 bits are the 256 planes. Lambda_F (`derive`) comes from the unit circuit's column names,
//! booleanity and recomposition constraints and the checker-v2 layout, never from the statement, and is checked to be a
//! bijection onto [0, n_pos). Σ's preimage (`sigma.txt`) is checked against this statement: the scheme, the point
//! choice, N, m, the SHA-256 of `commitment.txt` and of the leaf digests the public words carry.

use crate::circuit::{log2ceil, Circuit, Wire};
use crate::commitments::{from_limbs, unhex32, RowHash, Spec as CommitSpec, LIMBS};
use crate::field::{Ext, DEG, P, P64};
use crate::sha256::Sha256;
use crate::transcript::Transcript;
use std::collections::{HashMap, HashSet};

pub const TAG: &[u8] = b"verity-gkr/link/sigma-2xgf2_128/v2";
pub const ROOT_F_TAG: &[u8] = b"verity-gkr/link/root-f/v2";
pub const PLANES: usize = 256;
const LO: usize = 14;

// ---- GF(2^128) x GF(2^128) -----------------------------------------------------------------------------------------

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
    pub const ONE: G = G([1, 0, 1, 0]);
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
    /// The pair product: limbs 0-1 (point 0) and 2-3 (point 1) each in GF(2^128).
    #[inline(always)]
    pub fn mul(self, o: G) -> G {
        let (a, b) = (mul128([self.0[0], self.0[1]], [o.0[0], o.0[1]]), mul128([self.0[2], self.0[3]], [o.0[2], o.0[3]]));
        G([a[0], a[1], b[0], b[1]])
    }
}

/// Flock's GF(2^128): x^128 = x^7 + x^2 + x + 1.
#[inline(always)]
pub fn mul128(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    let mut r = [0u64; 4];
    for i in 0..2 {
        for j in 0..2 {
            let (lo, hi) = clmul(a[i], b[j]);
            r[i + j] ^= lo;
            r[i + j + 1] ^= hi;
        }
    }
    let h = [r[2], r[3]];
    let mut o = [r[0] ^ h[0], r[1] ^ h[1]];
    let mut ov = 0u64;
    for s in [1u32, 2, 7] {
        o[0] ^= h[0] << s;
        o[1] ^= (h[1] << s) | (h[0] >> (64 - s));
        ov ^= h[1] >> (64 - s);
    }
    o[0] ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    o
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
    /// `f(jb, pos(u, jb))` for `jb` in order, without the divisions of `pos`.
    #[inline(always)]
    pub fn unit_positions(&self, u: usize, mut f: impl FnMut(usize, usize)) {
        let (v, s) = (u / self.steps, u % self.steps);
        let mut jb = 0;
        for port in 0..2 {
            for jj in 0..self.k {
                let base = ((port * self.vus + v) * self.kk + self.k * s + jj) * self.bits;
                for i in 0..self.bits {
                    f(jb, base + i);
                    jb += 1;
                }
            }
        }
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
    let (mut hit, mut out) = (0usize, false);
    for u in 0..vus * steps {
        lay.unit_positions(u, |_, p| {
            if p >= n { out = true; return; }
            let (w, b) = (p >> 6, 1u64 << (p & 63));
            if seen[w] & b == 0 { seen[w] |= b; hit += 1; }
        });
        if out { return Err("link: Lambda_F leaves the linked positions".into()); }
    }
    // n_cells == n_pos cells on `hit` distinct positions
    if hit != lay.n_cells() { return Err("link: Lambda_F is not injective".into()); }
    Ok(lay)
}

// ---- the statement's link ------------------------------------------------------------------------------------------

/// The session's link context (`link.txt`, written from the Flock session record by `tools/cell_gate.py`):
/// `link session-2xgf2_128/v2`, `sigma`, `root_f_sha256`, `root_b_sha256` (hex), two `point` lines of m GF(2^128)
/// elements and one `y` line of two, each element 32 hex digits, the record's `{hi:016x}{lo:016x}`.
/// `live`: the session's prime rounds (`prime_coins.txt` beside `link.txt`), when the prime coins were live.
pub struct Spec { pub layout: Layout, pub sigma: [u8; 32], pub root_f_sha256: [u8; 32], pub root_b_sha256: [u8; 32],
                  pub points: Vec<G>, pub y: G, pub t_derive: f64, pub live: Option<crate::transcript::LiveCoins> }

fn f128(s: &str) -> Result<[u64; 2], String> {
    if s.len() != 32 || !s.bytes().all(|c| c.is_ascii_hexdigit()) { return Err(format!("link.txt: {s:?} is not a GF(2^128) element")); }
    let hi = u64::from_str_radix(&s[..16], 16).unwrap();
    let lo = u64::from_str_radix(&s[16..], 16).unwrap();
    Ok([lo, hi])
}

fn sigma_line<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    text.lines().find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix(' '))).ok_or_else(|| format!("sigma.txt: no {key} line"))
}

/// `link.txt` and `sigma.txt` in `d` against the unit circuit, `commitment.txt` and the public words (per VU: the
/// `y` words, then the x row and W column digest limbs).
pub fn load(d: &std::path::Path, text: &str, circ: &Circuit, steps: usize, vus: usize, commitment: &[u8], public: &[u32]) -> Result<Spec, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next() != Some("link session-2xgf2_128/v2") { return Err("link.txt: not a session-2xgf2_128/v2 link".into()); }
    let mut kv: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut pts: Vec<Vec<&str>> = Vec::new();
    for l in lines {
        let mut t = l.split_whitespace();
        let k = t.next().unwrap();
        if k == "point" { pts.push(t.collect()); } else if kv.insert(k, t.collect()).is_some() { return Err(format!("link.txt: {k} twice")); }
    }
    let one = |k: &str| -> Result<[u8; 32], String> {
        kv.get(k).and_then(|v| (v.len() == 1).then(|| unhex32(v[0])).flatten()).ok_or_else(|| format!("link.txt: no {k} line"))
    };
    let (sigma, root_f_sha256, root_b_sha256) = (one("sigma")?, one("root_f_sha256")?, one("root_b_sha256")?);
    let cs = CommitSpec::parse(std::str::from_utf8(commitment).map_err(|_| "commitment.txt: not UTF-8")?)?;
    if cs.hash != RowHash::Blake3 { return Err("link: the cell's scheme is blake3-keyed/row/v2 (Flock proves keyed BLAKE3 row leaves)".into()); }
    let (kk, bits) = (cs.k as usize, cs.x_bits as usize);
    if cs.w_bits as usize != bits { return Err("link: x and W word widths differ".into()); }
    let t0 = std::time::Instant::now();
    let layout = derive(circ, vus, steps, kk, bits)?;
    let t_derive = t0.elapsed().as_secs_f64();
    let m = layout.m();
    if pts.len() != 2 || pts.iter().any(|p| p.len() != m) { return Err(format!("link.txt: two points of {m} coordinates expected")); }
    let coord = |j: usize| -> Result<G, String> { let (a, b) = (f128(pts[0][j])?, f128(pts[1][j])?); Ok(G([a[0], a[1], b[0], b[1]])) };
    let points = (0..m).map(coord).collect::<Result<Vec<_>, _>>()?;
    let yv = kv.get("y").filter(|v| v.len() == 2).ok_or("link.txt: y is two GF(2^128) elements")?;
    let (y0, y1) = (f128(yv[0])?, f128(yv[1])?);
    let y = G([y0[0], y0[1], y1[0], y1[1]]);
    // Σ: its preimage hashes to the session's Σ, and every line this statement fixes agrees with it
    let st = std::fs::read(d.join("sigma.txt")).map_err(|e| format!("sigma.txt: {e}"))?;
    if Sha256::digest(&st) != sigma { return Err("link: sigma.txt does not hash to the session's Σ".into()); }
    let st = std::str::from_utf8(&st).map_err(|_| "sigma.txt: not UTF-8")?;
    let per = cs.n_y + 2 * LIMBS;
    if public.len() != vus * per { return Err(format!("link: {} public words, {} per VU expected", public.len(), per)); }
    let mut cat = Vec::with_capacity(64 * vus);
    for a in 0..2 {
        for v in 0..vus {
            let lo = v * per + cs.n_y + a * LIMBS;
            cat.extend_from_slice(&from_limbs(&public[lo..lo + LIMBS]).ok_or("link: a public digest limb is not below 2^16")?);
        }
    }
    let want = [
        ("verity/link-L/v2", String::new()),
        ("scheme", RowHash::Blake3.schema().to_string()),
        ("points", "2 gf2_128 x^128+x^7+x^2+x+1".to_string()),
        ("vus", vus.to_string()),
        ("k", kk.to_string()),
        ("bits", bits.to_string()),
        ("n", layout.n_pos().to_string()),
        ("m", m.to_string()),
        ("prime_commitment", Sha256::digest(commitment).iter().map(|b| format!("{b:02x}")).collect()),
        ("leaf_digests", format!("{} {}", 2 * vus, Sha256::digest(&cat).iter().map(|b| format!("{b:02x}")).collect::<String>())),
    ];
    if st.lines().next() != Some(want[0].0) { return Err("sigma.txt: not a verity/link-L/v2 preimage".into()); }
    for (k, v) in &want[1..] {
        if sigma_line(st, k)? != v { return Err(format!("link: Σ names {k} {:?}, this statement {v:?}", sigma_line(st, k)?)); }
    }
    let live = match std::fs::read_to_string(d.join("prime_coins.txt")) {
        Ok(t) => Some(crate::transcript::parse_coins(&t)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("prime_coins.txt: {e}")),
    };
    Ok(Spec { layout, sigma, root_f_sha256, root_b_sha256, points, y, t_derive, live })
}

// ---- the transcript part and the functional term -------------------------------------------------------------------

/// What the functional needs: the split eq tables, the byte tables of `rho^t`, and the value side `b`.
/// `tab[8 byte + v]` = `sum_{bit c of v} rho^(8 byte + c)`, padded to 8 lanes for the vector accumulate.
pub struct Term { pub lay: Layout, hi: Vec<G>, lo: Vec<G>, tab: Vec<[u32; 8]>, pub b: Ext }

/// The session's context as the prime transcript absorbs it (`gpu/link.py::context_bytes`).
pub fn context_bytes(spec: &Spec) -> Vec<u8> {
    let mut buf = TAG.to_vec();
    buf.extend_from_slice(&spec.sigma);
    buf.extend_from_slice(&spec.root_b_sha256);
    for k in 0..2 {
        for r in &spec.points {
            buf.extend_from_slice(&r.0[2 * k].to_le_bytes());
            buf.extend_from_slice(&r.0[2 * k + 1].to_le_bytes());
        }
    }
    for k in 0..2 {
        buf.extend_from_slice(&spec.y.0[2 * k].to_le_bytes());
        buf.extend_from_slice(&spec.y.0[2 * k + 1].to_le_bytes());
    }
    buf
}

/// root_F against the record, the session's context, the sigma messages and their checks, rho (`gpu/link.py::verify`),
/// then the term of `a`. `spare`: the challenges after rho with no absorb between (`verify.rs::open_coins`).
pub fn verify(spec: &Spec, tr: &mut Transcript, sigma: &[Ext], spare: usize, _threads: usize) -> Result<Term, String> {
    let lay = spec.layout;
    tr.live_at_root_f()?;
    let mut rf = ROOT_F_TAG.to_vec();
    rf.extend_from_slice(&tr.root_f_input());
    if Sha256::digest(&Sha256::digest(&rf)) != spec.root_f_sha256 {
        return Err("link: root_F in the session is not this proof's commitment (Ligero root and GKR messages)".into());
    }
    tr.absorb(&context_bytes(spec));
    let r = &spec.points;
    let lo_bits = LO.min(r.len());
    let lo = eq_table(&r[..lo_bits]);
    let hi = eq_table(&r[lo_bits..]);
    let y = spec.y;
    if sigma.len() != PLANES { return Err("link: sigma count".into()); }
    for (t, s) in sigma.iter().enumerate() {
        if s.0[1..].iter().any(|&c| c != 0) || s.0[0] as usize > lay.n_cells() {
            return Err(format!("link: sigma_{t} is not a canonical integer in [0, {}]", lay.n_cells()));
        }
        if (s.0[0] as u64 & 1) != y.bit(t) { return Err(format!("link: sigma_{t} parity disagrees with the binary side's y")); }
    }
    tr.absorb_exts(sigma);
    tr.reserve(1 + spare);
    let rho = tr.challenge();
    let mut pw = Vec::with_capacity(PLANES);
    let mut g = Ext::ONE;
    for _ in 0..PLANES { pw.push(g); g = g * rho; }
    let mut b = Ext::ZERO;
    for t in 0..PLANES { b += pw[t].scale(sigma[t].0[0]); }
    let mut tab = vec![[0u32; 8]; 32 * 256];
    for byte in 0..32 {
        for v in 1..256usize {
            let low = v & v.wrapping_neg();
            let e = Ext(std::array::from_fn(|d| tab[byte * 256 + (v ^ low)][d]));
            let x = e + pw[8 * byte + low.trailing_zeros() as usize];
            tab[byte * 256 + v][..DEG].copy_from_slice(&x.0);
        }
    }
    Ok(Term { lay, hi, lo, tab, b })
}

impl Term {
    /// `sum_t rho^t bit_t(e)` (lanes `DEG..8` stay 0), canonical.
    #[inline(always)]
    fn dot(&self, e: &G) -> [u32; 8] {
        let mut acc = [0u32; 8];
        for (li, &limb) in e.0.iter().enumerate() {
            let mut x = limb;
            for bi in 0..8 {
                let t = &self.tab[(li * 8 + bi) * 256 + (x & 255) as usize];
                for d in 0..8 {
                    let s = acc[d] + t[d];
                    acc[d] = s.min(s.wrapping_sub(P));
                }
                x >>= 8;
            }
        }
        acc
    }

    /// `out[col0 + jb] += sum_t rho^t bit_t(eq(r, pos(u, jb)))` for unit `u` of segment 0.
    #[inline]
    pub fn add_row(&self, u: usize, out: &mut [Ext]) {
        let lay = &self.lay;
        let mask = self.lo.len() - 1;
        let lo_bits = self.lo.len().trailing_zeros();
        let dst = &mut out[lay.col0..lay.col0 + lay.per_unit()];
        lay.unit_positions(u, |jb, p| {
            let acc = self.dot(&self.hi[p >> lo_bits].mul(self.lo[p & mask]));
            dst[jb] += Ext(std::array::from_fn(|d| acc[d]));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gmul_ref(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // bit-serial shift-and-add, the reference `gpu/link.py::gmul128` (x^128 = x^7 + x^2 + x + 1)
        let (mut x, mut r) = (a, [0u64; 2]);
        for t in 0..128 {
            if (b[t >> 6] >> (t & 63)) & 1 == 1 { r = [r[0] ^ x[0], r[1] ^ x[1]]; }
            let top = x[1] >> 63;
            x = [x[0] << 1, (x[1] << 1) | (x[0] >> 63)];
            if top == 1 { x[0] ^= 0x87; }
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
            let (ga, gb) = (G::from_bytes(&a), G::from_bytes(&b));
            let p = ga.mul(gb);
            assert_eq!([p.0[0], p.0[1]], gmul_ref([ga.0[0], ga.0[1]], [gb.0[0], gb.0[1]]));
            assert_eq!([p.0[2], p.0[3]], gmul_ref([ga.0[2], ga.0[3]], [gb.0[2], gb.0[3]]));
        }
        let top = [u64::MAX, u64::MAX];
        assert_eq!(mul128(top, top), gmul_ref(top, top));
        // Flock's F128 (ghash): x^127 * x = x^7 + x^2 + x + 1
        assert_eq!(mul128([0, 1 << 63], [2, 0]), [0x87, 0]);
    }

    #[test]
    fn unit_positions_is_pos() {
        for (k, bits, steps, vus) in [(16usize, 16usize, 3usize, 5usize), (32, 8, 2, 3)] {
            let lay = Layout { vus, steps, kk: k * steps, k, bits, col0: 0, ncols: 0 };
            for u in 0..vus * steps {
                let mut n = 0;
                lay.unit_positions(u, |jb, p| { assert_eq!(p, lay.pos(u, jb)); assert_eq!(jb, n); n += 1; });
                assert_eq!(n, lay.per_unit());
            }
        }
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
