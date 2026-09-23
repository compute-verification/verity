//! Verifier for the `ligerito-design` reference protocol (`backends/direct/ligerito/ref.py`, format `ligerito-ref/v1`).
//!
//! `ref.py` is the SPEC of the campaign's Ligerito PCS and differs from the `proto/pcs.py` dialect in [`crate::pcs`]:
//! mixed-radix indices with digit 0 least significant and an optional TERNARY digit 0 (tall dimension `3 * 2^k`); the
//! statement is a tensor claim `<f, w> = v` (`eq` on binary digits, Lagrange-3 on the ternary one); codeword rows in
//! NATURAL order (`row s = sum_r M[r, c] omega_n^{s r}`, no 4-step permutation); Merkle leaves = the row as little-endian
//! u32 words, ELEMENT-major for extension rows; sumcheck messages are the 3 COEFFICIENTS `(c0, c1, c2)` of the round
//! polynomial, low column digit first; every opened row of level `i` becomes the tensor claim `<(omega^{s r})_r, v_i>`
//! with weights `(1, omega^s, omega^{2s})` on the ternary digit; the query set and all claims on `v_i` are batched with
//! one fresh `alpha` per claim; and its Fiat-Shamir instance is BLAKE3 only ([`FsBlake3`]).  The coin order is the
//! transcript: `absorb(root_i)`; for `i >= 2`: `indices(S_{i-1})`, `absorb(rows || siblings)`, `challenge(1 + |S|)`;
//! then per column digit `absorb(c0 c1 c2)`, `challenge(1)`; after level `L`: `absorb(v_L)`, `indices(S_L)`,
//! `absorb(rows || siblings)`.
//!
//! The JSON fixtures `python -m backends.direct.ligerito.ref` prints are accepted bit for bit by [`load_fixtures`] +
//! [`verify`]; [`apply_ops`] replays the declarative tampers of `fixtures/ref/negatives.json` so the Rust verdict is
//! compared with `ref.verify`'s on every one.  One deliberate divergence: `ref.py` reduces row entries mod p when hashing
//! (so `x + p` verifies like `x`); this verifier rejects any non-canonical field element.

use crate::field::{pow, root_of_unity, Ext, EXT_DEG, P};
use crate::hash::{blake3, blake3_xof, hex};
use crate::json::Json;
use crate::merkle::{self, Digest};

// ------------------------------------------------------------------------------------------------- coins -------

/// The four-method coin protocol `ref.py` speaks (its `FSCoins`, or the proto's `LiveCoins`).
pub trait RefCoins {
    fn absorb(&mut self, label: &[u8], data: &[u8]);
    fn challenge(&mut self, label: &[u8], n: usize) -> Vec<Ext>;
    fn indices(&mut self, label: &[u8], domain: usize, count: usize) -> Vec<usize>;
    fn rounds(&self) -> usize;
}

/// `ref.FSCoins`: `state <- BLAKE3(state || u32 |label| || label || u64 |data| || data)` on absorb; a challenge of `n`
/// elements is the BLAKE3 XOF of `state || label || 0x01 || u32 n` read as `6n` little-endian u64 words reduced mod p;
/// `indices` reads u64 words mod `domain` from the XOF of `state || label || 0x03 || u32 domain || u32 count` (a whole
/// fresh pass with twice the words when a pass leaves the set short); every coin then advances
/// `state <- BLAKE3(state || label || 0x02)`.
pub struct FsBlake3 {
    state: [u8; 32],
    rounds: usize,
}

impl FsBlake3 {
    pub const DEFAULT_SEED: &'static [u8] = b"ligerito-ref/v1";

    pub fn new(seed: &[u8]) -> FsBlake3 {
        FsBlake3 { state: blake3(seed), rounds: 0 }
    }

    fn advance(&mut self, label: &[u8]) {
        let mut buf = Vec::with_capacity(32 + label.len() + 1);
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(label);
        buf.push(2);
        self.state = blake3(&buf);
        self.rounds += 1;
    }
}

impl RefCoins for FsBlake3 {
    fn absorb(&mut self, label: &[u8], data: &[u8]) {
        let mut buf = Vec::with_capacity(32 + 4 + label.len() + 8 + data.len());
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(&(label.len() as u32).to_le_bytes());
        buf.extend_from_slice(label);
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
        self.state = blake3(&buf);
    }

    fn challenge(&mut self, label: &[u8], n: usize) -> Vec<Ext> {
        let mut buf = Vec::with_capacity(32 + label.len() + 5);
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(label);
        buf.push(1);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        let raw = blake3_xof(&buf, 8 * EXT_DEG * n);
        let out = (0..n)
            .map(|i| {
                let mut c = [0u32; EXT_DEG];
                for d in 0..EXT_DEG {
                    let o = 8 * (i * EXT_DEG + d);
                    c[d] = (u64::from_le_bytes(raw[o..o + 8].try_into().unwrap()) % P as u64) as u32;
                }
                Ext(c)
            })
            .collect();
        self.advance(label);
        out
    }

    fn indices(&mut self, label: &[u8], domain: usize, count: usize) -> Vec<usize> {
        assert!(count <= domain, "indices: {count} from a domain of {domain}");
        let mut buf = Vec::with_capacity(32 + label.len() + 9);
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(label);
        buf.push(3);
        buf.extend_from_slice(&(domain as u32).to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        let mut need = 2 * count + 8;
        let mut out: Vec<usize> = Vec::new();
        while out.len() < count {
            let raw = blake3_xof(&buf, 8 * need);
            out.clear();
            for ch in raw.chunks_exact(8) {
                let v = (u64::from_le_bytes(ch.try_into().unwrap()) % domain as u64) as usize;
                if !out.contains(&v) {
                    out.push(v);
                }
                if out.len() == count {
                    break;
                }
            }
            need *= 2;
        }
        self.advance(label);
        out
    }

    fn rounds(&self) -> usize {
        self.rounds
    }
}

// ------------------------------------------------------------------------------------------------- tensors -----

/// A tensor over a digit range: one weight vector (length = the digit's radix) per digit, digit 0 first; it denotes
/// the vector `x -> prod_j w_j[x_j]` with `x = d_0 + rad_0 (d_1 + rad_1 (...))`.
pub type Tensor = Vec<Vec<Ext>>;

pub fn expand(t: &[Vec<Ext>]) -> Vec<Ext> {
    let mut v = vec![Ext::ONE];
    for w in t {
        let mut nv = Vec::with_capacity(v.len() * w.len());
        for wd in w {
            for x in &v {
                nv.push(x.mul(wd));
            }
        }
        v = nv;
    }
    v
}

/// Row digits (radix product `rows`) and column digits.
pub fn split(t: &[Vec<Ext>], rows: usize) -> Result<(Tensor, Tensor), String> {
    let (mut prod, mut m) = (1usize, 0usize);
    while prod < rows {
        if m == t.len() {
            return Err(format!("tensor split: {rows} rows exceed the tensor's {prod} entries"));
        }
        prod *= t[m].len();
        m += 1;
    }
    if prod != rows {
        return Err(format!("tensor split: digit product {prod} != {rows} rows"));
    }
    Ok((t[..m].to_vec(), t[m..].to_vec()))
}

/// `prod_j (w_j[0] (1 - r_j) + w_j[1] r_j)` over binary digits.
pub fn tensor_eval_binary(t: &[Vec<Ext>], r: &[Ext]) -> Result<Ext, String> {
    if t.len() != r.len() {
        return Err(format!("tensor_eval_binary: {} digits, {} challenges", t.len(), r.len()));
    }
    let mut out = Ext::ONE;
    for (w, rj) in t.iter().zip(r) {
        if w.len() != 2 {
            return Err("tensor_eval_binary: a column digit is not binary".into());
        }
        out = out.mul(&w[0].mul(&Ext::ONE.sub(rj)).add(&w[1].mul(rj)));
    }
    Ok(out)
}

pub fn eq_tensor(z: &[Ext]) -> Tensor {
    z.iter().map(|zj| vec![Ext::ONE.sub(zj), *zj]).collect()
}

/// Lagrange basis on {0, 1, 2} at `z`.
pub fn lagrange3(z: &Ext) -> Vec<Ext> {
    let one = Ext::ONE;
    let two = Ext::from_base(2);
    let inv2 = (P + 1) / 2;
    let l0 = z.sub(&one).mul(&z.sub(&two)).mul_base(inv2);
    let l1 = z.mul(&z.sub(&two)).neg();
    let l2 = z.mul(&z.sub(&one)).mul_base(inv2);
    vec![l0, l1, l2]
}

/// The evaluation point as a tensor: `eq` on binary digits, Lagrange-3 on a ternary digit.
pub fn point_tensor(z: &[Ext], radices: &[usize]) -> Result<Tensor, String> {
    if z.len() != radices.len() {
        return Err(format!("point has {} coordinates, the vector {} digits", z.len(), radices.len()));
    }
    z.iter()
        .zip(radices)
        .map(|(zj, &rad)| match rad {
            3 => Ok(lagrange3(zj)),
            2 => Ok(vec![Ext::ONE.sub(zj), *zj]),
            _ => Err(format!("radix {rad}")),
        })
        .collect()
}

/// The RS generator row `(omega_n^{s x})_x` over the digits `radices`.
pub fn rs_row_tensor(s: usize, n: usize, radices: &[usize]) -> Tensor {
    let w = pow(root_of_unity(n), s as u64);
    let mut stride = 1u64;
    radices
        .iter()
        .map(|&rad| {
            let base = pow(w, stride);
            stride *= rad as u64;
            (0..rad as u64).map(|d| Ext::from_base(pow(base, d))).collect()
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------- dims --------

/// `tall[i] x 2^wide_log2[i]` matrices, codeword length `2^n_log2[i]`, `queries[i]` opened rows per level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    pub tall: Vec<usize>,
    pub wide_log2: Vec<usize>,
    pub n_log2: Vec<usize>,
    pub queries: Vec<usize>,
}

impl Dims {
    pub fn levels(&self) -> usize {
        self.tall.len()
    }
    pub fn wide(&self, i: usize) -> usize {
        1 << self.wide_log2[i]
    }
    pub fn n(&self, i: usize) -> usize {
        1 << self.n_log2[i]
    }
    pub fn size(&self) -> usize {
        self.tall[0] << self.wide_log2[0]
    }

    /// The row digits of level `i`: `[3]` when `tall` is `3 * 2^k`, then `k` binary digits.
    pub fn row_radices(&self, i: usize) -> Result<Vec<usize>, String> {
        let t = self.tall[i];
        let odd = if t % 3 == 0 { 3 } else { 1 };
        let rest = t / odd;
        if t == 0 || !rest.is_power_of_two() {
            return Err(format!("tall[{i}] = {t} is neither 2^k nor 3 * 2^k"));
        }
        let mut r = Vec::new();
        if odd == 3 {
            r.push(3);
        }
        r.extend(std::iter::repeat(2).take(rest.trailing_zeros() as usize));
        Ok(r)
    }

    pub fn radices(&self) -> Result<Vec<usize>, String> {
        let mut r = self.row_radices(0)?;
        r.extend(std::iter::repeat(2).take(self.wide_log2[0]));
        Ok(r)
    }

    pub fn validate(&self) -> Result<(), String> {
        let l = self.levels();
        if l == 0 || self.wide_log2.len() != l || self.n_log2.len() != l || self.queries.len() != l {
            return Err("dims: level lists differ in length or are empty".into());
        }
        for i in 0..l {
            self.row_radices(i)?;
            if self.wide_log2[i] > 30 || self.n_log2[i] > 27 {
                return Err(format!("dims: level {i} too wide or too long"));
            }
            if self.n(i) < self.tall[i] {
                return Err(format!("dims: level {i}: codeword length {} < {} rows", self.n(i), self.tall[i]));
            }
            if self.queries[i] > self.n(i) || self.queries[i] == 0 {
                return Err(format!("dims: level {i}: {} queries from {} rows", self.queries[i], self.n(i)));
            }
            if i + 1 < l && self.tall[i] != self.tall[i + 1] << self.wide_log2[i + 1] {
                return Err(format!("dims: tall[{i}] != tall[{}] * wide[{}]", i + 1, i + 1));
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------- proof -------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Base(Vec<u32>),
    Ext(Vec<Ext>),
}

impl Row {
    pub fn bytes(&self) -> Vec<u8> {
        match self {
            Row::Base(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Row::Ext(v) => v.iter().flat_map(|x| x.to_bytes()).collect(),
        }
    }
    /// `<row, rbar>`.
    pub fn dot(&self, rbar: &[Ext]) -> Ext {
        match self {
            Row::Base(v) => crate::field::dot_ext_base(rbar, v),
            Row::Ext(v) => crate::field::dot_ext(v, rbar),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Row::Base(v) => v.len(),
            Row::Ext(v) => v.len(),
        }
    }
    pub fn is_base(&self) -> bool {
        matches!(self, Row::Base(_))
    }
}

#[derive(Clone, Debug)]
pub struct Level {
    pub sumcheck: Vec<[Ext; 3]>,
    pub opened_idx: Vec<usize>,
    pub opened_rows: Vec<Row>,
    pub siblings: Vec<Digest>,
}

#[derive(Clone, Debug)]
pub struct RefProof {
    pub dims: Dims,
    pub roots: Vec<Digest>,
    pub levels: Vec<Level>,
    pub final_vec: Vec<Ext>,
    pub value: Ext,
}

fn ext_bytes(v: &[Ext]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_bytes()).collect()
}

fn opening_bytes(rows: &[Row], sibs: &[Digest]) -> Vec<u8> {
    let mut b: Vec<u8> = rows.iter().flat_map(|r| r.bytes()).collect();
    for s in sibs {
        b.extend_from_slice(s);
    }
    b
}

fn merkle_check(root: &Digest, depth: usize, idx: &[usize], rows: &[Row], sibs: &[Digest]) -> Result<(), String> {
    let leaves: Vec<(usize, Digest)> = idx.iter().zip(rows).map(|(&i, r)| (i, blake3(&r.bytes()))).collect();
    merkle::verify_multi(root, &leaves, sibs, depth)
}

/// Verify `<f, w> = value` against the commitment `root`.  `Err(reason)` mirrors `ref.verify`'s reasons.
pub fn verify(root: &Digest, w: &[Vec<Ext>], value: &Ext, proof: &RefProof, coins: &mut dyn RefCoins) -> Result<(), String> {
    let dims = &proof.dims;
    dims.validate()?;
    let l = dims.levels();
    let radices = dims.radices()?;
    if w.iter().map(|t| t.len()).collect::<Vec<_>>() != radices {
        return Err("point tensor radices do not match dims".into());
    }
    if proof.roots.len() != l || proof.levels.len() != l || &proof.roots[0] != root {
        return Err("roots / levels shape or commitment mismatch".into());
    }
    if &proof.value != value {
        return Err("claimed value differs from the proof's value".into());
    }
    // a term: (gamma, row tensor, column tensor)
    let (ra, ca) = split(w, dims.tall[0])?;
    let mut terms: Vec<(Ext, Tensor, Tensor)> = vec![(Ext::ONE, ra, ca)];
    let mut total = *value;
    let mut rbar: Vec<Ext> = Vec::new();
    let mut carried: Vec<(Ext, Tensor)> = Vec::new();
    for i in 0..l {
        coins.absorb(b"root", &proof.roots[i]);
        if i >= 1 {
            let lp = &proof.levels[i - 1];
            let idx = coins.indices(b"queries", dims.n(i - 1), dims.queries[i - 1]);
            if idx != lp.opened_idx || lp.opened_rows.len() != idx.len() {
                return Err(format!("level {i}: opened indices are not the verifier's"));
            }
            check_rows(&lp.opened_rows, dims.wide(i - 1), i - 1 == 0).map_err(|e| format!("level {i}: {e}"))?;
            merkle_check(&proof.roots[i - 1], dims.n_log2[i - 1], &idx, &lp.opened_rows, &lp.siblings)
                .map_err(|e| format!("level {i}: Merkle multi-opening of level {} rejected ({e})", i - 1))?;
            coins.absorb(b"opened", &opening_bytes(&lp.opened_rows, &lp.siblings));
            // batch the carried claims and the opened rows with fresh alphas
            let alpha = coins.challenge(b"batch", 1 + idx.len());
            let mut rad_i = dims.row_radices(i)?;
            rad_i.extend(std::iter::repeat(2).take(dims.wide_log2[i]));
            let mut nt = Vec::with_capacity(carried.len() + idx.len());
            let mut ntotal = alpha[0].mul(&total);
            for (gamma, a) in &carried {
                let (ra, ca) = split(a, dims.tall[i])?;
                nt.push((alpha[0].mul(gamma), ra, ca));
            }
            for (m, (&s, row)) in idx.iter().zip(&lp.opened_rows).enumerate() {
                let y = row.dot(&rbar);
                ntotal = ntotal.add(&alpha[1 + m].mul(&y));
                let (ra, ca) = split(&rs_row_tensor(s, dims.n(i - 1), &rad_i), dims.tall[i])?;
                nt.push((alpha[1 + m], ra, ca));
            }
            terms = nt;
            total = ntotal;
        }
        // degree-2 sumcheck over the k'_i column digits, low digit first
        let msgs = &proof.levels[i].sumcheck;
        if msgs.len() != dims.wide_log2[i] {
            return Err(format!("level {}: sumcheck message shape", i + 1));
        }
        let mut r = Vec::with_capacity(msgs.len());
        for (j, [c0, c1, c2]) in msgs.iter().enumerate() {
            if c0.add(c0).add(c1).add(c2) != total {
                return Err(format!("level {}: sumcheck round {j}: g(0) + g(1) != claim", i + 1));
            }
            coins.absorb(b"sumcheck", &ext_bytes(&[*c0, *c1, *c2]));
            let rj = coins.challenge(b"r", 1)[0];
            total = c0.add(&rj.mul(&c1.add(&c2.mul(&rj))));
            r.push(rj);
        }
        rbar = expand(&eq_tensor(&r));
        carried = terms.iter().map(|(g, a, b)| Ok((g.mul(&tensor_eval_binary(b, &r)?), a.clone()))).collect::<Result<_, String>>()?;
    }
    let v = &proof.final_vec;
    if v.len() != dims.tall[l - 1] {
        return Err("final vector shape".into());
    }
    coins.absorb(b"final", &ext_bytes(v));
    let lp = &proof.levels[l - 1];
    let idx = coins.indices(b"queries", dims.n(l - 1), dims.queries[l - 1]);
    if idx != lp.opened_idx || lp.opened_rows.len() != idx.len() {
        return Err("last level: opened indices are not the verifier's".into());
    }
    check_rows(&lp.opened_rows, dims.wide(l - 1), l == 1).map_err(|e| format!("last level: {e}"))?;
    merkle_check(&proof.roots[l - 1], dims.n_log2[l - 1], &idx, &lp.opened_rows, &lp.siblings)
        .map_err(|e| format!("last level: Merkle multi-opening rejected ({e})"))?;
    coins.absorb(b"opened", &opening_bytes(&lp.opened_rows, &lp.siblings));
    // carried claims on v_L, checked directly
    let mut acc = Ext::ZERO;
    for (g, a) in &carried {
        acc = acc.add(&g.mul(&crate::field::dot_ext(&expand(a), v)));
    }
    if acc != total {
        return Err("final vector does not satisfy the carried claims".into());
    }
    // opened rows of the last codeword against the RS encoding of v_L
    let rad = dims.row_radices(l - 1)?;
    for (&s, row) in idx.iter().zip(&lp.opened_rows) {
        let y = row.dot(&rbar);
        if y != crate::field::dot_ext(&expand(&rs_row_tensor(s, dims.n(l - 1), &rad)), v) {
            return Err(format!("last level: opened row {s} is not the encoding of the final vector"));
        }
    }
    Ok(())
}

fn check_rows(rows: &[Row], wide: usize, base: bool) -> Result<(), String> {
    for r in rows {
        if r.len() != wide || r.is_base() != base {
            return Err("opened row shape".into());
        }
    }
    Ok(())
}

/// Paper (17) with unique decoding, per level, summed: `((1 + rate_i)/2)^{|S_i|} + (k'_i n_i + 2 k'_i + |S_{i-1}| + 1)/|F|`
/// as `-log2`.
pub fn soundness_bits(dims: &Dims) -> f64 {
    let mut total = 0f64;
    for i in 0..dims.levels() {
        let rate = dims.tall[i] as f64 / dims.n(i) as f64;
        let prox = ((1.0 + rate) / 2.0).powi(dims.queries[i] as i32);
        let k = dims.wide_log2[i] as f64;
        let s_prev = if i == 0 { 0 } else { dims.queries[i - 1] } as f64;
        let alg = (k * dims.n(i) as f64 + 2.0 * k + s_prev + 1.0) / (P as f64).powi(6);
        total += prox + alg;
    }
    -total.log2()
}

// ------------------------------------------------------------------------------------------------- JSON --------

fn num_u64(j: &Json) -> Option<u64> {
    match j {
        Json::Num(x) if *x >= 0.0 && x.fract() == 0.0 && *x < 1.8e19 => Some(*x as u64),
        _ => None,
    }
}

fn field_elem(j: &Json) -> Result<u32, String> {
    match num_u64(j) {
        Some(v) if v < P as u64 => Ok(v as u32),
        _ => Err("field element is not a canonical integer in [0, p)".into()),
    }
}

fn ext_of(j: &Json) -> Result<Ext, String> {
    let a = j.as_arr().ok_or("extension element is not a list")?;
    if a.len() != EXT_DEG {
        return Err(format!("extension element has {} coordinates", a.len()));
    }
    let mut c = [0u32; EXT_DEG];
    for (i, x) in a.iter().enumerate() {
        c[i] = field_elem(x)?;
    }
    Ok(Ext(c))
}

fn ext_list(j: &Json) -> Result<Vec<Ext>, String> {
    j.as_arr().ok_or("expected a list of extension elements")?.iter().map(ext_of).collect()
}

fn digest_of(j: &Json) -> Result<Digest, String> {
    let s = j.as_str().ok_or("digest is not a hex string")?;
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("digest is not 64 hex digits".into());
    }
    Ok(crate::hash::unhex32(s))
}

fn row_of(j: &Json) -> Result<Row, String> {
    let a = j.as_arr().ok_or("opened row is not a list")?;
    if a.iter().all(|x| matches!(x, Json::Num(_))) {
        Ok(Row::Base(a.iter().map(field_elem).collect::<Result<_, _>>()?))
    } else {
        Ok(Row::Ext(a.iter().map(ext_of).collect::<Result<_, _>>()?))
    }
}

/// `ref.proof_to_dict` output.
pub fn proof_from_json(j: &Json) -> Result<RefProof, String> {
    let get = |k: &str| j.get(k).ok_or(format!("proof: missing {k}"));
    if get("format")?.as_str() != Some("ligerito-ref/v1") {
        return Err("proof: format is not ligerito-ref/v1".into());
    }
    if get("hash")?.as_str() != Some("blake3") {
        return Err("proof: hash is not blake3 (ref.py ran without the blake3 package)".into());
    }
    let d = get("dims")?;
    let dl = |k: &str| d.get(k).and_then(|x| x.usize_list()).ok_or(format!("proof: dims.{k}"));
    let dims = Dims { tall: dl("tall")?, wide_log2: dl("wide_log2")?, n_log2: dl("n_log2")?, queries: dl("queries")? };
    let roots = get("roots")?.as_arr().ok_or("proof: roots")?.iter().map(digest_of).collect::<Result<_, _>>()?;
    let mut levels = Vec::new();
    for (i, lv) in get("levels")?.as_arr().ok_or("proof: levels")?.iter().enumerate() {
        let g = |k: &str| lv.get(k).ok_or(format!("proof: levels[{i}].{k}"));
        let sumcheck = g("sumcheck")?
            .as_arr()
            .ok_or("sumcheck")?
            .iter()
            .map(|m| {
                let v = ext_list(m)?;
                if v.len() != 3 {
                    return Err(format!("levels[{i}]: sumcheck message has {} coefficients", v.len()));
                }
                Ok([v[0], v[1], v[2]])
            })
            .collect::<Result<_, String>>()?;
        let opened_idx = g("opened_idx")?.usize_list().ok_or("opened_idx")?;
        let opened_rows = g("opened_rows")?.as_arr().ok_or("opened_rows")?.iter().map(row_of).collect::<Result<_, _>>()?;
        let siblings = g("siblings")?.as_arr().ok_or("siblings")?.iter().map(digest_of).collect::<Result<_, _>>()?;
        levels.push(Level { sumcheck, opened_idx, opened_rows, siblings });
    }
    let final_vec = ext_list(get("final")?)?;
    let value = ext_of(get("value")?)?;
    Ok(RefProof { dims, roots, levels, final_vec, value })
}

/// One `ref.fixture(...)` entry: the statement, the raw proof JSON (kept for tampering) and ref.py's verdict.
pub struct Fixture {
    pub name: String,
    pub seed: Vec<u8>,
    pub z: Vec<Ext>,
    pub commitment: Digest,
    pub proof: Json,
    pub expect_ok: bool,
    pub expect_reason: String,
}

pub fn load_fixtures(text: &str) -> Result<Vec<Fixture>, String> {
    let j = crate::json::parse(text)?;
    let arr = j.as_arr().ok_or("fixtures: not a list")?;
    arr.iter()
        .enumerate()
        .map(|(i, f)| {
            let get = |k: &str| f.get(k).ok_or(format!("fixture {i}: missing {k}"));
            let expect = get("expect")?;
            Ok(Fixture {
                name: f.get("name").and_then(|x| x.as_str()).unwrap_or(&format!("fixture[{i}]")).to_string(),
                seed: get("seed")?.as_str().ok_or("seed")?.as_bytes().to_vec(),
                z: ext_list(get("z")?)?,
                commitment: digest_of(get("commitment")?)?,
                proof: get("proof")?.clone(),
                expect_ok: expect.get("ok").and_then(|x| x.as_bool()).ok_or("expect.ok")?,
                expect_reason: expect.get("reason").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            })
        })
        .collect()
}

/// Verify a fixture's statement + proof (from the given proof JSON, normally `fx.proof`) with `FsBlake3(seed)`;
/// on acceptance the coin count, the paper's soundness figure and the opened-row count.
pub fn verify_fixture(fx: &Fixture, proof_json: &Json) -> Result<crate::pcs::Report, String> {
    let proof = proof_from_json(proof_json)?;
    let w = point_tensor(&fx.z, &proof.dims.radices()?)?;
    let mut coins = FsBlake3::new(&fx.seed);
    verify(&fx.commitment, &w, &proof.value, &proof, &mut coins)?;
    Ok(crate::pcs::Report {
        coins: coins.rounds(),
        soundness_log2: -soundness_bits(&proof.dims),
        opened_rows: proof.dims.queries.iter().sum(),
        terms_final: 1 + proof.dims.queries[..proof.dims.levels() - 1].iter().sum::<usize>(),
    })
}

/// A declarative tamper (`fixtures/ref/negatives.json`): `{"base": i, "ops": [{"path": [...], "op": ..., "v": ...}],
/// "expect": {...}}`; path elements are object keys or array indices (negative = from the end).
pub struct Tamper {
    pub name: String,
    pub base: usize,
    pub ops: Vec<Json>,
    pub expect_ok: bool,
    pub expect_reason: String,
}

pub fn load_tampers(text: &str) -> Result<Vec<Tamper>, String> {
    let j = crate::json::parse(text)?;
    j.as_arr()
        .ok_or("tampers: not a list")?
        .iter()
        .map(|t| {
            let expect = t.get("expect").ok_or("tamper: expect")?;
            Ok(Tamper {
                name: t.get("name").and_then(|x| x.as_str()).unwrap_or("?").to_string(),
                base: t.get("base").and_then(|x| x.as_usize()).ok_or("tamper: base")?,
                ops: t.get("ops").and_then(|x| x.as_arr()).ok_or("tamper: ops")?.to_vec(),
                expect_ok: expect.get("ok").and_then(|x| x.as_bool()).ok_or("tamper: expect.ok")?,
                expect_reason: expect.get("reason").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            })
        })
        .collect()
}

fn walk<'a>(mut node: &'a mut Json, path: &[Json]) -> Result<&'a mut Json, String> {
    for p in path {
        node = match (node, p) {
            (Json::Obj(kv), Json::Str(k)) => &mut kv.iter_mut().find(|(kk, _)| kk == k).ok_or(format!("path: no key {k}"))?.1,
            (Json::Arr(a), Json::Num(x)) => {
                let i = if *x < 0.0 { a.len() as i64 + *x as i64 } else { *x as i64 };
                a.get_mut(i as usize).ok_or(format!("path: index {x} out of range"))?
            }
            _ => return Err("path: element does not match the node".into()),
        };
    }
    Ok(node)
}

pub fn apply_ops(doc: &mut Json, ops: &[Json]) -> Result<(), String> {
    for op in ops {
        let path = op.get("path").and_then(|x| x.as_arr()).ok_or("op: path")?;
        let kind = op.get("op").and_then(|x| x.as_str()).ok_or("op: op")?;
        let v = op.get("v");
        let node = walk(doc, path)?;
        match kind {
            "add" | "addraw" | "xor" => {
                let a = num_u64(v.ok_or("op: v")?).ok_or("op: v not an integer")?;
                let x = num_u64(node).ok_or("op: target not an integer")?;
                let y = match kind {
                    "add" => (x + a) % P as u64,
                    "addraw" => x + a,
                    _ => x ^ a,
                };
                *node = Json::Num(y as f64);
            }
            "set" => *node = v.ok_or("op: v")?.clone(),
            "pop" => {
                let a = match node {
                    Json::Arr(a) => a,
                    _ => return Err("op: pop on a non-list".into()),
                };
                a.pop().ok_or("op: pop on an empty list")?;
            }
            "push" | "dup0" => {
                let a = match node {
                    Json::Arr(a) => a,
                    _ => return Err("op: push on a non-list".into()),
                };
                let item = if kind == "push" { v.ok_or("op: v")?.clone() } else { a.first().ok_or("op: dup0 on an empty list")?.clone() };
                a.push(item);
            }
            other => return Err(format!("op: unknown {other}")),
        }
    }
    Ok(())
}

pub fn describe(proof: &RefProof) -> String {
    let d = &proof.dims;
    format!(
        "N = {} = {:?} x 2^{:?}, n = 2^{:?}, queries {:?}, {} levels, soundness ~2^-{:.1}",
        d.size(),
        d.tall,
        d.wide_log2,
        d.n_log2,
        d.queries,
        d.levels(),
        soundness_bits(d)
    )
}

pub fn digest_hex(d: &Digest) -> String {
    hex(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xof_matches_the_python_blake3_package() {
        // blake3.blake3(d).digest(length=200) for d in ["", "abc", bytes(range(100)), 2000 bytes i % 251]
        let vectors = include_str!("../fixtures/ref/blake3_xof.txt");
        for line in vectors.lines() {
            let (n, hexout) = line.split_once(' ').unwrap();
            let n: usize = n.parse().unwrap();
            let data: Vec<u8> = match n {
                0 => vec![],
                3 => b"abc".to_vec(),
                100 => (0..100u8).collect(),
                _ => (0..n).map(|i| (i % 251) as u8).collect(),
            };
            assert_eq!(hex(&blake3_xof(&data, 200)), hexout, "xof of {n} bytes");
        }
    }

    #[test]
    fn tensors_expand_split_and_lagrange() {
        let z = Ext::from_base(2);
        let l = lagrange3(&z);
        assert_eq!(l, vec![Ext::ZERO, Ext::ZERO, Ext::ONE]);
        let l = lagrange3(&Ext::from_base(0));
        assert_eq!(l, vec![Ext::ONE, Ext::ZERO, Ext::ZERO]);
        // rs row tensor over [3, 2, 2] expands to omega^{s x}
        let n = 64;
        let t = rs_row_tensor(5, n, &[3, 2, 2]);
        let v = expand(&t);
        let w = pow(root_of_unity(n), 5);
        for x in 0..12 {
            assert_eq!(v[x], Ext::from_base(pow(w, x as u64)));
        }
        let (r, c) = split(&t, 6).unwrap();
        assert_eq!((r.len(), c.len()), (2, 1));
        assert!(split(&t, 4).is_err());
    }

    fn fixtures() -> Vec<Fixture> {
        load_fixtures(include_str!("../fixtures/ref/fixtures.json")).unwrap()
    }

    #[test]
    fn ref_py_fixtures_verify_bit_for_bit() {
        let fxs = fixtures();
        assert_eq!(fxs.len(), 4);
        let mut saw_ternary = false;
        for fx in &fxs {
            assert!(fx.expect_ok);
            let proof = proof_from_json(&fx.proof).unwrap();
            saw_ternary |= proof.dims.tall[0] % 3 == 0;
            verify_fixture(fx, &fx.proof).unwrap_or_else(|e| panic!("{}: {e}", fx.name));
            // a different seed: different coins, rejected
            let mut coins = FsBlake3::new(b"other");
            let w = point_tensor(&fx.z, &proof.dims.radices().unwrap()).unwrap();
            assert!(verify(&fx.commitment, &w, &proof.value, &proof, &mut coins).is_err());
            // soundness bound is finite and the fixtures are toys
            assert!(soundness_bits(&proof.dims) > 1.0 && soundness_bits(&proof.dims) < 60.0);
        }
        assert!(saw_ternary, "parameter-set-B shape (3 x 2^k tall) is among the fixtures");
    }

    #[test]
    fn ref_py_tampers_agree_with_python_verdicts() {
        let fxs = fixtures();
        let tampers = load_tampers(include_str!("../fixtures/ref/negatives.json")).unwrap();
        assert!(tampers.len() >= 76);
        let mut rejected = 0;
        for t in &tampers {
            let fx = &fxs[t.base];
            let mut doc = fx.proof.clone();
            apply_ops(&mut doc, &t.ops).unwrap_or_else(|e| panic!("{}: {e}", t.name));
            let got = verify_fixture(fx, &doc);
            // ref.py reduces row entries mod p when hashing; we reject non-canonical encodings instead
            let expect_ok = t.expect_ok && !t.name.ends_with("/noncanon");
            assert_eq!(got.is_ok(), expect_ok, "{}: rust {:?}, python {} ({})", t.name, got, t.expect_ok, t.expect_reason);
            if !expect_ok {
                rejected += 1;
                if !t.name.ends_with("/noncanon") && !t.expect_reason.starts_with("exception") {
                    // the reason should name the same stage as ref.py's (ref.py raised while deserialising ragged rows)
                    let stage = t.expect_reason.split(':').next().unwrap();
                    assert!(got.as_ref().unwrap_err().starts_with(stage), "{}: {:?} vs {}", t.name, got, t.expect_reason);
                }
            }
        }
        assert_eq!(rejected, tampers.len());
    }
}
