//! Proof bytes.  Two containers are read:
//!
//! * `LGRTOPV0` -- this crate's PCS proof container (statement + parameters + proof, all little-endian) written by
//!   the reference prover and `selftest --emit`.
//! * the `proto/pcs.py` `Proof.to_bytes` layout (lane ligerito-proto): u32 LE header length, JSON header
//!   (`n, kprime, queries, rate_log2, splits, coins, hash, beta, ext_degree, rows_shape, sib_len, final_len`), v,
//!   per round root + k'_i x (3, 6) messages, per opening the rows as u32 words (|S| x W, W = C or 6 C plane-major)
//!   + the sibling bytes, the final vector.  The evaluation point z is not in the proof: it is the statement
//!   (sidecar `<name>.z`, n x 24 B, or the caller's).
//!
//! Every length is checked before it is read; every field element must be a canonical (reduced) encoding.
//!
//! ~~~text
//! LGRTOPV0:
//! magic "LGRTOPV0"
//! u32 n, u32 L, u32 rate_log2, u32 domain_shift, u32 kprime[L], u32 queries[L], (u32 n1, u32 n2)[L]
//! z: n x ext(24 B), v: ext
//! roots: L x 32 B
//! for i = 1..L: k'_i x 3 x ext (g(0), g(1), g(2))
//! for i = 1..L: opening of round i: u32 nrows; rows: nrows x cols_i x (u32 | ext); u32 nsib; siblings: nsib x 32 B
//! y: 2^{k_L} x ext
//! ~~~

use crate::field::{Ext, BETA, EXT_DEG, P};
use crate::json;
use crate::merkle::Digest;
use crate::params::Params;
use crate::pcs::{row_from_words, Opening, Proof, Statement};

pub const MAGIC_PCS_V0: &[u8; 8] = b"LGRTOPV0";

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.remaining() < n {
            return Err(format!("truncated: need {n} bytes at offset {}, have {}", self.pos, self.remaining()));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    /// A bounded count (guards allocation from hostile lengths).
    pub fn count(&mut self, what: &str, max: usize) -> Result<usize, String> {
        let n = self.u32()? as usize;
        if n > max {
            return Err(format!("{what}: count {n} exceeds {max}"));
        }
        Ok(n)
    }

    pub fn base(&mut self) -> Result<u32, String> {
        let v = self.u32()?;
        if v >= P {
            return Err(format!("non-canonical base element at offset {}", self.pos - 4));
        }
        Ok(v)
    }

    pub fn ext(&mut self) -> Result<Ext, String> {
        let b = self.bytes(24)?;
        Ext::from_bytes(b).ok_or_else(|| format!("non-canonical extension element at offset {}", self.pos - 24))
    }

    pub fn exts(&mut self, n: usize) -> Result<Vec<Ext>, String> {
        (0..n).map(|_| self.ext()).collect()
    }

    pub fn digest(&mut self) -> Result<Digest, String> {
        Ok(self.bytes(32)?.try_into().unwrap())
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.remaining() != 0 {
            return Err(format!("{} trailing bytes", self.remaining()));
        }
        Ok(())
    }
}

pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer { buf: Vec::new() }
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn ext(&mut self, e: &Ext) {
        self.buf.extend_from_slice(&e.to_bytes());
    }
    pub fn exts(&mut self, v: &[Ext]) {
        for e in v {
            self.ext(e);
        }
    }
    pub fn digest(&mut self, d: &Digest) {
        self.buf.extend_from_slice(d);
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

const MAX_QUERIES: usize = 1 << 16;
const MAX_SIBLINGS: usize = 1 << 22;

fn read_rows(r: &mut Reader, nrows: usize, cols: usize, base: bool) -> Result<Vec<Vec<Ext>>, String> {
    let mut rows = Vec::with_capacity(nrows);
    for _ in 0..nrows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(if base { Ext::from_base(r.base()?) } else { r.ext()? });
        }
        rows.push(row);
    }
    Ok(rows)
}

fn write_rows(w: &mut Writer, rows: &[Vec<Ext>], base: bool) {
    for row in rows {
        for e in row {
            if base {
                w.u32(e.0[0]);
            } else {
                w.ext(e);
            }
        }
    }
}

fn read_opening(r: &mut Reader, p: &Params, i: usize) -> Result<Opening, String> {
    let nrows = r.count("opened rows", MAX_QUERIES)?;
    let rows = read_rows(r, nrows, p.cols(i), i == 1)?;
    let nsib = r.count("siblings", MAX_SIBLINGS)?;
    let siblings = (0..nsib).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
    Ok(Opening { rows, siblings })
}

fn write_opening(w: &mut Writer, op: &Opening, i: usize) {
    w.u32(op.rows.len() as u32);
    write_rows(w, &op.rows, i == 1);
    w.u32(op.siblings.len() as u32);
    for d in &op.siblings {
        w.digest(d);
    }
}

pub fn read_pcs_v0(buf: &[u8]) -> Result<(Statement, Proof), String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC_PCS_V0 {
        return Err("not an LGRTOPV0 proof".into());
    }
    let n = r.count("n", 40)?;
    let l = r.count("rounds", 16)?;
    let rate_log2 = r.count("rate", 8)?;
    let domain_shift = r.u32()?;
    let kprime = (0..l).map(|_| r.count("k'", 40)).collect::<Result<Vec<_>, _>>()?;
    let queries = (0..l).map(|_| r.count("|S|", MAX_QUERIES)).collect::<Result<Vec<_>, _>>()?;
    let mut splits = Vec::with_capacity(l);
    for _ in 0..l {
        splits.push((r.count("n1", 1 << 27)?, r.count("n2", 1 << 27)?));
    }
    let params = Params { n, kprime, queries, rate_log2, domain_shift, splits };
    params.validate()?;
    let z = r.exts(n)?;
    let v = r.ext()?;
    let roots = (0..l).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
    let mut sumcheck = Vec::with_capacity(l);
    for i in 1..=l {
        let mut msgs = Vec::with_capacity(params.kprime[i - 1]);
        for _ in 0..params.kprime[i - 1] {
            msgs.push([r.ext()?, r.ext()?, r.ext()?]);
        }
        sumcheck.push(msgs);
    }
    let mut openings = Vec::with_capacity(l);
    for i in 1..=l {
        openings.push(read_opening(&mut r, &params, i)?);
    }
    let y_final = r.exts(params.final_len())?;
    r.finish()?;
    Ok((Statement { z, v }, Proof { params, roots, sumcheck, openings, y_final }))
}

pub fn write_pcs_v0(stmt: &Statement, proof: &Proof) -> Vec<u8> {
    let p = &proof.params;
    let l = p.rounds();
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC_PCS_V0);
    w.u32(p.n as u32);
    w.u32(l as u32);
    w.u32(p.rate_log2 as u32);
    w.u32(p.domain_shift);
    for &k in &p.kprime {
        w.u32(k as u32);
    }
    for &q in &p.queries {
        w.u32(q as u32);
    }
    for &(n1, n2) in &p.splits {
        w.u32(n1 as u32);
        w.u32(n2 as u32);
    }
    w.exts(&stmt.z);
    w.ext(&stmt.v);
    for d in &proof.roots {
        w.digest(d);
    }
    for msgs in &proof.sumcheck {
        for g in msgs {
            w.exts(g);
        }
    }
    for i in 1..=l {
        write_opening(&mut w, &proof.openings[i - 1], i);
    }
    w.exts(&proof.y_final);
    w.buf
}

/// Is this a `proto/pcs.py` proof?  (u32 LE header length followed by '{'.)
pub fn looks_like_proto(buf: &[u8]) -> bool {
    buf.len() >= 5 && buf[4] == b'{' && (u32::from_le_bytes(buf[..4].try_into().unwrap()) as usize) < buf.len()
}

/// Read a `proto/pcs.py` proof; `z` is the evaluation point (the statement's).
pub fn read_proto(buf: &[u8], z: Vec<Ext>) -> Result<(Statement, Proof), String> {
    let mut r = Reader::new(buf);
    let hl = r.count("header", 1 << 20)?;
    let head = std::str::from_utf8(r.bytes(hl)?).map_err(|_| "proto header: not utf-8")?;
    let h = json::parse(head)?;
    let field = |k: &str| h.get(k).ok_or_else(|| format!("proto header: missing {k}"));
    let n = field("n")?.as_usize().ok_or("proto header: n")?;
    let kprime = field("kprime")?.usize_list().ok_or("proto header: kprime")?;
    let queries = field("queries")?.usize_list().ok_or("proto header: queries")?;
    let rate_log2 = field("rate_log2")?.as_usize().ok_or("proto header: rate_log2")?;
    if let Some(hash) = h.get("hash") {
        if hash.as_str() != Some("blake3") {
            return Err(format!("proto header: hash {hash:?} (this verifier: blake3)"));
        }
    }
    if let Some(b) = h.get("beta") {
        if b.as_usize() != Some(BETA as usize) {
            return Err(format!("proto header: beta {b:?} (this verifier: x^6 - {BETA})"));
        }
    }
    if let Some(d) = h.get("ext_degree") {
        if d.as_usize() != Some(EXT_DEG) {
            return Err(format!("proto header: ext_degree {d:?} (this verifier: {EXT_DEG})"));
        }
    }
    let l = kprime.len();
    let splits: Vec<(usize, usize)> = match h.get("splits") {
        Some(s) => s
            .as_arr()
            .ok_or("proto header: splits")?
            .iter()
            .map(|x| {
                let v = x.usize_list().ok_or("proto header: split")?;
                if v.len() != 2 {
                    return Err("proto header: split pair".to_string());
                }
                Ok((v[0], v[1]))
            })
            .collect::<Result<_, String>>()?,
        None => Vec::new(),
    };
    let mut params = Params::new(n, kprime, queries, rate_log2);
    if !splits.is_empty() {
        params.splits = splits;
    }
    params.validate()?;
    let rows_shape: Vec<Vec<usize>> = field("rows_shape")?
        .as_arr()
        .ok_or("proto header: rows_shape")?
        .iter()
        .map(|x| x.usize_list().ok_or("proto header: rows_shape entry".to_string()))
        .collect::<Result<_, _>>()?;
    let sib_len = field("sib_len")?.usize_list().ok_or("proto header: sib_len")?;
    let final_len = field("final_len")?.as_usize().ok_or("proto header: final_len")?;
    if rows_shape.len() != l || sib_len.len() != l {
        return Err(format!("proto header: {} rows_shape / {} sib_len for {l} rounds", rows_shape.len(), sib_len.len()));
    }
    if z.len() != n {
        return Err(format!("statement: z has {} coordinates, n = {n}", z.len()));
    }
    let v = r.ext()?;
    let mut roots = Vec::with_capacity(l);
    let mut sumcheck = Vec::with_capacity(l);
    for i in 1..=l {
        roots.push(r.digest()?);
        let mut msgs = Vec::with_capacity(params.kprime[i - 1]);
        for _ in 0..params.kprime[i - 1] {
            msgs.push([r.ext()?, r.ext()?, r.ext()?]);
        }
        sumcheck.push(msgs);
    }
    let mut openings = Vec::with_capacity(l);
    for i in 1..=l {
        let shape = &rows_shape[i - 1];
        if shape.len() != 2 || shape[0] > MAX_QUERIES {
            return Err(format!("proto header: rows_shape {shape:?}"));
        }
        let (nrows, width) = (shape[0], shape[1]);
        let base = i == 1;
        let cols = params.cols(i);
        if width != cols * if base { 1 } else { EXT_DEG } {
            return Err(format!("round {i} opening: row width {width}, expected {}", cols * if base { 1 } else { EXT_DEG }));
        }
        let mut rows = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            let words = (0..width).map(|_| r.base()).collect::<Result<Vec<_>, _>>()?;
            rows.push(row_from_words(&words, cols, base).ok_or("row words")?);
        }
        if sib_len[i - 1] % 32 != 0 || sib_len[i - 1] / 32 > MAX_SIBLINGS {
            return Err(format!("round {i} opening: sibling bytes {}", sib_len[i - 1]));
        }
        let siblings = (0..sib_len[i - 1] / 32).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
        openings.push(Opening { rows, siblings });
    }
    if final_len != params.final_len() {
        return Err(format!("proto header: final_len {final_len}, expected {}", params.final_len()));
    }
    let y_final = r.exts(final_len)?;
    r.finish()?;
    Ok((Statement { z, v }, Proof { params, roots, sumcheck, openings, y_final }))
}

/// Write a proof in the `proto/pcs.py` layout (tests: the two readers must agree).
pub fn write_proto(stmt: &Statement, proof: &Proof, coins: usize) -> Vec<u8> {
    let p = &proof.params;
    let l = p.rounds();
    let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let rows_shape = (1..=l)
        .map(|i| format!("[{},{}]", proof.openings[i - 1].rows.len(), p.cols(i) * if i == 1 { 1 } else { EXT_DEG }))
        .collect::<Vec<_>>()
        .join(",");
    let sib_len = proof.openings.iter().map(|o| (32 * o.siblings.len()).to_string()).collect::<Vec<_>>().join(",");
    let splits = p.splits.iter().map(|(a, b)| format!("[{a},{b}]")).collect::<Vec<_>>().join(",");
    let head = format!(
        "{{\"n\":{},\"kprime\":[{}],\"queries\":[{}],\"rate_log2\":{},\"splits\":[{}],\"coins\":{},\"hash\":\"blake3\",\"beta\":{},\"ext_degree\":{},\"rows_shape\":[{}],\"sib_len\":[{}],\"final_len\":{}}}",
        p.n,
        list(&p.kprime),
        list(&p.queries),
        p.rate_log2,
        splits,
        coins,
        BETA,
        EXT_DEG,
        rows_shape,
        sib_len,
        proof.y_final.len()
    );
    let mut w = Writer::new();
    w.u32(head.len() as u32);
    w.buf.extend_from_slice(head.as_bytes());
    w.ext(&stmt.v);
    for i in 1..=l {
        w.digest(&proof.roots[i - 1]);
        for g in &proof.sumcheck[i - 1] {
            w.exts(g);
        }
    }
    for i in 1..=l {
        for row in &proof.openings[i - 1].rows {
            for word in crate::pcs::row_words(row, i == 1) {
                w.u32(word);
            }
        }
        for d in &proof.openings[i - 1].siblings {
            w.digest(d);
        }
    }
    w.exts(&proof.y_final);
    w.buf
}

/// `<name>.z` sidecar: n x 24 B.
pub fn read_point(buf: &[u8]) -> Result<Vec<Ext>, String> {
    if buf.len() % 24 != 0 {
        return Err(format!("point file: {} bytes is not a multiple of 24", buf.len()));
    }
    let mut r = Reader::new(buf);
    r.exts(buf.len() / 24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refprover::{prove, random_point, random_vector};
    use crate::transcript::FiatShamir;

    #[test]
    fn pcs_v0_roundtrip_and_rejects_garbage() {
        let p = Params::new(8, vec![3, 2], vec![6, 6], 2);
        let f = random_vector(8, 5);
        let z = random_point(8, 6);
        let (stmt, proof) = prove(&p, &f, &z, &mut FiatShamir::new(FiatShamir::LABEL));
        let bytes = write_pcs_v0(&stmt, &proof);
        let (s2, p2) = read_pcs_v0(&bytes).unwrap();
        assert_eq!(s2.z, stmt.z);
        assert_eq!(s2.v, stmt.v);
        assert_eq!(p2, proof);
        assert_eq!(write_pcs_v0(&s2, &p2), bytes);
        for cut in [0, 7, 8, 20, 100, bytes.len() - 1] {
            assert!(read_pcs_v0(&bytes[..cut]).is_err());
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(read_pcs_v0(&longer).is_err());
        let mut bad = bytes.clone();
        let off = 8 + 4 * 4 + 4 * 2 * 2 + 8 * 2; // first z coordinate
        bad[off..off + 4].copy_from_slice(&P.to_le_bytes());
        assert!(read_pcs_v0(&bad).is_err());
    }

    #[test]
    fn proto_layout_roundtrip() {
        let p = Params::new(9, vec![3, 2], vec![6, 6], 2);
        let f = random_vector(9, 5);
        let z = random_point(9, 6);
        let mut t = FiatShamir::new(FiatShamir::LABEL);
        let (stmt, proof) = prove(&p, &f, &z, &mut t);
        let bytes = write_proto(&stmt, &proof, 0);
        assert!(looks_like_proto(&bytes));
        assert!(!looks_like_proto(&write_pcs_v0(&stmt, &proof)));
        let (s2, p2) = read_proto(&bytes, z.clone()).unwrap();
        assert_eq!(s2.v, stmt.v);
        assert_eq!(p2, proof);
        assert!(read_proto(&bytes[..bytes.len() - 1], z.clone()).is_err());
        assert!(read_proto(&bytes, z[..8].to_vec()).is_err());
        // the round-2 rows are plane-major words: coordinate a of entry c at a C + c
        let words = crate::pcs::row_words(&proof.openings[1].rows[0], false);
        let cols = p.cols(2);
        assert_eq!(words[cols + 1], proof.openings[1].rows[0][1].0[1]);
    }
}
