//! Proof bytes.  `LGRTOPV0` is this crate's provisional PCS proof container (statement + parameters + proof, all
//! little-endian) used by the reference prover and the fixtures until `proof.py`'s LGTO0001 lands; the reader for
//! that format is added next to it.  Every length is checked before it is read; every field element must be a
//! canonical (reduced) encoding.
//!
//! ~~~text
//! magic "LGRTOPV0"
//! u32 n, u32 L, u32 rate_log2, u32 domain_shift, u32 kprime[L], u32 queries[L]
//! z: n x ext(24 B), alpha: ext
//! roots: L x 32 B
//! for i = 2..L: opening of round i-1: u32 nrows; v: nrows x ext; rows: nrows x cols_{i-1} x (u32 | ext);
//!                                    u32 nsib; siblings: nsib x 32 B
//! for i = 1..L: k'_i x 3 x ext (g(0), g(1), g(2))
//! y: 2^{k_L} x ext
//! final opening: u32 nrows; rows: nrows x cols_L x (u32 | ext); u32 nsib; siblings
//! ~~~

use crate::field::{Ext, P};
use crate::merkle::Digest;
use crate::params::Params;
use crate::pcs::{Opening, Proof, Statement};

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

const MAX_QUERIES: usize = 1 << 16;
const MAX_SIBLINGS: usize = 1 << 22;

fn read_opening(r: &mut Reader, p: &Params, i: usize, with_v: bool) -> Result<Opening, String> {
    let nrows = r.count("opened rows", MAX_QUERIES)?;
    let v = if with_v { r.exts(nrows)? } else { Vec::new() };
    let rows = read_rows(r, nrows, p.cols(i), i == 1)?;
    let nsib = r.count("siblings", MAX_SIBLINGS)?;
    let siblings = (0..nsib).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
    Ok(Opening { v, rows, siblings })
}

fn write_opening(w: &mut Writer, op: &Opening, i: usize, with_v: bool) {
    w.u32(op.rows.len() as u32);
    if with_v {
        w.exts(&op.v);
    }
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
    let params = Params { n, kprime, queries, rate_log2, domain_shift };
    params.validate()?;
    let z = r.exts(n)?;
    let alpha = r.ext()?;
    let roots = (0..l).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
    let mut openings = Vec::with_capacity(l.saturating_sub(1));
    for i in 2..=l {
        openings.push(read_opening(&mut r, &params, i - 1, true)?);
    }
    let mut sumcheck = Vec::with_capacity(l);
    for i in 1..=l {
        let mut msgs = Vec::with_capacity(params.kprime[i - 1]);
        for _ in 0..params.kprime[i - 1] {
            msgs.push([r.ext()?, r.ext()?, r.ext()?]);
        }
        sumcheck.push(msgs);
    }
    let y_final = r.exts(params.final_len())?;
    let final_opening = read_opening(&mut r, &params, l, false)?;
    r.finish()?;
    Ok((Statement { z, alpha }, Proof { params, roots, openings, sumcheck, y_final, final_opening }))
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
    w.exts(&stmt.z);
    w.ext(&stmt.alpha);
    for d in &proof.roots {
        w.digest(d);
    }
    for i in 2..=l {
        write_opening(&mut w, &proof.openings[i - 2], i - 1, true);
    }
    for msgs in &proof.sumcheck {
        for g in msgs {
            w.exts(g);
        }
    }
    w.exts(&proof.y_final);
    write_opening(&mut w, &proof.final_opening, l, false);
    w.buf
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
        assert_eq!(s2.alpha, stmt.alpha);
        assert_eq!(p2, proof);
        assert_eq!(write_pcs_v0(&s2, &p2), bytes);
        // truncations and trailing bytes fail to parse (never panic)
        for cut in [0, 7, 8, 20, 100, bytes.len() - 1] {
            assert!(read_pcs_v0(&bytes[..cut]).is_err());
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(read_pcs_v0(&longer).is_err());
        // a non-canonical field element
        let mut bad = bytes.clone();
        let off = 8 + 4 * 4 + 4 * 2 * 2; // first z coordinate
        bad[off..off + 4].copy_from_slice(&P.to_le_bytes());
        assert!(read_pcs_v0(&bad).is_err());
    }
}
