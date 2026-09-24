//! `proof.bin` (gpu/README.md "Proof file").  The parser is the only place the file's own counts are read; every
//! count is compared against the statement-derived expectation by the verifier before use.

use crate::field::{Ext, P};
use crate::sha512::DIGEST_BYTES;

pub type Digest = [u8; DIGEST_BYTES];

pub struct Column { pub values: Vec<u32>, pub path: Vec<Digest> }

pub struct Opening { pub w: Vec<Ext>, pub qc: Vec<Ext>, pub columns: Vec<Column> }

pub struct Proof { pub root: Digest, pub msgs: Vec<Ext>, pub opening: Option<Opening>, pub bytes_read: usize }

pub const MAGIC: &[u8; 8] = b"VGKRBB6\0";
/// Version 2 = SHA-512 Merkle (the u32 after the version is the digest size, 64).  Version 1 (SHA-256, 32-byte
/// digests) is refused: its binding term alone is 2^-128, so a proof in it does not carry the claimed soundness.
pub const VERSION: u32 = 2;
/// Byte offset of the root in proof.bin (magic, version, digest size).
pub const ROOT_AT: usize = 16;

struct Rd<'a> { b: &'a [u8], pos: usize }
impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.b.len() { return Err("proof.bin truncated".into()); }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn ext(&mut self) -> Result<Ext, String> { Ext::from_bytes(self.take(24)?).ok_or_else(|| "proof.bin: non-canonical extension coordinate".to_string()) }
    fn exts(&mut self, n: u64, cap: usize, what: &str) -> Result<Vec<Ext>, String> {
        if n as usize > cap { return Err(format!("proof.bin: {what} count {n} exceeds the statement's bound {cap}")); }
        (0..n).map(|_| self.ext()).collect()
    }
}

impl Proof {
    /// `max_*`: upper bounds derived from the statement so a hostile file cannot make the parser allocate; the
    /// verifier then checks equality.
    pub fn parse(b: &[u8], max_msgs: usize, k: usize, t: usize, rows: usize, depth: usize) -> Result<Proof, String> {
        let mut r = Rd { b, pos: 0 };
        if r.take(8)? != MAGIC { return Err("proof.bin: bad magic".into()); }
        let ver = r.u32()?;
        if ver != VERSION { return Err(format!("proof.bin: version {ver} (this verifier takes version {VERSION}, SHA-512 Merkle)")); }
        let dlen = r.u32()? as usize;
        if dlen != DIGEST_BYTES { return Err(format!("proof.bin: digest size {dlen}, SHA-512 needs {DIGEST_BYTES}")); }
        let root: Digest = r.take(DIGEST_BYTES)?.try_into().unwrap();
        let n = r.u64()?;
        let msgs = r.exts(n, max_msgs, "message")?;
        let has = r.u64()?;
        let opening = match has {
            0 => None,
            1 => {
                let kk = r.u64()?;
                let w = r.exts(kk, k, "w")?;
                let nq = r.u64()?;
                let qc = r.exts(nq, 2 * k - 1, "qc")?;
                let (tt, rr, dd) = (r.u64()? as usize, r.u64()? as usize, r.u64()? as usize);
                if tt != t || rr != rows || dd != depth { return Err(format!("proof.bin: opening shape t={tt} rows={rr} depth={dd}, statement needs t={t} rows={rows} depth={depth}")); }
                let mut columns = Vec::with_capacity(t);
                for _ in 0..t {
                    let mut values = Vec::with_capacity(rows);
                    for _ in 0..rows {
                        let v = r.u32()?;
                        if v >= P { return Err("ligero: non-canonical column value".into()); }
                        values.push(v);
                    }
                    let mut path = Vec::with_capacity(depth);
                    for _ in 0..depth { path.push(r.take(DIGEST_BYTES)?.try_into().unwrap()); }
                    columns.push(Column { values, path });
                }
                Some(Opening { w, qc, columns })
            }
            _ => return Err("proof.bin: bad opening flag".into()),
        };
        if r.pos != b.len() { return Err("proof.bin: trailing bytes".into()); }
        Ok(Proof { root, msgs, opening, bytes_read: b.len() })
    }
}
