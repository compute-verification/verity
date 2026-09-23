//! Verifier coins.  Every challenge the verifier issues is a function of ONE coin slot (a byte string) expanded by
//! SHAKE-256: extension elements by `u64 LE mod p` per coordinate, row indices by `u64 LE mod m` with repeats
//! rejected.  Where the slot bytes come from is the `Coins` implementation:
//!
//! * `FiatShamir` -- the diagnostic mode: a BLAKE2b-256 state chain absorbing every prover message
//!   (`state = blake2b(state || tag || len_le64 || data)`), slot = `shake256(state || counter_le64)`; this is
//!   `backends/direct/ligerito/proto/transcript.py` (label `ligerito-proto/v1`) until `transcript.py`'s Coins
//!   protocol is posted, after which the tags / label are aligned there.
//! * `Replay` -- the interactive mode: the verifier's own coin slots recorded during a live session (a `.coins`
//!   file: for every coin, u32 LE length + bytes, in issue order); the transcript absorbs are then irrelevant and
//!   challenges are functions of the coins alone (PROTOCOL.md 8c).

use crate::field::{Ext, EXT_DEG, P};
use crate::hash::{blake2b_256, shake256};

pub trait Coins {
    /// Absorb a prover message (FS only; replay ignores it).
    fn absorb(&mut self, tag: &[u8], data: &[u8]);
    /// The next coin slot: `nbytes` bytes.
    fn slot(&mut self, nbytes: usize) -> Result<Vec<u8>, String>;
    /// Number of coins issued so far.
    fn coins(&self) -> usize;

    /// `count` extension elements from one coin.
    fn squeeze_ext(&mut self, count: usize) -> Result<Vec<Ext>, String> {
        let raw = self.slot(8 * count * EXT_DEG)?;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let mut c = [0u32; EXT_DEG];
            for d in 0..EXT_DEG {
                let o = 8 * (i * EXT_DEG + d);
                let u = u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
                c[d] = (u % P as u64) as u32;
            }
            out.push(Ext(c));
        }
        Ok(out)
    }

    /// `count` DISTINCT indices in [0, n) from one coin (draw order preserved).  The slot is 2 x count u64s; when
    /// the rejection leaves the set short (probability ~ (count/n)^count) more coins are drawn, as the Python does.
    fn squeeze_indices(&mut self, n: usize, count: usize) -> Result<Vec<usize>, String> {
        if count > n {
            return Err(format!("squeeze_indices: {count} distinct indices from {n} rows"));
        }
        let mut out = Vec::with_capacity(count);
        let mut seen = std::collections::HashSet::with_capacity(count);
        while out.len() < count {
            let raw = self.slot(8 * 2 * count)?;
            for ch in raw.chunks_exact(8) {
                let u = u64::from_le_bytes(ch.try_into().unwrap());
                let i = (u % n as u64) as usize;
                if seen.insert(i) {
                    out.push(i);
                    if out.len() == count {
                        break;
                    }
                }
            }
        }
        Ok(out)
    }
}

pub struct FiatShamir {
    state: [u8; 32],
    counter: u64,
    coins: usize,
}

impl FiatShamir {
    pub const LABEL: &'static [u8] = b"ligerito-proto/v1";

    pub fn new(label: &[u8]) -> FiatShamir {
        FiatShamir { state: blake2b_256(label), counter: 0, coins: 0 }
    }

    pub fn state(&self) -> [u8; 32] {
        self.state
    }
}

impl Coins for FiatShamir {
    fn absorb(&mut self, tag: &[u8], data: &[u8]) {
        let mut buf = Vec::with_capacity(32 + tag.len() + 8 + data.len());
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(tag);
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
        self.state = blake2b_256(&buf);
    }

    fn slot(&mut self, nbytes: usize) -> Result<Vec<u8>, String> {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&self.state);
        buf[32..].copy_from_slice(&self.counter.to_le_bytes());
        self.counter += 1;
        self.coins += 1;
        Ok(shake256(&buf, nbytes))
    }

    fn coins(&self) -> usize {
        self.coins
    }
}

/// Recorded coin slots (interactive replay).
pub struct Replay {
    slots: Vec<Vec<u8>>,
    next: usize,
}

impl Replay {
    pub fn new(slots: Vec<Vec<u8>>) -> Replay {
        Replay { slots, next: 0 }
    }

    /// `.coins` file: repeated (u32 LE length, bytes).
    pub fn parse(raw: &[u8]) -> Result<Replay, String> {
        let mut slots = Vec::new();
        let mut pos = 0;
        while pos < raw.len() {
            if pos + 4 > raw.len() {
                return Err("coins: truncated length".into());
            }
            let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + n > raw.len() {
                return Err("coins: truncated slot".into());
            }
            slots.push(raw[pos..pos + n].to_vec());
            pos += n;
        }
        Ok(Replay { slots, next: 0 })
    }

    pub fn serialize(slots: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for s in slots {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s);
        }
        out
    }

    pub fn exhausted(&self) -> bool {
        self.next == self.slots.len()
    }
}

impl Coins for Replay {
    fn absorb(&mut self, _tag: &[u8], _data: &[u8]) {}

    fn slot(&mut self, nbytes: usize) -> Result<Vec<u8>, String> {
        let s = self.slots.get(self.next).ok_or_else(|| format!("coins: slot {} missing (replay exhausted)", self.next))?;
        if s.len() != nbytes {
            return Err(format!("coins: slot {} holds {} bytes, the protocol needs {nbytes}", self.next, s.len()));
        }
        self.next += 1;
        Ok(s.clone())
    }

    fn coins(&self) -> usize {
        self.next
    }
}

/// A Fiat-Shamir transcript that also records its slots (to write a `.coins` file for the replay tests).
pub struct Recording<C: Coins> {
    pub inner: C,
    pub slots: Vec<Vec<u8>>,
}

impl<C: Coins> Recording<C> {
    pub fn new(inner: C) -> Recording<C> {
        Recording { inner, slots: Vec::new() }
    }
}

impl<C: Coins> Coins for Recording<C> {
    fn absorb(&mut self, tag: &[u8], data: &[u8]) {
        self.inner.absorb(tag, data)
    }
    fn slot(&mut self, nbytes: usize) -> Result<Vec<u8>, String> {
        let s = self.inner.slot(nbytes)?;
        self.slots.push(s.clone());
        Ok(s)
    }
    fn coins(&self) -> usize {
        self.inner.coins()
    }
}

/// Little-endian u32 serialization of extension elements (the transcript's absorb payload for vectors).
pub fn ext_bytes(v: &[Ext]) -> Vec<u8> {
    let mut out = Vec::with_capacity(24 * v.len());
    for e in v {
        out.extend_from_slice(&e.to_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fiat_shamir_matches_python_transcript() {
        // backends/direct/ligerito/proto/transcript.py:
        //   t = Transcript(); t.absorb(b"root", bytes(range(32))); t.squeeze_ext(2); t.squeeze_indices(1024, 5)
        // state after the absorb / first ext / indices -- pinned from hashlib (see the note's `## Pins`).
        let mut t = FiatShamir::new(FiatShamir::LABEL);
        assert_eq!(crate::hash::hex(&t.state()), crate::hash::hex(&blake2b_256(b"ligerito-proto/v1")));
        let root: Vec<u8> = (0..32u8).collect();
        t.absorb(b"root", &root);
        let mut buf = Vec::new();
        buf.extend_from_slice(&blake2b_256(b"ligerito-proto/v1"));
        buf.extend_from_slice(b"root");
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&root);
        assert_eq!(t.state(), blake2b_256(&buf));
        let e = t.squeeze_ext(2).unwrap();
        let mut sbuf = Vec::new();
        sbuf.extend_from_slice(&t.state());
        sbuf.extend_from_slice(&0u64.to_le_bytes());
        let raw = shake256(&sbuf, 8 * 2 * 6);
        let u0 = u64::from_le_bytes(raw[..8].try_into().unwrap());
        assert_eq!(e[0].0[0], (u0 % P as u64) as u32);
        let idx = t.squeeze_indices(1024, 5).unwrap();
        assert_eq!(idx.len(), 5);
        let mut s = idx.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 5);
        assert_eq!(t.coins(), 2);
    }

    #[test]
    fn replay_roundtrip() {
        let mut rec = Recording::new(FiatShamir::new(FiatShamir::LABEL));
        rec.absorb(b"x", b"hello");
        let a = rec.squeeze_ext(3).unwrap();
        let b = rec.squeeze_indices(100, 7).unwrap();
        let bytes = Replay::serialize(&rec.slots);
        let mut rp = Replay::parse(&bytes).unwrap();
        rp.absorb(b"ignored", b"");
        assert_eq!(rp.squeeze_ext(3).unwrap(), a);
        assert_eq!(rp.squeeze_indices(100, 7).unwrap(), b);
        assert!(rp.exhausted());
        assert!(rp.squeeze_ext(1).is_err());
        let mut rp2 = Replay::parse(&bytes).unwrap();
        assert!(rp2.squeeze_ext(4).is_err()); // wrong slot length
    }
}
