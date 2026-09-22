//! The GPU prover's Fiat-Shamir transcript (`gpu/transcript.py`), reproduced exactly:
//!
//!   state <- SHA-256(domain);  absorb: state <- SHA-256(state || msg)
//!   challenge: digests d_c = SHA-256(state || "chal" || ctr_LE64), ctr += 1 per digest; the eight LE u32 words of
//!   each digest are taken in order and accepted iff < p; six accepted words make one extension challenge; the
//!   words left in the last digest are discarded (a fresh digest starts the next challenge).
//!
//! This differs from the CPU crate's `transcript.rs` (one digest per *coordinate*, first accepted word only), so
//! the two transcripts are not interchangeable even at equal messages.

use crate::field::{Ext, DEG, P};
use crate::sha256::Sha256;

pub struct Transcript { state: [u8; 32], ctr: u64, pub slots: u64 }

impl Transcript {
    pub fn new(domain: &[u8]) -> Self { Transcript { state: Sha256::digest(domain), ctr: 0, slots: 0 } }
    pub fn absorb(&mut self, data: &[u8]) {
        let mut s = Sha256::new();
        s.update(&self.state);
        s.update(data);
        self.state = s.finalize();
    }
    pub fn absorb_ext(&mut self, x: Ext) { self.absorb(&x.to_bytes()); }
    pub fn absorb_exts(&mut self, xs: &[Ext]) {
        let mut buf = Vec::with_capacity(xs.len() * Ext::BYTES);
        for x in xs { buf.extend_from_slice(&x.to_bytes()); }
        self.absorb(&buf);
    }
    fn digest(&mut self) -> [u8; 32] {
        let mut s = Sha256::new();
        s.update(&self.state);
        s.update(b"chal");
        s.update(&self.ctr.to_le_bytes());
        self.ctr += 1;
        s.finalize()
    }
    pub fn challenge(&mut self) -> Ext {
        self.slots += 1;
        let mut c = [0u32; DEG];
        let mut got = 0;
        while got < DEG {
            let d = self.digest();
            for i in 0..8 {
                if got == DEG { break; }
                let v = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap());
                if v < P { c[got] = v; got += 1; }
            }
        }
        Ext(c)
    }
    pub fn challenges(&mut self, n: usize) -> Vec<Ext> { (0..n).map(|_| self.challenge()).collect() }
}
