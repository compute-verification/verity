//! SHA-256 (FIPS 180-4, in-tree so the crate has no dependencies) and the Fiat-Shamir transcript.
//! Challenges are drawn by rejection sampling so every coordinate is uniform in [0, p).

use crate::field::{Ext, Fp};

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub struct Sha256 {
    h: [u32; 8],
    buf: Vec<u8>,
    len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: Vec::new(),
            len: 0,
        }
    }
    pub fn update(&mut self, data: &[u8]) {
        self.len += data.len() as u64;
        self.buf.extend_from_slice(data);
        let mut off = 0;
        while self.buf.len() - off >= 64 {
            let block: [u8; 64] = self.buf[off..off + 64].try_into().unwrap();
            self.compress(&block);
            off += 64;
        }
        self.buf.drain(..off);
    }
    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (i, v) in [a, b, c, d, e, f, g, h].iter().enumerate() {
            self.h[i] = self.h[i].wrapping_add(*v);
        }
    }
    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.len * 8;
        let mut pad = vec![0x80u8];
        while (self.buf.len() + pad.len()) % 64 != 56 {
            pad.push(0);
        }
        pad.extend_from_slice(&bits.to_be_bytes());
        self.update(&pad);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.h[i].to_be_bytes());
        }
        out
    }
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut s = Sha256::new();
        s.update(data);
        s.finalize()
    }
}

/// One recorded transcript event (`dump-transcript`): what was absorbed, which challenge came out, or a label
/// marking the protocol phase the following events belong to.
#[derive(Clone, Debug)]
pub enum Event {
    Label(String),
    Absorb { bytes: Vec<u8>, state: [u8; 32] },
    Challenge { value: Ext, ctr_before: u64 },
}

/// Fiat-Shamir: state <- H(state || message); challenge_i <- H(state || "chal" || ctr) with rejection sampling
/// (`Fp::rejection_sample` on each digest, coordinate by coordinate, `DEGREE` coordinates per challenge).
/// `slots` counts every challenge drawn (`rounds.protocol` in the one-coin-per-message reading).
///
/// Latency emulation (lane latency, 2026-09-22): in a live proof the protocol is interactive (the verifier's coins
/// arrive over the network, `backends/ZK_CONSTRUCTION.md` 3.3), so every verifier message on the critical path is
/// one network round trip.  The prover marks those points with [`Transcript::round_trip`]: it counts them
/// (`trips` = `rounds.sequential_depth`, measured rather than derived) and, when `rtt` is set (`--rtt-ms`),
/// sleeps one RTT there -- the prover waiting for the challenge.  Only the *critical* chain is marked (the
/// deepest logUp table, the deepest segment: the other instances run in lock-step with it and cost no extra
/// trip, PROTOCOL.md 5.4).  The protocol and the bytes are unchanged; the hook is a delay, not a change.
pub struct Transcript {
    state: [u8; 32],
    ctr: u64,
    pub slots: u64,
    pub log: Option<Vec<Event>>,
    rtt: Option<std::time::Duration>,
    pub trips: u64,
    pub critical: bool,
    pub slept: std::time::Duration,
}

impl Transcript {
    pub fn new(domain: &[u8]) -> Self {
        Transcript {
            state: Sha256::digest(domain),
            ctr: 0,
            slots: 0,
            log: None,
            rtt: None,
            trips: 0,
            critical: false,
            slept: std::time::Duration::ZERO,
        }
    }
    /// Emulate a network with this round-trip time: every critical-path verifier message costs one sleep.
    pub fn set_rtt(&mut self, rtt: Option<std::time::Duration>) {
        self.rtt = rtt;
    }
    /// One verifier message on the critical path (a prover message was just absorbed and the next prover message
    /// depends on the challenge drawn next).  No-op unless `critical` is set for the current chain.
    pub fn round_trip(&mut self) {
        if !self.critical {
            return;
        }
        self.trips += 1;
        if let Some(d) = self.rtt {
            let t = std::time::Instant::now();
            std::thread::sleep(d);
            self.slept += t.elapsed();
        }
    }
    /// Start recording every absorb / challenge / label (for the transcript vectors).
    pub fn with_log(domain: &[u8]) -> Self {
        let mut t = Self::new(domain);
        t.log = Some(vec![Event::Absorb {
            bytes: domain.to_vec(),
            state: t.state,
        }]);
        t
    }
    pub fn label(&mut self, s: &str) {
        if let Some(l) = &mut self.log {
            l.push(Event::Label(s.to_string()));
        }
    }
    pub fn absorb(&mut self, data: &[u8]) {
        let mut s = Sha256::new();
        s.update(&self.state);
        s.update(data);
        self.state = s.finalize();
        if let Some(l) = &mut self.log {
            l.push(Event::Absorb {
                bytes: data.to_vec(),
                state: self.state,
            });
        }
    }
    pub fn absorb_ext(&mut self, x: Ext) {
        self.absorb(&x.to_bytes());
    }
    pub fn absorb_exts(&mut self, xs: &[Ext]) {
        let mut buf = Vec::with_capacity(xs.len() * Ext::BYTES);
        for x in xs {
            buf.extend_from_slice(&x.to_bytes());
        }
        self.absorb(&buf);
    }
    fn squeeze_fp(&mut self) -> Fp {
        loop {
            let mut s = Sha256::new();
            s.update(&self.state);
            s.update(b"chal");
            s.update(&self.ctr.to_le_bytes());
            self.ctr += 1;
            let d = s.finalize();
            if let Some(v) = Fp::rejection_sample(&d) {
                return v;
            }
        }
    }
    pub fn challenge(&mut self) -> Ext {
        self.slots += 1;
        let ctr_before = self.ctr;
        let value = Ext(std::array::from_fn(|_| self.squeeze_fp()));
        if let Some(l) = &mut self.log {
            l.push(Event::Challenge { value, ctr_before });
        }
        value
    }
    pub fn challenges(&mut self, n: usize) -> Vec<Ext> {
        (0..n).map(|_| self.challenge()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_vectors() {
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{:02x}", b)).collect::<String>();
        assert_eq!(
            hex(Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1000];
        assert_eq!(
            hex(Sha256::digest(&long)),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }
}
