//! Verifier coins -- the Rust side of `backends/direct/ligerito/transcript.py` (lane ligerito-proto, interface frozen
//! 17:45Z).  Every challenge is a function of ONE coin (round trip) expanded by SHAKE-256: extension elements by
//! `u64 LE mod p` per coordinate, row indices by `u64 LE mod domain` with repeats rejected.
//!
//! * `FiatShamir` (`FiatShamirCoins`): BLAKE2b-256 state chain, `state = blake2b(state || label || len_le64 || data)`
//!   on every absorb; `challenge(label, n)` absorbs (b"challenge", label) then squeezes `shake256(state ||
//!   counter_le64)`; `indices(label, domain, count)` absorbs (b"indices", label || domain_le64) then squeezes.  The
//!   diagnostic mode.
//! * `Live` (`LiveCoins`): one verifier coin (>= 16 bytes, 32 in practice) per round trip; the round's stream is
//!   `shake256(coin || sub_le64)` with `sub` counting the squeezes within that round; absorbs are no-ops (the
//!   verifier binds the prover's messages itself; PROTOCOL.md 8c: challenges are functions of coins alone).  The
//!   `.coins` file of a live session is the coin list, each u32 LE length + bytes, in round order.
//! * `Local` (`LocalCoins(seed)`): deterministic from a seed, ignores absorbs -- diagnostic only.

use crate::field::{Ext, EXT_DEG, P};
use crate::hash::{blake2b_256, shake256};

pub trait Coins {
    /// A prover message.
    fn absorb(&mut self, label: &[u8], data: &[u8]);
    /// Start a new round (coin) with `label`; subsequent `stream` calls belong to it.
    fn begin_round(&mut self, kind: &[u8], label: &[u8], extra: &[u8]) -> Result<(), String>;
    /// `nbytes` from the current round's stream.
    fn stream(&mut self, nbytes: usize) -> Vec<u8>;
    /// Sequential verifier messages so far.
    fn rounds(&self) -> usize;

    /// `n` extension elements; one coin.
    fn challenge(&mut self, label: &[u8], n: usize) -> Result<Vec<Ext>, String> {
        self.begin_round(b"challenge", label, &[])?;
        let raw = self.stream(8 * n * EXT_DEG);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = [0u32; EXT_DEG];
            for d in 0..EXT_DEG {
                let o = 8 * (i * EXT_DEG + d);
                // transcript.py: np.frombuffer(<u8).astype(np.int64) % P -- the 8 bytes are read as a SIGNED 64-bit
                // integer (two's complement) and reduced with a non-negative remainder.  Match it exactly.
                let u = i64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
                c[d] = u.rem_euclid(P as i64) as u32;
            }
            out.push(Ext(c));
        }
        Ok(out)
    }

    /// `count` DISTINCT positions in [0, domain), draw order preserved; one coin (the stream is re-squeezed when the
    /// rejection leaves the set short, as `_distinct_indices` does).
    fn indices(&mut self, label: &[u8], domain: usize, count: usize) -> Result<Vec<usize>, String> {
        if count > domain {
            return Err(format!("indices: {count} distinct positions from a domain of {domain}"));
        }
        self.begin_round(b"indices", label, &(domain as u64).to_le_bytes())?;
        let mut out = Vec::with_capacity(count);
        let mut seen = std::collections::HashSet::with_capacity(count);
        while out.len() < count {
            let raw = self.stream(8 * 2 * count);
            for ch in raw.chunks_exact(8) {
                let u = u64::from_le_bytes(ch.try_into().unwrap());
                let i = (u % domain as u64) as usize;
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
    rounds: usize,
}

impl FiatShamir {
    pub const LABEL: &'static [u8] = b"ligerito-proto/v1";

    pub fn new(label: &[u8]) -> FiatShamir {
        FiatShamir { state: blake2b_256(label), counter: 0, rounds: 0 }
    }

    pub fn state(&self) -> [u8; 32] {
        self.state
    }
}

impl Coins for FiatShamir {
    fn absorb(&mut self, label: &[u8], data: &[u8]) {
        let mut buf = Vec::with_capacity(32 + label.len() + 8 + data.len());
        buf.extend_from_slice(&self.state);
        buf.extend_from_slice(label);
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(data);
        self.state = blake2b_256(&buf);
    }

    fn begin_round(&mut self, kind: &[u8], label: &[u8], extra: &[u8]) -> Result<(), String> {
        self.rounds += 1;
        let mut l = label.to_vec();
        l.extend_from_slice(extra);
        self.absorb(kind, &l);
        Ok(())
    }

    fn stream(&mut self, nbytes: usize) -> Vec<u8> {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&self.state);
        buf[32..].copy_from_slice(&self.counter.to_le_bytes());
        self.counter += 1;
        shake256(&buf, nbytes)
    }

    fn rounds(&self) -> usize {
        self.rounds
    }
}

/// Verifier coins of a live session (or their replay).
pub struct Live {
    coins: Vec<Vec<u8>>,
    next: usize,
    sub: u64,
}

impl Live {
    pub fn new(coins: Vec<Vec<u8>>) -> Live {
        Live { coins, next: 0, sub: 0 }
    }

    /// `.coins` file: repeated (u32 LE length, bytes).
    pub fn parse(raw: &[u8]) -> Result<Live, String> {
        let mut coins = Vec::new();
        let mut pos = 0;
        while pos < raw.len() {
            if pos + 4 > raw.len() {
                return Err("coins: truncated length".into());
            }
            let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + n > raw.len() {
                return Err("coins: truncated coin".into());
            }
            coins.push(raw[pos..pos + n].to_vec());
            pos += n;
        }
        Ok(Live::new(coins))
    }

    pub fn serialize(coins: &[Vec<u8>]) -> Vec<u8> {
        let mut out = Vec::new();
        for s in coins {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s);
        }
        out
    }

    pub fn exhausted(&self) -> bool {
        self.next == self.coins.len()
    }

    /// `count` pseudo-random 32-byte coins (tests / selftest).
    pub fn random_coins(count: usize, seed: u64) -> Vec<Vec<u8>> {
        (0..count as u64).map(|i| shake256(&[seed.to_le_bytes(), i.to_le_bytes()].concat(), 32)).collect()
    }
}

impl Coins for Live {
    fn absorb(&mut self, _label: &[u8], _data: &[u8]) {}

    fn begin_round(&mut self, _kind: &[u8], _label: &[u8], _extra: &[u8]) -> Result<(), String> {
        let c = self.coins.get(self.next).ok_or_else(|| format!("coins: round {} has no verifier coin (replay exhausted)", self.next))?;
        if c.len() < 16 {
            return Err(format!("coins: coin {} is {} bytes, a verifier coin must be at least 16", self.next, c.len()));
        }
        self.next += 1;
        self.sub = 0;
        Ok(())
    }

    fn stream(&mut self, nbytes: usize) -> Vec<u8> {
        let coin = &self.coins[self.next - 1];
        let mut buf = coin.clone();
        buf.extend_from_slice(&self.sub.to_le_bytes());
        self.sub += 1;
        shake256(&buf, nbytes)
    }

    fn rounds(&self) -> usize {
        self.next
    }
}

/// `LocalCoins(seed)`: stream = shake256(b"local-coins" || seed_le64 || counter_le64), one counter per squeeze.
pub struct Local {
    seed: u64,
    counter: u64,
    rounds: usize,
}

impl Local {
    pub fn new(seed: u64) -> Local {
        Local { seed, counter: 0, rounds: 0 }
    }
}

impl Coins for Local {
    fn absorb(&mut self, _label: &[u8], _data: &[u8]) {}
    fn begin_round(&mut self, _kind: &[u8], _label: &[u8], _extra: &[u8]) -> Result<(), String> {
        self.rounds += 1;
        Ok(())
    }
    fn stream(&mut self, nbytes: usize) -> Vec<u8> {
        let mut buf = b"local-coins".to_vec();
        buf.extend_from_slice(&self.seed.to_le_bytes());
        buf.extend_from_slice(&self.counter.to_le_bytes());
        self.counter += 1;
        shake256(&buf, nbytes)
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
}

/// ligerito-sumcheck's diagnostic `sumcheck.LocalCoins(seed)`: `challenge(label, n)` = shake256(seed || b"|" || label || b"|" ||
/// rounds_le32) squeezed to 8 n 6 bytes, each coordinate reduced UNSIGNED (`astype(object) % P`); absorbs ignored.  Its
/// `indices` is not used by the sumchecks.  Only for that lane's fixtures.
pub struct SumcheckLocal {
    seed: Vec<u8>,
    rounds: u32,
    label: Vec<u8>,
}

impl SumcheckLocal {
    pub fn new(seed: &[u8]) -> SumcheckLocal {
        SumcheckLocal { seed: seed.to_vec(), rounds: 0, label: Vec::new() }
    }
}

impl Coins for SumcheckLocal {
    fn absorb(&mut self, _label: &[u8], _data: &[u8]) {}
    fn begin_round(&mut self, _kind: &[u8], label: &[u8], _extra: &[u8]) -> Result<(), String> {
        self.label = label.to_vec();
        Ok(())
    }
    fn stream(&mut self, nbytes: usize) -> Vec<u8> {
        let mut buf = self.seed.clone();
        buf.push(b'|');
        buf.extend_from_slice(&self.label);
        buf.push(b'|');
        buf.extend_from_slice(&self.rounds.to_le_bytes());
        self.rounds += 1;
        shake256(&buf, nbytes)
    }
    fn rounds(&self) -> usize {
        self.rounds as usize
    }
    fn challenge(&mut self, label: &[u8], n: usize) -> Result<Vec<Ext>, String> {
        self.begin_round(b"challenge", label, &[])?;
        let raw = self.stream(8 * n * EXT_DEG);
        Ok((0..n)
            .map(|i| {
                let mut c = [0u32; EXT_DEG];
                for d in 0..EXT_DEG {
                    let o = 8 * (i * EXT_DEG + d);
                    let u = u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
                    c[d] = (u % P as u64) as u32;
                }
                Ext(c)
            })
            .collect())
    }
}

/// A recorded transcript (ligerito-sumcheck's `RecordingCoins` log): every `absorb` must match the recorded label and
/// bytes, every `challenge` returns the recorded values.  The first mismatch is reported by the next coin draw (and by
/// `finish`), so a verifier that absorbs the wrong bytes is caught even though `absorb` cannot fail.
pub enum Recorded {
    Absorb { label: Vec<u8>, data: Vec<u8> },
    Challenge { label: Vec<u8>, values: Vec<Ext> },
}

pub struct Scripted {
    log: Vec<Recorded>,
    pos: usize,
    rounds: usize,
    error: Option<String>,
}

impl Scripted {
    pub fn new(log: Vec<Recorded>) -> Scripted {
        Scripted { log, pos: 0, rounds: 0, error: None }
    }

    /// Ok iff every recorded entry was consumed and every absorb matched.
    pub fn finish(&self) -> Result<(), String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        if self.pos != self.log.len() {
            return Err(format!("scripted coins: {} of {} entries consumed", self.pos, self.log.len()));
        }
        Ok(())
    }
}

impl Coins for Scripted {
    fn absorb(&mut self, label: &[u8], data: &[u8]) {
        if self.error.is_some() {
            return;
        }
        match self.log.get(self.pos) {
            Some(Recorded::Absorb { label: l, data: d }) if l == label && d == data => self.pos += 1,
            Some(Recorded::Absorb { label: l, data: d }) => {
                self.error = Some(format!(
                    "scripted coins: absorb {:?} ({} bytes) at entry {} but the record has {:?} ({} bytes){}",
                    String::from_utf8_lossy(label),
                    data.len(),
                    self.pos,
                    String::from_utf8_lossy(l),
                    d.len(),
                    if l == label { " -- bytes differ" } else { "" }
                ))
            }
            Some(Recorded::Challenge { label: l, .. }) => {
                self.error = Some(format!("scripted coins: absorb {:?} at entry {} where the record draws {:?}", String::from_utf8_lossy(label), self.pos, String::from_utf8_lossy(l)))
            }
            None => self.error = Some(format!("scripted coins: absorb {:?} after the end of the record", String::from_utf8_lossy(label))),
        }
    }
    fn begin_round(&mut self, _kind: &[u8], _label: &[u8], _extra: &[u8]) -> Result<(), String> {
        Err("scripted coins: only challenge() is recorded".into())
    }
    fn stream(&mut self, _nbytes: usize) -> Vec<u8> {
        Vec::new()
    }
    fn rounds(&self) -> usize {
        self.rounds
    }
    fn challenge(&mut self, label: &[u8], n: usize) -> Result<Vec<Ext>, String> {
        if let Some(e) = &self.error {
            return Err(e.clone());
        }
        match self.log.get(self.pos) {
            Some(Recorded::Challenge { label: l, values }) if l == label && values.len() == n => {
                self.pos += 1;
                self.rounds += 1;
                Ok(values.clone())
            }
            Some(Recorded::Challenge { label: l, values }) => Err(format!(
                "scripted coins: challenge {:?} x{n} at entry {} but the record has {:?} x{}",
                String::from_utf8_lossy(label),
                self.pos,
                String::from_utf8_lossy(l),
                values.len()
            )),
            Some(Recorded::Absorb { label: l, .. }) => Err(format!("scripted coins: challenge {:?} at entry {} where the record absorbs {:?}", String::from_utf8_lossy(label), self.pos, String::from_utf8_lossy(l))),
            None => Err(format!("scripted coins: challenge {:?} after the end of the record", String::from_utf8_lossy(label))),
        }
    }
}

/// Little-endian u32 serialization of extension elements (`_ext_bytes`).
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
    fn fiat_shamir_state_chain_matches_the_python_definition() {
        let mut t = FiatShamir::new(FiatShamir::LABEL);
        assert_eq!(t.state(), blake2b_256(b"ligerito-proto/v1"));
        let root: Vec<u8> = (0..32u8).collect();
        t.absorb(b"root", &root);
        let mut buf = Vec::new();
        buf.extend_from_slice(&blake2b_256(b"ligerito-proto/v1"));
        buf.extend_from_slice(b"root");
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&root);
        let s1 = blake2b_256(&buf);
        assert_eq!(t.state(), s1);
        // challenge(b"r", 2): absorb (b"challenge", b"r"), then shake256(state || 0)
        let e = t.challenge(b"r", 2).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&s1);
        buf.extend_from_slice(b"challenge");
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"r");
        let s2 = blake2b_256(&buf);
        assert_eq!(t.state(), s2);
        let mut sbuf = s2.to_vec();
        sbuf.extend_from_slice(&0u64.to_le_bytes());
        let raw = shake256(&sbuf, 8 * 2 * 6);
        for i in 0..12 {
            // signed read, as np.frombuffer(<u8).astype(np.int64) % P
            let u = i64::from_le_bytes(raw[8 * i..8 * i + 8].try_into().unwrap());
            assert_eq!(e[i / 6].0[i % 6], u.rem_euclid(P as i64) as u32);
        }
        // indices(b"S", 1024, 5): absorb (b"indices", b"S" || 1024_le64)
        let idx = t.indices(b"S", 1024, 5).unwrap();
        let mut buf = Vec::new();
        buf.extend_from_slice(&s2);
        buf.extend_from_slice(b"indices");
        buf.extend_from_slice(&9u64.to_le_bytes());
        buf.extend_from_slice(b"S");
        buf.extend_from_slice(&1024u64.to_le_bytes());
        assert_eq!(t.state(), blake2b_256(&buf));
        let mut s = idx.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 5);
        assert_eq!(t.rounds(), 2);
    }

    #[test]
    fn live_coins_replay_and_exhaust() {
        let coins = Live::random_coins(3, 9);
        let bytes = Live::serialize(&coins);
        let mut a = Live::parse(&bytes).unwrap();
        let mut b = Live::new(coins.clone());
        a.absorb(b"ignored", b"x");
        let ea = a.challenge(b"r", 2).unwrap();
        let eb = b.challenge(b"other-label", 2).unwrap();
        assert_eq!(ea, eb); // labels do not enter live coins
        assert_eq!(a.indices(b"S", 50, 40).unwrap(), b.indices(b"S", 50, 40).unwrap());
        // the stream derivation: shake256(coin || 0)
        let mut buf = coins[0].clone();
        buf.extend_from_slice(&0u64.to_le_bytes());
        let raw = shake256(&buf, 8 * 12);
        let u = i64::from_le_bytes(raw[..8].try_into().unwrap());
        assert_eq!(ea[0].0[0], u.rem_euclid(P as i64) as u32);
        a.challenge(b"z", 1).unwrap();
        assert!(a.exhausted());
        assert!(a.challenge(b"z", 1).is_err());
        let mut short = Live::new(vec![vec![1u8; 8]]);
        assert!(short.challenge(b"z", 1).is_err());
    }

    #[test]
    fn local_coins_are_seed_only() {
        let mut a = Local::new(5);
        let mut b = Local::new(5);
        a.absorb(b"x", b"y");
        assert_eq!(a.challenge(b"p", 3).unwrap(), b.challenge(b"q", 3).unwrap());
        assert_ne!(Local::new(6).challenge(b"p", 1).unwrap(), Local::new(5).challenge(b"p", 1).unwrap());
    }
}
