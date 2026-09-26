//! Zero-knowledge readiness for `verity/flock-circuit` (lane flock-circuit, M0): the masking hooks the prover API
//! carries now, as no-ops, and the prover's own randomness.
//!
//! The prover is a deterministic function of (statement, witness, [`ProverSeed`], verifier coins): the seed is an explicit input,
//! drawn from `getrandom` in a proving build and injectable only by a test harness (feature `seed-injection`), and no message byte
//! depends on thread scheduling (every reduction is a GF(2^128) sum, order-free) or on unseeded device state; the selftest
//! `prover_is_deterministic` checks proofs and transcript byte for byte.
//!
//! [`ProverRng`] is the only source of the prover's masking randomness (masks, salts, Reed–Solomon padding): a fresh 256-bit
//! seed per proof from the operating system (`getrandom`), expanded with ChaCha20 as independent streams addressed by
//! `(purpose, index)` (`set_stream` / `set_word_pos`), so a device kernel can expand the same words later. It never reads the
//! transcript, and the only constructor outside tests is [`ProverRng::from_os`]: no fixed or test seed is reachable in a
//! production build.
//!
//! [`Hooks`] is where M1's masking goes: a sumcheck round hook (`+ρ·g`), the Reed–Solomon padding `t_pad` of each Ligerito
//! level, a permutation of the claim points, and the mask slot's fill. [`NoMask`] implements them as the identity, and
//! [`Hooks::identity`] is what the statement and the backend identity record, so a NON_ZK and a ZK cell never share a ratio.

use flock_field::F128;
use rand_chacha::{
    ChaCha20Rng,
    rand_core::{RngCore, SeedableRng},
};
use serde_json::{Value, json};

/// ChaCha20 stream numbers, one per purpose.
pub const STREAM_MASK: u64 = 1;
pub const STREAM_SALT: u64 = 2;
pub const STREAM_PAD: u64 = 3;

/// The prover's seed: an explicit input of every proof, so the prover is a function of (statement, witness, seed, verifier coins).
/// A proving build can only draw it from the OS; a test harness (feature `seed-injection`, or unit tests) can inject one.
#[derive(Clone, Copy)]
pub struct ProverSeed([u8; 32]);

impl ProverSeed {
    pub fn from_os() -> Self {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("the OS random generator failed: refusing to prove without prover randomness");
        ProverSeed(seed)
    }

    #[cfg(any(test, feature = "seed-injection"))]
    pub fn injected(seed: [u8; 32]) -> Self {
        ProverSeed(seed)
    }
}

pub struct ProverRng {
    rng: ChaCha20Rng,
}

impl ProverRng {
    pub fn from_seed(seed: ProverSeed) -> Self {
        ProverRng { rng: ChaCha20Rng::from_seed(seed.0) }
    }

    /// A fresh proof's randomness from the OS, never from the transcript.
    pub fn from_os() -> Self {
        Self::from_seed(ProverSeed::from_os())
    }

    /// Bytes `[32 index, 32 index + out.len())` of stream `purpose` (independent of every other `(purpose, index)`).
    pub fn fill_at(&mut self, purpose: u64, index: u64, out: &mut [u8]) {
        self.rng.set_stream(purpose);
        self.rng.set_word_pos(8 * index as u128);
        self.rng.fill_bytes(out);
    }

    /// A 32-byte leaf salt (leaf `i` of Merkle tree `tree`).
    pub fn salt(&mut self, tree: u32, i: u64) -> [u8; 32] {
        let mut s = [0u8; 32];
        self.fill_at(STREAM_SALT, ((tree as u64) << 40) | i, &mut s);
        s
    }

    pub fn f128s(&mut self, purpose: u64, index: u64, n: usize) -> Vec<F128> {
        let mut b = vec![0u8; 16 * n];
        self.fill_at(purpose, index, &mut b);
        b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
    }
}

/// The prover's masking hooks. M0 implements them as the identity ([`NoMask`]).
pub trait Hooks {
    /// Called with each sumcheck round's message before it is sent (M1: add `ρ·g`, g the round's masking polynomial).
    fn sumcheck_round(&mut self, _round: usize, _msg: &mut [F128]) {}
    /// Reed–Solomon padding rows per Ligerito level (M1: at least the level's query count).
    fn t_pad(&self) -> usize {
        0
    }
    /// The claim points in the order they are opened (M1: a permutation keeping masks reachable).
    fn claim_points(&mut self, pts: Vec<Vec<F128>>) -> Vec<Vec<F128>> {
        pts
    }
    /// Fill block `block`'s mask slot (`words` 128-bit words of free cells).
    fn mask_fill(&mut self, _block: usize, words: usize) -> Vec<F128> {
        vec![F128::ZERO; words]
    }
    /// Whether the proof is zero knowledge: part of the statement and the backend identity.
    fn zk(&self) -> bool {
        false
    }
    fn identity(&self) -> Value {
        json!({"zk": self.zk(), "t_pad": self.t_pad(), "masks": "none", "sumcheck_mask": "none", "claim_permutation": "identity"})
    }
}

/// M0: every hook the identity; the prover randomness drawn per proof all the same (it salts the proof's Merkle leaves).
pub struct NoMask {
    pub rng: ProverRng,
}

impl NoMask {
    pub fn new(seed: ProverSeed) -> Self {
        NoMask { rng: ProverRng::from_seed(seed) }
    }
}

impl Hooks for NoMask {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_addressed_and_independent() {
        let mut r = ProverRng::from_seed(ProverSeed::injected([7; 32]));
        let a = r.salt(0, 5);
        let _ = r.salt(1, 0);
        assert_eq!(r.salt(0, 5), a, "a salt is a function of the seed and its address");
        assert_ne!(r.salt(0, 6), a);
        assert_ne!(r.salt(1, 5), a);
    }

    #[test]
    fn two_os_seeds_differ() {
        assert_ne!(ProverRng::from_os().salt(0, 0), ProverRng::from_os().salt(0, 0));
    }

    /// TODO(M1): two proofs of the same statement share no mask bit: prove one statement twice with masking on, take every
    /// mask cell, Reed–Solomon padding row and leaf salt of each, and assert the two sets are disjoint as bit positions with
    /// equal values no more often than chance (a chi-square on the XOR). Needs M1's masks.
    #[test]
    #[ignore = "M1: needs the masking hooks implemented"]
    fn two_proofs_share_no_mask_bits() {
        unimplemented!("M1");
    }
}
