//! Zero-knowledge readiness for `verity/flock-circuit` (lane flock-circuit, M0): the masking hooks the prover API
//! carries now, as no-ops, and the prover's own randomness.
//!
//! The prover is a deterministic function of (statement, witness, [`ProverSeed`], verifier coins): the seed is an explicit input,
//! drawn from `getrandom` in a proving build and injectable only by a test harness (feature `seed-injection`), and no message byte
//! depends on thread scheduling (every reduction is a GF(2^128) sum, order-free) or on unseeded device state; the selftest
//! `prover_is_deterministic` checks proofs and transcript byte for byte.
//!
//! [`ProverRng`] is the source of the masks and the Reed–Solomon padding: a fresh 256-bit seed per proof from the operating
//! system (`getrandom`), expanded with ChaCha20 as independent streams addressed by `(purpose, index)` (`set_stream` /
//! `set_word_pos`), so a device kernel can expand the same words later. It never reads the transcript, and the only constructor
//! outside tests is [`ProverRng::from_os`]: no fixed or test seed is reachable in a production build.
//!
//! The Merkle leaf salts of `hm96-sha512/v1` are NOT drawn from it: [`LeafSalts`] expands them with ChaCha20 from their own fresh
//! 256-bit OS key per proof (on the device, where the trees are built), modelled as uniform like the OS generator's own ChaCha20
//! output (red-team-hm96 F4). With an injected seed (a test harness only) the key comes from that seed, so a test proof is
//! reproducible. The hm96 key is not prover
//! randomness at all: it is the statement's constant (`circuit::leaf_scheme`).
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

/// ChaCha20 stream numbers, one per purpose (`STREAM_SALT` only for an injected seed's salt tape).
pub const STREAM_MASK: u64 = 1;
pub const STREAM_PAD: u64 = 2;
pub const STREAM_SALT: u64 = 3;
/// Bytes per `hm96-sha512/v1` leaf salt.
pub const SALT_BYTES: usize = 192;

/// The prover's seed: an explicit input of every proof, so the prover is a function of (statement, witness, seed, verifier coins).
/// A proving build can only draw it from the OS; a test harness (feature `seed-injection`, or unit tests) can inject one.
#[derive(Clone, Copy)]
pub struct ProverSeed {
    seed: [u8; 32],
    injected: bool,
}

impl ProverSeed {
    pub fn from_os() -> Self {
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).expect("the OS random generator failed: refusing to prove without prover randomness");
        ProverSeed { seed, injected: false }
    }

    #[cfg(any(test, feature = "seed-injection"))]
    pub fn injected(seed: [u8; 32]) -> Self {
        ProverSeed { seed, injected: true }
    }
}

/// The salts of one proof's `hm96-sha512/v1` Merkle leaves: a ChaCha20 expansion of a 256-bit salt key (Daniel, 2026-09-27;
/// this replaced salts drawn from `getrandom` leaf by leaf, which cost +96% prover time at m = 33). Leaf `i` of tree `id` is
/// ChaCha20 blocks `3i, 3i + 1, 3i + 2` under the key with nonce `id` ([`chacha20_block`]); the level-0 tree is id 0 (both reps
/// bind one root) and every later tree takes the next id, so no salt is used twice. The key is a fresh `getrandom` draw per proof
/// (an injected seed's `STREAM_SALT` bytes in a test harness only), never reused, never written anywhere, and zeroed on drop.
/// The device expands the same stream (`hm96_finish_leaves`).
pub struct LeafSalts {
    key: [u8; 32],
    next: u64,
}

impl LeafSalts {
    pub fn for_seed(seed: &ProverSeed) -> Self {
        let mut key = [0u8; 32];
        if seed.injected {
            let mut r = ChaCha20Rng::from_seed(seed.seed);
            r.set_stream(STREAM_SALT);
            r.fill_bytes(&mut key);
        } else {
            getrandom::fill(&mut key).expect("the OS random generator failed: refusing to commit without a salt key");
        }
        LeafSalts { key, next: 1 }
    }

    /// The salt key, for the device to expand; it stays in memory only.
    pub fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// The id of the next tree after level 0.
    pub fn next_id(&mut self) -> u64 {
        self.next += 1;
        self.next - 1
    }

    /// Tree `id`'s salts, `leaves` × 192 bytes.
    pub fn expand(&self, id: u64, leaves: usize) -> Vec<u8> {
        use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
        let mut b = vec![0u8; leaves * SALT_BYTES];
        b.par_chunks_mut(SALT_BYTES).enumerate().for_each(|(i, s)| {
            for j in 0..3 {
                s[64 * j..64 * j + 64].copy_from_slice(&chacha20_block(&self.key, 3 * i as u64 + j as u64, id));
            }
        });
        b
    }
}

impl Drop for LeafSalts {
    fn drop(&mut self) {
        for b in self.key.iter_mut() {
            unsafe { std::ptr::write_volatile(b, 0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

/// One ChaCha20 block (20 rounds): state = "expand 32-byte k", the key, a 64-bit block counter (words 12-13) and a 64-bit
/// nonce (words 14-15), output little-endian.
pub fn chacha20_block(key: &[u8; 32], counter: u64, nonce: u64) -> [u8; 64] {
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&[0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]);
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes(key[4 * i..4 * i + 4].try_into().unwrap());
    }
    s[12] = counter as u32;
    s[13] = (counter >> 32) as u32;
    s[14] = nonce as u32;
    s[15] = (nonce >> 32) as u32;
    let mut x = s;
    let qr = |x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize| {
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(16);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(12);
        x[a] = x[a].wrapping_add(x[b]);
        x[d] = (x[d] ^ x[a]).rotate_left(8);
        x[c] = x[c].wrapping_add(x[d]);
        x[b] = (x[b] ^ x[c]).rotate_left(7);
    };
    for _ in 0..10 {
        qr(&mut x, 0, 4, 8, 12);
        qr(&mut x, 1, 5, 9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7, 8, 13);
        qr(&mut x, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&x[i].wrapping_add(s[i]).to_le_bytes());
    }
    out
}

pub struct ProverRng {
    rng: ChaCha20Rng,
}

impl ProverRng {
    pub fn from_seed(seed: ProverSeed) -> Self {
        ProverRng { rng: ChaCha20Rng::from_seed(seed.seed) }
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

/// M0: every hook the identity; the prover randomness drawn per proof all the same (the selftests fill the mask slot from it).
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
        let a = r.f128s(STREAM_MASK, 5, 2);
        let _ = r.f128s(STREAM_PAD, 0, 2);
        assert_eq!(r.f128s(STREAM_MASK, 5, 2), a, "the words are a function of the seed and their address");
        assert_ne!(r.f128s(STREAM_MASK, 6, 2), a);
        assert_ne!(r.f128s(STREAM_PAD, 5, 2), a);
    }

    /// RFC 7539 section 2.3.2: key 00..1f, block counter 1, nonce 000000090000004a00000000 (words 13-15).
    #[test]
    fn chacha20_block_matches_rfc7539() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let b = chacha20_block(&key, 1 | (0x0900_0000u64 << 32), 0x4a00_0000);
        assert_eq!(crate::hex(&b[..16]), "10f1e7e4d13b5915500fdd1fa32071c4");
    }

    #[test]
    fn two_os_seeds_differ() {
        assert_ne!(ProverRng::from_os().f128s(STREAM_MASK, 0, 2), ProverRng::from_os().f128s(STREAM_MASK, 0, 2));
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
