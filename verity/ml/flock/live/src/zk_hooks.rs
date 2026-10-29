//! Zero-knowledge readiness for `verity/flock-circuit` (lane flock-circuit, M0): the masking hooks the prover API
//! carries now, as no-ops, and the prover's own randomness.
//!
//! Every random value the prover uses is the operating system's own bytes (`getrandom`), read where it is drawn and kept where it
//! is used again, so no premise on a stream generator is left on any of them (Daniel, 9 Oct, decision 4). Under the outer
//! firewall the Lean server reads them ([`set_os_source`]), and this process only carries them. A test harness (feature
//! `seed-injection`) can inject a [`ProverSeed`] instead, and then the prover is a deterministic function of (statement, witness,
//! seed, verifier coins): no message byte depends on thread scheduling (every reduction is a GF(2^128) sum, order-free) or on
//! unseeded device state; the selftest `prover_is_deterministic` checks proofs and transcript byte for byte.
//!
//! [`ProverRng`] is the source of the masks and the Reed–Solomon padding: the mask slot's words, the pads and their padding,
//! τ's salt, and Ligerito level 0's padding coefficients and extra lanes. In a proving build each sample reads the OS afresh, in
//! [`OS_PIECE`]-byte pieces on rayon's threads, whatever its `(purpose, index)`, so a value used twice is kept by its user: level
//! 0's hiding for the table's session, each rep's pads in the rep's state, each block's mask words with the table's witness
//! sources. Under an injected seed they are ChaCha20 streams of it, addressed by `(purpose, index)` (`set_stream` /
//! `set_word_pos`). It never reads the transcript, and outside tests every constructor takes a [`ProverSeed`], which a proving
//! build can only draw from the OS: no fixed or test seed is reachable in a production build.
//!
//! The Merkle leaf salts of `hm96-sha512/v1` are [`LeafSalts`]'s: the OS's bytes too, read tree by tree (decision 4(a)). With an
//! injected seed they are a ChaCha20 expansion of that seed instead, so a test proof is reproducible. The hm96 key is not prover
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

/// An injected seed's ChaCha20 stream numbers, one per purpose (`STREAM_SALT` for its salt tape); a proving build reads the OS
/// for every purpose.
pub const STREAM_MASK: u64 = 1;
pub const STREAM_PAD: u64 = 2;
pub const STREAM_SALT: u64 = 3;
/// An injected seed's coin-tree nonce (`coin_tree::SCHEME`); a proving build draws the nonce from the OS.
pub const STREAM_COIN_NONCE: u64 = 8;
/// Bytes per `hm96-sha512/v1` leaf salt.
pub const SALT_BYTES: usize = 192;
/// A session of several tables gives each its own randomness (the public-circuit ZK proof's §2.2 and gap 1): in a proving build
/// every table's samples are reads of their own; under an injected seed table `j` reads stream `purpose` at ChaCha20 stream
/// `purpose | j << TABLE_SHIFT`. Its salt trees are ids `j << TABLE_SHIFT` (level 0) and up, below the next table's. Table 0's
/// addresses are a one-table session's.
pub const TABLE_SHIFT: u32 = 32;
/// Words per piece of [`ProverRng::f128s_lanes`]'s parallel draw under an injected seed. Even: an index addresses 32 bytes,
/// two words, so every piece starts at a whole index.
const LANE_CHUNK: usize = 1 << 12;

/// The prover's seed: an explicit input of every proof. A proving build can only draw it from the OS, and then expands nothing
/// from it (every random value is read from the OS where it is sampled). A test harness (feature `seed-injection`, or unit tests)
/// can inject one, which the prover expands with ChaCha20, so it is a function of (statement, witness, seed, verifier coins).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ProverSeed {
    seed: [u8; 32],
    injected: bool,
}

impl ProverSeed {
    pub fn from_os() -> Self {
        let mut seed = [0u8; 32];
        os_fill(&mut seed);
        ProverSeed { seed, injected: false }
    }

    #[cfg(any(test, feature = "seed-injection"))]
    pub fn injected(seed: [u8; 32]) -> Self {
        ProverSeed { seed, injected: true }
    }
}

/// The salts of one proof's `hm96-sha512/v1` Merkle leaves: every leaf's 192 bytes read from the OS (`getrandom`, the source of
/// [`ProverSeed::from_os`]), a tree at a time in [`OS_PIECE`]-byte pieces on rayon's threads, never a stream generator's output
/// (Daniel, 9 Oct, decision 4(a)). They replace a ChaCha20 expansion of one 32-byte key per proof, chosen on 27 Sep because
/// salts read with one call per leaf cost +96% prover time at m = 33. Trees are numbered: the level-0 tree is id 0 (both reps
/// bind one root) and every later tree takes the next id. In a session of several tables, table `j` ([`LeafSalts::table`]) takes
/// ids from `j << TABLE_SHIFT` instead. A tree's salts are read once, so no salt is used twice and every commitment to a tree
/// sees one set: level 0's are kept for the proof ([`LeafSalts::level0`]: the session's `Commit`, both reps and the ZK server's
/// recheck all commit that tree), and a tree whose salts the ZK server's Lean finish drew ([`LeafSalts::preset`]) is kept
/// until the device's tree takes them (zeroed if dropped first). Salts are never written to a record. An
/// injected seed (a test harness only) expands them with ChaCha20 instead, leaf `i` of tree `id` being blocks
/// `3i, 3i + 1, 3i + 2` under its `STREAM_SALT` key with nonce `id` ([`chacha20_block`]), so a test proof repeats.
pub struct LeafSalts {
    /// an injected seed's salt key (on the heap, so moving a `LeafSalts` copies no key byte); `None` in every proving build
    key: Option<Box<[u8; 32]>>,
    /// the trees kept
    kept: std::sync::Arc<std::sync::Mutex<Kept>>,
    next: u64,
    level0: u64,
}

/// Bytes per `getrandom` call when [`LeafSalts`] reads a tree's salts.
pub const OS_PIECE: usize = 1 << 16;

#[derive(Default)]
struct Kept {
    level0: Option<std::sync::Arc<Vec<u8>>>,
    /// trees whose salts were sampled elsewhere ([`LeafSalts::preset`]), by tree id and leaf count, taken by the first draw
    preset: std::collections::BTreeMap<(u64, usize), ZeroOnDrop>,
}

/// The coin-tree nonce the prover sends in `Hello` (docs/coin-tree-v2.md §3): 32 fresh bytes from the OS, once per session,
/// never derived from the verifier's messages; an injected seed (a test harness) derives it, so a harness's sessions repeat.
pub fn coin_nonce(seed: &ProverSeed) -> [u8; 32] {
    let mut n = [0u8; 32];
    if seed.injected {
        let mut r = ChaCha20Rng::from_seed(seed.seed);
        r.set_stream(STREAM_COIN_NONCE);
        r.fill_bytes(&mut n);
    } else {
        os_fill(&mut n);
    }
    n
}

impl LeafSalts {
    pub fn for_seed(seed: &ProverSeed) -> Self {
        let key = seed.injected.then(|| {
            let mut key = Box::new([0u8; 32]);
            let mut r = ChaCha20Rng::from_seed(seed.seed);
            r.set_stream(STREAM_SALT);
            r.fill_bytes(&mut key[..]);
            key
        });
        LeafSalts { key, kept: Default::default(), next: 1, level0: 0 }
    }

    /// Table `table`'s salts in a session of several tables: the level-0 tree `table << TABLE_SHIFT` and every later tree the
    /// next id, below the next table's level 0 (table 0: [`Self::for_seed`]'s ids), with trees of its own.
    pub fn table(&self, table: u64) -> LeafSalts {
        assert!(table < 1 << TABLE_SHIFT, "a table index below 2^{TABLE_SHIFT}");
        let level0 = table << TABLE_SHIFT;
        LeafSalts { key: self.key.clone(), kept: Default::default(), next: level0 + 1, level0 }
    }

    /// The level-0 tree's id (both reps of a table bind its one root).
    pub fn level0_id(&self) -> u64 {
        self.level0
    }

    /// The id of the next tree after level 0.
    pub fn next_id(&mut self) -> u64 {
        assert!(self.next - self.level0 < 1 << TABLE_SHIFT, "a table's salt trees stay below the next table's level 0");
        self.next += 1;
        self.next - 1
    }

    /// The level-0 tree's salts, `leaves` × 192 bytes: read on the first call, the same bytes on every later one.
    pub fn level0(&self, leaves: usize) -> std::sync::Arc<Vec<u8>> {
        let mut k = self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = &k.level0 {
            assert_eq!(s.len(), leaves * SALT_BYTES, "level 0's salts are read for one tree size");
            return s.clone();
        }
        let s = std::sync::Arc::new(self.draw(self.level0, leaves));
        k.level0 = Some(s.clone());
        s
    }

    /// The level-0 tree's salts drawn elsewhere (the Lean session loop's, `Flock.Firewall.Level0`), before anything reads them:
    /// every later [`Self::level0`] answers these bytes.
    pub fn set_level0(&self, s: Vec<u8>) -> Result<(), String> {
        let mut k = self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if k.level0.is_some() {
            return Err("level 0's salts were read before the Lean loop drew them".into());
        }
        k.level0 = Some(std::sync::Arc::new(s));
        Ok(())
    }

    /// The salts of the next trees from here, drawn elsewhere (the Lean finish's for a firewalled rep's levels ≥ 1,
    /// `Flock.Firewall.Finish.hiSaltsOf`), in id order, each `leaves` × 192 bytes: the first draw of each such tree at that
    /// size takes them in place of its own read.
    pub fn preset(&self, trees: Vec<Vec<u8>>) -> Result<(), String> {
        let mut k = self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (j, s) in trees.into_iter().enumerate() {
            if s.is_empty() || s.len() % SALT_BYTES != 0 {
                return Err(format!("preset salts of {} bytes for a tree, not whole {SALT_BYTES}-byte leaves", s.len()));
            }
            let id = self.next + j as u64;
            k.preset.insert((id, s.len() / SALT_BYTES), ZeroOnDrop(s));
        }
        Ok(())
    }

    /// Tree `id`'s salts, `leaves` × 192 bytes (level 0's: [`Self::level0`]'s): its preset salts at that size, once, else read
    /// afresh. A device that samples a tree at another size than the Lean finish's gets salts of its own, its tree is not the
    /// finish's and its opening fails the ZK server's check, and nothing panics.
    pub fn expand(&self, id: u64, leaves: usize) -> Vec<u8> {
        if id == self.level0 {
            return self.level0(leaves).to_vec();
        }
        let preset = self.kept.lock().unwrap_or_else(std::sync::PoisonError::into_inner).preset.remove(&(id, leaves));
        match preset {
            Some(mut p) => std::mem::take(&mut p.0),
            None => self.draw(id, leaves),
        }
    }

    fn draw(&self, id: u64, leaves: usize) -> Vec<u8> {
        use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
        let mut b = vec![0u8; leaves * SALT_BYTES];
        match &self.key {
            Some(key) => b.par_chunks_mut(SALT_BYTES).enumerate().for_each(|(i, s)| {
                for j in 0..3 {
                    s[64 * j..64 * j + 64].copy_from_slice(&chacha20_block(key, 3 * i as u64 + j as u64, id));
                }
            }),
            None => os_fill(&mut b),
        }
        b
    }
}

/// Where a proving build's samples read the OS's bytes when not from `getrandom` itself: under the outer ZK server, the Lean
/// server's `flock-firewall tap` (`firewall::lean_tap`, PROTOCOL.md §10.4), set once before the process's first draw.
static OS_SOURCE: std::sync::OnceLock<std::sync::Mutex<Box<dyn FnMut(&mut [u8]) + Send>>> = std::sync::OnceLock::new();

/// Makes `source` the OS bytes of every later draw of this process; `Err` when a source is already set.
pub fn set_os_source(source: Box<dyn FnMut(&mut [u8]) + Send>) -> Result<(), String> {
    OS_SOURCE.set(std::sync::Mutex::new(source)).map_err(|_| "the prover's OS source is already set".to_string())
}

/// `out` read from the OS: from [`set_os_source`]'s source if one is set, else `getrandom` in [`OS_PIECE`]-byte pieces on
/// rayon's threads.
fn os_fill(out: &mut [u8]) {
    use rayon::prelude::{ParallelIterator, ParallelSliceMut};
    if let Some(s) = OS_SOURCE.get() {
        return (s.lock().unwrap_or_else(std::sync::PoisonError::into_inner))(out);
    }
    out.par_chunks_mut(OS_PIECE)
        .for_each(|p| getrandom::fill(p).expect("the OS random generator failed: refusing to prove without prover randomness"));
}

fn words(b: &[u8]) -> Vec<F128> {
    b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
}

impl Drop for LeafSalts {
    fn drop(&mut self) {
        if let Some(key) = self.key.as_mut() {
            zero(&mut key[..]);
        }
    }
}

fn zero(b: &mut [u8]) {
    for x in b.iter_mut() {
        unsafe { std::ptr::write_volatile(x, 0) };
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// A byte buffer holding key material, zeroed when dropped (a panic unwinding past it included).
pub struct ZeroOnDrop(pub Vec<u8>);

impl Drop for ZeroOnDrop {
    fn drop(&mut self) {
        zero(&mut self.0);
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

/// The prover's random words: in a proving build the OS's, read afresh by every sample (two draws at one address get different
/// words); under an injected seed, ChaCha20 stream `purpose | table << TABLE_SHIFT` from word `8 index`, a function of the seed
/// and the address alone.
pub struct ProverRng {
    /// an injected seed's generator; `None` in every proving build
    rng: Option<ChaCha20Rng>,
    table: u64,
}

impl ProverRng {
    pub fn from_seed(seed: ProverSeed) -> Self {
        Self::for_table(seed, 0)
    }

    /// Table `table`'s randomness in a session of several tables: under an injected seed every stream at
    /// `purpose | table << TABLE_SHIFT` (table 0: [`Self::from_seed`]'s); in a proving build reads of the table's own.
    pub fn for_table(seed: ProverSeed, table: u64) -> Self {
        assert!(table < 1 << TABLE_SHIFT, "a table index below 2^{TABLE_SHIFT}");
        ProverRng { rng: seed.injected.then(|| ChaCha20Rng::from_seed(seed.seed)), table }
    }

    /// A fresh proof's randomness from the OS, never from the transcript.
    pub fn from_os() -> Self {
        Self::from_seed(ProverSeed::from_os())
    }

    /// `out.len()` bytes at `(purpose, index)`: in a proving build read from the OS on every call; under an injected seed bytes
    /// `[32 index, 32 index + out.len())` of stream `purpose` (independent of every other `(purpose, index)`, and of every
    /// other table's).
    pub fn fill_at(&mut self, purpose: u64, index: u64, out: &mut [u8]) {
        assert!(purpose < 1 << TABLE_SHIFT, "a stream purpose below 2^{TABLE_SHIFT} (the table takes the stream's high bits)");
        match &mut self.rng {
            Some(r) => {
                r.set_stream(purpose | self.table << TABLE_SHIFT);
                r.set_word_pos(8 * index as u128);
                r.fill_bytes(out);
            }
            None => os_fill(out),
        }
    }

    pub fn f128s(&mut self, purpose: u64, index: u64, n: usize) -> Vec<F128> {
        let mut b = vec![0u8; 16 * n];
        self.fill_at(purpose, index, &mut b);
        words(&b)
    }

    /// `n` words for each of `lanes` lanes, on rayon's threads: in a proving build each lane read from the OS; under an injected
    /// seed lane `i` is stream `purpose` at index `i << 32` (`f128s(purpose, (i as u64) << 32, n)`), sampled in [`LANE_CHUNK`]-word
    /// pieces, each worker with its own copy of the generator: a word is a function of the seed and its address alone, so the
    /// draw is the sequential one's word for word.
    pub fn f128s_lanes(&self, purpose: u64, lanes: usize, n: usize) -> Vec<Vec<F128>> {
        use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator, ParallelSliceMut};
        let Some(rng) = &self.rng else {
            return (0..lanes).into_par_iter().map(|_| {
                let mut b = vec![0u8; 16 * n];
                os_fill(&mut b);
                words(&b)
            }).collect();
        };
        let mut out: Vec<Vec<F128>> = (0..lanes).into_par_iter().map(|_| vec![F128::ZERO; n]).collect();
        out.par_iter_mut()
            .enumerate()
            .flat_map(|(i, lane)| lane.par_chunks_mut(LANE_CHUNK).enumerate().map(move |(c, w)| (((i as u64) << 32) + (c * LANE_CHUNK / 2) as u64, w)))
            .for_each_init(|| ProverRng { rng: Some(rng.clone()), table: self.table }, |r, (at, w)| w.copy_from_slice(&r.f128s(purpose, at, w.len())));
        out
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
        Self::for_table(seed, 0)
    }

    /// Table `table`'s, in a session of several tables ([`ProverRng::for_table`]).
    pub fn for_table(seed: ProverSeed, table: u64) -> Self {
        NoMask { rng: ProverRng::for_table(seed, table) }
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

    /// Level 0's hiding draws its lanes in parallel (`f128s_lanes`): every lane's words are the sequential draw's, whatever the
    /// generator read before, in every table, at lane lengths that end inside a ChaCha20 block, at a piece's end, and inside
    /// a later piece.
    #[test]
    fn lanes_drawn_in_parallel_are_the_sequential_draw() {
        let seed = ProverSeed::injected([11; 32]);
        let c = LANE_CHUNK;
        for (table, purpose, lanes, n) in [(0, 7, 3000, 37usize), (0, 6, 4, 4096), (2, 7, 257, 1), (5, 6, 64, 130), (0, 6, 4, 3 * c + 5),
                                           (1, 7, 3, 2 * c), (0, 7, 2, c + 1), (3, 6, 5, 7 * c - 3)] {
            let mut r = ProverRng::for_table(seed, table);
            let _ = r.f128s(STREAM_PAD, 9, 5);
            let par = r.f128s_lanes(purpose, lanes, n);
            let mut s = ProverRng::for_table(seed, table);
            let seq: Vec<Vec<F128>> = (0..lanes).map(|i| s.f128s(purpose, (i as u64) << 32, n)).collect();
            assert!(par == seq, "table {table}, stream {purpose}: {lanes} lanes of {n} words");
            assert_eq!(par.len(), lanes);
        }
        assert!(ProverRng::from_seed(seed).f128s_lanes(7, 0, 8).is_empty());
    }

    /// RFC 7539 section 2.3.2: key 00..1f, block counter 1, nonce 000000090000004a00000000 (words 13-15).
    #[test]
    fn chacha20_block_matches_rfc7539() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let b = chacha20_block(&key, 1 | (0x0900_0000u64 << 32), 0x4a00_0000);
        assert_eq!(crate::hex(&b), concat!("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e",
                                           "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e"));
    }

    /// A proving build's salts are the OS's: no two trees, proofs or leaves share a salt, a tree's sample spans several
    /// `getrandom` pieces, and level 0's are read once.
    #[test]
    fn os_salts_are_fresh_per_tree_and_read_once_where_shared() {
        let leaves = 2 * OS_PIECE / SALT_BYTES + 3;
        let (mut a, b) = (LeafSalts::for_seed(&ProverSeed::from_os()), LeafSalts::for_seed(&ProverSeed::from_os()));
        let l0 = a.level0(leaves);
        assert_eq!(l0.len(), leaves * SALT_BYTES);
        assert!(std::sync::Arc::ptr_eq(&l0, &a.level0(leaves)), "level 0's salts are read once");
        assert_eq!(a.expand(a.level0_id(), leaves), *l0);
        let id = a.next_id();
        let (t1, t1_again) = (a.expand(id, leaves), a.expand(id, leaves));
        let mut seen = std::collections::BTreeSet::new();
        for s in [&l0[..], &t1, &t1_again, &b.level0(leaves)[..], &b.expand(id, leaves), &a.table(1).level0(leaves)[..]] {
            assert!(s.chunks(SALT_BYTES).all(|c| seen.insert(c.to_vec())), "a salt read twice");
        }
    }

    /// Salts drawn elsewhere for the next trees ([`LeafSalts::preset`], the Lean finish's): the first draw of each at its size
    /// takes them; a second draw, a sample at another size and a tree past them read their own.
    #[test]
    fn preset_salts_are_the_next_trees() {
        let mut a = LeafSalts::for_seed(&ProverSeed::from_os());
        let trees: Vec<Vec<u8>> = vec![vec![1; 4 * SALT_BYTES], vec![2; 2 * SALT_BYTES]];
        a.preset(trees.clone()).unwrap();
        let id = a.next_id();
        assert_eq!(a.expand(id, 4), trees[0], "the first tree is the preset one");
        assert_ne!(a.expand(id, 4), trees[0], "a second draw of it reads afresh");
        let id = a.next_id();
        let other = a.expand(id, 3);
        assert_eq!(other.len(), 3 * SALT_BYTES);
        assert!(other.chunks(SALT_BYTES).all(|c| c != &trees[1][..SALT_BYTES]), "a draw at another size reads its own");
        assert_eq!(a.expand(id, 2), trees[1], "the draw at the preset size still takes them");
        let id = a.next_id();
        assert_ne!(a.expand(id, 2), trees[1], "a tree past the preset ones reads afresh");
        assert!(a.preset(vec![vec![0; 5]]).is_err(), "a preset tree is whole leaves");
    }

    /// An injected seed's salts are its ChaCha20 tape: leaf `i` of tree `id` is blocks `3i..3i+2` under the `STREAM_SALT` key with
    /// nonce `id`, so a test proof repeats.
    #[test]
    fn an_injected_seed_expands_its_salts() {
        let seed = ProverSeed::injected([5; 32]);
        let mut key = [0u8; 32];
        let mut r = ChaCha20Rng::from_seed([5; 32]);
        r.set_stream(STREAM_SALT);
        r.fill_bytes(&mut key);
        let s = LeafSalts::for_seed(&seed);
        let all = s.expand(7, 5);
        for i in 0..5 {
            let want: Vec<u8> = (0..3).flat_map(|j| chacha20_block(&key, 3 * i as u64 + j, 7)).collect();
            assert_eq!(all[SALT_BYTES * i..SALT_BYTES * (i + 1)], want);
        }
        assert_eq!(LeafSalts::for_seed(&seed).expand(7, 5), all);
        assert_eq!(*s.level0(3), LeafSalts::for_seed(&seed).expand(0, 3));
    }

    /// Table 0 of a session of several tables reads a one-table session's words and salts; every other table reads none of
    /// them, and none of another table's.
    #[test]
    fn a_table_draws_its_own_words_and_salts() {
        let seed = ProverSeed::injected([7; 32]);
        let addrs = [(STREAM_MASK, 0u64, 64usize), (4, 0, 64), (4, 1 << 32, 64), (5, 1 << 24, 64), (6, 3 << 32, 64), (7, 67 << 32, 64)];
        let draw = |mut r: ProverRng| -> Vec<Vec<F128>> { addrs.iter().map(|&(s, i, n)| r.f128s(s, i, n)).collect() };
        let one = draw(ProverRng::from_seed(seed));
        let tables: Vec<Vec<Vec<F128>>> = (0..3).map(|j| draw(ProverRng::for_table(seed, j))).collect();
        assert_eq!(tables[0], one, "table 0 is a one-table session's randomness");
        let words = |t: &Vec<Vec<F128>>| -> std::collections::BTreeSet<(u64, u64)> { t.iter().flatten().map(|x| (x.lo, x.hi)).collect() };
        for a in 0..3 {
            for b in a + 1..3 {
                assert!(words(&tables[a]).is_disjoint(&words(&tables[b])), "tables {a} and {b} share a word");
            }
        }
        let s = LeafSalts::for_seed(&seed);
        let (mut s0, mut s1, mut one) = (s.table(0), s.table(1), LeafSalts::for_seed(&seed));
        assert_eq!((s0.level0_id(), s0.next_id(), s0.next_id()), (one.level0_id(), one.next_id(), one.next_id()));
        assert_eq!((s1.level0_id(), s1.next_id()), (1 << TABLE_SHIFT, (1 << TABLE_SHIFT) + 1));
        assert_eq!(*s1.level0(2), s.expand(1 << TABLE_SHIFT, 2), "one key, the table's own trees");
        assert_ne!(*s1.level0(2), *s0.level0(2));
        let os = LeafSalts::for_seed(&ProverSeed::from_os());
        assert_ne!(*os.table(1).level0(2), *os.table(0).level0(2), "each table reads its own salts");
    }

    #[test]
    fn two_os_seeds_differ() {
        assert_ne!(ProverRng::from_os().f128s(STREAM_MASK, 0, 2), ProverRng::from_os().f128s(STREAM_MASK, 0, 2));
    }

    /// A proving build's words are the OS's: two draws at one address, two tables of one seed at one address, and the lanes of
    /// a sample spanning several `getrandom` pieces share no word.
    #[test]
    fn os_draws_are_read_afresh_at_every_address() {
        let seed = ProverSeed::from_os();
        let (mut r, mut t1) = (ProverRng::from_seed(seed), ProverRng::for_table(seed, 1));
        let n = 2 * OS_PIECE / 16 + 3;
        let mut seen = std::collections::BTreeSet::new();
        let lanes = r.f128s_lanes(6, 3, n);
        assert!(lanes.iter().all(|l| l.len() == n) && lanes.len() == 3);
        for d in [r.f128s(STREAM_MASK, 5, n), r.f128s(STREAM_MASK, 5, n), t1.f128s(STREAM_MASK, 5, n)].iter().chain(&lanes).chain(&r.f128s_lanes(6, 3, n)) {
            assert!(d.iter().all(|x| seen.insert((x.lo, x.hi))), "a word read twice");
        }
    }

    /// Two proofs' masks share no bit beyond chance: the M1 draws of two OS seeds (the algebraic phase's pads and dot-product
    /// mask for both reps, the mask slot, Ligerito level 0's padding and extra lanes) XOR to a balanced bit string. The
    /// protocol-level version is `flock-circuit zkstat`'s cross-run XOR test.
    #[test]
    fn two_proofs_share_no_mask_bits() {
        let draw = |seed: ProverSeed| -> Vec<u8> {
            let mut r = ProverRng::from_seed(seed);
            let mut b = Vec::new();
            for (stream, index, n) in [(STREAM_MASK, 0u64, 1024usize), (4, 0, 300), (5, 0, 300), (4, 1 << 32, 300), (6, 0, 4096), (7, 5 << 32, 500)] {
                b.extend(r.f128s(stream, index, n).iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat()));
            }
            b
        };
        let (a, b) = (draw(ProverSeed::from_os()), draw(ProverSeed::from_os()));
        let n = (8 * a.len()) as f64;
        let ones: u64 = a.iter().zip(&b).map(|(x, y)| (x ^ y).count_ones() as u64).sum();
        let z = (ones as f64 - n / 2.0) / (n / 4.0).sqrt();
        assert!(z.abs() < 6.0, "the XOR of two proofs' masks is unbalanced: z = {z}");
        assert!(a.chunks(16).zip(b.chunks(16)).all(|(x, y)| x != y), "two proofs share a mask word");
    }
}
