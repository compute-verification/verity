//! The verifier's per-session coin commitment (lane flock-zk, M2; soundness ASSUMPTIONS.md §2 and §6, DESIGN.md §11.1).
//!
//! At `Hello`, before the prover's first message, the verifier draws every coin of the session and commits to all of them
//! at once: one SHA-512 Merkle root over Halevi–Micali (`hm96-sha512/v1`) commitments, one leaf per (stream, round), each
//! stream (a table's repetition, or the session's own link points) a subtree. Each round's coins are opened with the round's
//! answer, and the prover checks the opening against the root before it uses a coin ([`Checked`]). Binding (SHA-512
//! collision resistance) fixes every coin before the prover says anything; hiding (HM96, statistical) keeps each round's
//! coins unknown until the round's bytes are recorded. No coin is derived from another: no PRF.
//!
//! A leaf commits to a fixed block of `block` coins through a hash chain, so an opening reveals only the `n` coins the round
//! uses: `T_block = SHA-512(TAIL)`, `T_i = SHA-512(CHAIN ‖ c_i ‖ T_{i+1})`, the leaf's inner digest
//! `x = SHA-512(ROUND ‖ u32be(stream) ‖ u32be(round) ‖ T_0)`, the leaf `hm96::leaf(x, salt)`. An opening of `n` coins is
//! `c_0..c_{n-1}`, `T_n`, the salt and the path (stream subtree, then the streams' tree), as F128 words.

use flock_field::F128;
use sha2::{Digest, Sha512};

use crate::{Req, Resp, Transport};

pub const SCHEME: &str = "coin-tree/hm96-sha512/v1";
const TAIL: &[u8] = b"verity/coin-tree/tail/v1\0";
const CHAIN: &[u8] = b"verity/coin-tree/chain/v1\0";
const ROUND: &[u8] = b"verity/coin-tree/round/v1\0";
const PAD: &[u8] = b"verity/coin-tree/pad/v1\0";
pub const SALT_BYTES: usize = 192;
/// The session's own stream: round 0 is the link points drawn at `Commit`.
pub const SESSION: &str = "session";

type Hash = flock_hash::Digest;

/// The committed schedule's shape, fixed by the session parameters (both sides pin it through `Hello`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    /// every stream the session may open, in leaf order
    pub streams: Vec<String>,
    /// rounds per stream: `2^rounds_log`
    pub rounds_log: usize,
    /// coins per round, at most
    pub block: usize,
}

impl Spec {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"scheme": SCHEME, "streams": self.streams, "rounds_log": self.rounds_log, "block": self.block})
    }
    fn streams_log(&self) -> usize {
        self.streams.len().next_power_of_two().trailing_zeros() as usize
    }
    pub fn stream_index(&self, s: &str) -> Option<usize> {
        self.streams.iter().position(|x| x == s)
    }
    /// Words of an opening of `n` coins: the coins, `T_n` (4), the salt (12), the path (4 per node).
    pub fn opening_words(&self, n: usize) -> usize {
        n + 4 + SALT_BYTES / 16 + 4 * (self.rounds_log + self.streams_log())
    }
}

fn sha(parts: &[&[u8]]) -> [u8; 64] {
    let mut h = Sha512::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

// The scheme's digests are SHA-512's: the SHA-512 build's 64-byte nodes and hm96 leaves. A build without it (M0's older GPU
// binaries) keeps the types and refuses the scheme.
#[cfg(feature = "sha512")]
fn digest(b: &[u8]) -> Hash {
    flock_hash::digest_from(b)
}
#[cfg(feature = "sha512")]
fn bytes(h: &Hash) -> &[u8] {
    h.as_flattened()
}
#[cfg(feature = "sha512")]
fn hm96_leaf(x: &[u8; 64], salt: &[u8]) -> Hash {
    flock_merkle::hm96::leaf(&digest(x), salt)
}
#[cfg(not(feature = "sha512"))]
fn digest(_: &[u8]) -> Hash {
    panic!("{SCHEME} needs the SHA-512 build")
}
#[cfg(not(feature = "sha512"))]
fn bytes(h: &Hash) -> &[u8] {
    h.as_slice()
}
#[cfg(not(feature = "sha512"))]
fn hm96_leaf(_: &[u8; 64], _: &[u8]) -> Hash {
    panic!("{SCHEME} needs the SHA-512 build")
}

fn node(l: &Hash, r: &Hash) -> Hash {
    digest(&sha(&[bytes(l), bytes(r)]))
}

fn f128_bytes(x: &F128) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&x.lo.to_le_bytes());
    b[8..].copy_from_slice(&x.hi.to_le_bytes());
    b
}

fn words64(b: &[u8]) -> Vec<F128> {
    b.chunks(16).map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..16].try_into().unwrap()))).collect()
}

fn bytes64(w: &[F128]) -> Vec<u8> {
    w.iter().flat_map(f128_bytes).collect()
}

/// `T_i` for every `i` of one block (`T_block` last).
fn chain(block: &[F128]) -> Vec<[u8; 64]> {
    let mut t = vec![[0u8; 64]; block.len() + 1];
    t[block.len()] = sha(&[TAIL]);
    for i in (0..block.len()).rev() {
        t[i] = sha(&[CHAIN, &f128_bytes(&block[i]), &t[i + 1]]);
    }
    t
}

fn leaf(stream: usize, round: usize, t0: &[u8; 64], salt: &[u8]) -> Hash {
    let x = sha(&[ROUND, &(stream as u32).to_be_bytes(), &(round as u32).to_be_bytes(), t0]);
    hm96_leaf(&x, salt)
}

/// The verifier's side: every coin of the session, their commitment, and the openings.
pub struct CoinTree {
    pub spec: Spec,
    blocks: Vec<Vec<Vec<F128>>>,
    tails: Vec<Vec<Vec<[u8; 64]>>>,
    salts: Vec<Vec<Vec<u8>>>,
    /// per stream: its subtree, flat (leaves, then each level, root last)
    trees: Vec<Vec<Hash>>,
    /// the streams' tree, flat
    top: Vec<Hash>,
}

fn flat_tree(leaves: Vec<Hash>) -> Vec<Hash> {
    let n = leaves.len();
    let mut t = leaves;
    let (mut rd, mut len) = (0, n);
    while len > 1 {
        for i in 0..len / 2 {
            let x = node(&t[rd + 2 * i], &t[rd + 2 * i + 1]);
            t.push(x);
        }
        rd += len;
        len /= 2;
    }
    t
}

fn path(t: &[Hash], n: usize, mut j: usize) -> Vec<Hash> {
    let (mut rd, mut len) = (0, n);
    let mut out = Vec::new();
    while len > 1 {
        out.push(t[rd + (j ^ 1)]);
        rd += len;
        len /= 2;
        j /= 2;
    }
    out
}

impl CoinTree {
    /// Draw every coin and salt from `rand` (the OS in a verifier, a seeded tape in a test harness) and commit.
    pub fn draw(spec: Spec, rand: &mut dyn FnMut(&mut [u8])) -> Self {
        Self::draw_with(spec, rand, &|_, _, _| {})
    }

    /// [`Self::draw`] with `edit(stream, round, block)` applied to every block before the commitment: a verifier that commits
    /// to coins of its choosing (a test harness's malicious verifier).
    pub fn draw_with(spec: Spec, rand: &mut dyn FnMut(&mut [u8]), edit: &dyn Fn(&str, usize, &mut [F128])) -> Self {
        use rayon::prelude::*;
        let rounds = 1usize << spec.rounds_log;
        let mut blocks = Vec::new();
        let mut salts = Vec::new();
        for name in &spec.streams {
            let mut bs = Vec::with_capacity(rounds);
            let mut ss = Vec::with_capacity(rounds);
            for k in 0..rounds {
                let mut b = vec![0u8; 16 * spec.block];
                rand(&mut b);
                let mut w = words64(&b);
                edit(name, k, &mut w);
                bs.push(w);
                let mut s = vec![0u8; SALT_BYTES];
                rand(&mut s);
                ss.push(s);
            }
            blocks.push(bs);
            salts.push(ss);
        }
        let tails: Vec<Vec<Vec<[u8; 64]>>> = blocks.par_iter().map(|bs| bs.par_iter().map(|b| chain(b)).collect()).collect();
        let trees: Vec<Vec<Hash>> = (0..spec.streams.len()).into_par_iter()
            .map(|s| flat_tree((0..rounds).map(|k| leaf(s, k, &tails[s][k][0], &salts[s][k])).collect()))
            .collect();
        let width = 1usize << spec.streams_log();
        let mut tops: Vec<Hash> = trees.iter().map(|t| *t.last().unwrap()).collect();
        tops.resize(width, digest(&sha(&[PAD])));
        let top = flat_tree(tops);
        CoinTree { spec, blocks, tails, salts, trees, top }
    }

    pub fn root(&self) -> [u8; 64] {
        bytes(self.top.last().unwrap()).try_into().unwrap()
    }

    /// The root as the four F128 words `Hello` answers with.
    pub fn root_words(&self) -> Vec<F128> {
        words64(&self.root())
    }

    /// Round `round` of `stream`'s first `n` coins, and their opening words (`Spec::opening_words(n)` in all).
    pub fn open(&self, stream: usize, round: usize, n: usize) -> Result<(Vec<F128>, Vec<F128>), String> {
        if round >= 1 << self.spec.rounds_log || n > self.spec.block || stream >= self.spec.streams.len() {
            return Err(format!("coin tree: round {round} of {} ({n} coins) is outside the committed schedule", self.spec.streams[stream.min(self.spec.streams.len() - 1)]));
        }
        let coins = self.blocks[stream][round][..n].to_vec();
        let mut w = coins.clone();
        w.extend(words64(&self.tails[stream][round][n]));
        w.extend(words64(&self.salts[stream][round]));
        let p: Vec<Hash> = path(&self.trees[stream], 1 << self.spec.rounds_log, round).into_iter()
            .chain(path(&self.top, 1 << self.spec.streams_log(), stream)).collect();
        for h in p {
            w.extend(words64(bytes(&h)));
        }
        Ok((coins, w))
    }
}

/// The prover's check: `words` opens `n` coins of `stream`'s round `round` under `root`; returns the coins.
pub fn check(spec: &Spec, root: &[u8; 64], stream: usize, round: usize, n: usize, words: &[F128]) -> Result<Vec<F128>, String> {
    if words.len() != spec.opening_words(n) || round >= 1 << spec.rounds_log || n > spec.block {
        return Err(format!("coin opening of {} round {round}: {} words, want {}", spec.streams.get(stream).map(|s| s.as_str()).unwrap_or("?"),
                           words.len(), spec.opening_words(n)));
    }
    let coins = words[..n].to_vec();
    let mut t: [u8; 64] = bytes64(&words[n..n + 4]).try_into().unwrap();
    for c in coins.iter().rev() {
        t = sha(&[CHAIN, &f128_bytes(c), &t]);
    }
    let salt = bytes64(&words[n + 4..n + 4 + SALT_BYTES / 16]);
    let mut h = leaf(stream, round, &t, &salt);
    let mut at = n + 4 + SALT_BYTES / 16;
    let mut idx = round;
    for _ in 0..spec.rounds_log {
        let sib = digest(&bytes64(&words[at..at + 4]));
        h = if idx & 1 == 0 { node(&h, &sib) } else { node(&sib, &h) };
        idx /= 2;
        at += 4;
    }
    let mut idx = stream;
    for _ in 0..spec.streams_log() {
        let sib = digest(&bytes64(&words[at..at + 4]));
        h = if idx & 1 == 0 { node(&h, &sib) } else { node(&sib, &h) };
        idx /= 2;
        at += 4;
    }
    if bytes(&h) != root.as_slice() {
        return Err(format!("the verifier's coins for {} round {round} do not open its step-0 commitment", spec.streams[stream]));
    }
    Ok(coins)
}

/// The prover's transport under a committed coin schedule: takes the root from `Hello`'s answer, and checks every coin the
/// verifier sends (the link points, every round) against it before handing it on, stripped of its opening. A coin that does
/// not open the commitment stops the prover there (a refusal), before it answers anything that depends on it.
pub struct Checked<T: Transport> {
    pub inner: T,
    pub spec: Spec,
    root: Option<[u8; 64]>,
    rounds: std::collections::BTreeMap<String, usize>,
    /// every coin handed on, by stream and round (the simulator's extraction)
    pub opened: std::collections::BTreeMap<(String, usize), Vec<F128>>,
}

impl<T: Transport> Checked<T> {
    pub fn new(inner: T, spec: Spec) -> Self {
        Checked { inner, spec, root: None, rounds: Default::default(), opened: Default::default() }
    }
    pub fn root(&self) -> Option<[u8; 64]> {
        self.root
    }
    fn take(&mut self, stream: &str, n: usize, words: Vec<F128>) -> Resp {
        let Some(root) = self.root else { return Resp::Refused("PROVER-STOPPED: coins before the step-0 coin commitment".into()) };
        let Some(si) = self.spec.stream_index(stream) else { return Resp::Refused(format!("PROVER-STOPPED: {stream} is outside the committed schedule")) };
        let k = *self.rounds.get(stream).unwrap_or(&0);
        match check(&self.spec, &root, si, k, n, &words) {
            Ok(c) => {
                self.rounds.insert(stream.to_string(), k + 1);
                self.opened.insert((stream.to_string(), k), c.clone());
                Resp::Coins(c)
            }
            Err(e) => Resp::Refused(format!("PROVER-STOPPED: {e}")),
        }
    }
}

impl<T: Transport> Transport for Checked<T> {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let what = match &req {
            Req::Hello(_) => Some((String::new(), 0usize, true)),
            Req::Commit { .. } => Some((SESSION.to_string(), 0, false)),
            Req::Round { stream, n, .. } => Some((stream.clone(), *n as usize, false)),
            _ => None,
        };
        let r = self.inner.call(req)?;
        let Some((stream, n, hello)) = what else { return Ok(r) };
        match r {
            Resp::Coins(w) if hello => {
                if w.len() != 4 {
                    return Ok(Resp::Refused("PROVER-STOPPED: Hello's answer is not a 64-byte coin commitment".into()));
                }
                self.root = Some(bytes64(&w).try_into().unwrap());
                Ok(Resp::Coins(w))
            }
            Resp::Coins(w) => {
                // the link points' count is the statement's, not the request's: every word before the opening's tail
                let n = if stream == SESSION { w.len() - self.spec.opening_words(0) } else { n };
                Ok(self.take(&stream, n, w))
            }
            other => Ok(other),
        }
    }
    fn stats(&self) -> (u64, u64, u64) {
        self.inner.stats()
    }
    fn wait_s(&self) -> f64 {
        self.inner.wait_s()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tape(seed: u8) -> impl FnMut(&mut [u8]) {
        let mut ctr = 0u64;
        move |b: &mut [u8]| {
            for c in b.chunks_mut(64) {
                let x = sha(&[&[seed], &ctr.to_le_bytes()]);
                c.copy_from_slice(&x[..c.len()]);
                ctr += 1;
            }
        }
    }

    /// The keyed tree's spec (`coin-tree/hm96-sha512/v2`, docs/coin-tree-v2.md §9) rests on this reading of v1: its `v1_root` is
    /// this code's root over the spec's test tape, read without a key.
    #[test]
    fn v1_reproduces_the_v2_specs_v1_root() {
        let stream: Vec<u8> = (0..4096u64).flat_map(|i| sha(&[b"verity/coin-tree/v2/test-tape\0", &i.to_le_bytes()])).collect();
        let mut at = 0;
        let mut tape = |b: &mut [u8]| {
            b.copy_from_slice(&stream[at..at + b.len()]);
            at += b.len();
        };
        let spec = Spec { streams: ["session", "t/rep0", "t/rep1"].map(String::from).to_vec(), rounds_log: 2, block: 3 };
        let root: String = CoinTree::draw(spec, &mut tape).root().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(root, "33cfd9f396b0f775a8db99f0d3d38563601416c556b9ad46e49500dee6d22d79f1c5f655a8f979d4879afaa87e0623bd23e7776b517b5b0dbdfe2b2682ed12a7");
    }

    #[test]
    fn openings_verify_and_forgeries_do_not() {
        let spec = Spec { streams: vec![SESSION.into(), "t/rep0".into(), "t/rep1".into()], rounds_log: 4, block: 32 };
        let t = CoinTree::draw(spec.clone(), &mut tape(1));
        let root = t.root();
        for (s, k, n) in [(0, 0, 27), (1, 3, 1), (2, 15, 32), (1, 0, 0)] {
            let (c, w) = t.open(s, k, n).unwrap();
            assert_eq!(check(&spec, &root, s, k, n, &w).unwrap(), c);
            let mut bad = w.clone();
            if n > 0 {
                bad[0].lo ^= 1;
                assert!(check(&spec, &root, s, k, n, &bad).is_err(), "an altered coin opens");
            }
            assert!(check(&spec, &root, s, (k + 1) % 16, n, &w).is_err(), "another round's position opens");
            assert!(check(&spec, &root, (s + 1) % 3, k, n, &w).is_err(), "another stream's position opens");
        }
        // a coin prefix is bound: opening fewer coins reveals the same prefix
        let (c5, _) = t.open(1, 2, 5).unwrap();
        let (c9, w9) = t.open(1, 2, 9).unwrap();
        assert_eq!(c5[..], c9[..5]);
        assert!(check(&spec, &root, 1, 2, 5, &w9).is_err(), "a 9-coin opening is not a 5-coin one");
        assert_ne!(CoinTree::draw(spec.clone(), &mut tape(2)).root(), root);
    }
}
