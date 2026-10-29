//! The verifier's per-session coin commitment (lane flock-zk, C-Flock ZK; soundness ASSUMPTIONS.md §2 and §6, DESIGN.md §11.1), keyed
//! (`coin-tree/hm96-sha512/v2`, docs/coin-tree-v2.md).
//!
//! At `Hello`, before the prover's first message, the verifier samples every coin of the session and commits to all of them
//! at once: one SHA-512 Merkle root over Halevi–Micali commitments, one leaf per (stream, round), each stream (a table's
//! repetition, or the session's own link points) a subtree. Each round's coins are opened with the round's answer, and the
//! prover checks the opening against the root before it uses a coin ([`Checked`]). No coin is derived from another: no PRF.
//!
//! v2 keys the tree to the session. The prover's `Hello` carries a fresh 32-byte nonce `ν`, and every SHA-512 input of the
//! tree is `ν ‖ (v1's input)`, so a collision stored in the verifier's auxiliary input, made before `ν` existed, opens
//! nothing (binding is collision resistance of `x ↦ SHA-512(ν ‖ x)`). The verifier answers `Hello` with the root and a
//! fresh hm96 key `K` of its own, drawn for this session's tree, never reused and never revealed before the answer, so hm96's
//! hiding holds for every nonce the prover picks.
//!
//! A leaf commits to a fixed block of `block` coins through a hash chain, so an opening reveals only the `n` coins the round
//! uses: `T_block = H_ν(TAIL)`, `T_i = H_ν(CHAIN ‖ c_i ‖ T_{i+1})`, the round digest
//! `x = H_ν(ROUND ‖ u32be(stream) ‖ u32be(round) ‖ T_0)`, and the keyed hm96 leaf `H_ν(LEAF_PREFIX(K) ‖ x ⊕ M(K)·y ‖ c)` with
//! `c = H_ν(SALT_PREFIX ‖ y)` for the leaf's salt `y`. An opening of `n` coins is `c_0..c_{n-1}`, `T_n`, the salt and the path
//! (stream subtree, then the streams' tree), as F128 words.

use flock_field::F128;
use sha2::{Digest, Sha512};

use crate::{Req, Resp, Transport};

pub const SCHEME: &str = "coin-tree/hm96-sha512/v2";
pub const SALT_BYTES: usize = 192;
pub const KEY_BYTES: usize = 256;
pub const NONCE_BYTES: usize = 32;
/// The session's own stream: round 0 is the link points sampled at `Commit`.
pub const SESSION: &str = "session";
// hm96-sha512/v1's prefixes (verity/ml/commitments/hm96/PROTOCOL.md §2), zero-padded to 128 bytes
const SALT_TAG: &[u8] = b"verity/hm96-sha512/salt/v1\0";
const LEAF_TAG: &[u8] = b"verity/hm96-sha512/leaf/v1\0";

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

// The scheme's digests are SHA-512's: the SHA-512 build's 64-byte nodes and hm96 leaves. A build without it (the circuit prover's older GPU
// binaries) keeps the types and refuses the scheme.
#[cfg(feature = "sha512")]
fn digest(b: &[u8]) -> Hash {
    flock_hash::digest_from(b)
}
#[cfg(feature = "sha512")]
fn bytes(h: &Hash) -> &[u8] {
    h.as_flattened()
}
#[cfg(not(feature = "sha512"))]
fn digest(_: &[u8]) -> Hash {
    panic!("{SCHEME} needs the SHA-512 build")
}
#[cfg(not(feature = "sha512"))]
fn bytes(h: &Hash) -> &[u8] {
    h.as_slice()
}

fn padded128(parts: &[&[u8]]) -> [u8; 128] {
    let mut out = [0u8; 128];
    let mut at = 0;
    for p in parts {
        out[at..at + p.len()].copy_from_slice(p);
        at += p.len();
    }
    out
}

/// The tree's hashes (docs/coin-tree-v2.md §5): every SHA-512 input prefixed by the prover's nonce, v2's tags, and hm96 leaves
/// under the verifier's session key. The regression of the reading of v1 the spec rests on takes v1's: no prefix, v1's tags,
/// hm96's pinned key.
#[derive(Clone)]
pub struct Rules {
    prefix: Vec<u8>,
    key: Box<[u8; KEY_BYTES]>,
    /// the key as little-endian u64 words (bit `t` of the key is bit `t % 64` of word `t / 64`)
    key_words: Box<[u64; KEY_BYTES / 8]>,
    salt_prefix: [u8; 128],
    leaf_prefix: [u8; 128],
    tail: Vec<u8>,
    chain: Vec<u8>,
    round: Vec<u8>,
    pad: Vec<u8>,
}

impl Rules {
    pub fn v2(nonce: &[u8; NONCE_BYTES], key: &[u8; KEY_BYTES]) -> Self {
        Self::new(nonce.to_vec(), key, "v2")
    }

    fn new(prefix: Vec<u8>, key: &[u8; KEY_BYTES], version: &str) -> Self {
        let tag = |name: &str| format!("verity/coin-tree/{name}/{version}\0").into_bytes();
        let key_words = Box::new(std::array::from_fn(|w| u64::from_le_bytes(key[8 * w..8 * w + 8].try_into().unwrap())));
        Rules { prefix, key: Box::new(*key), key_words, salt_prefix: padded128(&[SALT_TAG]),
                leaf_prefix: padded128(&[LEAF_TAG, &Sha512::digest(key)]), tail: tag("tail"), chain: tag("chain"), round: tag("round"),
                pad: tag("pad") }
    }

    pub fn key(&self) -> &[u8; KEY_BYTES] {
        &self.key
    }

    fn h(&self, parts: &[&[u8]]) -> [u8; 64] {
        let mut s = Sha512::new();
        s.update(&self.prefix);
        for p in parts {
            s.update(p);
        }
        s.finalize().into()
    }

    /// hm96's `M(K)·y`: bit `i` is `⊕_j K_{i+j} y_j` (`i < 512`, `j < 1536`), bit `t` of a byte string being bit `t % 8` of
    /// byte `t / 8`. Each set bit of `y` adds the key's 512-bit window at it.
    fn mask(&self, y: &[u8]) -> [u8; 64] {
        let mut acc = [0u64; 8];
        for j in (0..8 * SALT_BYTES).filter(|&j| (y[j >> 3] >> (j & 7)) & 1 == 1) {
            let (q, r) = (j >> 6, j & 63);
            for (w, a) in acc.iter_mut().enumerate() {
                let hi = if r == 0 { 0 } else { self.key_words[q + w + 1] << (64 - r) };
                *a ^= (self.key_words[q + w] >> r) | hi;
            }
        }
        let mut out = [0u8; 64];
        for (w, a) in acc.iter().enumerate() {
            out[8 * w..8 * w + 8].copy_from_slice(&a.to_le_bytes());
        }
        out
    }

    /// `T_i` for every `i` of one block (`T_block` last).
    fn chain(&self, block: &[F128]) -> Vec<[u8; 64]> {
        let mut t = vec![[0u8; 64]; block.len() + 1];
        t[block.len()] = self.h(&[&self.tail]);
        for i in (0..block.len()).rev() {
            t[i] = self.h(&[&self.chain, &f128_bytes(&block[i]), &t[i + 1]]);
        }
        t
    }

    fn leaf(&self, stream: usize, round: usize, t0: &[u8; 64], salt: &[u8]) -> Hash {
        let x = self.h(&[&self.round, &(stream as u32).to_be_bytes(), &(round as u32).to_be_bytes(), t0]);
        let c = self.h(&[&self.salt_prefix, salt]);
        let m = self.mask(salt);
        let b: Vec<u8> = x.iter().zip(m).map(|(p, q)| p ^ q).collect();
        digest(&self.h(&[&self.leaf_prefix, &b, &c]))
    }

    fn node(&self, l: &Hash, r: &Hash) -> Hash {
        digest(&self.h(&[bytes(l), bytes(r)]))
    }

    fn pad(&self) -> Hash {
        digest(&self.h(&[&self.pad]))
    }
}

/// The prover's nonce from its `Hello` (`coins.nonce`: exactly 64 lowercase hex digits), as the verifier's R7 check and the
/// prover's [`Checked`] read it.
pub fn hello_nonce(hello: &str) -> Result<[u8; NONCE_BYTES], String> {
    let v: serde_json::Value = serde_json::from_str(hello).map_err(|e| format!("Hello is not JSON: {e}"))?;
    let n = v["coins"]["nonce"].as_str().ok_or("Hello carries no coin-tree nonce (coins.nonce)")?;
    if n.len() != 2 * NONCE_BYTES || !n.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return Err(format!("Hello's coin-tree nonce is not {} lowercase hex digits", 2 * NONCE_BYTES));
    }
    let mut out = [0u8; NONCE_BYTES];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&n[2 * i..2 * i + 2], 16).expect("hex digits");
    }
    Ok(out)
}

/// The session stream's coin count from the prover's `Hello`: the link points, `link.points · link.m` (the server's `Commit`
/// opens exactly these), 0 without a link.
pub fn hello_links(hello: &str) -> usize {
    let v: serde_json::Value = serde_json::from_str(hello).unwrap_or_default();
    let n = |k: &str| v["link"][k].as_u64().unwrap_or(0) as usize;
    n("points") * n("m")
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

/// The verifier's side: every coin of the session, their commitment, and the openings.
pub struct CoinTree {
    pub spec: Spec,
    rules: Rules,
    blocks: Vec<Vec<Vec<F128>>>,
    tails: Vec<Vec<Vec<[u8; 64]>>>,
    salts: Vec<Vec<Vec<u8>>>,
    /// per stream: its subtree, flat (leaves, then each level, root last)
    trees: Vec<Vec<Hash>>,
    /// the streams' tree, flat
    top: Vec<Hash>,
}

fn flat_tree(rules: &Rules, leaves: Vec<Hash>) -> Vec<Hash> {
    let n = leaves.len();
    let mut t = leaves;
    let (mut rd, mut len) = (0, n);
    while len > 1 {
        for i in 0..len / 2 {
            let x = rules.node(&t[rd + 2 * i], &t[rd + 2 * i + 1]);
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
    /// Draw the session key, then every coin and salt, from `rand` (the OS in a verifier, a seeded tape in a test harness) and
    /// commit under the prover's nonce.
    pub fn draw(spec: Spec, nonce: &[u8; NONCE_BYTES], rand: &mut dyn FnMut(&mut [u8])) -> Self {
        Self::draw_with(spec, nonce, rand, &|_, _, _| {})
    }

    /// [`Self::draw`] with `edit(stream, round, block)` applied to every block before the commitment: a verifier that commits
    /// to coins of its choosing (a test harness's malicious verifier). The key comes first (bit 2047 cleared), then v1's
    /// order: per stream and round, the block, then the salt.
    pub fn draw_with(spec: Spec, nonce: &[u8; NONCE_BYTES], rand: &mut dyn FnMut(&mut [u8]), edit: &dyn Fn(&str, usize, &mut [F128])) -> Self {
        let mut key = [0u8; KEY_BYTES];
        rand(&mut key);
        key[KEY_BYTES - 1] &= 0x7f;
        Self::build(spec, Rules::v2(nonce, &key), rand, edit)
    }

    fn build(spec: Spec, rules: Rules, rand: &mut dyn FnMut(&mut [u8]), edit: &dyn Fn(&str, usize, &mut [F128])) -> Self {
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
        let tails: Vec<Vec<Vec<[u8; 64]>>> = blocks.par_iter().map(|bs| bs.par_iter().map(|b| rules.chain(b)).collect()).collect();
        let trees: Vec<Vec<Hash>> = (0..spec.streams.len()).into_par_iter()
            .map(|s| flat_tree(&rules, (0..rounds).map(|k| rules.leaf(s, k, &tails[s][k][0], &salts[s][k])).collect()))
            .collect();
        let width = 1usize << spec.streams_log();
        let mut tops: Vec<Hash> = trees.iter().map(|t| *t.last().unwrap()).collect();
        tops.resize(width, rules.pad());
        let top = flat_tree(&rules, tops);
        CoinTree { spec, rules, blocks, tails, salts, trees, top }
    }

    pub fn root(&self) -> [u8; 64] {
        bytes(self.top.last().unwrap()).try_into().unwrap()
    }

    /// The session key (in the record, and in `Hello`'s answer: never before it).
    pub fn key(&self) -> &[u8; KEY_BYTES] {
        self.rules.key()
    }

    /// `Hello`'s answer: the root's 4 words, then the session key's 16.
    pub fn answer_words(&self) -> Vec<F128> {
        let mut w = words64(&self.root());
        w.extend(words64(self.key()));
        w
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

/// The prover's check: `words` opens `n` coins of `stream`'s round `round` under `root` and the session's `rules` (its own
/// nonce, the answer's key); returns the coins.
pub fn check(spec: &Spec, rules: &Rules, root: &[u8; 64], stream: usize, round: usize, n: usize, words: &[F128]) -> Result<Vec<F128>, String> {
    if words.len() != spec.opening_words(n) || round >= 1 << spec.rounds_log || n > spec.block {
        return Err(format!("coin opening of {} round {round}: {} words, want {}", spec.streams.get(stream).map(|s| s.as_str()).unwrap_or("?"),
                           words.len(), spec.opening_words(n)));
    }
    let coins = words[..n].to_vec();
    let mut t: [u8; 64] = bytes64(&words[n..n + 4]).try_into().unwrap();
    for c in coins.iter().rev() {
        t = rules.h(&[&rules.chain, &f128_bytes(c), &t]);
    }
    let salt = bytes64(&words[n + 4..n + 4 + SALT_BYTES / 16]);
    let mut h = rules.leaf(stream, round, &t, &salt);
    let mut at = n + 4 + SALT_BYTES / 16;
    let mut idx = round;
    for _ in 0..spec.rounds_log {
        let sib = digest(&bytes64(&words[at..at + 4]));
        h = if idx & 1 == 0 { rules.node(&h, &sib) } else { rules.node(&sib, &h) };
        idx /= 2;
        at += 4;
    }
    let mut idx = stream;
    for _ in 0..spec.streams_log() {
        let sib = digest(&bytes64(&words[at..at + 4]));
        h = if idx & 1 == 0 { rules.node(&h, &sib) } else { rules.node(&sib, &h) };
        idx /= 2;
        at += 4;
    }
    if bytes(&h) != root.as_slice() {
        return Err(format!("the verifier's coins for {} round {round} do not open its step-0 commitment", spec.streams[stream]));
    }
    Ok(coins)
}

/// The prover's transport under a committed coin schedule: takes its own nonce and the link points' count from the `Hello`
/// it sends and the root and session key from `Hello`'s answer (exactly 20 words, the key's bit 2047 zero), and checks every
/// coin the verifier sends (the link points, every round) against them before handing it on, stripped of its opening.
/// Anything else stops the prover there (a refusal), before it answers anything that depends on it. A firewalled session's
/// coin check of record is the Lean session loop's instead (`Flock.Firewall.Session`), which owns its connection.
pub struct Checked<T: Transport> {
    pub inner: T,
    pub spec: Spec,
    nonce: Option<[u8; NONCE_BYTES]>,
    links: usize,
    rules: Option<Rules>,
    root: Option<[u8; 64]>,
    rounds: std::collections::BTreeMap<String, usize>,
    /// every coin handed on, by stream and round (the simulator's extraction)
    pub opened: std::collections::BTreeMap<(String, usize), Vec<F128>>,
}

impl<T: Transport> Checked<T> {
    pub fn new(inner: T, spec: Spec) -> Self {
        Checked { inner, spec, nonce: None, links: 0, rules: None, root: None, rounds: Default::default(), opened: Default::default() }
    }
    pub fn root(&self) -> Option<[u8; 64]> {
        self.root
    }
    fn take(&mut self, stream: &str, n: usize, words: Vec<F128>) -> Result<Resp, String> {
        let (Some(root), Some(rules)) = (self.root, self.rules.as_ref()) else {
            return Ok(Resp::Refused("PROVER-STOPPED: coins before the step-0 coin commitment".into()));
        };
        let Some(si) = self.spec.stream_index(stream) else { return Ok(Resp::Refused(format!("PROVER-STOPPED: {stream} is outside the committed schedule"))) };
        let k = *self.rounds.get(stream).unwrap_or(&0);
        match check(&self.spec, rules, &root, si, k, n, &words) {
            Ok(c) => {
                self.rounds.insert(stream.to_string(), k + 1);
                self.opened.insert((stream.to_string(), k), c.clone());
                Ok(Resp::Coins(c))
            }
            Err(e) => Ok(Resp::Refused(format!("PROVER-STOPPED: {e}"))),
        }
    }
}

impl<T: Transport> Transport for Checked<T> {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let what = match &req {
            Req::Hello(h) => {
                match hello_nonce(h) {
                    Ok(n) => self.nonce = Some(n),
                    Err(e) => return Ok(Resp::Refused(format!("PROVER-STOPPED: {e}"))),
                }
                self.links = hello_links(h);
                Some((String::new(), 0usize, true))
            }
            Req::Commit { .. } => Some((SESSION.to_string(), 0, false)),
            Req::Round { stream, n, .. } => Some((stream.clone(), *n as usize, false)),
            _ => None,
        };
        let r = self.inner.call(req)?;
        let Some((stream, n, hello)) = what else { return Ok(r) };
        match r {
            Resp::Coins(w) if hello => {
                if w.len() != 4 + KEY_BYTES / 16 {
                    return Ok(Resp::Refused(format!("PROVER-STOPPED: Hello's answer has {} words, not the coin tree's root and session key (20)", w.len())));
                }
                let key: [u8; KEY_BYTES] = bytes64(&w[4..]).try_into().unwrap();
                if key[KEY_BYTES - 1] & 0x80 != 0 {
                    return Ok(Resp::Refused("PROVER-STOPPED: the verifier's session key has bit 2047 set".into()));
                }
                self.root = Some(bytes64(&w[..4]).try_into().unwrap());
                self.rules = Some(Rules::v2(&self.nonce.expect("the nonce of the Hello just sent"), &key));
                Ok(Resp::Coins(w))
            }
            Resp::Coins(w) => {
                // the link points' count is the statement's (`Hello`'s link), not the request's
                let n = if stream == SESSION { self.links } else { n };
                self.take(&stream, n, w)
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

    fn sha(parts: &[&[u8]]) -> [u8; 64] {
        let mut h = Sha512::new();
        for p in parts {
            h.update(p);
        }
        h.finalize().into()
    }

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

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

    /// docs/coin-tree-v2.md §9's tape: SHA-512("verity/coin-tree/v2/test-tape\0" ‖ u64le(i)), i = 0, 1, …, read in order.
    fn spec_tape() -> impl FnMut(&mut [u8]) {
        let stream: Vec<u8> = (0..4096u64).flat_map(|i| sha(&[b"verity/coin-tree/v2/test-tape\0", &i.to_le_bytes()])).collect();
        let mut at = 0;
        move |b: &mut [u8]| {
            b.copy_from_slice(&stream[at..at + b.len()]);
            at += b.len();
        }
    }

    fn spec_spec() -> Spec {
        Spec { streams: ["session", "t/rep0", "t/rep1"].map(String::from).to_vec(), rounds_log: 2, block: 3 }
    }

    const NONCE: [u8; NONCE_BYTES] = {
        let mut n = [0u8; NONCE_BYTES];
        let mut i = 0;
        while i < NONCE_BYTES {
            n[i] = i as u8;
            i += 1;
        }
        n
    };

    fn v1_rules() -> Rules {
        Rules::new(Vec::new(), &flock_merkle::hm96::default_key(), "v1")
    }

    /// The v2 spec rests on this reading of v1: its `v1_root` is v1's root over the spec's tape, read without a key (the
    /// generic keyed code under v1's rules: no prefix, v1's tags, hm96's pinned key).
    #[test]
    fn v1_reproduces_the_v2_specs_v1_root() {
        let root = CoinTree::build(spec_spec(), v1_rules(), &mut spec_tape(), &|_, _, _| {}).root();
        assert_eq!(hx(&root), "33cfd9f396b0f775a8db99f0d3d38563601416c556b9ad46e49500dee6d22d79f1c5f655a8f979d4879afaa87e0623bd23e7776b517b5b0dbdfe2b2682ed12a7");
    }

    /// The keyed hm96 pieces are hm96-sha512/v1's under its pinned key: the mask `M(K)·y` and both prefixes.
    #[test]
    fn keyed_hm96_is_hm96_under_the_pinned_key() {
        let r = v1_rules();
        let c = flock_merkle::hm96::consts();
        assert_eq!(r.salt_prefix, c.salt_prefix);
        assert_eq!(r.leaf_prefix, c.leaf_prefix);
        let mut t = tape(7);
        for _ in 0..8 {
            let mut y = [0u8; SALT_BYTES];
            t(&mut y);
            assert_eq!(r.mask(&y), flock_merkle::hm96::mask(&y));
        }
    }

    /// docs/coin-tree-v2.md §9, every vector: the key, the root, `Hello`'s coins object and answer frame, the opening of stream
    /// 1 round 2 (n = 2) and each intermediate, the tail and pad digests.
    #[test]
    fn v2_reproduces_the_specs_vectors() {
        let t = CoinTree::draw(spec_spec(), &NONCE, &mut spec_tape());
        let r = &t.rules;
        assert_eq!(hx(t.key()), "53d43b82d13a8d6cd586807a53763a5864e185bca221ce92e9d427b309dee61dba712538f39a2df44ab848082f9399912409587ef73bec0e93bb890a30618c786071e1daec3262a1a4172b6133844b9d6095bfc7e1c10907c13f346242ba55f6da7ce6933f81e691d1e5c423a13ecee129e4c48c3ce4accf0203a6629c9b2973d922a4ed7d7ac9a477ecca699155231c4558ca66b6291f43b23e3df6f2d43a80f69b5cdc1961ae302bbf547bdac99657384b447ddc8cff866d6429b101e6dfca9ec72e302c96d55bba508c16ed1abe3d1057c421c2ffbf1266c6485e90f8e1e22081b72242fe8396aca78cc5908947d6ebde3c7def16cc0dbb65ed499807e625");
        assert_eq!(hx(&t.root()), "468d899b6bde51c72568b9cd72b6d35c809454fe82b991cd46647fec631224cc37af0b859a1133ce849b690e6583b7a12ddb90b2126611e6dcbaad8daee35740");
        let mut coins = t.spec.to_json();
        coins["nonce"] = serde_json::json!(hx(&NONCE));
        assert_eq!(coins.to_string(), r#"{"block":3,"nonce":"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f","rounds_log":2,"scheme":"coin-tree/hm96-sha512/v2","streams":["session","t/rep0","t/rep1"]}"#);
        assert_eq!(hx(&Resp::Coins(t.answer_words()).encode()), format!("0214000000{}{}", hx(&t.root()), hx(t.key())));
        let (s, k, n) = (1, 2, 2);
        let (c, w) = t.open(s, k, n).unwrap();
        assert_eq!(c.iter().map(|x| hx(&f128_bytes(x))).collect::<Vec<_>>(), ["063456f31ef84a3d219bb0b5fd492f5e", "401968e9dd010aa780e6c0f421c2b05f"]);
        assert_eq!(w.len(), 34);
        assert_eq!(hx(&sha(&[&bytes64(&w)])), "6aef2a682b4c872097f6af4f4e167f45a42c11eaf3fb889cc538f90636fbc0f40f16bec75cc10e2bcfd74650086ea766e7a9caf995a1956c89841f108aeb02a3");
        assert_eq!(hx(&t.tails[s][k][n]), "9b2994f96eacf0551dc8d8c41ac00f8cce122d575342691400262daf8489fbbef9becea2e575a2e073dcfe696d676eee0bc349ff68dfdda8f278c378fec23c51");
        let t0 = t.tails[s][k][0];
        assert_eq!(hx(&t0), "b75c0977d0607fed217d48f346ff7a85f0e6874793939b0368f3286c8828dece613fd57cd88f037710bb7b9f180b1725afb2f47c94cf2ac57b383411f5045205");
        let salt = &t.salts[s][k];
        let x = r.h(&[&r.round, &(s as u32).to_be_bytes(), &(k as u32).to_be_bytes(), &t0]);
        let b: Vec<u8> = x.iter().zip(r.mask(salt)).map(|(p, q)| p ^ q).collect();
        assert_eq!(hx(&x), "c75f2678385f0b50c6dd3a849a024dad914a0741c35374601deb048d071522b5dbb403561feda7da12d3f510ed964c4784bff2c580f9439e2f5456ed3777f1dc");
        assert_eq!(hx(&b), "d03b0fdcadbbc02b5e8b6ad48a16aa16a9e1d499553c1429b396b5de1a12e8df2457d1dfa8908e4eb3b856e3c22dcc141e9809ea8731ea4a0e9bdcc077eaf043");
        assert_eq!(hx(&r.h(&[&r.salt_prefix, salt])), "5a5a8122e97ed2a930a1a15ef7cff32f4b325fe1b68eb63fd3fdf1cf8b1feaa2abaeb6ac59b4befc7a6ada40408d65773bb6f57c36518d4629f287d7b0a5c033");
        assert_eq!(hx(bytes(&r.leaf(s, k, &t0, salt))), "e370b4556e90d4e8947636c4bb3264314f2bfc716ec732a2907ff9fd79d164c5886113c94171638c7243ac95ffe7262eef0653517264ad50a3dd4cfa1515165a");
        assert_eq!(hx(&r.h(&[&r.tail])), "3ea637a0d4617c22acbf0b6fbb4ea41519715341e8cfa7ac44176184e300db5a0ce5e709add52b48901e928adeb88d9fea4fafa82a295044e44719a586e52176");
        assert_eq!(hx(&r.h(&[&r.pad])), "6e03778139201207311f8606aef149711173deeb9c783b33ba9524ceb81e0c74669c7984a3f2b8c1b82304e66a4ebd84a285046ed467226e30aff27ec64f6320");
        // the opening checks, and fails under another nonce, the pinned key, or no prefix
        assert_eq!(check(&t.spec, r, &t.root(), s, k, n, &w).unwrap(), c);
        for wrong in [Rules::v2(&[0; NONCE_BYTES], t.key()), Rules::v2(&NONCE, &flock_merkle::hm96::default_key()), Rules::new(Vec::new(), t.key(), "v2")] {
            assert!(check(&t.spec, &wrong, &t.root(), s, k, n, &w).is_err(), "an opening checks under another nonce or key");
        }
    }

    #[test]
    fn openings_verify_and_forgeries_do_not() {
        let spec = Spec { streams: vec![SESSION.into(), "t/rep0".into(), "t/rep1".into()], rounds_log: 4, block: 32 };
        let t = CoinTree::draw(spec.clone(), &NONCE, &mut tape(1));
        let (root, r) = (t.root(), t.rules.clone());
        for (s, k, n) in [(0, 0, 27), (1, 3, 1), (2, 15, 32), (1, 0, 0)] {
            let (c, w) = t.open(s, k, n).unwrap();
            assert_eq!(check(&spec, &r, &root, s, k, n, &w).unwrap(), c);
            let mut bad = w.clone();
            if n > 0 {
                bad[0].lo ^= 1;
                assert!(check(&spec, &r, &root, s, k, n, &bad).is_err(), "an altered coin opens");
            }
            assert!(check(&spec, &r, &root, s, (k + 1) % 16, n, &w).is_err(), "another round's position opens");
            assert!(check(&spec, &r, &root, (s + 1) % 3, k, n, &w).is_err(), "another stream's position opens");
        }
        // a coin prefix is bound: opening fewer coins reveals the same prefix
        let (c5, _) = t.open(1, 2, 5).unwrap();
        let (c9, w9) = t.open(1, 2, 9).unwrap();
        assert_eq!(c5[..], c9[..5]);
        assert!(check(&spec, &r, &root, 1, 2, 5, &w9).is_err(), "a 9-coin opening is not a 5-coin one");
        assert_ne!(CoinTree::draw(spec.clone(), &NONCE, &mut tape(2)).root(), root);
        assert_ne!(CoinTree::draw(spec, &[1; NONCE_BYTES], &mut tape(1)).root(), root, "the nonce keys the tree");
    }

    /// A verifier that answers `Hello` with fixed words.
    struct Answers(Vec<F128>, bool);

    impl Transport for Answers {
        fn call(&mut self, _: Req) -> Result<Resp, String> {
            self.1 = true;
            Ok(Resp::Coins(self.0.clone()))
        }
    }

    fn hello(nonce: Option<&str>) -> String {
        let mut coins = spec_spec().to_json();
        if let Some(n) = nonce {
            coins["nonce"] = serde_json::json!(n);
        }
        serde_json::json!({"coins": coins}).to_string()
    }

    /// The prover's refusals (§4): an answer that is not 20 words, a key with bit 2047 set, a `Hello` without a well-formed
    /// nonce (never sent).
    #[test]
    fn the_prover_refuses_a_malformed_answer_or_its_own_nonceless_hello() {
        let t = CoinTree::draw(spec_spec(), &NONCE, &mut spec_tape());
        let refused = |words: Vec<F128>, nonce: Option<&str>| -> (bool, bool) {
            let mut c = Checked::new(Answers(words, false), spec_spec());
            let r = c.call(Req::Hello(hello(nonce))).unwrap();
            (matches!(r, Resp::Refused(ref e) if e.starts_with("PROVER-STOPPED")), c.inner.1)
        };
        let good = hx(&NONCE);
        assert_eq!(refused(t.answer_words(), Some(&good)), (false, true));
        assert_eq!(refused(t.answer_words()[..4].to_vec(), Some(&good)), (true, true), "a 4-word answer (v1's)");
        let mut high = t.answer_words();
        high[19].hi |= 1 << 63;
        assert_eq!(refused(high, Some(&good)), (true, true), "a key with bit 2047 set");
        for bad in [None, Some(&good[..62]), Some(&good.to_uppercase()[..])] {
            assert_eq!(refused(t.answer_words(), bad), (true, false), "a Hello without a well-formed nonce is not sent");
        }
    }

    /// A verifier that answers each request with the next of its answers.
    struct Script(std::collections::VecDeque<Resp>);

    impl Transport for Script {
        fn call(&mut self, _: Req) -> Result<Resp, String> {
            self.0.pop_front().ok_or_else(|| "the script has no answer left".to_string())
        }
    }

    /// One case of `coin_open_vectors.json`: the prover's `Hello`, `Hello`'s answer (none: the verifier answers it without
    /// coins), then the openings the verifier answers, each `(stream, n, words)`.
    struct Case {
        case: &'static str,
        hello: String,
        answer: Option<Vec<F128>>,
        opens: Vec<(&'static str, usize, Vec<F128>)>,
    }

    /// The case through [`Checked`]: each step the Lean check takes (`flock-firewall coins`'s `hello`, `answer`, `open`) with
    /// what Rust answered, the coins handed on or the refusal, which ends the case.
    fn run(c: &Case) -> serde_json::Value {
        use serde_json::json;
        let mut script = std::collections::VecDeque::from([c.answer.clone().map_or(Resp::Ok, Resp::Coins)]);
        script.extend(c.opens.iter().map(|(_, _, w)| Resp::Coins(w.clone())));
        let left = script.len();
        let mut ch = Checked::new(Script(script), spec_spec());
        let mut steps = vec![json!({"hello": c.hello})];
        let r = ch.call(Req::Hello(c.hello.clone())).unwrap();
        if let (true, Some(w)) = (ch.inner.0.len() < left, &c.answer) {
            steps.push(json!({"answer": hx(&Resp::Coins(w.clone()).encode())}));
        }
        if let Resp::Refused(why) = r {
            steps.last_mut().unwrap()["stop"] = json!(why);
            return json!({"case": c.case, "steps": steps});
        }
        for (stream, n, w) in &c.opens {
            let req = if *stream == SESSION {
                Req::Commit { root_f: vec![], roots: vec![], publics: vec![] }
            } else {
                Req::Round { stream: stream.to_string(), content: vec![], n: *n as u32 }
            };
            let n = if *stream == SESSION { ch.links } else { *n };
            let mut step = json!({"open": stream, "n": n, "frame": hx(&Resp::Coins(w.clone()).encode())});
            match ch.call(req).unwrap() {
                Resp::Coins(coins) => step["coins"] = json!(hx(&bytes64(&coins))),
                Resp::Refused(why) => {
                    step["stop"] = json!(why);
                    steps.push(step);
                    break;
                }
                _ => panic!("{}: an answer that is neither coins nor a refusal", c.case),
            }
            steps.push(step);
        }
        json!({"case": c.case, "steps": steps})
    }

    /// `coin_open_vectors.json`, which `test_lean_coin_open.py` holds the Lean check (`Flock.Firewall.Coins`,
    /// `flock-firewall coins`) to, word for word: honest and altered openings of the §9 tree through [`Checked`], each step
    /// with Rust's answer. `COIN_OPEN_VECTORS=write` rewrites the file from these cases.
    #[test]
    fn checked_answers_the_lean_checks_vectors() {
        let t = CoinTree::draw(spec_spec(), &NONCE, &mut spec_tape());
        let open = |s: usize, k: usize, n: usize| t.open(s, k, n).unwrap().1;
        let with = |mut w: Vec<F128>, f: &dyn Fn(&mut Vec<F128>)| {
            f(&mut w);
            w
        };
        let good = hx(&NONCE);
        let hello_of = |nonce: serde_json::Value, link: serde_json::Value| -> String {
            let mut coins = spec_spec().to_json();
            if !nonce.is_null() {
                coins["nonce"] = nonce;
            }
            let mut h = serde_json::json!({"coins": coins, "profile": "fast100", "reps": 2});
            if !link.is_null() {
                h["link"] = link;
            }
            h.to_string()
        };
        let link = |points: u64, m: u64| serde_json::json!({"m": m, "points": points, "sigma": "ab".repeat(64)});
        let hello = hello_of(serde_json::json!(good), link(2, 1));
        let ans = t.answer_words();
        let rep0 = |w: Vec<F128>| vec![("t/rep0", 2, w)];
        let at = |i: usize, bit: u32| move |w: &mut Vec<F128>| if bit < 64 { w[i].lo ^= 1 << bit } else { w[i].hi ^= 1 << (bit - 64) };
        let case = |case, hello: &String, answer: Option<Vec<F128>>, opens| Case { case, hello: hello.clone(), answer, opens };
        let cases = vec![
            case("honest", &hello, Some(ans.clone()), vec![(SESSION, 0, open(0, 0, 2)), ("t/rep0", 2, open(1, 0, 2)), ("t/rep0", 0, open(1, 1, 0)),
                 ("t/rep0", 3, open(1, 2, 3)), ("t/rep0", 1, open(1, 3, 1)), ("t/rep1", 2, open(2, 0, 2)), ("t/rep1", 3, open(2, 1, 3))]),
            case("a coin's bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(0, 0)))),
            case("the last coin's top bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(1, 127)))),
            case("T_n's bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(2, 0)))),
            case("T_n's top bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(5, 127)))),
            case("the salt's bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(6, 0)))),
            case("the salt's top bit", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(17, 127)))),
            case("the round's path node", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(18, 0)))),
            case("the streams' path node", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &at(33, 127)))),
            case("a word short", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &|w| { w.pop(); }))),
            case("a word long", &hello, Some(ans.clone()), rep0(with(open(1, 0, 2), &|w| w.push(F128::ZERO)))),
            case("no words", &hello, Some(ans.clone()), rep0(vec![])),
            case("round 1's opening for round 0", &hello, Some(ans.clone()), rep0(open(1, 1, 2))),
            case("t/rep1's opening for t/rep0", &hello, Some(ans.clone()), rep0(open(2, 0, 2))),
            case("the session's opening for t/rep0", &hello, Some(ans.clone()), rep0(open(0, 0, 2))),
            case("3 coins for 2", &hello, Some(ans.clone()), rep0(open(1, 0, 3))),
            case("2 coins for 3", &hello, Some(ans.clone()), vec![("t/rep0", 3, open(1, 0, 2))]),
            case("4 coins, past the block", &hello, Some(ans.clone()), vec![("t/rep0", 4, with(open(1, 0, 3), &|w| w.push(F128::ZERO)))]),
            case("round 4, past the schedule", &hello, Some(ans.clone()),
                 vec![("t/rep0", 1, open(1, 0, 1)), ("t/rep0", 1, open(1, 1, 1)), ("t/rep0", 1, open(1, 2, 1)), ("t/rep0", 1, open(1, 3, 1)), ("t/rep0", 1, open(1, 3, 1))]),
            case("a stream outside the tree", &hello, Some(ans.clone()), vec![("t/rep2", 2, open(2, 0, 2))]),
            case("the session's count is Hello's", &hello, Some(ans.clone()), vec![(SESSION, 0, open(0, 0, 3))]),
            case("a Hello of 3 link points", &hello_of(serde_json::json!(good), link(1, 3)), Some(ans.clone()), vec![(SESSION, 0, open(0, 0, 3))]),
            case("a Hello of link points past the block", &hello_of(serde_json::json!(good), link(2, 2)), Some(ans.clone()),
                 vec![(SESSION, 0, with(open(0, 0, 3), &|w| w.push(F128::ZERO)))]),
            case("another nonce", &hello_of(serde_json::json!("11".repeat(32)), link(2, 1)), Some(ans.clone()), rep0(open(1, 0, 2))),
            case("another key", &hello, Some(with(ans.clone(), &at(4, 0))), rep0(open(1, 0, 2))),
            case("another root", &hello, Some(with(ans.clone(), &at(0, 0))), rep0(open(1, 0, 2))),
            case("a 4-word answer", &hello, Some(ans[..4].to_vec()), rep0(open(1, 0, 2))),
            case("a 21-word answer", &hello, Some(with(ans.clone(), &|w| w.push(F128::ZERO))), rep0(open(1, 0, 2))),
            case("the key's bit 2047", &hello, Some(with(ans.clone(), &at(19, 127))), rep0(open(1, 0, 2))),
            case("Hello answered without coins", &hello, None, rep0(open(1, 0, 2))),
            case("a Hello without a nonce", &hello_of(serde_json::Value::Null, link(2, 1)), Some(ans.clone()), rep0(open(1, 0, 2))),
            case("a nonce that is not a string", &hello_of(serde_json::json!(5), link(2, 1)), Some(ans.clone()), rep0(open(1, 0, 2))),
            case("an uppercase nonce", &hello_of(serde_json::json!(good.to_uppercase()), link(2, 1)), Some(ans.clone()), rep0(open(1, 0, 2))),
            case("a short nonce", &hello_of(serde_json::json!(&good[..62]), link(2, 1)), Some(ans.clone()), rep0(open(1, 0, 2))),
        ];
        let ran: Vec<serde_json::Value> = cases.iter().map(run).collect();
        let built = serde_json::json!({
            "comment": "The coin tree's opening checks (`coin-tree/hm96-sha512/v2`) on docs/coin-tree-v2.md §9's tree, which coin_tree.rs's test holds Rust's `Checked` to and test_lean_coin_open.py holds the Lean check (`Flock.Firewall.Coins`, `flock-firewall coins`) to, word for word. Each case is the steps the Lean check takes: the prover's `hello`, `Hello`'s `answer` (`Resp::Coins`' frame, hex), then each `open` of `n` coins of a stream's next round (its frame); an open's `coins` are the ones handed on (hex), and a step's `stop` is the refusal, which ends the case.",
            "cases": ran,
        });
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/coin_open_vectors.json");
        if std::env::var("COIN_OPEN_VECTORS").as_deref() == Ok("write") {
            std::fs::write(path, serde_json::to_string_pretty(&built).unwrap() + "\n").unwrap();
        }
        let file: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for (want, got) in file["cases"].as_array().unwrap().iter().zip(built["cases"].as_array().unwrap()) {
            assert_eq!(want, got, "{}: coin_open_vectors.json is not what Checked answers (COIN_OPEN_VECTORS=write rewrites it)", got["case"]);
        }
        assert_eq!(file, built, "coin_open_vectors.json's cases are not these (COIN_OPEN_VECTORS=write rewrites it)");
        for c in built["cases"].as_array().unwrap() {
            let honest = ["honest", "a Hello of 3 link points"].contains(&c["case"].as_str().unwrap());
            assert_eq!(c["steps"].as_array().unwrap().iter().all(|s| s.get("stop").is_none()), honest, "{}: refused exactly when altered", c["case"]);
        }
    }
}
