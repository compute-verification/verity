//! Live coins from a seed the verifier commits at step 0 (lane flock-circuit; the malicious-verifier ZK groundwork of
//! B-Ligero PROTOCOL §5.3): the verifier draws a 256-bit seed and nonce from the OS before anything else, sends the commitment
//! `SHA-256(TAG || nonce || seed)` in answer to `Hello`, derives every coin from the seed with `verity.randomness.derive`
//! (byte-compatible: `tests/randomness/vectors.json`'s formats) and reveals seed and nonce in its verdict, so the prover can
//! check that every coin it was sent was fixed before it committed. The commitment scheme is named (`SCHEME`) and swappable:
//! HM96, as B-Ligero's coin commitment, is the target.

use sha2::{Digest, Sha256};

pub const SCHEME: &str = "coin-commit/sha256";
pub const TAG: &[u8] = b"verity/flock-circuit/coin-commit\0";
pub const DOMAIN: &str = "verity/flock-circuit/coins/v1";
const DERIVE: &[u8] = b"verity/randomness/derive/v1\0";
const STREAM: &[u8] = b"verity/randomness/stream/v1\0";

/// A `verity.randomness` part: bytes, a string or a nonnegative integer.
pub enum Part<'a> {
    B(&'a [u8]),
    S(&'a str),
    I(u64),
}

/// `frame(part)` = tag || len as 8 bytes big-endian || body.
pub fn frame(p: &Part<'_>) -> Vec<u8> {
    let (tag, body): (u8, Vec<u8>) = match p {
        Part::B(b) => (b'b', b.to_vec()),
        Part::S(s) => (b's', s.as_bytes().to_vec()),
        Part::I(i) => {
            let b = i.to_be_bytes();
            let z = b.iter().position(|&c| c != 0).unwrap_or(7);
            (b'i', b[z..].to_vec())
        }
    };
    let mut out = vec![tag];
    out.extend((body.len() as u64).to_be_bytes());
    out.extend(body);
    out
}

/// `derive(source, domain, context)` (context keys sorted, as the Python sorts them).
pub fn derive(source: &[u8], domain: &str, context: &[(&str, Part<'_>)]) -> [u8; 32] {
    let mut ctx: Vec<(&str, Vec<u8>)> = context.iter().map(|(k, v)| (*k, frame(v))).collect();
    ctx.sort_by(|a, b| a.0.cmp(b.0));
    let flat: Vec<u8> = ctx.into_iter().flat_map(|(k, v)| [frame(&Part::S(k)), v].concat()).collect();
    let mut h = Sha256::new();
    h.update(DERIVE);
    h.update(frame(&Part::B(source)));
    h.update(frame(&Part::S(domain)));
    h.update(frame(&Part::B(&flat)));
    h.finalize().into()
}

/// The first `n` bytes of `Key(key).stream(index)`.
pub fn stream(key: &[u8; 32], index: &[Part<'_>], n: usize) -> Vec<u8> {
    let inner: Vec<u8> = index.iter().flat_map(frame).collect();
    let mut head = STREAM.to_vec();
    head.extend(key);
    head.extend(frame(&Part::B(&inner)));
    let mut out = Vec::with_capacity(n + 32);
    let mut c = 0u64;
    while out.len() < n {
        let mut h = Sha256::new();
        h.update(&head);
        h.update(c.to_be_bytes());
        out.extend(h.finalize());
        c += 1;
    }
    out.truncate(n);
    out
}

pub fn commitment(seed: &[u8; 32], nonce: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TAG);
    h.update(nonce);
    h.update(seed);
    h.finalize().into()
}

/// The coin key of a session: the seed, bound to the session's hello (its parameters, Σ included).
pub fn coin_key(seed: &[u8; 32], hello: &str) -> [u8; 32] {
    derive(seed, DOMAIN, &[("hello", Part::S(hello))])
}

/// The bytes of the `k`-th coin request of the session (`n` coins of 16 bytes).
pub fn coins(key: &[u8; 32], k: u64, n: usize) -> Vec<u8> {
    stream(key, &[Part::S("coins"), Part::I(k)], 16 * n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// verity.randomness: derive(bytes(range(32)), "verity/flock-circuit/coins/v1", {"sigma": bytes(32), "statement": b"\x01\x02"})
    /// and its stream(("coins", 3), 48), from the Python.
    #[test]
    fn matches_verity_randomness() {
        let src: Vec<u8> = (0..32).collect();
        let k = derive(&src, DOMAIN, &[("statement", Part::B(&[1, 2])), ("sigma", Part::B(&[0; 32]))]);
        assert_eq!(crate::hex(&k), "d8cb901a7b56a293ee85597aea7358b24dd732b2e57b377b0b264474fe56e0fb");
        assert_eq!(crate::hex(&stream(&k, &[Part::S("coins"), Part::I(3)], 48)),
                   "6b53857d54c89827969433fbeed8be96fc8471bcc872af7fed757cdfa23f44838f6348a60b2f0964ec1b67ae3a8a2272");
    }
}
