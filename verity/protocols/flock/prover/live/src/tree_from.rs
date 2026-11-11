//! The challenger's coin trees (`flock-circuit serve --tree-from HOST:PORT [--tree-secret-file F]`): the verifier's coin
//! server takes each session's keyed coin tree ([`crate::coin_tree`]) from the challenger, the auditor's device on a host the
//! prover doesn't control, instead of sampling it here. At `Hello` the challenger samples the tree under the prover's nonce and
//! answers its root and session key; it opens each round itself, after what the round follows (the link points after
//! `root_F` and each table's `root_B`, [`link_after`]; every other round after its message's SHA-512, the record's
//! `msg_sha512`), and logs each before it answers. The client checks every opening against the tree it sampled
//! ([`Challenger::tree_open`]) before the challenger's live-coin part answers the prover, so the session's record is the one a tree sampled
//! here would give, and the challenger's release log is what the audit holds it to.
//!
//! The frames are `verity.protocols.spot.device.remote`'s: `u32be len(head) ‖ u32be 0 ‖ head`, the head a JSON
//! object with its `type`. The client sends `hello` (with the session's `secret` in hex when it has one), then
//! `tree {nonce, spec}` per session and `tree-open {after, n, round, stream, tree}` per round, then `close`. A refusal, a
//! broken stream, a malformed answer or an opening that doesn't open its tree ends the connection, and every request after
//! it is refused: a retry never resamples.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::{Arc, Mutex};

use flock_field::F128;
use serde_json::{Value, json};
use sha2::{Digest, Sha512};

use crate::coin_tree::{self, KEY_BYTES, NONCE_BYTES, Rules, Spec};
use crate::hex;

const MAX_HEAD: usize = 1 << 20;
const TREE_FORM: &str = "the challenger's coin tree is {key, nonce, root} in lowercase hex (256, 32 and 64 bytes, the key's \
                         bit 2047 clear), under the nonce asked for";

/// One connection to the challenger's session, and the coin trees it sampled on it.
pub struct Challenger {
    stream: TcpStream,
    ended: Option<String>,
    trees: BTreeMap<[u8; 64], (Spec, Rules)>,
}

/// A [`Challenger`] the challenger's live-coin part's sessions share: one request and its answer at a time.
pub type Shared = Arc<Mutex<Challenger>>;

/// What a session's link points are opened after: the SHA-512 of its `root_F` and each table's `root_B`, in table-name
/// order (`challenger.link_after`, `Spot.Challenger.linkAfter`).
pub fn link_after<'a>(root_f: &[u8], root_b_by_name: impl IntoIterator<Item = &'a Vec<u8>>) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(root_f);
    for b in root_b_by_name {
        h.update(b);
    }
    h.finalize().into()
}

/// `s` as lowercase hex digits, two to a byte.
pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return None;
    }
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()).collect()
}

/// An opening word as the challenger writes it: 32 lowercase hex digits, the high 64 bits first.
fn word(s: &str) -> Option<F128> {
    let b = unhex(s).filter(|b| b.len() == 16)?;
    Some(F128::new(u64::from_be_bytes(b[8..].try_into().unwrap()), u64::from_be_bytes(b[..8].try_into().unwrap())))
}

impl Challenger {
    /// The challenger at `address` (`HOST:PORT`), greeted with the session's `secret` when it has one. Its answer to the
    /// hello (the session's commitment, or a withheld session's layout) is checked for its form and not read: the coin
    /// server asks it for coin trees alone.
    pub fn connect(address: &str, secret: Option<&[u8]>) -> Result<Self, String> {
        let stream = TcpStream::connect(address).map_err(|e| format!("the challenger at {address}: {e}"))?;
        stream.set_nodelay(true).map_err(|e| format!("the challenger at {address}: {e}"))?;
        let mut c = Challenger { stream, ended: None, trees: BTreeMap::new() };
        let mut hello = json!({"type": "hello"});
        if let Some(s) = secret {
            hello["secret"] = json!(hex(s));
        }
        c.ask(&hello, &["commitment", "layout"])?;
        Ok(c)
    }

    fn end(&mut self, why: String) -> String {
        let _ = self.stream.shutdown(Shutdown::Both);
        self.ended = Some(why.clone());
        why
    }

    fn send(&mut self, head: &Value) -> Result<(), String> {
        let h = serde_json::to_vec(head).expect("a JSON head");
        let mut frame = Vec::with_capacity(8 + h.len());
        frame.extend((h.len() as u32).to_be_bytes());
        frame.extend(0u32.to_be_bytes());
        frame.extend(h);
        self.stream.write_all(&frame).map_err(|e| format!("the stream broke ({e})"))
    }

    fn recv(&mut self) -> Result<Value, String> {
        let mut hdr = [0u8; 8];
        self.stream.read_exact(&mut hdr).map_err(|e| format!("the stream broke ({e})"))?;
        let (nh, nb) = (u32::from_be_bytes(hdr[..4].try_into().unwrap()) as usize, u32::from_be_bytes(hdr[4..].try_into().unwrap()));
        if nh > MAX_HEAD || nb != 0 {
            return Err(format!("a frame is a head of at most {MAX_HEAD} bytes and no body, not {nh} + {nb}"));
        }
        let mut h = vec![0u8; nh];
        self.stream.read_exact(&mut h).map_err(|e| format!("the stream broke ({e})"))?;
        let v: Value = serde_json::from_slice(&h).map_err(|e| format!("a frame's head is not JSON ({e})"))?;
        if !v["type"].is_string() {
            return Err("a frame's head is an object with a type".into());
        }
        Ok(v)
    }

    /// `head` sent, and the answer of one of `kinds` read: its value (`null` for `closed`).
    fn ask(&mut self, head: &Value, kinds: &[&str]) -> Result<Value, String> {
        if let Some(e) = &self.ended {
            return Err(format!("the challenger's session ended ({e}): a retry never redraws"));
        }
        let reply = match self.send(head).and_then(|_| self.recv()) {
            Ok(r) => r,
            Err(e) => return Err(self.end(e)),
        };
        let o = reply.as_object().expect("recv reads an object");
        let kind = reply["type"].as_str().unwrap_or_default();
        if kind == "refused" && o.len() == 2 && reply["reason"].is_string() {
            return Err(self.end(format!("the challenger refused: {}", reply["reason"].as_str().unwrap())));
        }
        let form = if kind == "closed" { o.len() == 1 } else { o.len() == 2 && o.contains_key(kind) };
        if !kinds.contains(&kind) || !form {
            return Err(self.end(format!("the challenger did not answer with {}", kinds.join(" or "))));
        }
        Ok(reply.get(kind).cloned().unwrap_or(Value::Null))
    }

    /// The coin tree the challenger samples of `spec` under the prover's `nonce`: its root and session key, `Hello`'s answer.
    pub fn tree(&mut self, spec: &Spec, nonce: &[u8; NONCE_BYTES]) -> Result<([u8; 64], [u8; KEY_BYTES]), String> {
        let t = self.ask(&json!({"type": "tree", "nonce": hex(nonce), "spec": spec.to_json()}), &["tree"])?;
        let field = |k: &str, n: usize| t[k].as_str().and_then(unhex).filter(|b| b.len() == n);
        let (root, key): ([u8; 64], [u8; KEY_BYTES]) =
            match (t.as_object().map(|o| o.len()), field("key", KEY_BYTES), field("nonce", NONCE_BYTES), field("root", 64)) {
                (Some(3), Some(key), Some(got), Some(root)) if key[KEY_BYTES - 1] & 0x80 == 0 && got == nonce => {
                    (root.try_into().unwrap(), key.try_into().unwrap())
                }
                _ => return Err(self.end(TREE_FORM.into())),
            };
        self.trees.insert(root, (spec.clone(), Rules::v2(nonce, &key)));
        Ok((root, key))
    }

    /// Round `round` of `stream`'s first `n` coins of the tree this connection sampled under `root`, opened by the
    /// challenger after `after`: the coins and their opening words, checked against the tree ([`coin_tree::check`]).
    pub fn tree_open(&mut self, root: &[u8; 64], stream: &str, round: usize, n: usize, after: &[u8; 64]) -> Result<(Vec<F128>, Vec<F128>), String> {
        let Some((spec, rules, si)) = self.trees.get(root).and_then(|(s, r)| s.stream_index(stream).map(|i| (s.clone(), r.clone(), i))) else {
            return Err(format!("no coin tree this session sampled under that root has a stream {stream}"));
        };
        let ws = self.ask(&json!({"type": "tree-open", "after": hex(after), "n": n, "round": round, "stream": stream, "tree": hex(root)}),
                          &["tree-coins"])?;
        let Some(w) = ws.as_array().and_then(|a| a.iter().map(|w| w.as_str().and_then(word)).collect::<Option<Vec<F128>>>()) else {
            return Err(self.end("a coin tree's round is its opening words, each 32 lowercase hex digits".into()));
        };
        match coin_tree::check(&spec, &rules, root, si, round, n, &w) {
            Ok(coins) => Ok((coins, w)),
            Err(e) => Err(self.end(format!("the challenger's coin tree round: {e}"))),
        }
    }

    /// The session closed: the challenger signs its close and answers once it is written.
    pub fn close(&mut self) -> Result<(), String> {
        self.ask(&json!({"type": "close"}), &["closed"])?;
        self.end("the client closed the session".into());
        Ok(())
    }
}

/// A challenger in this process, for tests: it serves one connection, samples each tree from a ChaCha20 tape as a coin
/// server's test harness does (`coin_tape`), and returns every head the client sent.
#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use crate::coin_tree::CoinTree;
    use rand_chacha::{ChaCha20Rng, rand_core::{RngCore, SeedableRng}};
    use std::net::TcpListener;

    #[derive(Clone, Copy)]
    pub(crate) enum How {
        Honest,
        /// each tree under another nonce than the one asked for
        OtherNonce,
        /// each round's first coin altered after the tree is committed
        Tampered,
        /// every tree refused
        Refuses,
    }

    fn spec_of(v: &Value) -> Spec {
        Spec { streams: v["streams"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect(),
               rounds_log: v["rounds_log"].as_u64().unwrap() as usize, block: v["block"].as_u64().unwrap() as usize }
    }

    pub(crate) fn challenger(tape: [u8; 32], how: How) -> (String, std::thread::JoinHandle<Vec<Value>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = l.local_addr().unwrap().to_string();
        let served = std::thread::spawn(move || {
            let mut c = Challenger { stream: l.accept().unwrap().0, ended: None, trees: BTreeMap::new() };
            let (mut heads, mut trees) = (Vec::new(), BTreeMap::<String, CoinTree>::new());
            while let Ok(head) = c.recv() {
                heads.push(head.clone());
                let reply = match (head["type"].as_str().unwrap(), how) {
                    ("hello", _) => json!({"type": "commitment", "commitment": {}}),
                    ("tree", How::Refuses) => json!({"type": "refused", "reason": "no tree for this session"}),
                    ("tree", _) => {
                        let mut nonce: [u8; NONCE_BYTES] = unhex(head["nonce"].as_str().unwrap()).unwrap().try_into().unwrap();
                        if let How::OtherNonce = how {
                            nonce[0] ^= 1;
                        }
                        let mut r = ChaCha20Rng::from_seed(tape);
                        let t = CoinTree::draw(spec_of(&head["spec"]), &nonce, &mut |b: &mut [u8]| r.fill_bytes(b));
                        let answer = json!({"key": hex(t.key()), "nonce": hex(&nonce), "root": hex(&t.root())});
                        trees.insert(hex(&t.root()), t);
                        json!({"type": "tree", "tree": answer})
                    }
                    ("tree-open", _) => {
                        let t = &trees[head["tree"].as_str().unwrap()];
                        let si = t.spec.stream_index(head["stream"].as_str().unwrap()).unwrap();
                        let at = |k: &str| head[k].as_u64().unwrap() as usize;
                        let (_, mut w) = t.open(si, at("round"), at("n")).unwrap();
                        if let How::Tampered = how {
                            w[0].lo ^= 1;
                        }
                        json!({"type": "tree-coins", "tree-coins": w.iter().map(|x| format!("{:016x}{:016x}", x.hi, x.lo)).collect::<Vec<_>>()})
                    }
                    ("close", _) => json!({"type": "closed"}),
                    (other, _) => json!({"type": "refused", "reason": format!("no {other} here")}),
                };
                if c.send(&reply).is_err() {
                    break;
                }
            }
            heads
        });
        (address, served)
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{How, challenger};
    use super::*;

    fn spec() -> Spec {
        Spec { streams: vec![coin_tree::SESSION.into(), "t/rep0".into()], rounds_log: 2, block: 3 }
    }

    /// The challenger's tree is answered under the nonce asked for, each round checked against it, and `close` is the last
    /// request: every one after it is refused.
    #[test]
    fn a_challengers_tree_opens_under_its_root_and_nothing_follows_close() {
        let (address, served) = challenger([1; 32], How::Honest);
        let mut c = Challenger::connect(&address, Some(b"secret".as_slice())).unwrap();
        let (root, key) = c.tree(&spec(), &[7; NONCE_BYTES]).unwrap();
        assert_eq!(key[KEY_BYTES - 1] & 0x80, 0);
        let (coins, w) = c.tree_open(&root, "t/rep0", 1, 2, &[4; 64]).unwrap();
        assert_eq!((coins.len(), w.len(), &w[..2]), (2, spec().opening_words(2), &coins[..]));
        assert!(c.tree_open(&[0; 64], "t/rep0", 1, 2, &[4; 64]).unwrap_err().contains("no coin tree this session sampled"));
        c.close().unwrap();
        assert!(c.tree(&spec(), &[7; NONCE_BYTES]).unwrap_err().contains("a retry never redraws"));
        let heads = served.join().unwrap();
        assert_eq!(heads[0], json!({"type": "hello", "secret": hex(b"secret")}));
        assert_eq!(heads[1], json!({"type": "tree", "nonce": hex(&[7; NONCE_BYTES]), "spec": spec().to_json()}));
        assert_eq!(heads[2], json!({"type": "tree-open", "after": hex(&[4; 64]), "n": 2, "round": 1, "stream": "t/rep0", "tree": hex(&root)}));
        assert_eq!((heads.len(), &heads[3]), (4, &json!({"type": "close"})));
    }

    /// A tree under another nonce, a round that doesn't open its tree, and a refusal each end the connection: nothing is
    /// asked of the challenger after them.
    #[test]
    fn another_nonce_a_tampered_round_or_a_refusal_ends_the_challengers_session() {
        for (how, why) in [(How::OtherNonce, "under the nonce asked for"), (How::Tampered, "the challenger's coin tree round"),
                           (How::Refuses, "the challenger refused: no tree for this session")] {
            let (address, served) = challenger([1; 32], how);
            let mut c = Challenger::connect(&address, None).unwrap();
            let e = match c.tree(&spec(), &[7; NONCE_BYTES]) {
                Ok((root, _)) => c.tree_open(&root, coin_tree::SESSION, 0, 2, &[4; 64]).unwrap_err(),
                Err(e) => e,
            };
            assert!(e.contains(why), "{e}");
            assert!(c.close().unwrap_err().contains("a retry never redraws"));
            assert!(served.join().unwrap().iter().all(|h| h["type"] != "close"));
        }
    }

    #[test]
    fn link_after_is_the_sha512_of_root_f_then_each_root_b_in_table_name_order() {
        let roots: BTreeMap<String, Vec<u8>> = [("b".to_string(), vec![2u8; 64]), ("a".to_string(), vec![1u8; 64])].into();
        let want: [u8; 64] = Sha512::digest([vec![9u8; 64], vec![1; 64], vec![2; 64]].concat()).into();
        assert_eq!(link_after(&[9; 64], roots.values()), want);
    }
}
