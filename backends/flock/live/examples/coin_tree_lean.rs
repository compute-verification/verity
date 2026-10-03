//! Vectors of the keyed coin tree (`coin-tree/hm96-sha512/v2`) for the soundness package's layout
//! (`FlockSoundness/Discharge/ZkTree/Layout.lean`): what `CoinTree::draw` makes of a known tape, for
//! `backends/flock/tests/test_lean_coin_layout.py`, written to the path in its argument
//! (`cargo run --example coin_tree_lean --features sha512 -- OUT.json`, as `backends/flock/tests/lean/coin_tree.sh` does).
//! An example rather than an integration test: the SHA-512 build compiles the library and `flock-circuit`, not every bin.
//!
//! * `draws`: draws on splitmix64 tapes (byte `a` of the tape is byte `a % 8` of output `a / 8`, little-endian): the live
//!   spec (`Zk.coinSpec`: the session stream and one table's two reps, 2^8 rounds of 512 coins) and a small one; for each,
//!   the key, the root, openings `(stream, round, n)` as their coins and salt, and the link points of `m` coordinates
//!   (session round 0's `2·m` coins, in runs of `m`).
//! * `mask`: docs/coin-tree-v2.md §9's draw, its key and the salt of stream 1 round 2, and that leaf's mask `M(K)·y`,
//!   which is `x ⊕ b` for §9's round digest `x` and leaf input `b` (pinned in `coin_tree.rs`'s tests).
#[cfg(feature = "sha512")]
mod vectors {
    use flock_field::F128;
    use flock_live::coin_tree::{CoinTree, Spec, NONCE_BYTES, SALT_BYTES};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha512};

    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

    fn mix(s: u64) -> u64 {
        let z = (s ^ (s >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn splitmix(seed: u64) -> impl FnMut(&mut [u8]) {
        let mut at = 0u64;
        move |b: &mut [u8]| {
            for x in b.iter_mut() {
                let w = mix(seed.wrapping_add((at / 8 + 1).wrapping_mul(GAMMA)));
                *x = (w >> (8 * (at % 8))) as u8;
                at += 1;
            }
        }
    }

    fn spec_tape() -> impl FnMut(&mut [u8]) {
        let stream: Vec<u8> = (0..4096u64)
            .flat_map(|i| {
                let mut h = Sha512::new();
                h.update(b"verity/coin-tree/v2/test-tape\0");
                h.update(i.to_le_bytes());
                h.finalize().to_vec()
            })
            .collect();
        let mut at = 0;
        move |b: &mut [u8]| {
            b.copy_from_slice(&stream[at..at + b.len()]);
            at += b.len();
        }
    }

    fn hx(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn word(x: &F128) -> String {
        let mut b = x.lo.to_le_bytes().to_vec();
        b.extend(x.hi.to_le_bytes());
        hx(&b)
    }

    fn nonce() -> [u8; NONCE_BYTES] {
        std::array::from_fn(|i| i as u8)
    }

    fn spec(streams: &[&str], rounds_log: usize, block: usize) -> Spec {
        Spec { streams: streams.iter().map(|s| s.to_string()).collect(), rounds_log, block }
    }

    /// The opening's coins and its salt (the 12 words after the coins and `T_n`).
    fn opening(t: &CoinTree, s: usize, k: usize, n: usize) -> Value {
        let (c, w) = t.open(s, k, n).unwrap();
        let salt: Vec<u8> = w[n + 4..n + 4 + SALT_BYTES / 16].iter().flat_map(|x| {
            let mut b = x.lo.to_le_bytes().to_vec();
            b.extend(x.hi.to_le_bytes());
            b
        }).collect();
        json!({"stream": s, "round": k, "n": n, "coins": c.iter().map(word).collect::<Vec<_>>(), "salt": hx(&salt)})
    }

    fn draw(name: &str, seed: u64, sp: Spec, opens: &[(usize, usize, usize)], m: usize) -> Value {
        let t = CoinTree::draw(sp.clone(), &nonce(), &mut splitmix(seed));
        let (c, _) = t.open(0, 0, 2 * m).unwrap();
        for &(s, k, n) in opens {
            assert!(t.open(s, k, n).is_ok());
        }
        assert!(t.open(1, 1 << sp.rounds_log, 1).is_err() && t.open(1, 0, sp.block + 1).is_err());
        json!({
            "name": name, "seed": seed, "spec": {"streams": sp.streams, "rounds_log": sp.rounds_log, "block": sp.block},
            "nonce": hx(&nonce()), "key": hx(t.key()), "root": hx(&t.root()),
            "openings": opens.iter().map(|&(s, k, n)| opening(&t, s, k, n)).collect::<Vec<_>>(),
            "points": {"m": m, "points": c.chunks(m).map(|p| p.iter().map(word).collect::<Vec<_>>()).collect::<Vec<_>>()},
        })
    }

    fn xor(a: &str, b: &str) -> String {
        let v = |s: &str| (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect::<Vec<_>>();
        hx(&v(a).iter().zip(v(b)).map(|(p, q)| p ^ q).collect::<Vec<_>>())
    }

    pub fn write(path: &str) {
        let live = draw("live", 0x5eed, spec(&["session", "t/rep0", "t/rep1"], 8, 512),
            &[(1, 0, 1), (1, 17, 5), (2, 255, 512), (2, 3, 0), (1, 254, 511), (0, 0, 54), (0, 255, 512)], 27);
        let small = draw("small", 7, spec(&["session", "a/rep0", "a/rep1", "b/rep0"], 2, 3),
            &[(0, 0, 3), (1, 1, 2), (2, 3, 3), (3, 2, 1), (3, 3, 0)], 1);

        // docs/coin-tree-v2.md §9: its key and root, and x and b of stream 1 round 2
        let t = CoinTree::draw(spec(&["session", "t/rep0", "t/rep1"], 2, 3), &nonce(), &mut spec_tape());
        assert_eq!(hx(&t.root()), "468d899b6bde51c72568b9cd72b6d35c809454fe82b991cd46647fec631224cc37af0b859a1133ce849b690e6583b7a12ddb90b2126611e6dcbaad8daee35740");
        let x = "c75f2678385f0b50c6dd3a849a024dad914a0741c35374601deb048d071522b5dbb403561feda7da12d3f510ed964c4784bff2c580f9439e2f5456ed3777f1dc";
        let b = "d03b0fdcadbbc02b5e8b6ad48a16aa16a9e1d499553c1429b396b5de1a12e8df2457d1dfa8908e4eb3b856e3c22dcc141e9809ea8731ea4a0e9bdcc077eaf043";
        let o = opening(&t, 1, 2, 2);
        let mask = json!({"key": hx(t.key()), "salt": o["salt"], "mask": xor(x, b)});

        let out = json!({"draws": [live, small], "mask": mask});
        std::fs::write(path, serde_json::to_string_pretty(&out).unwrap() + "\n").unwrap();
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: coin_tree_lean OUT.json");
    #[cfg(feature = "sha512")]
    vectors::write(&path);
    #[cfg(not(feature = "sha512"))]
    panic!("coin_tree_lean needs the sha512 feature ({path} not written)");
}
