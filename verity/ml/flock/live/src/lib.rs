//! Live verifier coins for Flock (the flock-128-r2 profile: two `Fast100` runs, soundness squared only with live coins).
//!
//! * [`LiveChallenger`] is the prover's `Challenger`. It never derives a challenge: every squeeze sends the bytes the
//!   prover absorbed since its previous squeeze (the "round", framed exactly like `FsChallenger`'s SHA-256
//!   transcript-v2 absorb stream, so the CUDA prover's host challenger produces identical rounds) and waits for the
//!   verifier's coins. Forks open live child streams.
//! * [`Server`] is the verifier's side of a session. It issues a coin only after it holds the round that precedes it,
//!   draws coins from the OS, pins the session parameters (profile, reps, flavour, statement), refuses a rep-1 coin
//!   unless rep 1 committed rep 0's root, and refuses every coin until the link context (root_F, points, y) arrived.
//!   With a [`LinkSpec`] the link context is a real exchange (flock-link L1): `Commit` carries root_F, every table's
//!   root_B and its public words; the server checks the public words against its statement, then draws the link
//!   points from the OS as its own coin slot; `Link` carries y; only then are Flock's coins issued, and each rep's
//!   first round must bind exactly the committed root_B. The replay hands the recorded points and y to the statement
//!   verifier, which opens them in every rep (L2).
//! * [`ReplayChallenger`] runs the ordinary Flock verifier over the recorded session: at every squeeze it checks that the
//!   proof's absorbed bytes hash to the round the server recorded before it issued that coin, then returns that coin.
//!   The session record ([`Server::record_json`]) is the evidence; a proof file alone verifies nothing.
//! * Live prime coins (route (a), [`SessionConfig::prime_live`]): the prime (A-GKR) prover's every challenge round is a
//!   `Prime` request carrying its SHA-256 transcript state (which chains every message it absorbed since the domain,
//!   the coins included) and the number of BabyBear^6 challenges; the server answers with OS-drawn words below p. Prime
//!   rounds need `Hello`; at least one must precede `Commit` (root_F); none is issued between `Commit` and y. The
//!   record keeps every round (state, count, words) and the round count before `Commit`; the prime verifier replays
//!   its transcript against them (`archive/gkr/verifier/src/transcript.rs`).
//! * [`Server::from_record`] rebuilds a finished session from its record alone (G2: the record keeps root_F, every
//!   root_B and the committed public words), so [`Server::finish`] re-verifies the Flock side offline.

pub mod chunk;
pub mod coin_seed;
pub mod coin_tree;
pub mod ir_block;
pub mod ir_frame;
pub mod ir_sampling;
pub mod ir_tail;
pub mod lookup;
pub mod circuit;
pub mod firewall;
pub mod glue;
pub mod proved_scope;
pub mod pure_block;
pub mod record_text;
#[cfg(feature = "glue")]
pub mod tables;
pub mod typed;
pub mod vllm_block;
pub mod zk_hooks;
#[cfg(feature = "sha512")]
pub mod zk_veil;
#[cfg(feature = "sha512")]
pub mod sha512_native;
// needs flock-vtime-b684b12.patch, which the pods apply with the SHA-512 patches
#[cfg(feature = "sha512")]
pub mod session_verify;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod gpu_circuit;

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    net::TcpStream,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use flock_field::F128;
use flock_transcript::challenger::Challenger;
use serde_json::{Value, json};
use sha2::{Digest, Sha256, Sha512};

pub const OP_DOMAIN: u8 = 0x01;
pub const OP_LABEL: u8 = 0x02;
pub const OP_OBSERVE: u8 = 0x03;
pub const OP_SQUEEZE: u8 = 0x04;
pub const OP_BYTES: u8 = 0x05;
pub const KIND_SCALAR: u8 = 0x01;
pub const KIND_SLICE: u8 = 0x02;

pub const LABEL_MIXED: &[u8] = b"flock-mixed-v1";
pub const LABEL_R1CS: &[u8] = b"flock-r1cs-v0";

/// Every thread's stack, the prover's and the verifier's, ours and Flock's (its pools and their helper threads, patched to
/// `flock_core::THREAD_STACK`): Rust's 2 MiB default overflows on a large statement's proof
/// (`RMSNormCudaVector8_v1{BLOCK=256}`; `SiluMul_v2{I=1536}` at m=27).
pub const THREAD_STACK: usize = flock_core::THREAD_STACK;

/// Rayon's global pool on [`THREAD_STACK`] stacks. First thing in a binary's `main`, before any parallel iterator builds the
/// pool with the default stack; a second call does nothing.
pub fn init_threads() {
    let _ = rayon::ThreadPoolBuilder::new().stack_size(THREAD_STACK).build_global();
}

/// `std::thread::spawn` on a [`THREAD_STACK`] stack.
pub fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> std::thread::JoinHandle<T> {
    std::thread::Builder::new().stack_size(THREAD_STACK).spawn(f).expect("a thread")
}

/// `Scope::spawn` on a [`THREAD_STACK`] stack.
pub fn spawn_scoped<'s, 'e, T: Send + 's>(sc: &'s std::thread::Scope<'s, 'e>, f: impl FnOnce() -> T + Send + 's)
                                          -> std::thread::ScopedJoinHandle<'s, T> {
    flock_core::spawn_scoped(sc, f)
}

// ---------------------------------------------------------------------------------------------------------------
// Framing (transcript-v2: 16-byte header [op][kind][0;6][len u64 LE], payload zero-padded to 16).
// ---------------------------------------------------------------------------------------------------------------

pub fn push_header(out: &mut Vec<u8>, op: u8, kind: u8, len: u64) {
    let mut h = [0u8; 16];
    h[0] = op;
    h[1] = kind;
    h[8..].copy_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&h);
}

pub fn push_padded(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(b);
    let rem = b.len() % 16;
    if rem != 0 {
        out.resize(out.len() + 16 - rem, 0);
    }
}

fn push_f128(out: &mut Vec<u8>, v: F128) {
    out.extend_from_slice(&v.lo.to_le_bytes());
    out.extend_from_slice(&v.hi.to_le_bytes());
}

pub fn frame_domain(domain: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    push_header(&mut v, OP_DOMAIN, 0, domain.len() as u64);
    push_padded(&mut v, domain);
    v
}

/// One round a framed challenger elsewhere absorbed (the CUDA prover's `FsChallenger`: every item since the last squeeze,
/// then the squeeze), made again through `ch`: each item by the call that frames it (an observe's kind picks
/// `observe_f128` or `observe_f128_slice`), then the squeeze. So a wrapper in front of the live stream (M1's masking) sees
/// the calls the CPU prover makes. The stream's domain frame is `ch`'s own. Returns the round's `n` coins.
pub fn replay_round<C: Challenger>(ch: &mut C, content: &[u8], n: usize) -> Result<Vec<F128>, String> {
    let body = |at: usize, w: usize| content.get(at..at + w).ok_or_else(|| "a frame overruns its round".to_string());
    let mut at = 0usize;
    while at < content.len() {
        let h = body(at, 16).map_err(|_| "a truncated frame header".to_string())?;
        let (op, kind, len) = (h[0], h[1], u64::from_le_bytes(h[8..].try_into().unwrap()) as usize);
        at += 16;
        let padded = len.div_ceil(16) * 16;
        match op {
            OP_DOMAIN => {
                body(at, padded)?;
                at += padded;
            }
            OP_LABEL => {
                ch.observe_label(body(at, len)?);
                at += padded;
            }
            OP_BYTES => {
                ch.observe_bytes(body(at, len)?);
                at += padded;
            }
            OP_OBSERVE => {
                let vs: Vec<F128> = body(at, 16 * len)?.chunks_exact(16)
                    .map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap())))
                    .collect();
                match kind {
                    KIND_SCALAR if len == 1 => ch.observe_f128(vs[0]),
                    KIND_SLICE => ch.observe_f128_slice(&vs),
                    _ => return Err(format!("an observe of kind {kind} and length {len}")),
                }
                at += 16 * len;
            }
            OP_SQUEEZE if len == 0 && kind == KIND_SLICE => {
                ch.sample_f128_vec(0);
            }
            OP_SQUEEZE => {
                if at != content.len() {
                    return Err("a squeeze before the round's end".into());
                }
                if len != n {
                    return Err(format!("the round squeezes {len} coins, the caller asked for {n}"));
                }
                return match kind {
                    KIND_SCALAR if n == 1 => Ok(vec![ch.sample_f128()]),
                    KIND_SLICE => Ok(ch.sample_f128_vec(n)),
                    _ => Err(format!("a squeeze of kind {kind} and length {n}")),
                };
            }
            x => return Err(format!("unknown frame op {x}")),
        }
    }
    Err("a round without its squeeze".into())
}

pub fn sha256(b: &[u8]) -> [u8; 32] {
    Sha256::digest(b).into()
}

/// The unit draw (the Lean verifier's PROTOCOL.md §7.3): the verifier draws the units to prove in the clear from its OS
/// randomness, after registration and before the statement is fixed, with `verity.ml.randomness`'s exact samplers read straight
/// from OS bytes (no derivation). The draw is the object `{"law": "subset", "population": N, "k": K, "units": [...]}`,
/// `{"law": "bernoulli", "population": N, "p": "num/den", "units": [...]}` or `{"law": "stratified", "population": N, "k": K,
/// "strata": [{"template", "units": [[lo, hi], ...], "k"}, ...], "units": [...]}`, units ascending. A stratified law is the
/// verifier's own expansion of `stratified:K` over its program and query (core's `one_stage.draw.derive`, Lean
/// `HmRow.stratifiedLaw`): this module checks its form (Lean `Flock.Draw.strataProblem`) and draws by it, never derives it.
pub mod unit_draw {
    use serde_json::{Value, json};

    /// A stratum of the stratified law: a template's units as ascending, disjoint, non-adjacent half-open ranges, and its draw.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Stratum {
        pub template: String,
        pub units: Vec<(u64, u64)>,
        pub k: u64,
    }

    impl Stratum {
        pub fn size(&self) -> u64 {
            self.units.iter().map(|(lo, hi)| hi - lo).sum()
        }

        /// The stratum's `j`-th unit, `j < size`.
        fn unit(&self, mut j: u64) -> u64 {
            for &(lo, hi) in &self.units {
                if j < hi - lo {
                    return lo + j;
                }
                j -= hi - lo;
            }
            unreachable!("a stratum's local index is below its size")
        }

        fn holds(&self, u: u64) -> bool {
            self.units.iter().any(|&(lo, hi)| lo <= u && u < hi)
        }

        fn to_json(&self) -> Value {
            json!({"template": self.template, "units": self.units.iter().map(|&(lo, hi)| json!([lo, hi])).collect::<Vec<_>>(), "k": self.k})
        }

        /// Exactly `{template, units, k}`, its units non-empty `[lo, hi)` ranges, ascending, disjoint and not adjacent.
        fn of_json(v: &Value) -> Result<Stratum, String> {
            let bad = || "U2: a stratum is exactly {template, units, k}".to_string();
            let o = v.as_object().ok_or_else(bad)?;
            if o.keys().map(|k| k.as_str()).ne(["k", "template", "units"]) {
                return Err(bad());
            }
            let template = v["template"].as_str().ok_or_else(bad)?.to_string();
            let k = v["k"].as_u64().ok_or("U2: a stratum's k")?;
            let ranges = || format!("U2: stratum '{template}': units are non-empty [lo, hi) ranges");
            let units: Vec<(u64, u64)> = v["units"].as_array().ok_or_else(ranges)?.iter().map(|r| match r.as_array().map(|p| p.as_slice()) {
                Some([lo, hi]) => match (lo.as_u64(), hi.as_u64()) {
                    (Some(lo), Some(hi)) if lo < hi => Ok((lo, hi)),
                    _ => Err(ranges()),
                },
                _ => Err(ranges()),
            }).collect::<Result<_, _>>()?;
            if units.is_empty() {
                return Err(ranges());
            }
            if units.windows(2).any(|w| w[1].0 <= w[0].1) {
                return Err(format!("U2: stratum '{template}': ranges ascending, disjoint and not adjacent"));
            }
            Ok(Stratum { template, units, k })
        }
    }

    /// A stratum's draw under the stratified rule (core's `draw.stratum_k`, Lean `Flock.Draw.stratumK`):
    /// `max(1, round(k · size / population))`, rounding half up.
    pub fn stratum_k(k: u64, size: u64, population: u64) -> u64 {
        ((2 * k as u128 * size as u128 + population as u128) / (2 * population as u128)).max(1) as u64
    }

    /// Why a stratified law isn't well formed over `population` units (Lean `Flock.Draw.strataProblem`), or `None`:
    /// `1 <= k <= population`, distinct templates, each stratum's `k` the rule's, the ranges tiling `[0, population)` exactly.
    pub fn strata_problem(k: u64, strata: &[Stratum], population: u64) -> Option<String> {
        if strata.is_empty() || k < 1 || k > population {
            return Some("U2: strata and 1 <= k <= population".into());
        }
        for (i, s) in strata.iter().enumerate() {
            if strata[..i].iter().any(|t| t.template == s.template) {
                return Some(format!("U2: template '{}' twice", s.template));
            }
            if s.k != stratum_k(k, s.size(), population) {
                return Some(format!("U2: stratum '{}': k is not the rule's", s.template));
            }
        }
        let mut ranges: Vec<(u64, u64)> = strata.iter().flat_map(|s| s.units.iter().copied()).collect();
        ranges.sort_unstable();
        let mut at = 0;
        for (lo, hi) in ranges {
            if lo != at {
                return Some("U2: the strata don't tile the population".into());
            }
            at = hi;
        }
        (at != population).then(|| "U2: the strata don't tile the population".into())
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum Law {
        Subset { k: u64 },
        Bernoulli { num: u64, den: u64 },
        /// One stratum per template, a uniform `k_s`-subset of each, independently (Lean `Flock.Draw.Law.stratified`).
        Stratified { k: u64, strata: Vec<Stratum> },
    }

    impl Law {
        /// `subset:K` or `bernoulli:NUM/DEN`.
        pub fn parse(s: &str) -> Result<Law, String> {
            let bad = || format!("a draw law is subset:K or bernoulli:NUM/DEN, not {s:?}");
            match s.split_once(':') {
                Some(("subset", k)) => Ok(Law::Subset { k: k.parse().map_err(|_| bad())? }),
                Some(("bernoulli", p)) => {
                    let (a, b) = p.split_once('/').ok_or_else(bad)?;
                    Ok(Law::Bernoulli { num: a.parse().map_err(|_| bad())?, den: b.parse().map_err(|_| bad())? })
                }
                _ => Err(bad()),
            }
        }

        /// A law object: `{"law": "stratified", "k": K, "strata": [...]}` (the verifier's own expansion, `draw.derive`'s),
        /// `{"law": "subset", "k": K}` or `{"law": "bernoulli", "p": "num/den"}`.
        pub fn of_json(v: &Value) -> Result<Law, String> {
            let keys: Vec<&str> = v.as_object().ok_or("a draw law is an object")?.keys().map(|k| k.as_str()).collect();
            match (v["law"].as_str(), keys.as_slice()) {
                (Some("stratified"), ["k", "law", "strata"]) => {
                    let k = v["k"].as_u64().ok_or("a stratified law's k")?;
                    let strata = v["strata"].as_array().ok_or("a stratified law's strata")?.iter().map(Stratum::of_json)
                        .collect::<Result<_, _>>()?;
                    Ok(Law::Stratified { k, strata })
                }
                (Some("subset"), ["k", "law"]) => Ok(Law::Subset { k: v["k"].as_u64().ok_or("a subset law's k")? }),
                (Some("bernoulli"), ["law", "p"]) => Law::parse(&format!("bernoulli:{}", v["p"].as_str().unwrap_or(""))),
                _ => Err(format!("a draw law object is {{law: stratified, k, strata}}, {{law: subset, k}} or {{law: bernoulli, p}}, not {v}")),
            }
        }

        /// A draw of each stratum's first `k_s` units (or the first `k`): a stand-in whose tables are every draw's.
        pub fn first(&self, population: u64) -> Option<Value> {
            match self {
                Law::Subset { k } => Some(json!({"law": "subset", "population": population, "k": k, "units": (0..*k).collect::<Vec<_>>()})),
                Law::Stratified { k, strata } => {
                    let mut units: Vec<u64> = strata.iter().flat_map(|s| (0..s.k.min(s.size())).map(|j| s.unit(j))).collect();
                    units.sort_unstable();
                    Some(json!({"law": "stratified", "population": population, "k": k,
                                "strata": strata.iter().map(Stratum::to_json).collect::<Vec<_>>(), "units": units}))
                }
                Law::Bernoulli { .. } => None,
            }
        }
    }

    fn os_bytes(n: usize) -> Vec<u8> {
        let mut b = vec![0u8; n];
        getrandom::fill(&mut b).expect("the OS random generator failed: refusing to draw units without it");
        b
    }

    /// `uniform(n)`: big-endian words of `w = ceil((bitlen(n) + 64) / 8)` OS bytes, the first `x < 2^(8w) - 2^(8w) mod n`,
    /// reduced mod `n`.
    pub fn uniform(n: u64) -> u64 {
        assert!(n > 0 && n < 1 << 56, "uniform: 0 < n < 2^56");
        let w = (64 - n.leading_zeros() + 64).div_ceil(8) as usize;
        let m = 1u128 << (8 * w);
        let lim = m - m % n as u128;
        loop {
            let x = os_bytes(w).iter().fold(0u128, |a, &y| (a << 8) | y as u128);
            if x < lim {
                return (x % n as u128) as u64;
            }
        }
    }

    /// A uniformly random `k`-subset of `[0, n)`, sorted: the partial Fisher-Yates of verity.ml.randomness's Key.subset.
    fn subset(k: u64, n: u64) -> Vec<u64> {
        let mut swapped: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
        let at = |m: &std::collections::BTreeMap<u64, u64>, i: u64| *m.get(&i).unwrap_or(&i);
        let mut out = Vec::new();
        for i in 0..k {
            let j = i + uniform(n - i);
            let (vi, vj) = (at(&swapped, i), at(&swapped, j));
            swapped.insert(i, vj);
            swapped.insert(j, vi);
            out.push(vj);
        }
        out.sort_unstable();
        out
    }

    /// The draw over `[0, population)`. A stratified draw is one `subset` per stratum, each on its own OS bytes, local index
    /// `j` mapped to the stratum's `j`-th unit, the union ascending (Lean `Flock.Draw.stratified`).
    pub fn draw(law: &Law, population: u64) -> Result<Value, String> {
        let v = match law {
            Law::Subset { k } => {
                if *k > population {
                    return Err(format!("subset: k = {k} > population {population}"));
                }
                json!({"law": "subset", "population": population, "k": k, "units": subset(*k, population)})
            }
            Law::Bernoulli { num, den } => {
                if *den == 0 || num > den {
                    return Err(format!("bernoulli: p = {num}/{den} is not a probability"));
                }
                let units: Vec<u64> = (0..population).filter(|_| uniform(*den) < *num).collect();
                json!({"law": "bernoulli", "population": population, "p": format!("{num}/{den}"), "units": units})
            }
            Law::Stratified { k, strata } => {
                if let Some(why) = strata_problem(*k, strata, population) {
                    return Err(why);
                }
                let mut units: Vec<u64> = strata.iter().flat_map(|s| subset(s.k.min(s.size()), s.size()).into_iter().map(|j| s.unit(j)).collect::<Vec<_>>())
                    .collect();
                units.sort_unstable();
                json!({"law": "stratified", "population": population, "k": k,
                       "strata": strata.iter().map(Stratum::to_json).collect::<Vec<_>>(), "units": units})
            }
        };
        check(&v)?;
        Ok(v)
    }

    /// U2: exactly the law's keys; units ascending, distinct and in `[0, N)`; a subset has K <= N units; a Bernoulli p is
    /// `num/den` with `0 <= num <= den`, `den > 0`, no unit at p = 0 and every unit at p = 1; a stratified draw's strata are
    /// well formed and tile the population (`strata_problem`), and exactly `k_s` of its units fall in each. Returns the units.
    pub fn check(v: &Value) -> Result<Vec<usize>, String> {
        let o = v.as_object().ok_or("U2: the draw is not an object")?;
        let n = v["population"].as_u64().ok_or("U2: population")?;
        let units: Vec<u64> = v["units"].as_array().ok_or("U2: units")?.iter().map(|u| u.as_u64().ok_or("U2: a unit"))
            .collect::<Result<_, _>>()?;
        if units.windows(2).any(|w| w[0] >= w[1]) || units.iter().any(|&u| u >= n) {
            return Err("U2: units are not ascending, distinct and inside the population".into());
        }
        let keys: Vec<&str> = o.keys().map(|k| k.as_str()).collect();
        match v["law"].as_str() {
            Some("subset") => {
                let k = v["k"].as_u64().ok_or("U2: k")?;
                if keys != ["k", "law", "population", "units"] || k > n || units.len() as u64 != k {
                    return Err("U2: not a subset draw of k <= N units".into());
                }
            }
            Some("bernoulli") => {
                let (a, b) = v["p"].as_str().and_then(|p| p.split_once('/')).ok_or("U2: p")?;
                let (num, den): (u64, u64) = (a.parse().map_err(|_| "U2: p")?, b.parse().map_err(|_| "U2: p")?);
                if keys != ["law", "p", "population", "units"] || den == 0 || num > den || (num == 0 && !units.is_empty())
                    || (num == den && units.len() as u64 != n)
                {
                    return Err("U2: not a Bernoulli draw".into());
                }
            }
            Some("stratified") => {
                if keys != ["k", "law", "population", "strata", "units"] {
                    return Err("U2: unit_draw's keys are not law, population, k, strata, units".into());
                }
                let k = v["k"].as_u64().ok_or("U2: k")?;
                let strata: Vec<Stratum> = v["strata"].as_array().ok_or("U2: unit_draw.strata is not a list")?.iter().map(Stratum::of_json)
                    .collect::<Result<_, _>>()?;
                if let Some(why) = strata_problem(k, &strata, n) {
                    return Err(why);
                }
                for s in &strata {
                    let drawn = units.iter().filter(|&&u| s.holds(u)).count() as u64;
                    if drawn != s.k {
                        return Err(format!("U2: stratum '{}' holds {drawn} drawn units, not its k = {}", s.template, s.k));
                    }
                }
            }
            _ => return Err("U2: an unknown law".into()),
        }
        Ok(units.into_iter().map(|u| u as usize).collect())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn law() -> Law {
            // two templates, one in two ranges: 10 + 6 units, k = 5 -> k_s = round(50/16) = 3 and round(30/16) = 2
            Law::of_json(&json!({"law": "stratified", "k": 5, "strata": [
                {"template": "A", "units": [[0, 4], [10, 16]], "k": 3}, {"template": "B", "units": [[4, 10]], "k": 2}]})).unwrap()
        }

        #[test]
        fn a_stratified_draw_takes_each_strata_k_from_its_own_units() {
            for _ in 0..200 {
                let d = draw(&law(), 16).unwrap();
                let units = check(&d).unwrap();
                assert_eq!(units.len(), 5);
                assert_eq!(units.iter().filter(|&&u| (4..10).contains(&u)).count(), 2, "{d}");
            }
        }

        #[test]
        fn every_unit_of_a_stratum_is_drawn() {
            let mut seen = [false; 16];
            for _ in 0..400 {
                for u in check(&draw(&law(), 16).unwrap()).unwrap() {
                    seen[u] = true;
                }
            }
            assert!(seen.iter().all(|&s| s));
        }

        #[test]
        fn the_rule_rounds_half_up_with_a_floor_of_one() {
            assert_eq!((stratum_k(5, 10, 16), stratum_k(5, 6, 16), stratum_k(1, 1, 1000), stratum_k(2, 3, 4)), (3, 2, 1, 2));
        }

        #[test]
        fn a_malformed_law_or_draw_is_refused() {
            let Law::Stratified { k, strata } = law() else { unreachable!() };
            assert!(strata_problem(k, &strata, 17).is_some(), "the strata don't tile 17 units");
            let mut wrong = strata.clone();
            wrong[1].k = 3;
            assert!(strata_problem(k, &wrong, 16).is_some(), "k_s not the rule's");
            let mut twice = strata.clone();
            twice[1].template = "A".into();
            assert!(strata_problem(k, &twice, 16).is_some(), "a template twice");
            assert!(draw(&law(), 17).is_err());
            let d = draw(&law(), 16).unwrap();
            let mut moved = d.clone();
            let u: Vec<u64> = (0..5).collect();
            moved["units"] = json!(u);
            assert!(check(&moved).is_err(), "four of A's units and one of B's");
            let mut extra = d.clone();
            extra["p"] = json!("1/2");
            assert!(check(&extra).is_err(), "a key no stratified draw has");
            assert!(Law::of_json(&json!({"law": "stratified", "k": 5, "strata": [{"template": "A", "units": [[0, 4], [4, 16]], "k": 5}]})).is_err(),
                    "adjacent ranges");
            assert_eq!(Law::of_json(&json!({"law": "subset", "k": 3})).unwrap(), Law::Subset { k: 3 });
        }

        #[test]
        fn the_stand_in_draw_has_every_draws_count() {
            let d = law().first(16).unwrap();
            assert_eq!(check(&d).unwrap(), vec![0, 1, 2, 4, 5]);
        }
    }
}

/// Whether `d` is `b`'s digest under the hash its length names (SHA-512 for 64 bytes, else SHA-256).
fn same_digest(d: &[u8], b: &[u8]) -> bool {
    if d.len() == 64 { Sha512::digest(b)[..] == d[..] } else { Sha256::digest(b)[..] == d[..] }
}

/// The bytes of Flock Merkle digests, whatever width the linked Flock commits with (32, or 64 under verity's SHA-512 patch).
pub fn merkle_digest_bytes(d: &[flock_hash::Digest]) -> &[u8] {
    // SAFETY: a digest is a plain byte array (`[u8; 32]` or `[[u8; 32]; 2]`): no padding, every byte initialized.
    unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, std::mem::size_of_val(d)) }
}

/// One Flock Merkle digest from the first `size_of::<Digest>()` bytes of `b`.
pub fn merkle_digest_read(b: &[u8]) -> flock_hash::Digest {
    let mut d = flock_hash::Digest::default();
    let n = std::mem::size_of::<flock_hash::Digest>();
    // SAFETY: as in `merkle_digest_bytes`; `b` holds at least `n` bytes (the slice index panics otherwise).
    unsafe { std::ptr::copy_nonoverlapping(b[..n].as_ptr(), &mut d as *mut _ as *mut u8, n) };
    d
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn f128_hex(v: &F128) -> String {
    format!("{:016x}{:016x}", v.hi, v.lo)
}

/// Step 0 replayed from a session record alone: the recorded seed opens the recorded commitment, and every coin request of the
/// session (each stream round and the link points, by their session-wide index `g`, however the reps interleave) is
/// `coin_seed::coins(coin_key(seed, hello), g, n)`, the indices `0 .. requests` each exactly once. Returns the request count.
pub fn replay_coin_seed(rec: &Value) -> Result<u64, String> {
    let cs = &rec["coin_seed"];
    if cs.is_null() {
        return Err("the record has no step-0 seed".into());
    }
    let hb = |k: &str| -> Result<Vec<u8>, String> {
        let s = cs[k].as_str().ok_or(format!("coin_seed.{k} missing (the verdict was never issued?)"))?;
        (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16)).collect::<Result<_, _>>().map_err(|e| e.to_string())
    };
    let h32 = |k: &str| -> Result<[u8; 32], String> { hb(k)?.try_into().map_err(|_| format!("coin_seed.{k} is not 32 bytes")) };
    let (seed, nonce, commit) = (h32("seed")?, h32("nonce")?, hb("commitment")?);
    if cs["scheme"] != coin_seed::SCHEME || cs["domain"] != coin_seed::DOMAIN || coin_seed::commitment(&seed, &nonce)[..] != commit[..] {
        return Err("the recorded seed does not open the recorded commitment under the pinned scheme".into());
    }
    let hello = cs["context"]["hello"].as_str().ok_or("coin_seed.context.hello missing")?;
    if rec["hello"].as_str() != Some(hello) {
        return Err("the derivation context is not the session's hello".into());
    }
    let key = coin_seed::coin_key(&seed, hello);
    let mut got: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    let mut put = |g: Option<u64>, coins: Vec<String>, what: String| -> Result<(), String> {
        let g = g.ok_or(format!("{what}: no coin-request index"))?;
        if got.insert(g, coins).is_some() {
            return Err(format!("coin request {g} recorded twice"));
        }
        Ok(())
    };
    for st in rec["streams"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        for r in st["rounds"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let coins = r["coins"].as_array().map(|a| a.iter().filter_map(|c| c.as_str().map(String::from)).collect()).unwrap_or_default();
            put(r["g"].as_u64(), coins, format!("{} round {}", st["stream"], r["k"]))?;
        }
    }
    if !rec["link"].is_null() {
        let pts: Vec<String> = rec["link"]["points"].as_array().map(|a| a.iter().flat_map(|p| p.as_array().cloned().unwrap_or_default())
            .filter_map(|c| c.as_str().map(String::from)).collect()).unwrap_or_default();
        put(rec["link"]["points_g"].as_u64(), pts, "the link points".into())?;
    }
    let requests = cs["requests"].as_u64().ok_or("coin_seed.requests missing")?;
    if got.keys().copied().ne(0..requests) {
        return Err(format!("the recorded coin requests are not exactly 0..{requests}"));
    }
    for (g, coins) in &got {
        let want: Vec<String> = coin_seed::coins(&key, *g, coins.len()).chunks(16)
            .map(|c| f128_hex(&F128 { lo: u64::from_le_bytes(c[..8].try_into().unwrap()), hi: u64::from_le_bytes(c[8..].try_into().unwrap()) })).collect();
        if &want != coins {
            return Err(format!("coin request {g} is not the seed's derivation"));
        }
    }
    Ok(requests)
}

/// Stream names: rep roots are `<table>/rep<r>`, children `<parent>/f<i>`.
pub fn rep_stream(table: &str, rep: usize) -> String {
    format!("{table}/rep{rep}")
}

fn child_stream(parent: &str, idx: u32) -> String {
    format!("{parent}/f{idx}")
}

// ---------------------------------------------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Req {
    Hello(String),
    Link(Vec<u8>),
    /// flock-link L1: root_F (the prime side's first commitment), each table's root_B, each table's public words.
    Commit { root_f: Vec<u8>, roots: Vec<(String, Vec<u8>)>, publics: Vec<(String, Vec<u8>)> },
    Open { stream: String, parent: String, label: Vec<u8>, at: u64 },
    Round { stream: String, content: Vec<u8>, n: u32 },
    Proof { table: String, rep: u32, bytes: Vec<u8> },
    Finish,
    /// A round-trip probe on the open connection: answered `Ok`, touches no session state.
    Ping(Vec<u8>),
    /// A live prime round: the prime transcript's state (32 bytes) and the number of BabyBear^6 challenges wanted.
    Prime { state: Vec<u8>, n: u32 },
    /// Registration done: the verifier draws the units to prove (`unit_draw`), before `Hello` fixes the statement.
    Register,
}

#[derive(Clone, Debug)]
pub enum Resp {
    Ok,
    Coins(Vec<F128>),
    Refused(String),
    Verdict(String),
    /// A prime round's coins: 6 n words, each uniform below [`BABYBEAR_P`].
    Words(Vec<u32>),
    /// The unit draw (canonical JSON; `null` when the session draws none).
    Draw(String),
}

pub const BABYBEAR_P: u32 = 2013265921;
/// Coordinates per prime challenge (BabyBear^6).
pub const PRIME_DEG: usize = 6;
pub const PRIME_MODE: &str = "live-coins/v1";

fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_bytes(w: &mut Vec<u8>, b: &[u8]) {
    put_u32(w, b.len() as u32);
    w.extend_from_slice(b);
}

struct Cur<'a>(&'a [u8], usize);
impl Cur<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.0.get(self.1..self.1 + 4).ok_or("short frame")?;
        self.1 += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.0.get(self.1..self.1 + 8).ok_or("short frame")?;
        self.1 += 8;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<Vec<u8>, String> {
        let n = self.u32()? as usize;
        let b = self.0.get(self.1..self.1 + n).ok_or("short frame")?;
        self.1 += n;
        Ok(b.to_vec())
    }
    fn string(&mut self) -> Result<String, String> {
        String::from_utf8(self.bytes()?).map_err(|e| e.to_string())
    }
}

impl Req {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Vec::new();
        match self {
            Req::Hello(s) => {
                w.push(1);
                put_bytes(&mut w, s.as_bytes());
            }
            Req::Link(b) => {
                w.push(2);
                put_bytes(&mut w, b);
            }
            Req::Open { stream, parent, label, at } => {
                w.push(3);
                put_bytes(&mut w, stream.as_bytes());
                put_bytes(&mut w, parent.as_bytes());
                put_bytes(&mut w, label);
                w.extend_from_slice(&at.to_le_bytes());
            }
            Req::Round { stream, content, n } => {
                w.push(4);
                put_bytes(&mut w, stream.as_bytes());
                put_bytes(&mut w, content);
                put_u32(&mut w, *n);
            }
            Req::Proof { table, rep, bytes } => {
                w.push(5);
                put_bytes(&mut w, table.as_bytes());
                put_u32(&mut w, *rep);
                put_bytes(&mut w, bytes);
            }
            Req::Finish => w.push(6),
            Req::Ping(b) => {
                w.push(9);
                put_bytes(&mut w, b);
            }
            Req::Commit { root_f, roots, publics } => {
                w.push(7);
                put_bytes(&mut w, root_f);
                for list in [roots, publics] {
                    put_u32(&mut w, list.len() as u32);
                    for (t, b) in list {
                        put_bytes(&mut w, t.as_bytes());
                        put_bytes(&mut w, b);
                    }
                }
            }
            Req::Prime { state, n } => {
                w.push(8);
                put_bytes(&mut w, state);
                put_u32(&mut w, *n);
            }
            Req::Register => w.push(10),
        }
        w
    }

    pub fn decode(b: &[u8]) -> Result<Req, String> {
        let mut c = Cur(b, 1);
        Ok(match b.first().ok_or("empty frame")? {
            1 => Req::Hello(c.string()?),
            2 => Req::Link(c.bytes()?),
            3 => Req::Open { stream: c.string()?, parent: c.string()?, label: c.bytes()?, at: c.u64()? },
            4 => Req::Round { stream: c.string()?, content: c.bytes()?, n: c.u32()? },
            5 => Req::Proof { table: c.string()?, rep: c.u32()?, bytes: c.bytes()? },
            6 => Req::Finish,
            9 => Req::Ping(c.bytes()?),
            7 => {
                let root_f = c.bytes()?;
                let mut lists = Vec::new();
                for _ in 0..2 {
                    let n = c.u32()? as usize;
                    let mut l = Vec::new();
                    for _ in 0..n.min(1 << 10) {
                        l.push((c.string()?, c.bytes()?));
                    }
                    lists.push(l);
                }
                let publics = lists.pop().unwrap();
                let roots = lists.pop().unwrap();
                Req::Commit { root_f, roots, publics }
            }
            8 => Req::Prime { state: c.bytes()?, n: c.u32()? },
            10 => Req::Register,
            t => return Err(format!("unknown request tag {t}")),
        })
    }
}

impl Resp {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Vec::new();
        match self {
            Resp::Ok => w.push(1),
            Resp::Coins(v) => {
                w.push(2);
                put_u32(&mut w, v.len() as u32);
                for c in v {
                    push_f128(&mut w, *c);
                }
            }
            Resp::Refused(s) => {
                w.push(3);
                put_bytes(&mut w, s.as_bytes());
            }
            Resp::Verdict(s) => {
                w.push(4);
                put_bytes(&mut w, s.as_bytes());
            }
            Resp::Words(v) => {
                w.push(5);
                put_u32(&mut w, v.len() as u32);
                for x in v {
                    put_u32(&mut w, *x);
                }
            }
            Resp::Draw(s) => {
                w.push(6);
                put_bytes(&mut w, s.as_bytes());
            }
        }
        w
    }

    pub fn decode(b: &[u8]) -> Result<Resp, String> {
        let mut c = Cur(b, 1);
        Ok(match b.first().ok_or("empty frame")? {
            1 => Resp::Ok,
            2 => {
                let n = c.u32()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let lo = c.u64()?;
                    let hi = c.u64()?;
                    v.push(F128 { lo, hi });
                }
                Resp::Coins(v)
            }
            3 => Resp::Refused(c.string()?),
            4 => Resp::Verdict(c.string()?),
            5 => {
                let n = c.u32()? as usize;
                Resp::Words((0..n).map(|_| c.u32()).collect::<Result<_, _>>()?)
            }
            6 => Resp::Draw(c.string()?),
            t => return Err(format!("unknown response tag {t}")),
        })
    }
}

pub fn write_frame(w: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    w.write_all(&(b.len() as u32).to_le_bytes())?;
    w.write_all(b)?;
    w.flush()
}

pub fn read_frame(r: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let mut b = vec![0u8; u32::from_le_bytes(n) as usize];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// How the prover reaches the verifier.
pub trait Transport: Send {
    fn call(&mut self, req: Req) -> Result<Resp, String>;
    /// Bytes sent and received so far, and the number of calls.
    fn stats(&self) -> (u64, u64, u64) {
        (0, 0, 0)
    }
    /// Seconds the prover spent waiting inside `call` (the latency live coins add).
    fn wait_s(&self) -> f64 {
        0.0
    }
}

pub struct TcpTransport {
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
    sent: u64,
    recv: u64,
    calls: u64,
    wait: std::time::Duration,
}

impl TcpTransport {
    pub fn connect(addr: &str) -> std::io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        s.set_nodelay(true)?;
        Ok(Self { r: BufReader::new(s.try_clone()?), w: BufWriter::new(s), sent: 0, recv: 0, calls: 0, wait: Default::default() })
    }
}

impl TcpTransport {
    /// Round trips of a 64-byte `Ping` on this open connection, before the session starts (not counted in the session's
    /// bytes, calls or wait): (median ms, min ms, n).
    pub fn ping_rtt(&mut self, n: usize) -> (f64, f64, usize) {
        let b = Req::Ping(vec![0u8; 64]).encode();
        let mut ts = Vec::with_capacity(n);
        for _ in 0..n {
            let t = std::time::Instant::now();
            if write_frame(&mut self.w, &b).is_err() || read_frame(&mut self.r).is_err() {
                break;
            }
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        if ts.is_empty() {
            return (f64::NAN, f64::NAN, 0);
        }
        let mut v = ts.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (v[v.len() / 2], v[0], v.len())
    }
}

impl TcpTransport {
    /// One round trip of the request frame `b` as it is, encoded or not (the wire negatives' doctored frames).
    pub fn call_raw(&mut self, b: &[u8]) -> Result<Resp, String> {
        let t = std::time::Instant::now();
        self.sent += b.len() as u64 + 4;
        self.calls += 1;
        write_frame(&mut self.w, b).map_err(|e| format!("send: {e}"))?;
        let r = read_frame(&mut self.r).map_err(|e| format!("recv: {e}"))?;
        self.recv += r.len() as u64 + 4;
        self.wait += t.elapsed();
        Resp::decode(&r)
    }
}

impl Transport for TcpTransport {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        self.call_raw(&req.encode())
    }
    fn stats(&self) -> (u64, u64, u64) {
        (self.sent, self.recv, self.calls)
    }
    fn wait_s(&self) -> f64 {
        self.wait.as_secs_f64()
    }
}

/// The server in the same process (tests and negatives); the messages still go through encode/decode.
pub struct InProc(pub Arc<Mutex<Server>>);

impl Transport for InProc {
    fn call(&mut self, req: Req) -> Result<Resp, String> {
        let req = Req::decode(&req.encode())?;
        let resp = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).handle(req);
        Resp::decode(&resp.encode())
    }
}

pub type SharedTransport = Arc<Mutex<Box<dyn Transport>>>;

pub fn shared(t: impl Transport + 'static) -> SharedTransport {
    Arc::new(Mutex::new(Box::new(t)))
}

// ---------------------------------------------------------------------------------------------------------------
// Prover side
// ---------------------------------------------------------------------------------------------------------------

/// Adversarial provers for the negatives: what the prover commits to the verifier differs from what it puts in the
/// proof. `round` counts the stream's coin-issuing rounds from 0.
#[derive(Clone, Debug)]
pub enum Tamper {
    /// Flip a byte of round `round`'s committed content: the proof's message at that point is not the one the coin
    /// was issued against (a message altered after its coin).
    Alter { stream: String, round: usize },
    /// Commit round `round` without its messages and send them at the start of the next round (a message sent after
    /// its coin).
    Defer { stream: String, round: usize },
    /// Absorb `pow_nonce` at every PoW site but put a different nonce in the proof.
    Nonce,
}

/// The prover's challenger in a live session. See the crate docs.
pub struct LiveChallenger {
    stream: String,
    link: SharedTransport,
    pending: Vec<u8>,
    rounds: u64,
    forks: AtomicU32,
    /// The value `grind_pow` returns and absorbs: `0`, the only nonce the verifier accepts (R4). Live coins need no
    /// grinding; a nonzero value models a prover that chooses its nonce (negative 4), which the verifier refuses.
    pub pow_nonce: u64,
    pub tamper: Option<Tamper>,
    deferred: Vec<u8>,
}

impl LiveChallenger {
    /// A rep's root stream. The server refuses its coins until the stream is opened (`open_rep`).
    pub fn new(link: SharedTransport, table: &str, rep: usize, domain: &[u8]) -> Result<Self, String> {
        let stream = rep_stream(table, rep);
        match link.lock().unwrap().call(Req::Open {
            stream: stream.clone(),
            parent: String::new(),
            label: domain.to_vec(),
            at: 0,
        })? {
            Resp::Ok => {}
            Resp::Refused(e) => return Err(e),
            r => return Err(format!("unexpected {r:?}")),
        }
        Ok(Self::stream(link, stream, frame_domain(domain)))
    }

    fn stream(link: SharedTransport, stream: String, pending: Vec<u8>) -> Self {
        Self {
            stream,
            link,
            pending,
            rounds: 0,
            forks: AtomicU32::new(0),
            pow_nonce: 0,
            tamper: None,
            deferred: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.stream
    }

    /// One round: `content` is everything absorbed since the previous squeeze, squeeze header included.
    pub fn round(&mut self, mut content: Vec<u8>, n: usize) -> Vec<F128> {
        let k = self.rounds as usize;
        if !self.deferred.is_empty() {
            let mut c = std::mem::take(&mut self.deferred);
            c.extend_from_slice(&content);
            content = c;
        }
        match &self.tamper {
            Some(Tamper::Alter { stream, round }) if *stream == self.stream && *round == k => {
                content[0] ^= 1;
            }
            Some(Tamper::Defer { stream, round }) if *stream == self.stream && *round == k => {
                let tail = content.split_off(content.len() - 16);
                self.deferred = content;
                content = tail;
            }
            _ => {}
        }
        self.rounds += 1;
        let resp = self.link.lock().unwrap().call(Req::Round { stream: self.stream.clone(), content, n: n as u32 });
        match resp {
            Ok(Resp::Coins(c)) if c.len() == n => c,
            Ok(Resp::Refused(e)) => panic!("LIVE-REFUSED {}: {e}", self.stream),
            other => panic!("LIVE-ERROR {}: {other:?}", self.stream),
        }
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let content = std::mem::take(&mut self.pending);
        self.round(content, n)
    }

    fn open_child(&self, label: &[u8]) -> String {
        let idx = self.forks.fetch_add(1, Ordering::SeqCst);
        let child = child_stream(&self.stream, idx);
        let resp = self.link.lock().unwrap().call(Req::Open {
            stream: child.clone(),
            parent: self.stream.clone(),
            label: label.to_vec(),
            at: self.rounds,
        });
        match resp {
            Ok(Resp::Ok) => child,
            other => panic!("LIVE-REFUSED open {child}: {other:?}"),
        }
    }
}

/// Absorb and squeeze methods shared by the live and the replay challenger: both must frame identically.
macro_rules! framed_challenger {
    () => {
        fn observe_label(&mut self, label: &[u8]) {
            push_header(&mut self.pending, OP_LABEL, 0, label.len() as u64);
            push_padded(&mut self.pending, label);
        }
        fn observe_f128(&mut self, v: F128) {
            push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
            push_f128(&mut self.pending, v);
        }
        fn observe_f128_slice(&mut self, vs: &[F128]) {
            push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
            for v in vs {
                push_f128(&mut self.pending, *v);
            }
        }
        fn observe_bytes(&mut self, b: &[u8]) {
            push_header(&mut self.pending, OP_BYTES, 0, b.len() as u64);
            push_padded(&mut self.pending, b);
        }
        fn sample_f128(&mut self) -> F128 {
            self.squeeze(KIND_SCALAR, 1)[0]
        }
        fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
            self.squeeze(KIND_SLICE, n)
        }
    };
}

impl Challenger for LiveChallenger {
    framed_challenger!();

    /// R4: no grinding. The nonce is absorbed like any message; the coin after it is the verifier's.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        let n = self.pow_nonce;
        self.observe_bytes(&n.to_le_bytes());
        if matches!(self.tamper, Some(Tamper::Nonce)) { n ^ 1 } else { n }
    }

    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        unreachable!("the prover's challenger never verifies")
    }

    /// R3: the child is a live stream of its own, opened at this position of the parent (after the parent's seed
    /// round, so the parent's messages before the fork are committed first); its coins come from the verifier.
    fn fork(&mut self, label: &'static [u8]) -> Self {
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let name = self.open_child(label);
        let mut c = LiveChallenger::stream(self.link.clone(), name, frame_domain(label));
        c.observe_f128_slice(&seed);
        c.pow_nonce = self.pow_nonce;
        c.tamper = self.tamper.clone();
        c
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Verifier side: coin server
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Binding {
    /// `UnionInstance::bind_statement`: label, registry digest, counts (u64 LE each), root.
    Mixed { registry_digest: [u8; 32], counts_le: Vec<u8> },
    /// `verify_ligerito` / the CUDA prover: label, statement digest, root.
    R1cs { statement_digest: [u8; 32] },
    /// `UnionInstance::bind_statement_circuit`: the Mixed binding, then `flock-circuit-stmt-v2`, the circuit digest
    /// and the digest of the public words (pinned from the committed public words, checked by the statement).
    Circuit { registry_digest: [u8; 32], counts_le: Vec<u8>, circuit_digest: [u8; 32] },
}

pub const LABEL_CIRCUIT: &[u8] = b"flock-circuit-stmt-v2";

/// flock-link: the link context the verifier requires before Flock's first coin.
#[derive(Clone, Debug)]
pub struct LinkSpec {
    /// Σ, the link statement digest (scheme, Λ, N, m, the two sub-proof statements): recorded, and sent in `Hello`.
    pub sigma: Vec<u8>,
    /// Coordinates per link point (the linked position space is `{0,1}^m`).
    pub m: usize,
    /// Independent GF(2^128) points (2: the Schwartz–Zippel term squares).
    pub points: usize,
}

#[derive(Clone, Debug)]
pub struct TableSpec {
    pub name: String,
    pub binding: Binding,
}

/// Fixed by the verifier's configuration, never by the prover (R7).
#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub profile: String,
    pub reps: usize,
    pub flavor: String,
    pub tables: Vec<TableSpec>,
    pub require_link: bool,
    /// `None`: the pre-flock-link stub (opaque `Link` bytes). `Some`: the real exchange.
    pub link: Option<LinkSpec>,
    /// The prime prover's coins are served in this session (`Prime` rounds); `Commit` needs at least one before it.
    pub prime_live: bool,
    /// Step 0 (`coin_seed::SCHEME`): the verifier commits to a seed in answer to `Hello`, derives every coin from it and reveals
    /// it in its verdict. Off for every statement that does not ask for it.
    pub coin_seed: bool,
    /// The coin server keeps every round's bytes and the replay compares the proof's messages with them byte for byte (the
    /// round digests stay, as framing); the record carries them. Off for every statement that does not ask for it.
    pub retain_rounds: bool,
    /// The hash of every digest the record binds (root_F, root_B, publics, link, rounds, proofs, stream roots): SHA-256, or
    /// SHA-512 for a statement that asks (SHA-512 on every binding path; named in `Hello`).
    pub sha512_digests: bool,
    /// The draw this statement proves (its public header's `unit_draw`, under Σ); the record keeps it (U1).
    pub unit_draw: Option<Value>,
    /// Draw units live: after registration (`Register`) the server draws them under this law over the population.
    pub draw_law: Option<(unit_draw::Law, u64)>,
    /// M2 (`coin_tree::SCHEME`): at `Hello` the verifier commits to every coin of the session (one HM96 leaf per stream and
    /// round under one SHA-512 root) and opens each round's coins with its answer; the prover checks every opening first.
    pub coin_tree: Option<coin_tree::Spec>,
    /// Or serve this draw, the verifier's own (made outside the server, as `docs/audit-protocols.md` recommends), at `Register`.
    pub draw_given: Option<Value>,
}

impl SessionConfig {
    pub fn flock_128_r2(tables: Vec<TableSpec>) -> Self {
        Self { profile: "fast100".into(), reps: 2, flavor: "rs".into(), tables, require_link: true, link: None, prime_live: false, coin_seed: false,
               retain_rounds: false, sha512_digests: false, unit_draw: None, draw_law: None, draw_given: None, coin_tree: None }
    }

    /// The record's digest of `b`.
    pub fn dg(&self, b: &[u8]) -> Vec<u8> {
        if self.sha512_digests { Sha512::digest(b).to_vec() } else { Sha256::digest(b).to_vec() }
    }

    /// The record's digest keys' suffix (`root_f_sha256` or `root_f_sha512`, ...).
    pub fn dk(&self) -> &'static str {
        if self.sha512_digests { "sha512" } else { "sha256" }
    }

    /// E0: a link exchange is only sound with the R5 gate (no coin before y), so `link: Some` implies `require_link`.
    /// Live prime coins are ordered against root_F and y, so they need the exchange.
    pub fn check(&self) -> Result<(), String> {
        if self.link.is_some() && !self.require_link {
            return Err("E0: a link exchange (link: Some) without the R5 gate (require_link: false) is refused: y could follow Flock's coins".into());
        }
        if self.coin_tree.is_some() && (self.coin_seed || self.prime_live) {
            return Err("a committed coin schedule replaces the step-0 seed, and serves no live prime coins".into());
        }
        if let Some(t) = &self.coin_tree
            && (t.streams.first().map(|x| x.as_str()) != Some(coin_tree::SESSION)
                || self.tables.iter().flat_map(|tb| (0..self.reps).map(move |r| rep_stream(&tb.name, r))).any(|x| !t.streams.contains(&x)))
        {
            return Err("the committed coin schedule must hold the session's stream first, then every table's every rep".into());
        }
        if self.coin_seed && self.prime_live {
            return Err("step-0 coins derive the GF(2^128) coins only; live prime coins with a committed seed are not built".into());
        }
        if self.prime_live && self.link.is_none() {
            return Err("live prime coins without a link exchange are refused: their order is against root_F and y".into());
        }
        Ok(())
    }

    pub fn hello(&self) -> String {
        self.hello_value().to_string()
    }

    /// The `Hello` of a session under the keyed coin tree (`coin_tree::SCHEME`): the prover's fresh nonce in `coins.nonce`, 64
    /// lowercase hex digits; the verifier's own `Hello` with it is what R7 compares byte for byte.
    pub fn hello_with_nonce(&self, nonce: &[u8; coin_tree::NONCE_BYTES]) -> String {
        let mut h = self.hello_value();
        h["coins"]["nonce"] = json!(hex(nonce));
        h.to_string()
    }

    fn hello_value(&self) -> Value {
        let mut h = json!({"profile": self.profile, "reps": self.reps, "flavor": self.flavor,
               "tables": self.tables.iter().map(|t| t.name.clone()).collect::<Vec<_>>()});
        if let Some(l) = &self.link {
            h["link"] = json!({"sigma": hex(&l.sigma), "m": l.m, "points": l.points});
        }
        if self.prime_live {
            h["prime"] = json!(PRIME_MODE);
        }
        if self.coin_seed {
            h["coins"] = json!(coin_seed::SCHEME);
        }
        if let Some(t) = &self.coin_tree {
            h["coins"] = t.to_json();
        }
        if self.retain_rounds {
            h["rounds"] = json!("bytes retained");
        }
        if self.sha512_digests {
            h["digests"] = json!("sha512");
        }
        h
    }
}

/// One live prime round: the state the prover committed, and the words issued after it.
#[derive(Clone, Debug)]
pub struct PrimeRound {
    pub state: [u8; 32],
    pub n: u32,
    pub words: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct RoundRec {
    pub sha: Vec<u8>,
    pub len: usize,
    pub n: u32,
    pub coins: Vec<F128>,
}

#[derive(Clone, Debug, Default)]
pub struct StreamRec {
    pub parent: String,
    pub label: Vec<u8>,
    pub at: u64,
    pub rounds: Vec<RoundRec>,
    pub cap: Option<Vec<u8>>,
}

/// The statement-specific verifier: decode `proof`, pin the flavour and the PCS parameters from the session
/// configuration, run the Flock verifier with `ch`, and return the flattened root the proof carries.
pub trait StatementVerifier: Send {
    fn verify(&self, table: &str, rep: usize, proof: &[u8], ch: &mut ReplayChallenger, link: &LinkCtx<'_>)
        -> Result<Vec<u8>, String>;
    fn domain(&self, table: &str, rep: usize) -> Vec<u8>;
    /// Check a table's committed public words against the verifier's own statement (the endpoints: flock-link C4)
    /// and return the public-words digest its first round must bind (`None`: the binding carries no public words).
    fn statement_publics(&self, table: &str, publics: &[u8]) -> Result<Option<[u8; 32]>, String> {
        if publics.is_empty() { Ok(None) } else { Err(format!("{table}: this statement takes no public words")) }
    }
    /// The phase timers of this verifier's last `verify` call (wall-clock seconds, for the verdict; never part of a proof),
    /// where it keeps them.
    fn take_phases(&self) -> Option<Value> {
        None
    }
    /// Why the statement is one this verifier refuses whatever its sessions hold ([`proved_scope`]), or `None`.
    fn refusal(&self) -> Option<String> {
        None
    }
}

/// What the verifier's record holds for one table when its proofs are replayed: the link points it drew, the y the
/// prover sent for this table (one per point), and the table's committed public words. Empty without a [`LinkSpec`].
pub struct LinkCtx<'a> {
    pub points: &'a [Vec<F128>],
    pub y: &'a [F128],
    pub publics: &'a [u8],
}

#[derive(Clone, Debug)]
struct CommitRec {
    root_f: Vec<u8>,
    roots: BTreeMap<String, Vec<u8>>,
    publics: BTreeMap<String, Vec<u8>>,
    publics_digest: BTreeMap<String, Option<[u8; 32]>>,
}

/// The drawn units' statement: from the draw, the session configuration (carrying the draw) and the verifier.
pub type DrawHook = Box<dyn FnMut(&Value) -> Result<(SessionConfig, Box<dyn StatementVerifier>), String> + Send>;
/// `(stream, round, block)`: a malicious verifier's edit of a committed block ([`coin_tree::CoinTree::draw_with`]).
pub type CoinEdit = Box<dyn Fn(&str, usize, &mut [F128]) + Send>;

pub struct Server {
    pub cfg: SessionConfig,
    verifier: Box<dyn StatementVerifier>,
    hello: Option<String>,
    link: Option<Vec<u8>>,
    commit: Option<CommitRec>,
    points: Vec<Vec<F128>>,
    y: Vec<F128>,
    streams: BTreeMap<String, StreamRec>,
    proofs: BTreeMap<(String, usize), Vec<u8>>,
    prime: Vec<PrimeRound>,
    /// Prime rounds issued before `Commit` (the prime verifier's root_F must follow exactly these).
    prime_before_commit: Option<usize>,
    aborted: Option<String>,
    verdict: Option<Value>,
    urandom: Option<File>,
    /// rebuilt from a record ([`Server::from_record`]): it replays `Hello` without drawing coins (it has no OS source)
    replaying: bool,
    /// step 0: (seed, nonce, commitment, the coin key once `Hello` is accepted, coin requests issued)
    seed: Option<([u8; 32], [u8; 32], [u8; 64], Option<[u8; 32]>, u64)>,
    /// the global index of the last seed-derived coin request, and of the link points' request
    last_g: Option<u64>,
    points_g: Option<u64>,
    /// with a step-0 seed: (stream, round) -> the session-wide index of the coin request that answered it
    /// (`coin_seed::coins(key, g, n)`), so the derivation replays from the record however the reps' rounds interleave
    round_g: BTreeMap<(String, usize), u64>,
    /// with `retain_rounds`: (stream, round) -> the round's bytes as received
    round_msg: BTreeMap<(String, usize), Vec<u8>>,
    /// Rebuilds the statement for the drawn units (`SessionConfig::draw_law`): its configuration (with `unit_draw`) and verifier.
    on_draw: Option<DrawHook>,
    /// a test harness's step-0 seed and nonce (feature `seed-injection` only), used at `Hello` instead of the OS's
    seed_inject: Option<([u8; 32], [u8; 32])>,
    /// M2: the committed coin schedule (drawn at `Hello`), and a test harness's random tape for it (the verifier's coins as a
    /// function of the tape: the simulator's rewinding model)
    coin_tree: Option<coin_tree::CoinTree>,
    /// the prover's nonce from its `Hello` (the keyed coin tree's prefix; in the record)
    coin_nonce: Option<[u8; coin_tree::NONCE_BYTES]>,
    coin_tape: Option<[u8; 32]>,
    /// a test harness's malicious verifier: its edit of the committed blocks (`coin_tree::CoinTree::draw_with`)
    coin_edit: Option<CoinEdit>,
    t0: std::time::Instant,
    /// Seconds spent handling everything but `Finish` (the verifier's compute inside the prover's coin waits).
    handle_s: f64,
}

fn parse_frame<'a>(c: &'a [u8], at: &mut usize, op: u8) -> Result<&'a [u8], String> {
    let h = c.get(*at..*at + 16).ok_or("binding: short round")?;
    if h[0] != op || h[1] != 0 || h[2..8] != [0u8; 6] {
        return Err(format!("binding: expected op {op} at byte {at}, found op {} kind {}", h[0], h[1]));
    }
    let len = u64::from_le_bytes(h[8..16].try_into().unwrap()) as usize;
    let p = c.get(*at + 16..*at + 16 + len).ok_or("binding: short payload")?;
    *at += 16 + len.div_ceil(16) * 16;
    Ok(p)
}

impl Server {
    /// Panics on a configuration [`SessionConfig::check`] refuses; [`Server::try_new`] returns the refusal.
    pub fn new(cfg: SessionConfig, verifier: Box<dyn StatementVerifier>) -> Self {
        Self::try_new(cfg, verifier).unwrap_or_else(|e| panic!("{e}"))
    }

    /// A server that draws units at `Register` (its configuration's `draw_law`) and takes the drawn statement from `hook`.
    pub fn with_draw(mut self, hook: DrawHook) -> Self {
        self.on_draw = Some(hook);
        self
    }

    pub fn try_new(cfg: SessionConfig, verifier: Box<dyn StatementVerifier>) -> Result<Self, String> {
        cfg.check()?;
        Ok(Self {
            cfg,
            verifier,
            hello: None,
            link: None,
            commit: None,
            points: Vec::new(),
            y: Vec::new(),
            streams: BTreeMap::new(),
            proofs: BTreeMap::new(),
            prime: Vec::new(),
            prime_before_commit: None,
            aborted: None,
            verdict: None,
            urandom: Some(File::open("/dev/urandom").expect("/dev/urandom")),
            replaying: false,
            seed: None,
            last_g: None,
            points_g: None,
            round_g: BTreeMap::new(),
            round_msg: BTreeMap::new(),
            on_draw: None,
            seed_inject: None,
            coin_tree: None,
            coin_nonce: None,
            coin_tape: None,
            coin_edit: None,
            t0: std::time::Instant::now(),
            handle_s: 0.0,
        })
    }

    /// A test harness's step-0 seed (deterministic verifier coins for transcript comparisons).
    #[cfg(feature = "seed-injection")]
    pub fn inject_coin_seed(&mut self, seed: [u8; 32], nonce: [u8; 32]) {
        self.seed_inject = Some((seed, nonce));
    }

    /// A test harness's random tape for the committed coin schedule (a verifier whose coins are a function of its tape).
    #[cfg(feature = "seed-injection")]
    pub fn inject_coin_tape(&mut self, tape: [u8; 32]) {
        self.coin_tape = Some(tape);
    }

    /// A test harness's malicious verifier: it commits to its tape's coins with `edit` applied to every block.
    #[cfg(feature = "seed-injection")]
    pub fn inject_coin_edit(&mut self, edit: CoinEdit) {
        self.coin_edit = Some(edit);
    }

    /// The committed coin schedule's root (after `Hello`).
    pub fn coin_root(&self) -> Option<[u8; 64]> {
        self.coin_tree.as_ref().map(|t| t.root())
    }

    /// M2: `n` coins of `stream`'s round `round` from the committed schedule, and the answer's words (coins, then opening).
    fn committed_coins(&mut self, stream: &str, round: usize, n: usize) -> Result<(Vec<F128>, Vec<F128>), String> {
        let t = self.coin_tree.as_ref().ok_or("no committed coin schedule")?;
        let si = t.spec.stream_index(stream).ok_or(format!("{stream} is outside the committed coin schedule"))?;
        t.open(si, round, n)
    }

    /// The live prime rounds, and how many preceded `Commit`.
    pub fn prime_rounds(&self) -> (&[PrimeRound], Option<usize>) {
        (&self.prime, self.prime_before_commit)
    }

    /// A server rebuilt from a record ([`Server::from_record`]) has no coin source: the points it draws while replaying
    /// `Commit` are zeros, replaced at once by the recorded ones, and it never answers a prover.
    fn urandom(&mut self, b: &mut [u8]) {
        match self.urandom.as_mut() {
            Some(f) => f.read_exact(b).expect("urandom"),
            None => b.fill(0),
        }
    }

    /// Words uniform below p (rejection sampling on OS u32s).
    fn prime_words(&mut self, n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        let mut b = [0u8; 4 * 64];
        while out.len() < n {
            self.urandom(&mut b);
            for c in b.chunks(4) {
                let v = u32::from_le_bytes(c.try_into().unwrap()) & 0x7fff_ffff;
                if v < BABYBEAR_P && out.len() < n {
                    out.push(v);
                }
            }
        }
        out
    }

    /// The proofs received, by (table, rep): kept beside the session record (E1).
    pub fn proofs(&self) -> &BTreeMap<(String, usize), Vec<u8>> {
        &self.proofs
    }

    fn refuse(&mut self, why: String) -> Resp {
        if self.aborted.is_none() {
            self.aborted = Some(why.clone());
        }
        Resp::Refused(why)
    }

    fn coins(&mut self, n: usize) -> Vec<F128> {
        let mut b = vec![0u8; 16 * n];
        match self.seed.as_mut() {
            Some((_, _, _, Some(key), k)) => {
                b = coin_seed::coins(key, *k, n);
                self.last_g = Some(*k);
                *k += 1;
            }
            _ => {
                self.last_g = None;
                self.urandom(&mut b);
            }
        }
        b.chunks(16)
            .map(|c| F128 {
                lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
            })
            .collect()
    }

    /// The root a rep's first round commits: domain, then the table's statement binding.
    fn parse_binding(&self, table: &TableSpec, content: &[u8]) -> Result<Vec<u8>, String> {
        let mut at = 0;
        parse_frame(content, &mut at, OP_DOMAIN)?;
        let (label, digest) = match &table.binding {
            Binding::Mixed { registry_digest, .. } | Binding::Circuit { registry_digest, .. } => (LABEL_MIXED, registry_digest),
            Binding::R1cs { statement_digest } => (LABEL_R1CS, statement_digest),
        };
        if parse_frame(content, &mut at, OP_LABEL)? != label {
            return Err("binding: wrong statement label".into());
        }
        if parse_frame(content, &mut at, OP_BYTES)? != digest {
            return Err("binding: statement digest differs from the configured statement".into());
        }
        if let Binding::Mixed { counts_le, .. } | Binding::Circuit { counts_le, .. } = &table.binding
            && parse_frame(content, &mut at, OP_BYTES)? != counts_le.as_slice()
        {
            return Err("binding: table counts differ from the configured statement".into());
        }
        let cap = parse_frame(content, &mut at, OP_BYTES)?.to_vec();
        if let Binding::Circuit { circuit_digest, .. } = &table.binding {
            if parse_frame(content, &mut at, OP_LABEL)? != LABEL_CIRCUIT {
                return Err("binding: no circuit statement after the root".into());
            }
            if parse_frame(content, &mut at, OP_BYTES)? != circuit_digest {
                return Err("F2: circuit digest differs from the configured statement".into());
            }
            let want = self.commit.as_ref().and_then(|c| c.publics_digest.get(&table.name).cloned().flatten());
            if Some(parse_frame(content, &mut at, OP_BYTES)?) != want.as_ref().map(|d| d.as_slice()) {
                return Err("C4: public words bound in the proof differ from the committed, checked ones".into());
            }
        }
        Ok(cap)
    }

    fn root_of(&self, stream: &str) -> Option<(TableSpec, usize)> {
        let (t, r) = stream.rsplit_once("/rep")?;
        let rep: usize = r.parse().ok()?;
        self.cfg.tables.iter().find(|x| x.name == t).map(|x| (x.clone(), rep))
    }

    pub fn handle(&mut self, req: Req) -> Resp {
        if matches!(req, Req::Ping(_)) {
            return Resp::Ok;
        }
        let finish = matches!(req, Req::Finish);
        let t = std::time::Instant::now();
        let r = self.handle_inner(req);
        if !finish {
            self.handle_s += t.elapsed().as_secs_f64();
        }
        r
    }

    fn handle_inner(&mut self, req: Req) -> Resp {
        if let Some(why) = &self.aborted {
            return Resp::Refused(format!("session aborted: {why}"));
        }
        if self.verdict.is_some() {
            return Resp::Refused("session finished: the verdict is issued".into());
        }
        match req {
            Req::Register => {
                if self.hello.is_some() || self.cfg.unit_draw.is_some() {
                    return self.refuse("U: the units are drawn once, after registration and before Hello".into());
                }
                let drawn = match (&self.cfg.draw_given, &self.cfg.draw_law) {
                    (Some(d), _) => unit_draw::check(d).map(|_| d.clone()),
                    (None, Some((law, population))) => unit_draw::draw(law, *population),
                    (None, None) => return Resp::Draw("null".into()),
                };
                let draw = match drawn {
                    Ok(d) => d,
                    Err(e) => return self.refuse(format!("U: {e}")),
                };
                let Some(hook) = self.on_draw.as_mut() else { return self.refuse("U: a draw law without the drawn statement".into()) };
                match hook(&draw) {
                    Ok((cfg, verifier)) if cfg.unit_draw.as_ref() == Some(&draw) => {
                        self.cfg = cfg;
                        self.verifier = verifier;
                        Resp::Draw(draw.to_string())
                    }
                    Ok(_) => self.refuse("U: the drawn statement does not carry the draw".into()),
                    Err(e) => self.refuse(format!("U: the drawn statement: {e}")),
                }
            }
            Req::Hello(h) => {
                // one Hello per session: the coin tree's session key is drawn for it alone, never reused
                if self.hello.is_some() {
                    return self.refuse("R7: a second Hello in one session".into());
                }
                // the keyed coin tree: the prover's nonce is the one field the verifier's own Hello lacks
                let nonce = match self.cfg.coin_tree.is_some().then(|| coin_tree::hello_nonce(&h)) {
                    Some(Err(e)) => return self.refuse(format!("R7: {e}")),
                    Some(Ok(n)) => Some(n),
                    None => None,
                };
                let want = match &nonce {
                    Some(n) => self.cfg.hello_with_nonce(n),
                    None => self.cfg.hello(),
                };
                if h != want {
                    return self.refuse(format!("R7: session parameters {h} differ from the verifier's {want}"));
                }
                // a live server draws the session's key and coins from the OS: without a source it refuses, rather than answer
                // Ok and leave the prover to fail on a missing commitment
                if self.cfg.coin_tree.is_some() && self.urandom.is_none() && !self.replaying {
                    return self.refuse("coin tree: no OS random source to draw the session's key and coins".into());
                }
                if let (Some(spec), Some(nonce)) = (self.cfg.coin_tree.clone(), nonce)
                    && self.urandom.is_some()
                {
                    let edit: &dyn Fn(&str, usize, &mut [F128]) = match &self.coin_edit { Some(e) => e.as_ref(), None => &|_, _, _| {} };
                    let tree = match self.coin_tape {
                        Some(t) => {
                            use rand_chacha::{ChaCha20Rng, rand_core::{RngCore, SeedableRng}};
                            let mut r = ChaCha20Rng::from_seed(t);
                            coin_tree::CoinTree::draw_with(spec, &nonce, &mut |b: &mut [u8]| r.fill_bytes(b), edit)
                        }
                        None => {
                            let mut f = File::open("/dev/urandom").expect("/dev/urandom");
                            coin_tree::CoinTree::draw_with(spec, &nonce, &mut |b: &mut [u8]| f.read_exact(b).expect("urandom"), edit)
                        }
                    };
                    // the session key leaves the verifier here, with the root, and not before
                    let w = tree.answer_words();
                    self.coin_tree = Some(tree);
                    self.coin_nonce = Some(nonce);
                    self.hello = Some(h);
                    return Resp::Coins(w);
                }
                if self.cfg.coin_seed && self.urandom.is_some() {
                    let (mut seed, mut nonce) = ([0u8; 32], [0u8; 32]);
                    match self.seed_inject.take() {
                        Some((s, n)) => (seed, nonce) = (s, n),
                        None => {
                            self.urandom(&mut seed);
                            self.urandom(&mut nonce);
                        }
                    }
                    let c = coin_seed::commitment(&seed, &nonce);
                    self.seed = Some((seed, nonce, c, Some(coin_seed::coin_key(&seed, &h)), 0));
                    self.hello = Some(h);
                    let w = |i: usize| u64::from_le_bytes(c[8 * i..8 * i + 8].try_into().unwrap());
                    return Resp::Coins((0..4).map(|i| F128 { lo: w(2 * i), hi: w(2 * i + 1) }).collect());
                }
                self.hello = Some(h);
                Resp::Ok
            }
            Req::Link(b) => {
                if self.link.is_some() {
                    return self.refuse("link context sent twice".into());
                }
                if let Some(spec) = self.cfg.link.clone() {
                    if self.commit.is_none() {
                        return self.refuse("L1: y sent before root_F, root_B and the link points".into());
                    }
                    let want = 16 * spec.points * self.cfg.tables.len();
                    if b.len() != want {
                        return self.refuse(format!("L1: y is {} bytes, want {want}", b.len()));
                    }
                    self.y = b.chunks(16).map(|c| F128 {
                        lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                        hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
                    }).collect();
                }
                self.link = Some(b);
                Resp::Ok
            }
            Req::Commit { root_f, roots, publics } => {
                let Some(spec) = self.cfg.link.clone() else {
                    return self.refuse("L1: this session has no link exchange".into());
                };
                if self.hello.is_none() {
                    return self.refuse("R7: roots committed before the session parameters".into());
                }
                if self.commit.is_some() {
                    return self.refuse("L1: roots committed twice (the link points are drawn once)".into());
                }
                if root_f.is_empty() {
                    return self.refuse("L1/C2: no root_F: the points must follow both sides' roots".into());
                }
                if self.cfg.prime_live && self.prime.is_empty() {
                    return self.refuse("P-live: root_F committed before any live prime coin (the prime proof's coins before root_F are the verifier's)".into());
                }
                // A statement with no link points (flock-pure) commits its Σ in root_F's place.
                if spec.points == 0 && root_f != spec.sigma {
                    return self.refuse("statement: Commit names another statement than the verifier's Σ".into());
                }
                let roots: BTreeMap<String, Vec<u8>> = roots.into_iter().collect();
                let publics: BTreeMap<String, Vec<u8>> = publics.into_iter().collect();
                let names: Vec<String> = self.cfg.tables.iter().map(|t| t.name.clone()).collect();
                if roots.keys().cloned().collect::<Vec<_>>() != names || roots.values().any(|r| r.is_empty()) {
                    return self.refuse(format!("L1/L3: root_B must be committed for exactly the tables {names:?}, together"));
                }
                if publics.keys().any(|k| !names.contains(k)) {
                    return self.refuse("public words for a table outside the statement".into());
                }
                let mut pd = BTreeMap::new();
                for n in &names {
                    match self.verifier.statement_publics(n, publics.get(n).map(|v| v.as_slice()).unwrap_or(&[])) {
                        Ok(d) => { pd.insert(n.clone(), d); }
                        Err(e) => return self.refuse(e),
                    }
                }
                self.commit = Some(CommitRec { root_f, roots, publics, publics_digest: pd });
                self.prime_before_commit = Some(self.prime.len());
                if self.coin_tree.is_some() {
                    let (c, w) = match self.committed_coins(coin_tree::SESSION, 0, spec.points * spec.m) {
                        Ok(x) => x,
                        Err(e) => return self.refuse(e),
                    };
                    self.points = c.chunks(spec.m).map(|p| p.to_vec()).collect();
                    return Resp::Coins(w);
                }
                let pts: Vec<Vec<F128>> = self.coins(spec.points * spec.m).chunks(spec.m).map(|p| p.to_vec()).collect();
                self.points_g = self.last_g;
                self.points = pts;
                Resp::Coins(self.points.concat())
            }
            Req::Open { stream, parent, label, at } => {
                if self.hello.is_none() {
                    return self.refuse("R7: stream opened before the session parameters".into());
                }
                if self.streams.contains_key(&stream) {
                    return self.refuse(format!("stream {stream} opened twice"));
                }
                if parent.is_empty() {
                    let Some((table, rep)) = self.root_of(&stream) else {
                        return self.refuse(format!("unknown root stream {stream}"));
                    };
                    if rep >= self.cfg.reps {
                        return self.refuse(format!("R7: rep {rep} beyond reps = {}", self.cfg.reps));
                    }
                    if label != self.verifier.domain(&table.name, rep) {
                        return self.refuse(format!("stream {stream}: wrong transcript domain"));
                    }
                    if rep > 0 && self.streams.get(&rep_stream(&table.name, rep - 1)).and_then(|s| s.cap.as_ref()).is_none() {
                        return self.refuse(format!("R1: {stream} opened before rep {} committed its root", rep - 1));
                    }
                } else {
                    if self.cfg.coin_tree.is_some() {
                        return self.refuse(format!("fork {stream}: a committed coin schedule has no forked streams"));
                    }
                    let Some(p) = self.streams.get(&parent) else {
                        return self.refuse(format!("fork of unknown stream {parent}"));
                    };
                    // F3: a fork follows at least one issued round of its parent, and its rep root's binding round.
                    let root = parent.split('/').take(2).collect::<Vec<_>>().join("/");
                    if at == 0 || self.streams.get(&root).and_then(|r| r.cap.as_ref()).is_none() {
                        return self.refuse(format!("F3: fork {stream} opened at parent position {at}, before {root}'s binding round"));
                    }
                    if p.rounds.len() as u64 != at {
                        return self.refuse(format!("R3: fork {stream} claims parent position {at}, parent has issued {}", p.rounds.len()));
                    }
                    let idx = self.streams.keys().filter(|k| k.starts_with(&format!("{parent}/f"))
                        && !k[parent.len() + 2..].contains('/')).count();
                    if stream != child_stream(&parent, idx as u32) {
                        return self.refuse(format!("R3: fork stream {stream} is not {parent}'s next child"));
                    }
                }
                self.streams.insert(stream, StreamRec { parent, label, at, ..Default::default() });
                Resp::Ok
            }
            Req::Round { stream, content, n } => {
                if (self.cfg.require_link || self.cfg.link.is_some()) && self.link.is_none() {
                    return self.refuse("R5: coin requested before root_F, the link points and y were received".into());
                }
                let Some(rec) = self.streams.get(&stream) else {
                    return self.refuse(format!("round on unopened stream {stream}"));
                };
                if n == 0 || n > 1 << 16 {
                    return self.refuse(format!("bad coin count {n}"));
                }
                if rec.parent.is_empty() && rec.rounds.is_empty() {
                    let (table, rep) = self.root_of(&stream).expect("root checked at open");
                    let cap = match self.parse_binding(&table, &content) {
                        Ok(c) => c,
                        Err(e) => return self.refuse(format!("{stream}: {e}")),
                    };
                    if rep > 0 {
                        let r0 = self.streams[&rep_stream(&table.name, 0)].cap.clone();
                        if r0.as_ref() != Some(&cap) {
                            return self.refuse(format!(
                                "R1: {stream} committed root {} but rep 0 committed {}; no rep-{rep} coin is issued",
                                hex(&sha256(&cap)[..8]),
                                r0.map(|c| hex(&sha256(&c)[..8])).unwrap_or_default()
                            ));
                        }
                    }
                    if self.cfg.link.is_some() {
                        let rb = self.commit.as_ref().and_then(|c| c.roots.get(&table.name));
                        if rb != Some(&cap) {
                            return self.refuse(format!("L1: {stream} binds a root other than the committed root_B"));
                        }
                    }
                    self.streams.get_mut(&stream).unwrap().cap = Some(cap);
                }
                let at = self.streams[&stream].rounds.len();
                let (coins, answer) = if self.coin_tree.is_some() {
                    match self.committed_coins(&stream, at, n as usize) {
                        Ok(x) => x,
                        Err(e) => return self.refuse(e),
                    }
                } else {
                    let c = self.coins(n as usize);
                    (c.clone(), c)
                };
                if let Some(g) = self.last_g {
                    self.round_g.insert((stream.clone(), at), g);
                }
                self.streams.get_mut(&stream).unwrap().rounds.push(RoundRec { sha: self.cfg.dg(&content), len: content.len(), n, coins: coins.clone() });
                if self.cfg.retain_rounds {
                    self.round_msg.insert((stream, at), content);
                }
                Resp::Coins(answer)
            }
            Req::Proof { table, rep, bytes } => {
                self.proofs.insert((table, rep as usize), bytes);
                Resp::Ok
            }
            Req::Ping(_) => Resp::Ok,
            Req::Finish => {
                let v = self.finish();
                Resp::Verdict(v.to_string())
            }
            Req::Prime { state, n } => {
                if !self.cfg.prime_live {
                    return self.refuse("P-live: this session serves no prime coins".into());
                }
                if self.hello.is_none() {
                    return self.refuse("P-live/R7: prime coin requested before the session parameters".into());
                }
                if self.commit.is_some() && self.link.is_none() {
                    return self.refuse("P-live/R5: prime coin requested after root_F and before y".into());
                }
                let Ok(state) = <[u8; 32]>::try_from(state.as_slice()) else {
                    return self.refuse("P-live: a prime round commits a 32-byte transcript state".into());
                };
                if n == 0 || n > 1 << 12 {
                    return self.refuse(format!("P-live: bad prime coin count {n}"));
                }
                let words = self.prime_words(PRIME_DEG * n as usize);
                self.prime.push(PrimeRound { state, n, words: words.clone() });
                Resp::Words(words)
            }
        }
    }

    /// Verify every table's every rep against the recorded session (R2 replay, R7 pins, R8 verdict).
    pub fn finish(&mut self) -> Value {
        let mut reps_out = Vec::new();
        let mut ok = self.aborted.is_none();
        let mut why = self.aborted.clone().unwrap_or_default();
        let t = std::time::Instant::now();
        let record = Arc::new(self.streams.clone());
        let msgs = Arc::new(self.round_msg.clone());
        let record_clone_s = t.elapsed().as_secs_f64();
        let n_pts = self.cfg.link.as_ref().map(|l| l.points).unwrap_or(0);
        for (ti, table) in self.cfg.tables.clone().into_iter().enumerate() {
            let publics = self.commit.as_ref().and_then(|c| c.publics.get(&table.name).cloned()).unwrap_or_default();
            let y: Vec<F128> = self.y.get(ti * n_pts..(ti + 1) * n_pts).map(|s| s.to_vec()).unwrap_or_default();
            for rep in 0..self.cfg.reps {
                let root = rep_stream(&table.name, rep);
                // wall-clock seconds by phase, for the verdict only
                let mut timing = json!({});
                let res: Result<(), String> = (|| {
                    let bytes = self.proofs.get(&(table.name.clone(), rep)).ok_or(format!("R7: no proof for {root} (lone rep)"))?;
                    let rec = record.get(&root).ok_or(format!("no live session stream for {root}"))?;
                    let mut ch = ReplayChallenger::with_messages(record.clone(), msgs.clone(), &root, &self.verifier.domain(&table.name, rep));
                    // A replay failure is the primary reason: the verifier's own error on a proof whose messages
                    // were not the committed ones is a consequence of it.
                    if self.cfg.link.is_some() && (self.points.len() != n_pts || y.len() != n_pts) {
                        return Err(format!("{root}: no complete link context in the session"));
                    }
                    let ctx = LinkCtx { points: &self.points, y: &y, publics: &publics };
                    let tv = std::time::Instant::now();
                    let cap = self.verifier.verify(&table.name, rep, bytes, &mut ch, &ctx);
                    timing["verifier_s"] = json!(tv.elapsed().as_secs_f64());
                    timing["replay_squeeze_s"] = json!(ch.squeeze_s());
                    let cap = cap.map_err(|e| match ch.first_failure() {
                        Some(r) => format!("{r} (then the verifier: {e})"),
                        None => e,
                    })?;
                    let tf = std::time::Instant::now();
                    let fin = ch.finish();
                    timing["replay_finish_s"] = json!(tf.elapsed().as_secs_f64());
                    fin?;
                    if rec.cap.as_ref() != Some(&cap) {
                        return Err(format!("{root}: proof root differs from the root committed in the session"));
                    }
                    Ok(())
                })();
                if let Some(p) = self.verifier.take_phases() {
                    timing["phases"] = p;
                }
                if let Err(e) = &res {
                    if ok {
                        why = e.clone();
                    }
                    ok = false;
                }
                reps_out.push(json!({"stream": root, "ok": res.is_ok(), "why": res.err().unwrap_or_default(), "timing": timing}));
            }
        }
        if self.cfg.prime_live {
            // the prime proof itself is verified against these rounds by the prime verifier; here only their shape
            let before = self.prime_before_commit.unwrap_or(0);
            let bad = if before == 0 {
                Some("P-live: no live prime round before root_F".to_string())
            } else if self.prime.len() <= before {
                Some("P-live: no live prime round after y (rho and the Ligero queries)".to_string())
            } else {
                self.prime.iter().position(|r| r.words.len() != PRIME_DEG * r.n as usize || r.words.iter().any(|&w| w >= BABYBEAR_P))
                    .map(|k| format!("P-live: prime round {k} is malformed"))
            };
            if let Some(e) = bad {
                if ok {
                    why = e;
                }
                ok = false;
            }
        }
        // the Lean verifier's verdict on a statement it refuses is the refusal, whatever the session; the verdict keeps the one
        // the session would otherwise get, for the selftest's checks of everything else
        let mut without_refusal = None;
        if let Some(e) = self.verifier.refusal() {
            without_refusal = Some(json!({"accepted": ok, "why": why}));
            ok = false;
            why = e;
        }
        let rounds: usize = self.streams.values().map(|s| s.rounds.len()).sum();
        let empty: usize = self.streams.values().map(|s| s.rounds.iter().filter(|r| r.len <= 16).count()).sum();
        let v = json!({
            "accepted": ok, "why": why, "reps": reps_out, "streams": self.streams.len(), "rounds": rounds,
            "rounds_without_prover_message": empty, "prime_rounds": self.prime.len(), "prime_before_commit": self.prime_before_commit,
            "verify_s": t.elapsed().as_secs_f64(), "session_s": self.t0.elapsed().as_secs_f64(), "handle_s": self.handle_s,
            "record_clone_s": record_clone_s,
        });
        let mut v = v;
        if let Some((seed, nonce, c, _, k)) = &self.seed {
            v["coin_seed"] = json!({"scheme": coin_seed::SCHEME, "domain": coin_seed::DOMAIN, "commitment": hex(c), "seed": hex(seed),
                                    "nonce": hex(nonce), "coin_requests": k});
        }
        if let Some(w) = without_refusal {
            v["without_refusal"] = w;
        }
        self.verdict = Some(v.clone());
        v
    }

    /// The session record (R8): what the verifier saw and issued, and its verdict.
    pub fn record_json(&self) -> Value {
        let (dk, dg) = (self.cfg.dk(), |b: &[u8]| hex(&self.cfg.dg(b)));
        let streams: Vec<Value> = self
            .streams
            .iter()
            .map(|(name, s)| {
                let mut sv = json!({
                    "stream": name, "parent": s.parent, "label": String::from_utf8_lossy(&s.label), "at": s.at,
                    "rounds": s.rounds.iter().enumerate().map(|(k, r)| {
                        let mut v = json!({
                            "k": k, "g": self.round_g.get(&(name.clone(), k)), "n": r.n, "msg_len": r.len,
                            "coins": r.coins.iter().map(f128_hex).collect::<Vec<_>>(),
                        });
                        v[format!("msg_{dk}")] = json!(hex(&r.sha));
                        if let Some(m) = self.round_msg.get(&(name.clone(), k)) {
                            v["msg"] = json!(hex(m));
                        }
                        v
                    }).collect::<Vec<_>>(),
                });
                sv[format!("root_{dk}")] = json!(s.cap.as_ref().map(|c| dg(c)));
                sv
            })
            .collect();
        let link = self.commit.as_ref().map(|c| { let mut v = json!({
            "mode": "exchange",
            "sigma": self.cfg.link.as_ref().map(|l| hex(&l.sigma)),
            "points": self.points.iter().map(|p| p.iter().map(f128_hex).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "points_g": self.points_g,
            "y": self.y.iter().map(f128_hex).collect::<Vec<_>>(),
            // G2: the committed words themselves, so the Flock side replays from the record alone
            "root_f": hex(&c.root_f),
            "root_b": c.roots.iter().map(|(t, r)| (t.clone(), json!(hex(r)))).collect::<serde_json::Map<_, _>>(),
            "publics": c.publics.iter().map(|(t, p)| (t.clone(), json!(hex(p)))).collect::<serde_json::Map<_, _>>(),
        });
            v[format!("root_f_{dk}")] = json!(dg(&c.root_f));
            v[format!("root_b_{dk}")] = json!(c.roots.iter().map(|(t, r)| (t.clone(), json!(dg(r)))).collect::<serde_json::Map<_, _>>());
            v[format!("publics_{dk}")] = json!(c.publics.iter().map(|(t, p)| (t.clone(), json!(dg(p)))).collect::<serde_json::Map<_, _>>());
            v
        });
        let prime = self.cfg.prime_live.then(|| json!({
            "mode": PRIME_MODE, "before_commit": self.prime_before_commit,
            "rounds": self.prime.iter().map(|r| json!({"state": hex(&r.state), "n": r.n, "words": r.words})).collect::<Vec<_>>(),
        }));
        // link_sha256 covers the whole exchange in exchange mode: root_F, every root_B, the points and y.
        let link_sha = match (&self.commit, &self.link) {
            (Some(c), Some(l)) => {
                let mut b = c.root_f.clone();
                for r in c.roots.values() { b.extend_from_slice(r); }
                for p in self.points.iter().flatten() { push_f128(&mut b, *p); }
                b.extend_from_slice(l);
                Some(dg(&b))
            }
            (None, Some(l)) if self.cfg.link.is_none() => Some(dg(l)),
            _ => None,
        };
        let mut rec = json!({
            "kind": "flock-live-session/v1",
            "hello": self.hello, "link": link,
            "link_mode": if self.cfg.link.is_some() { "exchange" } else { "stub" },
            "config": {"profile": self.cfg.profile, "reps": self.cfg.reps, "flavor": self.cfg.flavor,
                       "require_link": self.cfg.require_link, "prime_live": self.cfg.prime_live,
                       "tables": self.cfg.tables.iter().map(|t| json!({"name": t.name, "binding": format!("{:?}", t.binding)})).collect::<Vec<_>>()},
            "unit_draw": self.cfg.unit_draw,
            "aborted": self.aborted, "verdict": self.verdict, "streams": streams, "prime": prime,
            "coin_tree": self.coin_tree.as_ref().map(|t| json!({"spec": t.spec.to_json(), "nonce": self.coin_nonce.map(|n| hex(&n)),
                                                                 "key": hex(t.key()), "root": hex(&t.root())})),
            "coin_seed": self.seed.as_ref().map(|(seed, nonce, c, _, k)| {
                let mut v = json!({"scheme": coin_seed::SCHEME, "domain": coin_seed::DOMAIN, "context": {"hello": self.hello}, "commitment": hex(c),
                                   "requests": k});
                // the opening only once the verdict is issued: until then the seed is the verifier's secret
                if self.verdict.is_some() {
                    v["seed"] = json!(hex(seed));
                    v["nonce"] = json!(hex(nonce));
                }
                v
            }),
        });
        rec[format!("link_{dk}")] = json!(link_sha);
        rec[format!("proof_{dk}")] = json!(self.proofs.iter().map(|((t, r), b)| {
            let mut v = json!({"stream": rep_stream(t, *r), "bytes": b.len()});
            v[dk] = json!(dg(b));
            v
        }).collect::<Vec<_>>());
        rec
    }

    /// G2: the finished session `rec` (a [`Server::record_json`]) rebuilt under the verifier's own configuration and
    /// statement, with `proofs` (by table and rep) as received; [`Server::finish`] then replays it. Everything the live
    /// server checked when a message arrived is checked again here from the record: the hello, the committed public
    /// words (C4), each rep's bound root against root_B and rep 0, the link context's shape, the prime rounds' order,
    /// the proofs' hashes and `link_sha256`. A record is evidence only under the verifier's custody: nothing here
    /// authenticates who wrote it.
    pub fn from_record(cfg: SessionConfig, verifier: Box<dyn StatementVerifier>, rec: &Value,
                       proofs: BTreeMap<(String, usize), Vec<u8>>) -> Result<Server, String> {
        let mut s = Server::try_new(cfg, verifier)?;
        s.urandom = None;
        s.replaying = true;
        let bad = |w: &str| format!("record: {w}");
        let hx = |v: &Value, w: &str| -> Result<Vec<u8>, String> {
            let t = v.as_str().ok_or_else(|| bad(&format!("{w} is not a hex string")))?;
            if t.len() % 2 != 0 {
                return Err(bad(&format!("{w}: odd hex length")));
            }
            (0..t.len() / 2).map(|i| u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(|_| bad(&format!("{w}: not hex")))).collect()
        };
        let f128s = |v: &Value, w: &str| -> Result<Vec<F128>, String> {
            v.as_array().ok_or_else(|| bad(&format!("{w} is not a list")))?.iter().map(|x| {
                let b = hx(x, w)?;
                if b.len() != 16 {
                    return Err(bad(&format!("{w}: a GF(2^128) element is 16 bytes")));
                }
                Ok(F128 { hi: u64::from_be_bytes(b[..8].try_into().unwrap()), lo: u64::from_be_bytes(b[8..].try_into().unwrap()) })
            }).collect()
        };
        if rec["kind"] != "flock-live-session/v1" {
            return Err(bad("not a flock-live-session/v1 record"));
        }
        if !rec["aborted"].is_null() {
            return Err(bad(&format!("the live session aborted: {}", rec["aborted"])));
        }
        let hello = rec["hello"].as_str().ok_or_else(|| bad("no hello"))?.to_string();
        if let Resp::Refused(e) = s.handle(Req::Hello(hello)) {
            return Err(bad(&e));
        }
        let prime_rec = &rec["prime"];
        let before = if s.cfg.prime_live {
            if prime_rec["mode"] != PRIME_MODE {
                return Err(bad("the configuration serves live prime coins, the record has none"));
            }
            let b = prime_rec["before_commit"].as_u64().ok_or_else(|| bad("prime.before_commit"))? as usize;
            let rounds = prime_rec["rounds"].as_array().ok_or_else(|| bad("prime.rounds"))?;
            if b > rounds.len() {
                return Err(bad("prime.before_commit exceeds the prime rounds"));
            }
            for r in rounds {
                let state = hx(&r["state"], "prime state")?;
                let n = r["n"].as_u64().ok_or_else(|| bad("prime n"))? as u32;
                let words: Vec<u32> = r["words"].as_array().ok_or_else(|| bad("prime words"))?.iter()
                    .map(|w| w.as_u64().filter(|&x| x < BABYBEAR_P as u64).map(|x| x as u32).ok_or_else(|| bad("a prime word is not below p")))
                    .collect::<Result<_, _>>()?;
                if state.len() != 32 || n == 0 || words.len() != PRIME_DEG * n as usize {
                    return Err(bad("a malformed prime round"));
                }
                s.prime.push(PrimeRound { state: state.try_into().unwrap(), n, words });
            }
            Some(b)
        } else {
            if !prime_rec.is_null() {
                return Err(bad("prime rounds in a session configured without them"));
            }
            None
        };
        // U1: the record keeps a draw exactly when the statement carries one, and they are equal; U2: it is well formed
        if rec["unit_draw"] != s.cfg.unit_draw.clone().unwrap_or(Value::Null) {
            return Err(bad("U1: the record's unit draw is not the statement's"));
        }
        if let Some(d) = &s.cfg.unit_draw {
            unit_draw::check(d).map_err(|e| bad(&e))?;
        }
        let l = &rec["link"];
        if s.cfg.link.is_some() {
            if rec["link_mode"] != "exchange" || l["mode"] != "exchange" {
                return Err(bad("link_mode is not exchange"));
            }
            if l["sigma"].as_str() != s.cfg.link.as_ref().map(|k| hex(&k.sigma)).as_deref() {
                return Err(bad("link.sigma is not the verifier's Σ"));
            }
            let root_f = hx(&l["root_f"], "root_f")?;
            let names: Vec<String> = s.cfg.tables.iter().map(|t| t.name.clone()).collect();
            let mut roots = Vec::new();
            let mut publics = Vec::new();
            let (dk, dg) = (s.cfg.dk(), |b: &[u8]| hex(&s.cfg.dg(b)));
            for n in &names {
                let rb = hx(&l["root_b"][n], "root_b")?;
                if l[format!("root_b_{dk}")][n].as_str() != Some(dg(&rb).as_str()) {
                    return Err(bad(&format!("{n}: root_b differs from root_b_{dk}")));
                }
                roots.push((n.clone(), rb));
                // a statement that commits no public words has no entry for the table (the server records none)
                let pk = &l[format!("publics_{dk}")][n];
                let p = if l["publics"][n].is_null() && pk.is_null() { Vec::new() } else { hx(&l["publics"][n], "publics")? };
                if !(p.is_empty() && pk.is_null()) && pk.as_str() != Some(dg(&p).as_str()) {
                    return Err(bad(&format!("{n}: publics differ from publics_{dk}")));
                }
                publics.push((n.clone(), p));
            }
            if l[format!("root_f_{dk}")].as_str() != Some(dg(&root_f).as_str()) {
                return Err(bad(&format!("root_f differs from root_f_{dk}")));
            }
            // the prime rounds before Commit were issued before it; the rest after y
            let tail = before.map(|b| s.prime.split_off(b)).unwrap_or_default();
            match s.handle(Req::Commit { root_f, roots, publics }) {
                Resp::Coins(_) => {}
                Resp::Refused(e) => return Err(bad(&e)),
                r => return Err(bad(&format!("commit: {r:?}"))),
            }
            s.prime.extend(tail);
            let pts = l["points"].as_array().ok_or_else(|| bad("link.points"))?;
            let spec = s.cfg.link.clone().unwrap();
            s.points = pts.iter().map(|p| f128s(p, "point")).collect::<Result<_, _>>()?;
            if s.points.len() != spec.points || s.points.iter().any(|p| p.len() != spec.m) {
                return Err(bad("the link points are not the configured shape"));
            }
            let y = f128s(&l["y"], "y")?;
            let mut yb = Vec::new();
            for v in &y {
                yb.extend_from_slice(&v.lo.to_le_bytes());
                yb.extend_from_slice(&v.hi.to_le_bytes());
            }
            if let Resp::Refused(e) = s.handle(Req::Link(yb)) {
                return Err(bad(&e));
            }
        }
        let streams = rec["streams"].as_array().ok_or_else(|| bad("no streams"))?;
        for st in streams {
            let name = st["stream"].as_str().ok_or_else(|| bad("stream name"))?.to_string();
            let mut sr = StreamRec {
                parent: st["parent"].as_str().ok_or_else(|| bad("stream parent"))?.to_string(),
                label: st["label"].as_str().ok_or_else(|| bad("stream label"))?.as_bytes().to_vec(),
                at: st["at"].as_u64().ok_or_else(|| bad("stream at"))?,
                ..Default::default()
            };
            for (k, r) in st["rounds"].as_array().ok_or_else(|| bad("stream rounds"))?.iter().enumerate() {
                if r["k"].as_u64() != Some(k as u64) {
                    return Err(bad(&format!("{name}: rounds out of order")));
                }
                let dk = s.cfg.dk();
                let sha = hx(&r[format!("msg_{dk}")], "the round digest")?;
                let coins = f128s(&r["coins"], "coins")?;
                let n = r["n"].as_u64().ok_or_else(|| bad("round n"))? as u32;
                if sha.len() != s.cfg.dg(b"").len() || coins.len() != n as usize {
                    return Err(bad(&format!("{name} round {k}: malformed")));
                }
                if let Some(g) = r["g"].as_u64() {
                    s.round_g.insert((name.to_string(), sr.rounds.len()), g);
                }
                if s.cfg.retain_rounds {
                    let m = hx(&r["msg"], "a retained round's bytes")?;
                    if s.cfg.dg(&m)[..] != sha[..] || Some(m.len() as u64) != r["msg_len"].as_u64() {
                        return Err(bad(&format!("{name} round {k}: the retained bytes are not the round's msg_{dk} / msg_len")));
                    }
                    s.round_msg.insert((name.to_string(), sr.rounds.len()), m);
                }
                sr.rounds.push(RoundRec { sha, len: r["msg_len"].as_u64().unwrap_or(0) as usize, n, coins });
            }
            if sr.parent.is_empty() {
                let Some((table, rep)) = s.root_of(&name) else { return Err(bad(&format!("unknown root stream {name}"))) };
                if rep >= s.cfg.reps || sr.label != s.verifier.domain(&table.name, rep) {
                    return Err(bad(&format!("{name}: beyond the configured reps, or the wrong domain")));
                }
                // every rep binds the one committed root_B (the live server refused any other before its first coin)
                let rb = s.commit.as_ref().and_then(|c| c.roots.get(&table.name)).cloned().ok_or_else(|| bad("no root_B"))?;
                if st[format!("root_{}", s.cfg.dk())].as_str() != Some(hex(&s.cfg.dg(&rb)).as_str()) {
                    return Err(bad(&format!("R1/L1: {name} bound a root other than the committed root_B")));
                }
                sr.cap = Some(rb);
            }
            s.streams.insert(name, sr);
        }
        let dk = s.cfg.dk();
        for p in rec[format!("proof_{dk}")].as_array().ok_or_else(|| bad("the proofs' digests"))? {
            let (t, r) = p["stream"].as_str().and_then(|x| x.rsplit_once("/rep")).ok_or_else(|| bad("proof stream"))?;
            let rep: usize = r.parse().map_err(|_| bad("proof rep"))?;
            let b = proofs.get(&(t.to_string(), rep)).ok_or_else(|| bad(&format!("no proof file for {t}/rep{rep}")))?;
            if p[dk].as_str() != Some(hex(&s.cfg.dg(b)).as_str()) {
                return Err(bad(&format!("the proof for {t}/rep{rep} is not the one the verifier received")));
            }
        }
        s.proofs = proofs;
        let got = s.record_json();
        if got[format!("link_{dk}")] != rec[format!("link_{dk}")] {
            return Err(bad(&format!("link_{dk} differs from the exchange it covers")));
        }
        Ok(s)
    }

    pub fn aborted(&self) -> Option<&str> {
        self.aborted.as_deref()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Verifier side: replay
// ---------------------------------------------------------------------------------------------------------------

struct ReplayState {
    record: Arc<BTreeMap<String, StreamRec>>,
    /// the rounds' bytes where the coin server retained them (`SessionConfig::retain_rounds`): compared byte for byte
    msgs: Arc<BTreeMap<(String, usize), Vec<u8>>>,
    used: BTreeMap<String, usize>,
    failures: Vec<String>,
    /// time spent checking rounds at squeezes, every stream of the rep (the verdict's `replay_squeeze_s`)
    squeeze_ns: u64,
}

/// The verifier's challenger over a recorded session. See the crate docs.
pub struct ReplayChallenger {
    stream: String,
    st: Arc<Mutex<ReplayState>>,
    pending: Vec<u8>,
    forks: AtomicU32,
    rounds: u64,
}

impl ReplayChallenger {
    pub fn new(record: Arc<BTreeMap<String, StreamRec>>, root: &str, domain: &[u8]) -> Self {
        Self::with_messages(record, Arc::new(BTreeMap::new()), root, domain)
    }

    pub fn with_messages(record: Arc<BTreeMap<String, StreamRec>>, msgs: Arc<BTreeMap<(String, usize), Vec<u8>>>, root: &str, domain: &[u8]) -> Self {
        let st = Arc::new(Mutex::new(ReplayState { record, msgs, used: BTreeMap::new(), failures: Vec::new(), squeeze_ns: 0 }));
        Self::stream(st, root.to_string(), frame_domain(domain))
    }

    fn stream(st: Arc<Mutex<ReplayState>>, stream: String, pending: Vec<u8>) -> Self {
        st.lock().unwrap().used.entry(stream.clone()).or_insert(0);
        Self { stream, st, pending, forks: AtomicU32::new(0), rounds: 0 }
    }

    fn fail(&self, why: String) {
        self.st.lock().unwrap().failures.push(why);
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let content = std::mem::take(&mut self.pending);
        let k = self.rounds as usize;
        self.rounds += 1;
        let t = std::time::Instant::now();
        let mut st = self.st.lock().unwrap();
        *st.used.get_mut(&self.stream).unwrap() = k + 1;
        let rec = st.record.get(&self.stream).and_then(|s| s.rounds.get(k)).cloned();
        let coins = match rec {
            None => {
                st.failures.push(format!("R2: {} round {k}: the verifier issued no such coin", self.stream));
                vec![F128::default(); n]
            }
            Some(r) => {
                let kept = st.msgs.get(&(self.stream.clone(), k)).map(|m| *m == content);
                if r.n as usize != n {
                    st.failures.push(format!("R2: {} round {k}: {n} coins requested, {} issued", self.stream, r.n));
                } else if kept == Some(false) {
                    st.failures.push(format!("R2: {} round {k}: the proof's messages differ from the bytes the verifier kept before the coin", self.stream));
                } else if kept.is_none() && !same_digest(&r.sha, &content) {
                    st.failures.push(format!(
                        "R2: {} round {k}: the proof's messages differ from those committed before the coin",
                        self.stream
                    ));
                }
                let mut c = r.coins;
                c.resize(n, F128::default());
                c
            }
        };
        st.squeeze_ns += t.elapsed().as_nanos() as u64;
        coins
    }

    pub fn first_failure(&self) -> Option<String> {
        self.st.lock().unwrap().failures.first().cloned()
    }

    /// Seconds spent checking the rep's rounds at its squeezes (every stream of the rep).
    pub fn squeeze_s(&self) -> f64 {
        self.st.lock().unwrap().squeeze_ns as f64 * 1e-9
    }

    /// After the Flock verifier returns: no replay failure, and every recorded round of this rep's streams consumed.
    pub fn finish(&self) -> Result<(), String> {
        let st = self.st.lock().unwrap();
        if let Some(f) = st.failures.first() {
            return Err(f.clone());
        }
        let prefix = format!("{}/", self.stream);
        for (name, rec) in st.record.iter().filter(|(n, _)| **n == self.stream || n.starts_with(&prefix)) {
            let used = st.used.get(name).copied().unwrap_or(0);
            if used != rec.rounds.len() {
                return Err(format!("R2: {name}: {} rounds recorded, the proof consumed {used}", rec.rounds.len()));
            }
        }
        Ok(())
    }
}

impl Challenger for ReplayChallenger {
    framed_challenger!();

    fn grind_pow(&mut self, _bits: u32) -> u64 {
        unreachable!("the verifier's challenger never grinds")
    }

    /// R4: the nonce is a message like any other (bound by the round digest), and must be `0`: the coin does not depend on
    /// it, so the verifier refuses every other nonce, as the Lean verifier's decoder does (D21).
    fn verify_pow(&mut self, nonce: u64, _bits: u32) -> bool {
        self.observe_bytes(&nonce.to_le_bytes());
        nonce == 0
    }

    fn fork(&mut self, label: &'static [u8]) -> Self {
        let seed = [self.sample_f128(), self.sample_f128()];
        self.fork_from_seed(seed, label)
    }

    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        let idx = self.forks.fetch_add(1, Ordering::SeqCst);
        let name = child_stream(&self.stream, idx);
        {
            let st = self.st.lock().unwrap();
            match st.record.get(&name) {
                None => {
                    drop(st);
                    self.fail(format!("R3: fork {name}: no live child stream in the session"));
                }
                Some(r) if r.parent != self.stream || r.label != label || r.at != self.rounds => {
                    drop(st);
                    self.fail(format!("R3: fork {name}: opened at a different parent position or label"));
                }
                _ => {}
            }
        }
        let mut c = ReplayChallenger::stream(self.st.clone(), name, frame_domain(label));
        c.observe_f128_slice(&seed);
        c
    }
}

/// A challenger that frames exactly as [`LiveChallenger`] and [`ReplayChallenger`] do, answers each squeeze from a known coin
/// schedule, and keeps each round's content: the rounds a live prover would send for a transcript whose coins are known in
/// advance (M2's simulator, after its extraction run). A request whose size differs from the schedule's round is recorded in
/// `mismatch` and answered with zeros.
pub struct RecordingChallenger {
    pending: Vec<u8>,
    schedule: Vec<Vec<F128>>,
    /// every round's content and coin count, in order
    pub rounds: Vec<(Vec<u8>, usize)>,
    pub mismatch: Option<String>,
}

impl RecordingChallenger {
    pub fn new(domain: &[u8], schedule: Vec<Vec<F128>>) -> Self {
        Self { pending: frame_domain(domain), schedule, rounds: Vec::new(), mismatch: None }
    }

    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let k = self.rounds.len();
        self.rounds.push((std::mem::take(&mut self.pending), n));
        match self.schedule.get(k) {
            Some(c) if c.len() == n => c.clone(),
            other => {
                self.mismatch.get_or_insert(format!("round {k}: {n} coins requested, the schedule has {:?}", other.map(|c| c.len())));
                vec![F128::default(); n]
            }
        }
    }
}

impl Challenger for RecordingChallenger {
    framed_challenger!();

    fn grind_pow(&mut self, _bits: u32) -> u64 {
        unreachable!("the recording challenger replays a verifier")
    }

    fn verify_pow(&mut self, nonce: u64, _bits: u32) -> bool {
        self.observe_bytes(&nonce.to_le_bytes());
        true
    }

    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the single-table path does not fork")
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    fn f(x: u64) -> F128 {
        F128::new(x, !x)
    }

    /// Every call shape the provers make, framed by one challenger, is framed identically by another replaying its rounds.
    #[test]
    fn replayed_rounds_frame_what_the_original_framed() {
        let coins: Vec<Vec<F128>> = [1usize, 3, 2, 1, 5].iter().enumerate().map(|(k, &n)| (0..n).map(|i| f(100 * k as u64 + i as u64)).collect()).collect();
        let mut a = RecordingChallenger::new(b"domain", coins.clone());
        a.observe_label(b"label");
        a.observe_bytes(&[7u8; 21]);
        a.observe_f128(f(1));
        a.observe_f128_slice(&[f(2)]);
        assert_eq!(a.sample_f128(), coins[0][0]);
        a.observe_f256(flock_field::F256::new(f(3), f(4)));
        a.observe_f128_slice(&[f(5), f(6), f(7)]);
        a.sample_f128_vec(3);
        a.observe_bytes(&0u64.to_le_bytes());
        a.sample_f256();
        a.observe_f128_slice(&[]);
        a.sample_f128_vec(0);
        a.observe_label(b"");
        a.sample_f128();
        a.observe_f128(f(8));
        a.sample_f128_vec(5);
        assert!(a.mismatch.is_none());
        let mut b = RecordingChallenger::new(b"domain", coins.clone());
        for (k, (content, n)) in a.rounds.iter().enumerate() {
            assert_eq!(replay_round(&mut b, content, *n).unwrap(), coins[k], "round {k}");
        }
        assert_eq!(a.rounds, b.rounds);
    }

    #[test]
    fn a_round_is_refused_unless_it_ends_in_its_squeeze() {
        let mut a = RecordingChallenger::new(b"d", vec![vec![f(1)], vec![f(2), f(3)]]);
        a.observe_f128(f(9));
        a.sample_f128();
        a.sample_f128_vec(2);
        let (r0, r1) = (a.rounds[0].0.clone(), a.rounds[1].0.clone());
        let mut b = RecordingChallenger::new(b"d", vec![vec![f(1)], vec![f(2), f(3)]]);
        assert!(replay_round(&mut b, &r0, 2).is_err(), "the count differs");
        assert!(replay_round(&mut b, &r0[..r0.len() - 16], 1).is_err(), "no squeeze");
        assert!(replay_round(&mut b, &[r1.clone(), r1].concat(), 2).is_err(), "a squeeze mid-round");
        assert!(replay_round(&mut b, &r0[..20], 1).is_err(), "truncated");
    }
}

#[cfg(test)]
mod coin_server_tests {
    use super::*;

    struct NoProofs;

    impl StatementVerifier for NoProofs {
        fn verify(&self, _: &str, _: usize, _: &[u8], _: &mut ReplayChallenger, _: &LinkCtx<'_>) -> Result<Vec<u8>, String> {
            Err("no proofs in this test".into())
        }
        fn domain(&self, _: &str, _: usize) -> Vec<u8> {
            Vec::new()
        }
    }

    fn cfg() -> SessionConfig {
        let mut c = SessionConfig::flock_128_r2(vec![TableSpec { name: "t".into(), binding: Binding::R1cs { statement_digest: [0; 32] } }]);
        c.coin_tree = Some(coin_tree::Spec { streams: vec![coin_tree::SESSION.into(), rep_stream("t", 0), rep_stream("t", 1)], rounds_log: 2, block: 3 });
        c
    }

    /// A coin-tree server answers `Hello` with the root and its session key (20 words), once per session; without an OS
    /// random source (and not replaying a record) it refuses.
    #[test]
    fn a_coin_tree_server_answers_hello_once_and_refuses_without_an_os_source() {
        let hello = cfg().hello_with_nonce(&[7; coin_tree::NONCE_BYTES]);
        let mut s = Server::try_new(cfg(), Box::new(NoProofs)).unwrap();
        assert!(matches!(s.handle(Req::Hello(hello.clone())), Resp::Coins(ref w) if w.len() == 20));
        assert!(matches!(s.handle(Req::Hello(hello.clone())), Resp::Refused(ref e) if e.contains("a second Hello")));
        let mut s = Server::try_new(cfg(), Box::new(NoProofs)).unwrap();
        s.urandom = None;
        assert!(matches!(s.handle(Req::Hello(hello)), Resp::Refused(ref e) if e.contains("no OS random source")));
    }
}

#[cfg(test)]
mod thread_stack_tests {
    /// About 1 KiB a frame: 16 Ki frames need some 16 MiB, past the 2 MiB default and inside `THREAD_STACK`.
    fn deep(n: usize) -> usize {
        let a = std::hint::black_box([n as u8; 1024]);
        if n == 0 { 0 } else { deep(n - 1) + usize::from(a[0] == n as u8) }
    }

    #[test]
    fn a_spawned_thread_takes_a_recursion_past_the_default_stack() {
        assert_eq!(super::spawn(|| deep(16 << 10)).join().unwrap(), 16 << 10);
        assert_eq!(std::thread::scope(|sc| super::spawn_scoped(sc, || deep(16 << 10)).join().unwrap()), 16 << 10);
    }

    /// Flock's own threads: `run_hetero_chunks`' and `run_hetero_chunks_stateful`'s two helper threads beside the calling
    /// pool (here one on `THREAD_STACK`, as `init_threads` builds the global one), and the all-core pool. A chunk on a pool
    /// worker waits until a helper has run one, so a helper always takes the recursion.
    #[test]
    fn flocks_helpers_and_pools_take_a_recursion_past_the_default_stack() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).stack_size(super::THREAD_STACK).build().unwrap();
        let chunk = |helped: &AtomicBool| {
            let d = deep(16 << 10);
            if rayon::current_thread_index().is_none() {
                helped.store(true, Ordering::Release);
            }
            while !helped.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            d
        };
        let helped = AtomicBool::new(false);
        pool.install(|| flock_core::run_hetero_chunks(64, |_| assert_eq!(chunk(&helped), 16 << 10)));
        let helped = AtomicBool::new(false);
        let states = pool.install(|| flock_core::run_hetero_chunks_stateful(64, || 0usize, |s, _| *s += chunk(&helped)));
        assert_eq!(states.iter().sum::<usize>(), 64 * (16 << 10));
        assert_eq!(flock_core::all_core_pool().install(|| deep(16 << 10)), 16 << 10);
    }
}
