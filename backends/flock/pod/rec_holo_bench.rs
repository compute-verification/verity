//! Pricing kernels for a hidden-circuit (holographic) argument on upstream Flock b684b12, copied into
//! crates/flock-prover/examples by 77-rec-holo-bench.sh. One JSON line per measurement on stdout:
//! - `mul`: F128 multiplications per second, one dependent chain per thread (latency) and over 2^24-element vectors (memory)
//! - `gkr`: product_gkr::prove_batched (the wiring permutation argument) at 2^mu cells: prove/verify seconds, proof bytes
//! - `sumcheck3`: a degree-3 sumcheck over 2^mu terms of v*e1*e2 (a sparse matrix evaluation's core), seconds
//! - `lincheck`: upstream's lincheck prover at 2^m rows with blocks of 2^k_log rows (at least 8 blocks: k_log ≤ m − 3),
//!   LC_D random nonzeros per row in each of A_0 and B_0, seconds
//! - `comb`: the lincheck's circuit-dependent sum comb[c] = α·Σ e[i] over A's column c + Σ e[i] over B's, as a parallel
//!   gather over a column-major list of CB_D random rows per column in each of A and B, at 2^k_log columns, seconds
//! env: GKR_MUS="20 22 24" SC_MUS="20 22 24" LC_CASES="20:11 20:17" LC_D=15 CB_KLOGS="21 24" CB_D=15 RUNS=3
use std::{env::var, fs::read_to_string, hint::black_box, time::Instant};

use flock_core::{
    challenger::FsChallenger,
    field::F128,
    lincheck::{QuirkyPoint, SkipPoint, SparseMatrixCircuit, prove},
    product_gkr::{prove_batched, verify_batched},
    r1cs::SparseBinaryMatrix,
    test_rng::Rng,
};
use flock_prover::init_perf_thread_pool;
use rayon::prelude::*;

fn peak_rss_mb() -> u64 {
    read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .map_or(0, |kb: u64| kb / 1024)
}

fn random_rows(k_log: usize, d: usize, seed: u64) -> SparseBinaryMatrix {
    let k = 1usize << k_log;
    let rows = (0..k)
        .into_par_iter()
        .map(|r| {
            let mut g = Rng::new(seed ^ ((r as u64) << 1 | 1));
            let mut v: Vec<usize> = Vec::with_capacity(d + 4);
            while v.len() < d.min(k) {
                v.push(g.below(k));
                v.sort_unstable();
                v.dedup();
            }
            v
        })
        .collect();
    SparseBinaryMatrix::new(k, k, rows)
}

fn mus(key: &str, default: &str) -> Vec<usize> {
    var(key).unwrap_or_else(|_| default.into()).split([' ', ',']).filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn chain(n: usize, seed: u64) -> F128 {
    let x = F128::new(0x8765_4321_0fed_cba9, 0x1357_9bdf_2468_ace0);
    let mut a: [F128; 8] = std::array::from_fn(|j| F128::new(seed ^ (j as u64 + 1), 0x55));
    for _ in 0..n {
        for v in a.iter_mut() {
            *v *= x;
        }
    }
    a.iter().fold(F128::ZERO, |s, &v| s + v)
}

fn sumcheck3(v: &[F128], e1: &[F128], e2: &[F128], rng: &mut Rng) -> Vec<[F128; 4]> {
    let w = F128::new(2, 0);
    let (mut v, mut e1, mut e2) = (v.to_vec(), e1.to_vec(), e2.to_vec());
    let mut rounds = Vec::new();
    while v.len() > 1 {
        let h = v.len() / 2;
        let ev = (0..h)
            .into_par_iter()
            .map(|i| {
                let (a0, a1, b0, b1, c0, c1) = (v[i], v[i + h], e1[i], e1[i + h], e2[i], e2[i + h]);
                let (da, db, dc) = (a0 + a1, b0 + b1, c0 + c1);
                [a0 * b0 * c0, a1 * b1 * c1, (a0 + w * da) * (b0 + w * db) * (c0 + w * dc), da * db * dc]
            })
            .reduce(|| [F128::ZERO; 4], |x, y| [x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]]);
        rounds.push(ev);
        let r = rng.f128();
        let fold = |t: &mut Vec<F128>| {
            let (lo, hi) = t.split_at(h);
            let n: Vec<F128> = lo.par_iter().zip(hi.par_iter()).map(|(&a, &b)| a + r * (a + b)).collect();
            *t = n;
        };
        fold(&mut v);
        fold(&mut e1);
        fold(&mut e2);
    }
    rounds
}

fn main() {
    let threads = init_perf_thread_pool().unwrap_or_else(rayon::current_num_threads);
    let runs: usize = var("RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut rng = Rng::new(0x5EC_4010);

    let n = 1usize << 24;
    let t = Instant::now();
    black_box(chain(n, 1));
    let one = (8 * n) as f64 / t.elapsed().as_secs_f64();
    let t = Instant::now();
    black_box((0..threads).into_par_iter().map(|j| chain(n, j as u64)).reduce(|| F128::ZERO, |a, b| a + b));
    let all = (8 * n * threads) as f64 / t.elapsed().as_secs_f64();
    let (a, b) = (rng.f128_vec(n), rng.f128_vec(n));
    let t = Instant::now();
    black_box(a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x * y).reduce(|| F128::ZERO, |s, p| s + p));
    let vec = n as f64 / t.elapsed().as_secs_f64();
    println!(r#"{{"kind":"mul","threads":{threads},"chain_1t_per_s":{one:.4e},"chain_all_per_s":{all:.4e},"vector_dot_per_s":{vec:.4e}}}"#);
    drop((a, b));

    for mu in mus("GKR_MUS", "20 22 24") {
        let cells = 1usize << mu;
        let w: Vec<F128> = (0..cells).map(|_| if rng.next_u64() & 1 == 1 { F128::ONE } else { F128::ZERO }).collect();
        let sigma = rng.permutation(cells);
        let g: Vec<F128> = (0..cells).map(|i| w[sigma[i]]).collect();
        let (mut tp, mut tv, mut bytes) = (Vec::new(), Vec::new(), 0);
        for run in 0..=runs {
            let mut chp = FsChallenger::new(b"rec-holo-gkr");
            let t = Instant::now();
            let (proof, _) = prove_batched(&w, &g, &sigma, None, &mut chp);
            let p = t.elapsed().as_secs_f64();
            let mut chv = FsChallenger::new(b"rec-holo-gkr");
            let t = Instant::now();
            verify_batched(mu, &proof, None, &mut chv).expect("verify_batched accepts");
            let v = t.elapsed().as_secs_f64();
            if run > 0 {
                tp.push(p);
                tv.push(v);
            }
            bytes = bincode::serialize(&proof).unwrap().len();
        }
        let p = median(tp);
        println!(
            r#"{{"kind":"gkr","mu":{mu},"threads":{threads},"prove_s":{p:.6},"ns_per_cell":{:.3},"verify_s":{:.6},"proof_bytes":{bytes},"peak_rss_mb":{}}}"#,
            p * 1e9 / cells as f64,
            median(tv),
            peak_rss_mb()
        );
    }

    let cd: usize = var("CB_D").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    for k_log in mus("CB_KLOGS", "21 24") {
        let k = 1usize << k_log;
        let idx = |salt: u64| -> Vec<u32> {
            (0..k * cd)
                .into_par_iter()
                .map(|j| {
                    let mut x = (j as u64 ^ salt).wrapping_add(0x9E37_79B9_7F4A_7C15);
                    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    ((x ^ (x >> 31)) as usize & (k - 1)) as u32
                })
                .collect()
        };
        let (ia, ib) = (idx(0xA), idx(0xB));
        let e = rng.f128_vec(k);
        let alpha = rng.f128();
        let mut ts = Vec::new();
        for run in 0..=runs {
            let t = Instant::now();
            let comb: Vec<F128> = (0..k)
                .into_par_iter()
                .map(|c| {
                    let s = |v: &[u32]| v[c * cd..(c + 1) * cd].iter().fold(F128::ZERO, |acc, &i| acc + e[i as usize]);
                    alpha * s(&ia[..]) + s(&ib[..])
                })
                .collect();
            black_box(&comb);
            if run > 0 {
                ts.push(t.elapsed().as_secs_f64());
            }
        }
        let s = median(ts);
        println!(
            r#"{{"kind":"comb","k_log":{k_log},"nnz_per_col":{},"threads":{threads},"s":{s:.6},"ns_per_nonzero":{:.3},"peak_rss_mb":{}}}"#,
            2 * cd,
            s * 1e9 / (2 * cd * k) as f64,
            peak_rss_mb()
        );
    }

    let d: usize = var("LC_D").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    for case in var("LC_CASES").unwrap_or_else(|_| "20:11 20:20".into()).split([' ', ',']).filter(|t| !t.is_empty()) {
        let (m, k_log) = case.split_once(':').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())).unwrap();
        let (a_0, b_0) = (random_rows(k_log, d, 0xA0 ^ m as u64), random_rows(k_log, d, 0xB0 ^ m as u64));
        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ts = Vec::new();
        for run in 0..=runs {
            let mut z = vec![0u8; (1usize << m) / 8];
            rng.fill_bytes(&mut z);
            let x = QuirkyPoint {
                z_skip: SkipPoint::Phi8(rng.f128()),
                x_inner_rest: rng.f128_vec(k_log - 6),
                x_outer: rng.f128_vec(m - k_log),
            };
            let mut ch = FsChallenger::new(b"rec-holo-lincheck");
            let t = Instant::now();
            black_box(prove(&z, m, k_log, 6, &circuit, &x, &mut ch));
            if run > 0 {
                ts.push(t.elapsed().as_secs_f64());
            }
        }
        let s = median(ts);
        println!(
            r#"{{"kind":"lincheck","m":{m},"k_log":{k_log},"nnz_per_row":{},"threads":{threads},"prove_s":{s:.6},"ns_per_row":{:.3},"peak_rss_mb":{}}}"#,
            2 * d,
            s * 1e9 / (1usize << m) as f64,
            peak_rss_mb()
        );
    }

    for mu in mus("SC_MUS", "20 22 24") {
        let terms = 1usize << mu;
        let (v, e1, e2) = (rng.f128_vec(terms), rng.f128_vec(terms), rng.f128_vec(terms));
        let mut ts = Vec::new();
        for run in 0..=runs {
            let t = Instant::now();
            black_box(sumcheck3(&v, &e1, &e2, &mut rng));
            if run > 0 {
                ts.push(t.elapsed().as_secs_f64());
            }
        }
        let s = median(ts);
        println!(
            r#"{{"kind":"sumcheck3","mu":{mu},"threads":{threads},"prove_s":{s:.6},"ns_per_term":{:.3}}}"#,
            s * 1e9 / terms as f64
        );
    }
}
