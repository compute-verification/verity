//! What a first-class lookup costs Flock's prover against the committed bits it replaces, on CPU: upstream's packed
//! zerocheck (`zerocheck::prove_packed`, the work every committed bit pays at a fixed m) per bit, against upstream's
//! grand-product GKR (`product_gkr::prove_batched`, the char-2 lookup's multiset check, no commitment) per leaf. A lookup is
//! one leaf of its side's product circuit; `prove_batched` runs two circuits of N leaves in lockstep, so a lookup whose table
//! side is small costs about half its per-leaf figure.
//!
//! Build and run from a flock b684b12 checkout, with this file as `crates/flock-core/examples/lookup_cost.rs`:
//!
//!   cargo build --release -p flock-core --example lookup_cost
//!   RAYON_NUM_THREADS=1 target/release/examples/lookup_cost [zc_mmin zc_mmax gkr_mumin gkr_mumax]
use std::{env, time::Instant};

use flock_core::{challenger::FsChallenger, field::F128, product_gkr::prove_batched, test_rng::Rng, zerocheck::prove_packed};

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let a: Vec<usize> = env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (zmin, zmax, gmin, gmax) = (a.first().copied().unwrap_or(22), a.get(1).copied().unwrap_or(26),
                                    a.get(2).copied().unwrap_or(18), a.get(3).copied().unwrap_or(22));
    let threads = rayon::current_num_threads();
    let mut rng = Rng::new(20261001);
    for m in zmin..=zmax {
        let n = (1usize << m) / 8;
        let mut x = vec![0u8; n];
        let mut y = vec![0u8; n];
        rng.fill_bytes(&mut x);
        rng.fill_bytes(&mut y);
        let z: Vec<u8> = x.iter().zip(&y).map(|(p, q)| p & q).collect();
        let s = best(3, || prove_packed(&x, &y, &z, m, &mut FsChallenger::new(b"verity-lookup-cost-zc")));
        println!("{}", serde_json::json!({"kind": "zerocheck", "m": m, "bits": 1u64 << m, "threads": threads, "prove_s": s,
                                          "s_per_bit": s / (1u64 << m) as f64}));
    }
    for mu in gmin..=gmax {
        let n = 1usize << mu;
        let f = rng.f128_vec(n);
        let g = f.clone();
        let sigma: Vec<usize> = (0..n).collect();
        let s = best(3, || prove_batched(&f, &g, &sigma, None, &mut FsChallenger::new(b"verity-lookup-cost-gkr")));
        println!("{}", serde_json::json!({"kind": "product_gkr_batched", "mu": mu, "leaves": n, "threads": threads, "prove_s": s,
                                          "s_per_leaf": s / n as f64}));
    }
    let _ = F128::ONE;
}
