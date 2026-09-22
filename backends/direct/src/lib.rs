//! Candidate B build lane, wave 1: the WP2 transition-unit contracts as direct AIR constraints over Goldilocks,
//! proved with Plonky3 `uni-stark` (FRI + Keccak Merkle).  `NON_ZK_PROOF_DIAGNOSTIC`, `authentication = excluded`.
//!
//! * `params`  the frozen widths (mirrors `checker/params.py::REAL`)
//! * `program` the unit as a straight-line program over an abstract context (mirrors `checker/step.py`)
//! * `ctx`     the witness interpreter and the AIR interpreter; range checks as 2-bit limbs, lookups as one-hot
//! * `air`     the one-row-per-unit AIR, the trace, and a concrete constraint evaluator
//! * `stark`   the Plonky3 configuration

pub mod air;
pub mod challenger;
pub mod rtt;
pub mod ctx;
pub mod lookup;
pub mod lookup_stark;
pub mod params;
pub mod program;
pub mod stark;

use program::UnitIo;
use rand::{Rng, SeedableRng};

/// `step_output`: the word the relation assigns to `x` through the honest hints (the `pack.y` column).
pub fn step_output(c: u64, a: &[u64], b: &[u64]) -> u64 {
    let lay = air::layout();
    let io = UnitIo { c, a: a.to_vec(), b: b.to_vec(), y: 0 };
    let (row, _) = air::gen_row(&io, &[]);
    air::f_to_u64(row[lay.col("pack.y")])
}

/// Random finite words: every exponent field below 255.
pub fn random_units(n: usize, seed: u64) -> Vec<UnitIo> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let lay = air::layout();
    let ycol = lay.col("pack.y");
    (0..n)
        .map(|_| {
            let finite16 = |r: &mut rand::rngs::StdRng| loop {
                let w: u64 = r.gen_range(0..1u64 << 16);
                if (w >> 7) & 0xFF != 0xFF {
                    return w;
                }
            };
            let c = loop {
                let w: u64 = rng.gen_range(0..1u64 << 32);
                if (w >> 23) & 0xFF != 0xFF {
                    break w;
                }
            };
            let a: Vec<u64> = (0..16).map(|_| finite16(&mut rng)).collect();
            let b: Vec<u64> = (0..16).map(|_| finite16(&mut rng)).collect();
            let mut io = UnitIo { c, a, b, y: 0 };
            let (row, _) = air::gen_row(&io, &[]);
            io.y = air::f_to_u64(row[ycol]);
            io
        })
        .collect()
}
