//! A served window's bits committed with Flock's PCS (Ligerito over F_{2^128}, `hm96-sha512` column leaves) and opened at
//! evaluation points, against hashing the same bytes as hm96 rows. One JSON object per line on stdout:
//! `{"kind": "pcs", hash, m, log_inv_rate, claims, bytes, commit_s, open_s, verify_s, proof_bytes}` and
//! `{"kind": "hm96_rows", m, bytes, row_bytes, hash_s}`.
//! env: M (window sizes as log2 of its bits, default "26 28 30"), RATES (log_inv_rate, default "1 2"), CLAIMS (claims per
//! opening, default "1 16"), ROW_BYTES (hm96 row size, default 4096), HASH (the PCS's Merkle leaves: `hm96-sha512`, the
//! default, or `sha512`), ROWS_BASELINE (0 skips the hm96 rows' timing).

use std::{env, time::Instant};

use flock_core::{
    challenger::FsChallenger,
    pcs::{
        LOG_PACKING, OpeningGrinding, PcsParams, commit,
        ligerito::{LigeritoProfile, prover_config_for, verifier_config_for},
        open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding,
        verify_opening_batch_ligerito_mixed_with_grinding,
    },
    zerocheck::{PaddingSpec, multilinear::lagrange_weights_naive, univariate_skip::build_eq},
};
use flock_field::{F128, test_rng::Rng};
use flock_hash::HashKind;
use rayon::prelude::*;
use sha2::{Digest, Sha512};

const K_SKIP: usize = 6;
const INITIAL_K: usize = 6;

fn list(name: &str, default: &str) -> Vec<usize> {
    env::var(name).unwrap_or_else(|_| default.into()).split_whitespace().map(|s| s.parse().expect(name)).collect()
}

/// SplitMix64: a seeded stream of words, the window's stand-in values.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `2^m` seeded bits, packed as `pack_witness` lays them out: bit `i` of word `w` is bit `128·w + i` of the window, the
/// low 64 in `F128`'s `lo` half.
fn window(m: usize, seed: u64) -> (Vec<u128>, Vec<F128>) {
    let mut s = seed;
    let words: Vec<u128> =
        (0..1usize << (m - LOG_PACKING)).map(|_| u128::from(splitmix(&mut s)) | (u128::from(splitmix(&mut s)) << 64)).collect();
    let packed = words.iter().map(|w| F128::new(*w as u64, (*w >> 64) as u64)).collect();
    (words, packed)
}

fn bit(words: &[u128], i: usize) -> bool {
    (words[i / 128] >> (i % 128)) & 1 == 1
}

/// The ring-switched claim's value: the skip-variable bit slices' multilinear extension at `(z_skip, x_outer)`.
fn claim(words: &[u128], m: usize, z_skip: F128, x_outer: &[F128]) -> F128 {
    let lambda = lagrange_weights_naive(K_SKIP, z_skip);
    let eq_outer = build_eq(x_outer);
    (0..1usize << (m - K_SKIP))
        .into_par_iter()
        .map(|o| {
            let mut inner = F128::ZERO;
            for (s, l) in lambda.iter().enumerate() {
                if bit(words, (o << K_SKIP) + s) {
                    inner += *l;
                }
            }
            eq_outer[o] * inner
        })
        .reduce(|| F128::ZERO, |a, b| a + b)
}

fn pcs_point(m: usize, log_inv_rate: usize, claims: usize, hash: &str) {
    let (words, packed) = window(m, 0x5e_7ed ^ m as u64);
    let merkle_hash = match hash {
        "hm96-sha512" => HashKind::Hm96Sha512,
        "sha512" => HashKind::Sha512,
        other => panic!("HASH {other:?}: hm96-sha512 or sha512"),
    };
    let params = PcsParams {
        m,
        log_inv_rate,
        log_batch_size: INITIAL_K,
        profile: Default::default(),
        num_lanes: None,
        merkle_hash,
    };
    let t = Instant::now();
    let (commitment, prover_data) = commit(&packed, &params);
    let commit_s = t.elapsed().as_secs_f64();

    let mut rng = Rng::new(0xc1a1_0000 ^ m as u64);
    let points: Vec<(F128, Vec<F128>)> =
        (0..claims).map(|_| (rng.f128(), (0..m - K_SKIP).map(|_| rng.f128()).collect())).collect();
    let x_outers: Vec<&[F128]> = points.iter().map(|(_, x)| x.as_slice()).collect();
    let log_n = m - LOG_PACKING;
    let profile = LigeritoProfile::Fast;
    let p_cfg = prover_config_for(log_n, INITIAL_K, profile).expect("prover config");
    let v_cfg = verifier_config_for(log_n, INITIAL_K, profile).expect("verifier config");

    let t = Instant::now();
    let mut ch = FsChallenger::new(b"verity/served-pcs/v0");
    let proof = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_grinding(
        packed,
        &prover_data,
        &commitment,
        &x_outers,
        &[],
        &[],
        &PaddingSpec::dense(m),
        &p_cfg,
        OpeningGrinding::disabled(),
        &mut ch,
    );
    let open_s = t.elapsed().as_secs_f64();

    let values: Vec<F128> = points.iter().map(|(z, x)| claim(&words, m, *z, x)).collect();
    let weights: Vec<Vec<F128>> = points.iter().map(|(z, _)| lagrange_weights_naive(K_SKIP, *z)).collect();
    let weight_refs: Vec<&[F128]> = weights.iter().map(|w| w.as_slice()).collect();
    let t = Instant::now();
    let mut ch = FsChallenger::new(b"verity/served-pcs/v0");
    verify_opening_batch_ligerito_mixed_with_grinding(
        &commitment,
        &values,
        &weight_refs,
        &x_outers,
        &[],
        &proof,
        &v_cfg,
        OpeningGrinding::disabled(),
        &mut ch,
    )
    .unwrap_or_else(|e| panic!("the verifier refused an honest opening: {e:?}"));
    let verify_s = t.elapsed().as_secs_f64();
    let proof_bytes = bincode::serialize(&proof).expect("proof bytes").len();
    println!(
        "{{\"kind\":\"pcs\",\"hash\":\"{hash}\",\"m\":{m},\"log_inv_rate\":{log_inv_rate},\"claims\":{claims},\"bytes\":{},\"commit_s\":{commit_s:.4},\
         \"open_s\":{open_s:.4},\"verify_s\":{verify_s:.4},\"proof_bytes\":{proof_bytes}}}",
        1usize << (m - 3)
    );
}

/// The same bytes as hm96-sha512 rows of `row_bytes`: per row, the inner digest, the salt's digest over 192 bytes and the
/// leaf over both (`verity/ml/commitments/hm96/PROTOCOL.md`), in parallel over rows as a committer would.
fn hm96_rows(m: usize, row_bytes: usize) {
    let bytes = 1usize << (m - 3);
    let data = vec![0xa5u8; bytes];
    let salt = [0x5au8; 192];
    let t = Instant::now();
    let leaves: usize = data
        .par_chunks(row_bytes)
        .map(|row| {
            let x = Sha512::new().chain_update([0u8; 128]).chain_update(row).finalize();
            let c = Sha512::new().chain_update([1u8; 128]).chain_update(salt).finalize();
            let leaf = Sha512::new().chain_update([2u8; 128]).chain_update(x).chain_update(c).finalize();
            usize::from(leaf[0] == 0)
        })
        .sum();
    let hash_s = t.elapsed().as_secs_f64();
    std::hint::black_box(leaves);
    println!("{{\"kind\":\"hm96_rows\",\"m\":{m},\"bytes\":{bytes},\"row_bytes\":{row_bytes},\"hash_s\":{hash_s:.4}}}");
}

fn main() {
    let row_bytes = list("ROW_BYTES", "4096")[0];
    let hash = env::var("HASH").unwrap_or_else(|_| "hm96-sha512".into());
    let baseline = env::var("ROWS_BASELINE").map_or(true, |v| v != "0");
    for m in list("M", "26 28 30") {
        if baseline {
            hm96_rows(m, row_bytes);
        }
        for rate in list("RATES", "1 2") {
            for claims in list("CLAIMS", "1 16") {
                pcs_point(m, rate, claims, &hash);
            }
        }
    }
}
