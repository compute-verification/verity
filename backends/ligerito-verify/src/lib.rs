//! ligerito-verify: the independent verifier for the Ligerito backend (see `pcs` for the protocol, `main.rs` for
//! the CLI).  Module map: `field` (BabyBear + F_{p^6}), `hash` (BLAKE3 / BLAKE2b / SHAKE-256), `merkle`
//! (multi-openings), `transcript` (coins: Fiat-Shamir diagnostic, live coins, local), `claims` (tensor terms),
//! `params` (dimensions + soundness), `pcs` (the PCS verifier), `format` + `json` (proof bytes), `refprover`
//! (test oracle), `sumcheck` (relation-side round verification: plain deg-2 and Gruen deg-3 forms).

pub mod claims;
pub mod field;
pub mod format;
pub mod hash;
pub mod json;
pub mod lgsc3;
pub mod merkle;
pub mod params;
pub mod pcs;
pub mod refprover;
pub mod refpcs;
pub mod relation;
pub mod sumcheck;
pub mod transcript;

#[cfg(test)]
mod tests {
    use crate::params::Params;
    use crate::pcs::{verify, Proof, Statement};
    use crate::refprover::{prove, random_point, random_vector, tamper};
    use crate::transcript::{Coins, FiatShamir, Live, Local};

    fn fs() -> FiatShamir {
        FiatShamir::new(FiatShamir::LABEL)
    }

    fn honest(p: &Params, seed: u64, coins: &mut dyn Coins) -> (Statement, Proof) {
        let f = random_vector(p.n, seed);
        let z = random_point(p.n, seed + 1);
        prove(p, &f, &z, coins)
    }

    fn check_params(p: Params) {
        // Fiat-Shamir
        let (stmt, proof) = honest(&p, 42, &mut fs());
        let rep = verify(&stmt, &proof, &mut fs()).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        let expected_coins = p.kprime.iter().sum::<usize>() + 2 * p.rounds() - 1;
        assert_eq!(rep.coins, expected_coins);
        for (what, s, pf) in tamper(&stmt, &proof) {
            assert!(verify(&s, &pf, &mut fs()).is_err(), "{p:?}: accepted {what}");
        }
        // live coins (interactive replay): the prover and the verifier see the same coin list
        let coins = Live::random_coins(expected_coins, 7);
        let (stmt, proof) = honest(&p, 43, &mut Live::new(coins.clone()));
        let mut lv = Live::new(coins.clone());
        verify(&stmt, &proof, &mut lv).unwrap_or_else(|e| panic!("{p:?} live: {e}"));
        assert!(lv.exhausted());
        for (what, s, pf) in tamper(&stmt, &proof) {
            assert!(verify(&s, &pf, &mut Live::new(coins.clone())).is_err(), "{p:?}: accepted {what} under live coins");
        }
        // a proof made under one coin list fails under another
        assert!(verify(&stmt, &proof, &mut Live::new(Live::random_coins(expected_coins, 8))).is_err());
        assert!(verify(&stmt, &proof, &mut fs()).is_err());
        // local coins
        let (stmt, proof) = honest(&p, 44, &mut Local::new(3));
        verify(&stmt, &proof, &mut Local::new(3)).unwrap();
        assert!(verify(&stmt, &proof, &mut Local::new(4)).is_err());
    }

    #[test]
    fn two_rounds_small() {
        check_params(Params::new(8, vec![3, 2], vec![6, 6], 2));
    }

    #[test]
    fn three_rounds() {
        check_params(Params::new(12, vec![4, 3, 2], vec![12, 8, 8], 2));
    }

    #[test]
    fn one_committed_round_is_ligero() {
        check_params(Params::new(8, vec![4], vec![10], 2));
    }

    #[test]
    fn coset_domain_and_split() {
        let mut p = Params::new(9, vec![3, 3], vec![8, 8], 2);
        p.domain_shift = 31;
        p.splits = vec![(32, 8), (4, 8)];
        check_params(p);
    }

    #[test]
    fn rate_one_eighth() {
        check_params(Params::new(9, vec![3, 2], vec![8, 8], 3));
    }

    #[test]
    fn spliced_commitment_is_rejected() {
        // Fixed (live) coins: commit f, but run the rest of the protocol for f' = f + e_0.  The opened rows of
        // round 1 come from f (so they pass the Merkle check), the later rounds from f'.  The batched claim then
        // disagrees with y.
        let p = Params::new(8, vec![3, 2], vec![6, 6], 2);
        let coins = Live::random_coins(8, 1);
        let f = random_vector(p.n, 7);
        let mut f2 = f.clone();
        f2[0] = crate::field::add(f2[0], 1);
        let z = random_point(p.n, 8);
        let (stmt2, proof2) = prove(&p, &f2, &z, &mut Live::new(coins.clone()));
        let (stmt1, proof1) = prove(&p, &f, &z, &mut Live::new(coins.clone()));
        let mut spliced = proof2.clone();
        spliced.roots[0] = proof1.roots[0];
        spliced.openings[0] = proof1.openings[0].clone();
        assert!(verify(&stmt2, &spliced, &mut Live::new(coins.clone())).is_err());
        verify(&stmt2, &proof2, &mut Live::new(coins.clone())).unwrap();
        verify(&stmt1, &proof1, &mut Live::new(coins)).unwrap();
    }

    #[test]
    fn statement_must_match_the_transcript() {
        let p = Params::new(8, vec![3, 2], vec![6, 6], 2);
        let (stmt, proof) = honest(&p, 3, &mut fs());
        assert!(verify(&stmt, &proof, &mut FiatShamir::new(b"other-label")).is_err());
        let f = random_vector(p.n, 3);
        let z2 = random_point(p.n, 99);
        let v2 = crate::refprover::evaluate(&f, &z2);
        assert!(verify(&Statement { z: z2, v: v2 }, &proof, &mut fs()).is_err());
        // header binding: the same proof re-labelled with different declared parameters fails
        let mut pf = proof.clone();
        pf.params.queries[0] += 1;
        assert!(verify(&stmt, &pf, &mut fs()).is_err());
    }

    #[test]
    fn soundness_report() {
        let p = Params::new(12, vec![4, 3, 2], vec![12, 8, 8], 2);
        let (stmt, proof) = honest(&p, 1, &mut fs());
        let rep = verify(&stmt, &proof, &mut fs()).unwrap();
        assert_eq!(rep.opened_rows, 28);
        assert_eq!(rep.terms_final, 1 + 12 + 8);
        assert!(rep.soundness_log2 < 0.0);
    }

    /// Proofs written by the Python prototype (`ligerito-proto` b4d7e12, `proto/pcs.py::prove` on CPU, seed 1) must
    /// verify bit-for-bit under Fiat-Shamir, and byte flips anywhere in the header, the claimed evaluation, a root,
    /// the sumcheck messages, the opened rows, the sibling digests, the final vector, or the point must be rejected.
    fn proto_fixture(proof: &[u8], z: &[u8], coins: usize, opened_rows: usize) {
        use crate::format::{read_point, read_proto};
        let zp = read_point(z).unwrap();
        let (stmt, pf) = read_proto(proof, zp.clone()).unwrap();
        let rep = verify(&stmt, &pf, &mut fs()).unwrap();
        assert_eq!(rep.coins, coins);
        assert_eq!(rep.opened_rows, opened_rows);
        // negatives: flip one bit at ~40 evenly spaced positions (the header length word excepted); parse or verify
        // must fail every time
        let step = (proof.len() / 40).max(1);
        for pos in (4..proof.len()).step_by(step) {
            let mut b = proof.to_vec();
            b[pos] ^= 1;
            let ok = read_proto(&b, zp.clone()).and_then(|(s, p)| verify(&s, &p, &mut fs()));
            assert!(ok.is_err(), "accepted a flipped bit at byte {pos}");
        }
        for pos in (0..z.len()).step_by(24) {
            let mut zb = z.to_vec();
            zb[pos] ^= 1;
            let zz = read_point(&zb).unwrap();
            let (s, p) = read_proto(proof, zz).unwrap();
            assert!(verify(&s, &p, &mut fs()).is_err(), "accepted a flipped point coordinate at byte {pos}");
        }
    }

    #[test]
    fn proto_fixture_n10_one_round() {
        proto_fixture(
            include_bytes!("../fixtures/proto/n10_1r.proof"),
            include_bytes!("../fixtures/proto/n10_1r.z"),
            5,
            24,
        );
    }

    /// Written by this crate's `selftest --n 12 --kprime 4,3,2 --queries 16,16,16 --rate-log2 2,3,4 --emit` in proto's
    /// container with the 7bc2fdd per-round `rate_log2` list; for the proto lane to cross-verify on the pod.
    #[test]
    fn rust_written_proto_container_with_per_round_rates() {
        use crate::format::{read_point, read_proto};
        let proof = include_bytes!("../fixtures/rust/n12_rates234.proof");
        let z = read_point(include_bytes!("../fixtures/rust/n12_rates234.z")).unwrap();
        let (stmt, pf) = read_proto(proof, z).unwrap();
        assert_eq!(pf.params.rate_log2, vec![2, 3, 4]);
        assert!(!pf.params.header_rate_scalar);
        assert_eq!(pf.params.header_json(), "{\"n\": 12, \"kprime\": [4, 3, 2], \"queries\": [16, 16, 16], \"rate\": [2, 3, 4]}");
        verify(&stmt, &pf, &mut fs()).unwrap();
    }

    #[test]
    fn proto_fixture_n12_three_rounds() {
        proto_fixture(
            include_bytes!("../fixtures/proto/n12.proof"),
            include_bytes!("../fixtures/proto/n12.z"),
            14,
            48,
        );
    }
}
