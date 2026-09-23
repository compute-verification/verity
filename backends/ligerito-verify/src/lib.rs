//! ligerito-verify: the independent verifier for the Ligerito backend (see `pcs` for the protocol, `main.rs` for
//! the CLI).  Module map: `field` (BabyBear + F_{p^6}), `hash` (BLAKE3 / BLAKE2b / SHAKE-256), `merkle`
//! (multi-openings), `transcript` (coins: Fiat-Shamir diagnostic or live replay), `claims` (tensor terms),
//! `params` (dimensions + soundness), `pcs` (the PCS verifier), `format` (proof bytes), `refprover` (test oracle).

pub mod claims;
pub mod field;
pub mod format;
pub mod hash;
pub mod merkle;
pub mod params;
pub mod pcs;
pub mod refprover;
pub mod transcript;

#[cfg(test)]
mod tests {
    use crate::field::Ext;
    use crate::params::Params;
    use crate::pcs::{verify, Proof, Statement};
    use crate::refprover::{prove, random_point, random_vector, tamper};
    use crate::transcript::{FiatShamir, Recording, Replay};

    fn fs() -> FiatShamir {
        FiatShamir::new(FiatShamir::LABEL)
    }

    fn honest(p: &Params, seed: u64) -> (Statement, Proof, Vec<Vec<u8>>) {
        let f = random_vector(p.n, seed);
        let z = random_point(p.n, seed + 1);
        let mut rec = Recording::new(fs());
        let (stmt, proof) = prove(p, &f, &z, &mut rec);
        (stmt, proof, rec.slots)
    }

    fn check_params(p: Params) {
        let (stmt, proof, slots) = honest(&p, 42);
        let rep = verify(&stmt, &proof, &mut fs()).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        assert_eq!(rep.coins, slots.len());
        // the same proof under replayed coins
        let mut rp = Replay::new(slots.clone());
        verify(&stmt, &proof, &mut rp).unwrap();
        assert!(rp.exhausted());
        // every negative rejects (FS)
        for (what, s, pf) in tamper(&stmt, &proof) {
            assert!(verify(&s, &pf, &mut fs()).is_err(), "{p:?}: accepted {what}");
            let mut rp = Replay::new(slots.clone());
            assert!(verify(&s, &pf, &mut rp).is_err(), "{p:?}: accepted {what} under replay");
        }
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
    fn coset_domain() {
        let mut p = Params::new(9, vec![3, 3], vec![8, 8], 2);
        p.domain_shift = 31;
        check_params(p);
    }

    #[test]
    fn rate_one_eighth() {
        check_params(Params::new(9, vec![3, 2], vec![8, 8], 3));
    }

    #[test]
    fn spliced_commitment_is_rejected() {
        // Fixed (replayed) coins: commit f, but run the rest of the protocol for f' = f + e_0.  The opened rows of
        // round 1 and their symbols come from f (so they pass the row . r-bar = v check and the Merkle check), the
        // later rounds from f'.  The batched claim then disagrees with y.
        let p = Params::new(8, vec![3, 2], vec![6, 6], 2);
        let f = random_vector(p.n, 7);
        let mut f2 = f.clone();
        f2[0] = crate::field::add(f2[0], 1);
        let z = random_point(p.n, 8);
        let mut rec = Recording::new(fs());
        let (stmt2, proof2) = prove(&p, &f2, &z, &mut rec);
        let slots = rec.slots;
        let (_stmt1, proof1) = prove(&p, &f, &z, &mut Replay::new(slots.clone()));
        let mut spliced = proof2.clone();
        spliced.roots[0] = proof1.roots[0];
        spliced.openings[0] = proof1.openings[0].clone();
        assert!(verify(&stmt2, &spliced, &mut Replay::new(slots.clone())).is_err());
        // sanity: both honest proofs verify under the same coins
        verify(&stmt2, &proof2, &mut Replay::new(slots.clone())).unwrap();
        verify(&_stmt1, &proof1, &mut Replay::new(slots)).unwrap();
    }

    #[test]
    fn statement_must_match_the_transcript() {
        let p = Params::new(8, vec![3, 2], vec![6, 6], 2);
        let (stmt, proof, _) = honest(&p, 3);
        // a different FS label changes every challenge
        let mut t = FiatShamir::new(b"other-label");
        assert!(verify(&stmt, &proof, &mut t).is_err());
        // a proof for one point does not verify at another point with the correct value there
        let f = random_vector(p.n, 3);
        let z2 = random_point(p.n, 99);
        let alpha2 = crate::refprover::evaluate(&f, &z2);
        let s2 = Statement { z: z2, alpha: alpha2 };
        assert!(verify(&s2, &proof, &mut fs()).is_err());
        let _ = Ext::ZERO;
    }

    #[test]
    fn soundness_report() {
        let p = Params::new(12, vec![4, 3, 2], vec![12, 8, 8], 2);
        let (stmt, proof, _) = honest(&p, 1);
        let rep = verify(&stmt, &proof, &mut fs()).unwrap();
        assert_eq!(rep.opened_rows, 28);
        assert_eq!(rep.terms_final, 1 + 12 + 8);
        assert!(rep.soundness_log2 < 0.0);
    }
}
