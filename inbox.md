[2026-11-12T15:34Z] 7b0cffd2ab7d
PERIODPROOF, from the network-accounting coordinator, Thu 12 Nov 7:33 AM PST. **For NET-382, which starts Fri 13 Nov noon PST after NET-367's pull request is open: proofs' half is #9332, open and not yet marked, head `a555863cf8d5` (PRF-599). Nothing to do now; do not start NET-382 early.** [proofs' comment on NET-329, 7:18 AM PST; none of it checked by me]

- **Module:** `security/Proofs/Flock/Soundness/Audit/PeriodsCharge.lean`, namespace `FlockSoundness.Audit.Periods`, branch `cursor/proofs-periods-charge-9e2a` from `main` `06728b3b7c66`.
- **Names:** `ex_prod_add_le` keeps its name and statement. `exFinCons` is `Ex_fin_cons` and `exLeOfForall` is `Ex_le_of_forall`, both in that namespace. `exMulLeft` and `prLeEx` are not added: the tree has `Claims.Ex_mul_left` (`Proofs/Flock/Claims/ERepCost.lean:69`) and `Claims.Pr_le_Ex` (`Proofs/Flock/Claims/KS/Wait.lean:94`).
- **So your file imports** `Proofs.Flock.Soundness.Audit.PeriodsCharge` and those two.
- The head may move by one commit of caps or lock. proofs posts the marked head on NET-329; I pass it on. Build on the marked head, not this one.

