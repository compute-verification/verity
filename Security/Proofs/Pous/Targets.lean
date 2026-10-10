import Proofs.Pous.Statements

/-!
# Open targets (untrusted)

Nothing here is trusted, and nothing here is in the default build: it is the list of what to prove next. Every target is
stated in `Security/Proofs/Pous/Statements.lean` or `Security/Models/Pous/Guarantees.lean`; `grade.sh` grades a proof of
`Pous.Guarantees.X` given as `theorem solution`. The names below are the draft names the prover files use. The proved
statements are in `Security/Proofs/Pous.lean`.
-/

namespace PousTargets

open Pous

/-! ## Open -/

theorem a2_lemmaA_m1 : Guarantees.A2LemmaAM1 := sorry
theorem a5_rabin_gcd : Guarantees.A5RabinGcd := sorry
theorem a7_m1_params : Guarantees.A7Params := sorry
theorem m1_meets : Guarantees.M1Meets := sorry

/-! ## Not stated yet

* A2 in M2/M3 (the identity factor and the `(M − g)!/M!` pool, on the `Dist` event that A3′'s `G` carries); A7′ has
  the M2 numbers only.
* `k·π` in place of `(Σ_{j<k} B^j)·π` in Theorem 1(ii) sequential (a stopped-process argument; unproved).
* B4 (certificate format for a tile graph's depth robustness).
* N10′, a column-aware restatement of the refuted N10 (being drafted).
* A P3 bound with local decoding beyond two layers, and a bridge from B2's graph model to `TopoDAG`.
* Report Theorem 2 (no black-box reduction) is a meta-theorem about reductions and has no statement in this model;
  Theorem A and the lattice/NTRU bounds are standard-model statements about specific algebra.
-/

end PousTargets
