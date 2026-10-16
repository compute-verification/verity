import Definitions.Pouw.RelativeSecurity
import Models.Pouw.Assumptions.RelativeSecurity

/-!
# Relative security's guarantees for the staged window (trusted; draft)

Each guarantee is a `def G : Prop` over `Definitions.Pouw.RelativeSecurity` with the named premises it rests on as
hypotheses (`Models.Pouw.Assumptions.RelativeSecurity`). Its proof is the theorem
`Pouw.SecurityProofs.RelativeSecurity.G : Pouw.Guarantees.RelativeSecurity.G`.
-/

namespace Pouw.Guarantees.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

/-- **Theorem S for the staged spot check**, with no premise. In every staged play (`R` rounds of `M` stages, each
stage's error table fixed before its draw, one check a round of a drawn stage and cell), if every stage's error is of
degree at most 2 on the cube, the play is accepted with deficiency at least `D` with probability at most
`e^{−D/4}`: the luck ¾ (a nonzero degree-2 error is nonzero on a quarter of the cube), through the stages'
supermartingale. -/
def TheoremSStaged : Prop :=
  ∀ (C V A : Type) [Fintype C] [Fintype V] [DecidableEq V] [CommRing A] (M R : ℕ) (P : StagedPlay C V A M R),
    (∀ x w s c, CubeQuad (P.errAt x w s c)) → ∀ D : ℝ,
      prob (fun x => P.game.accepts x ∧ D ≤ P.game.deficiency x) ≤ Real.exp (-(D / 4))

/-- **The staged window's service floor**, the theorem of record: Theorem K at chain factor 1 (Lemma G, s32 wrap read
mod 2), the luck ¾ and the stage floors summed. In every machine `Mc` and for every service `S` whose seed is `R`
rounds of `St` stage draws on the re-encoded design (7 planes at `wRe`, integers mod `2^32`), under the three
premises, every adversary that commits first has its accepted runs charged below
`(R − 160 ln 2) · St · κ L n n' B − Vmax` with probability at most `2^{−40} + R (δb + δz)`. -/
def StagedFloorK : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St n L B n' : ℕ) (e : S.Seed ≃ Paths (DStage St n L 7 B n') R)
    (run : RunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < St → 0 < κ → 0 < L * (n * n' * B) → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundStaged Mc S run e δb δz → KScopeAffine Mc S run wRe → ChargeVF3f Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 160 * Real.log 2) * (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

end Pouw.Guarantees.RelativeSecurity
