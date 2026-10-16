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

/-- **Theorem S with overlapping draws**, with no premise. In every play over `St` draws a round (`R` rounds, each
round's error tables fixed before its first draw, one check a round of a cell-block drawn among every stage's, at
every draw's drawn bits), if every cell-block's error is of degree at most 2 on the cube of its own draw's bits at
every bit point of the other draws', the play is accepted with deficiency `ovDef` at least `D` with probability at
most `e^{−D/4}`: the luck ¾ per draw, with the other draws' bits given. This is the luck `OverlapFloorK` reads. -/
def TheoremSOverlap : Prop :=
  ∀ (St : ℕ) (κ ι : Fin St → Type) [∀ t, Fintype (κ t)] [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] (A : Type)
    [CommRing A] (R : ℕ) (P : OvPlay κ ι A R), (∀ x w c U, CubeQuad fun z => P.errAt x w c (mergeAt U c.1 z)) →
    ∀ D : ℝ, prob (fun x => P.game.accepts x ∧ D ≤ P.game.deficiency x) ≤ Real.exp (-(D / 4))

/-- **The overlapped window's service floor**: Theorem K∥ (K at chain factor 1 per row of one draw at every bit
point of the other draws', Lemma G over F₂), the luck ¾ per draw and every stage's cell-blocks counted once. In
every machine `Mc` and for every service `S` whose seed is `R` rounds of `St` stage draws of any shapes (stage `t`:
`n t` rows, `B t` blocks of `L`, `n' t` columns, 7 planes at `wRe`, integers mod `2^32`) overlapping at any depth,
under the three premises, every adversary that commits first has its accepted runs charged below
`(R − 160 ln 2) · κ L Σ_t n_t n'_t B_t − Vmax` with probability at most `2^{−40} + R (δb + δz)`. -/
def OverlapFloorK : Prop :=
  ∀ (Mc : Machine) (S : Service) (R St L : ℕ) (n B n' : Fin St → ℕ) (e : S.Seed ≃ Paths (OSpot St n L 7 B n') R)
    (run : ORunMap Mc S R St n L 7 B n' (ZMod (2 ^ 32))) (δb δz κ Vmax : ℝ),
    0 < κ → 0 < L * ∑ t, n t * n' t * B t → 0 ≤ δb → 0 ≤ δz →
    ServiceSoundOverlap Mc S run e δb δz → KScopeOverlap Mc S run wRe → ChargeVF3fOverlap Mc S run e κ Vmax →
    ServiceFloor Mc S ((R - 160 * Real.log 2) * (κ * L * ((∑ t, n t * n' t * B t : ℕ) : ℝ)) - Vmax)
      (((2 : ℝ) ^ 40)⁻¹ + R * (δb + δz))

end Pouw.Guarantees.RelativeSecurity
