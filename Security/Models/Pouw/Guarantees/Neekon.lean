import Definitions.Pouw.Neekon
import Models.Pouw.Assumptions.Neekon

/-!
# Dither's Theorem B in Neekon's two-stage definition (trusted)

The guarantee is a `def G : Prop` over `Definitions.Pouw.Neekon` with assumption H
(`Models.Pouw.Assumptions.Neekon`) as a hypothesis. Its proof is `Pouw.SecurityProofs.Neekon.TheoremB :
Pouw.Guarantees.Neekon.TheoremB`.

**Theorem A and H.** Theorem K's floor of record, `StagedFloorKOneSided` (`Models.Pouw.Guarantees.RelativeSecurity`),
bounds the product count of the runs in its scope (`KScopeOneSided`: one-sided products, charge-pure, straight-line),
not every algorithm, so it gives H's conclusion only for those runs. Dither's transcript is its targets'
table: with the draw read as the design's bits (`DSeed`), `T(i, j, ℓ)` is `dPrefix` of cell `(j, i)` at segment `ℓ` on
the design of shape `(n, 32, 7, K, m)` and weights `wRe`, with `B − 64` and `A` committed
(`Pouw.SecurityProofs.Neekon.transcript_eq_dPrefix`). The rest of the bridge isn't small:
- H's adversary is one algorithm of `Mc` with a `time`; the floor's is a straight-line program per stage, given by an
  analysis map (`RunMap`) that its premises constrain. Restricting H to runs in K's scope is a premise on that map;
- the floor is over `R` rounds of a spot check, one cell-block a round, and its bound is `(R − 274 ln 2) · St · κ L n n'
  B − Vmax`, which is negative at one round. H is one draw checked whole. The one-draw form needs the per-stage
  count (`Slp`'s `L` products per certified cell-block, by charging purity) with the parity luck 0.854 per word, which
  gives H's conclusion at `δ > 0.854` with `tHonest = κ · 32 · n m K` and `εH = Vmax / tHonest`. Those lemmas are in
  `Proofs.Pouw.RelativeSecurity`, and that statement is not proved here. Its last step is
  `Pouw.SecurityProofs.Neekon.le_time_of_runFloor`: if `M₂`'s correct runs charged below `F` have probability at most
  `κ < δ`, then `time(M₂) ≥ F`, H's conclusion at `F = (1 − εH) tHonest`.
-/

namespace Pouw.Guarantees.Neekon

open Pouw.RelativeSecurity Pouw.Neekon Pouw.Neekon.Assumptions

/-- **Dither's Theorem B**: under H, Dither's ideal protocol is secure in Neekon's two-stage definition, with
`M' = Solve` and `η = 0`. In every machine `Mc`, if `Solve` computes the transcript and the edited product `(A + R)B`
and costs at most `tHonest + tStrip` (the honest transcript and the strip `(ÃΠ)Uᵀ`), then H at `(εH, δ, tHonest)`
gives `NeekonSecure ε δ 0` with `1 + ε = (1 + tStrip / tHonest) / (1 − εH)`: every adversary accepted with
probability at least `δ` has `time(M₂) ≥ (1 − εH) tHonest`, and `Solve` costs at most `1 + ε` times that. -/
def TheoremB : Prop :=
  ∀ (Mc : Machine) (m K n : ℕ)
    (solve : Mc.Alg (DSeed m K n × DInst m K n) (DTrans m K n × Matrix (Fin m) (Fin n) Z32))
    (εH δ tHonest tStrip : ℝ),
    0 < tHonest → εH < 1 →
    (∀ σ x, Mc.eval solve (σ, x) = (transcript σ x, editA σ x * x.2)) →
    Mc.time solve ≤ tHonest + tStrip →
    DitherH Mc m K n εH δ tHonest →
    NeekonSecure (dither Mc m K n solve) ((1 + tStrip / tHonest) / (1 - εH) - 1) δ 0

end Pouw.Guarantees.Neekon
