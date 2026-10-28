import Definitions.Accounting.Assumptions.TS
import Definitions.Accounting.Assumptions.TS.Fixed

/-!
# `UnitFixed` from a fixed word (trusted)

Option A (`Definitions.Accounting.TS.Fixed`). Each `def G : Prop` is proved as `Accounting.SecurityProofs.TS.G`
(`Proofs.Accounting.TS.Fixed`), with its non-vacuity witness in `Proofs.Accounting.TS.FixedMeets`.

* `UnitFixedOfWords`: `UnitFixed` (E1(c) per unit, with its fold clause) on a word layer whose credited units are fixed
  under option A, from `ModesRun` and `CountsFixed`. A good unit (`DrawnLayer.Good`: the run paid for its late cone) is
  computed, and so is every unit steering it, since a unit steering it is in its late cone; so its words are fixed words
  and `CountsFixed` counts them. A unit not skipped whole folds its words (`ModesRun`).
* `AdditiveFixed`: additive operands don't decide fixedness. A computed unit's work (its words less what its additive
  operands add) is its work on its multiplicative operands under its noise, and when those are bound before its coin
  they are the same on every continuation from that coin: fixed by σ_{<c}, whatever its additive operands are, which may
  be written after it (an honest K-chunk's running sum reads its own round's coin).

`UnitFixed` itself stays an assumption, and its readers are unchanged; retiring it is a later step.
-/

namespace Accounting.Guarantees.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity

/-- **`UnitFixed` from option A's fixed word.** On a word layer whose units of positive weight are fixed under the
weights (option A: each multiplicative operand bound before the unit's coin, or written by a unit always proved or
priced at its public fan-out, `WordLayer.Priced`), with nonnegative weights, if the modes are the run's (`ModesRun`)
and the labelling counts fixed words as fixed (`CountsFixed`), then `UnitFixed` holds of its drawn layer, at the same
modes, weights and folded words. -/
def UnitFixedOfWords : Prop :=
  ∀ {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] {C B : Type} [Fintype C] [Fintype B]
    {S m s : ℕ} {J Q Pos V N W : Type} [Fintype J] [Nonempty V] [AddCommGroup W]
    (P : Play (Paths C m × B) S K T L U) (pf : WordLayer C B S m s J Q Pos V N W)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ),
    (∀ w j, 0 ≤ wt w j) → (∀ w j, wt w j ≠ 0 → pf.Fixed wt (w, j)) → ModesRun pf mode → CountsFixed P pf wt folded →
      UnitFixed P pf.toDrawnLayer mode wt folded

/-- **Additive operands don't decide fixedness.** For a unit `u` whose multiplicative operands are all bound before its
coin, on every path where the run computed it, its work is its work on its multiplicative operands under its noise,
and those operands are the same on every continuation from its coin (round `u.1`'s coin `coin u`): a function of the
coins before it. Its additive operands, and so its words, may differ there. -/
def AdditiveFixed : Prop :=
  ∀ {C B : Type} [Fintype C] [Fintype B] {S m s : ℕ} {J Q Pos V N W : Type} [Nonempty V] [AddCommGroup W]
    (pf : WordLayer C B S m s J Q Pos V N W) (u : Fin S × J), (∀ k, pf.role u k = .mul → pf.Bound u k) →
    ∀ x, pf.Computed x u →
      pf.work x u = pf.prod u (pf.mulView x u) (pf.noise x u) ∧
        ∀ z, pf.mulView (spliceAt u.1 (pf.coin u.1 u.2) x z) u = pf.mulView x u

end Accounting.Guarantees.TS
