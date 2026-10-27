import Properties.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Pous.Period
import Proofs.Pouw.Prob

/-!
# PoUS's per-period guarantee is the coins model's `.pous` input

At any value `ξ` of the period's coins, PoUS's event over its own coordinate (it accepts while the capacity account held
fewer than `S · len` bit-ticks) maps, through `RealizesSpaceOfPeriodHeld`'s coin equivalence, into `PeriodHeld`'s event
at the other coins `other ξ`, whose probability `Pous.Period.SecurityProofs.periodHeld` bounds by `δ + ε_max`. The
bound holds at the challenge times PoUS grades (`Definitions.Accounting.JointBridge.SpaceOfPeriodHeld`'s header).
-/

namespace Accounting.SecurityProofs.Bridge

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-period guarantee at a period that realizes it** (`Accounting.Bridge.SpaceOfPeriodHeld`). -/
theorem SpaceOfPeriodHeld : Accounting.Bridge.SpaceOfPeriodHeld := by
  intro _ _ hH D Q L hP ι α _ _ d C I _ Ω _ r ΩW ΩQ run P round c p hR hdev hS ξ
  have h := Pous.Period.SecurityProofs.periodHeld hH D Q L hP ι α ΩW ΩQ d run P round c.S hdev hS
    (hR.other ξ)
  refine le_trans ?_ ((Pouw.pr_equiv hR.coins _).trans_le h)
  refine Pouw.pr_mono fun x hx => ?_
  obtain ⟨ha, hn⟩ := hx
  have hA := hR.accepts _ ha
  have hH' := hR.held (Function.update ξ (r.own .pous) x)
  rw [hR.other_update, Function.update_self] at hA hH'
  refine ⟨hA.1, hA.2, ?_⟩
  have hn' : ¬ c.S * (r.win.len : ℝ) ≤ (r.held (Function.update ξ (r.own .pous) x) : ℝ) := hn
  rw [hH', hR.len] at hn'
  exact lt_of_not_ge hn'

end Accounting.SecurityProofs.Bridge
