import Definitions.Accounting.JointBridge.SpaceOfPeriodHeld
import Proofs.Pous.Period
import Proofs.Pouw.Prob

/-!
# PoUS's per-period guarantee is `JointPeriod`'s `.pous` input

At any value `ξ` of the period's coins, PoUS's event over its own coordinate (it accepts while the capacity account held
fewer than `S · len` bit-ticks) maps, through `RealizesSpaceOfPeriodHeld`'s coin equivalence, into `PeriodHeld`'s event
at the other coins `other ξ`, whose probability `Pous.Period.SecurityProofs.periodHeld` bounds by `δ + ε_max`. The
bound holds at the challenge times PoUS grades (`Definitions.Accounting.JointBridge.SpaceOfPeriodHeld`'s header).
-/

namespace Accounting.SecurityProofs.Bridge

open Pous Pous.EndToEnd PousCheck PousCheck.Properties Pous.Period
open scoped ENNReal

set_option exponentiation.threshold 20000

/-- **PoUS's per-period guarantee at a period that realizes it.** Under `PeriodHeld`'s premises (`AcceptsDecodes`,
`PublishesRoot`, `CoinCommitHides`, `P2RingAuditSecure`) and its `RingDevice` on the joint run the period realizes,
with the certificate within the ring game's state bound, PoUS's audit of the period has `PeriodRun.Sound` at error
`δ + ε_max` (`1% + 2^-128`). -/
theorem SpaceOfPeriodHeld (hd : AcceptsDecodes) (hp : PublishesRoot)
    (hH : Pous.P2.Assumptions.CoinCommitHides coinView ((2⁻¹ : ℝ≥0∞) ^ 131))
    (D Q L : ℕ) (hP : Pous.P2.Assumptions.P2RingAuditSecure D Q L)
    {ι α : Type} [Fintype ι] [Fintype α] {d : Device ι} {C : Verity.Circuit} {I : Type} [DecidableEq I]
    {Ω : I → Type} [∀ i, Fintype (Ω i)] (r : PeriodRun d α C Ω) {ΩW ΩQ : Type}
    (run : Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (round : ℕ) (c : Certificates) (p : Price)
    (hR : RealizesSpaceOfPeriodHeld r run (fun ω => P.Accepts ω ∧ ¬ P.Break ω))
    (hdev : RingDevice run P D Q L round)
    (hS : c.S * (run.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ Pous.EndToEnd.S) :
    r.Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) := by
  intro ξ
  have h := Pous.Period.SecurityProofs.periodHeld hd hp hH D Q L hP ι α ΩW ΩQ d run P round c.S hdev hS
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
