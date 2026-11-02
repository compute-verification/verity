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
  intro _ _ D Q L hP ι α _ _ d C I _ Ω _ r ΩW ΩQ run P τ round c p hR hdev hinst hS ξ
  have h := Pous.Period.SecurityProofs.periodHeld D Q L hP ι α ΩW ΩQ d run P τ round c.S hdev hinst hS
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

/-- **`SpaceOfPeriodHeld` holds at the honest instance**: its premises on the code, setup's `AcceptsDecodes` and
`PublishesRoot`, are proved, and its device premises hold at the lookup strategy on a run whose capacity account holds
all of `C`'s bits at a tick of the capacity window on every coin (`Pous.Period.SecurityProofs.periodHeld_honest`). -/
theorem SpaceOfPeriodHeld_honest {ι : Type} [Fintype ι] {d : Accounting.Device ι} {α ΩW ΩQ : Type} [Fintype α]
    (run : Accounting.Run d α ΩW ΩQ ((ringRO (2 ^ 19) 16448 (2 ^ 16448 - 21065)).Ω × HmCoins × Streams))
    (P : Audits ΩW ΩQ) (D Q L round : ℕ) (hs : P.sid.size = 32)
    (hfull : ∀ ω, ∃ t ∈ (run.win .capacity).ticks, FULL ≤ (run.schedule ω t).mem (run.acct .capacity)) :
    AcceptsDecodes ∧ PublishesRoot ∧ DeviceModel run P (Pous.Period.SecurityProofs.lookup ΩW ΩQ) D Q L round ∧
      Instantiates P (Pous.Period.SecurityProofs.lookup ΩW ΩQ) :=
  ⟨PousCheck.Properties.SecurityProofs.acceptsDecodes, PousCheck.Properties.SecurityProofs.publishesRoot,
    Pous.Period.SecurityProofs.periodHeld_honest run P D Q L round hs hfull⟩

/-- **`SpaceOfPeriodHeld` is nontrivial**: a period whose PoUS verdict accepts every coin while its claim fails is not
`Sound` at `δ + ε_max`. -/
theorem SpaceOfPeriodHeld_nontrivial {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]
    {C : Verity.Circuit} {I : Type} [DecidableEq I] {Ω : I → Type} [∀ i, Fintype (Ω i)] [∀ i, Nonempty (Ω i)]
    (r : Accounting.PeriodRun d α C Ω) (c : Accounting.Certificates) (p : Accounting.Price)
    (h : ∀ ξ, r.accepts .pous ξ ∧ ¬ r.Holds c p .pous ξ) :
    ¬ r.Sound c p .pous ((Params.δ : ℝ) + Params.εMax.toReal) := by
  intro hs
  obtain ⟨ξ⟩ : Nonempty ((i : I) → Ω i) := inferInstance
  have h1 : Pouw.pr (fun x : Ω (r.own .pous) =>
      r.accepts .pous (Function.update ξ (r.own .pous) x) ∧
        ¬ r.Holds c p .pous (Function.update ξ (r.own .pous) x)) = 1 :=
    Pouw.pr_eq_one fun x => h (Function.update ξ (r.own .pous) x)
  exact not_le.2 Pous.Period.SecurityProofs.bound_toReal_lt_one (h1.symm.trans_le (hs ξ))

end Accounting.SecurityProofs.Bridge
