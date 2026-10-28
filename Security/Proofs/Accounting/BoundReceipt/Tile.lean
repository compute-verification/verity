import Mathlib
import Definitions.Accounting.Guarantees.BoundReceipt
import Proofs.Pouw.Prob

/-!
# A tile's output is unpredictable below its work: from PoUW's named assumptions

A checked unit whose credit is above `T/(1 − γ₀)` puts the checked units' total credit above it, when every credit is
nonnegative; `TT` and `TT_NCP_U` bound the probability of the second.
-/

namespace Accounting.SecurityProofs

open Accounting.Guarantees

/-- `TileUnpredictable` from a bound on the checked units' total credit, for nonnegative credits. -/
theorem tileUnpredictable_of_total {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : Pouw.CostModel Q R S) (P : Pouw.Protocol Q R S) (credit : Pouw.Shape → ℝ) (hc : ∀ sh, 0 ≤ credit sh)
    (D : Pouw.UnitLayout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ)
    (h : ∀ U : Pouw.Workload, U.InDomain D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Pouw.Adversary CM,
      A.Bounded U T q →
        Pouw.pr (fun x : (Q → R) × S =>
          T / (1 - γ₀) < ∑ u ∈ Pouw.checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), credit (U.layout.shape u))
          ≤ ε q U.layout.N) :
    TileUnpredictable CM P credit D γ₀ ε := by
  intro U hU T hT q hq A hA u hu
  refine (Pouw.pr_mono (fun x hx => ?_)).trans (h U hU T hT q hq A hA)
  exact hu.trans_le (Finset.single_le_sum (f := fun u => credit (U.layout.shape u)) (fun i _ => hc _) hx)

/-- **`TT` gives `TileUnpredictable` at `m·k·n`** (`Accounting.Guarantees.TileUnpredictableOfTT`). -/
theorem TileUnpredictableOfTT : Accounting.Guarantees.TileUnpredictableOfTT :=
  fun _ _ _ _ _ _ _ CM P D γ₀ ε hTT =>
    tileUnpredictable_of_total CM P _ (fun _ => Nat.cast_nonneg _) D γ₀ ε hTT.2

/-- **`TT_NCP_U` gives `TileUnpredictable` at `3·m·k·n`** (`Accounting.Guarantees.TileUnpredictableOfNCPU`). -/
theorem TileUnpredictableOfNCPU : Accounting.Guarantees.TileUnpredictableOfNCPU :=
  fun β _ _ _ _ _ _ _ CM P D γ₀ ε hside hTT =>
    tileUnpredictable_of_total CM (P.toProtocolU β) _ (fun _ => by positivity) D γ₀ ε (hTT hside).2

/-- **More compared entries and less credit keep `TileUnpredictable`** (`Accounting.Guarantees.TileUnpredictableMono`):
a unit whose checked values are right on `P'`'s entries is right on `P`'s. -/
theorem TileUnpredictableMono : Accounting.Guarantees.TileUnpredictableMono := by
  classical
  intro Q R S _ _ _ _ CM P P' credit credit' D γ₀ ε hY hidx hcr h U hU T hT q hq A hA u hu
  refine (Pouw.pr_mono (fun x hx => ?_)).trans (h U hU T hT q hq A hA u (hu.trans_le (hcr _)))
  rw [Pouw.checkedSet, Finset.mem_filter] at hx ⊢
  obtain ⟨hr, c, hc, hact, hck⟩ := hx
  exact ⟨hr, c, hc, hact, fun y hy => by rw [← hY]; exact hck y (hidx _ hy)⟩

/-- **`TT_OUT`'s shape at W1 gives `TileUnpredictablePearlC` at W1**
(`Accounting.Guarantees.TileUnpredictablePearlCOfTTOut`). -/
theorem TileUnpredictablePearlCOfTTOut : Accounting.Guarantees.TileUnpredictablePearlCOfTTOut := by
  classical
  intro Q R S _ _ _ _ dv salt P credit D γ₀ ε hc hTT U hU T hT q hq A hA u hu
  refine (Pouw.pr_mono (fun x hx => ?_)).trans (hTT U hU T hT q hq A hA)
  have hcorrect : ∀ v ∈ Pouw.PearlC.correctSet P U x.1 x.2 (A.transcript U x.1 x.2), ∃ c : Pouw.PearlC.Commit,
      Pouw.PearlC.actOf (A.transcript U x.1 x.2) v = c.act ∧ Pouw.PearlC.Correct P U x.1 x.2 v c := by
    intro v hv
    rw [Pouw.PearlC.correctSet, Finset.mem_filter] at hv
    obtain ⟨-, c, hτ, hcor⟩ := hv
    exact ⟨c, by simp [Pouw.PearlC.actOf, hτ], hcor⟩
  obtain ⟨c, hact, hcor⟩ := hcorrect u hx
  refine hu.trans_le ((hc x.1 x.2 U u c hcor).2.trans ?_)
  rw [← hact]
  refine Finset.single_le_sum
    (f := fun v => P.credit x.1 x.2 U v (Pouw.PearlC.actOf (A.transcript U x.1 x.2) v)) (fun v hv => ?_) hx
  obtain ⟨c', hact', hcor'⟩ := hcorrect v hv
  simpa only [hact'] using (hc x.1 x.2 U v c' hcor').1

end Accounting.SecurityProofs
