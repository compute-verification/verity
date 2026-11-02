import Definitions.Nci.Guarantees.Formalism
import Proofs.Core.Circuit
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The training formalism v1: Fit

A partition into `B` blocks of at most `c` gates has `|C| ≤ B c` (`card_computed_le`), so every `(c, k)`-partition gives
`B c / |C| ≥ 1` once `|C| > 0`; `B > 0` full blocks of `c > 0` gates that read within `k` have `|C| = B c` and give
`ρ = 1` (`slowdown_eq_one`). Fit bounds what a session or tree block reads beyond `θ` by the sizes of its pieces.
-/

namespace Nci.Formalism

open Finset
open Verity (Declaration Partition)
open scoped ENNReal

theorem card_computed_le {d : Declaration} {B c : ℕ} (P : Partition d.D B) (h : ∀ u, (P.gates u).card ≤ c) :
    d.D.computed.card ≤ B * c := by
  rw [← P.sum_card_gates]
  calc ∑ u, (P.gates u).card ≤ ∑ _u : Fin B, c := Finset.sum_le_sum fun u _ => h u
    _ = B * c := by simp

/-- Full blocks that read within `k` give `ρ = 1`. -/
theorem slowdown_eq_one {d : Declaration} {B c k : ℕ} (P : Partition d.D B) (hfull : ∀ u, (P.gates u).card = c)
    (hreads : ∀ u, (d.unregistered (P.gates u)).card ≤ k) (hB : 0 < B) (hc : 0 < c) : slowdown d c k = 1 := by
  have hC : d.D.computed.card = B * c := by
    rw [← P.sum_card_gates]
    simp [hfull]
  have hpos : (d.D.computed.card : ℝ≥0∞) ≠ 0 := by
    rw [hC]
    exact_mod_cast (Nat.mul_pos hB hc).ne'
  have hne : (d.D.computed.card : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  unfold slowdown
  apply le_antisymm
  · refine iInf_le_of_le B (iInf_le_of_le P (iInf_le_of_le ⟨fun u => (hfull u).le, hreads⟩ ?_))
    rw [← hC]
    exact (ENNReal.div_self hpos hne).le
  · refine le_iInf fun B' => le_iInf fun P' => le_iInf fun hP' => ?_
    rw [ENNReal.le_div_iff_mul_le (Or.inl hpos) (Or.inl hne), one_mul]
    exact_mod_cast card_computed_le P' hP'.1

theorem card_biUnion_le_mul {α : Type} [DecidableEq α] {m b : ℕ} (f : Fin m → Finset α) (h : ∀ i, (f i).card ≤ b) :
    (univ.biUnion f).card ≤ m * b :=
  card_biUnion_le.trans ((Finset.sum_le_sum fun i _ => h i).trans (by simp))

namespace SecurityProofs

theorem FitServing : Guarantees.FitServing := by
  intro d B P c k Δ s τ hP hB hc hfit
  refine slowdown_eq_one P (fun u => (hP u).1) (fun u => ?_) hB hc
  obtain ⟨-, desc, data, hdesc, hdata, hsub⟩ := hP u
  calc (d.unregistered (P.gates u)).card ≤ (desc ∪ univ.biUnion data).card := card_le_card hsub
    _ ≤ desc.card + (univ.biUnion data).card := card_union_le _ _
    _ ≤ Δ + s * τ := Nat.add_le_add hdesc (card_biUnion_le_mul data hdata)
    _ ≤ k := hfit

theorem FitHonestServing : Guarantees.FitHonestServing := fun d B P c k s τ hP hB hc hfit =>
  FitServing d B P c k 0 s τ hP hB hc hfit

theorem FitTraining : Guarantees.FitTraining := by
  intro d B P c k Δ Δgrad s τ hP hB hc hfit
  refine slowdown_eq_one P (fun u => (hP u).1) (fun u => ?_) hB hc
  obtain ⟨-, desc, state, data, hdesc, hstate, hdata, hsub⟩ := hP u
  calc (d.unregistered (P.gates u)).card
        ≤ (desc ∪ univ.biUnion state ∪ univ.biUnion data).card := card_le_card hsub
    _ ≤ (desc ∪ univ.biUnion state).card + (univ.biUnion data).card := card_union_le _ _
    _ ≤ desc.card + (univ.biUnion state).card + (univ.biUnion data).card :=
        Nat.add_le_add_right (card_union_le _ _) _
    _ ≤ Δ + 2 * Δgrad + s * τ :=
        Nat.add_le_add (Nat.add_le_add hdesc (card_biUnion_le_mul state hstate)) (card_biUnion_le_mul data hdata)
    _ ≤ k := hfit

end SecurityProofs

end Nci.Formalism
