module

public import Definitions.Nci.Guarantees.Formalism
public import Proofs.Core.Circuit
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The training formalism v2: Fit

A partition into `B` blocks of at most `c` gates has `|C| ≤ B c` (`card_computed_le`), so every partition into inference
blocks gives `B c / |C| ≥ 1` once `|C| > 0`. A partition into inference blocks prices `ρ` from above
(`slowdown_le_partitionSlowdown`), and one whose every block holds at least `c - d` gates has `|C| ≥ B (c - d)`, so
`B c / |C| ≤ c / (c - d)` (`partitionSlowdown_le`); `B > 0` full blocks of `c > 0` gates that read within `k` give
`ρ = 1` (`slowdown_eq_one`). Fit bounds what a compute, reduction or update application block reads beyond `θ` by the
sizes of its pieces (`compute_reads_le`, `training_reads_le`).
-/

@[expose] public section

namespace Nci.Formalism

open Finset
open Verity (RegisteredCircuit Partition)
open scoped ENNReal

theorem card_computed_le {d : RegisteredCircuit} {B c : ℕ} (P : Partition d.D B) (h : ∀ u, (P.gates u).card ≤ c) :
    d.D.computed.card ≤ B * c := by
  rw [← P.sum_card_gates]
  calc ∑ u, (P.gates u).card ≤ ∑ _u : Fin B, c := Finset.sum_le_sum fun u _ => h u
    _ = B * c := by simp

/-- A partition into inference blocks prices `ρ` from above. -/
theorem slowdown_le_partitionSlowdown {d : RegisteredCircuit} {B c k : ℕ} {P : Partition d.D B} (hP : P.IsCK d c k) :
    slowdown d c k ≤ partitionSlowdown d B c :=
  iInf_le_of_le B (iInf_le_of_le P (iInf_le_of_le hP le_rfl))

/-- Blocks of at least `c - dg` gates give `B c / |C| ≤ c / (c - dg)`. -/
theorem partitionSlowdown_le {d : RegisteredCircuit} {B c dg : ℕ} (P : Partition d.D B)
    (h : ∀ u, c - dg ≤ (P.gates u).card) : partitionSlowdown d B c ≤ (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞) := by
  have hC : B * (c - dg) ≤ d.D.computed.card := by
    rw [← P.sum_card_gates]
    calc B * (c - dg) = ∑ _u : Fin B, (c - dg) := by simp
      _ ≤ ∑ u, (P.gates u).card := Finset.sum_le_sum fun u _ => h u
  unfold partitionSlowdown
  rcases Nat.eq_zero_or_pos c with rfl | hc
  · simp
  rcases Nat.eq_zero_or_pos (c - dg) with hcd | hcd
  · rw [hcd, Nat.cast_zero, ENNReal.div_zero (by exact_mod_cast hc.ne')]
    exact le_top
  rcases Nat.eq_zero_or_pos B with rfl | hB
  · simp
  calc ((B * c : ℕ) : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)
      ≤ ((B * c : ℕ) : ℝ≥0∞) / ((B * (c - dg) : ℕ) : ℝ≥0∞) := ENNReal.div_le_div_left (by exact_mod_cast hC) _
    _ = (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞) := by
        rw [Nat.cast_mul, Nat.cast_mul,
          ENNReal.mul_div_mul_left _ _ (by exact_mod_cast hB.ne') (ENNReal.natCast_ne_top B)]

/-- Full blocks that read within `k` give `ρ = 1`. -/
theorem slowdown_eq_one {d : RegisteredCircuit} {B c k : ℕ} (P : Partition d.D B) (hb : d.D.Boolean)
    (hfull : ∀ u, (P.gates u).card = c) (hreads : ∀ u, (d.unregistered (P.gates u)).card ≤ k) (hB : 0 < B)
    (hc : 0 < c) : slowdown d c k = 1 := by
  have hC : d.D.computed.card = B * c := by
    rw [← P.sum_card_gates]
    simp [hfull]
  have hpos : (d.D.computed.card : ℝ≥0∞) ≠ 0 := by
    rw [hC]
    exact_mod_cast (Nat.mul_pos hB hc).ne'
  have hne : (d.D.computed.card : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  apply le_antisymm
  · refine (slowdown_le_partitionSlowdown (P := P) ⟨hb, fun u => (hfull u).le, hreads⟩).trans ?_
    rw [partitionSlowdown, ← hC]
    exact (ENNReal.div_self hpos hne).le
  · refine le_iInf fun B' => le_iInf fun P' => le_iInf fun hP' => ?_
    rw [partitionSlowdown, ENNReal.le_div_iff_mul_le (Or.inl hpos) (Or.inl hne), one_mul]
    exact_mod_cast card_computed_le P' hP'.card

theorem card_biUnion_le_mul {α : Type} [DecidableEq α] {m b : ℕ} (f : Fin m → Finset α) (h : ∀ i, (f i).card ≤ b) :
    (univ.biUnion f).card ≤ m * b :=
  card_biUnion_le.trans ((Finset.sum_le_sum fun i _ => h i).trans (by simp))

/-- A compute block reads at most `Δ + s τ` values beyond `θ`. -/
theorem compute_reads_le {d : RegisteredCircuit} {I : Finset (Fin d.D.N)} {Δ s τ : ℕ} (h : ComputeBlock d I Δ s τ) :
    (d.unregistered I).card ≤ Δ + s * τ := by
  obtain ⟨desc, data, hdesc, hdata, hsub⟩ := h
  calc (d.unregistered I).card ≤ (desc ∪ univ.biUnion data).card := card_le_card hsub
    _ ≤ desc.card + (univ.biUnion data).card := card_union_le _ _
    _ ≤ Δ + s * τ := Nat.add_le_add hdesc (card_biUnion_le_mul data hdata)

/-- A training block that fits reads at most `k` values beyond `θ`. -/
theorem training_reads_le {d : RegisteredCircuit} {I : Finset (Fin d.D.N)} {Δ Δt Δgrad s τ k : ℕ}
    (h : ComputeBlock d I Δ s τ ∨ ReductionBlock d I Δgrad ∨ UpdateBlock d I Δt Δgrad)
    (hfit : TrainingFits Δ Δt Δgrad s τ k) : (d.unregistered I).card ≤ k := by
  obtain ⟨h1, h2, h3⟩ := hfit
  rcases h with h | ⟨state, hstate, hsub⟩ | ⟨weights, state, hweights, hstate, hsub⟩
  · exact (compute_reads_le h).trans h1
  · calc (d.unregistered I).card ≤ (univ.biUnion state).card := card_le_card hsub
      _ ≤ 2 * Δgrad := card_biUnion_le_mul state hstate
      _ ≤ k := h2
  · calc (d.unregistered I).card ≤ (weights ∪ state).card := card_le_card hsub
      _ ≤ weights.card + state.card := card_union_le _ _
      _ ≤ Δt + Δgrad := Nat.add_le_add hweights hstate
      _ ≤ k := h3

namespace SecurityProofs

theorem FitServing : Guarantees.FitServing := by
  intro d B P c k dg Δ s τ hb hP hfit
  exact (slowdown_le_partitionSlowdown (k := k)
    ⟨hb, fun u => (hP u).2.1, fun u => (compute_reads_le (hP u).2.2).trans hfit⟩).trans
    (partitionSlowdown_le P fun u => (hP u).1)

theorem FitHonestServing : Guarantees.FitHonestServing := by
  intro d B P c k s τ hb hP hfit hB hc
  exact slowdown_eq_one P hb (fun u => le_antisymm (hP u).2.1 (by simpa using (hP u).1))
    (fun u => (compute_reads_le (hP u).2.2).trans hfit) hB hc

theorem FitTraining : Guarantees.FitTraining := by
  intro d B P c k dg Δ Δt Δgrad s τ hb hP hfit
  exact (slowdown_le_partitionSlowdown (k := k)
    ⟨hb, fun u => (hP u).2.1, fun u => training_reads_le (hP u).2.2 hfit⟩).trans
    (partitionSlowdown_le P fun u => (hP u).1)

end SecurityProofs

end Nci.Formalism
