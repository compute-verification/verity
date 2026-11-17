module

public import Definitions.Core.Interaction
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob
import Std.Tactic.BVDecide.Normalize.Prop

/-! The period sum: per-window leak bounds summed over a period by a union bound. -/

@[expose] public section

namespace NetTiming

open Claims Finset

/-- **The union bound** for `Claims.Pr` over a finite family of events. -/
theorem Pr_exists_mem_le_sum {Ω ι : Type} [Fintype Ω] (T : Finset ι) (p : ι → Ω → Prop) :
    Pr (fun ω => ∃ w ∈ T, p w ω) ≤ ∑ w ∈ T, Pr (p w) := by
  classical
  unfold Pr Claims.prUniform
  rw [← Finset.sum_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  have hsub : (Finset.univ.filter fun ω => ∃ w ∈ T, p w ω) ⊆ T.biUnion fun w => Finset.univ.filter (p w) := by
    intro ω hω
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_biUnion] at hω ⊢
    exact hω
  have hcard : (((Finset.univ.filter fun ω => ∃ w ∈ T, p w ω).card : ℕ) : ℝ) ≤
      ((∑ w ∈ T, (Finset.univ.filter (p w)).card : ℕ) : ℝ) :=
    Nat.cast_le.2 ((Finset.card_le_card hsub).trans Finset.card_biUnion_le)
  push_cast at hcard
  convert hcard

theorem monitoringPeriodLeak_le_sum {Ω ι : Type} [Fintype Ω] (T : Finset ι) (acc : ι → Ω → Prop) (moved : ι → Ω → ℝ)
    (b ε : ι → ℝ) (total : Ω → ℝ) (a : ℝ)
    (htotal : ∀ ω, (∀ w ∈ T, acc w ω) → total ω ≤ a + ∑ w ∈ T, moved w ω)
    (hwin : ∀ w ∈ T, Pr (fun ω => acc w ω ∧ b w < moved w ω) ≤ ε w) :
    Pr (fun ω => (∀ w ∈ T, acc w ω) ∧ a + ∑ w ∈ T, b w < total ω) ≤ ∑ w ∈ T, ε w := by
  calc Pr (fun ω => (∀ w ∈ T, acc w ω) ∧ a + ∑ w ∈ T, b w < total ω)
      ≤ Pr (fun ω => ∃ w ∈ T, acc w ω ∧ b w < moved w ω) := by
        refine Pr_mono fun ω h => ?_
        obtain ⟨hacc, hlt⟩ := h
        by_contra hno
        have hle : ∑ w ∈ T, moved w ω ≤ ∑ w ∈ T, b w :=
          Finset.sum_le_sum fun w hw => le_of_not_gt fun hgt => hno ⟨w, hw, hacc w hw, hgt⟩
        linarith [htotal ω hacc]
    _ ≤ ∑ w ∈ T, Pr (fun ω => acc w ω ∧ b w < moved w ω) := Pr_exists_mem_le_sum T _
    _ ≤ ∑ w ∈ T, ε w := Finset.sum_le_sum hwin

end NetTiming
