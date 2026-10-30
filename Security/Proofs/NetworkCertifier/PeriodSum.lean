import Proofs.Flock.Claims.Prob
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# The period sum: per-window leak bounds summed over a period

A period `T` is a finite set of windows over one coin space `Ω` (the audit's draws over the period). Each window `w`
has its own leak bound: except with probability `ε w`, an accepted audit of `w` leaves the sender at most `b w` bits
through the wardens in `w` (`Pr[acc w ∧ b w < moved w] ≤ ε w`). This is the per-window guarantee's conclusion, whose
error is `δ + ε_ks + ε_bind`, taken here as a hypothesis family, so the sum doesn't depend on the per-window
statement's form. `periodLeak_le_sum` sums these bounds by a union bound (`Pr_exists_mem_le_sum`): except with
probability `Σ_w ε w`, when every window's audit is accepted, the period's total is at most `a + Σ_w b w`. The total
is bounded by the period-level terms `a` (`log2 #Ss + Σ_c log2 (cap_c + 1)` and whatever else the period charges once)
plus each window's own bits (`htotal`).
-/

namespace NetTiming

open Claims Finset

/-- **The union bound** for `Claims.Pr` over a finite family of events. -/
theorem Pr_exists_mem_le_sum {Ω ι : Type} [Fintype Ω] (T : Finset ι) (p : ι → Ω → Prop) :
    Pr (fun ω => ∃ w ∈ T, p w ω) ≤ ∑ w ∈ T, Pr (p w) := by
  classical
  unfold Pr
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

/-- **The period sum.** Over a period `T` of windows on one coin space, if each window's accepted audit leaves more
than `b w` bits moved in it with probability at most `ε w`, and when every window's audit is accepted the period's
total is at most the period-level terms `a` plus the windows' own bits, then every window's audit is accepted and the
period's total exceeds `a + Σ_w b w` with probability at most `Σ_w ε w`. -/
theorem periodLeak_le_sum {Ω ι : Type} [Fintype Ω] (T : Finset ι) (acc : ι → Ω → Prop) (moved : ι → Ω → ℝ)
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
