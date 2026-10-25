import Models.Pouw.Assumptions.Pinned
import Proofs.Pouw.Prob
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Models.Pouw.Guarantees.Game

/-!
# (c) Work-weighted sampling: `δ_s = (1 − ε_s)^t`
-/

namespace Pouw.Proofs

open Finset

variable {w : ℕ → ℕ} {N : ℕ}

theorem card_ticket (w : ℕ → ℕ) (N : ℕ) : Fintype.card (Ticket w N) = ∑ u ∈ range N, w u := by
  rw [Fintype.card_sigma]
  simp only [Fintype.card_fin]
  exact Fin.sum_univ_eq_sum_range w N

theorem card_good_tickets (K : Finset ℕ) :
    ((univ : Finset (Ticket w N)).filter fun τ => ((τ.1 : ℕ) ∈ K)).card
      = ∑ u ∈ range N, if u ∈ K then w u else 0 := by
  classical
  have h : ((univ : Finset (Ticket w N)).filter fun τ => ((τ.1 : ℕ) ∈ K))
      = ((univ : Finset (Fin N)).filter fun u : Fin N => (u : ℕ) ∈ K).sigma fun _ => univ := by
    ext τ; simp
  rw [h, Finset.card_sigma]
  simp only [Finset.card_univ, Fintype.card_fin]
  rw [Finset.sum_filter]
  exact Fin.sum_univ_eq_sum_range (fun u => if u ∈ K then w u else 0) N

theorem sum_split (K : Finset ℕ) :
    (∑ u ∈ range N, if u ∈ K then w u else 0) + ∑ u ∈ range N \ K, w u = ∑ u ∈ range N, w u := by
  classical
  rw [Finset.sdiff_eq_filter, Finset.sum_filter, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun u _ => ?_
  split_ifs <;> simp

/-- One work-weighted sample lands in `K` with probability `(∑_{u ∈ K} w u)/∑_u w u`; `t` independent samples all
do with that to the `t`. -/
theorem pr_all_in (K : Finset ℕ) (t : ℕ) :
    pr (fun c : Fin t → Ticket w N => AuditAccepts K c)
      = (((∑ u ∈ range N, if u ∈ K then w u else 0 : ℕ) : ℝ) / (∑ u ∈ range N, w u : ℕ)) ^ t := by
  classical
  rw [pr_eq_card _ (Fintype.piFinset fun _ => (univ : Finset (Ticket w N)).filter fun τ => ((τ.1 : ℕ) ∈ K))
    (fun c => by simp [Fintype.mem_piFinset, AuditAccepts])]
  rw [Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Fintype.card_fun,
    Fintype.card_fin, card_good_tickets, card_ticket]
  push_cast
  rw [div_pow]

/-- **(c)** `Pr[accept ∧ ∑_{u∉K} w(u) > ε_s·∑_u w(u)] ≤ (1 − ε_s)^t`. -/
theorem workWeightedSampling : Pouw.Guarantees.WorkWeightedSampling := by
  intro α _ w N t K εs hεs
  have hp : (0 : ℝ) ≤ (1 - εs) ^ t := pow_nonneg (by linarith) t
  refine pr_prod_le _ _ hp fun a => ?_
  by_cases hbad : εs * ∑ u ∈ range N, (w u : ℝ) < ∑ u ∈ range N \ K a, (w u : ℝ)
  · refine (pr_mono (P' := fun c : Fin t → Ticket w N => AuditAccepts (K a) c) fun c h => h.1).trans ?_
    rw [pr_all_in]
    have hsplit := sum_split (w := w) (N := N) (K a)
    have hsplit' : ((∑ u ∈ range N, if u ∈ K a then w u else 0 : ℕ) : ℝ) + ∑ u ∈ range N \ K a, (w u : ℝ)
        = ∑ u ∈ range N, (w u : ℝ) := by exact_mod_cast hsplit
    have hWpos : (0 : ℝ) < ∑ u ∈ range N, (w u : ℝ) := by
      have hsub : ∑ u ∈ range N \ K a, (w u : ℝ) ≤ ∑ u ∈ range N, (w u : ℝ) :=
        Finset.sum_le_sum_of_subset_of_nonneg Finset.sdiff_subset fun _ _ _ => by positivity
      rcases lt_or_ge 0 εs with h | h
      · nlinarith [hsub]
      · have : 0 ≤ ∑ u ∈ range N \ K a, (w u : ℝ) := Finset.sum_nonneg fun _ _ => by positivity
        by_contra hW
        push Not at hW
        have h0 : ∑ u ∈ range N, (w u : ℝ) = 0 := le_antisymm hW (Finset.sum_nonneg fun _ _ => by positivity)
        rw [h0] at hbad hsub
        linarith
    apply pow_le_pow_left₀ (by positivity)
    rw [Nat.cast_sum (s := range N) (f := w), div_le_iff₀ hWpos]
    linarith
  · refine le_of_eq_of_le (pr_eq_zero fun c h => hbad h.2) hp

end Pouw.Proofs
