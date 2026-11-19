module

public import Proofs.Spot.Sampling.AmountRate
import Mathlib.Tactic

/-!
# What a scheme gets back over a period at a rate

`Scheme.getsBackPeriod_rate` proves `Scheme.GetsBackPeriodRate` from `Adaptive.rate_budget` and the scheme's charge
inequality. Where every phase is accepted with its log meeting the conditions and none breaks, the charge inequality
summed over the phases puts the period's `over` above its `under` and `T` charges by at most the cost of the phases'
incorrect blocks; above `⌊ln(1/δ) · DEN/NUM⌋` that cost has probability at most `δ` plus the rounds' drawn-unit terms by
the rate over adaptive rounds, read through the period's run tree (`hplay`).
-/

@[expose] public section

namespace Verity.Scheme

open Finset Spot.Sampling.Adaptive
open scoped ENNReal

/-- Some phase breaks with probability at most `T · δscheme`, by the union bound over the phases. -/
theorem pr_exists_breaks_le (S : Scheme) {R : Type} (T : ℕ) (phase : Fin T → R → S.Run) (pr : (R → Prop) → ℝ≥0∞)
    (hmono : ∀ p q : R → Prop, (∀ r, p r → q r) → pr p ≤ pr q)
    (hunion : ∀ p q : R → Prop, pr (fun r => p r ∨ q r) ≤ pr p + pr q) (hempty : pr (fun _ => False) = 0)
    (hbreak : ∀ t, pr (fun r => S.breaks (phase t r)) ≤ S.δscheme) :
    pr (fun r => ∃ t, S.breaks (phase t r)) ≤ T * S.δscheme := by
  classical
  have key : ∀ s : Finset (Fin T), pr (fun r => ∃ t ∈ s, S.breaks (phase t r)) ≤ s.card * S.δscheme := by
    intro s
    induction s using Finset.induction_on with
    | empty =>
      refine (hmono _ (fun _ => False) ?_).trans (by simp [hempty])
      rintro r ⟨t, ht, _⟩
      exact absurd ht (Finset.notMem_empty t)
    | insert a s ha ih =>
      refine (hmono _ (fun r => S.breaks (phase a r) ∨ ∃ t ∈ s, S.breaks (phase t r)) ?_).trans ?_
      · rintro r ⟨t, ht, hb⟩
        rcases Finset.mem_insert.1 ht with rfl | ht
        · exact Or.inl hb
        · exact Or.inr ⟨t, ht, hb⟩
      · refine (hunion _ _).trans ?_
        rw [Finset.card_insert_of_notMem ha, Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_comm]
        exact add_le_add ih (hbreak a)
  have h := key Finset.univ
  rw [Finset.card_univ, Fintype.card_fin] at h
  exact (hmono _ _ fun r ⟨t, hb⟩ => ⟨t, Finset.mem_univ t, hb⟩).trans h

/-- `exp(−(NUM/DEN) · (⌊ln(1/δ) · DEN/NUM⌋ + 1)) ≤ δ` for `δ > 0`. -/
theorem ofReal_exp_floor_succ_le {num den : ℕ} (hnum : 0 < num) (hden : 0 < den) {δ : ℝ≥0∞} (hδ : 0 < δ) :
    ENNReal.ofReal (Real.exp (-((num : ℝ) / den * ((⌊Real.log δ.toReal⁻¹ * den / num⌋₊ : ℕ) + 1)))) ≤ δ := by
  by_cases htop : δ = ⊤
  · rw [htop]; exact le_top
  have hδr : 0 < δ.toReal := ENNReal.toReal_pos hδ.ne' htop
  have hnum' : (0 : ℝ) < num := by exact_mod_cast hnum
  have hden' : (0 : ℝ) < den := by exact_mod_cast hden
  have hlt : Real.log δ.toReal⁻¹ * den / num < (⌊Real.log δ.toReal⁻¹ * den / num⌋₊ : ℝ) + 1 :=
    Nat.lt_floor_add_one _
  have hlog : Real.log δ.toReal⁻¹ ≤ (num : ℝ) / den * ((⌊Real.log δ.toReal⁻¹ * den / num⌋₊ : ℕ) + 1) := by
    rw [div_lt_iff₀ hnum'] at hlt
    rw [div_mul_eq_mul_div, le_div_iff₀ hden']
    nlinarith
  calc ENNReal.ofReal (Real.exp (-((num : ℝ) / den * ((⌊Real.log δ.toReal⁻¹ * den / num⌋₊ : ℕ) + 1))))
      ≤ ENNReal.ofReal (Real.exp (-Real.log δ.toReal⁻¹)) :=
        ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2 (neg_le_neg hlog))
    _ = δ := by
        rw [Real.log_inv, neg_neg, Real.exp_log hδr, ENNReal.ofReal_toReal htop]

/-- **What a scheme gets back over a period at a rate.** The period's run tree `P` is one prover strategy's
(`Adaptive.Play`): each round's design and incorrect blocks fixed by the history before its coin, so the phases'
provers adapt to one another, and a rejected phase ends it. Every round's design meets the rate `NUM/DEN` at
amounts `a` (`hL`), every round's drawn-unit bound is `err` (`hD`), and the period's runs are the tree's: every phase
accepted, none breaking, with the phases' incorrect blocks costing at least `M` in all, is at most as likely as the
tree's accepting with harm at least `M` at amounts `a` (`hplay`). -/
theorem getsBackPeriod_rate (S : Scheme) {R : Type} (T : ℕ) (phase : Fin T → R → S.Run) (pr : (R → Prop) → ℝ≥0∞)
    (hmono : ∀ p q : R → Prop, (∀ r, p r → q r) → pr p ≤ pr q)
    (hunion : ∀ p q : R → Prop, pr (fun r => p r ∨ q r) ≤ pr p + pr q) (hempty : pr (fun _ => False) = 0)
    (hbreak : ∀ t, pr (fun r => S.breaks (phase t r)) ≤ S.δscheme)
    (num den : ℕ) (hnum : 0 < num) (hden : 0 < den) {n N : ℕ} (a : Fin n → ℕ) (P : Play n N) {err : ℕ → ℝ≥0∞}
    (hL : P.SamplingDesigns fun L => L.Design fun u => (a u : ℝ) * num / den) (hD : P.Drawn err)
    (hplay : ∀ M : ℝ, pr (fun r => (∀ t, S.accepts (phase t r)) ∧ (∀ t, ¬ S.breaks (phase t r)) ∧
        M ≤ ∑ t, ∑ b ∈ S.incorrect (phase t r), (S.cost b : ℝ)) ≤
      P.prEv (fun u => (a u : ℝ)) (fun acc x => acc = true ∧ M ≤ x) 0)
    {δ : ℝ≥0∞} (hδ : 0 < δ) :
    S.GetsBackPeriodRate T phase pr num den δ (∑ m ∈ range N, err m) := by
  classical
  unfold Scheme.GetsBackPeriodRate
  set F := ⌊Real.log δ.toReal⁻¹ * den / num⌋₊ with hF
  set M : ℝ := (F : ℝ) + 1 with hM
  refine (hmono _ (fun r => (∃ t, S.breaks (phase t r)) ∨ ((∀ t, S.accepts (phase t r)) ∧
      (∀ t, ¬ S.breaks (phase t r)) ∧ M ≤ ∑ t, ∑ b ∈ S.incorrect (phase t r), (S.cost b : ℝ))) ?_).trans ?_
  · rintro r ⟨hall, hlt⟩
    by_cases hb : ∃ t, S.breaks (phase t r)
    · exact Or.inl hb
    · push_neg at hb
      refine Or.inr ⟨fun t => (hall t).1, hb, ?_⟩
      have hc : ∀ t, S.over (phase t r) ≤
          S.under (phase t r) + ∑ b ∈ S.incorrect (phase t r), S.cost b + S.charges :=
        fun t => S.charge _ (hall t).1 (hall t).2 (hb t)
      have hsum : ∑ t, S.over (phase t r) ≤
          ∑ t, S.under (phase t r) + ∑ t, ∑ b ∈ S.incorrect (phase t r), S.cost b + T * S.charges := by
        calc ∑ t, S.over (phase t r)
            ≤ ∑ t, (S.under (phase t r) + ∑ b ∈ S.incorrect (phase t r), S.cost b + S.charges) :=
              Finset.sum_le_sum fun t _ => hc t
          _ = _ := by
              rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
                Fintype.card_fin, smul_eq_mul]
      have hnat : F + 1 ≤ ∑ t, ∑ b ∈ S.incorrect (phase t r), S.cost b := by omega
      have hcast : ((∑ t, ∑ b ∈ S.incorrect (phase t r), S.cost b : ℕ) : ℝ) =
          ∑ t, ∑ b ∈ S.incorrect (phase t r), (S.cost b : ℝ) := by push_cast; rfl
      rw [hM, ← hcast]
      exact_mod_cast hnat
  · have hrate := rate_budget a num den M P hL hD
    have hexp : ENNReal.ofReal (Real.exp (-((num : ℝ) / den * M))) ≤ δ := by
      rw [hM, hF]; exact ofReal_exp_floor_succ_le hnum hden hδ
    calc pr (fun r => (∃ t, S.breaks (phase t r)) ∨ ((∀ t, S.accepts (phase t r)) ∧
          (∀ t, ¬ S.breaks (phase t r)) ∧ M ≤ ∑ t, ∑ b ∈ S.incorrect (phase t r), (S.cost b : ℝ)))
        ≤ pr (fun r => ∃ t, S.breaks (phase t r)) + pr (fun r => (∀ t, S.accepts (phase t r)) ∧
          (∀ t, ¬ S.breaks (phase t r)) ∧ M ≤ ∑ t, ∑ b ∈ S.incorrect (phase t r), (S.cost b : ℝ)) := hunion _ _
      _ ≤ T * S.δscheme + (δ + ∑ m ∈ range N, err m) :=
          add_le_add (pr_exists_breaks_le S T phase pr hmono hunion hempty hbreak)
            ((hplay M).trans (hrate.trans (add_le_add hexp le_rfl)))
      _ = δ + ∑ m ∈ range N, err m + T * S.δscheme := by rw [add_comm]

end Verity.Scheme
