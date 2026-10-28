import Definitions.Accounting.Guarantees.TS
import Proofs.Pouw.RelativeSecurity

/-!
# The tensor–scalar model's round lemmas

The plays' tail bounds: one bet a round under (L), the deficiency against the deficient rounds, and SAMPLING's budget
(EPS1-SPEC §2.4, §2.7). They are PoUW's round bounds
(`Pouw.RelativeSecurity`): `luck_rounds` conditions on the first round's coins (`prob_eq_expect`, `expect_prod`) as
`expect_weight_le` does, and `budget_tail` is `prob_accepts_le` at `c = 1/Λ`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Accounting.Guarantees.TS Pouw.RelativeSecurity

/-! The helpers sit in `Rounds` so that they can't clash with a name in the file that imports this one. -/

namespace Rounds

theorem expect_add {Ω : Type*} [Fintype Ω] (f g : Ω → ℝ) :
    expect (fun ω => f ω + g ω) = expect f + expect g := by
  unfold expect
  rw [Finset.sum_add_distrib, add_div]

theorem prob_false {Ω : Type*} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

/-- An indicator below an event's: its expectation is at most the event's probability. -/
theorem expect_ind_le_prob {Ω : Type*} [Fintype Ω] {P Q : Ω → Prop} {inst : DecidablePred P}
    (h : ∀ ω, P ω → Q ω) : expect (fun ω => if P ω then (1 : ℝ) else 0) ≤ prob Q := by
  classical
  rw [prob_eq_expect Q]
  refine expect_mono fun ω => ?_
  by_cases hP : P ω
  · rw [ite_eq_left hP, ite_eq_left (h ω hP)]
  · rw [ite_eq_right hP]
    split_ifs <;> norm_num

/-- `Λ` bounding the play is PoUW's exponential round bound at `c = 1/Λ`. -/
theorem expBound_of_budgetAt {Ω : Type} [Fintype Ω] {Λ : ℝ} :
    ∀ {n : ℕ} (g : RoundGame Ω n), RoundGame.BudgetAt Λ g → g.ExpBound (1 / Λ)
  | _, .done, _ => trivial
  | _, .round d pass next, h => by
    classical
    obtain ⟨-, hsum, hnext⟩ := h
    refine ⟨?_, fun a => expBound_of_budgetAt (next a) (hnext a)⟩
    unfold expect
    rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
    · simp [h0]
    rw [div_le_one (by exact_mod_cast h0)]
    refine le_of_eq_of_le (Finset.sum_congr rfl fun ω _ => ?_) hsum
    dsimp only
    split_ifs
    · rw [one_div_mul_eq_div]
    · rfl

end Rounds

/-- D-ideal's step 2, one bet a round: under (L) at `p`, the ideal game accepts with at least `k` deficient rounds
with probability at most `p^k`. -/
theorem luck_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ) (hp : 0 ≤ p)
    (hL : RoundGame.LuckAt p g) (k : ℕ) :
    prob (fun x => g.accepts x ∧ k ≤ Accounting.TS.RoundGame.deficientRounds g x) ≤ p ^ k := by
  classical
  induction g generalizing k with
  | done =>
    rcases k with _ | j
    · rw [pow_zero]; exact prob_le_one _
    · refine (prob_mono (q := fun _ => False) fun x hx => ?_).trans (Rounds.prob_false.trans_le (pow_nonneg hp _))
      simp [Accounting.TS.RoundGame.deficientRounds] at hx
  | round d pass next ih =>
    obtain ⟨-, hpass, hnext⟩ := hL
    rcases k with _ | j
    · rw [pow_zero]; exact prob_le_one _
    rw [prob_eq_expect, expect_prod]
    calc _ ≤ expect (fun a => p ^ j * (if pass a ∧ 0 < d a then (1 : ℝ) else 0) +
            p ^ (j + 1) * (if ¬0 < d a then (1 : ℝ) else 0)) := expect_mono fun a => ?_
      _ = p ^ j * prob (fun a => pass a ∧ 0 < d a) + p ^ (j + 1) * prob (fun a => ¬0 < d a) := by
          rw [Rounds.expect_add, expect_mul_left, expect_mul_left, ← prob_eq_expect, ← prob_eq_expect]
      _ ≤ p ^ j * (p * prob (fun a => 0 < d a)) + p ^ (j + 1) * prob (fun a => ¬0 < d a) := by
          gcongr
      _ = p ^ (j + 1) * (prob (fun a => 0 < d a) + prob (fun a => ¬0 < d a)) := by ring
      _ ≤ p ^ (j + 1) := mul_le_of_le_one_right (pow_nonneg hp _) (prob_add_not_le _)
    by_cases hda : 0 < d a
    · by_cases hpa : pass a
      · rw [ite_eq_left ⟨hpa, hda⟩, ite_eq_right (not_not.mpr hda), mul_one, mul_zero, add_zero]
        refine (Rounds.expect_ind_le_prob (Q := fun y => (next a).accepts y ∧
          j ≤ Accounting.TS.RoundGame.deficientRounds (next a) y) fun y hy => ?_).trans (ih a (hnext a) j)
        simp only [RoundGame.accepts, Accounting.TS.RoundGame.deficientRounds, ite_eq_left hda] at hy
        exact ⟨hy.1.2, by omega⟩
      · rw [ite_eq_right fun h => hpa h.1, ite_eq_right (not_not.mpr hda), mul_zero, mul_zero, add_zero]
        refine (Rounds.expect_ind_le_prob (Q := fun _ => False) fun y hy => hpa ?_).trans Rounds.prob_false.le
        simp only [RoundGame.accepts] at hy
        exact hy.1.1
    · rw [ite_eq_right fun h => hda h.2, ite_eq_left hda, mul_zero, zero_add, mul_one]
      refine (Rounds.expect_ind_le_prob (Q := fun y => (next a).accepts y ∧
        j + 1 ≤ Accounting.TS.RoundGame.deficientRounds (next a) y) fun y hy => ?_).trans (ih a (hnext a) (j + 1))
      simp only [RoundGame.accepts, Accounting.TS.RoundGame.deficientRounds, ite_eq_right hda] at hy
      exact ⟨hy.1.2, by omega⟩

/-- A round's share is at most 1, so the deficiency is at most the number of deficient rounds. -/
theorem deficiency_le_rounds {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (p : ℝ)
    (hL : RoundGame.LuckAt p g) (x : Paths Ω S) :
    g.deficiency x ≤ Accounting.TS.RoundGame.deficientRounds g x := by
  induction g with
  | done => simp [RoundGame.deficiency, Accounting.TS.RoundGame.deficientRounds]
  | round d pass next ih =>
    obtain ⟨hd, -, hnext⟩ := hL
    simp only [RoundGame.deficiency, Accounting.TS.RoundGame.deficientRounds, Nat.cast_add]
    refine add_le_add ?_ (ih x.1 (hnext x.1) x.2)
    split_ifs with h
    · simpa using (hd x.1).2
    · simpa using (not_lt.mp h)

/-- Lemma C's step 1, SAMPLING's budget: when `Λ` bounds the play, the ideal game accepts with deficiency at least
`Λ·t` with probability at most `e^{−t}` (the product over rounds is a supermartingale). -/
theorem budget_tail {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (Λ : ℝ) (hΛ : 0 < Λ)
    (hB : RoundGame.BudgetAt Λ g) (t : ℝ) :
    prob (fun x => g.accepts x ∧ Λ * t ≤ g.deficiency x) ≤ Real.exp (-t) := by
  have h := g.prob_accepts_le (c := 1 / Λ) (by positivity) (Rounds.expBound_of_budgetAt g hB) (Λ * t)
  rwa [one_div, inv_mul_cancel_left₀ hΛ.ne'] at h

end Accounting.SecurityProofs.TS
