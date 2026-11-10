module

public import Proofs.Pouw.Sampling
public import Definitions.Pouw.Game.ClopperPearson
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Ring

/-!
# Clopper–Pearson's upper bound is a miss-bound table (`missBound_cpUpper`)

`t` uniform samples from a type of `a + b` elements land `k` times among `a` of them for `C(t, k)·a^k·b^(t−k)` of the
`(a + b)^t` samples (`card_count`), so the count is `Bin(t, a/(a+b))` (`pr_count_le`). Over work-weighted tickets, a set
`S` holding the share `p` of the tickets is sampled `X ~ Bin(t, p)` times. The bound at `X` falls below `p` only if `X` is
at most the largest `x ≤ t` whose bound is below `p`, and `Pr_p[X ≤ x] ≤ δ` there: some share `r < p` is in the set
`cpUpper` takes the infimum of, and `p` lies between `r` and `1`. So Clopper–Pearson's table at `δ` is a miss bound at
`δ`, for every `δ ≥ 0` (`missBound_cpUpper`), and at no `δ < 0` (`missBound_cpUpper_nontrivial`).
-/

@[expose] public section

namespace Pouw.Proofs

open Finset

section Count

variable {α : Type} [Fintype α] (P : α → Prop) [DecidablePred P]

omit [Fintype α] in
theorem count_cons {t : ℕ} (x : α) (c : Fin t → α) :
    (univ.filter fun i => P ((Fin.cons x c : Fin (t + 1) → α) i)).card
      = (if P x then 1 else 0) + (univ.filter fun i => P (c i)).card := by
  rw [card_filter, card_filter, Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]

theorem card_count_succ (t k : ℕ) :
    ((univ : Finset (Fin (t + 1) → α)).filter fun c => (univ.filter fun i => P (c i)).card = k).card
      = (univ.filter P).card *
          ((univ : Finset (Fin t → α)).filter fun c => 1 + (univ.filter fun i => P (c i)).card = k).card
        + (univ.filter fun x => ¬ P x).card *
          ((univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card = k).card := by
  calc ((univ : Finset (Fin (t + 1) → α)).filter fun c => (univ.filter fun i => P (c i)).card = k).card
      = ∑ c : Fin (t + 1) → α, if (univ.filter fun i => P (c i)).card = k then 1 else 0 := card_filter _ _
    _ = ∑ xc : α × (Fin t → α),
          if (univ.filter fun i => P ((Fin.cons xc.1 xc.2 : Fin (t + 1) → α) i)).card = k then 1 else 0 :=
        ((Fin.consEquiv fun _ => α).sum_comp
          (fun c : Fin (t + 1) → α => if (univ.filter fun i => P (c i)).card = k then 1 else 0)).symm
    _ = ∑ x : α, if P x then
            ((univ : Finset (Fin t → α)).filter fun c => 1 + (univ.filter fun i => P (c i)).card = k).card
          else ((univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card = k).card := by
        rw [Fintype.sum_prod_type]
        refine sum_congr rfl fun x _ => ?_
        simp only [count_cons]
        by_cases hx : P x
        · simp only [hx, ↓reduceIte]
          exact (card_filter _ _).symm
        · simp only [hx, ↓reduceIte, zero_add]
          exact (card_filter _ _).symm
    _ = _ := by rw [sum_ite, sum_const, sum_const, smul_eq_mul, smul_eq_mul]

/-- **`t` uniform samples land `k` times in `P`** for `C(t, k)·a^k·b^(t−k)` of them, `a` the elements in `P` and `b` the
rest. -/
theorem card_count (t k : ℕ) :
    ((univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card = k).card
      = t.choose k * (univ.filter P).card ^ k * (univ.filter fun x => ¬ P x).card ^ (t - k) := by
  induction t generalizing k with
  | zero =>
    rcases k with _ | k
    · simp
    · simp
  | succ t ih =>
    rw [card_count_succ]
    rcases k with _ | j
    · have h0 : ((univ : Finset (Fin t → α)).filter fun c => 1 + (univ.filter fun i => P (c i)).card = 0) = ∅ :=
        filter_eq_empty_iff.mpr fun c _ => by omega
      rw [h0, card_empty, ih]
      simp only [mul_zero, zero_add, Nat.choose_zero_right, pow_zero, Nat.sub_zero, one_mul]
      ring
    · have h1 : ((univ : Finset (Fin t → α)).filter fun c => 1 + (univ.filter fun i => P (c i)).card = j + 1)
          = (univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card = j :=
        filter_congr fun c _ => by omega
      rw [h1, ih, ih, Nat.choose_succ_succ', Nat.add_sub_add_right]
      rcases Nat.lt_or_ge j t with hj | hj
      · obtain ⟨m, rfl⟩ : ∃ m, t = j + 1 + m := ⟨t - (j + 1), by omega⟩
        have e1 : j + 1 + m - j = m + 1 := by omega
        have e2 : j + 1 + m - (j + 1) = m := by omega
        rw [e1, e2]
        ring
      · rw [Nat.choose_eq_zero_of_lt (by omega : t < j + 1)]
        ring

/-- **The count of `t` uniform samples in `P` is binomial**: it is at most `x` with probability `Pr_p[X ≤ x]`,
`p = a/(a + b)`. -/
theorem pr_count_le (t x : ℕ) (hα : 0 < Fintype.card α) :
    pr (fun c : Fin t → α => (univ.filter fun i => P (c i)).card ≤ x)
      = binomCdf t ((univ.filter P).card / Fintype.card α) x := by
  classical
  have hcard : ((univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card ≤ x).card
      = ∑ k ∈ range (x + 1),
          t.choose k * (univ.filter P).card ^ k * (univ.filter fun y => ¬ P y).card ^ (t - k) := by
    rw [card_eq_sum_card_fiberwise (f := fun c : Fin t → α => (univ.filter fun i => P (c i)).card)
      (t := range (x + 1)) ?maps]
    · refine sum_congr rfl fun k hk => ?_
      rw [filter_filter, ← card_count P t k]
      congr 1
      refine filter_congr fun c _ => ?_
      have := mem_range.mp hk
      constructor
      · exact fun h => h.2
      · intro h
        exact ⟨by omega, h⟩
    · intro c hc
      exact mem_range.mpr (Nat.lt_succ_of_le (mem_filter.mp hc).2)
  have hn : (0 : ℝ) < Fintype.card α := by exact_mod_cast hα
  have hab : ((univ.filter P).card : ℝ) + (univ.filter fun y => ¬ P y).card = Fintype.card α := by
    have h := card_filter_add_card_filter_not (s := (univ : Finset α)) P
    rw [card_univ] at h
    exact_mod_cast h
  have h1 : 1 - ((univ.filter P).card : ℝ) / Fintype.card α
      = ((univ.filter fun y => ¬ P y).card : ℝ) / Fintype.card α := by
    rw [eq_div_iff hn.ne', sub_mul, div_mul_cancel₀ _ hn.ne']
    linarith
  rw [pr_eq_card _ ((univ : Finset (Fin t → α)).filter fun c => (univ.filter fun i => P (c i)).card ≤ x)
    (fun c => by simp), hcard, Fintype.card_fun, Fintype.card_fin, binomCdf, h1]
  push_cast
  rw [sum_div]
  refine sum_congr rfl fun k _ => ?_
  rcases le_or_gt k t with hk | hk
  · obtain ⟨m, rfl⟩ : ∃ m, t = k + m := ⟨t - k, by omega⟩
    rw [Nat.add_sub_cancel_left, div_pow, div_pow, pow_add]
    field_simp
  · rw [Nat.choose_eq_zero_of_lt hk]
    simp

end Count

/-- **Clopper–Pearson's upper bound at `δ ≥ 0` is a miss bound at `δ`**: over `t` work-weighted samples, whatever the
set `S` of units, the bound at the samples `S` drew falls below `S`'s share of the tickets with probability at most
`δ`. -/
theorem missBound_cpUpper (t : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) : MissBound t (cpUpper t δ) δ := by
  classical
  intro N w S hS
  have hAW : ∑ g ∈ S, (w g : ℝ) ≤ ∑ g ∈ range N, (w g : ℝ) :=
    sum_le_sum_of_subset_of_nonneg hS fun _ _ _ => Nat.cast_nonneg _
  have hW0 : 0 ≤ ∑ g ∈ range N, (w g : ℝ) := sum_nonneg fun _ _ => Nat.cast_nonneg _
  rcases hW0.eq_or_lt with hW | hW
  · refine (pr_eq_zero fun c h => ?_).trans_le hδ
    rw [← hW, mul_zero] at h
    linarith
  have hp1 : (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ) ≤ 1 := (div_le_one hW).mpr hAW
  have hev : ∀ c : Fin t → Ticket w N,
      cpUpper t δ (drawsIn S c) * ∑ g ∈ range N, (w g : ℝ) < ∑ g ∈ S, (w g : ℝ) →
        cpUpper t δ (drawsIn S c) < (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ) :=
    fun c h => (lt_div_iff₀ hW).mpr h
  have hle : ∀ c : Fin t → Ticket w N, drawsIn S c ≤ t :=
    fun c => (card_le_univ _).trans (Fintype.card_fin t).le
  by_cases hG : ∃ k ≤ t, cpUpper t δ k < (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ)
  · let G := (range (t + 1)).filter fun k => cpUpper t δ k < (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ)
    obtain ⟨k₀, hk₀, hk₀'⟩ := hG
    have hne : G.Nonempty := ⟨k₀, mem_filter.mpr ⟨mem_range.mpr (by omega), hk₀'⟩⟩
    have hx0 : cpUpper t δ (G.max' hne) < (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ) :=
      (mem_filter.mp (G.max'_mem hne)).2
    refine (pr_mono (P' := fun c : Fin t → Ticket w N => drawsIn S c ≤ G.max' hne) fun c h =>
      G.le_max' _ (mem_filter.mpr ⟨mem_range.mpr (Nat.lt_succ_of_le (hle c)), hev c h⟩)).trans ?_
    have hpos : 0 < Fintype.card (Ticket w N) := by
      rw [card_ticket]
      exact_mod_cast hW
    have hshare : ((univ.filter fun τ : Ticket w N => ((τ.1 : ℕ) ∈ S)).card : ℝ) / (Fintype.card (Ticket w N) : ℝ)
        = (∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ) := by
      rw [card_good_tickets, card_ticket, sum_ite_mem, inter_eq_right.mpr hS]
      simp only [Nat.cast_sum]
    have hcount : pr (fun c : Fin t → Ticket w N => drawsIn S c ≤ G.max' hne)
        = binomCdf t ((∑ g ∈ S, (w g : ℝ)) / ∑ g ∈ range N, (w g : ℝ)) (G.max' hne) := by
      rw [← hshare]
      exact pr_count_le (fun τ : Ticket w N => ((τ.1 : ℕ) ∈ S)) t _ hpos
    rw [hcount]
    unfold cpUpper at hx0
    obtain ⟨r, hr, hrp⟩ := exists_lt_of_csInf_lt ⟨1, by simp⟩ hx0
    rcases hr.2.2 with h1 | hq
    · rw [h1] at hrp
      linarith
    · exact hq _ hrp.le hp1
  · exact (pr_eq_zero fun c h => hG ⟨drawsIn S c, hle c, hev c h⟩).trans_le hδ

/-- **`missBound_cpUpper`'s honest witness**: its premise holds at δ = 1/100, the δ Pearl+'s run is credited at. -/
theorem missBound_cpUpper_honest : (0 : ℝ) ≤ 1 / 100 := by norm_num

/-- **`missBound_cpUpper`'s nontrivial witness**: without its premise the bound fails, since no probability is below a
negative δ. -/
theorem missBound_cpUpper_nontrivial : ¬ MissBound 1 (cpUpper 1 (-1)) (-1) := by
  intro h
  have h' := (pr_nonneg _).trans (h 0 (fun _ => 0) ∅ (empty_subset _))
  norm_num at h'

end Pouw.Proofs
