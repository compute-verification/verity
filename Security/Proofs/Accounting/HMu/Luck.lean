import Definitions.Accounting.Guarantees.HMu
import Proofs.Accounting.Tag.LittlewoodOfford

/-!
# Lemma B, and (H-μ) from the coins

`lemmaB` sums (H-μ) over the at most `Z` skipped values (Inj) leaves on each deficient view. The coin bounds count bit
vectors at each committed `ω₀`: Erdős's Littlewood–Offord bound (`Accounting.TagProofs.card_level_le_choose`, moved
from `Fin u` to any finite index), or the pairing of each bit vector with the one that flips a bit of nonzero weight.
-/

namespace Accounting.HMuProofs

open Accounting.HMu Pouw.RelativeSecurity Finset

section Prob

variable {Ω : Type} [Fintype Ω]

theorem prob_eq_card (p : Ω → Prop) [DecidablePred p] :
    prob p = ((univ.filter p).card : ℝ) / Fintype.card Ω := by
  unfold prob
  congr

theorem prob_nonneg (p : Ω → Prop) : 0 ≤ prob p := by
  classical
  rw [prob_eq_card]
  positivity

theorem prob_mono {p q : Ω → Prop} (h : ∀ ω, p ω → q ω) : prob p ≤ prob q := by
  classical
  rw [prob_eq_card, prob_eq_card]
  gcongr with ω
  exact h ω

theorem prob_congr {p q : Ω → Prop} (h : ∀ ω, p ω ↔ q ω) : prob p = prob q :=
  le_antisymm (prob_mono fun ω => (h ω).1) (prob_mono fun ω => (h ω).2)

theorem prob_eq_zero {p : Ω → Prop} (h : ∀ ω, ¬p ω) : prob p = 0 := by
  classical
  rw [prob_eq_card, filter_false_of_mem fun ω _ => h ω, card_empty, Nat.cast_zero, zero_div]

theorem prob_exists_le {α : Type} (s : Finset α) (P : α → Ω → Prop) :
    prob (fun ω => ∃ x ∈ s, P x ω) ≤ ∑ x ∈ s, prob (P x) := by
  classical
  simp only [prob_eq_card, ← sum_div]
  gcongr
  calc ((univ.filter fun ω => ∃ x ∈ s, P x ω).card : ℝ)
      ≤ ((s.biUnion fun x => univ.filter (P x)).card : ℝ) := by
        exact_mod_cast card_le_card fun ω hω => by simpa using hω
    _ ≤ ∑ x ∈ s, ((univ.filter (P x)).card : ℝ) := by exact_mod_cast card_biUnion_le

open Classical in
/-- A bound at every fibre of `f` is a bound. -/
theorem prob_le_of_fibres {O : Type} (f : Ω → O) (p q : Ω → Prop) (c : ℝ)
    (h : ∀ o, prob (fun ω => p ω ∧ f ω = o) ≤ c * prob (fun ω => q ω ∧ f ω = o)) :
    prob p ≤ c * prob q := by
  have key : ∀ r : Ω → Prop, prob r = ∑ o ∈ univ.image f, prob (fun ω => r ω ∧ f ω = o) := by
    intro r
    simp only [prob_eq_card, ← sum_div]
    congr 1
    rw [card_eq_sum_card_fiberwise (f := f) (t := univ.image f) (by intro x _; exact mem_image_of_mem f (mem_univ x))]
    push_cast
    simp only [filter_filter]
  rw [key p, key q, mul_sum]
  exact sum_le_sum fun o _ => h o

theorem prob_eq_one [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  rw [prob_eq_card, filter_true_of_mem fun ω _ => h ω, card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

/-- On a product, a bound on every slice `{a} × B` is a bound. -/
theorem prob_le_of_slices {A B : Type} [Fintype A] [Fintype B] (p q : A × B → Prop) (c : ℝ)
    (h : ∀ a, prob (fun b => p (a, b)) ≤ c * prob (fun b => q (a, b))) :
    prob p ≤ c * prob q := by
  classical
  have key : ∀ r : A × B → Prop, prob r = (∑ a, prob (fun b => r (a, b))) / Fintype.card A := by
    intro r
    simp only [prob_eq_card, ← sum_div, div_div, Fintype.card_prod]
    rw [mul_comm (Fintype.card B : ℝ)]
    congr 1
    rw [card_filter, Fintype.sum_prod_type]
    push_cast
    simp only [card_filter]
    push_cast
    rfl
    exact Nat.cast_mul _ _
  rw [key p, key q, mul_div_assoc', mul_sum]
  gcongr with a
  exact h a

end Prob

section Coins

variable {Ω₀ O J G : Type} [Fintype Ω₀] [Fintype J] [DecidableEq J]

/-- A bound on the bit vectors at every committed `ω₀` that meets `P` bounds the probability given any view of the
committed part. -/
theorem prob_slice_le (V : Ω₀ → O) (o : O) (P : Ω₀ → Prop) (X : Ω₀ → (J → Bool) → Prop) (c : ℝ) (hc : 0 ≤ c)
    (h : ∀ a, P a → prob (X a) ≤ c) :
    prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o ∧ P ω.1 ∧ X ω.1 ω.2) ≤
      c * prob (fun ω : Ω₀ × (J → Bool) => V ω.1 = o) := by
  apply prob_le_of_slices
  intro a
  by_cases hV : V a = o
  · rw [prob_eq_one (p := fun b : J → Bool => V (a, b).1 = o) fun _ => hV, mul_one]
    by_cases hP : P a
    · exact (prob_mono fun b hb => hb.2.2).trans (h a hP)
    · rw [prob_eq_zero fun b hb => hP hb.2.1]
      exact hc
  · rw [prob_eq_zero fun b hb => hV hb.1]
    exact mul_nonneg hc (prob_nonneg _)

omit [DecidableEq J] in
theorem coinSum_comp_equiv {u : ℕ} (eq : J ≃ Fin u) (v : J → ℤ) (e : J → Bool) :
    coinSum v e = Accounting.Tag.unreadSum (fun i => v (eq.symm i)) (fun i => e (eq.symm i)) := by
  unfold coinSum Accounting.Tag.unreadSum
  exact (Equiv.sum_comp eq.symm (fun j => if e j then v j else 0)).symm

/-- Erdős's Littlewood–Offord bound over any finite index. -/
theorem card_coinSum_le (v : J → ℤ) (hv : ∀ j, v j ≠ 0) (t : ℤ) :
    (univ.filter fun e : J → Bool => coinSum v e = t).card ≤
      (Fintype.card J).choose (Fintype.card J / 2) := by
  let eq : J ≃ Fin (Fintype.card J) := Fintype.equivFin J
  refine le_trans (le_of_eq ?_)
    (Accounting.TagProofs.card_level_le_choose (fun i => v (eq.symm i)) (fun i => hv _) t)
  apply card_equiv (Equiv.arrowCongr eq (Equiv.refl Bool))
  intro e
  simp only [mem_filter, mem_univ, true_and, coinSum_comp_equiv eq]
  rfl

/-- (H-μ) from the coins at `μ = C(u, ⌊u/2⌋)/2^u`. -/
theorem prob_coinSum_le (a : ℤ) (v : J → ℤ) (hv : ∀ j, v j ≠ 0) (t : ℤ) :
    prob (fun e : J → Bool => a + coinSum v e = t) ≤
      ((Fintype.card J).choose (Fintype.card J / 2) : ℝ) / 2 ^ Fintype.card J := by
  rw [prob_congr (q := fun e : J → Bool => coinSum v e = t - a) fun e => by constructor <;> intro h <;> omega,
    prob_eq_card, Fintype.card_fun, Fintype.card_bool]
  push_cast
  gcongr
  exact_mod_cast card_coinSum_le v hv (t - a)

theorem coinSum_update [AddCommMonoid G] (v : J → G) (e : J → Bool) (j : J) (b : Bool) :
    coinSum v (Function.update e j b) = (if b then v j else 0) + ∑ k ∈ univ.erase j, (if e k then v k else 0) := by
  unfold coinSum
  rw [← add_sum_erase _ _ (mem_univ j)]
  congr 1
  · simp
  · exact sum_congr rfl fun k hk => by rw [Function.update_of_ne (ne_of_mem_erase hk)]

/-- A bit of nonzero weight pairs each bit vector with one of another sum: at most half share a sum. -/
theorem prob_coinSum_half [AddCommGroup G] (a : G) (v : J → G) (j : J) (hj : v j ≠ 0) (t : G) :
    prob (fun e : J → Bool => a + coinSum v e = t) ≤ 1 / 2 := by
  classical
  set S := univ.filter fun e : J → Bool => a + coinSum v e = t
  let fl : (J → Bool) → (J → Bool) := fun e => Function.update e j (!e j)
  have hfl : ∀ e, fl (fl e) = e := by intro e; simp [fl]
  have hne : ∀ e, coinSum v (fl e) ≠ coinSum v e := by
    intro e h
    have h2 := coinSum_update v e j (e j)
    rw [Function.update_eq_self] at h2
    simp only [fl] at h
    rw [coinSum_update, h2] at h
    cases hb : e j <;> simp [hb] at h <;> exact hj (by simpa using h)
  have hmap : S.card ≤ (univ \ S).card := by
    apply card_le_card_of_injOn fl
    · intro e he
      have he' : a + coinSum v e = t := by simpa [S] using he
      have hfe : a + coinSum v (fl e) ≠ t := fun h => hne e (add_left_cancel (h.trans he'.symm))
      simpa [S] using hfe
    · intro e₁ _ e₂ _ h
      rw [← hfl e₁, ← hfl e₂]
      exact congrArg fl h
  have hsum := card_sdiff_add_card_eq_card (subset_univ S)
  rw [card_univ, Fintype.card_fun, Fintype.card_bool] at hsum
  have h2 : (2 : ℝ) * S.card ≤ 2 ^ Fintype.card J := by exact_mod_cast (by omega : 2 * S.card ≤ 2 ^ Fintype.card J)
  rw [prob_eq_card, Fintype.card_fun, Fintype.card_bool]
  push_cast
  rw [div_le_iff₀ (by positivity)]
  linarith

/-- Receipt luck at each committed `ω₀`: at most `Z` skipped values give the committed receipt, each of probability at
most `m` on the bit vectors. -/
theorem prob_luck_le {β : Type} (X : Ω₀ → (J → Bool) → G) (hon : Ω₀ → G → β) (com : Ω₀ → β) (d : Ω₀ → ℝ)
    (P : Ω₀ → Prop) (Z : ℕ) (m : ℝ) (hm : 0 ≤ m) (hX : ∀ a, P a → ∀ x, prob (fun e => X a e = x) ≤ m) :
    prob (fun ω : Ω₀ × (J → Bool) =>
        (∀ b, ∃ s : Finset G, s.card ≤ Z ∧ ∀ e, hon ω.1 (X ω.1 e) = b → X ω.1 e ∈ s) ∧ P ω.1 ∧
          (com ω.1 = hon ω.1 (X ω.1 ω.2) ∧ 0 < d ω.1)) ≤
      Z * m * prob (fun ω : Ω₀ × (J → Bool) => 0 < d ω.1) := by
  apply prob_le_of_slices
  intro a
  by_cases hd : 0 < d a
  · rw [prob_eq_one (p := fun b : J → Bool => 0 < d (a, b).1) fun _ => hd, mul_one]
    by_cases hI : ∀ b, ∃ s : Finset G, s.card ≤ Z ∧ ∀ e, hon a (X a e) = b → X a e ∈ s
    · by_cases hP : P a
      · obtain ⟨s, hs, hsX⟩ := hI (com a)
        calc _ ≤ prob (fun e => ∃ x ∈ s, X a e = x) :=
              prob_mono fun e he => ⟨X a e, hsX e he.2.2.1.symm, rfl⟩
          _ ≤ ∑ x ∈ s, prob (fun e => X a e = x) := prob_exists_le s _
          _ ≤ ∑ _x ∈ s, m := sum_le_sum fun x _ => hX a hP x
          _ = s.card * m := by rw [sum_const, nsmul_eq_mul]
          _ ≤ Z * m := by gcongr
      · rw [prob_eq_zero fun b hb => hP hb.2.1]
        positivity
    · rw [prob_eq_zero fun b hb => hI hb.1]
      positivity
  · rw [prob_eq_zero fun b hb => hd hb.2.2.2, prob_eq_zero (p := fun b : J → Bool => 0 < d (a, b).1) fun b hb => hd hb,
      mul_zero]

end Coins

/-- **Lemma B**: on each deficient view, the round is lucky only where its skipped value is one of the at most `Z` that
give the committed receipt, each of point mass at most `μ`. -/
theorem lemmaB : Accounting.Guarantees.HMu.LemmaB := by
  intro Ω O α β _ R μ Z ⟨hpm, hinj⟩
  apply prob_le_of_fibres R.view
  intro o
  by_cases hd : 0 < R.d o
  · obtain ⟨s, hs, hsX⟩ := hinj o hd (R.committed o)
    have hnn : 0 ≤ μ * prob (fun ω => R.view ω = o) := by
      by_cases hex : ∃ ω, R.view ω = o
      · obtain ⟨ω, hω⟩ := hex
        exact (prob_nonneg _).trans (hpm o hd (R.skipped ω))
      · push Not at hex
        rw [prob_eq_zero hex, mul_zero]
    calc prob (fun ω => R.lucky ω ∧ R.view ω = o)
        ≤ prob (fun ω => ∃ x ∈ s, R.view ω = o ∧ R.skipped ω = x) := by
          apply prob_mono
          rintro ω ⟨⟨hc, -⟩, hv⟩
          subst hv
          exact ⟨R.skipped ω, hsX ω rfl hc.symm, rfl, rfl⟩
      _ ≤ ∑ x ∈ s, prob (fun ω => R.view ω = o ∧ R.skipped ω = x) := prob_exists_le s _
      _ ≤ ∑ _x ∈ s, μ * prob (fun ω => R.view ω = o) := sum_le_sum fun x _ => hpm o hd x
      _ = s.card * (μ * prob (fun ω => R.view ω = o)) := by rw [sum_const, nsmul_eq_mul]
      _ ≤ Z * (μ * prob (fun ω => R.view ω = o)) := by gcongr
      _ = Z * μ * prob (fun ω => 0 < R.d (R.view ω) ∧ R.view ω = o) := by
          rw [prob_congr (p := fun ω => 0 < R.d (R.view ω) ∧ R.view ω = o) (q := fun ω => R.view ω = o)
            fun ω => ⟨fun h => h.2, fun h => ⟨by rw [h]; exact hd, h⟩⟩]
          ring
  · rw [prob_eq_zero (p := fun ω => R.lucky ω ∧ R.view ω = o) fun ω h => hd (h.2 ▸ h.1.2),
      prob_eq_zero (p := fun ω => 0 < R.d (R.view ω) ∧ R.view ω = o) fun ω h => hd (h.2 ▸ h.1), mul_zero]

theorem coinPointMass : Accounting.Guarantees.HMu.CoinPointMass := by
  intro Ω₀ O J _ _ _ V a v o t
  exact prob_slice_le V o (fun ω₀ => ∀ j, v ω₀ j ≠ 0) (fun ω₀ e => a ω₀ + coinSum (v ω₀) e = t) _ (by positivity)
    fun ω₀ hv => prob_coinSum_le (a ω₀) (v ω₀) hv t

theorem coinPointMassHalf : Accounting.Guarantees.HMu.CoinPointMassHalf := by
  intro Ω₀ O J G _ _ _ _ V a v o t
  exact prob_slice_le V o (fun ω₀ => ∃ j, v ω₀ j ≠ 0) (fun ω₀ e => a ω₀ + coinSum (v ω₀) e = t) _ (by norm_num)
    fun ω₀ ⟨j, hj⟩ => prob_coinSum_half (a ω₀) (v ω₀) j hj t

theorem receiptLuckOfCoins : Accounting.Guarantees.HMu.ReceiptLuckOfCoins := by
  intro Ω₀ J β _ _ _ R Z
  exact prob_luck_le (fun a e => R.skipped (a, e)) R.honest R.committed R.d (fun a => ∀ j, R.weight a j ≠ 0) Z _
    (by positivity) fun ω₀ hv x => prob_coinSum_le (R.base ω₀) (R.weight ω₀) hv x

theorem receiptLuckOfCoin : Accounting.Guarantees.HMu.ReceiptLuckOfCoin := by
  intro Ω₀ J G β _ _ _ _ R Z
  exact prob_luck_le (fun a e => R.skipped (a, e)) R.honest R.committed R.d (fun a => ∃ j, R.weight a j ≠ 0) Z _
    (by norm_num) fun ω₀ ⟨j, hj⟩ x => prob_coinSum_half (R.base ω₀) (R.weight ω₀) j hj x

end Accounting.HMuProofs
