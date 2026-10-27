import Definitions.Accounting.Guarantees.TS

/-!
# The tensor slots and the output count

T1, its tightness and Lemma 2's count (EPS1-TENSOR).

* `TensorSlotCount`: an injective map of `N` committed words into slot writes gives `N ≤ Σ_i w_i n_i`, and
  `w_i ≤ ρ_T c_i` (`div_le_rhoT`) turns it into `N/ρ_T ≤ Σ_i c_i n_i`.
* `TensorSlotTight`: both ratios lie in `(0, 1]`, so their product is `1` iff both are.
* `OutputCount`: the head writes number `Σ_t ω(cls t) = Σ_k ω_k n_k`, and an image is no larger than its type.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Finset

/-- Every form writes at most `ρ_T` slots a pipe cycle. -/
theorem div_le_rhoT {ι : Type} [Fintype ι] (w c : ι → ℕ) (i : ι) : (w i : ℝ) / c i ≤ rhoT w c :=
  le_ciSup (f := fun i => (w i : ℝ) / c i) (Set.finite_range _).bddAbove i

theorem rhoT_nonneg {ι : Type} [Fintype ι] [Nonempty ι] (w c : ι → ℕ) : 0 ≤ rhoT w c :=
  le_trans (by positivity) (div_le_rhoT w c (Classical.arbitrary ι))

/-- `N` distinct slot writes need `N ≤ Σ_i n_i w_i`. -/
theorem card_le_slots {ι : Type} [Fintype ι] (w n : ι → ℕ) (N : ℕ) (slot : Fin N → SlotWrite ι w n)
    (hinj : Function.Injective slot) : N ≤ ∑ i, n i * w i := by
  simpa [SlotWrite, Fintype.card_sigma, Fintype.card_prod] using Fintype.card_le_of_injective slot hinj

/-- T1. -/
theorem TensorSlotCount : Accounting.Guarantees.TS.TensorSlotCount := by
  intro ι _ _ w c n N slot hc hinj
  have hN : (N : ℝ) ≤ ∑ i, (n i : ℝ) * w i := by exact_mod_cast card_le_slots w n N slot hinj
  have hw : ∀ i, (w i : ℝ) ≤ rhoT w c * c i := fun i =>
    (div_le_iff₀ (by exact_mod_cast hc i)).1 (div_le_rhoT w c i)
  rcases (rhoT_nonneg w c).eq_or_lt with h0 | hpos
  · rw [← h0, div_zero]
    positivity
  · rw [div_le_iff₀ hpos, sum_mul]
    refine hN.trans (sum_le_sum fun i _ => ?_)
    calc (n i : ℝ) * w i ≤ n i * (rhoT w c * c i) := mul_le_mul_of_nonneg_left (hw i) (by positivity)
      _ = c i * n i * rhoT w c := by ring

/-- T1's tightness. -/
theorem TensorSlotTight : Accounting.Guarantees.TS.TensorSlotTight := by
  intro ρH ρT dH L hρH hle hdH hdL
  have hρT : 0 < ρT := hρH.trans_le hle
  have hL : 0 < L := hdH.trans_le hdL
  have ha : ρH / ρT ≤ 1 := (div_le_one hρT).2 hle
  have hb : dH / L ≤ 1 := (div_le_one hL).2 hdL
  have ha0 : 0 < ρH / ρT := div_pos hρH hρT
  constructor
  · intro h
    have hab : ρH / ρT * (dH / L) = 1 := by linarith
    have ha1 : ρH / ρT = 1 := by
      refine le_antisymm ha ?_
      by_contra hlt
      have : ρH / ρT * (dH / L) < 1 :=
        (mul_le_of_le_one_right ha0.le hb).trans_lt (lt_of_not_ge hlt)
      linarith
    rw [ha1, one_mul] at hab
    exact ⟨((div_eq_one_iff_eq hL.ne').1 hab).symm, (div_eq_one_iff_eq hρT.ne').1 ha1⟩
  · rintro ⟨hL1, hρ1⟩
    rw [hL1, hρ1, div_self hρT.ne', div_self hdH.ne']
    ring

/-- The head writes number `Σ_k ω_k n_k`. -/
theorem card_headWrite {K : Type} [Fintype K] [DecidableEq K] (ω : K → ℕ) (G : ℕ) (cls : Fin G → K) :
    Fintype.card (HeadWrite ω G cls) = ∑ k, ω k * classCount cls k := by
  rw [Fintype.card_sigma, ← sum_fiberwise univ cls fun t => Fintype.card (Fin (ω (cls t)))]
  refine sum_congr rfl fun k _ => ?_
  rw [sum_congr rfl fun t ht => by rw [Fintype.card_fin, (mem_filter.1 ht).2], sum_const, smul_eq_mul,
    classCount, mul_comm]

/-- Lemma 2's count. -/
theorem OutputCount : Accounting.Guarantees.TS.OutputCount := by
  intro K _ _ ω G cls Nc head
  have h : (univ.image head).card ≤ ∑ k, ω k * classCount cls k :=
    (card_le_univ _).trans (card_headWrite ω G cls).le
  exact_mod_cast h

end Accounting.SecurityProofs.TS
