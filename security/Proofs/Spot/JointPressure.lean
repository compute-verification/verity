module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Probability.Distributions.Uniform
public import Proofs.Spot.ExfiltrationBound

/-! One sample for several claims: the max rule and the union bound. -/

@[expose] public section

namespace Population.Exfil

open Finset

noncomputable def maxRule {I U : Type} (claims : Finset I) (κ : I → ℝ) (harm : I → U → ℝ) (u : U) : ℝ :=
  claims.fold max 0 fun i => κ i * harm i u

theorem le_maxRule {I U : Type} {claims : Finset I} {i : I} (hi : i ∈ claims) (κ : I → ℝ) (harm : I → U → ℝ)
    (u : U) : κ i * harm i u ≤ maxRule claims κ harm u :=
  (Finset.le_fold_max _).2 (Or.inr ⟨i, hi, le_refl _⟩)

theorem maxRule_le {I U : Type} {claims : Finset I} {κ : I → ℝ} {harm : I → U → ℝ} {a : U → ℝ} (u : U)
    (h0 : 0 ≤ a u) (h : ∀ i ∈ claims, κ i * harm i u ≤ a u) : maxRule claims κ harm u ≤ a u :=
  (Finset.fold_max_le _).2 ⟨h0, h⟩

theorem pressure_maxRule {Ω U I : Type} {μ : PMF Ω} {claims : Finset I} {κ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 ≤ κ i) (harm : I → U → ℝ) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (B : List Ω → Finset U) (a : U → ℝ) (ha : ∀ u, 0 ≤ a u ∧ a u ≤ 1) (hmax : ∀ u, maxRule claims κ harm u ≤ a u)
    (hleak : ∀ i ∈ claims, ∀ h, leak i h ≤ ∑ u ∈ B h, harm i u)
    (hesc : ∀ h, Verity.Game.prPMFReal μ (· ∈ esc h) ≤ ∏ u ∈ B h, (1 - a u)) :
    ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩ := fun i hi =>
  pressure_of_sample (hκ i hi) ⟨leak i, esc⟩ B (harm i) a ha
    (fun u => (le_maxRule hi κ harm u).trans (hmax u)) (hleak i hi) hesc

variable {Ω : Type} [Fintype Ω] {μ : PMF Ω}

open Classical in
theorem prob_exists_le {I : Type} (n : ℕ) (s : Finset I) (E : I → (Fin n → Ω) → Prop) :
    prob μ n (fun cs => ∃ i ∈ s, E i cs) ≤ ∑ i ∈ s, prob μ n (E i) := by
  simp only [prob_eq_ex]
  calc ex μ n _ ≤ ex μ n (fun cs => ∑ i ∈ s, if E i cs then (1 : ℝ) else 0) := ex_mono fun cs => by
          split_ifs with h
          · obtain ⟨i, hi, he⟩ := h
            refine le_trans (le_of_eq ?_) (single_le_sum (f := fun j => if E j cs then (1 : ℝ) else 0)
              (fun j _ => by split_ifs <;> norm_num) hi)
            simp [he]
          · exact sum_nonneg fun j _ => by split_ifs <;> norm_num
    _ = ∑ i ∈ s, ex μ n (fun cs => if E i cs then (1 : ℝ) else 0) := by
        simp only [ex, mul_sum]
        exact sum_comm

theorem joint_tail {I : Type} [DecidableEq Ω] {claims : Finset I} {κ ℓ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 ≤ κ i) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) (x : I → ℝ) :
    prob μ n (fun cs => ∃ i ∈ claims, x i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, Real.exp (-(κ i * (x i - ℓ i))) :=
  (prob_exists_le n claims _).trans
    (sum_le_sum fun i hi => tail_cum (hκ i hi) (hP i hi) (hL i hi) n h (x i))

theorem joint_tail_delta {I : Type} [DecidableEq Ω] {claims : Finset I} {κ ℓ δ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 < κ i) (hδ : ∀ i ∈ claims, 0 < δ i) (leak : I → List Ω → ℝ) (esc : List Ω → Finset Ω)
    (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) :
    prob μ n (fun cs => ∃ i ∈ claims, ℓ i + Real.log (1 / δ i) / κ i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, δ i := by
  refine (joint_tail (fun i hi => (hκ i hi).le) leak esc hP hL n h _).trans (le_of_eq (sum_congr rfl ?_))
  intro i hi
  have hk := (hκ i hi).ne'
  rw [show -(κ i * (ℓ i + Real.log (1 / δ i) / κ i - ℓ i)) = Real.log (δ i) by
    rw [one_div, Real.log_inv]; field_simp; ring]
  exact Real.exp_log (hδ i hi)

noncomputable def claimCum [DecidableEq Ω] {I : Type} (withheld : I → Prop) [DecidablePred withheld]
    (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (i : I) (n : ℕ) (h : List Ω) (cs : Fin n → Ω) : ℝ :=
  if withheld i then Strategy.cumPrev ⟨leak i, esc⟩ n h cs else Strategy.cum ⟨leak i, esc⟩ n h cs

theorem joint_tail_withheld {I : Type} [DecidableEq Ω] {claims : Finset I} {κ ℓ : I → ℝ}
    (withheld : I → Prop) [DecidablePred withheld] (hκ : ∀ i ∈ claims, 0 ≤ κ i) (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (hP : ∀ i ∈ claims, DetectionPressure μ (κ i) ⟨leak i, esc⟩)
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) (x : I → ℝ) :
    prob μ n (fun cs => ∃ i ∈ claims, x i ≤ claimCum withheld leak esc i n h cs)
      ≤ ∑ i ∈ claims, Real.exp (-(κ i * (x i - if withheld i then 0 else ℓ i))) :=
  (prob_exists_le n claims _).trans (sum_le_sum fun i hi => by
    by_cases hw : withheld i
    · simp only [claimCum, hw, ↓reduceIte, sub_zero]
      exact tail_cumPrev (hκ i hi) (hP i hi) n h (x i)
    · simp only [claimCum, hw, ↓reduceIte]
      exact tail_cum (hκ i hi) (hP i hi) (hL i hi) n h (x i))

theorem joint_of_maxRule {I U : Type} [DecidableEq Ω] {claims : Finset I} {κ ℓ δ : I → ℝ}
    (hκ : ∀ i ∈ claims, 0 < κ i) (hδ : ∀ i ∈ claims, 0 < δ i) (harm : I → U → ℝ) (leak : I → List Ω → ℝ)
    (esc : List Ω → Finset Ω) (B : List Ω → Finset U) (a : U → ℝ) (ha : ∀ u, 0 ≤ a u ∧ a u ≤ 1)
    (hmax : ∀ u, maxRule claims κ harm u ≤ a u) (hleak : ∀ i ∈ claims, ∀ h, leak i h ≤ ∑ u ∈ B h, harm i u)
    (hesc : ∀ h, Verity.Game.prPMFReal μ (· ∈ esc h) ≤ ∏ u ∈ B h, (1 - a u))
    (hL : ∀ i ∈ claims, LeakBounded (ℓ i) ⟨leak i, esc⟩) (n : ℕ) (h : List Ω) :
    prob μ n (fun cs => ∃ i ∈ claims, ℓ i + Real.log (1 / δ i) / κ i ≤ Strategy.cum ⟨leak i, esc⟩ n h cs)
      ≤ ∑ i ∈ claims, δ i :=
  joint_tail_delta hκ hδ leak esc
    (pressure_maxRule (fun i hi => (hκ i hi).le) harm leak esc B a ha hmax hleak hesc) hL n h

noncomputable def subsetSample (U : Type) [Fintype U] (k : ℕ) (hk : k ≤ Fintype.card U) : PMF (Finset U) :=
  PMF.uniformOfFinset (powersetCard k univ) (powersetCard_nonempty.2 (by rwa [card_univ]))

theorem subsetSample_toReal {U : Type} [Fintype U] (k : ℕ) (hk : k ≤ Fintype.card U) (S : Finset U) :
    (subsetSample U k hk S).toReal = if S.card = k then 1 / ((Fintype.card U).choose k : ℝ) else 0 := by
  rw [subsetSample, PMF.uniformOfFinset_apply]
  simp only [mem_powersetCard, subset_univ, true_and, card_powersetCard, card_univ]
  split_ifs <;> simp

def misses {U : Type} [Fintype U] [DecidableEq U] (B : Finset U) : Finset (Finset U) :=
  univ.filter fun S => Disjoint S B

theorem subsetSample_acceptProb {U : Type} [Fintype U] [DecidableEq U] (k : ℕ) (hk : k ≤ Fintype.card U)
    (B : Finset U) :
    Verity.Game.prPMFReal (subsetSample U k hk) (· ∈ misses B) =
      ((Fintype.card U - B.card).choose k : ℝ) / (Fintype.card U).choose k := by
  have hset : (misses B).filter (fun S => S.card = k) = Bᶜ.powersetCard k := by
    ext S
    simp [misses, mem_powersetCard, subset_compl_iff_disjoint_right]
  rw [Verity.Game.prPMFReal_mem]
  simp only [subsetSample_toReal]
  rw [← sum_filter, hset, sum_const, card_powersetCard, card_compl, nsmul_eq_mul, mul_one_div]

theorem choose_pred_ratio (M k : ℕ) (hM : 0 < M) :
    ((M - 1).choose k : ℝ) * M = (M.choose k : ℝ) * (M - k : ℕ) := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_lt hM
  simp only [zero_add, Nat.add_sub_cancel]
  exact_mod_cast Nat.choose_mul_succ_eq m k

theorem choose_ratio_le (N k : ℕ) (hk : k ≤ N) (hN : 0 < N) :
    ∀ b : ℕ, ((N - b).choose k : ℝ) / N.choose k ≤ (1 - k / N) ^ b := by
  have hc : (0 : ℝ) < N.choose k := by exact_mod_cast Nat.choose_pos hk
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have h1 : (0 : ℝ) ≤ 1 - k / N := by rw [sub_nonneg, div_le_one hNr]; exact_mod_cast hk
  rcases Nat.eq_zero_or_pos k with rfl | hk0
  · intro b; simp
  intro b
  induction b with
  | zero => simp [hc.ne']
  | succ b ih =>
    rw [pow_succ]
    rcases Nat.lt_or_ge (N - (b + 1)) k with hlt | hge
    · rw [Nat.choose_eq_zero_of_lt hlt, Nat.cast_zero, zero_div]
      exact mul_nonneg (pow_nonneg h1 _) h1
    · have hM : 0 < N - b := by omega
      have hMr : (0 : ℝ) < ((N - b : ℕ) : ℝ) := by exact_mod_cast hM
      rw [show N - (b + 1) = N - b - 1 by omega]
      have hstep : ((N - b - 1).choose k : ℝ) = ((N - b).choose k : ℝ) * (((N - b - k : ℕ) : ℝ) / (N - b : ℕ)) := by
        rw [mul_div_assoc', eq_div_iff hMr.ne']
        exact choose_pred_ratio (N - b) k hM
      have hratio : (((N - b - k : ℕ) : ℝ) / (N - b : ℕ)) ≤ 1 - k / N := by
        rw [Nat.cast_sub (show k ≤ N - b by omega), sub_div, div_self hMr.ne']
        have hle : ((N - b : ℕ) : ℝ) ≤ N := by exact_mod_cast Nat.sub_le N b
        linarith [div_le_div_of_nonneg_left (Nat.cast_nonneg k : (0 : ℝ) ≤ k) hMr hle]
      rw [hstep, mul_div_right_comm]
      exact mul_le_mul ih hratio (div_nonneg (Nat.cast_nonneg _) hMr.le) (pow_nonneg h1 _)

theorem subsetSample_acceptProb_le {U : Type} [Fintype U] [DecidableEq U] (k : ℕ) (hk : k ≤ Fintype.card U)
    (hN : 0 < Fintype.card U) (B : Finset U) :
    Verity.Game.prPMFReal (subsetSample U k hk) (· ∈ misses B) ≤ ∏ _u ∈ B, (1 - (k : ℝ) / Fintype.card U) := by
  rw [subsetSample_acceptProb, prod_const]
  exact choose_ratio_le _ k hk hN B.card

theorem pressure_subsetSample {U : Type} [Fintype U] [DecidableEq U] {κ : ℝ} (hκ : 0 ≤ κ) (k : ℕ)
    (hk : k ≤ Fintype.card U) (hN : 0 < Fintype.card U) (E : U → ℝ)
    (hκE : ∀ u, κ * E u ≤ (k : ℝ) / Fintype.card U) (A : Strategy (Finset U)) (B : List (Finset U) → Finset U)
    (hA : ∀ h, A.esc h = misses (B h)) (hleak : ∀ h, A.leak h ≤ ∑ u ∈ B h, E u) :
    DetectionPressure (subsetSample U k hk) κ A := by
  have hNr : (0 : ℝ) < Fintype.card U := by exact_mod_cast hN
  refine pressure_of_sample hκ A B E (fun _ => (k : ℝ) / Fintype.card U)
    (fun _ => ⟨div_nonneg (Nat.cast_nonneg k) hNr.le, by rw [div_le_one hNr]; exact_mod_cast hk⟩) hκE hleak
    fun h => ?_
  rw [hA h]
  exact subsetSample_acceptProb_le k hk hN (B h)

end Population.Exfil
