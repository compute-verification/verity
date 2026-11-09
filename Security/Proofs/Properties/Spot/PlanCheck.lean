module

public import Properties.Spot.PlanCheck
import Proofs.Flock.Soundness.Audit.Law
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Exfil.Bernoulli
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Laws
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The planner's checker is sound

`Properties.Spot.LeakCheckSound`, `LeakCountSound` and `WorkCheckSound`, from four facts about `Spot.PlanCheck`'s
arithmetic:

* `expSum_cast`, `expCovers_sound`: `expSum x N` is the exponential series' partial sum, at most `e^x` for `x ≥ 0`
  (`Real.sum_le_exp_of_nonneg`), so `expCovers x δ` gives `1 ≤ e^x · δ`;
* `sum_choose_le_two_pow`, `sum_choose_le_exp`: `∑_{j ≤ K} C(n, j)` is at most `2^n`, and at most `(e · n / K)^K` for
  `1 ≤ K ≤ n` (the binomial theorem at `K / n`);
* `locBits_le_upper`: `locBits n K ≤ locBitsUpper n K`, with `log₂ e ≤ log2eUp` from `ln 2 > 0.6931471803`;
* `rateCovers_sound`: a design's `rateCovers` gives `(1 − num/den)^(count + 1) < delta`, since `1 − q < e^{−q}`.

The leak check is then `bern_certifies`, its count `bern_miss_lt`, and the work check the per-unit escape `∏ (1 − q_u) ≤ e^{−λ ∑ w_u / C}`.
Each check passes at a design the planner emits (`_honest`), and each conclusion fails at a design a check accepting
everything would pass (`_nontrivial`).
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open Finset FlockSoundness.Audit FlockSoundness.Audit.Partitioning
open _root_.Spot.PlanCheck

/-! ## The exponential series -/

theorem fact_eq (n : ℕ) : fact n = n.factorial := by
  induction n with
  | zero => rfl
  | succ n ih => rw [fact, ih, Nat.factorial_succ]

theorem expSum_cast (x : ℚ) (N : ℕ) :
    ((expSum x N : ℚ) : ℝ) = ∑ i ∈ range N, (x : ℝ) ^ i / (i.factorial : ℝ) := by
  induction N with
  | zero => simp [expSum]
  | succ N ih => rw [expSum, sum_range_succ, ← ih, fact_eq]; push_cast; ring

theorem expCovers_sound {x δ : ℚ} (h : expCovers x δ = true) : 0 < (δ : ℝ) ∧ 1 ≤ Real.exp x * δ := by
  simp only [expCovers, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hδ, hy⟩, hs⟩ := h
  set y := min x ((lnCap δ : ℕ) : ℚ) with hyd
  have hyx : (y : ℝ) ≤ x := by exact_mod_cast min_le_left _ _
  have hy0 : (0 : ℝ) ≤ y := by exact_mod_cast hy
  have hδR : (0 : ℝ) < δ := by exact_mod_cast hδ
  have hsum : ((expSum y (4 * lnCap δ + 128) : ℚ) : ℝ) ≤ Real.exp y := by
    rw [expSum_cast]; exact Real.sum_le_exp_of_nonneg hy0 _
  have hs' : (1 : ℝ) ≤ ((expSum y (4 * lnCap δ + 128) : ℚ) : ℝ) * δ := by exact_mod_cast hs
  refine ⟨hδR, ?_⟩
  calc (1 : ℝ) ≤ ((expSum y (4 * lnCap δ + 128) : ℚ) : ℝ) * δ := hs'
    _ ≤ Real.exp y * δ := mul_le_mul_of_nonneg_right hsum hδR.le
    _ ≤ Real.exp x * δ := mul_le_mul_of_nonneg_right (Real.exp_le_exp.2 hyx) hδR.le

/-- `e^{-x} ≤ δ` from `1 ≤ e^x · δ`. -/
theorem exp_neg_le_of {x δ : ℝ} (h : 1 ≤ Real.exp x * δ) : Real.exp (-x) ≤ δ := by
  have h1 : Real.exp (-x) * (Real.exp x * δ) = δ := by
    rw [← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul]
  have h2 := mul_le_mul_of_nonneg_left h (Real.exp_pos (-x)).le
  rwa [h1, mul_one] at h2

/-! ## The location term -/

theorem one_le_sum_choose (n K : ℕ) : 1 ≤ ∑ j ∈ range (K + 1), n.choose j := by
  have := single_le_sum (f := fun j => n.choose j) (fun _ _ => Nat.zero_le _) (mem_range.2 (Nat.succ_pos K))
  simpa using this

theorem sum_choose_le_two_pow (n K : ℕ) : ∑ j ∈ range (K + 1), n.choose j ≤ 2 ^ n := by
  rw [← Nat.sum_range_choose n]
  calc ∑ j ∈ range (K + 1), n.choose j = ∑ j ∈ (range (K + 1)).filter (· < n + 1), n.choose j := by
        rw [sum_filter]
        refine sum_congr rfl fun j _ => ?_
        split_ifs with h
        · rfl
        · exact Nat.choose_eq_zero_of_lt (by omega)
    _ ≤ ∑ j ∈ range (n + 1), n.choose j :=
        sum_le_sum_of_subset fun j hj => by simp only [mem_filter, mem_range] at hj ⊢; omega

theorem sum_choose_le_exp {n K : ℕ} (hK : 0 < K) (hKn : K ≤ n) :
    ((∑ j ∈ range (K + 1), n.choose j : ℕ) : ℝ) ≤ (Real.exp 1 * (n / K)) ^ K := by
  have hK' : (0 : ℝ) < K := by exact_mod_cast hK
  have hn : (0 : ℝ) < n := by exact_mod_cast lt_of_lt_of_le hK hKn
  set p : ℝ := K / n with hp
  have hp0 : 0 < p := div_pos hK' hn
  have hp1 : p ≤ 1 := (div_le_one hn).2 (by exact_mod_cast hKn)
  have hbin : (1 + p) ^ n = ∑ j ∈ range (n + 1), p ^ j * (n.choose j : ℝ) := by
    rw [add_comm 1 p, add_pow]; simp only [one_pow, mul_one]
  have h1 : ((∑ j ∈ range (K + 1), n.choose j : ℕ) : ℝ) * p ^ K ≤ (1 + p) ^ n := by
    rw [hbin]
    push_cast
    rw [sum_mul]
    calc ∑ j ∈ range (K + 1), (n.choose j : ℝ) * p ^ K
        ≤ ∑ j ∈ range (K + 1), p ^ j * (n.choose j : ℝ) :=
          sum_le_sum fun j hj => by
            rw [mul_comm]
            exact mul_le_mul_of_nonneg_right
              (pow_le_pow_of_le_one hp0.le hp1 (by simp only [mem_range] at hj; omega)) (Nat.cast_nonneg _)
      _ ≤ ∑ j ∈ range (n + 1), p ^ j * (n.choose j : ℝ) :=
          sum_le_sum_of_subset_of_nonneg (fun j hj => by simp only [mem_range] at hj ⊢; omega)
            fun j _ _ => by positivity
  have h3 : (1 + p) ^ n ≤ Real.exp K := by
    calc (1 + p) ^ n ≤ Real.exp p ^ n :=
          pow_le_pow_left₀ (by positivity) (by linarith [Real.add_one_le_exp p]) n
      _ = Real.exp (n * p) := (Real.exp_nat_mul p n).symm
      _ = Real.exp K := by rw [hp]; congr 1; field_simp
  have hpK : 0 < p ^ K := pow_pos hp0 K
  have h4 : ((∑ j ∈ range (K + 1), n.choose j : ℕ) : ℝ) ≤ Real.exp K / p ^ K := by
    rw [le_div_iff₀ hpK]; linarith
  calc _ ≤ Real.exp K / p ^ K := h4
    _ = (Real.exp 1 * (n / K)) ^ K := by
      rw [show Real.exp (K : ℝ) = Real.exp 1 ^ K by rw [← Real.exp_nat_mul, mul_one], hp, mul_pow, div_pow,
        div_pow]
      field_simp

theorem log2RatioUp_spec {n K a : ℕ} (h : log2RatioUp n K = some a) :
    n ^ 2 ^ log2Prec ≤ K ^ 2 ^ log2Prec * 2 ^ a := by
  unfold log2RatioUp at h
  simp only at h
  split_ifs at h with h1 h2
  · obtain rfl := Option.some.inj h; exact h1
  · obtain rfl := Option.some.inj h; exact h2

theorem log2e_le : Real.logb 2 (Real.exp 1) ≤ (log2eUp : ℝ) := by
  have hv : (log2eUp : ℝ) = 10000000000 / 6931471803 := by unfold log2eUp; push_cast; norm_num
  rw [Real.logb, Real.log_exp, hv, div_le_div_iff₀ (Real.log_pos one_lt_two) (by norm_num)]
  have := Real.log_two_gt_d9
  norm_num at this ⊢
  linarith

theorem locBits_le_n (n K : ℕ) : locBits n K ≤ n := by
  have hpos : (0 : ℝ) < ((∑ j ∈ range (K + 1), n.choose j : ℕ) : ℝ) := by exact_mod_cast one_le_sum_choose n K
  unfold locBits
  calc Real.logb 2 _ ≤ Real.logb 2 ((2 : ℝ) ^ n) :=
        Real.logb_le_logb_of_le one_lt_two hpos (by exact_mod_cast sum_choose_le_two_pow n K)
    _ = n := by rw [Real.logb_pow, Real.logb_self_eq_one one_lt_two, mul_one]

theorem locBits_le_entropy {n K a : ℕ} (hK : 0 < K) (hKn : K < n) (h : log2RatioUp n K = some a) :
    locBits n K ≤ K * ((a : ℝ) / (2 : ℝ) ^ log2Prec + (log2eUp : ℝ)) := by
  have hK' : (0 : ℝ) < K := by exact_mod_cast hK
  have hn : (0 : ℝ) < n := by exact_mod_cast lt_trans hK hKn
  have hpos : (0 : ℝ) < ((∑ j ∈ range (K + 1), n.choose j : ℕ) : ℝ) := by exact_mod_cast one_le_sum_choose n K
  have hnK : (0 : ℝ) < n / K := div_pos hn hK'
  have hs : (0 : ℝ) < (2 : ℝ) ^ log2Prec := by positivity
  have hr : Real.logb 2 (n / K) ≤ (a : ℝ) / (2 : ℝ) ^ log2Prec := by
    rw [le_div_iff₀ hs]
    have hspec : ((n : ℝ) / K) ^ (2 ^ log2Prec) ≤ (2 : ℝ) ^ a := by
      have hc : ((n ^ 2 ^ log2Prec : ℕ) : ℝ) ≤ ((K ^ 2 ^ log2Prec * 2 ^ a : ℕ) : ℝ) := by
        exact_mod_cast log2RatioUp_spec h
      push_cast at hc
      rw [div_pow, div_le_iff₀ (by positivity)]
      linarith
    have := Real.logb_le_logb_of_le one_lt_two (by positivity) hspec
    rw [Real.logb_pow, Real.logb_pow, Real.logb_self_eq_one one_lt_two] at this
    push_cast at this
    linarith
  unfold locBits
  calc Real.logb 2 _ ≤ Real.logb 2 ((Real.exp 1 * (n / K)) ^ K) :=
        Real.logb_le_logb_of_le one_lt_two hpos (sum_choose_le_exp hK hKn.le)
    _ = K * (Real.logb 2 (Real.exp 1) + Real.logb 2 (n / K)) := by
        rw [Real.logb_pow, Real.logb_mul (Real.exp_pos 1).ne' hnK.ne']
    _ ≤ K * ((a : ℝ) / (2 : ℝ) ^ log2Prec + (log2eUp : ℝ)) := by
        apply mul_le_mul_of_nonneg_left _ hK'.le
        linarith [log2e_le]

theorem locBits_le_upper (n K : ℕ) : locBits n K ≤ (locBitsUpper n K : ℝ) := by
  unfold locBitsUpper
  split_ifs with h0 hn
  · subst h0; simp [locBits]
  · push_cast; exact locBits_le_n n K
  · split
    · next a ha =>
      push_cast
      exact le_min (locBits_le_n n K) (locBits_le_entropy (Nat.pos_of_ne_zero h0) (by omega) ha)
    · push_cast; exact locBits_le_n n K

/-! ## The leak check -/

theorem rateCovers_sound {d : Leak} (hden : d.den ≠ 0) (hnum : d.num ≤ d.den) (h : d.rateCovers = true) :
    (1 - (d.num : ℝ) / d.den) ^ (d.count + 1) < d.delta := by
  have hD : (0 : ℝ) < d.den := by exact_mod_cast Nat.pos_of_ne_zero hden
  unfold Leak.rateCovers at h
  split_ifs at h with he
  · simp only [decide_eq_true_eq] at h
    rw [he, div_self hD.ne', sub_self, zero_pow (Nat.succ_ne_zero _)]
    exact_mod_cast h
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨hδ1, hc⟩ := h
    obtain ⟨-, hexp⟩ := expCovers_sound hc
    push_cast at hexp
    have hq0 : 0 ≤ (d.num : ℝ) / d.den := by positivity
    have hq1 : (d.num : ℝ) / d.den ≤ 1 := (div_le_one hD).2 (by exact_mod_cast hnum)
    have hδ1' : (d.delta : ℝ) < 1 := by exact_mod_cast hδ1
    have hqpos : 0 < (d.num : ℝ) / d.den := by
      rcases hq0.lt_or_eq with h | h
      · exact h
      · exfalso; rw [← h, zero_mul, Real.exp_zero, one_mul] at hexp; linarith
    have hlt : 1 - (d.num : ℝ) / d.den < Real.exp (-((d.num : ℝ) / d.den)) := by
      have := Real.add_one_lt_exp (neg_ne_zero.2 hqpos.ne'); linarith
    calc (1 - (d.num : ℝ) / d.den) ^ (d.count + 1) < Real.exp (-((d.num : ℝ) / d.den)) ^ (d.count + 1) :=
          pow_lt_pow_left₀ hlt (by linarith) (Nat.succ_ne_zero _)
      _ = Real.exp (-((d.num : ℝ) / d.den * (d.count + 1))) := by
          rw [← Real.exp_nat_mul]; push_cast; ring_nf
      _ ≤ d.delta := exp_neg_le_of hexp

theorem leakCheckSound : LeakCheckSound := by
  intro d v h
  unfold leakBound at h
  split_ifs at h with hc
  obtain ⟨hden, hnum, hrate⟩ := hc
  obtain rfl := Option.some.inj h
  have hesc := rateCovers_sound hden hnum hrate
  obtain ⟨k, hk⟩ : ∃ k, d.den = k + 1 := ⟨d.den - 1, by omega⟩
  rw [hk] at hnum hesc
  refine ⟨k, hk, FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil.bern_certifies hnum hesc ?_⟩
  have := locBits_le_upper d.units d.count
  push_cast
  linarith

/-- Independent samples at `num / N` let more than `K` units escape with at most `(1 − num/N)^(K + 1)`. -/
theorem bern_miss_lt {n num N K : ℕ} [NeZero N] (h : num ≤ N) {δ : ℝ}
    (hesc : (1 - (num : ℝ) / N) ^ (K + 1) < δ) : (Law.bernoulli n num N).miss (K + 1) < ENNReal.ofReal δ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have ht0 : 0 ≤ 1 - (num : ℝ) / N := by
    rw [sub_nonneg, div_le_one hN]
    exact_mod_cast h
  have ht1 : 1 - (num : ℝ) / N ≤ 1 := by
    have : 0 ≤ (num : ℝ) / N := by positivity
    linarith
  have hδ : 0 < δ := lt_of_le_of_lt (pow_nonneg ht0 _) hesc
  have hbig : ∀ B : Finset (Fin n), K + 1 ≤ B.card →
      (Law.bernoulli n num N).escape B ≤ ENNReal.ofReal ((1 - (num : ℝ) / N) ^ (K + 1)) := by
    intro B hB
    rw [ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (Law.escape_le_one _ B))
      (pow_nonneg ht0 _), FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil.bern_escape_toReal h]
    exact pow_le_pow_of_le_one ht0 ht1 hB
  have hmiss_le : (Law.bernoulli n num N).miss (K + 1) ≤ ENNReal.ofReal ((1 - (num : ℝ) / N) ^ (K + 1)) :=
    iSup₂_le fun B hB => hbig B hB
  exact lt_of_le_of_lt hmiss_le ((ENNReal.ofReal_lt_ofReal_iff hδ).2 hesc)

theorem leakCountSound : LeakCountSound := by
  intro d v h
  unfold leakBound at h
  split_ifs at h with hc
  obtain ⟨hden, hnum, hrate⟩ := hc
  have hesc := rateCovers_sound hden hnum hrate
  obtain ⟨k, hk⟩ : ∃ k, d.den = k + 1 := ⟨d.den - 1, by omega⟩
  rw [hk] at hnum hesc
  exact ⟨k, hk, bern_miss_lt hnum hesc⟩

/-! ## The work check -/

theorem lamOf_none {den : ℕ} {C : ℚ} : ∀ {ss : List Rate}, lamOf den C ss = none →
    ∀ r ∈ ss, r.num < den → r.work ≤ 0
  | [], _ => by simp
  | r :: rs, h => by
    intro s hs hsn
    by_cases hr : r.num < den ∧ 0 < r.work
    · simp [lamOf, hr] at h
    · simp only [lamOf, hr, ↓reduceIte] at h
      rcases List.mem_cons.1 hs with rfl | hs
      · by_contra hw; exact hr ⟨hsn, lt_of_not_ge hw⟩
      · exact lamOf_none h s hs hsn

theorem lamOf_nonneg {den : ℕ} {C : ℚ} (hC : 0 ≤ C) : ∀ {ss : List Rate} {l : ℚ}, lamOf den C ss = some l → 0 ≤ l
  | [], _, h => by simp [lamOf] at h
  | r :: rs, l, h => by
    simp only [lamOf] at h
    split_ifs at h with hr
    · have hv : 0 ≤ (r.num : ℚ) / den * C / r.work :=
        div_nonneg (mul_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) hC) hr.2.le
      cases hrest : lamOf den C rs with
      | none => simp only [hrest, Option.some.injEq] at h; subst h; exact hv
      | some a => simp only [hrest, Option.some.injEq] at h; subst h; exact le_min (lamOf_nonneg hC hrest) hv
    · exact lamOf_nonneg hC h

theorem lamOf_le {den : ℕ} {C : ℚ} : ∀ {ss : List Rate} {l : ℚ}, lamOf den C ss = some l →
    ∀ r ∈ ss, r.num < den → 0 < r.work → l ≤ (r.num : ℚ) / den * C / r.work
  | [], _, h => by simp [lamOf] at h
  | r :: rs, l, h => by
    intro s hs hsn hsw
    simp only [lamOf] at h
    split_ifs at h with hr
    · cases hrest : lamOf den C rs with
      | none =>
        simp only [hrest, Option.some.injEq] at h; subst h
        rcases List.mem_cons.1 hs with rfl | hs
        · exact le_rfl
        · exact absurd (lamOf_none hrest s hs hsn) (not_le.2 hsw)
      | some a =>
        simp only [hrest, Option.some.injEq] at h; subst h
        rcases List.mem_cons.1 hs with rfl | hs
        · exact min_le_right _ _
        · exact (min_le_left _ _).trans (lamOf_le hrest s hs hsn hsw)
    · rcases List.mem_cons.1 hs with rfl | hs
      · exact absurd ⟨hsn, hsw⟩ hr
      · exact lamOf_le h s hs hsn hsw

theorem workCheckSound : WorkCheckSound := by
  intro d h
  simp only [checkWork, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hden, hC⟩, hδ0⟩, -⟩, hall⟩, hlam⟩ := h
  obtain ⟨k, hk⟩ : ∃ k, d.den = k + 1 := ⟨d.den - 1, by omega⟩
  refine ⟨k, hk, fun n σ hσ B hB => ?_⟩
  have hst : ∀ u, (σ u).num ≤ d.den ∧ 0 ≤ (σ u).work := fun u => by simpa using hall _ (hσ u)
  have hD : (0 : ℝ) < d.den := by exact_mod_cast Nat.pos_of_ne_zero hden
  have hCR : (0 : ℝ) < d.target := by exact_mod_cast hC
  rw [FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal.bern_escape_toReal (num := fun u => (σ u).num)
    (den := k + 1) (fun u => by rw [← hk]; exact (hst u).1), ← hk]
  have hfac0 : ∀ u, 0 ≤ 1 - ((σ u).num : ℝ) / d.den := fun u => by
    rw [sub_nonneg, div_le_one hD]; exact_mod_cast (hst u).1
  by_cases hfull : ∃ u ∈ B, (σ u).num = d.den
  · obtain ⟨u, hu, he⟩ := hfull
    rw [prod_eq_zero hu (by rw [he, div_self hD.ne', sub_self])]
    exact_mod_cast hδ0.le
  push Not at hfull
  have hlt : ∀ u ∈ B, (σ u).num < d.den := fun u hu => lt_of_le_of_ne (hst u).1 (hfull u hu)
  revert hlam
  cases hl : lamOf d.den d.target d.strata with
  | none =>
    intro _
    exfalso
    have hw : ∀ u ∈ B, ((σ u).work : ℝ) ≤ 0 := fun u hu => by
      exact_mod_cast lamOf_none hl _ (hσ u) (hlt u hu)
    have : ∑ u ∈ B, ((σ u).work : ℝ) ≤ 0 := sum_nonpos hw
    linarith
  | some l =>
    intro hlam
    obtain ⟨-, hexp⟩ := expCovers_sound hlam
    have hl0 : (0 : ℝ) ≤ l := by exact_mod_cast lamOf_nonneg hC.le hl
    have hq : ∀ u ∈ B, (l : ℝ) * (σ u).work / d.target ≤ ((σ u).num : ℝ) / d.den := fun u hu => by
      rcases (hst u).2.lt_or_eq with hw | hw
      · have hR : ((l : ℚ) : ℝ) ≤ ((((σ u).num : ℚ) / d.den * d.target / (σ u).work : ℚ) : ℝ) := by
          exact_mod_cast lamOf_le hl _ (hσ u) (hlt u hu) hw
        push_cast at hR
        have hwR : (0 : ℝ) < (σ u).work := by exact_mod_cast hw
        rw [div_le_iff₀ hCR]
        calc (l : ℝ) * (σ u).work ≤ ((σ u).num / d.den * d.target / (σ u).work) * (σ u).work :=
              mul_le_mul_of_nonneg_right hR hwR.le
          _ = (σ u).num / d.den * d.target := by field_simp
      · rw [← hw]; push_cast; rw [mul_zero, zero_div]; positivity
    have hfac : ∀ u ∈ B, 1 - ((σ u).num : ℝ) / d.den ≤ Real.exp (-((l : ℝ) * (σ u).work / d.target)) :=
      fun u hu => by
        have h1 := Real.add_one_le_exp (-(((σ u).num : ℝ) / d.den))
        have h2 := Real.exp_le_exp.2 (neg_le_neg (hq u hu))
        linarith
    calc ∏ u ∈ B, (1 - ((σ u).num : ℝ) / d.den) ≤ ∏ u ∈ B, Real.exp (-((l : ℝ) * (σ u).work / d.target)) :=
          prod_le_prod₀ (fun u _ => hfac0 u) hfac
      _ = Real.exp (-((l : ℝ) / d.target * ∑ u ∈ B, ((σ u).work : ℝ))) := by
          rw [← Real.exp_sum, mul_sum, ← sum_neg_distrib]
          exact congrArg Real.exp (sum_congr rfl fun u _ => by ring)
      _ ≤ Real.exp (-(l : ℝ)) := by
          apply Real.exp_le_exp.2
          have : (l : ℝ) ≤ (l : ℝ) / d.target * ∑ u ∈ B, ((σ u).work : ℝ) := by
            rw [div_mul_eq_mul_div, le_div_iff₀ hCR]; exact mul_le_mul_of_nonneg_left hB hl0
          linarith
      _ ≤ d.delta := exp_neg_le_of hexp

/-! ## Witnesses -/

/-- The leak check passes at a planner's design: `1000` units of `8` bits sampled at `1/2` certify `9` incorrect ones at
`δ = 1/100`. -/
theorem leakCheckSound_honest :
    (leakBound { units := 1000, width := 8, num := 1, den := 2, count := 9, delta := 1 / 100 }).isSome = true := by
  decide +kernel

/-- No design certifies `−1` bits: the conclusion fails where a check accepting everything would return `−1`. -/
theorem leakCheckSound_nontrivial :
    ∀ k : ℕ, ¬ CertifiesLeak (Law.bernoulli 1 0 (k + 1)) (fun _ => 0) ((1 / 2 : ℚ) : ℝ) ((-1 : ℚ) : ℝ) := by
  intro _ ⟨H, K, _, _, hℓ⟩
  have h0 : 0 ≤ locBits 1 K := by
    unfold locBits
    exact Real.logb_nonneg one_lt_two (by exact_mod_cast one_le_sum_choose 1 K)
  have : (0 : ℝ) ≤ H := Nat.cast_nonneg H
  push_cast at hℓ
  linarith

/-- The leak check passes at the design that samples every proof block, which certifies no incorrect one. -/
theorem leakCountSound_honest :
    (leakBound { units := 64, width := 1, num := 1, den := 1, count := 0, delta := 1 / 100 }).isSome = true := by
  decide +kernel

/-- A unit never sampled always escapes: the count's conclusion fails where a check accepting everything would pass. -/
theorem leakCountSound_nontrivial :
    ∀ k : ℕ, ¬ (Law.bernoulli 1 0 (k + 1)).miss (0 + 1) < ENNReal.ofReal ((1 / 2 : ℚ) : ℝ) := by
  intro k h
  have he := Law.escape_le_miss (Law.bernoulli 1 0 (k + 1)) (K := 0 + 1) (B := {0}) (by simp)
  have h1 : ((Law.bernoulli 1 0 (k + 1)).escape {0}).toReal = 1 := by
    rw [FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil.bern_escape_toReal (Nat.zero_le _)]
    simp
  have hlt := lt_of_le_of_lt he h
  have := (ENNReal.toReal_lt_toReal (ne_top_of_lt hlt) ENNReal.ofReal_ne_top).2 hlt
  rw [h1, ENNReal.toReal_ofReal (by norm_num)] at this
  norm_num at this

/-- The work check passes at a planner's design: two strata at rates proportional to their work. -/
theorem workCheckSound_honest :
    checkWork { strata := [⟨1, 1⟩, ⟨2, 2⟩], den := 1000, target := 4606, delta := 1 / 100 } = true := by
  decide +kernel

/-- A stratum never sampled doesn't meet a target: its one unit, doing all the work, always escapes. -/
theorem workCheckSound_nontrivial :
    ∀ k : ℕ, ¬ MeetsTarget (Law.bernoulliOf (n := 1) (fun _ => 0) (k + 1)) (fun _ => ((1 : ℚ) : ℝ))
      ((1 : ℚ) : ℝ) ((1 / 2 : ℚ) : ℝ) := by
  intro _ h
  have := h {0} (by simp)
  rw [FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal.bern_escape_toReal (fun _ => Nat.zero_le _)] at this
  norm_num at this

end Properties.Spot.SecurityProofs
