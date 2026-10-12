import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.Fp8Atom.H1TLemmas
import Proofs.Pouw.Fp8Atom.Proofs
import Proofs.Pouw.Fp8Atom.H1TProofs

/-!
# The atom as two truncations (`H1TAtomValue`)

With the accumulator `+0` and no NaN code, `hopperStep` sums the 32 exact products in one group: each is truncated
toward zero to the grid `2^(E − 13)` of the largest product exponent `E` (`aligned_eq_truncZ`), the truncated products
are summed exactly, and the sum is truncated toward zero to 14 significant bits (`normalize`). Every nonzero product
has an exponent in `[−12, 16]` and every aligned product is below `2^15`, so the sum's exponent is in `[−25, 23]`: no
saturation, no subnormal, and `packTerm` writes the term exactly.
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## The step on the zero accumulator -/

theorem fp32Value_zero : fp32Value 0 = .fin Term.zero := rfl

theorem zip_ofFn (a b : Fin 32 → Code) : List.zip (List.ofFn a) (List.ofFn b) = List.ofFn fun l => (a l, b l) := by
  apply List.ext_getElem <;> simp

theorem hopperStep_zero (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    hopperStep 0 a b = packValue (groupStepTotal 14 (-139) (.fin Term.zero)
      ((List.ofFn fun l => product (a l) (b l)).map Value.fin)) := by
  unfold hopperStep Pipeline.step hopperE4m3K32
  simp only [Pipeline.fold, fp32Value_zero, zip_ofFn]
  congr 2
  rw [List.take_of_length_le (by simp), List.map_ofFn, List.map_ofFn]
  congr 1
  funext l
  simp [productTotal, ha l, hb l]

theorem groupStepTotal_fin (ts : List Term) :
    groupStepTotal 14 (-139) (.fin Term.zero) (ts.map Value.fin) =
      let t := groupSum 14 (-139) (Term.zero :: ts)
      if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t := by
  unfold groupStepTotal
  have h1 : ((Value.fin Term.zero :: ts.map Value.fin).any Value.isNaN) = false := by
    simp [List.any_map, Function.comp_def, Value.isNaN]
  have h2 : ∀ s, ((Value.fin Term.zero :: ts.map Value.fin).any (Value.isInf s)) = false := by
    intro s; simp [List.any_map, Function.comp_def, Value.isInf]
  have h3 : (Value.fin Term.zero :: ts.map Value.fin).filterMap Value.finite? = Term.zero :: ts := by
    simp [List.filterMap_map, Function.comp_def, Value.finite?]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.and_self]

/-! ## The alignment exponent -/

theorem maxExp_ge (z : ℤ) (ts : List Term) : z ≤ maxExp z ts := by
  induction ts generalizing z with
  | nil => exact le_refl _
  | cons t ts ih =>
    unfold maxExp at *
    rw [List.foldl_cons]
    refine le_trans ?_ (ih _)
    split_ifs with h
    · exact h.2.le
    · exact le_refl _

theorem le_maxExp (z : ℤ) (ts : List Term) : ∀ t ∈ ts, t.mag ≠ 0 → t.exp ≤ maxExp z ts := by
  induction ts generalizing z with
  | nil => intro t ht; cases ht
  | cons s ts ih =>
    intro t ht hm
    unfold maxExp at *
    rw [List.foldl_cons]
    rcases List.mem_cons.1 ht with rfl | ht
    · refine le_trans ?_ (maxExp_ge _ ts)
      split_ifs with h
      · exact le_refl _
      · exact not_lt.1 fun h' => h ⟨hm, h'⟩
    · exact ih _ t ht hm

theorem maxExp_le (z B : ℤ) (ts : List Term) (hz : z ≤ B) (h : ∀ t ∈ ts, t.mag ≠ 0 → t.exp ≤ B) :
    maxExp z ts ≤ B := by
  induction ts generalizing z with
  | nil => exact hz
  | cons s ts ih =>
    unfold maxExp at *
    rw [List.foldl_cons]
    apply ih
    · split_ifs with h'
      · exact h s (List.mem_cons_self) h'.1
      · exact hz
    · exact fun t ht => h t (List.mem_cons_of_mem _ ht)

theorem maxExp_zero_cons (z : ℤ) (ts : List Term) : maxExp z (Term.zero :: ts) = maxExp z ts := by
  unfold maxExp; simp [Term.zero]

/-! ## The first truncation, on values -/

theorem truncTo_mul {g : ℚ} (hg : 0 < g) (q : ℚ) : truncTo g (q * g) = (truncZ q : ℚ) * g := by
  unfold truncTo truncZ
  rw [abs_mul, abs_of_pos hg, mul_div_cancel_right₀ _ hg.ne']
  rcases lt_or_ge q 0 with h | h
  · have h' : q * g < 0 := mul_neg_of_neg_of_pos h hg
    rw [ite_eq_left h', ite_eq_right (not_le.2 h), abs_of_neg h, ← Int.natCast_floor_eq_floor (by linarith)]
    push_cast; ring
  · have h' : ¬ q * g < 0 := not_lt.2 (mul_nonneg h hg.le)
    rw [ite_eq_right h', ite_eq_left h, abs_of_nonneg h, ← Int.natCast_floor_eq_floor h]
    push_cast; ring

theorem truncTo_zero (g : ℚ) : truncTo g 0 = 0 := by simp [truncTo]

theorem two_zpow_ne (k : ℤ) : (2 : ℚ) ^ k ≠ 0 := zpow_ne_zero _ two_ne_zero

/-- An aligned term, on the grid `2^(E − 13)`, is its value truncated toward zero to the grid. -/
theorem aligned_value {E : ℤ} {t : Term} (hE : t.mag ≠ 0 → t.exp ≤ E) :
    (aligned 14 E t : ℚ) * 2 ^ (E - 13) = truncTo (2 ^ (E - 13)) t.value := by
  by_cases hm : t.mag = 0
  · have : t.value = 0 := by simp [Term.value, hm]
    rw [this, truncTo_zero]; simp [aligned, hm]
  · have hv := value_eq (w := 14) (by norm_num) (hE hm)
    rw [show (E - (14 : ℕ) + 1 : ℤ) = E - 13 by push_cast; ring] at hv
    rw [hv, aligned_eq_truncZ (by norm_num), truncTo_mul (two_zpow_pos _)]

/-! ## The word -/

theorem fp32Value_packTerm {t : Term} (h1 : 2 ^ 23 ≤ t.mag) (h2 : t.mag < 2 ^ 24) (e1 : -126 ≤ t.exp)
    (e2 : t.exp ≤ 127) : fp32Value (packTerm t) = .fin t := by
  obtain ⟨neg, exp, mag⟩ := t
  simp only at h1 h2 e1 e2
  have hm0 : mag ≠ 0 := by omega
  have hb : mag &&& (1 <<< 23) ≠ 0 := by
    rw [Nat.one_shiftLeft, Nat.and_two_pow]
    have : mag.testBit 23 = true := by
      rw [Nat.testBit_eq_decide_div_mod_eq, decide_eq_true_iff]; omega
    simp [this]
  have hlow : mag &&& 0x7FFFFF = mag % 2 ^ 23 := by
    rw [show (0x7FFFFF : ℕ) = 2 ^ 23 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod]
  have hm : 1 <<< 23 ||| mag % 2 ^ 23 = mag := by
    rw [← Nat.shiftLeft_add_eq_or_of_lt (Nat.mod_lt _ (by norm_num)), Nat.one_shiftLeft]; omega
  unfold packTerm
  simp only [hm0, ite_false, not_lt.2 e2, hb, hlow]
  have hef : ((exp + 127 - 0).toNat : ℤ) = exp + 127 := by omega
  generalize (exp + 127 - 0).toNat = ef at hef
  have hlt : ef * 2 ^ 23 + mag % 2 ^ 23 < 2 ^ 31 := by
    have : ef ≤ 254 := by omega
    have : mag % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
    nlinarith
  have hw : ((if neg then 1 else 0) <<< 31 ||| ef <<< 23 ||| mag % 2 ^ 23) =
      (if neg then 1 else 0) * 2 ^ 31 + (ef * 2 ^ 23 + mag % 2 ^ 23) := by
    rw [Nat.or_assoc, ← Nat.shiftLeft_add_eq_or_of_lt (Nat.mod_lt _ (by norm_num)),
      ← Nat.shiftLeft_add_eq_or_of_lt (by rw [Nat.shiftLeft_eq]; exact hlt), Nat.shiftLeft_eq, Nat.shiftLeft_eq]
  rw [hw]
  unfold fp32Value
  simp only [Nat.shiftRight_eq_div_pow]
  have hm2 : mag % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  generalize mag % 2 ^ 23 = r at hm2 hm hlt ⊢
  have hs : (if neg then 1 else 0) ≤ 1 := by split <;> omega
  generalize hsd : (if neg then 1 else 0) = s at hs
  have ha : (s * 2 ^ 31 + (ef * 2 ^ 23 + r)) / 2 ^ 23 % 256 = ef := by
    norm_num only at hlt hm2 ⊢; omega
  have hc : (s * 2 ^ 31 + (ef * 2 ^ 23 + r)) % 2 ^ 23 = r := by
    norm_num only at hlt hm2 ⊢; omega
  have hd : (s * 2 ^ 31 + (ef * 2 ^ 23 + r)) / 2 ^ 31 % 2 = s := by
    norm_num only at hlt hm2 ⊢; omega
  rw [ha, hc, hd, ite_eq_right (by omega), ite_eq_right (by omega), hm]
  have hn : (s == 1) = neg := by cases neg <;> simp at hsd <;> simp [← hsd]
  rw [hn]
  congr 2; omega

theorem wordQ_packTerm {t : Term} (h1 : 2 ^ 23 ≤ t.mag) (h2 : t.mag < 2 ^ 24) (e1 : -126 ≤ t.exp)
    (e2 : t.exp ≤ 127) : wordQ (packTerm t) = some t.value := by
  simp [wordQ, fp32Value_packTerm h1 h2 e1 e2, Value.finite?]

/-! ## The second truncation -/

theorem floor_shift (n L : ℕ) : (⌊(n : ℚ) * 2 ^ ((13 : ℤ) - L)⌋ : ℚ) =
    ((if 14 < L + 1 then n >>> (L + 1 - 14) else n <<< (14 - (L + 1)) : ℕ) : ℚ) := by
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, ← Nat.floor_div_eq_div (K := ℚ)]
    have : (13 : ℤ) - L = -((L + 1 - 14 : ℕ) : ℤ) := by omega
    rw [this, zpow_neg, zpow_natCast, ← div_eq_mul_inv]
    have hnn : (0 : ℚ) ≤ (n : ℚ) / 2 ^ (L + 1 - 14) := by positivity
    rw [← Int.natCast_floor_eq_floor hnn]
    push_cast; rfl
  · rw [Nat.shiftLeft_eq]
    have : (13 : ℤ) - L = ((14 - (L + 1) : ℕ) : ℤ) := by omega
    rw [this, zpow_natCast]
    have e : (n : ℚ) * 2 ^ (14 - (L + 1)) = ((n * 2 ^ (14 - (L + 1)) : ℕ) : ℤ) := by push_cast; ring
    rw [e, Int.floor_intCast]; push_cast; ring

theorem binade_nat_mul {n : ℕ} (hn : n ≠ 0) (k : ℤ) : binade ((n : ℚ) * 2 ^ k) = n.log2 + k := by
  have h1 := Nat.log2_self_le hn
  have h2 := @Nat.lt_log2_self n
  have hq : (0 : ℚ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  apply binade_unique (mul_pos hq (two_zpow_pos _))
  · rw [zpow_add₀ two_ne_zero, zpow_natCast]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast h1) (two_zpow_pos _).le
  · rw [show (n.log2 : ℤ) + k + 1 = ((n.log2 + 1 : ℕ) : ℤ) + k by push_cast; ring, zpow_add₀ two_ne_zero, zpow_natCast]
    exact mul_lt_mul_of_pos_right (by exact_mod_cast h2) (two_zpow_pos _)

/-- `normalize` on a nonzero sum at an exponent in `[−12, 16]`, below `2^21`: a normal term, the sum truncated toward
zero to 14 significant bits. -/
theorem normalize_spec {E total : ℤ} (h0 : total ≠ 0) (hE1 : -12 ≤ E) (hE2 : E ≤ 16) (hT : total.natAbs < 2 ^ 21) :
    2 ^ 23 ≤ (normalize 14 E total).mag ∧ (normalize 14 E total).mag < 2 ^ 24 ∧
      -126 ≤ (normalize 14 E total).exp ∧ (normalize 14 E total).exp ≤ 127 ∧
      (normalize 14 E total).value = trunc14 ((total : ℚ) * 2 ^ (E - 13)) := by
  have hn : total.natAbs ≠ 0 := by omega
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self total.natAbs
  have hL : total.natAbs.log2 < 21 := by
    by_contra h
    have : 2 ^ 21 ≤ 2 ^ total.natAbs.log2 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hq := floor_shift total.natAbs total.natAbs.log2
  set L := total.natAbs.log2 with hLdef
  set m0 := (if 14 < L + 1 then total.natAbs >>> (L + 1 - 14) else total.natAbs <<< (14 - (L + 1)))
    with hm0
  have hb1 : 2 ^ 13 ≤ m0 ∧ m0 < 2 ^ 14 := by
    rw [hm0]; split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow]
      constructor
      · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]; rwa [show 13 + (L + 1 - 14) = L by omega]
      · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]; rwa [show 14 + (L + 1 - 14) = L + 1 by omega]
    · rw [Nat.shiftLeft_eq]
      constructor
      · calc 2 ^ 13 = 2 ^ L * 2 ^ (14 - (L + 1)) := by rw [← pow_add]; congr 1; omega
          _ ≤ _ := Nat.mul_le_mul_right _ hL1
      · calc total.natAbs * 2 ^ (14 - (L + 1)) < 2 ^ (L + 1) * 2 ^ (14 - (L + 1)) :=
              Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
          _ = 2 ^ 14 := by rw [← pow_add]; congr 1; omega
  have hbl : bitLength total.natAbs = L + 1 := by unfold bitLength; rw [ite_eq_right hn]
  have hne : ¬ (E + ((L + 1 : ℕ) : ℤ) - ((14 : ℕ) : ℤ) < -126) := by omega
  unfold normalize
  simp only [h0, ite_false, hbl, hne, ← hm0]
  have hfa : fromAdder 14 m0 = m0 * 2 ^ 10 := by simp [fromAdder, Nat.shiftLeft_eq]
  rw [hfa, ite_eq_right (by omega)]
  dsimp only
  refine ⟨by omega, by omega, by omega, by omega, ?_⟩
  -- the value
  have hy0 : (total : ℚ) * 2 ^ (E - 13) ≠ 0 := mul_ne_zero (by exact_mod_cast h0) (two_zpow_ne _)
  have habs : |(total : ℚ) * 2 ^ (E - 13)| = (total.natAbs : ℚ) * 2 ^ (E - 13) := by
    rw [abs_mul, abs_of_pos (two_zpow_pos _), Nat.cast_natAbs, Int.cast_abs]
  have hbin : binade |(total : ℚ) * 2 ^ (E - 13)| = L + (E - 13) := by
    rw [habs, binade_nat_mul hn]
  unfold trunc14 truncTo Term.value
  rw [ite_eq_right hy0, hbin, habs]
  have hdiv : (total.natAbs : ℚ) * 2 ^ (E - 13) / 2 ^ ((L : ℤ) + (E - 13) - 13) =
      (total.natAbs : ℚ) * 2 ^ ((13 : ℤ) - L) := by
    rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; ring
  rw [hdiv, hq]
  have hs : (total : ℚ) * 2 ^ (E - 13) < 0 ↔ total < 0 := by
    rw [mul_neg_iff]; constructor
    · rintro (⟨-, h⟩ | ⟨h, -⟩)
      · exact absurd h (not_lt.2 (two_zpow_pos _).le)
      · exact_mod_cast h
    · intro h; right; exact ⟨by exact_mod_cast h, two_zpow_pos _⟩
  have hpow : (2 : ℚ) ^ (E + ((L + 1 : ℕ) : ℤ) - ((14 : ℕ) : ℤ) - 23) =
      2 ^ ((L : ℤ) + (E - 13) - 13) / 2 ^ 10 := by
    rw [eq_div_iff (by positivity), ← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  rw [hpow]
  by_cases hneg : total < 0
  · simp only [hneg, decide_true, ite_true, hs.2 hneg]
    push_cast; field_simp; norm_num
  · simp only [hneg, decide_false, ite_false, Bool.false_eq_true, (not_congr hs).2 hneg]
    push_cast; field_simp; norm_num

/-! ## The atom -/

theorem sig_le : ∀ c : Code, c.sig ≤ 15 := by decide +kernel

theorem expOr1_le : ∀ c : Code, c.expOr1 ≤ 15 := by decide +kernel

theorem product_mag_lt (a b : Code) : (product a b).mag < 2 ^ 25 := by
  unfold product
  dsimp only
  split_ifs
  · simp [Term.zero]
  · dsimp only
    rw [Nat.shiftLeft_eq]
    have := Nat.mul_le_mul (sig_le a) (sig_le b)
    omega

theorem product_exp (a b : Code) (h : (product a b).mag ≠ 0) :
    -12 ≤ (product a b).exp ∧ (product a b).exp ≤ 16 := by
  unfold product at *
  dsimp only at *
  split_ifs at * with h'
  · simp [Term.zero] at h
  · dsimp only
    have := expOr1_pos a; have := expOr1_pos b; have := expOr1_le a; have := expOr1_le b
    omega

theorem aligned_natAbs_lt (E : ℤ) (t : Term) (hm : t.mag < 2 ^ 25) : (aligned 14 E t).natAbs < 2 ^ 15 := by
  rw [aligned_eq (by norm_num)]
  have h1 : t.mag / 2 ^ shiftOf 14 E t ≤ t.mag / 2 ^ 10 :=
    Nat.div_le_div_left (Nat.pow_le_pow_right (by norm_num) (by unfold shiftOf; omega)) (by positivity)
  have h2 : t.mag / 2 ^ 10 < 2 ^ 15 := by omega
  split <;> simp only [neg_one_mul, one_mul, Int.natAbs_neg, Int.natAbs_natCast] <;> omega

theorem val_eq {c : Code} (h : c.isNaN = false) : decode c = some (val c) := by
  rw [val, decode_eq c h]; rfl

theorem alignedSum_eq (E : ℤ) (a b : Fin 32 → Code) :
    alignedSum 14 E (Term.zero :: List.ofFn fun l => product (a l) (b l)) =
      ∑ l, aligned 14 E (product (a l) (b l)) := by
  have hz : aligned 14 E Term.zero = 0 := by simp [aligned, Term.zero]
  rw [alignedSum, List.map_cons, List.sum_cons, hz, zero_add, List.map_ofFn, List.sum_ofFn]
  simp only [Function.comp_apply]

/-- **The atom is two truncations.** -/
theorem h1tAtomValue : H1TAtomValue := by
  intro a b ha hb
  rw [hopperStep_zero a b ha hb, groupStepTotal_fin]
  have hE : maxExp (-139) (Term.zero :: List.ofFn fun l => product (a l) (b l)) = gridExp a b :=
    maxExp_zero_cons _ _
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ gridExp a b := fun l hm =>
    le_maxExp _ _ _ (List.mem_ofFn.2 ⟨l, rfl⟩) hm
  have htot : ((alignedSum 14 (gridExp a b) (Term.zero :: List.ofFn fun l => product (a l) (b l)) : ℤ) : ℚ) *
      2 ^ (gridExp a b - 13) = preSum a b := by
    rw [alignedSum_eq, preSum]; push_cast; rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_value (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  unfold groupSum
  rw [hE]
  have hsum := alignedSum_eq (gridExp a b) a b
  generalize alignedSum 14 (gridExp a b) (Term.zero :: List.ofFn fun l => product (a l) (b l)) = total at htot hsum
  by_cases h0 : total = 0
  · subst h0
    have : preSum a b = 0 := by rw [← htot]; simp
    rw [this]
    simp [normalize, Term.zero, packValue, packTerm, wordQ, fp32Value_zero, Value.finite?, Term.value, trunc14]
  · have hex : ∃ l, (product (a l) (b l)).mag ≠ 0 := by
      by_contra hc
      simp only [not_exists, not_not] at hc
      apply h0
      rw [hsum]
      exact Finset.sum_eq_zero fun l _ => by simp [aligned, hc l]
    obtain ⟨l0, hl0⟩ := hex
    have hE1 : -12 ≤ gridExp a b := le_trans (product_exp _ _ hl0).1 (hle l0 hl0)
    have hE2 : gridExp a b ≤ 16 := by
      apply maxExp_le _ _ _ (by norm_num)
      intro t ht hm
      obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      exact (product_exp _ _ hm).2
    have hT : total.natAbs < 2 ^ 21 := by
      have h1 : |total| ≤ ∑ l : Fin 32, |aligned 14 (gridExp a b) (product (a l) (b l))| := by
        rw [hsum]; exact Finset.abs_sum_le_sum_abs _ _
      have h2 : ∑ l : Fin 32, |aligned 14 (gridExp a b) (product (a l) (b l))| ≤ ∑ _l : Fin 32, (2 ^ 15 : ℤ) := by
        refine Finset.sum_le_sum fun l _ => ?_
        have := aligned_natAbs_lt (gridExp a b) (product (a l) (b l)) (product_mag_lt _ _)
        rw [Int.abs_eq_natAbs]; omega
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
      rw [Int.abs_eq_natAbs] at h1
      omega
    obtain ⟨m1, m2, e1, e2, hv⟩ := normalize_spec h0 hE1 hE2 hT
    rw [ite_eq_right (fun h => absurd h.2 (not_lt.2 e2))]
    simp only [packValue]
    rw [wordQ_packTerm m1 m2 e1 e2, hv, htot]

/-! ## `trunc14`: monotone, and its cell -/

/-- `trunc14` on a positive value. -/
def tpos (t : ℚ) : ℚ := (⌊t / 2 ^ (binade t - 13)⌋ : ℚ) * 2 ^ (binade t - 13)

theorem trunc14_pos {t : ℚ} (ht : 0 < t) : trunc14 t = tpos t := by
  unfold trunc14 truncTo tpos
  rw [ite_eq_right ht.ne', abs_of_pos ht, ite_eq_right (not_lt.2 ht.le), one_mul]

theorem trunc14_neg (y : ℚ) : trunc14 (-y) = -trunc14 y := by
  unfold trunc14 truncTo
  rcases lt_trichotomy y 0 with h | h | h
  · rw [ite_eq_right (by linarith), ite_eq_right h.ne, abs_neg, ite_eq_right (by linarith), ite_eq_left h]; ring
  · subst h; simp
  · rw [ite_eq_right (by linarith), ite_eq_right h.ne', abs_neg, ite_eq_left (by linarith),
      ite_eq_right (not_lt.2 h.le)]; ring

theorem tpos_spec {t : ℚ} (ht : 0 < t) :
    2 ^ binade t ≤ tpos t ∧ tpos t ≤ t ∧ t - tpos t < 2 ^ (binade t - 13) := by
  obtain ⟨b1, b2⟩ := binade_spec ht
  have hG := two_zpow_pos (binade t - 13)
  have hq : (2 : ℚ) ^ 13 ≤ t / 2 ^ (binade t - 13) := by
    rw [le_div_iff₀ hG, ← zpow_natCast, ← zpow_add₀ two_ne_zero]; convert b1 using 2; push_cast; ring
  have hf1 := Int.floor_le (t / 2 ^ (binade t - 13))
  have hf2 := Int.lt_floor_add_one (t / 2 ^ (binade t - 13))
  have hf3 : (2 : ℚ) ^ 13 ≤ ⌊t / 2 ^ (binade t - 13)⌋ := by
    have : (2 ^ 13 : ℤ) ≤ ⌊t / 2 ^ (binade t - 13)⌋ := Int.le_floor.2 (by exact_mod_cast hq)
    exact_mod_cast this
  unfold tpos
  refine ⟨?_, ?_, ?_⟩
  · calc (2 : ℚ) ^ binade t = 2 ^ 13 * 2 ^ (binade t - 13) := by
          rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
      _ ≤ _ := mul_le_mul_of_nonneg_right hf3 hG.le
  · calc _ ≤ t / 2 ^ (binade t - 13) * 2 ^ (binade t - 13) := mul_le_mul_of_nonneg_right hf1 hG.le
      _ = t := div_mul_cancel₀ _ hG.ne'
  · have : t < ((⌊t / 2 ^ (binade t - 13)⌋ : ℚ) + 1) * 2 ^ (binade t - 13) := by
      calc t = t / 2 ^ (binade t - 13) * 2 ^ (binade t - 13) := (div_mul_cancel₀ _ hG.ne').symm
        _ < _ := mul_lt_mul_of_pos_right hf2 hG
    linarith

theorem tpos_pos {t : ℚ} (ht : 0 < t) : 0 < tpos t := lt_of_lt_of_le (two_zpow_pos _) (tpos_spec ht).1

theorem binade_tpos {t : ℚ} (ht : 0 < t) : binade (tpos t) = binade t := by
  obtain ⟨h1, h2, -⟩ := tpos_spec ht
  exact binade_unique (tpos_pos ht) h1 (lt_of_le_of_lt h2 (binade_spec ht).2)

theorem tpos_mono {s t : ℚ} (hs : 0 < s) (h : s ≤ t) : tpos s ≤ tpos t := by
  have ht : 0 < t := lt_of_lt_of_le hs h
  have hb := binade_mono hs h
  rcases hb.lt_or_eq with hb | hb
  · refine le_of_lt ?_
    calc tpos s ≤ s := (tpos_spec hs).2.1
      _ < 2 ^ (binade s + 1) := (binade_spec hs).2
      _ ≤ 2 ^ binade t := zpow_le_zpow_right₀ one_le_two (by omega)
      _ ≤ tpos t := (tpos_spec ht).1
  · unfold tpos
    rw [hb]
    have hG := two_zpow_pos (binade t - 13)
    apply mul_le_mul_of_nonneg_right _ hG.le
    exact_mod_cast Int.floor_mono (div_le_div_of_nonneg_right h hG.le)

theorem trunc14_zero : trunc14 0 = 0 := by simp [trunc14]

theorem trunc14_of_neg {y : ℚ} (h : y < 0) : trunc14 y = -tpos (-y) := by
  rw [← trunc14_pos (by linarith), ← trunc14_neg, neg_neg]

theorem trunc14_lt_zero {y : ℚ} (h : y < 0) : trunc14 y < 0 := by
  rw [trunc14_of_neg h]; linarith [tpos_pos (show 0 < -y by linarith)]

theorem trunc14_gt_zero {y : ℚ} (h : 0 < y) : 0 < trunc14 y := by
  rw [trunc14_pos h]; exact tpos_pos h

/-- **L7: `trunc14` is monotone.** -/
theorem trunc14Mono : Trunc14Mono := by
  intro y y' h
  rcases lt_trichotomy y 0 with hy | hy | hy
  · rcases lt_trichotomy y' 0 with hy' | hy' | hy'
    · rw [trunc14_of_neg hy, trunc14_of_neg hy']; exact neg_le_neg (tpos_mono (by linarith) (by linarith))
    · rw [hy', trunc14_zero]; exact (trunc14_lt_zero hy).le
    · linarith [trunc14_lt_zero hy, trunc14_gt_zero hy']
  · subst hy
    rw [trunc14_zero]
    rcases h.lt_or_eq with h | h
    · exact (trunc14_gt_zero h).le
    · rw [← h, trunc14_zero]
  · rw [trunc14_pos hy, trunc14_pos (lt_of_lt_of_le hy h)]
    exact tpos_mono hy h

theorem trunc14_abs (y : ℚ) : |trunc14 y| = trunc14 |y| := by
  rcases lt_trichotomy y 0 with h | h | h
  · rw [abs_of_neg h, trunc14_neg, abs_of_neg (trunc14_lt_zero h)]
  · subst h; simp [trunc14_zero]
  · rw [abs_of_pos h, abs_of_pos (trunc14_gt_zero h)]

theorem trunc14_eq_zero {y : ℚ} (h : trunc14 y = 0) : y = 0 := by
  rcases lt_trichotomy y 0 with hy | hy | hy
  · linarith [trunc14_lt_zero hy]
  · exact hy
  · linarith [trunc14_gt_zero hy]

theorem abs_trunc14_sub (y : ℚ) : |trunc14 y - y| = abs (trunc14 (abs y) - abs y) := by
  rcases lt_or_ge y 0 with h | h
  · rw [abs_of_neg h, trunc14_neg, ← abs_neg]; ring_nf
  · rw [abs_of_nonneg h]

/-- **L3: the cell.** -/
theorem trunc14Cell : Trunc14Cell := by
  intro y y' hy
  have hay := abs_pos.2 hy
  have hcell : |trunc14 y - y| < 2 ^ (binade |y| - 13) := by
    rw [abs_trunc14_sub, trunc14_pos hay]
    have := tpos_spec hay
    rw [abs_of_nonpos (by linarith)]
    linarith
  refine ⟨hcell, fun he => ?_⟩
  have hy' : y' ≠ 0 := fun h0 => hy (trunc14_eq_zero (by rw [he, h0, trunc14_zero]))
  have hay' := abs_pos.2 hy'
  have he' : tpos |y| = tpos |y'| := by rw [← trunc14_pos hay, ← trunc14_pos hay', ← trunc14_abs, ← trunc14_abs, he]
  have hb : binade |y| = binade |y'| := by rw [← binade_tpos hay, ← binade_tpos hay', he']
  obtain ⟨-, a1, a2⟩ := tpos_spec hay
  obtain ⟨-, c1, c2⟩ := tpos_spec hay'
  rw [← hb, ← he'] at c2
  rw [← he'] at c1
  have hss : (y < 0 ↔ y' < 0) := by
    constructor
    · intro h
      by_contra h'
      rcases (not_lt.1 h').lt_or_eq with h' | h'
      · linarith [trunc14_lt_zero h, trunc14_gt_zero h']
      · exact hy' h'.symm
    · intro h
      by_contra h'
      rcases (not_lt.1 h').lt_or_eq with h' | h'
      · linarith [trunc14_lt_zero h, trunc14_gt_zero h']
      · exact hy h'.symm
  have : |y - y'| = |(|y| - |y'|)| := by
    rcases lt_or_ge y 0 with h | h
    · rw [abs_of_neg h, abs_of_neg (hss.1 h), ← abs_neg]; ring_nf
    · rw [abs_of_nonneg h, abs_of_nonneg (not_lt.1 fun h' => not_lt.2 h (hss.2 h'))]
  rw [this, abs_lt]; constructor <;> linarith

end Pouw.Fp8Atom.H1T.Proofs
