import Definitions.Pouw.PearlC.DeviceKernelWref
import Proofs.Pouw.PearlC.FormingP
import Proofs.Pouw.PearlC.AccumProofs
import Proofs.Pouw.PearlC.FreezeP
import Proofs.Pouw.PearlC.FreezePPure

/-!
# The honest words are FP32 words

The honest transcript commits `wordOf v` for each reference value `v` (`Game.honestTranscript`), and `wordOf v` reads
back as `v` exactly when some FP32 word has the value `v` (`wordOf_spec`). This file proves that one does for every
ticket and useful value of Pearl-C at a device record `devAt ⟨[32], w, z⟩ G pr` with `w ≥ 24`, sm_120 v1 included, on
every row of at most `2^24 − 1` atoms (the domain's `k ≤ 2^16` gives at most `2^11`):

- **A useful value** is `(peelExactP …).bind wordQ`: a word's value or `none`, and `none` is an infinity's.
- **A ticket value** is the promoted chain `rn(… rn(0 + x₁) … + x_g)`, and `rn` has no exponent range, so the total is
  an FP32 value only if it stays on FP32's grid `2^−149` (`onGrid_rn`) and below `2^128`. Each step adds at most
  `32·448² < 2^23` to the accumulator's magnitude (`step_abs_le`: both truncations are toward zero), so a group's word
  is at most `2^23` per atom (`groupWord_abs_le`) and the total at most `2^23` per atom of the row
  (`foldl_promote_bound`, `chainP_word`).
-/

namespace Pouw.PearlC.HonestWords

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Accum

/-! ## FP32 values -/

/-- On FP32's finest grid, the subnormals' spacing `2^−149`. -/
def OnGrid (x : ℚ) : Prop := ∃ j : ℤ, x = j * 2 ^ (-149 : ℤ)

theorem onGrid_zero : OnGrid 0 := ⟨0, by simp⟩

theorem onGrid_add {x y : ℚ} (hx : OnGrid x) (hy : OnGrid y) : OnGrid (x + y) := by
  obtain ⟨i, rfl⟩ := hx
  obtain ⟨j, rfl⟩ := hy
  exact ⟨i + j, by push_cast; ring⟩

theorem onGrid_neg {x : ℚ} (hx : OnGrid x) : OnGrid (-x) := by
  obtain ⟨i, rfl⟩ := hx
  exact ⟨-i, by push_cast; ring⟩

theorem onGrid_abs {x : ℚ} (hx : OnGrid x) : OnGrid |x| := by
  rcases abs_choice x with h | h <;> rw [h]
  · exact hx
  · exact onGrid_neg hx

theorem onGrid_pow_mul (n : ℤ) {e : ℤ} (he : -149 ≤ e) : OnGrid (n * 2 ^ e) := by
  refine ⟨n * 2 ^ (e + 149).toNat, ?_⟩
  have h : (2 : ℚ) ^ e = 2 ^ ((e + 149).toNat : ℤ) * 2 ^ (-149 : ℤ) := by
    rw [← zpow_add₀ two_ne_zero]; congr 1; omega
  rw [h, zpow_natCast]; push_cast; ring

theorem onGrid_term (t : Term) (h : t.mag ≠ 0 → -126 ≤ t.exp) : OnGrid t.value := by
  by_cases hm : t.mag = 0
  · exact ⟨0, by simp [Term.value, hm]⟩
  have he := h hm
  have e : t.value = (((if t.neg then -1 else 1) * t.mag : ℤ) : ℚ) * 2 ^ (t.exp - 23) := by
    unfold Term.value; split <;> push_cast <;> ring
  rw [e]
  exact onGrid_pow_mul _ (by omega)

theorem onGrid_wordQ {w : ℕ} {x : ℚ} (h : wordQ w = some x) : OnGrid x := by
  obtain ⟨t, ht, rfl⟩ := PeelExactProofs.wordQ_fin h
  exact onGrid_term t (FreezeP.fp32Value_floored w t ht)

theorem rn_zero : rn 0 = 0 := by simp [rn_eq, rp_zero]

theorem onGrid_rp {t : ℚ} (ht : 0 < t) (hg : OnGrid t) : OnGrid (rp t) := by
  have hu : ulp t = 2 ^ (binade t - 23) := by rw [ulp, abs_of_pos ht]
  by_cases hb : -126 ≤ binade t
  · rw [rp, hu]; exact onGrid_pow_mul _ (by omega)
  · replace hb := not_le.1 hb
    obtain ⟨j, hj⟩ := hg
    rw [rp_of_int (n := j * 2 ^ (-126 - binade t).toNat)]
    · exact ⟨j, hj⟩
    rw [hu]
    generalize binade t = b at hb ⊢
    rw [hj]
    push_cast
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_div_assoc, ← zpow_sub₀ two_ne_zero]
    congr 2; ring

theorem onGrid_rn {x : ℚ} (hx : OnGrid x) : OnGrid (rn x) := by
  rcases eq_or_ne x 0 with rfl | h0
  · rw [rn_zero]; exact onGrid_zero
  rw [rn_eq]
  have h := onGrid_rp (abs_pos.2 h0) (onGrid_abs hx)
  split
  · rw [neg_one_mul]; exact onGrid_neg h
  · rw [one_mul]; exact h

/-- `rn` keeps a bound `M·2^e` with a 24-bit `M`: the bound is an FP32 value, and `rn` is monotone and odd. -/
theorem abs_rn_le {x : ℚ} {M : ℕ} {e : ℤ} (hM : M < 2 ^ 24) (h : |x| ≤ M * 2 ^ e) : |rn x| ≤ M * 2 ^ e := by
  have hr : Representable ((M : ℚ) * 2 ^ e) :=
    ⟨M, e, by rw [abs_of_nonneg (by positivity)]; exact_mod_cast hM, by push_cast; rfl⟩
  have h1 := rn_mono (abs_le.1 h).2
  have h2 := rn_mono (abs_le.1 h).1
  rw [rn_exact hr] at h1
  rw [rn_neg, rn_exact hr] at h2
  exact abs_le.2 ⟨h2, h1⟩

/-- **An FP32 value has a word**: a 24-bit significand on the grid `2^−149` below `2^128`. -/
theorem exists_wordQ {q : ℚ} (hg : OnGrid q) (hr : Representable q) (hq : |q| < 2 ^ 128) :
    ∃ w, wordQ w = some q := by
  rcases eq_or_ne q 0 with rfl | h0
  · exact ⟨0, Freeze.wordQ_zero⟩
  set sb : ℕ := if q < 0 then 1 else 0 with hsb
  have hs : sb ≤ 1 := by rw [hsb]; split <;> omega
  have hsgn : (if (sb == 1) = true then (-1 : ℚ) else 1) * |q| = q := by
    rw [hsb]; by_cases h : q < 0
    · simp [h, abs_of_neg h]
    · simp [h, abs_of_nonneg (not_lt.1 h)]
  have ha : 0 < |q| := abs_pos.2 h0
  obtain ⟨hb1, hb2⟩ := binade_spec ha
  have hb128 : binade |q| ≤ 127 := by
    have h : (2 : ℚ) ^ binade |q| < 2 ^ (128 : ℤ) := lt_of_le_of_lt hb1 (by exact_mod_cast hq)
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 h
    omega
  by_cases hn : -126 ≤ binade |q|
  · obtain ⟨N, hN⟩ := sig_int hr h0
    obtain ⟨s1, s2⟩ := sig_bounds h0
    rw [hN] at s1 s2
    have hN1 : (2 ^ 23 : ℤ) ≤ N := by exact_mod_cast s1
    have hN2 : N < 2 ^ 24 := by exact_mod_cast s2
    refine ⟨sb * 2 ^ 31 + (binade |q| + 127).toNat * 2 ^ 23 + (N.toNat - 2 ^ 23), ?_⟩
    rw [wordQ, AtomErr.fp32Value_word sb _ _ hs (by omega) (by omega), ite_eq_right (by omega)]
    simp only [Value.finite?, Option.map_some, Option.some.injEq, Term.value]
    have hq' : |q| = N * 2 ^ (binade |q| - 23) := by
      have hu : ulp q = 2 ^ (binade |q| - 23) := rfl
      rw [← hN, hu, div_mul_cancel₀ _ (two_zpow_pos _).ne']
    have hm : ((2 ^ 23 + (N.toNat - 2 ^ 23) : ℕ) : ℚ) = N := by
      have : ((2 ^ 23 + (N.toNat - 2 ^ 23) : ℕ) : ℤ) = N := by push_cast; omega
      exact_mod_cast this
    have hx : (((binade |q| + 127).toNat : ℕ) : ℤ) - 127 - 23 = binade |q| - 23 := by omega
    rw [hm, hx, mul_assoc, ← hq', hsgn]
  · replace hn := not_le.1 hn
    obtain ⟨j, hj⟩ := hg
    have hj0 : j ≠ 0 := by rintro rfl; simp at hj; exact h0 hj
    have hjq : |q| = (j.natAbs : ℚ) * 2 ^ (-149 : ℤ) := by
      rw [hj, abs_mul, abs_of_pos (two_zpow_pos _), Nat.cast_natAbs, Int.cast_abs]
    have hj23 : j.natAbs < 2 ^ 23 := by
      have h1 : |q| < 2 ^ (-126 : ℤ) :=
        lt_of_lt_of_le hb2 (zpow_le_zpow_right₀ one_le_two (by omega))
      rw [hjq, show (2 : ℚ) ^ (-126 : ℤ) = 2 ^ 23 * 2 ^ (-149 : ℤ) by
        rw [show (2 : ℚ) ^ 23 = 2 ^ (23 : ℤ) by norm_cast, ← zpow_add₀ two_ne_zero]; norm_num] at h1
      have := lt_of_mul_lt_mul_right h1 (two_zpow_pos _).le
      exact_mod_cast this
    refine ⟨sb * 2 ^ 31 + 0 * 2 ^ 23 + j.natAbs, ?_⟩
    rw [wordQ, AtomErr.fp32Value_word sb _ _ hs (by omega) hj23, ite_eq_left rfl, ite_eq_right (by omega)]
    simp only [Value.finite?, Option.map_some, Option.some.injEq, Term.value]
    rw [show (-126 : ℤ) - 23 = -149 by norm_num, mul_assoc, ← hjq, hsgn]

/-! ## The words a step writes -/

/-- A term as `normalize` leaves it and `packTerm` encodes it: zero, or a magnitude below `2^24` at exponent `−126` or
more that is normal (`2^23` or more) or at exponent `−126`. -/
def Canon (t : Term) : Prop :=
  t.mag = 0 ∨ (t.mag < 2 ^ 24 ∧ -126 ≤ t.exp ∧ (2 ^ 23 ≤ t.mag ∨ t.exp = -126))

theorem wordQ_packTerm_canon {t : Term} (hc : Canon t) (he : t.mag ≠ 0 → t.exp ≤ 127) :
    wordQ (packTerm t) = some t.value := by
  rcases hc with h0 | ⟨h24, h126, h23 | hsub⟩
  · have hv : t.value = 0 := by simp [Term.value, h0]
    rw [hv, show packTerm t = 0 by simp [packTerm, h0]]
    exact Freeze.wordQ_zero
  · exact wordQ_packTerm h23 h24 h126 (he (by omega))
  · by_cases h23 : 2 ^ 23 ≤ t.mag
    · exact wordQ_packTerm h23 h24 h126 (by omega)
    replace h23 := not_le.1 h23
    by_cases h0 : t.mag = 0
    · have hv : t.value = 0 := by simp [Term.value, h0]
      rw [hv, show packTerm t = 0 by simp [packTerm, h0]]
      exact Freeze.wordQ_zero
    have hb : t.mag &&& 1 <<< 23 = 0 := by
      rw [Nat.one_shiftLeft, Nat.and_two_pow]
      have : t.mag.testBit 23 = false := Nat.testBit_lt_two_pow h23
      simp [this]
    have hmask : t.mag &&& 0x7FFFFF = t.mag := by
      rw [show (0x7FFFFF : Nat) = 2 ^ 23 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod, Nat.mod_eq_of_lt h23]
    have h1 : ¬ 127 < t.exp := by omega
    rw [AccumProofs.packTerm_fin t h0 h1 (AccumProofs.packTerm_exp t h1), hmask, ite_eq_left hb, hsub]
    have hs : (if t.neg = true then 1 else 0) ≤ 1 := by split <;> omega
    rw [show ((-126 : ℤ) + 127 - 1).toNat = 0 by norm_num, wordQ,
      AtomErr.fp32Value_word _ _ _ hs (by norm_num) h23, ite_eq_left rfl, ite_eq_right h0]
    simp only [Value.finite?, Option.map_some, Option.some.injEq, Term.value, hsub]
    cases t.neg <;> simp

theorem fromAdder_ge {w : ℕ} (hw : 24 ≤ w) (m : ℕ) : fromAdder w m = m / 2 ^ (w - 24) := by
  rw [fromAdder, ite_eq_left hw, Nat.shiftRight_eq_div_pow]

theorem two_zpow_sub_natCast (a b : ℕ) (h : b ≤ a) : (2 : ℚ) ^ ((a : ℤ) - b) = 2 ^ (a - b) := by
  rw [← zpow_natCast, Nat.cast_sub h]

/-- `normalize`'s last two steps on a `w`-bit magnitude `M1` at exponent `e`: below `−126` the shift to the subnormal
grid, then the return to 24 bits. Both truncate toward zero. -/
theorem normalize_tail {w : ℕ} (hw : 24 ≤ w) (neg : Bool) (e : ℤ) (M1 : ℕ) (hlo : 2 ^ (w - 1) ≤ M1)
    (hhi : M1 < 2 ^ w) :
    Canon (if fromAdder w (if e < -126 then M1 >>> (-126 - e).toNat else M1) = 0 then Term.zero
      else ⟨neg, if e < -126 then -126 else e, fromAdder w (if e < -126 then M1 >>> (-126 - e).toNat else M1)⟩) ∧
    |(if fromAdder w (if e < -126 then M1 >>> (-126 - e).toNat else M1) = 0 then Term.zero
      else (⟨neg, if e < -126 then -126 else e,
        fromAdder w (if e < -126 then M1 >>> (-126 - e).toNat else M1)⟩ : Term)).value| ≤ M1 * 2 ^ (e - w + 1) := by
  have hpos : (0 : ℚ) ≤ M1 * 2 ^ (e - w + 1) := mul_nonneg (Nat.cast_nonneg _) (two_zpow_pos _).le
  -- the shift to the subnormal grid: `M2` at exponent `e2`, with `M2·2^(e2 − e) ≤ M1`
  obtain ⟨M2, e2, hM2, he2, hv2, hhi2, hcan2⟩ : ∃ (M2 : ℕ) (e2 : ℤ),
      (if e < -126 then M1 >>> (-126 - e).toNat else M1) = M2 ∧ (if e < -126 then -126 else e) = e2 ∧
      (M2 : ℚ) * 2 ^ (e2 - e) ≤ M1 ∧ M2 < 2 ^ w ∧ -126 ≤ e2 ∧ (2 ^ (w - 1) ≤ M2 ∨ e2 = -126) := by
    by_cases hs : e < -126
    · refine ⟨M1 >>> (-126 - e).toNat, -126, by rw [ite_eq_left hs], by rw [ite_eq_left hs], ?_, ?_, le_rfl, Or.inr rfl⟩
      · rw [Nat.shiftRight_eq_div_pow, show (-126 : ℤ) - e = (((-126 - e).toNat : ℕ) : ℤ) by omega, zpow_natCast]
        exact_mod_cast Nat.div_mul_le_self _ _
      · exact lt_of_le_of_lt (Nat.shiftRight_le _ _) hhi
    · refine ⟨M1, e, by rw [ite_eq_right hs], by rw [ite_eq_right hs], by simp, hhi, by omega, Or.inl hlo⟩
  rw [hM2, he2]
  rw [fromAdder_ge hw]
  set M3 := M2 / 2 ^ (w - 24) with hM3
  have hv3 : (M3 : ℚ) * 2 ^ (w - 24) ≤ M2 := by exact_mod_cast Nat.div_mul_le_self _ _
  have hhi3 : M3 < 2 ^ 24 := by
    rw [hM3, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add, show 24 + (w - 24) = w by omega]; exact hhi2
  by_cases h0 : M3 = 0
  · rw [ite_eq_left h0]
    exact ⟨Or.inl rfl, by simpa [Term.zero, Term.value] using hpos⟩
  rw [ite_eq_right h0]
  refine ⟨Or.inr ⟨hhi3, hcan2.1, ?_⟩, ?_⟩
  · rcases hcan2.2 with h | h
    · left
      rw [hM3, Nat.le_div_iff_mul_le (by positivity), ← pow_add]
      exact le_trans (Nat.pow_le_pow_right (by norm_num) (by omega)) h
    · exact Or.inr h
  · rw [FreezePPure.term_abs_value]
    dsimp only
    calc (M3 : ℚ) * 2 ^ (e2 - 23) = ((M3 : ℚ) * 2 ^ (w - 24)) * 2 ^ (e2 - e) * 2 ^ (e - w + 1) := by
          rw [← two_zpow_sub_natCast w 24 hw, mul_assoc, mul_assoc, ← zpow_add₀ two_ne_zero,
            ← zpow_add₀ two_ne_zero]
          congr 2; push_cast; ring
      _ ≤ (M2 : ℚ) * 2 ^ (e2 - e) * 2 ^ (e - w + 1) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hv3 (two_zpow_pos _).le) (two_zpow_pos _).le
      _ ≤ M1 * 2 ^ (e - w + 1) := mul_le_mul_of_nonneg_right hv2 (two_zpow_pos _).le

/-- **`normalize` truncates toward zero** at any width `w ≥ 24`, subnormals included: a `Canon` term no larger than the
sum on its grid `2^(E − w + 1)`. -/
theorem normalize_le {w : ℕ} (hw : 24 ≤ w) (E total : ℤ) :
    Canon (normalize w E total) ∧ |(normalize w E total).value| ≤ |(total : ℚ)| * 2 ^ (E - w + 1) := by
  by_cases h0 : total = 0
  · subst h0; simp [normalize, Canon, Term.zero, Term.value]
  have hTq : |(total : ℚ)| = (total.natAbs : ℚ) := by rw [Nat.cast_natAbs, Int.cast_abs]
  have hn : total.natAbs ≠ 0 := by omega
  have hbl : bitLength total.natAbs = total.natAbs.log2 + 1 := by unfold bitLength; rw [ite_eq_right hn]
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self total.natAbs
  unfold normalize
  rw [ite_eq_right h0, hTq]
  dsimp only
  generalize total.natAbs = T at hn hbl hL1 hL2 ⊢
  generalize T.log2 = L at hbl hL1 hL2
  generalize bitLength T = bl at hbl ⊢
  subst hbl
  have hlo : 2 ^ (w - 1) ≤ (if w < L + 1 then T >>> (L + 1 - w) else T <<< (w - (L + 1))) := by
    split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow, Nat.le_div_iff_mul_le (by positivity), ← pow_add]
      exact le_trans (Nat.pow_le_pow_right (by norm_num) (by omega)) hL1
    · rw [Nat.shiftLeft_eq]
      calc 2 ^ (w - 1) = 2 ^ L * 2 ^ (w - (L + 1)) := by rw [← pow_add]; congr 1; omega
        _ ≤ _ := Nat.mul_le_mul_right _ hL1
  have hhi : (if w < L + 1 then T >>> (L + 1 - w) else T <<< (w - (L + 1))) < 2 ^ w := by
    split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]
      exact lt_of_lt_of_le hL2 (Nat.pow_le_pow_right (by norm_num) (by omega))
    · rw [Nat.shiftLeft_eq]
      calc T * 2 ^ (w - (L + 1)) < 2 ^ (L + 1) * 2 ^ (w - (L + 1)) :=
            Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
        _ = 2 ^ w := by rw [← pow_add]; congr 1; omega
  have hv1 : ((if w < L + 1 then T >>> (L + 1 - w) else T <<< (w - (L + 1)) : ℕ) : ℚ) *
      2 ^ (((L + 1 : ℕ) : ℤ) - w) ≤ T := by
    split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow, two_zpow_sub_natCast (L + 1) w (by omega)]
      exact_mod_cast Nat.div_mul_le_self _ _
    · rw [Nat.shiftLeft_eq, show (((L + 1 : ℕ) : ℤ) - w) = -((w - (L + 1) : ℕ) : ℤ) by omega, zpow_neg,
        zpow_natCast]
      push_cast
      rw [mul_assoc, mul_inv_cancel₀ (by positivity), mul_one]
  generalize (if w < L + 1 then T >>> (L + 1 - w) else T <<< (w - (L + 1))) = M1 at hlo hhi hv1 ⊢
  have key := normalize_tail hw (decide (total < 0)) (E + ((L + 1 : ℕ) : ℤ) - w) M1 hlo hhi
  refine ⟨key.1, key.2.trans ?_⟩
  calc (M1 : ℚ) * 2 ^ (E + ((L + 1 : ℕ) : ℤ) - w - w + 1) =
        (M1 : ℚ) * 2 ^ (((L + 1 : ℕ) : ℤ) - w) * 2 ^ (E - w + 1) := by
        rw [mul_assoc, ← zpow_add₀ two_ne_zero]; congr 2; ring
    _ ≤ T * 2 ^ (E - w + 1) := mul_le_mul_of_nonneg_right hv1 (two_zpow_pos _).le

/-- **A group sum is no larger than the sum of its terms' magnitudes**, at any width `w ≥ 24`: each term truncated
toward zero to the group's grid, then the sum truncated toward zero by `normalize`. -/
theorem groupSum_le {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (ts : List Term) :
    Canon (groupSum w z ts) ∧ |(groupSum w z ts).value| ≤ (ts.map fun t => |t.value|).sum := by
  unfold groupSum
  set E := maxExp z ts
  obtain ⟨hc, hv⟩ := normalize_le hw E (alignedSum w E ts)
  refine ⟨hc, hv.trans ?_⟩
  have hE : ∀ t ∈ ts, t.mag ≠ 0 → t.exp ≤ E := le_maxExp z ts
  clear_value E
  clear hc hv
  have hg : (0 : ℚ) < 2 ^ (E - w + 1) := two_zpow_pos _
  unfold alignedSum
  induction ts with
  | nil => simp
  | cons t ts ih =>
    have ht := FormingP.alignedW_value (w := w) (hE t List.mem_cons_self)
    have h1 : |(aligned w E t : ℚ)| * 2 ^ (E - w + 1) ≤ |t.value| := by
      rw [← abs_of_pos hg, ← abs_mul, ht]; exact truncTo_abs_le hg _
    have h2 := ih fun s hs => hE s (List.mem_cons_of_mem _ hs)
    simp only [List.map_cons, List.sum_cons, Int.cast_add]
    calc |(aligned w E t : ℚ) + ((ts.map (aligned w E)).sum : ℤ)| * 2 ^ (E - w + 1)
        ≤ (|(aligned w E t : ℚ)| + |(((ts.map (aligned w E)).sum : ℤ) : ℚ)|) * 2 ^ (E - w + 1) := by
          gcongr; exact abs_add_le _ _
      _ ≤ _ := by rw [add_mul]; exact add_le_add h1 h2

theorem wordQ_packValue_nan : wordQ (packValue .nan) = none := by decide

theorem wordQ_packValue_inf (s : Bool) : wordQ (packValue (.inf s)) = none := by cases s <;> decide

/-- A total group step with a finite result: every input finite, and the result the group sum of their terms. -/
theorem groupStepTotal_fin_inv {w : ℕ} {z : ℤ} {acc : Value} {ps : List Value} {t : Term}
    (h : groupStepTotal w z acc ps = .fin t) :
    t = groupSum w z ((acc :: ps).filterMap Value.finite?) ∧ ∀ v ∈ acc :: ps, ∃ t', v = .fin t' := by
  unfold groupStepTotal at h
  dsimp only at h
  split_ifs at h with h1 h2 h3 h4 h5
  have hv : ∀ v ∈ acc :: ps, ∃ t', v = .fin t' := by
    intro v hv
    rcases v with _ | s | t'
    · exact absurd (List.any_eq_true.2 ⟨_, hv, rfl⟩) h1
    · cases s
      · exact absurd (List.any_eq_true.2 ⟨_, hv, by simp [Value.isInf]⟩) h3
      · exact absurd (List.any_eq_true.2 ⟨_, hv, by simp [Value.isInf]⟩) h4
    · exact ⟨t', rfl⟩
  exact ⟨(Value.fin.inj h).symm, hv⟩

/-- **One step adds at most `32·448²`**: at a pipeline `⟨[32], w, z⟩` with `w ≥ 24`, a finite word out of the step has
a finite accumulator in, and its magnitude exceeds the accumulator's by at most `32·448² < 2^23`. -/
theorem step_abs_le {w : ℕ} (hw : 24 ≤ w) (z : ℤ) {acc : ℕ} (a b : List Code) {y : ℚ}
    (h : wordQ ((⟨[32], w, z⟩ : Pipeline).step acc a b) = some y) :
    ∃ x, wordQ acc = some x ∧ |y| ≤ |x| + 32 * 448 ^ 2 := by
  have hs : (⟨[32], w, z⟩ : Pipeline).step acc a b =
      packValue (groupStepTotal w z (fp32Value acc) (((a.zip b).take 32).map fun x => productTotal x.1 x.2)) := rfl
  rw [hs] at h
  generalize hv : groupStepTotal w z (fp32Value acc) (((a.zip b).take 32).map fun x => productTotal x.1 x.2) = v at h
  rcases v with _ | s | t
  · rw [wordQ_packValue_nan] at h; exact absurd h (by simp)
  · rw [wordQ_packValue_inf] at h; exact absurd h (by simp)
  obtain ⟨ht, hfin⟩ := groupStepTotal_fin_inv hv
  obtain ⟨tc, htc⟩ := hfin _ List.mem_cons_self
  refine ⟨tc.value, by simp [wordQ, htc, Value.finite?], ?_⟩
  obtain ⟨hcan, habs⟩ := groupSum_le hw z ((fp32Value acc :: _).filterMap Value.finite?)
  rw [← ht] at hcan habs
  have hy : wordQ (packValue (.fin t)) = some t.value := by
    show wordQ (packTerm t) = _
    refine wordQ_packTerm_canon hcan fun hm => ?_
    unfold groupStepTotal at hv
    dsimp only at hv
    split_ifs at hv with h1 h2 h3 h4 h5
    by_contra hc
    rw [← ht] at h5
    exact h5 ⟨hm, by omega⟩
  rw [hy, Option.some.injEq] at h
  rw [← h]
  refine habs.trans ?_
  rw [htc, List.filterMap_cons, Value.finite?, List.map_cons, List.sum_cons]
  gcongr
  -- the products
  have hlen : (((a.zip b).take 32).map fun x => productTotal x.1 x.2).length ≤ 32 := by simp
  have hprod : ∀ t' ∈ (((a.zip b).take 32).map fun x => productTotal x.1 x.2).filterMap Value.finite?,
      |t'.value| ≤ 448 ^ 2 := by
    intro t' ht'
    obtain ⟨v, hv', hvt⟩ := List.mem_filterMap.1 ht'
    obtain ⟨⟨ca, cb⟩, -, rfl⟩ := List.mem_map.1 hv'
    unfold productTotal at hvt
    by_cases hn : (ca.isNaN || cb.isNaN) = true
    · rw [ite_eq_left hn] at hvt; exact absurd hvt (by simp [Value.finite?])
    rw [ite_eq_right hn] at hvt
    simp only [Value.finite?, Option.some.injEq] at hvt
    subst hvt
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true] at hn
    rw [productExact _ _ _ _ (val_eq hn.1) (val_eq hn.2), abs_mul]
    have h1 := val_abs_le hn.1
    have h2 := val_abs_le hn.2
    calc |val ca| * |val cb| ≤ 448 * 448 := mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
      _ = 448 ^ 2 := by norm_num
  generalize (((a.zip b).take 32).map fun x => productTotal x.1 x.2) = ps at hlen hprod
  calc ((ps.filterMap Value.finite?).map fun t => |t.value|).sum
      ≤ ((ps.filterMap Value.finite?).map fun _ => (448 ^ 2 : ℚ)).sum := List.sum_le_sum fun t' ht' => hprod t' ht'
    _ = (ps.filterMap Value.finite?).length * 448 ^ 2 := by simp
    _ ≤ 32 * 448 ^ 2 := by
      gcongr
      exact_mod_cast le_trans (List.length_filterMap_le _ _) hlen

/-! ## The promoted chain -/

theorem foldl_step_abs_le {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (g : List AtomIn) :
    ∀ (acc : ℕ) {y : ℚ},
      wordQ (g.foldl (fun v x => (⟨[32], w, z⟩ : Pipeline).step v (List.ofFn x.1) (List.ofFn x.2)) acc) = some y →
        ∃ x, wordQ acc = some x ∧ |y| ≤ |x| + g.length * 2 ^ 23 := by
  induction g with
  | nil => intro acc y h; exact ⟨y, h, by simp⟩
  | cons a g ih =>
    intro acc y h
    rw [List.foldl_cons] at h
    obtain ⟨x', hx', hy⟩ := ih _ h
    obtain ⟨x, hx, hxy⟩ := step_abs_le hw z _ _ hx'
    refine ⟨x, hx, ?_⟩
    simp only [List.length_cons]
    push_cast
    linarith

/-- **A group's word is at most `2^23` per atom.** -/
theorem groupWord_abs_le {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (g : List AtomIn) {y : ℚ}
    (h : wordQ (groupWordP ⟨[32], w, z⟩ g) = some y) : |y| ≤ g.length * 2 ^ 23 := by
  obtain ⟨x, hx, hy⟩ := foldl_step_abs_le hw z g 0 h
  rw [Freeze.wordQ_zero, Option.some.injEq] at hx
  subst hx
  simpa using hy

/-- The promotions keep the total on the grid, with a 24-bit significand, at most `2^23` per atom promoted so far. -/
theorem foldl_promote_bound {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (gs : List (List AtomIn)) :
    ∀ (t : ℚ) (A : ℕ), OnGrid t → Representable t → |t| ≤ A * 2 ^ 23 → A + (gs.map List.length).sum < 2 ^ 24 →
      ∀ q, (gs.map (groupWordP ⟨[32], w, z⟩)).foldl promote (some t) = some q →
        OnGrid q ∧ Representable q ∧ |q| ≤ ((A + (gs.map List.length).sum : ℕ) : ℚ) * 2 ^ 23 := by
  induction gs with
  | nil =>
    intro t A hg hr ht _ q h
    simp only [List.map_nil, List.foldl_nil, Option.some.injEq] at h
    subst h
    simpa using ⟨hg, hr, ht⟩
  | cons g gs ih =>
    intro t A hg hr ht hA q h
    simp only [List.map_cons, List.foldl_cons] at h
    simp only [List.map_cons, List.sum_cons] at hA ⊢
    rcases hx : wordQ (groupWordP ⟨[32], w, z⟩ g) with _ | x
    · have hp : promote (some t) (groupWordP ⟨[32], w, z⟩ g) = none := by simp [promote, hx]
      rw [hp, Freeze.foldl_promote_none] at h
      exact absurd h (by simp)
    have hp : promote (some t) (groupWordP ⟨[32], w, z⟩ g) = some (rnAdd t x) := by simp [promote, hx]
    rw [hp] at h
    have hxb := groupWord_abs_le hw z g hx
    have hs : |t + x| ≤ ((A + g.length : ℕ) : ℚ) * 2 ^ 23 := by
      push_cast
      linarith [abs_add_le t x]
    have hs' : |t + x| ≤ ((A + g.length : ℕ) : ℚ) * 2 ^ (23 : ℤ) := by rw [zpow_ofNat]; exact hs
    have hrn : |rnAdd t x| ≤ ((A + g.length : ℕ) : ℚ) * 2 ^ 23 := by
      have := abs_rn_le (M := A + g.length) (by omega) hs'
      rwa [zpow_ofNat] at this
    have key := ih (rnAdd t x) (A + g.length) (onGrid_rn (onGrid_add hg (onGrid_wordQ hx)))
      (Freeze.rn_representable _) hrn (by omega) q h
    rwa [show A + (g.length + (gs.map List.length).sum) = A + g.length + (gs.map List.length).sum by ring]

theorem chunksOf_length_sum {α : Type} (n : ℕ) :
    ∀ l : List α, ((Freeze.chunksOf n l).map List.length).sum = l.length
  | [] => by simp [Freeze.chunksOf]
  | x :: xs => by
    rw [Freeze.chunksOf_cons, List.map_cons, List.sum_cons, chunksOf_length_sum n (xs.drop n)]
    simp only [List.length_cons, List.length_take, List.length_drop]
    omega
termination_by l => l.length
decreasing_by simp; omega

theorem toChunks_length_sum {α : Type} (G : ℕ) (l : List α) : ((l.toChunks G).map List.length).sum = l.length := by
  rcases G with _ | n
  · cases l with
    | nil => rfl
    | cons x xs => simp [AccumProofs.toChunks_zero_cons]
  · rw [Freeze.toChunks_succ]
    exact chunksOf_length_sum n l

theorem exists_wordQ_none : ∃ wd, wordQ wd = none := ⟨_, wordQ_packValue_inf false⟩

/-- **The promoted chain's value is an FP32 word's** (or `none`, an infinity's), on a row of fewer than `2^24` atoms at
a step `⟨[32], w, z⟩` with `w ≥ 24`: its magnitude is at most `2^23` per atom. -/
theorem chainP_word {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (G : ℕ) (atoms : List AtomIn) (hk : atoms.length < 2 ^ 24) :
    ∃ wd, wordQ wd = chainP ⟨[32], w, z⟩ G atoms := by
  rcases h : chainP ⟨[32], w, z⟩ G atoms with _ | q
  · exact exists_wordQ_none
  have hr0 : Representable 0 := ⟨0, 0, by norm_num, by simp⟩
  obtain ⟨hg, hr, hq⟩ := foldl_promote_bound hw z (atoms.toChunks G) 0 0 onGrid_zero hr0 (by simp)
    (by rw [toChunks_length_sum]; omega) q h
  rw [toChunks_length_sum, zero_add] at hq
  refine exists_wordQ hg hr (lt_of_le_of_lt hq ?_)
  have h24 : (atoms.length : ℚ) < 2 ^ 24 := by exact_mod_cast hk
  calc (atoms.length : ℚ) * 2 ^ 23 < 2 ^ 24 * 2 ^ 23 := mul_lt_mul_of_pos_right h24 (by norm_num)
    _ ≤ 2 ^ 128 := by norm_num

theorem chainAt_word {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (G : ℕ) {m n k : ℕ} (hk : k / 32 < 2 ^ 24)
    (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) :
    ∃ wd, wordQ wd = chainAt ⟨[32], w, z⟩ G A B i j :=
  chainP_word hw z G _ (by unfold rowAtoms; rw [List.length_ofFn]; exact hk)

theorem bind_wordQ_word (x : Option ℕ) : ∃ wd, wordQ wd = x.bind wordQ := by
  rcases x with _ | v
  · exact exists_wordQ_none
  · exact ⟨v, rfl⟩

theorem atNat_word {p q : ℕ} (M : Matrix (Fin p) (Fin q) (Option ℚ)) (hM : ∀ i j, ∃ wd, wordQ wd = M i j)
    (i j : ℕ) : ∃ wd, wordQ wd = atNat M i j := by
  unfold atNat
  split_ifs with h
  · exact hM _ _
  · exact exists_wordQ_none

theorem unitUsefulDev_word {Q R S : Type} (p : Pipeline) (G : ℕ) (pr : Prices) (sem : PearlCSem Q R S) (H : Q → R)
    (s : S) (U : Workload) (u : ℕ) (act : Codes) (i : Fin (U.layout.shape u).m) (j : Fin (U.layout.shape u).n) :
    ∃ wd, wordQ wd = sem.unitUsefulDev (devAt p G pr) H s U u act i j :=
  bind_wordQ_word _

/-- **The honest words at a device record on a step `⟨[32], w, z⟩` with `w ≥ 24`**, under rev1 with the honest kernel's
`W_ref`: every ticket and useful value of a unit of `k/32 < 2^24` atoms per row is an FP32 word's, or `none`, an
infinity's, whatever the noise, the activations and the weights. -/
theorem honestWords_devRev1K {Q R S : Type} {w : ℕ} (hw : 24 ≤ w) (z : ℤ) (G : ℕ) (pr : Prices)
    (sem : PearlCSem Q R S) (ρ c : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes)
    (hk : (U.layout.shape u).k / 32 < 2 ^ 24) (i j : ℕ) :
    (∃ wd, wordQ wd = (pearlCProtocolDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c).ticket H s U u act i j) ∧
      ∃ wd, wordQ wd = (pearlCProtocolDevRev1K (devAt ⟨[32], w, z⟩ G pr) sem ρ c).useful H s U u act i j :=
by
  refine ⟨?_, ?_⟩
  · exact atNat_word (chainAt ⟨[32], w, z⟩ G (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2)
      (fun _ _ => chainAt_word hw z G hk _ _ _ _) i j
  · exact atNat_word (sem.unitUsefulDev (devAt ⟨[32], w, z⟩ G pr) H s U u act)
      (fun _ _ => unitUsefulDev_word _ _ _ _ _ _ _ _ _ _ _) i j

end Pouw.PearlC.HonestWords
