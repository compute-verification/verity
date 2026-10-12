import Proofs.Pouw.PearlC.HonestWords

/-!
# One group's step on a flat support: the cast reads a monotone function of the sum

The pieces of the salt-dead two-ends lemma (`Cap/SaltDead.lean`) that do not mention the support.

* **Rounding to a grid that coarsens with the binade is monotone** (`grid_mono`): for any monotone rounding
  `r : ℚ → ℤ` that fixes the integers, `x ↦ r(x / u(x))·u(x)` with `u(x) = 2^(max (binade x) emin − k)` is monotone on
  the positive rationals. E4M3's encoder is the instance `r = roundEven`, `emin = −6`, `k = 3`; the step's normalization
  is `r = ⌊·⌋`, `emin = −126`, `k = p − 1`.
* **E4M3's encoder** (`codeVal_e4m3_mono`, `codeVal_e4m3_neg`, `codeVal_e4m3_sat`): `codeVal ∘ e4m3` is monotone, odd,
  and `448` from `464` up; every code's value is at most `448`; `e4m3Mag` is at most `126`.
* **The normalization** (`normalize_nat`): on a nonzero sum `n`, `normalize w E n` is a canonical term (`Canon`) of
  value `truncF (min w 24) (n·2^(E − w + 1))`, the sum truncated toward zero to `min w 24` significant bits with
  FP32's subnormals; its exponent is `E + bitLength n − w` when that is at least `−126`. A negated sum negates the term
  (`normalize_neg`).
* **The read** (`codeW_mono`): the value the cast reads from a group's sum `t` at exponent `E`, through the overflow
  branch (`±inf`, read as `±480`), `packValue` and `wordVal`, then `codeVal ∘ e4m3`, is monotone in `t` for every
  width `w ≥ 1` and every `E`.
* **The FP32 word's sign** (`f32Word_sign`, `e4m3Word_signs`): `f32Word y` is its sign bit plus the word of `|y|`, a
  finite word or `+inf`; so `e4m3Word (f32Word y)` and `e4m3Word (f32Word (−y))` differ in the sign bit alone, below
  `126` in their other bits.
-/

namespace Pouw.PearlC.Cap

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC

/-! ## Monotone and odd -/

theorem monotone_of_odd {α : Type*} [AddCommGroup α] [LinearOrder α] [IsOrderedAddMonoid α] {f : α → ℚ}
    (hp : ∀ a b, 0 ≤ a → a ≤ b → f a ≤ f b) (ho : ∀ x, f (-x) = -f x) : Monotone f := by
  have h0 : f 0 = 0 := by have := ho 0; rw [neg_zero] at this; linarith
  intro a b hab
  rcases le_total 0 a with ha | ha
  · exact hp a b ha hab
  rcases le_total 0 b with hb | hb
  · have h1 := hp 0 (-a) le_rfl (neg_nonneg.2 ha)
    have h2 := hp 0 b le_rfl hb
    rw [ho, h0] at h1
    linarith
  · have := hp (-b) (-a) (neg_nonneg.2 hb) (neg_le_neg hab)
    rw [ho, ho] at this
    linarith

/-! ## Rounding to a grid that coarsens with the binade -/

/-- **Rounding to a binade grid is monotone.** For a monotone `r : ℚ → ℤ` fixing the integers, the rounding
`x ↦ r(x / u x)·u x` on the grid `u x = 2^(max (binade x) emin − k)` is monotone on the positive rationals: in one
grid it is `r`'s monotonicity; across a binade `2^B` of the larger, `2^B` lies on both grids, below the larger's
rounding and above the smaller's. -/
theorem grid_mono (r : ℚ → ℤ) (hr : Monotone r) (hri : ∀ n : ℤ, r n = n) (emin : ℤ) (k : ℕ) {a b : ℚ}
    (ha : 0 < a) (hab : a ≤ b) :
    (r (a / 2 ^ (max (binade a) emin - k)) : ℚ) * 2 ^ (max (binade a) emin - k) ≤
      (r (b / 2 ^ (max (binade b) emin - k)) : ℚ) * 2 ^ (max (binade b) emin - k) := by
  have hb : 0 < b := lt_of_lt_of_le ha hab
  have hbin := binade_mono ha hab
  set Ea := max (binade a) emin with hEa
  set Eb := max (binade b) emin with hEb
  have hle : Ea ≤ Eb := max_le_max hbin le_rfl
  rcases hle.lt_or_eq with hlt | heq
  · have hEb' : Eb = binade b := by
      rcases le_total (binade b) emin with h | h
      · exfalso; rw [hEb, max_eq_right h] at hlt; exact absurd (le_max_right (binade a) emin) (not_le.2 hlt)
      · exact max_eq_left h
    obtain ⟨hb1, -⟩ := binade_spec hb
    obtain ⟨-, ha2⟩ := binade_spec ha
    -- the larger rounds to at least `2^Eb`
    have hgb : (0 : ℚ) < 2 ^ (Eb - k) := two_zpow_pos _
    have hkk : ((2 : ℤ) ^ k : ℤ) = ((2 : ℚ) ^ k : ℚ) := by push_cast; ring
    have h1 : ((2 ^ k : ℤ) : ℚ) ≤ b / 2 ^ (Eb - k) := by
      rw [le_div_iff₀ hgb, hkk, ← zpow_natCast, ← zpow_add₀ two_ne_zero, show (k : ℤ) + (Eb - k) = Eb by ring,
        hEb']
      exact hb1
    have h2 : (2 ^ k : ℤ) ≤ r (b / 2 ^ (Eb - k)) := by rw [← hri (2 ^ k)]; exact hr h1
    have h3 : (2 : ℚ) ^ Eb ≤ (r (b / 2 ^ (Eb - k)) : ℚ) * 2 ^ (Eb - k) := by
      have : ((2 ^ k : ℤ) : ℚ) * 2 ^ (Eb - k) ≤ (r (b / 2 ^ (Eb - k)) : ℚ) * 2 ^ (Eb - k) :=
        mul_le_mul_of_nonneg_right (by exact_mod_cast h2) hgb.le
      rwa [hkk, ← zpow_natCast, ← zpow_add₀ two_ne_zero, show (k : ℤ) + (Eb - k) = Eb by ring] at this
    -- the smaller rounds to at most `2^Eb`
    have hga : (0 : ℚ) < 2 ^ (Ea - k) := two_zpow_pos _
    obtain ⟨N, hN⟩ : ∃ N : ℕ, (N : ℤ) = Eb - Ea + k := ⟨(Eb - Ea + k).toNat, Int.toNat_of_nonneg (by omega)⟩
    have hNq : ((2 ^ N : ℤ) : ℚ) * 2 ^ (Ea - k) = 2 ^ Eb := by
      push_cast
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero, hN]; congr 1; ring
    have h4 : a / 2 ^ (Ea - k) ≤ ((2 ^ N : ℤ) : ℚ) := by
      rw [div_le_iff₀ hga, hNq]
      calc a ≤ 2 ^ (binade a + 1) := ha2.le
        _ ≤ 2 ^ Eb := zpow_le_zpow_right₀ one_le_two (by omega)
    have h5 : r (a / 2 ^ (Ea - k)) ≤ 2 ^ N := by rw [← hri (2 ^ N)]; exact hr h4
    have h6 : (r (a / 2 ^ (Ea - k)) : ℚ) * 2 ^ (Ea - k) ≤ 2 ^ Eb := by
      rw [← hNq]; exact mul_le_mul_of_nonneg_right (by exact_mod_cast h5) hga.le
    linarith
  · rw [heq]
    have hg : (0 : ℚ) < 2 ^ (Eb - k) := two_zpow_pos _
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast hr (div_le_div_of_nonneg_right hab hg.le)) hg.le

/-! ## E4M3's encoder -/

theorem codes_le_448 : ∀ c : Code, codeVal c ≤ 448 := by decide +kernel

theorem codeVal_flip : ∀ c : Code, codeVal ⟨(128 + c.val) % 256, Nat.mod_lt _ (by decide)⟩ = -codeVal c := by
  decide +kernel

theorem codeVal_e4m3_le (x : ℚ) : codeVal (e4m3 x) ≤ 448 := codes_le_448 _

/-- E4M3 saturates from `464` up. -/
theorem codeVal_e4m3_sat {x : ℚ} (hx : 464 ≤ x) : codeVal (e4m3 x) = 448 := by
  have h1 : ¬ x < 0 := by linarith
  have h2 : x ≠ 0 := by intro h; rw [h] at hx; norm_num at hx
  have e : e4m3 x = ⟨126, by decide⟩ := by
    unfold e4m3 e4m3Mag
    simp only [h1, ↓reduceIte, abs_of_nonneg (by linarith : (0 : ℚ) ≤ x), h2, hx, zero_add]
  rw [e]; decide +kernel

theorem codeVal_e4m3_zero : codeVal (e4m3 0) = 0 := by decide +kernel

theorem codeVal_e4m3_neg (x : ℚ) : codeVal (e4m3 (-x)) = -codeVal (e4m3 x) := by
  rcases lt_trichotomy x 0 with h | rfl | h
  · have hx : ¬ (-x < 0) := by linarith
    have e : e4m3 x = ⟨(128 + (e4m3 (-x)).val) % 256, Nat.mod_lt _ (by decide)⟩ := by
      unfold e4m3; simp only [h, hx, abs_neg, ↓reduceIte, zero_add, Fin.mk.injEq]; omega
    rw [e, codeVal_flip]; ring
  · rw [neg_zero, codeVal_e4m3_zero, neg_zero]
  · have hx : -x < 0 := by linarith
    have hx' : ¬ x < 0 := by linarith
    have e : e4m3 (-x) = ⟨(128 + (e4m3 x).val) % 256, Nat.mod_lt _ (by decide)⟩ := by
      unfold e4m3; simp only [hx, hx', abs_neg, ↓reduceIte, zero_add, Fin.mk.injEq]; omega
    rw [e, codeVal_flip]

theorem codeVal_e4m3_mono_nonneg {a b : ℚ} (ha : 0 ≤ a) (hab : a ≤ b) : codeVal (e4m3 a) ≤ codeVal (e4m3 b) := by
  rcases le_or_gt 464 b with hb | hb
  · rw [codeVal_e4m3_sat hb]; exact codeVal_e4m3_le a
  have hb0 : 0 ≤ b := ha.trans hab
  have ha' : |a| < 464 := by rw [abs_of_nonneg ha]; linarith
  have hb' : |b| < 464 := by rw [abs_of_nonneg hb0]; linarith
  rw [Forming.codeVal_e4m3 ha', Forming.codeVal_e4m3 hb']
  simp only [not_lt.2 ha, not_lt.2 hb0, ↓reduceIte, one_mul, abs_of_nonneg ha, abs_of_nonneg hb0, Forming.e4m3Ulp]
  rcases ha.lt_or_eq with ha0 | rfl
  · exact grid_mono roundEven (fun _ _ h => roundEven_mono h) roundEven_intCast (-6) 3 ha0 hab
  · have h0 : roundEven 0 = 0 := by simpa using roundEven_intCast 0
    simp only [zero_div, h0, Int.cast_zero, zero_mul]
    have : 0 ≤ roundEven (b / 2 ^ (max (binade b) (-6) - 3)) := by
      rw [← h0]; exact roundEven_mono (div_nonneg hb0 (two_zpow_pos _).le)
    exact mul_nonneg (by exact_mod_cast this) (two_zpow_pos _).le

/-- **`codeVal ∘ e4m3` is monotone.** -/
theorem codeVal_e4m3_mono : Monotone fun x : ℚ => codeVal (e4m3 x) :=
  monotone_of_odd (fun _ _ ha hab => codeVal_e4m3_mono_nonneg ha hab) codeVal_e4m3_neg

/-- `e4m3Mag` never reaches the NaN magnitude `127`: it is `126` (448) at most. -/
theorem e4m3Mag_le {t : ℚ} (ht : 0 ≤ t) : e4m3Mag t ≤ 126 := by
  unfold e4m3Mag
  split_ifs with h0 h464
  · omega
  · omega
  · dsimp only
    have htp : 0 < t := lt_of_le_of_ne ht (Ne.symm h0)
    obtain ⟨hb1, hb2⟩ := binade_spec htp
    have hb9 : binade t < 9 := by
      have : (2 : ℚ) ^ binade t < 2 ^ (9 : ℤ) := lt_of_le_of_lt hb1 (by norm_num; linarith)
      exact (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 this
    set E := max (binade t) (-6) with hE
    have hu : (0 : ℚ) < 2 ^ (E - 3) := two_zpow_pos _
    have k16 : (2 : ℚ) ^ (E - 3) * 16 = 2 ^ (E + 1) := by
      rw [show (16 : ℚ) = 2 ^ (4 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
    have hq16 : t / 2 ^ (E - 3) < 16 := by
      rw [div_lt_iff₀ hu, mul_comm, k16]
      exact lt_of_lt_of_le hb2 (zpow_le_zpow_right₀ one_le_two (by omega))
    have hR0 : 0 ≤ roundEven (t / 2 ^ (E - 3)) :=
      le_trans (Int.floor_nonneg.2 (div_nonneg ht hu.le)) (floor_le_roundEven _)
    have hR16 : roundEven (t / 2 ^ (E - 3)) ≤ 16 := by
      have h1 := roundEven_le (t / 2 ^ (E - 3))
      have h2 : ⌊t / 2 ^ (E - 3)⌋ < 16 := Int.floor_lt.2 (by exact_mod_cast hq16)
      omega
    have htop : E = 8 → roundEven (t / 2 ^ (E - 3)) ≤ 14 := by
      intro h8
      have hq : t / 2 ^ (E - 3) < 29 / 2 := by
        rw [h8, div_lt_iff₀ (by norm_num)]; norm_num; linarith
      have h15 : ((roundEven (t / 2 ^ (E - 3)) : ℤ) : ℚ) < 15 := by
        have := AtomErr.abs_roundEven_sub_le (t / 2 ^ (E - 3))
        rw [abs_le] at this; linarith
      have : roundEven (t / 2 ^ (E - 3)) < 15 := by exact_mod_cast h15
      omega
    have hE6 : -6 ≤ E := le_max_right _ _
    have hE8 : E ≤ 8 := max_le (by omega) (by norm_num)
    rcases (show E ≤ 7 ∨ E = 8 by omega) with h | h
    · omega
    · have := htop h; omega

/-! ## Truncation to `p` bits with FP32's subnormals -/

/-- Truncation toward zero to `p` significant bits, on FP32's exponent range from below: the grid is
`2^(max (binade |y|) (−126) − (p − 1))` (`FormingP.truncP` above `2^−126`). -/
def truncF (p : ℕ) (y : ℚ) : ℚ :=
  if y = 0 then 0 else truncTo (2 ^ (max (binade |y|) (-126) - ((p : ℤ) - 1))) y

theorem truncTo_of_pos {g y : ℚ} (hy : 0 < y) : truncTo g y = (⌊y / g⌋ : ℚ) * g := by
  simp [truncTo, not_lt.2 hy.le, abs_of_pos hy]

theorem truncF_nonneg (p : ℕ) {y : ℚ} (hy : 0 ≤ y) : 0 ≤ truncF p y := by
  unfold truncF
  split_ifs with h0
  · exact le_rfl
  · have hy' : 0 < y := lt_of_le_of_ne hy (Ne.symm h0)
    rw [truncTo_of_pos hy']
    exact mul_nonneg (by exact_mod_cast Int.floor_nonneg.2 (div_nonneg hy (two_zpow_pos _).le)) (two_zpow_pos _).le

theorem truncF_neg (p : ℕ) (y : ℚ) : truncF p (-y) = -truncF p y := by
  unfold truncF
  rcases eq_or_ne y 0 with rfl | h
  · simp
  · rw [ite_eq_right (neg_ne_zero.2 h), ite_eq_right h, abs_neg, truncTo_neg]

theorem truncF_mono_nonneg {p : ℕ} (hp : 1 ≤ p) {a b : ℚ} (ha : 0 ≤ a) (hab : a ≤ b) : truncF p a ≤ truncF p b := by
  rcases ha.lt_or_eq with ha0 | rfl
  · have hb0 : 0 < b := lt_of_lt_of_le ha0 hab
    have hk : ((p : ℤ) - 1) = ((p - 1 : ℕ) : ℤ) := by omega
    unfold truncF
    rw [ite_eq_right ha0.ne', ite_eq_right hb0.ne', truncTo_of_pos ha0, truncTo_of_pos hb0, abs_of_pos ha0, abs_of_pos hb0, hk]
    exact grid_mono (fun x => ⌊x⌋) Int.floor_mono Int.floor_intCast (-126) (p - 1) ha0 hab
  · have : truncF p 0 = 0 := by simp [truncF]
    rw [this]; exact truncF_nonneg p (le_trans le_rfl hab)

/-! ## The normalization -/

theorem bitLength_of_ne {n : ℕ} (hn : n ≠ 0) : bitLength n = n.log2 + 1 := by
  unfold bitLength; rw [ite_eq_right hn]

/-- The adder's `w` bits of `n` from its leading bit `L`: a right shift if `n` is wider, else a left shift. -/
theorem shift_bounds {w n L : ℕ} (hw : 1 ≤ w) (hL1 : 2 ^ L ≤ n) (hL2 : n < 2 ^ (L + 1)) :
    2 ^ (w - 1) ≤ (if w < L + 1 then n >>> (L + 1 - w) else n <<< (w - (L + 1))) ∧
      (if w < L + 1 then n >>> (L + 1 - w) else n <<< (w - (L + 1))) < 2 ^ w := by
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow]
    constructor
    · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]; rwa [show w - 1 + (L + 1 - w) = L by omega]
    · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]; rwa [show w + (L + 1 - w) = L + 1 by omega]
  · rw [Nat.shiftLeft_eq]
    constructor
    · calc 2 ^ (w - 1) = 2 ^ L * 2 ^ (w - (L + 1)) := by rw [← pow_add]; congr 1; omega
        _ ≤ _ := Nat.mul_le_mul_right _ hL1
    · calc n * 2 ^ (w - (L + 1)) < 2 ^ (L + 1) * 2 ^ (w - (L + 1)) :=
            Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
        _ = 2 ^ w := by rw [← pow_add]; congr 1; omega

/-- A further right shift by `d` of the adder's bits is the adder's bits of `n` read from `L + d`. -/
theorem shift_shiftRight (w n L d : ℕ) :
    (if w < L + 1 then n >>> (L + 1 - w) else n <<< (w - (L + 1))) >>> d =
      (if w < L + d + 1 then n >>> (L + d + 1 - w) else n <<< (w - (L + d + 1))) := by
  by_cases h : w < L + 1
  · rw [ite_eq_left h, ite_eq_left (by omega), ← Nat.shiftRight_add]; congr 1; omega
  · rw [ite_eq_right h, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
    by_cases h' : w < L + d + 1
    · rw [ite_eq_left h', Nat.shiftRight_eq_div_pow,
        show 2 ^ d = 2 ^ (w - (L + 1)) * 2 ^ (L + d + 1 - w) by rw [← pow_add]; congr 1; omega,
        mul_comm n, Nat.mul_div_mul_left _ _ (by positivity)]
    · rw [ite_eq_right h', Nat.shiftLeft_eq,
        show 2 ^ (w - (L + 1)) = 2 ^ (w - (L + d + 1)) * 2 ^ d by rw [← pow_add]; congr 1; omega,
        ← mul_assoc, Nat.mul_div_cancel _ (by positivity)]

/-- **The normalization of a nonzero natural sum.** At any width `w ≥ 1` and exponent `E`, `normalize w E n` is a
canonical, nonnegative term whose value is `n·2^(E − w + 1)` truncated to `min w 24` bits with FP32's subnormals; its
exponent is `e = E + bitLength n − w` when `e ≥ −126` (and the term is nonzero), else `−126` or the term is zero. -/
theorem normalize_nat {w : ℕ} (hw : 1 ≤ w) (E : ℤ) {n : ℕ} (hn : n ≠ 0) :
    (normalize w E n).neg = false ∧ HonestWords.Canon (normalize w E n) ∧
      (normalize w E n).value = truncF (min w 24) ((n : ℚ) * 2 ^ (E - w + 1)) ∧
      (-126 ≤ E + ((n.log2 + 1 : ℕ) : ℤ) - w →
        (normalize w E n).mag ≠ 0 ∧ (normalize w E n).exp = E + ((n.log2 + 1 : ℕ) : ℤ) - w) ∧
      (E + ((n.log2 + 1 : ℕ) : ℤ) - w < -126 → (normalize w E n).mag ≠ 0 → (normalize w E n).exp = -126) := by
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self n
  set L := n.log2 with hLdef
  set m0 := (if w < L + 1 then n >>> (L + 1 - w) else n <<< (w - (L + 1))) with hm0
  have hb1 : 2 ^ (w - 1) ≤ m0 ∧ m0 < 2 ^ w := shift_bounds hw hL1 hL2
  have hn0 : (n : ℤ) ≠ 0 := by exact_mod_cast hn
  have hbl : bitLength n = L + 1 := bitLength_of_ne hn
  have hX0 : (n : ℚ) * 2 ^ (E - w + 1) ≠ 0 := mul_ne_zero (by exact_mod_cast hn) (two_zpow_ne _)
  have hXp : 0 < (n : ℚ) * 2 ^ (E - w + 1) := mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hn) (two_zpow_pos _)
  have hbin : binade |(n : ℚ) * 2 ^ (E - w + 1)| = L + (E - w + 1) := by
    rw [abs_of_pos hXp, binade_nat_mul hn]
  have hmin : 1 ≤ min w 24 := le_min hw (by norm_num)
  have hdec : decide ((n : ℤ) < 0) = false := by simp
  unfold normalize
  simp only [hn0, ite_false, Int.natAbs_natCast, hbl, ← hm0, hdec]
  by_cases hsub : E + ((L + 1 : ℕ) : ℤ) - (w : ℤ) < -126
  · -- the subnormal branch
    simp only [hsub, ite_true]
    set d := (-126 - (E + ((L + 1 : ℕ) : ℤ) - (w : ℤ))).toNat with hd
    have hd1 : 1 ≤ d := by omega
    have hsh := shift_shiftRight w n L d
    rw [← hm0] at hsh
    have hfl := FormingP.fromAdder_floor w n (L + d)
    rw [← hsh] at hfl
    set M := fromAdder w (m0 >>> d) with hM
    -- the magnitude is below `2^23`
    have hm2 : m0 >>> d < 2 ^ (w - 1) := by
      rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity)]
      calc m0 < 2 ^ w := hb1.2
        _ = 2 ^ (w - 1) * 2 ^ 1 := by rw [← pow_add]; congr 1; omega
        _ ≤ 2 ^ (w - 1) * 2 ^ d := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by norm_num) hd1)
    have hM23 : M < 2 ^ 23 := by
      rw [hM]; unfold fromAdder
      split_ifs with h
      · rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]
        exact lt_of_lt_of_le hm2 (Nat.pow_le_pow_right (by norm_num) (by omega))
      · rw [Nat.shiftLeft_eq]
        calc (m0 >>> d) * 2 ^ (24 - w) < 2 ^ (w - 1) * 2 ^ (24 - w) :=
              Nat.mul_lt_mul_of_pos_right hm2 (by positivity)
          _ = 2 ^ 23 := by rw [← pow_add]; congr 1; omega
    -- the value
    have hval : ((M : ℕ) : ℚ) * 2 ^ ((-126 : ℤ) - 23) = truncF (min w 24) ((n : ℚ) * 2 ^ (E - w + 1)) := by
      unfold truncF
      rw [ite_eq_right hX0, hbin, max_eq_right (by omega), truncTo_of_pos hXp, hfl]
      have hdiv : (n : ℚ) * 2 ^ (E - w + 1) / 2 ^ ((-126 : ℤ) - ((min w 24 : ℕ) - 1)) =
          (n : ℚ) * 2 ^ (((min w 24 : ℕ) : ℤ) - 1 - ((L + d : ℕ) : ℤ)) := by
        rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; push_cast; omega
      rw [hdiv, ← Int.natCast_floor_eq_floor (mul_nonneg (Nat.cast_nonneg _) (two_zpow_pos _).le)]
      push_cast
      rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne_zero]
      congr 2; omega
    by_cases hM0 : M = 0
    · rw [ite_eq_left hM0]
      refine ⟨rfl, Or.inl rfl, ?_, fun h => absurd h (by omega), fun _ h => absurd rfl h⟩
      rw [← hval, hM0]; simp [Term.value, Term.zero]
    · rw [ite_eq_right hM0]
      refine ⟨rfl, Or.inr ⟨show M < 2 ^ 24 by omega, show (-126 : ℤ) ≤ -126 from le_rfl, Or.inr rfl⟩, ?_,
        fun h => absurd h (by omega), fun _ _ => rfl⟩
      rw [← hval]; simp [Term.value]
  · -- the normal branch (`FormingP.normalizeW_spec`'s computation)
    simp only [hsub, ite_false]
    have hfl := FormingP.fromAdder_floor w n L
    rw [← hm0] at hfl
    have hfa : 2 ^ 23 ≤ fromAdder w m0 ∧ fromAdder w m0 < 2 ^ 24 := by
      unfold fromAdder
      split_ifs with h
      · rw [Nat.shiftRight_eq_div_pow]
        constructor
        · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]
          exact le_trans (Nat.pow_le_pow_right (by norm_num) (by omega)) hb1.1
        · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]
          exact lt_of_lt_of_le hb1.2 (Nat.pow_le_pow_right (by norm_num) (by omega))
      · rw [Nat.shiftLeft_eq]
        constructor
        · calc 2 ^ 23 = 2 ^ (w - 1) * 2 ^ (24 - w) := by rw [← pow_add]; congr 1; omega
            _ ≤ _ := Nat.mul_le_mul_right _ hb1.1
        · calc m0 * 2 ^ (24 - w) < 2 ^ w * 2 ^ (24 - w) := Nat.mul_lt_mul_of_pos_right hb1.2 (by positivity)
            _ = 2 ^ 24 := by rw [← pow_add]; congr 1; omega
    rw [ite_eq_right (by omega : fromAdder w m0 ≠ 0)]
    refine ⟨rfl, Or.inr ⟨hfa.2, show (-126 : ℤ) ≤ E + ((L + 1 : ℕ) : ℤ) - w by omega, Or.inl hfa.1⟩, ?_,
      fun _ => ⟨show fromAdder w m0 ≠ 0 by omega, rfl⟩, fun h => h.elim⟩
    unfold truncF
    rw [ite_eq_right hX0, hbin, max_eq_left (by omega), truncTo_of_pos hXp]
    have hdiv : (n : ℚ) * 2 ^ (E - w + 1) / 2 ^ ((L : ℤ) + (E - w + 1) - ((min w 24 : ℕ) - 1)) =
        (n : ℚ) * 2 ^ (((min w 24 : ℕ) : ℤ) - 1 - L) := by
      rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; ring
    rw [hdiv, ← Int.natCast_floor_eq_floor (mul_nonneg (Nat.cast_nonneg _) (two_zpow_pos _).le)]
    simp only [Term.value, Bool.false_eq_true, ite_false, one_mul]
    rw [hfl]
    push_cast
    rw [mul_assoc, ← zpow_natCast, ← zpow_add₀ two_ne_zero]
    congr 2; omega

/-- A term with its sign flipped (zero stays zero). -/
def negT (T : Term) : Term := if T.mag = 0 then T else ⟨!T.neg, T.exp, T.mag⟩

theorem negT_mag (T : Term) : (negT T).mag = T.mag := by unfold negT; split_ifs <;> rfl

theorem negT_exp (T : Term) : (negT T).exp = T.exp := by unfold negT; split_ifs <;> rfl

theorem negT_value (T : Term) : (negT T).value = -T.value := by
  obtain ⟨neg, exp, mag⟩ := T
  unfold negT
  split_ifs with h
  · simp only at h; simp [Term.value, h]
  · cases neg <;> simp [Term.value]

theorem negT_canon {T : Term} (h : HonestWords.Canon T) : HonestWords.Canon (negT T) := by
  unfold negT; split_ifs
  · exact h
  · exact h

/-- A negated sum normalizes to the negated term. -/
theorem normalize_neg (w : ℕ) (E t : ℤ) : normalize w E (-t) = negT (normalize w E t) := by
  by_cases h : t = 0
  · subst h; simp [normalize, negT, Term.zero]
  · have hd : decide (-t < 0) = !decide (t < 0) := by
      rcases lt_or_gt_of_ne h with h' | h'
      · simp [h', le_of_lt h']
      · simp [h', not_lt.2 h'.le]
    unfold normalize
    simp only [neg_eq_zero, h, ite_false, Int.natAbs_neg, hd]
    generalize fromAdder w _ = M
    unfold negT
    by_cases hM : M = 0
    · simp [hM, Term.zero]
    · simp [hM]

theorem canon_normalize {w : ℕ} (hw : 1 ≤ w) (E t : ℤ) : HonestWords.Canon (normalize w E t) := by
  rcases le_total 0 t with ht | ht
  · obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le ht
    rcases eq_or_ne n 0 with rfl | hn
    · exact Or.inl (by simp [normalize, Term.zero])
    · exact (normalize_nat hw E hn).2.1
  · obtain ⟨n, hn'⟩ := Int.eq_ofNat_of_zero_le (neg_nonneg.2 ht)
    have : t = -(n : ℤ) := by omega
    rw [this, normalize_neg]
    rcases eq_or_ne n 0 with rfl | hn
    · exact negT_canon (Or.inl (by simp [normalize, Term.zero]))
    · exact negT_canon (normalize_nat hw E hn).2.1

/-! ## The read -/

/-- The group step's outgoing value from its normalized term: `±inf` above exponent 127. -/
def ovfV (T : Term) : Value := if T.mag ≠ 0 ∧ 127 < T.exp then .inf T.neg else .fin T

/-- **The value the cast reads** from a group's sum `t` at the alignment exponent `E`: `wordVal` of the word the step
writes (`pipeStep_fin`'s `packValue` of the normalized term, or of its infinity). -/
def readW (w : ℕ) (E t : ℤ) : ℚ := PearlC.wordVal (packValue (ovfV (normalize w E t)))

/-- The code's value: `e4m3` of the read. -/
def codeW (w : ℕ) (E t : ℤ) : ℚ := codeVal (e4m3 (readW w E t))

theorem wordVal_ovfV {T : Term} (hc : HonestWords.Canon T) :
    PearlC.wordVal (packValue (ovfV T)) =
      if T.mag ≠ 0 ∧ 127 < T.exp then (if T.neg then -480 else 480) else T.value := by
  unfold ovfV
  split_ifs with h hn
  · simp only [PearlC.wordVal, HonestWords.wordQ_packValue_inf, hn]
    rw [ite_eq_left (by decide)]
  · simp only [PearlC.wordVal, HonestWords.wordQ_packValue_inf, hn]
    rw [ite_eq_right (by decide)]
  · simp only [PearlC.wordVal, packValue,
      HonestWords.wordQ_packTerm_canon hc (fun hm => not_lt.1 fun h' => h ⟨hm, h'⟩)]

theorem readW_neg {w : ℕ} (hw : 1 ≤ w) (E t : ℤ) : readW w E (-t) = -readW w E t := by
  have hc := canon_normalize hw E t
  unfold readW
  rw [normalize_neg, wordVal_ovfV (negT_canon hc), wordVal_ovfV hc, negT_mag, negT_exp, negT_value]
  by_cases h : (normalize w E t).mag ≠ 0 ∧ 127 < (normalize w E t).exp
  · have hn : (negT (normalize w E t)).neg = !(normalize w E t).neg := by unfold negT; rw [ite_eq_right h.1]
    rw [ite_eq_left h, ite_eq_left h, hn]
    cases (normalize w E t).neg <;> norm_num
  · rw [ite_eq_right h, ite_eq_right h]

theorem codeW_neg {w : ℕ} (hw : 1 ≤ w) (E t : ℤ) : codeW w E (-t) = -codeW w E t := by
  unfold codeW; rw [readW_neg hw, codeVal_e4m3_neg]

/-- The read of a natural sum: `480` (the overflow branch) when `E + bitLength n − w` exceeds 127, else the
truncated sum. -/
theorem readW_nat {w : ℕ} (hw : 1 ≤ w) (E : ℤ) (n : ℕ) :
    readW w E n = if n ≠ 0 ∧ 127 < E + ((n.log2 + 1 : ℕ) : ℤ) - w then 480
      else truncF (min w 24) ((n : ℚ) * 2 ^ (E - w + 1)) := by
  unfold readW
  rcases eq_or_ne n 0 with rfl | hn
  · have h0 : normalize w E ((0 : ℕ) : ℤ) = Term.zero := by simp [normalize]
    rw [h0, wordVal_ovfV (Or.inl rfl)]
    simp [Term.zero, Term.value, truncF]
  obtain ⟨hneg, hc, hv, hhi, hlo⟩ := normalize_nat hw E hn
  rw [wordVal_ovfV hc, hneg, hv]
  by_cases h126 : -126 ≤ E + ((n.log2 + 1 : ℕ) : ℤ) - w
  · obtain ⟨hm, he⟩ := hhi h126
    simp only [hm, he, hn, ne_eq, not_false_eq_true, true_and, Bool.false_eq_true, ite_false]
  · have h1 : ¬ ((normalize w E n).mag ≠ 0 ∧ 127 < (normalize w E n).exp) := by
      rintro ⟨hm, h⟩; rw [hlo (by omega) hm] at h; omega
    rw [ite_eq_right h1, ite_eq_right (by omega)]

theorem codeW_mono_nat {w : ℕ} (hw : 1 ≤ w) (E : ℤ) {a b : ℕ} (hab : a ≤ b) : codeW w E a ≤ codeW w E b := by
  have hp : 1 ≤ min w 24 := le_min hw (by norm_num)
  have hg : (0 : ℚ) ≤ 2 ^ (E - w + 1) := (two_zpow_pos _).le
  unfold codeW
  rw [readW_nat hw, readW_nat hw]
  split_ifs with ha hb hb
  · exact le_rfl
  · exfalso
    obtain ⟨ha0, h⟩ := ha
    have hb0 : b ≠ 0 := by omega
    have hlog : a.log2 ≤ b.log2 := (Nat.le_log2 hb0).2 (le_trans (Nat.log2_self_le ha0) hab)
    exact hb ⟨hb0, by omega⟩
  · rw [codeVal_e4m3_sat (x := 480) (by norm_num)]; exact codeVal_e4m3_le _
  · have hab' : (a : ℚ) * 2 ^ (E - w + 1) ≤ (b : ℚ) * 2 ^ (E - w + 1) :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hab) hg
    exact codeVal_e4m3_mono_nonneg (truncF_nonneg _ (mul_nonneg (Nat.cast_nonneg _) hg))
      (truncF_mono_nonneg hp (mul_nonneg (Nat.cast_nonneg _) hg) hab')

/-- **The read is monotone in the group's sum**, at every width `w ≥ 1` and alignment exponent `E`. -/
theorem codeW_mono {w : ℕ} (hw : 1 ≤ w) (E : ℤ) : Monotone (codeW w E) := by
  refine monotone_of_odd (fun a b ha hab => ?_) (codeW_neg hw E)
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le ha
  obtain ⟨m, rfl⟩ := Int.eq_ofNat_of_zero_le (le_trans ha hab)
  exact codeW_mono_nat hw E (by exact_mod_cast hab)

/-! ## The FP32 word's sign -/

/-- `f32Word y` is its sign bit plus the word of `|y|`. -/
theorem f32Word_sign (y : ℚ) : f32Word y = (if y < 0 then 2 ^ 31 else 0) + f32Word |y| := by
  rcases eq_or_ne y 0 with rfl | hy
  · simp [f32Word]
  · have ha : |y| ≠ 0 := abs_ne_zero.2 hy
    simp only [f32Word, abs_abs, not_lt.2 (abs_nonneg y), ↓reduceIte, ha, zero_add]
    split_ifs <;> omega

/-- The word of `|y|` is `+inf` or a finite word: exponent field below 255 and a 23-bit mantissa. -/
theorem f32Word_abs_form (y : ℚ) :
    f32Word |y| = 0x7F800000 ∨ ∃ e m : ℕ, e < 255 ∧ m < 2 ^ 23 ∧ f32Word |y| = e * 2 ^ 23 + m := by
  rcases eq_or_ne y 0 with rfl | hy
  · right; exact ⟨0, 0, by norm_num, by norm_num, by simp [f32Word]⟩
  have ha : 0 < |y| := abs_pos.2 hy
  obtain ⟨-, hb2⟩ := binade_spec ha
  set E₀ := max (binade |y|) (-126) with hE₀
  have hu : (0 : ℚ) < 2 ^ (E₀ - 23) := two_zpow_pos _
  have hlt : |y| / 2 ^ (E₀ - 23) < 2 ^ 24 := by
    rw [div_lt_iff₀ hu]
    calc |y| < 2 ^ (binade |y| + 1) := hb2
      _ ≤ 2 ^ (E₀ + 1) := zpow_le_zpow_right₀ one_le_two (by omega)
      _ = 2 ^ 24 * 2 ^ (E₀ - 23) := by
        rw [show (2 : ℚ) ^ 24 = 2 ^ (24 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
  have hN : (roundEven (|y| / 2 ^ (E₀ - 23))).toNat ≤ 2 ^ 24 := by
    have h1 := roundEven_le (|y| / 2 ^ (E₀ - 23))
    have h2 : ⌊|y| / 2 ^ (E₀ - 23)⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hlt)
    omega
  simp only [f32Word, abs_abs, not_lt.2 (abs_nonneg y), ↓reduceIte, ha.ne', zero_add, ← hE₀]
  generalize (roundEven (|y| / 2 ^ (E₀ - 23))).toNat = N₀ at hN ⊢
  split_ifs
  all_goals first
    | (exfalso; omega)
    | (left; decide)
    | (right; exact ⟨0, N₀, by norm_num, by assumption, by ring⟩)
    | (right; exact ⟨(E₀ + 1 + 127).toNat, 0, by omega, by norm_num, by omega⟩)
    | (right; exact ⟨(E₀ + 127).toNat, N₀ - 2 ^ 23, by omega, by omega, by omega⟩)

theorem abs_term_value (b b' : Bool) (ex : ℤ) (mg : ℕ) :
    |Term.value ⟨b, ex, mg⟩| = |Term.value ⟨b', ex, mg⟩| := by
  simp only [Term.value]; cases b <;> cases b' <;> simp [abs_mul]

/-- An E4M3 code below `126` in its magnitude bits and that code with the sign bit set: neither is NaN, they share
the significand and exponent, and only the second is negative. -/
theorem code_pair : ∀ c d : Code, c.val ≤ 126 → d.val = 128 + c.val →
    c.isNaN = false ∧ c.sign = false ∧ d.isNaN = false ∧ d.sign = true ∧ d.sig = c.sig ∧ d.expOr1 = c.expOr1 := by
  decide +kernel

/-- **The cast of an FP32 product and of its negation**: `e4m3Word (f32Word y)` and `e4m3Word (f32Word (−y))` are one
magnitude code `K ≤ 126` with the sign bit of `y < 0` and of `0 < y` respectively. -/
theorem e4m3Word_signs (y : ℚ) : ∃ K : ℕ, K ≤ 126 ∧
    (e4m3Word (f32Word y)).val = (if y < 0 then 128 else 0) + K ∧
      (e4m3Word (f32Word (-y))).val = (if 0 < y then 128 else 0) + K := by
  have h1 := f32Word_sign y
  have h2 := f32Word_sign (-y)
  simp only [abs_neg, neg_lt_zero] at h2
  have key : ∃ K : ℕ, K ≤ 126 ∧ ∀ sb : ℕ, sb ≤ 1 → (e4m3Word (sb * 2 ^ 31 + f32Word |y|)).val = 128 * sb + K := by
    rcases f32Word_abs_form y with hU | ⟨e, m, he, hm, hU⟩
    · refine ⟨126, le_rfl, fun sb hsb => ?_⟩
      rw [hU]
      interval_cases sb <;> decide +kernel
    · obtain ⟨x0, hx0⟩ : ∃ x0 : ℚ, ∀ sb : ℕ, sb ≤ 1 → ∃ x, wordQ (sb * 2 ^ 31 + (e * 2 ^ 23 + m)) = some x ∧
          |x| = x0 := by
        refine ⟨|(if e = 0 then (if m = 0 then Term.zero else ⟨false, -126, m⟩)
          else (⟨false, (e : ℤ) - 127, 2 ^ 23 + m⟩ : Term)).value|, fun sb hsb => ?_⟩
        rw [← add_assoc, wordQ, AtomErr.fp32Value_word sb e m hsb he hm]
        split_ifs <;> simp only [Value.finite?, Option.map_some, Option.some.injEq, exists_eq_left'] <;>
          first | rfl | exact abs_term_value _ _ _ _
      refine ⟨e4m3Mag x0, ?_, fun sb hsb => ?_⟩
      · obtain ⟨x, -, hx⟩ := hx0 0 (by norm_num)
        rw [← hx]; exact e4m3Mag_le (abs_nonneg _)
      obtain ⟨x, hx, hxa⟩ := hx0 sb hsb
      have hK : e4m3Mag x0 ≤ 126 := by rw [← hxa]; exact e4m3Mag_le (abs_nonneg _)
      have hs : ((sb * 2 ^ 31 + (e * 2 ^ 23 + m)) >>> 31) % 2 = sb := by
        rw [Nat.shiftRight_eq_div_pow]; norm_num only at hm he hsb ⊢; omega
      rw [hU]
      unfold e4m3Word
      simp only [hs, hx, hxa]
      interval_cases sb <;> simp <;> omega
  obtain ⟨K, hK, hKs⟩ := key
  refine ⟨K, hK, ?_, ?_⟩
  · have := hKs (if y < 0 then 1 else 0) (by split_ifs <;> norm_num)
    rw [h1]
    split_ifs at this ⊢ <;> simpa using this
  · have := hKs (if 0 < y then 1 else 0) (by split_ifs <;> norm_num)
    rw [h2]
    split_ifs at this ⊢ <;> simpa using this

end Pouw.PearlC.Cap
