import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.PearlC.Defs
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# The FP32 values in order, and a value's neighbours (the cap circuit's certificates)

The cap circuit (`verity/protocols/accounting/work/pouw/circuit/cap.py`, `Pc8CapWord`) is given only `lo` and `hi` of each of Freeze's
certificates. It takes `c` and `d` as their neighbours in FP32 value order (`_Cmp.pred`, `_Cmp.succ`): one step on the
sign-magnitude encoding, and `∓2^-149` at `±0`. This file puts the FP32 values in order on ℤ, so that "the next value"
is an index step:

* `gridInt j` is the `j`-th nonnegative FP32 value in units of `2^-149`: `j` itself for `j < 2^23` (the subnormals),
  else `(2^23 + j mod 2^23)·2^(j/2^23 − 1)`, with no top exponent. `gridQ k` is `±gridInt |k| · 2^-149`, strictly
  monotone in `k` (`gridQ_strictMono`), with both zeros at `0`.
* `OnGrid x`: `x` is some `gridQ k`. Every finite FP32 word's value is (`wordQ_eq_gridQ`: the value of `w` is
  `gridQ (wordIdx w)`), and so is `rnAdd` of two values on the grid (`rnAdd_onGrid`): a multiple of `2^-149` rounded to
  24 bits is still one. So every promoted total from `+0` is on the grid (`gridOpt_foldl_promote`).
* `wordIdx w` reads a word's sign-magnitude encoding as an integer. The circuit's neighbours of a word (`predWord`,
  `succWord`, `_Cmp.pred` and `_Cmp.succ` word for word) step its index by one (`wordIdx_predWord`, `wordIdx_succWord`).
-/

namespace Pouw.PearlC.Cap

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.PearlC

/-! ## The grid -/

/-- The `j`-th nonnegative FP32 value in units of `2^-149`, with no top exponent: `j < 2^23` is the subnormal `j`, else the
normal `(2^23 + j mod 2^23)·2^(j/2^23 − 1)` (exponent field `j/2^23`, mantissa `j mod 2^23`). -/
def gridInt (j : ℕ) : ℕ := if j < 2 ^ 23 then j else (2 ^ 23 + j % 2 ^ 23) * 2 ^ (j / 2 ^ 23 - 1)

/-- The FP32 values in units of `2^-149`, in value order on ℤ: `gridZ (−k) = −gridZ k`. -/
def gridZ (k : ℤ) : ℤ := if k < 0 then -(gridInt k.natAbs : ℤ) else (gridInt k.natAbs : ℤ)

/-- **The `k`-th FP32 value** in value order, `0` at `k = 0` (both zeros), with no top exponent. -/
def gridQ (k : ℤ) : ℚ := (gridZ k : ℚ) * 2 ^ (-149 : ℤ)

/-- `x` is an FP32 value (with no top exponent). -/
def OnGrid (x : ℚ) : Prop := ∃ k : ℤ, gridQ k = x

/-- A promoted total that is `none` or on the grid. -/
def GridOpt (t : Option ℚ) : Prop := ∀ y, t = some y → OnGrid y

theorem gridInt_zero : gridInt 0 = 0 := rfl

theorem gridInt_lt_succ (j : ℕ) : gridInt j < gridInt (j + 1) := by
  unfold gridInt
  rcases Nat.lt_or_ge (j + 1) (2 ^ 23) with h | h
  · rw [ite_eq_left (by omega), ite_eq_left h]; omega
  · rcases Nat.lt_or_ge j (2 ^ 23) with h' | h'
    · have hj : j + 1 = 2 ^ 23 := by omega
      rw [ite_eq_left h', ite_eq_right (by omega), hj]
      norm_num
      omega
    · rw [ite_eq_right (by omega), ite_eq_right (by omega)]
      obtain ⟨e, he⟩ : ∃ e, j / 2 ^ 23 = e + 1 := ⟨j / 2 ^ 23 - 1, by omega⟩
      by_cases hm : (j + 1) % 2 ^ 23 = 0
      · have h1 : (j + 1) / 2 ^ 23 = e + 2 := by omega
        have h2 : j % 2 ^ 23 = 2 ^ 23 - 1 := by omega
        rw [hm, h1, h2, he, show e + 2 - 1 = e + 1 by omega, show e + 1 - 1 = e by omega, pow_succ]
        have : 0 < 2 ^ e := Nat.two_pow_pos e
        norm_num
        omega
      · have h1 : (j + 1) / 2 ^ 23 = j / 2 ^ 23 := by omega
        have h2 : (j + 1) % 2 ^ 23 = j % 2 ^ 23 + 1 := by omega
        rw [h1, h2]
        exact Nat.mul_lt_mul_of_pos_right (by omega) (Nat.two_pow_pos _)

theorem gridInt_strictMono : StrictMono gridInt := strictMono_nat_of_lt_succ gridInt_lt_succ

theorem gridZ_lt_succ (k : ℤ) : gridZ k < gridZ (k + 1) := by
  unfold gridZ
  rcases lt_trichotomy k (-1) with h | h | h
  · rw [ite_eq_left (by omega), ite_eq_left (by omega)]
    have e : k.natAbs = (k + 1).natAbs + 1 := by omega
    rw [e]
    have := gridInt_lt_succ (k + 1).natAbs
    omega
  · subst h
    norm_num [gridInt]
  · rw [ite_eq_right (by omega), ite_eq_right (by omega)]
    have e : (k + 1).natAbs = k.natAbs + 1 := by omega
    rw [e]
    have := gridInt_lt_succ k.natAbs
    omega

theorem gridZ_strictMono : StrictMono gridZ := strictMono_int_of_lt_succ gridZ_lt_succ

/-- **The FP32 values are in order**: `gridQ` is strictly monotone. -/
theorem gridQ_strictMono : StrictMono gridQ := fun _ _ h =>
  mul_lt_mul_of_pos_right (Int.cast_lt.2 (gridZ_strictMono h)) (two_zpow_pos _)

theorem gridQ_lt_iff {a b : ℤ} : gridQ a < gridQ b ↔ a < b := gridQ_strictMono.lt_iff_lt

theorem gridQ_le_iff {a b : ℤ} : gridQ a ≤ gridQ b ↔ a ≤ b := gridQ_strictMono.le_iff_le

theorem gridQ_zero : gridQ 0 = 0 := by simp [gridQ, gridZ, gridInt]

theorem onGrid_gridQ (k : ℤ) : OnGrid (gridQ k) := ⟨k, rfl⟩

/-- **No FP32 value lies strictly between two neighbours**: below `gridQ k` an FP32 value is at most `gridQ (k − 1)`. -/
theorem le_pred_of_lt {x : ℚ} (hx : OnGrid x) {k : ℤ} (h : x < gridQ k) : x ≤ gridQ (k - 1) := by
  obtain ⟨j, rfl⟩ := hx
  exact gridQ_le_iff.2 (by have := gridQ_lt_iff.1 h; omega)

/-- Above `gridQ k` an FP32 value is at least `gridQ (k + 1)`. -/
theorem succ_le_of_lt {x : ℚ} (hx : OnGrid x) {k : ℤ} (h : gridQ k < x) : gridQ (k + 1) ≤ x := by
  obtain ⟨j, rfl⟩ := hx
  exact gridQ_le_iff.2 (by have := gridQ_lt_iff.1 h; omega)

/-! ## The grid is the multiples of `2^-149` with 24-bit significands -/

/-- `n` has a significand of at most 24 bits. -/
def Sig24 (n : ℕ) : Prop := ∃ M s : ℕ, M < 2 ^ 24 ∧ n = M * 2 ^ s

theorem sig24_gridInt (j : ℕ) : Sig24 (gridInt j) := by
  unfold gridInt
  split_ifs with h
  · exact ⟨j, 0, by omega, by simp⟩
  · exact ⟨2 ^ 23 + j % 2 ^ 23, j / 2 ^ 23 - 1, by omega, rfl⟩

theorem normalize24 : ∀ (s M : ℕ), M < 2 ^ 24 → 2 ^ 23 ≤ M * 2 ^ s →
    ∃ M' s', 2 ^ 23 ≤ M' ∧ M' < 2 ^ 24 ∧ M * 2 ^ s = M' * 2 ^ s'
  | s, M, h1, h2 =>
    if hM : 2 ^ 23 ≤ M then ⟨M, s, hM, h1, rfl⟩
    else
      match s with
      | 0 => absurd h2 (by simp only [pow_zero, mul_one]; omega)
      | s + 1 => by
        have e : M * 2 ^ (s + 1) = 2 * M * 2 ^ s := by ring
        obtain ⟨M', s', a, b, c⟩ := normalize24 s (2 * M) (by omega) (by omega)
        exact ⟨M', s', a, b, by rw [e, c]⟩

theorem exists_gridInt {n : ℕ} (h : Sig24 n) : ∃ j, gridInt j = n := by
  obtain ⟨M, s, hM, rfl⟩ := h
  rcases Nat.lt_or_ge (M * 2 ^ s) (2 ^ 23) with h | h
  · exact ⟨M * 2 ^ s, by rw [gridInt, ite_eq_left h]⟩
  · obtain ⟨M', s', h1, h2, e⟩ := normalize24 s M hM h
    refine ⟨(s' + 1) * 2 ^ 23 + (M' - 2 ^ 23), ?_⟩
    have d1 : ((s' + 1) * 2 ^ 23 + (M' - 2 ^ 23)) / 2 ^ 23 = s' + 1 := by omega
    have d2 : ((s' + 1) * 2 ^ 23 + (M' - 2 ^ 23)) % 2 ^ 23 = M' - 2 ^ 23 := by omega
    rw [e, gridInt, ite_eq_right (by omega), d1, d2, show s' + 1 - 1 = s' by omega, show 2 ^ 23 + (M' - 2 ^ 23) = M' by omega]

theorem onGrid_iff {x : ℚ} : OnGrid x ↔ ∃ n : ℤ, Sig24 n.natAbs ∧ x = n * 2 ^ (-149 : ℤ) := by
  constructor
  · rintro ⟨k, rfl⟩
    refine ⟨gridZ k, ?_, rfl⟩
    unfold gridZ
    split_ifs <;> simpa using sig24_gridInt k.natAbs
  · rintro ⟨n, hn, rfl⟩
    obtain ⟨j, hj⟩ := exists_gridInt hn
    refine ⟨if n < 0 then -(j : ℤ) else j, ?_⟩
    have e : gridZ (if n < 0 then -(j : ℤ) else j) = n := by
      unfold gridZ
      by_cases h : n < 0
      · have hj0 : 0 < j := by
          rcases Nat.eq_zero_or_pos j with h0 | h0
          · rw [h0, gridInt_zero] at hj; omega
          · exact h0
        rw [ite_eq_left h, ite_eq_left (by omega), Int.natAbs_neg, Int.natAbs_natCast, hj]
        omega
      · rw [ite_eq_right h, ite_eq_right (by omega), Int.natAbs_natCast, hj]
        omega
    rw [gridQ, e]

/-- An FP32 value has a 24-bit significand. -/
theorem onGrid_representable {x : ℚ} (h : OnGrid x) : Representable x := by
  obtain ⟨n, ⟨M, s, hM, hn⟩, rfl⟩ := onGrid_iff.1 h
  have hn' : (n : ℚ) = (if n < 0 then -(M : ℤ) else (M : ℤ) : ℤ) * (2 : ℚ) ^ (s : ℤ) := by
    rw [zpow_natCast]
    split_ifs with hneg
    · have : n = -((M * 2 ^ s : ℕ) : ℤ) := by omega
      rw [this]; push_cast; ring
    · have : n = ((M * 2 ^ s : ℕ) : ℤ) := by omega
      rw [this]; push_cast; ring
  refine ⟨if n < 0 then -(M : ℤ) else M, (s : ℤ) - 149, ?_, ?_⟩
  · split_ifs <;> simp only [abs_neg, Nat.abs_cast] <;> exact_mod_cast hM
  · rw [hn', mul_assoc, ← zpow_add₀ two_ne_zero]
    congr 2

/-! ## The FP32 add stays on the grid -/

/-- A multiple of `2^-149` rounded to 24 bits is an FP32 value: exact below `2^24·2^-149`, and above it the rounding unit
is at least `2^-148`. -/
theorem rn_onGrid (n : ℤ) : OnGrid (rn (n * 2 ^ (-149 : ℤ))) := by
  rcases lt_or_ge |n| (2 ^ 24) with h | h
  · rw [rn_exact ⟨n, -149, h, rfl⟩]
    have hn : (n.natAbs : ℤ) < 2 ^ 24 := by rw [Int.natCast_natAbs]; exact h
    exact onGrid_iff.2 ⟨n, ⟨n.natAbs, 0, by omega, by simp⟩, rfl⟩
  set x : ℚ := (n : ℚ) * 2 ^ (-149 : ℤ) with hxdef
  have habs : |x| = (|n| : ℚ) * 2 ^ (-149 : ℤ) := by
    rw [hxdef, abs_mul, abs_of_pos (two_zpow_pos _)]
  have hax : (2 : ℚ) ^ (-125 : ℤ) ≤ |x| := by
    rw [habs, show (-125 : ℤ) = 24 + -149 by norm_num, zpow_add₀ two_ne_zero]
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast h) (two_zpow_pos _).le
  have hx : x ≠ 0 := by
    intro e; rw [e, abs_zero] at hax; linarith [two_zpow_pos (-125 : ℤ)]
  obtain ⟨_, hb2⟩ := binade_spec (abs_pos.2 hx)
  have hb : -126 < binade |x| := by
    have := (zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 (lt_of_le_of_lt hax hb2)
    omega
  obtain ⟨hs1, hs2⟩ := sig_bounds hx
  set R := roundEven (|x| / ulp x) with hRdef
  have hR1 : (2 ^ 23 : ℤ) ≤ R := le_trans (Int.le_floor.2 (by exact_mod_cast hs1)) (floor_le_roundEven _)
  have hR2 : R ≤ 2 ^ 24 := by
    have hf : ⌊|x| / ulp x⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hs2)
    have := roundEven_le (|x| / ulp x)
    omega
  set t : ℕ := (binade |x| + 126).toNat with htdef
  have hulp : ulp x = ((2 ^ t : ℕ) : ℚ) * 2 ^ (-149 : ℤ) := by
    unfold ulp
    rw [Nat.cast_pow, Nat.cast_ofNat, ← zpow_natCast, ← zpow_add₀ two_ne_zero]
    congr 1
    omega
  have hrn : rn x = (((if x < 0 then -1 else 1) * R * ((2 ^ t : ℕ) : ℤ) : ℤ) : ℚ) * 2 ^ (-149 : ℤ) := by
    unfold rn
    rw [← hRdef, hulp]
    split_ifs <;> push_cast <;> ring
  rw [hrn]
  refine onGrid_iff.2 ⟨_, ?_, rfl⟩
  have hn : ((if x < 0 then -1 else 1) * R * ((2 ^ t : ℕ) : ℤ)).natAbs = R.natAbs * 2 ^ t := by
    rw [Int.natAbs_mul, Int.natAbs_mul, Int.natAbs_natCast]
    split_ifs <;> simp
  rw [hn]
  rcases hR2.lt_or_eq with hlt | heq
  · exact ⟨R.natAbs, t, by omega, rfl⟩
  · refine ⟨2 ^ 23, t + 1, by norm_num, ?_⟩
    rw [heq, pow_succ]
    norm_num
    ring

/-- **The FP32 add of two FP32 values is one.** -/
theorem rnAdd_onGrid {a b : ℚ} (ha : OnGrid a) (hb : OnGrid b) : OnGrid (rnAdd a b) := by
  obtain ⟨na, -, rfl⟩ := onGrid_iff.1 ha
  obtain ⟨nb, -, rfl⟩ := onGrid_iff.1 hb
  unfold rnAdd
  rw [← add_mul, ← Int.cast_add]
  exact rn_onGrid _

/-! ## Words -/

/-- The index of an FP32 word's value: its sign-magnitude encoding read as an integer, `±(w mod 2^31)`. -/
def wordIdx (w : ℕ) : ℤ := if (w >>> 31) % 2 = 1 then -((w % 2 ^ 31 : ℕ) : ℤ) else ((w % 2 ^ 31 : ℕ) : ℤ)

theorem lor_two_pow_23 {m : ℕ} (hm : m < 2 ^ 23) : (1 <<< 23 ||| m) = 2 ^ 23 + m := by
  rw [Nat.shiftLeft_eq, one_mul]
  have := Nat.two_pow_add_eq_or_of_lt hm 1
  rw [mul_one] at this
  exact this.symm

theorem gridZ_wordIdx (w : ℕ) :
    gridZ (wordIdx w) = (if (w >>> 31) % 2 = 1 then -1 else 1) * (gridInt (w % 2 ^ 31) : ℤ) := by
  unfold wordIdx gridZ
  generalize w % 2 ^ 31 = j
  by_cases hs : (w >>> 31) % 2 = 1
  · rw [ite_eq_left hs, ite_eq_left hs]
    rcases Nat.eq_zero_or_pos j with h0 | h0
    · subst h0; simp [gridInt]
    · rw [ite_eq_left (by omega), Int.natAbs_neg, Int.natAbs_natCast]
      ring
  · rw [ite_eq_right hs, ite_eq_right hs, ite_eq_right (by omega), Int.natAbs_natCast]
    ring

/-- **A finite word's value is `gridQ` of its index.** -/
theorem wordQ_eq_gridQ {w : ℕ} {x : ℚ} (h : wordQ w = some x) : x = gridQ (wordIdx w) := by
  unfold wordQ fp32Value at h
  dsimp only at h
  have hsh23 : w >>> 23 = w / 2 ^ 23 := Nat.shiftRight_eq_div_pow w 23
  have hj : w % 2 ^ 31 = (w / 2 ^ 23 % 256) * 2 ^ 23 + w % 2 ^ 23 := by omega
  rw [hsh23] at h
  rw [gridQ, gridZ_wordIdx]
  split_ifs at h with h1 h2 h3 h4 <;>
    simp only [Value.finite?, Option.map_some, Option.map_none, reduceCtorEq, Option.some.injEq] at h <;> subst h
  · -- zero
    have hz : w % 2 ^ 31 = 0 := by omega
    rw [hz, gridInt_zero]
    simp [Term.value, Term.zero]
  · -- subnormal
    have hg : gridInt (w % 2 ^ 31) = w % 2 ^ 23 := by
      rw [show w % 2 ^ 31 = w % 2 ^ 23 by omega, gridInt, ite_eq_left (Nat.mod_lt _ (by norm_num))]
    rw [hg]
    unfold Term.value
    dsimp only
    rw [show ((-126 : ℤ) - 23) = -149 by norm_num]
    generalize w % 2 ^ 23 = m
    by_cases hs : (w >>> 31) % 2 = 1 <;> simp [hs]
  · -- normal
    have hg : gridInt (w % 2 ^ 31) = (2 ^ 23 + w % 2 ^ 23) * 2 ^ (w / 2 ^ 23 % 256 - 1) := by
      rw [gridInt, ite_eq_right (by omega)]
      congr 2 <;> omega
    rw [hg, lor_two_pow_23 (Nat.mod_lt _ (by norm_num))]
    unfold Term.value
    dsimp only
    have he : 1 ≤ w / 2 ^ 23 % 256 := by omega
    generalize w % 2 ^ 23 = m
    generalize w / 2 ^ 23 % 256 = E at he
    have hpow : ((2 : ℚ) ^ ((E : ℤ) - 127 - 23)) = ((2 ^ (E - 1) : ℕ) : ℚ) * 2 ^ (-149 : ℤ) := by
      rw [Nat.cast_pow, Nat.cast_ofNat, ← zpow_natCast, ← zpow_add₀ two_ne_zero]
      congr 1
      omega
    rw [hpow]
    by_cases hs : (w >>> 31) % 2 = 1 <;> simp [hs] <;> ring

theorem wordQ_onGrid {w : ℕ} {x : ℚ} (h : wordQ w = some x) : OnGrid x := ⟨_, (wordQ_eq_gridQ h).symm⟩

/-- `_Cmp.pred` on a 32-bit word: `0x80000001` (`−2^-149`) at `±0`, else one step on the encoding, away from zero when
negative and toward it when positive. -/
def predWord (w : ℕ) : ℕ :=
  if w % 2 ^ 31 = 0 then 0x80000001 else if (w >>> 31) % 2 = 1 then (w + 1) % 2 ^ 32 else (w + 0xFFFFFFFF) % 2 ^ 32

/-- `_Cmp.succ` on a 32-bit word: `0x00000001` (`2^-149`) at `±0`, else one step on the encoding, toward zero when
negative and away from it when positive. -/
def succWord (w : ℕ) : ℕ :=
  if w % 2 ^ 31 = 0 then 0x00000001 else if (w >>> 31) % 2 = 1 then (w + 0xFFFFFFFF) % 2 ^ 32 else (w + 1) % 2 ^ 32

/-- **The circuit's `pred` is one index down** on a word whose magnitude is not the top encoding. -/
theorem wordIdx_predWord {w : ℕ} (hn : w % 2 ^ 31 + 1 < 2 ^ 31) :
    wordIdx (predWord w) = wordIdx w - 1 := by
  unfold predWord wordIdx
  simp only [Nat.shiftRight_eq_div_pow]
  split_ifs <;> omega

/-- **The circuit's `succ` is one index up** on a word whose magnitude is not the top encoding. -/
theorem wordIdx_succWord {w : ℕ} (hn : w % 2 ^ 31 + 1 < 2 ^ 31) :
    wordIdx (succWord w) = wordIdx w + 1 := by
  unfold succWord wordIdx
  simp only [Nat.shiftRight_eq_div_pow]
  split_ifs <;> first | omega | (exfalso; assumption)

/-- The index on four encodings: `0x00000001` is `2^-149`, `0x00800000` the least normal `2^-126`, `0x3F800000` is `1`, and
`−0` (`0x80000000`) is `+0`'s index. -/
theorem wordIdx_examples :
    gridQ (wordIdx 0x00000001) = 2 ^ (-149 : ℤ) ∧ gridQ (wordIdx 0x00800000) = 2 ^ (-126 : ℤ) ∧
      gridQ (wordIdx 0x3F800000) = 1 ∧ wordIdx 0x80000000 = 0 ∧ gridQ (wordIdx 0xBF800000) = -1 := by
  refine ⟨?_, ?_, ?_, by decide, ?_⟩ <;> norm_num [gridQ, gridZ, gridInt, wordIdx]

/-! ## Promoted totals stay on the grid -/

theorem gridOpt_some {y : ℚ} (h : OnGrid y) : GridOpt (some y) := by
  rintro _ ⟨⟩; exact h

theorem gridOpt_promote {t : Option ℚ} (ht : GridOpt t) (w : ℕ) : GridOpt (promote t w) := by
  intro y hy
  cases t with
  | none => simp [promote] at hy
  | some t' =>
    cases hx : wordQ w with
    | none => simp [promote, hx] at hy
    | some x =>
      simp only [promote, hx, Option.some.injEq] at hy
      subst hy
      exact rnAdd_onGrid (ht t' rfl) (wordQ_onGrid hx)

theorem gridOpt_foldl_promote (ws : List ℕ) {t : Option ℚ} (ht : GridOpt t) : GridOpt (ws.foldl promote t) := by
  induction ws generalizing t with
  | nil => exact ht
  | cons w ws ih => exact ih (gridOpt_promote ht w)

theorem promote_none (w : ℕ) : promote none w = none := rfl

theorem foldl_promote_none (ws : List ℕ) : ws.foldl promote none = none := by
  induction ws with
  | nil => rfl
  | cons w ws ih => exact ih

end Pouw.PearlC.Cap
