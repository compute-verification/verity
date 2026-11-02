import Mathlib.Tactic.FinCases
import Proofs.Pouw.Fp8Atom.H1TSliceProofs
import Proofs.Pouw.Fp8Atom.H1TProofs

/-!
# The one-flip clauses (`H1TOneFlip`, M4a and M4b)

The involution toggles one raw bit: bit 31 of `r17`, bit 15 of `r16` or bit 31 of `r16`, which `extract` reads as tag
lane `29 + k`'s sign bit and nothing else (`extract_rawFlip`). On the slice salts it is `flipS τ k`, and by
`H1TSlice` it moves the slice's block-1 sum by `−2·v·t` and its block-2 sum by `+2·v·t`, with `|v| ≥ 32F`.

* A flip of lane 0 moves the flipped fresh-step word by at least `4096F − 256F ≥ 960` (the cell of each sum is at
  most `128F`, `t₀ ≥ 64`), in the direction of `∓v` (`fresh_move`), and so a block-1 running word by at least
  `960 − 512 = 448` (L4: the ulp is at most 512; `run_move`).
* `I` against `I`: other slices (flip one), the two blocks of one slice (they move in opposite directions), and two
  columns in one block (a flip of a lane where their tuples differ moves the difference of the sums by
  `2·|v|·|Δt| ≥ 256F`, and equal truncations lie within `128F`).
* A block-1 `R_q` against `I_p`: if `I_p` is not of slice `q`, flip slice `q`. If it is, flip slice `q − 1`: then
  `I_p` and `I_q` stay, and `RN(R_(q−1) + I_q) = I_p` with `|I_p| < 2^23` puts `R_(q−1) + I_q` within `1/4` of
  `I_p` at both salts, while `R_(q−1)` moves by at least 448.
* A block-1 `R_q` against a later `R_q'` of block 1: flip slice `q'`.
* A block-2 `R_p` against a later `I_p'`: flip slice `p' − T`. The running word moves weakly one way (L7), and the
  block-2 fresh-step word strictly the other.

By column (M4b), a flip of lane `k` of slice `q`:
* Two block-1 `R_q` of columns whose tuples are at least 72 apart in lane `k` (`FarTags`): both `R_(q−1)` stay, and
  the difference of the two sums moves by `2·|v|·|Δt| ≥ 4608F`. Equal roundings put each pair of sums within 512
  (L4), and the four cells take less than `512F`, so it cannot be absorbed (`sep_run_columns`).
* A block-2 `R_p` against a block-1 word of slice `q` of a column whose tuple has the opposite sign in lane `k`
  (`OppLater`), for `p < T + q`: `R_p` reads slice `q` only through `I_q`, so it moves weakly one way (L7), and the
  other column's `I_q` strictly the other (`sep_opp_fresh`). So does its `R_q` if its value in lane `k` is at least
  36 in magnitude: `I_q` then moves by more than 512 and `R_(q−1)` stays (`sep_opp_run`).
-/

namespace Pouw.Fp8Atom.H1T.Proofs

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T

/-! ## The raw flip -/

theorem bitAt_eq (w c : ℕ) : bitAt w c = w.testBit c := by
  rw [Nat.testBit_eq_decide_div_mod_eq, bitAt, Nat.shiftRight_eq_div_pow]
  cases h : w / 2 ^ c % 2 == 1 <;> simp_all

theorem bitAt_xor (w b c : ℕ) : bitAt (w ^^^ 2 ^ b) c = (bitAt w c ^^ decide (b = c)) := by
  rw [bitAt_eq, bitAt_eq, Nat.testBit_xor, Nat.testBit_two_pow]

theorem bits3_xor (w b c : ℕ) (h : b < c ∨ c + 3 ≤ b) : bits3 (w ^^^ 2 ^ b) c = bits3 w c := by
  unfold bits3
  congr 1
  apply Nat.eq_of_testBit_eq
  intro i
  rw [show (8 : ℕ) = 2 ^ 3 from rfl, Nat.testBit_mod_two_pow, Nat.testBit_mod_two_pow, Nat.testBit_shiftRight,
    Nat.testBit_shiftRight, Nat.testBit_xor, Nat.testBit_two_pow]
  by_cases hi : i < 3
  · have : ¬ b = c + i := by omega
    simp [this]
  · simp [hi]

/-- Bit `b < 32` of a raw word toggled. -/
def flipBit (w : Fin (2 ^ 32)) (b : ℕ) (hb : b < 32) : Fin (2 ^ 32) :=
  ⟨w.val ^^^ 2 ^ b, Nat.xor_lt_two_pow w.isLt (Nat.pow_lt_pow_right (by norm_num) hb)⟩

theorem flipBit_flipBit (w : Fin (2 ^ 32)) (b : ℕ) (hb : b < 32) : flipBit (flipBit w b hb) b hb = w := by
  ext; simp [flipBit]

theorem bitAt_flipBit (w : Fin (2 ^ 32)) (b : ℕ) (hb : b < 32) (c : ℕ) :
    bitAt (flipBit w b hb).val c = (bitAt w.val c ^^ decide (b = c)) := bitAt_xor _ _ _

theorem bits3_flipBit (w : Fin (2 ^ 32)) (b : ℕ) (hb : b < 32) (c : ℕ) (h : b < c ∨ c + 3 ≤ b) :
    bits3 (flipBit w b hb).val c = bits3 w.val c := bits3_xor _ _ _ h

/-- Tag lane `29 + k`'s sign bit toggled in the raw words: bit 31 of `r17`, bit 15 of `r16` or bit 31 of `r16`. -/
def rawFlip (k : Fin 3) (r : RawSlice) : RawSlice :=
  if k.val = 0 then Function.update r 17 (flipBit (r 17) 31 (by norm_num))
  else if k.val = 1 then Function.update r 16 (flipBit (r 16) 15 (by norm_num))
  else Function.update r 16 (flipBit (r 16) 31 (by norm_num))

theorem rawFlip_rawFlip (k : Fin 3) (r : RawSlice) : rawFlip k (rawFlip k r) = r := by
  unfold rawFlip
  split_ifs <;> simp [Function.update_idem, flipBit_flipBit, Function.update_eq_self]

theorem extract_rawFlip (k : Fin 3) (r : RawSlice) : extract (rawFlip k r) = negFlip k (extract r) := by
  have hpos : ∀ l : Fin 29, (⟨l.val / 2, by have := l.isLt; omega⟩ : Fin 18) ≠ 16 ∧
      (⟨l.val / 2, by have := l.isLt; omega⟩ : Fin 18) ≠ 17 := by
    intro l; constructor <;> intro e <;> have := congrArg Fin.val e <;> simp at this <;> omega
  have e1 : (extract (rawFlip k r)).pos = (extract r).pos := by
    funext l
    simp only [extract]
    unfold rawFlip
    split_ifs <;> simp only [Function.update_of_ne (hpos l).1, Function.update_of_ne (hpos l).2]
  have e2 : (extract (rawFlip k r)).neg = Function.update (extract r).neg k (!(extract r).neg k) := by
    funext k'
    fin_cases k <;> fin_cases k' <;>
      simp [extract, rawFlip, pick3, bitAt_flipBit, Function.update_of_ne, Function.update_self]
  have e3 : (extract (rawFlip k r)).man = (extract r).man := by
    funext k'
    fin_cases k <;> fin_cases k' <;>
      simp [extract, rawFlip, pick3, Function.update_of_ne, Function.update_self, bits3_flipBit]
  unfold negFlip
  rw [← e1, ← e2, ← e3]

/-- The raw flip of slice `τ` of a row's salt. -/
def psi {T : ℕ} (τ : Fin T) (k : Fin 3) (s : Fin T → RawSlice) : Fin T → RawSlice :=
  Function.update s τ (rawFlip k (s τ))

theorem psi_invol {T : ℕ} (τ : Fin T) (k : Fin 3) : Function.Involutive (psi τ k) := by
  intro s
  simp [psi, Function.update_idem, rawFlip_rawFlip, Function.update_eq_self]

/-- Slice `τ` of `σ` with tag lane `k`'s sign bit flipped. -/
def flipS (τ : ℕ) (k : Fin 3) (σ : ℕ → SliceSalt) : ℕ → SliceSalt := Function.update σ τ (negFlip k (σ τ))

theorem flipS_self (τ : ℕ) (k : Fin 3) (σ : ℕ → SliceSalt) : flipS τ k σ τ = negFlip k (σ τ) := by
  simp [flipS]

theorem flipS_ne {τ t : ℕ} (k : Fin 3) (σ : ℕ → SliceSalt) (h : t ≠ τ) : flipS τ k σ t = σ t := by
  simp [flipS, h]

theorem rowSalt_psi (U : Unit) (i : Fin U.m) (τ : Fin U.T) (k : Fin 3) (s : Fin U.T → RawSlice) :
    U.rowSalt (fun _ => psi τ k s) i = flipS τ.val k (U.rowSalt (fun _ => s) i) := by
  funext t
  unfold Unit.rowSalt
  by_cases ht : t = τ.val
  · subst ht
    rw [flipS_self, dite_eq_left τ.isLt, dite_eq_left τ.isLt]
    simp [psi, extract_rawFlip]
  · rw [flipS_ne _ _ ht]
    split_ifs with h
    · have : (⟨t, h⟩ : Fin U.T) ≠ τ := fun e => ht (congrArg Fin.val e)
      simp [psi, Function.update_of_ne this]
    · rfl

/-! ## The block sums under a flip -/

/-- The sum `trunc14` truncates at position `p`: block 1 of slice `p`, or block 2 of slice `p − T`. -/
def bsum (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) : ℚ :=
  if p < T then preSum (formSlice (realLanes x p) (σ p)).1 (colLanes b j p)
  else preSum (formSlice (realLanes x (p - T)) (σ (p - T))).2 (colLanes b j (p - T))

/-- The slice position `p` reads. -/
def slc (T p : ℕ) : ℕ := if p < T then p else p - T

/-- Slice `τ`'s floor `F`. -/
def fl (x : ℕ → Code) (τ : ℕ) : ℚ := floorF (sliceMax (realLanes x τ))

/-- Tag lane `k`'s value in slice `τ`. -/
def tv (x : ℕ → Code) (σ : ℕ → SliceSalt) (τ : ℕ) (k : Fin 3) : ℚ := tagVal (fl x τ) ((σ τ).neg k) ((σ τ).man k)

theorem bsum_flip_ne {T : ℕ} {x b : ℕ → Code} (j : ℕ) (σ : ℕ → SliceSalt) {τ p : ℕ} (k : Fin 3)
    (h : slc T p ≠ τ) : bsum T x b j (flipS τ k σ) p = bsum T x b j σ p := by
  unfold bsum slc at *
  split_ifs at * with hp <;> rw [flipS_ne _ _ h]

theorem fresh_fixed {T : ℕ} {x b : ℕ → Code} (j : ℕ) (σ : ℕ → SliceSalt) {τ p : ℕ} (k : Fin 3)
    (h : slc T p ≠ τ) : freshVal T x b j (flipS τ k σ) p = freshVal T x b j σ p := by
  unfold freshVal freshWord
  unfold slc at h
  split_ifs at * with hp <;> rw [flipS_ne _ _ h]

theorem slc_lt {T p : ℕ} (hp : p < 2 * T) : slc T p < T := by unfold slc; split_ifs <;> omega

theorem fl_index {x : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (τ : ℕ) : ∃ fi : ℕ, fi ≤ 4 ∧ fl x τ = fOf fi := by
  obtain ⟨fi, hfi, hF, -⟩ := floorF_index (realLanes x τ) (fun l => hx _)
  exact ⟨fi, hfi, hF⟩

section
variable {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
include hx hb

theorem freshVal_bsum (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) : freshVal T x b j σ p = trunc14 (bsum T x b j σ p) :=
  freshVal_value hx hb j σ p

theorem bsum_flip_one (j : ℕ) (σ : ℕ → SliceSalt) {τ : ℕ} (hτ : τ < T) (k : Fin 3) :
    bsum T x b j (flipS τ k σ) τ = bsum T x b j σ τ - 2 * tv x σ τ k * val (tuple j k) := by
  unfold bsum; rw [ite_eq_left hτ, ite_eq_left hτ, flipS_self]
  exact ((h1tSlice _ (σ τ) b j τ (fun l => hx _) hb).2.2 k).2.2.1

theorem bsum_flip_two (j : ℕ) (σ : ℕ → SliceSalt) {τ : ℕ} (_hτ : τ < T) (k : Fin 3) :
    bsum T x b j (flipS τ k σ) (T + τ) = bsum T x b j σ (T + τ) + 2 * tv x σ τ k * val (tuple j k) := by
  unfold bsum; rw [ite_eq_right (by omega), ite_eq_right (by omega), Nat.add_sub_cancel_left, flipS_self]
  exact ((h1tSlice _ (σ τ) b j τ (fun l => hx _) hb).2.2 k).2.2.2

theorem bsum_lt (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) {fi : ℕ} (hF : fl x (slc T p) = fOf fi) :
    |bsum T x b j σ p| < 2 ^ 21 * fOf fi := by
  have hS := h1tSlice (realLanes x (slc T p)) (σ (slc T p)) b j (slc T p) (fun l => hx _) hb
  unfold fl at hF
  rw [hF] at hS
  by_cases hp : p < T
  · have e : slc T p = p := ite_eq_left hp
    rw [e] at hS
    unfold bsum; rw [ite_eq_left hp]; exact hS.1.2
  · have e : slc T p = p - T := ite_eq_right hp
    rw [e] at hS
    unfold bsum; rw [ite_eq_right hp]; exact hS.2.1.2

end

/-! ## Cells and moves -/

theorem fOf_ge (fi : ℕ) : 1 / 4 ≤ fOf fi := by
  rw [fOf_eq, show (1 / 4 : ℚ) = 2 ^ (-2 : ℤ) by norm_num]
  exact zpow_le_zpow_right₀ (by norm_num) (by omega)

theorem cell_le {fi : ℕ} {y : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) (h0 : y ≠ 0) :
    (2 : ℚ) ^ (binade |y| - 13) ≤ 128 * fOf fi := by
  have hb := (binade_spec (abs_pos.2 h0)).1
  have e : (2 : ℚ) ^ 21 * fOf fi = 2 ^ ((fi : ℤ) + 19) := by
    rw [fOf_eq, ← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
  have e2 : 128 * fOf fi = (2 : ℚ) ^ ((fi : ℤ) + 5) := by
    rw [fOf_eq, show (128 : ℚ) = 2 ^ (7 : ℤ) by norm_num, ← zpow_add₀ two_ne_zero]; congr 1; ring
  rw [e] at hy
  have := (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).1 (lt_of_le_of_lt hb hy)
  rw [e2]; exact zpow_le_zpow_right₀ (by norm_num) (by omega)

theorem trunc14_near {fi : ℕ} {y : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) : |trunc14 y - y| ≤ 128 * fOf fi := by
  rcases eq_or_ne y 0 with rfl | h0
  · rw [trunc14_zero, sub_zero, abs_zero]; have := fOf_pos fi; linarith
  · exact le_trans (trunc14Cell y y h0).1.le (cell_le hy h0)

theorem trunc14_eq_near {fi : ℕ} {y z : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) (h : trunc14 y = trunc14 z) :
    |y - z| < 128 * fOf fi := by
  rcases eq_or_ne y 0 with rfl | h0
  · rw [trunc14_zero] at h
    have := trunc14_eq_zero h.symm
    subst this; rw [sub_zero, abs_zero]; have := fOf_pos fi; linarith
  · exact lt_of_lt_of_le ((trunc14Cell y z h0).2 h) (cell_le hy h0)

theorem trunc14_move {fi : ℕ} {y y' : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) (hy' : |y'| < 2 ^ 21 * fOf fi) :
    |y' - y| - 256 * fOf fi ≤ |trunc14 y' - trunc14 y| := by
  have h1 := trunc14_near hy
  have h2 := trunc14_near hy'
  have e : y' - y = (y' - trunc14 y') + (trunc14 y' - trunc14 y) + (trunc14 y - y) := by ring
  have a1 := abs_add_le ((y' - trunc14 y') + (trunc14 y' - trunc14 y)) (trunc14 y - y)
  have a2 := abs_add_le (y' - trunc14 y') (trunc14 y' - trunc14 y)
  rw [abs_sub_comm] at h2
  rw [e]; linarith

theorem rn_sep {y y' : ℚ} (hy : |y| < 2 ^ 33) (hy' : |y'| < 2 ^ 33) : |y' - y| - 512 ≤ |rn y' - rn y| := by
  have h1 := rn_err_small hy
  have h2 := rn_err_small hy'
  have e : y' - y = (y' - rn y') + (rn y' - rn y) + (rn y - y) := by ring
  have a1 := abs_add_le ((y' - rn y') + (rn y' - rn y)) (rn y - y)
  have a2 := abs_add_le (y' - rn y') (rn y' - rn y)
  rw [abs_sub_comm] at h2
  rw [e]; linarith

/-- A value that rounds to below `2^23` in magnitude is within a quarter of it. -/
theorem rn_small {y w : ℚ} (h : rn y = w) (hw : |w| < 2 ^ 23) : |y - w| ≤ 1 / 4 := by
  have hr : Representable ((2 : ℚ) ^ 23) := ⟨2 ^ 23, 0, by norm_num, by norm_num⟩
  have e23 : rn ((2 : ℚ) ^ 23) = 2 ^ 23 := rn_exact hr
  have en : rn (-(2 : ℚ) ^ 23) = -2 ^ 23 := rn_exact (representable_neg hr)
  obtain ⟨w1, w2⟩ := abs_lt.1 hw
  have hy : |y| < 2 ^ 23 := by
    rw [abs_lt]; constructor
    · by_contra hc
      have := rn_mono (show y ≤ -(2 : ℚ) ^ 23 by linarith)
      rw [en, h] at this; linarith
    · by_contra hc
      have := rn_mono (show (2 : ℚ) ^ 23 ≤ y by linarith)
      rw [e23, h] at this; linarith
  rcases eq_or_ne y 0 with rfl | hy0
  · have : rn 0 = 0 := rn_exact ⟨0, 0, by norm_num, by simp⟩
    rw [this] at h; subst h; norm_num
  · have hb := (binade_spec (abs_pos.2 hy0)).1
    have e : (2 : ℚ) ^ (23 : ℕ) = 2 ^ (23 : ℤ) := (zpow_natCast 2 23).symm
    rw [e] at hy
    have h23 : binade |y| < 23 :=
      (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).1 (lt_of_le_of_lt hb hy)
    have hu : ulp y ≤ 1 / 2 := by
      unfold ulp
      calc (2 : ℚ) ^ (binade |y| - 23) ≤ 2 ^ (-1 : ℤ) := zpow_le_zpow_right₀ (by norm_num) (by omega)
        _ = 1 / 2 := by norm_num
    have := rn_err y
    rw [h, abs_sub_comm] at this
    linarith

theorem tagVal_abs_ge (F : ℚ) (hF : 0 < F) (neg : Bool) (m : Fin 8) : 32 * F ≤ |tagVal F neg m| := by
  have hm0 : (0 : ℚ) ≤ m := Nat.cast_nonneg _
  unfold tagVal
  rw [abs_mul, abs_mul, show |(if neg then (-1 : ℚ) else 1)| = 1 by split_ifs <;> simp, one_mul,
    abs_of_pos (by positivity), abs_of_pos (by positivity)]
  nlinarith

theorem A1_abs_ge : ∀ i : Fin 62, 32 * 512 ≤ |intVal (A1 i)| := by decide +kernel

/-- Every tag lane's tuple value is at least 32 in magnitude (lane 0's is in `[64, 448]`, lanes 1 and 2's in
`±[32, 448]`). -/
theorem tuple_abs_ge (j : ℕ) (k : Fin 3) : 32 ≤ |val (tuple j k)| := by
  rw [val_eq_intVal (tuple_facts j k).1, abs_div, abs_of_pos (by norm_num : (0 : ℚ) < 512), le_div_iff₀ (by norm_num),
    ← Int.cast_abs]
  have : 32 * 512 ≤ |intVal (tuple j k)| := by
    unfold tuple pick3
    split_ifs
    · exact le_trans (by have := (A0_facts ⟨j % 23, Nat.mod_lt _ (by norm_num)⟩).2.2; omega) (le_abs_self _)
    · exact A1_abs_ge _
    · exact A1_abs_ge _
  exact_mod_cast this

theorem trunc14_near_lt {fi : ℕ} {y : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) : |trunc14 y - y| < 128 * fOf fi := by
  rcases eq_or_ne y 0 with rfl | h0
  · rw [trunc14_zero, sub_zero, abs_zero]; have := fOf_pos fi; linarith
  · exact lt_of_lt_of_le (trunc14Cell y y h0).1 (cell_le hy h0)

theorem trunc14_move_lt {fi : ℕ} {y y' : ℚ} (hy : |y| < 2 ^ 21 * fOf fi) (hy' : |y'| < 2 ^ 21 * fOf fi) :
    |y' - y| - 256 * fOf fi < |trunc14 y' - trunc14 y| := by
  have h1 := trunc14_near_lt hy
  have h2 := trunc14_near_lt hy'
  have e : y' - y = (y' - trunc14 y') + (trunc14 y' - trunc14 y) + (trunc14 y - y) := by ring
  have a1 := abs_add_le ((y' - trunc14 y') + (trunc14 y' - trunc14 y)) (trunc14 y - y)
  have a2 := abs_add_le (y' - trunc14 y') (trunc14 y' - trunc14 y)
  rw [abs_sub_comm] at h2
  rw [e]; linarith

theorem tv_ne_zero {x : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (σ : ℕ → SliceSalt) (τ : ℕ) (k : Fin 3) :
    tv x σ τ k ≠ 0 := by
  obtain ⟨fi, -, hF⟩ := fl_index hx τ
  have hv := tagVal_abs_ge (fl x τ) (by rw [hF]; exact fOf_pos _) ((σ τ).neg k) ((σ τ).man k)
  intro h0; unfold tv at h0; rw [h0, abs_zero, hF] at hv; linarith [fOf_pos fi]

/-- A flip of lane `k` moves a sum by `2·v·w` with `|v| ≥ 32F`: at least `64F·c` if `|w| ≥ c`. -/
theorem tv_move_ge {x : ℕ → Code} {σ : ℕ → SliceSalt} {τ fi : ℕ} (hF : fl x τ = fOf fi) (k : Fin 3) {c w : ℚ}
    (hc : 0 ≤ c) (hw : c ≤ |w|) : 64 * fOf fi * c ≤ |2 * tv x σ τ k * w| := by
  have hv := tagVal_abs_ge (fl x τ) (by rw [hF]; exact fOf_pos _) ((σ τ).neg k) ((σ τ).man k)
  rw [hF] at hv
  rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℚ) < 2)]
  unfold tv; rw [hF]
  have := mul_le_mul hv hw hc (abs_nonneg _)
  linarith

theorem tv_opp {x : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (σ : ℕ → SliceSalt) (τ : ℕ) (k : Fin 3) {t t' : ℚ}
    (h : t * t' < 0) : tv x σ τ k * t * (tv x σ τ k * t') < 0 := by
  have h2 := mul_self_pos.2 (tv_ne_zero hx σ τ k)
  calc tv x σ τ k * t * (tv x σ τ k * t') = (tv x σ τ k * tv x σ τ k) * (t * t') := by ring
    _ < 0 := mul_neg_of_pos_of_neg h2 h

/-! ## A flip of lane 0 moves the flipped words -/

/-- A flip of lane 0 of slice `slc T p` moves `I_p` by at least 960: strictly down in block 1 and up in block 2 when
the lane's tag value `v` is positive, and the other way when it is negative. -/
theorem fresh_move {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) {p : ℕ} (hp : p < 2 * T) :
    960 ≤ |freshVal T x b j (flipS (slc T p) 0 σ) p - freshVal T x b j σ p| ∧
    (0 < tv x σ (slc T p) 0 → (p < T ↔ freshVal T x b j (flipS (slc T p) 0 σ) p < freshVal T x b j σ p)) ∧
    (tv x σ (slc T p) 0 < 0 → (p < T ↔ freshVal T x b j σ p < freshVal T x b j (flipS (slc T p) 0 σ) p)) := by
  obtain ⟨fi, -, hF⟩ := fl_index hx (slc T p)
  have hy := bsum_lt hx hb j σ p hF
  have hy' := bsum_lt hx hb j (flipS (slc T p) 0 σ) p hF
  have hv := tagVal_abs_ge (fl x (slc T p)) (by rw [hF]; exact fOf_pos _) ((σ (slc T p)).neg 0) ((σ (slc T p)).man 0)
  rw [hF] at hv
  have ht := tuple0_ge j
  have hF4 := fOf_ge fi
  have hd : bsum T x b j (flipS (slc T p) 0 σ) p =
      bsum T x b j σ p - (if p < T then 1 else -1) * (2 * tv x σ (slc T p) 0 * val (tuple j 0)) := by
    by_cases h : p < T
    · have e : slc T p = p := ite_eq_left h
      rw [ite_eq_left h, one_mul, e]
      exact bsum_flip_one hx hb j σ h 0
    · have e : slc T p = p - T := ite_eq_right h
      rw [ite_eq_right h, e]
      have := bsum_flip_two hx hb (T := T) j σ (τ := p - T) (by omega) 0
      rw [show T + (p - T) = p by omega] at this
      rw [this]; ring
  have habs : 4096 * fOf fi ≤ |2 * tv x σ (slc T p) 0 * val (tuple j 0)| := by
    rw [abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℚ) < 2),
      abs_of_pos (by linarith : (0 : ℚ) < val (tuple j 0))]
    unfold tv; rw [hF]; nlinarith [fOf_pos fi]
  have hmove := trunc14_move hy hy'
  rw [freshVal_bsum hx hb, freshVal_bsum hx hb]
  have h960 : 960 ≤ |trunc14 (bsum T x b j (flipS (slc T p) 0 σ) p) - trunc14 (bsum T x b j σ p)| := by
    have : |bsum T x b j (flipS (slc T p) 0 σ) p - bsum T x b j σ p| =
        |2 * tv x σ (slc T p) 0 * val (tuple j 0)| := by
      rw [hd]; split_ifs <;> simp [abs_neg]
    rw [this] at hmove; nlinarith
  have hne : trunc14 (bsum T x b j (flipS (slc T p) 0 σ) p) ≠ trunc14 (bsum T x b j σ p) := by
    intro e; rw [e, sub_self, abs_zero] at h960; norm_num at h960
  set d := 2 * tv x σ (slc T p) 0 * val (tuple j 0) with hddef
  set y := bsum T x b j σ p
  set y' := bsum T x b j (flipS (slc T p) 0 σ) p
  refine ⟨h960, fun hvp => ?_, fun hvn => ?_⟩
  · have hdp : 0 < d := by rw [hddef]; have : 0 < val (tuple j 0) := by linarith
                           positivity
    constructor
    · intro h
      rw [ite_eq_left h, one_mul] at hd
      exact lt_of_le_of_ne (trunc14Mono (by linarith)) hne
    · intro hlt; by_contra h
      rw [ite_eq_right h] at hd
      have := trunc14Mono (show y ≤ y' by linarith)
      linarith
  · have hdn : d < 0 := by
      rw [hddef]; have : 0 < val (tuple j 0) := by linarith
      nlinarith
    constructor
    · intro h
      rw [ite_eq_left h, one_mul] at hd
      exact lt_of_le_of_ne (trunc14Mono (by linarith)) (Ne.symm hne)
    · intro hlt; by_contra h
      rw [ite_eq_right h] at hd
      have := trunc14Mono (show y' ≤ y by linarith)
      linarith

theorem run_congr {T : ℕ} {x b : ℕ → Code} {j : ℕ} {σ σ' : ℕ → SliceSalt} {q : ℕ}
    (h : ∀ t ≤ q, freshVal T x b j σ' t = freshVal T x b j σ t) : runVal T x b j σ' q = runVal T x b j σ q :=
  le_antisymm (h1tChainMono T x b j σ' σ q fun t ht => (h t ht).le)
    (h1tChainMono T x b j σ σ' q fun t ht => (h t ht).ge)

theorem run_fixed {T : ℕ} {x b : ℕ → Code} (j : ℕ) (σ : ℕ → SliceSalt) {τ q : ℕ} (k : Fin 3) (hq : q < τ)
    (hτ : τ < T) : runVal T x b j (flipS τ k σ) q = runVal T x b j σ q :=
  run_congr fun t ht => fresh_fixed (b := b) j σ k (by unfold slc; split_ifs <;> omega)

theorem run1_sum_lt {T : ℕ} {x b : ℕ → Code} (hT : T ≤ 1130) (hx : ∀ r, (x r).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (j : ℕ) (σ : ℕ → SliceSalt) {p : ℕ} (hp : p + 1 < T) :
    |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 := by
  have hR := run_bound hT (fun t => (fresh_facts hx hb j σ t).2) p (by omega)
  rw [ite_eq_left (show p < T by omega)] at hR
  have hI := (fresh_facts (T := T) hx hb j σ (p + 1)).2
  rw [ite_eq_left hp] at hI
  have hq : (p : ℚ) + 2 ≤ 1130 := by exact_mod_cast (show p + 2 ≤ 1130 by omega)
  have := abs_add_le (runVal T x b j σ p) (freshVal T x b j σ (p + 1))
  norm_num; linarith

theorem fresh_lt {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) : |freshVal T x b j σ p| < 2 ^ 23 := by
  have := (fresh_facts (T := T) hx hb j σ p).2
  split_ifs at this <;> norm_num <;> linarith

/-- A flip of lane 0 of slice `q < T` moves the block-1 running word `R_q` by at least 448. -/
theorem run_move {T : ℕ} {x b : ℕ → Code} (hT : T ≤ 1130) (hx : ∀ r, (x r).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (j : ℕ) (σ : ℕ → SliceSalt) {q : ℕ} (hq : q < T) :
    448 ≤ |runVal T x b j (flipS q 0 σ) q - runVal T x b j σ q| := by
  have hs : slc T q = q := ite_eq_left hq
  have hm := (fresh_move hx hb j σ (p := q) (show q < 2 * T by omega)).1
  rw [hs] at hm
  cases q with
  | zero => show 448 ≤ |freshVal T x b j (flipS 0 0 σ) 0 - freshVal T x b j σ 0|; linarith
  | succ q =>
    have hfix := run_fixed (x := x) (b := b) j σ 0 (Nat.lt_succ_self q) hq
    show 448 ≤ |rnAdd (runVal T x b j (flipS (q + 1) 0 σ) q) (freshVal T x b j (flipS (q + 1) 0 σ) (q + 1)) -
      rnAdd (runVal T x b j σ q) (freshVal T x b j σ (q + 1))|
    rw [hfix]
    have h1 := run1_sum_lt hT hx hb j σ hq
    have h2 := run1_sum_lt hT hx hb j (flipS (q + 1) 0 σ) hq
    rw [hfix] at h2
    have := rn_sep h1 h2
    unfold rnAdd
    rw [show runVal T x b j σ q + freshVal T x b j (flipS (q + 1) 0 σ) (q + 1) -
      (runVal T x b j σ q + freshVal T x b j σ (q + 1)) =
      freshVal T x b j (flipS (q + 1) 0 σ) (q + 1) - freshVal T x b j σ (q + 1) by ring] at this
    linarith

/-! ## The pairs one flip separates -/

/-- A credited word of one row and column as a function of the row's slice salts. -/
def W (T : ℕ) (x b : ℕ → Code) (j : ℕ) : Pos T → (ℕ → SliceSalt) → ℚ
  | .inl p, σ => freshVal T x b j σ p.val
  | .inr p, σ => runVal T x b j σ p.val.val

/-- The flip of lane `k` of slice `τ` separates `A` and `B` wherever they agree. -/
def Sep (τ : ℕ) (k : Fin 3) (A B : (ℕ → SliceSalt) → ℚ) : Prop :=
  ∀ σ, A σ = B σ → A (flipS τ k σ) ≠ B (flipS τ k σ)

theorem Sep.symm {τ : ℕ} {k : Fin 3} {A B : (ℕ → SliceSalt) → ℚ} (h : Sep τ k A B) : Sep τ k B A :=
  fun σ e e' => h σ e.symm e'.symm

/-- If a flip moves `A` weakly and `B` strictly, in directions of opposite signs `u` and `u'`, it separates them. -/
theorem sep_of_opposite {τ : ℕ} {k : Fin 3} {A B u u' : (ℕ → SliceSalt) → ℚ} (hu : ∀ σ, u σ * u' σ < 0)
    (hA : ∀ σ, (0 ≤ u σ → A (flipS τ k σ) ≤ A σ) ∧ (u σ ≤ 0 → A σ ≤ A (flipS τ k σ)))
    (hB : ∀ σ, (0 < u' σ → B (flipS τ k σ) < B σ) ∧ (u' σ < 0 → B σ < B (flipS τ k σ))) : Sep τ k A B := by
  intro σ e e'
  have h := hu σ
  rcases lt_or_ge (u σ) 0 with hn | hn
  · have hu' : 0 < u' σ := by nlinarith
    have := (hA σ).2 hn.le
    have := (hB σ).1 hu'
    linarith
  · have hu' : u' σ < 0 := by
      rcases hn.lt_or_eq with hp | h0
      · nlinarith
      · rw [← h0, zero_mul] at h; exact absurd h (lt_irrefl 0)
    have := (hA σ).1 hn
    have := (hB σ).2 hu'
    linarith

theorem bsum_flip {T : ℕ} {x b : ℕ → Code} (hx : ∀ r, (x r).isNaN = false) (hb : ∀ r, (b r).isNaN = false)
    (j : ℕ) (σ : ℕ → SliceSalt) {p : ℕ} (hp : p < 2 * T) (k : Fin 3) :
    bsum T x b j (flipS (slc T p) k σ) p =
      bsum T x b j σ p - (if p < T then 1 else -1) * (2 * tv x σ (slc T p) k * val (tuple j k)) := by
  by_cases h : p < T
  · have e : slc T p = p := ite_eq_left h
    rw [ite_eq_left h, one_mul, e]
    exact bsum_flip_one hx hb j σ h k
  · have e : slc T p = p - T := ite_eq_right h
    rw [ite_eq_right h, e]
    have := bsum_flip_two hx hb (T := T) j σ (τ := p - T) (by omega) k
    rw [show T + (p - T) = p by omega] at this
    rw [this]; ring

theorem slc_block2 {T τ : ℕ} : slc T (T + τ) = τ := by
  unfold slc; rw [ite_eq_right (by omega)]; omega

section
variable {T : ℕ} {x b b' : ℕ → Code} (hT : T ≤ 1130) (hx : ∀ r, (x r).isNaN = false)
  (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false)

include hx hb in
/-- `I` against `I` of another slice: flip the first one's slice. -/
theorem sep_fresh_slices (j j' : ℕ) {p p' : ℕ} (hp : p < 2 * T) (h : slc T p' ≠ slc T p) :
    Sep (slc T p) 0 (fun σ => freshVal T x b j σ p) (fun σ => freshVal T x b' j' σ p') := by
  intro σ e e'
  beta_reduce at e e'
  have hm := (fresh_move hx hb j σ hp).1
  rw [e', fresh_fixed j' σ 0 h, ← e, sub_self, abs_zero] at hm
  norm_num at hm

include hx hb hb' in
/-- The two blocks of one slice: a flip of lane 0 moves them in opposite directions. -/
theorem sep_fresh_blocks (j j' : ℕ) {τ : ℕ} (hτ : τ < T) :
    Sep τ 0 (fun σ => freshVal T x b j σ τ) (fun σ => freshVal T x b' j' σ (T + τ)) := by
  intro σ e e'
  beta_reduce at e e'
  have m1 := fresh_move hx hb (T := T) j σ (p := τ) (by omega)
  have m2 := fresh_move hx hb' (T := T) j' σ (p := T + τ) (by omega)
  have hs1 : slc T τ = τ := ite_eq_left hτ
  rw [hs1] at m1
  rw [slc_block2] at m2
  obtain ⟨fi, -, hF⟩ := fl_index hx τ
  have hv := tagVal_abs_ge (fl x τ) (by rw [hF]; exact fOf_pos _) ((σ τ).neg 0) ((σ τ).man 0)
  have hne2 : freshVal T x b' j' (flipS τ 0 σ) (T + τ) ≠ freshVal T x b' j' σ (T + τ) := by
    intro h; have := m2.1; rw [h, sub_self, abs_zero] at this; norm_num at this
  rcases lt_or_gt_of_ne (show tv x σ τ 0 ≠ 0 by
    intro h0; unfold tv at h0; rw [h0, abs_zero] at hv; linarith [fOf_pos fi, hF ▸ hv]) with hn | hp
  · have a1 := (m1.2.2 hn).1 hτ
    have a2 : ¬ freshVal T x b' j' σ (T + τ) < freshVal T x b' j' (flipS τ 0 σ) (T + τ) :=
      fun h => absurd ((m2.2.2 hn).2 h) (by omega)
    rw [e, e'] at a1; exact a2 a1
  · have a1 := (m1.2.1 hp).1 hτ
    have a2 : ¬ freshVal T x b' j' (flipS τ 0 σ) (T + τ) < freshVal T x b' j' σ (T + τ) :=
      fun h => absurd ((m2.2.1 hp).2 h) (by omega)
    rw [e, e'] at a1; exact a2 a1

include hx hb hb' in
/-- Two columns at one position: flip a lane where their tuples differ. -/
theorem sep_fresh_columns {j j' : ℕ} {p : ℕ} (hp : p < 2 * T) {k : Fin 3} (hk : tuple j k ≠ tuple j' k) :
    Sep (slc T p) k (fun σ => freshVal T x b j σ p) (fun σ => freshVal T x b' j' σ p) := by
  intro σ e e'
  beta_reduce at e e'
  obtain ⟨fi, -, hF⟩ := fl_index hx (slc T p)
  rw [freshVal_bsum hx hb, freshVal_bsum hx hb'] at e e'
  have n1 := trunc14_eq_near (bsum_lt hx hb j σ p hF) e
  have n2 := trunc14_eq_near (bsum_lt hx hb j (flipS (slc T p) k σ) p hF) e'
  rw [bsum_flip hx hb j σ hp k, bsum_flip hx hb' j' σ hp k] at n2
  have hv := tagVal_abs_ge (fl x (slc T p)) (by rw [hF]; exact fOf_pos _) ((σ (slc T p)).neg k) ((σ (slc T p)).man k)
  rw [hF] at hv
  have ht := tuple_sep j j' k hk
  have hsep : 256 * fOf fi ≤
      |2 * tv x σ (slc T p) k * val (tuple j k) - 2 * tv x σ (slc T p) k * val (tuple j' k)| := by
    rw [show 2 * tv x σ (slc T p) k * val (tuple j k) - 2 * tv x σ (slc T p) k * val (tuple j' k) =
      2 * tv x σ (slc T p) k * (val (tuple j k) - val (tuple j' k)) by ring, abs_mul, abs_mul,
      abs_of_pos (by norm_num : (0 : ℚ) < 2)]
    unfold tv; rw [hF]; nlinarith [fOf_pos fi, abs_nonneg (val (tuple j k) - val (tuple j' k))]
  set A := 2 * tv x σ (slc T p) k * val (tuple j k)
  set B := 2 * tv x σ (slc T p) k * val (tuple j' k)
  set y := bsum T x b j σ p
  set z := bsum T x b' j' σ p
  by_cases hpT : p < T
  · rw [ite_eq_left hpT, one_mul, one_mul] at n2
    have e4 : A - B = (y - z) - (y - A - (z - B)) := by ring
    have := abs_sub (y - z) (y - A - (z - B))
    rw [← e4] at this; linarith
  · rw [ite_eq_right hpT] at n2
    have e4 : A - B = (y - (-1) * A - (z - (-1) * B)) - (y - z) := by ring
    have := abs_sub (y - (-1) * A - (z - (-1) * B)) (y - z)
    rw [← e4] at this; linarith

include hT hx hb in
/-- A block-1 running word against a fresh-step word of another slice: flip the running word's slice. -/
theorem sep_run_fresh (j j' : ℕ) {q p : ℕ} (hq : q < T) (hs : slc T p ≠ q) :
    Sep q 0 (fun σ => runVal T x b j σ q) (fun σ => freshVal T x b' j' σ p) := by
  intro σ e e'
  beta_reduce at e e'
  have hm := run_move hT hx hb j σ hq
  rw [e', fresh_fixed j' σ 0 hs, ← e, sub_self, abs_zero] at hm
  norm_num at hm

include hT hx hb hb' in
/-- `R_q` against a fresh-step word of slice `q`: flip slice `q − 1`, which moves `R_(q−1)` by at least 448 but not the
fresh-step words, while `RN(R_(q−1) + I_q)` equal to a word below `2^23` pins `R_(q−1)` to within a quarter. -/
theorem sep_run_fresh_same (j j' : ℕ) {q p : ℕ} (hq : q < T) (hq1 : 1 ≤ q) (hs : slc T p = q) :
    Sep (q - 1) 0 (fun σ => runVal T x b j σ q) (fun σ => freshVal T x b' j' σ p) := by
  obtain ⟨r, rfl⟩ : ∃ r, q = r + 1 := ⟨q - 1, by omega⟩
  rw [Nat.add_sub_cancel]
  intro σ e e'
  beta_reduce at e e'
  have hm := run_move hT hx hb j σ (q := r) (by omega)
  have hfq : freshVal T x b j (flipS r 0 σ) (r + 1) = freshVal T x b j σ (r + 1) :=
    fresh_fixed j σ 0 (by unfold slc; rw [ite_eq_left hq]; omega)
  have hfp : freshVal T x b' j' (flipS r 0 σ) p = freshVal T x b' j' σ p := fresh_fixed j' σ 0 (by omega)
  change rnAdd (runVal T x b j σ r) (freshVal T x b j σ (r + 1)) = _ at e
  change rnAdd (runVal T x b j (flipS r 0 σ) r) (freshVal T x b j (flipS r 0 σ) (r + 1)) = _ at e'
  rw [hfq, hfp] at e'
  have s1 := rn_small e (fresh_lt hx hb' j' σ p)
  have s2 := rn_small e' (fresh_lt hx hb' j' σ p)
  have e5 : runVal T x b j (flipS r 0 σ) r - runVal T x b j σ r =
      (runVal T x b j (flipS r 0 σ) r + freshVal T x b j σ (r + 1) - freshVal T x b' j' σ p) -
      (runVal T x b j σ r + freshVal T x b j σ (r + 1) - freshVal T x b' j' σ p) := by ring
  have := abs_sub (runVal T x b j (flipS r 0 σ) r + freshVal T x b j σ (r + 1) - freshVal T x b' j' σ p)
    (runVal T x b j σ r + freshVal T x b j σ (r + 1) - freshVal T x b' j' σ p)
  rw [← e5] at this
  linarith

include hT hx hb' in
/-- Two block-1 running words at different positions: flip the later one's slice. -/
theorem sep_run_run (j j' : ℕ) {q q' : ℕ} (h : q < q') (hq' : q' < T) :
    Sep q' 0 (fun σ => runVal T x b j σ q) (fun σ => runVal T x b' j' σ q') := by
  intro σ e e'
  beta_reduce at e e'
  have hm := run_move hT hx hb' j' σ hq'
  rw [← e', run_fixed j σ 0 h hq', e, sub_self, abs_zero] at hm
  norm_num at hm

include hx hb hb' in
/-- A block-2 running word `R_p` against a later fresh-step word `I_(T+τ)`: flip slice `τ`, which moves `R_p` weakly
(L7) and `I_(T+τ)` strictly the other way. -/
theorem sep_run2_fresh (j j' : ℕ) {p τ : ℕ} (hp : T ≤ p) (hτ : τ < T) (h : p < T + τ) :
    Sep τ 0 (fun σ => runVal T x b j σ p) (fun σ => freshVal T x b' j' σ (T + τ)) := by
  intro σ e e'
  beta_reduce at e e'
  have m1 := fresh_move hx hb (T := T) j σ (p := τ) (by omega)
  have m2 := fresh_move hx hb' (T := T) j' σ (p := T + τ) (by omega)
  have hs1 : slc T τ = τ := ite_eq_left hτ
  rw [hs1] at m1
  rw [slc_block2] at m2
  obtain ⟨fi, -, hF⟩ := fl_index hx τ
  have hv := tagVal_abs_ge (fl x τ) (by rw [hF]; exact fOf_pos _) ((σ τ).neg 0) ((σ τ).man 0)
  have hne2 : freshVal T x b' j' (flipS τ 0 σ) (T + τ) ≠ freshVal T x b' j' σ (T + τ) := by
    intro h; have := m2.1; rw [h, sub_self, abs_zero] at this; norm_num at this
  have hfix : ∀ t ≤ p, t ≠ τ → freshVal T x b j (flipS τ 0 σ) t = freshVal T x b j σ t := fun t ht htτ =>
    fresh_fixed j σ 0 (by unfold slc; split_ifs <;> omega)
  rcases lt_or_gt_of_ne (show tv x σ τ 0 ≠ 0 by
    intro h0; unfold tv at h0; rw [h0, abs_zero] at hv; linarith [fOf_pos fi, hF ▸ hv]) with hn | hpos
  · have a1 := (m1.2.2 hn).1 hτ
    have hmono := h1tChainMono T x b j σ (flipS τ 0 σ) p fun t ht => by
      by_cases htτ : t = τ
      · subst htτ; exact a1.le
      · exact (hfix t ht htτ).ge
    have a2 : freshVal T x b' j' (flipS τ 0 σ) (T + τ) < freshVal T x b' j' σ (T + τ) :=
      lt_of_le_of_ne (not_lt.1 fun h => absurd ((m2.2.2 hn).2 h) (by omega)) hne2
    rw [e, e'] at hmono; linarith
  · have a1 := (m1.2.1 hpos).1 hτ
    have hmono := h1tChainMono T x b j (flipS τ 0 σ) σ p fun t ht => by
      by_cases htτ : t = τ
      · subst htτ; exact a1.le
      · exact (hfix t ht htτ).le
    have a2 : freshVal T x b' j' σ (T + τ) < freshVal T x b' j' (flipS τ 0 σ) (T + τ) :=
      lt_of_le_of_ne (not_lt.1 fun h => absurd ((m2.2.1 hpos).2 h) (by omega)) (Ne.symm hne2)
    rw [e, e'] at hmono; linarith

/-! ### By column: O1 for far tuples, and O2 against a later slice of an opposite-signed column -/

include hx hb in
/-- A flip of lane `k` of slice `q < T` moves `I_q`'s sum by `−2·v·t`, and so `I_q` by more than `2·|v·t| − 256F`,
weakly down if `v·t ≥ 0` and weakly up if `v·t ≤ 0` (L7). -/
theorem fresh1_flip (j : ℕ) (σ : ℕ → SliceSalt) {q : ℕ} (hq : q < T) (k : Fin 3) {fi : ℕ} (hF : fl x q = fOf fi) :
    |2 * tv x σ q k * val (tuple j k)| - 256 * fOf fi <
      |freshVal T x b j (flipS q k σ) q - freshVal T x b j σ q| ∧
    (0 ≤ tv x σ q k * val (tuple j k) → freshVal T x b j (flipS q k σ) q ≤ freshVal T x b j σ q) ∧
    (tv x σ q k * val (tuple j k) ≤ 0 → freshVal T x b j σ q ≤ freshVal T x b j (flipS q k σ) q) := by
  have hs : slc T q = q := ite_eq_left hq
  have hF' : fl x (slc T q) = fOf fi := by rw [hs]; exact hF
  have hy := bsum_lt hx hb j σ q hF'
  have hy' := bsum_lt hx hb j (flipS q k σ) q hF'
  have hd := bsum_flip_one hx hb j σ hq k
  rw [freshVal_bsum hx hb, freshVal_bsum hx hb]
  refine ⟨?_, fun h => trunc14Mono ?_, fun h => trunc14Mono ?_⟩
  · have := trunc14_move_lt hy hy'
    rwa [show bsum T x b j (flipS q k σ) q - bsum T x b j σ q = -(2 * tv x σ q k * val (tuple j k)) by
      rw [hd]; ring, abs_neg] at this
  · rw [hd]; linarith
  · rw [hd]; linarith

include hx hb in
/-- With `|t| ≥ 32`, that move of `I_q` is strict: at least `2048F − 256F`. -/
theorem fresh1_strict (j : ℕ) (σ : ℕ → SliceSalt) {q : ℕ} (hq : q < T) (k : Fin 3) :
    (0 < tv x σ q k * val (tuple j k) → freshVal T x b j (flipS q k σ) q < freshVal T x b j σ q) ∧
    (tv x σ q k * val (tuple j k) < 0 → freshVal T x b j σ q < freshVal T x b j (flipS q k σ) q) := by
  obtain ⟨fi, -, hF⟩ := fl_index hx q
  obtain ⟨hm, hdn, hup⟩ := fresh1_flip hx hb j σ hq k hF
  have h0 : freshVal T x b j (flipS q k σ) q ≠ freshVal T x b j σ q := by
    intro e
    have := tv_move_ge (σ := σ) hF k (by norm_num : (0 : ℚ) ≤ 32) (tuple_abs_ge j k)
    rw [e, sub_self, abs_zero] at hm
    linarith [fOf_pos fi]
  exact ⟨fun h => lt_of_le_of_ne (hdn h.le) h0, fun h => lt_of_le_of_ne (hup h.le) (Ne.symm h0)⟩

include hT hx hb in
/-- With `|t| ≥ 36`, `I_q` moves by more than `2304F − 256F ≥ 512` and `R_(q−1)` stays, so the block-1 running word
`R_q` moves strictly the same way (L4: the rounding error is at most 256). -/
theorem run1_strict (j : ℕ) (σ : ℕ → SliceSalt) {q : ℕ} (hq : q < T) (hq1 : 1 ≤ q) (k : Fin 3)
    (ht : 36 ≤ |val (tuple j k)|) :
    (0 < tv x σ q k * val (tuple j k) → runVal T x b j (flipS q k σ) q < runVal T x b j σ q) ∧
    (tv x σ q k * val (tuple j k) < 0 → runVal T x b j σ q < runVal T x b j (flipS q k σ) q) := by
  obtain ⟨r, rfl⟩ : ∃ r, q = r + 1 := ⟨q - 1, by omega⟩
  obtain ⟨fi, -, hF⟩ := fl_index hx (r + 1)
  obtain ⟨hm, hdn, hup⟩ := fresh1_flip hx hb j σ hq k hF
  have hfix := run_fixed (x := x) (b := b) j σ k (Nat.lt_succ_self r) hq
  have h1 := run1_sum_lt hT hx hb j σ hq
  have h2 := run1_sum_lt hT hx hb j (flipS (r + 1) k σ) hq
  rw [hfix] at h2
  have hs := rn_sep h1 h2
  have hmv := tv_move_ge (σ := σ) hF k (by norm_num : (0 : ℚ) ≤ 36) ht
  have hF4 := fOf_ge fi
  have e1 : runVal T x b j σ (r + 1) = rn (runVal T x b j σ r + freshVal T x b j σ (r + 1)) := rfl
  have e2 : runVal T x b j (flipS (r + 1) k σ) (r + 1) =
      rn (runVal T x b j (flipS (r + 1) k σ) r + freshVal T x b j (flipS (r + 1) k σ) (r + 1)) := rfl
  rw [e1, e2, hfix]
  set R := runVal T x b j σ r
  set I := freshVal T x b j σ (r + 1)
  set I' := freshVal T x b j (flipS (r + 1) k σ) (r + 1)
  rw [show R + I' - (R + I) = I' - I by ring] at hs
  have hne : rn (R + I') ≠ rn (R + I) := by
    intro h; rw [h, sub_self, abs_zero] at hs; linarith
  exact ⟨fun h => lt_of_le_of_ne (rn_mono (by linarith [hdn h.le])) hne,
    fun h => lt_of_le_of_ne (rn_mono (by linarith [hup h.le])) (Ne.symm hne)⟩

include hx hb in
/-- A running word `R_p` before slice `q`'s block 2 (`p < T + q`) reads slice `q` only through `I_q`, so it moves
weakly the way `I_q` does (L7). -/
theorem run_weak (j : ℕ) (σ : ℕ → SliceSalt) {p q : ℕ} (hq : q < T) (h : p < T + q) (k : Fin 3) :
    (0 ≤ tv x σ q k * val (tuple j k) → runVal T x b j (flipS q k σ) p ≤ runVal T x b j σ p) ∧
    (tv x σ q k * val (tuple j k) ≤ 0 → runVal T x b j σ p ≤ runVal T x b j (flipS q k σ) p) := by
  obtain ⟨fi, -, hF⟩ := fl_index hx q
  obtain ⟨-, hdn, hup⟩ := fresh1_flip hx hb j σ hq k hF
  have hfix : ∀ t ≤ p, t ≠ q → freshVal T x b j (flipS q k σ) t = freshVal T x b j σ t := fun t ht htq =>
    fresh_fixed j σ k (by unfold slc; split_ifs <;> omega)
  refine ⟨fun hu => h1tChainMono T x b j (flipS q k σ) σ p fun t ht => ?_,
    fun hu => h1tChainMono T x b j σ (flipS q k σ) p fun t ht => ?_⟩
  · by_cases htq : t = q
    · subst htq; exact hdn hu
    · exact (hfix t ht htq).le
  · by_cases htq : t = q
    · subst htq; exact hup hu
    · exact (hfix t ht htq).ge

include hx hb hb' in
/-- O2 against a fresh-step word: `R_p` (`p < T + q`) of column `j` against the block-1 `I_q` of a column `j'` whose
tuple has the opposite sign in lane `k`. A flip of lane `k` of slice `q` moves them in opposite directions. -/
theorem sep_opp_fresh {j j' p q : ℕ} (hq : q < T) (h : p < T + q) {k : Fin 3}
    (hk : val (tuple j k) * val (tuple j' k) < 0) :
    Sep q k (fun σ => runVal T x b j σ p) (fun σ => freshVal T x b' j' σ q) :=
  sep_of_opposite (u := fun σ => tv x σ q k * val (tuple j k)) (u' := fun σ => tv x σ q k * val (tuple j' k))
    (fun σ => tv_opp hx σ q k hk) (fun σ => run_weak hx hb j σ hq h k) (fun σ => fresh1_strict hx hb' j' σ hq k)

include hT hx hb hb' in
/-- O2 against a running word: the same against the block-1 `R_q`, if column `j'`'s value in lane `k` is at least 36
in magnitude. -/
theorem sep_opp_run {j j' p q : ℕ} (hq : q < T) (hq1 : 1 ≤ q) (h : p < T + q) {k : Fin 3}
    (hk : val (tuple j k) * val (tuple j' k) < 0) (ht : 36 ≤ |val (tuple j' k)|) :
    Sep q k (fun σ => runVal T x b j σ p) (fun σ => runVal T x b' j' σ q) :=
  sep_of_opposite (u := fun σ => tv x σ q k * val (tuple j k)) (u' := fun σ => tv x σ q k * val (tuple j' k))
    (fun σ => tv_opp hx σ q k hk) (fun σ => run_weak hx hb j σ hq h k)
    (fun σ => run1_strict hT hx hb' j' σ hq hq1 k ht)

include hT hx hb hb' in
/-- O1 for far tuples: two columns' block-1 running words at one position `q ≥ 1`, with tuples at least 72 apart in
lane `k`. A flip of lane `k` of slice `q` keeps both `R_(q−1)` and moves the difference of the two sums by
`2·|v|·|Δt| ≥ 4608F`. Equal roundings put each pair of sums within 512 (L4), and the four cells take less than
`512F`, so `4608F − 512F ≥ 1024` is too much to absorb. -/
theorem sep_run_columns {j j' q : ℕ} (hq : q < T) (hq1 : 1 ≤ q) {k : Fin 3}
    (hk : 72 ≤ |val (tuple j k) - val (tuple j' k)|) :
    Sep q k (fun σ => runVal T x b j σ q) (fun σ => runVal T x b' j' σ q) := by
  obtain ⟨r, rfl⟩ : ∃ r, q = r + 1 := ⟨q - 1, by omega⟩
  intro σ e e'
  obtain ⟨fi, -, hF⟩ := fl_index hx (r + 1)
  have hs : slc T (r + 1) = r + 1 := ite_eq_left hq
  have hF' : fl x (slc T (r + 1)) = fOf fi := by rw [hs]; exact hF
  have f1 := run_fixed (x := x) (b := b) j σ k (Nat.lt_succ_self r) hq
  have f2 := run_fixed (x := x) (b := b') j' σ k (Nat.lt_succ_self r) hq
  have y1 := run1_sum_lt hT hx hb j σ hq
  have y2 := run1_sum_lt hT hx hb j (flipS (r + 1) k σ) hq
  have z1 := run1_sum_lt hT hx hb' j' σ hq
  have z2 := run1_sum_lt hT hx hb' j' (flipS (r + 1) k σ) hq
  rw [f1] at y2
  rw [f2] at z2
  have r1 := abs_le.1 (rn_err_small y1)
  have r2 := abs_le.1 (rn_err_small y2)
  have r3 := abs_le.1 (rn_err_small z1)
  have r4 := abs_le.1 (rn_err_small z2)
  have c1 : |freshVal T x b j σ (r + 1) - bsum T x b j σ (r + 1)| < 128 * fOf fi := by
    rw [freshVal_bsum hx hb]; exact trunc14_near_lt (bsum_lt hx hb j σ (r + 1) hF')
  have c2 : |freshVal T x b j (flipS (r + 1) k σ) (r + 1) - bsum T x b j (flipS (r + 1) k σ) (r + 1)| <
      128 * fOf fi := by
    rw [freshVal_bsum hx hb]; exact trunc14_near_lt (bsum_lt hx hb j (flipS (r + 1) k σ) (r + 1) hF')
  have c3 : |freshVal T x b' j' σ (r + 1) - bsum T x b' j' σ (r + 1)| < 128 * fOf fi := by
    rw [freshVal_bsum hx hb']; exact trunc14_near_lt (bsum_lt hx hb' j' σ (r + 1) hF')
  have c4 : |freshVal T x b' j' (flipS (r + 1) k σ) (r + 1) - bsum T x b' j' (flipS (r + 1) k σ) (r + 1)| <
      128 * fOf fi := by
    rw [freshVal_bsum hx hb']; exact trunc14_near_lt (bsum_lt hx hb' j' (flipS (r + 1) k σ) (r + 1) hF')
  have d1 := bsum_flip_one hx hb j σ hq k
  have d2 := bsum_flip_one hx hb' j' σ hq k
  have hmv := tv_move_ge (σ := σ) hF k (by norm_num : (0 : ℚ) ≤ 72) hk
  rw [show 2 * tv x σ (r + 1) k * (val (tuple j k) - val (tuple j' k)) =
    2 * tv x σ (r + 1) k * val (tuple j k) - 2 * tv x σ (r + 1) k * val (tuple j' k) by ring] at hmv
  have hF4 := fOf_ge fi
  change rn (runVal T x b j σ r + freshVal T x b j σ (r + 1)) =
    rn (runVal T x b' j' σ r + freshVal T x b' j' σ (r + 1)) at e
  change rn (runVal T x b j (flipS (r + 1) k σ) r + freshVal T x b j (flipS (r + 1) k σ) (r + 1)) =
    rn (runVal T x b' j' (flipS (r + 1) k σ) r + freshVal T x b' j' (flipS (r + 1) k σ) (r + 1)) at e'
  rw [f1, f2] at e'
  obtain ⟨c1a, c1b⟩ := abs_lt.1 c1
  obtain ⟨c2a, c2b⟩ := abs_lt.1 c2
  obtain ⟨c3a, c3b⟩ := abs_lt.1 c3
  obtain ⟨c4a, c4b⟩ := abs_lt.1 c4
  rw [d1] at c2a c2b
  rw [d2] at c4a c4b
  rcases le_abs.1 hmv with hm | hm <;> linarith

/-! ### Near the start of block 2: a flip of the last slice (M4c) -/

include hT hx hb in
/-- Every running word and the next fresh-step word sum to less than `2^33` in magnitude (L4). -/
theorem run_sum_lt (j : ℕ) (σ : ℕ → SliceSalt) {p : ℕ} (hp : p + 1 < 2 * T) :
    |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 := by
  by_cases h1 : p + 1 < T
  · exact run1_sum_lt hT hx hb j σ h1
  · have hR := run_bound hT (fun t => (fresh_facts hx hb j σ t).2) p (by omega)
    have hI := (fresh_facts (T := T) hx hb j σ (p + 1)).2
    rw [ite_eq_right h1] at hI
    have hTq : (T : ℚ) ≤ 1130 := by exact_mod_cast hT
    have := abs_add_le (runVal T x b j σ p) (freshVal T x b j σ (p + 1))
    by_cases h2 : p < T
    · rw [ite_eq_left h2] at hR
      have hq : (p : ℚ) + 1 ≤ T := by exact_mod_cast (show p + 1 ≤ T by omega)
      norm_num; linarith
    · rw [ite_eq_right h2] at hR
      have hq : (p : ℚ) + 1 ≤ 2 * T := by exact_mod_cast (show p + 1 ≤ 2 * T by omega)
      norm_num; linarith

include hT hx hb in
/-- A flip of slice `τ = T − 1` reaches `R_(τ+d)` (`d < T`) only through `I_τ` and the `d + 1` roundings from `R_τ`
on, since the block-2 words before `I_(2T−1)` read earlier slices. Each rounding takes at most 512 of the move (L4). -/
theorem run_last_chain (j : ℕ) (σ : ℕ → SliceSalt) {τ : ℕ} (hτ : τ + 1 = T) (h1 : 1 ≤ τ) (k : Fin 3) {d : ℕ}
    (hd : d < T) :
    |freshVal T x b j (flipS τ k σ) τ - freshVal T x b j σ τ| - 512 * (d + 1) ≤
      |runVal T x b j (flipS τ k σ) (τ + d) - runVal T x b j σ (τ + d)| := by
  induction d with
  | zero =>
    obtain ⟨r, rfl⟩ : ∃ r, τ = r + 1 := ⟨τ - 1, by omega⟩
    have hq : r + 1 < T := by omega
    have hfix := run_fixed (x := x) (b := b) j σ k (Nat.lt_succ_self r) hq
    have y1 := run1_sum_lt hT hx hb j σ hq
    have y2 := run1_sum_lt hT hx hb j (flipS (r + 1) k σ) hq
    rw [hfix] at y2
    have hs := rn_sep y1 y2
    show _ ≤ |rnAdd (runVal T x b j (flipS (r + 1) k σ) r) (freshVal T x b j (flipS (r + 1) k σ) (r + 1)) -
      rnAdd (runVal T x b j σ r) (freshVal T x b j σ (r + 1))|
    rw [hfix]; unfold rnAdd
    rw [show runVal T x b j σ r + freshVal T x b j (flipS (r + 1) k σ) (r + 1) -
      (runVal T x b j σ r + freshVal T x b j σ (r + 1)) =
      freshVal T x b j (flipS (r + 1) k σ) (r + 1) - freshVal T x b j σ (r + 1) by ring] at hs
    push_cast
    linarith
  | succ d ih =>
    have ih := ih (by omega)
    have hp : τ + d + 1 < 2 * T := by omega
    have hs : slc T (τ + d + 1) ≠ τ := by unfold slc; split_ifs <;> omega
    have hfix := fresh_fixed (x := x) (b := b) j σ k hs
    have y1 := run_sum_lt hT hx hb j σ hp
    have y2 := run_sum_lt hT hx hb j (flipS τ k σ) hp
    rw [hfix] at y2
    have hsep := rn_sep y1 y2
    show _ ≤ |rnAdd (runVal T x b j (flipS τ k σ) (τ + d)) (freshVal T x b j (flipS τ k σ) (τ + d + 1)) -
      rnAdd (runVal T x b j σ (τ + d)) (freshVal T x b j σ (τ + d + 1))|
    rw [hfix]; unfold rnAdd
    rw [show runVal T x b j (flipS τ k σ) (τ + d) + freshVal T x b j σ (τ + d + 1) -
      (runVal T x b j σ (τ + d) + freshVal T x b j σ (τ + d + 1)) =
      runVal T x b j (flipS τ k σ) (τ + d) - runVal T x b j σ (τ + d) by ring] at hsep
    push_cast at ih ⊢
    linarith

include hT hx hb in
/-- An `Early` block-2 running word `R_p` moves under a flip of its large lane `k` of slice `T − 1`: `I_(T−1)` moves
by more than `64F·|t| − 256F ≥ 512·(p − T + 2)`, and the `p − T + 2` roundings take at most that. -/
theorem run_early_moves (j : ℕ) (σ : ℕ → SliceSalt) {p : ℕ} (hp : T ≤ p) (hp2 : p + 3 ≤ 2 * T) {k : Fin 3}
    (hk : 32 * ((p + 2 - T : ℕ) : ℚ) + 4 ≤ |val (tuple j k)|) :
    runVal T x b j (flipS (T - 1) k σ) p ≠ runVal T x b j σ p := by
  obtain ⟨fi, -, hF⟩ := fl_index hx (T - 1)
  obtain ⟨hm, -, -⟩ := fresh1_flip hx hb j σ (show T - 1 < T by omega) k hF
  have hc := run_last_chain hT hx hb j σ (τ := T - 1) (by omega) (by omega) k (d := p - T + 1) (by omega)
  rw [show T - 1 + (p - T + 1) = p by omega] at hc
  have hmv := tv_move_ge (σ := σ) hF k (by positivity) hk
  have hF4 := fOf_ge fi
  have e : ((p + 2 - T : ℕ) : ℚ) = ((p - T + 1 : ℕ) : ℚ) + 1 := by
    rw [show p + 2 - T = (p - T + 1) + 1 by omega]; push_cast; ring
  rw [e] at hmv
  intro h
  rw [h, sub_self, abs_zero] at hc
  have hD : (0 : ℚ) ≤ ((p - T + 1 : ℕ) : ℚ) := Nat.cast_nonneg _
  nlinarith [mul_nonneg (sub_nonneg.2 hF4) (by linarith : (0 : ℚ) ≤ ((p - T + 1 : ℕ) : ℚ) + 1)]

include hT hx hb in
/-- An `Early` block-2 running word against a word that skips the last slice: a flip of the large lane of slice
`T − 1` moves the first and keeps the second. -/
theorem sep_early {j j' : ℕ} (a a' : Pos T) (hr : Run2 a) (he : Early j a) (hs : SkipsLast a') :
    ∃ k : Fin 3, Sep (T - 1) k (W T x b j a) (W T x b' j' a') := by
  obtain ⟨k, hk⟩ := he
  refine ⟨k, fun σ e e' => ?_⟩
  have hfix : W T x b' j' a' (flipS (T - 1) k σ) = W T x b' j' a' σ := by
    rcases a' with p' | ⟨q', hq1', hq2'⟩
    · simp only [SkipsLast, posIdx, Sum.elim_inl, Sum.isLeft_inl, true_and] at hs
      exact fresh_fixed (b := b') j' σ k (by unfold slc; split_ifs <;> omega)
    · simp only [SkipsLast, posIdx, Sum.elim_inr, Sum.isLeft_inr, Bool.false_eq_true, false_and, or_false] at hs
      exact run_fixed (b := b') j' σ k (show q'.val < T - 1 by omega) (by omega)
  rcases a with p | ⟨q, hq1, hq2⟩
  · exact absurd hr.1 (by simp)
  · exact run_early_moves hT hx hb j σ hr.2 hq2 hk (e'.trans (hfix.trans e.symm))

end

/-! ## Every pair that is not `Hard`, and every word that is not `Stuck` -/

theorem pair_sep {T : ℕ} {x b b' : ℕ → Code} (hT : T ≤ 1130) (hx : ∀ r, (x r).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (hb' : ∀ r, (b' r).isNaN = false) {j j' : ℕ}
    (htup : j ≠ j' → ∃ k, tuple j k ≠ tuple j' k) (a a' : Pos T) (hne : j ≠ j' ∨ a ≠ a')
    (hH : ¬ Hard j j' a a') : ∃ τ < T, ∃ k : Fin 3, Sep τ k (W T x b j a) (W T x b' j' a') := by
  rcases a with p | ⟨q, hq1, hq2⟩ <;> rcases a' with p' | ⟨q', hq1', hq2'⟩
  · have hp := p.isLt
    have hp' := p'.isLt
    by_cases hs : slc T p'.val = slc T p.val
    · by_cases hpp : p.val = p'.val
      · have hj : j ≠ j' := by
          rcases hne with h | h
          · exact h
          · exact absurd (congrArg Sum.inl (Fin.ext hpp)) h
        obtain ⟨k, hk⟩ := htup hj
        refine ⟨slc T p.val, slc_lt hp, k, ?_⟩
        show Sep _ k (fun σ => freshVal T x b j σ p.val) (fun σ => freshVal T x b' j' σ p'.val)
        rw [← hpp]; exact sep_fresh_columns hx hb hb' hp hk
      · by_cases h1 : p.val < T
        · have h2 : p'.val = T + p.val := by unfold slc at hs; split_ifs at hs <;> omega
          refine ⟨p.val, h1, 0, ?_⟩
          show Sep _ 0 (fun σ => freshVal T x b j σ p.val) (fun σ => freshVal T x b' j' σ p'.val)
          rw [h2]; exact sep_fresh_blocks hx hb hb' j j' h1
        · have h2 : p.val = T + p'.val ∧ p'.val < T := by unfold slc at hs; split_ifs at hs <;> omega
          refine ⟨p'.val, h2.2, 0, ?_⟩
          show Sep _ 0 (fun σ => freshVal T x b j σ p.val) (fun σ => freshVal T x b' j' σ p'.val)
          rw [h2.1]; exact (sep_fresh_blocks hx hb' hb j' j h2.2).symm
    · exact ⟨slc T p.val, slc_lt hp, 0, sep_fresh_slices hx hb j j' hp hs⟩
  · have hp := p.isLt
    by_cases hq : q'.val < T
    · by_cases hs : slc T p.val = q'.val
      · exact ⟨q'.val - 1, by omega, 0, (sep_run_fresh_same hT hx hb' hb j' j hq hq1' hs).symm⟩
      · exact ⟨q'.val, hq, 0, (sep_run_fresh hT hx hb' j' j hq hs).symm⟩
    · by_cases hlt : q'.val < p.val
      · refine ⟨p.val - T, by omega, 0, ?_⟩
        have := (sep_run2_fresh hx hb' hb j' j (τ := p.val - T) (not_lt.1 hq) (by omega) (by omega)).symm
        rwa [show T + (p.val - T) = p.val by omega] at this
      · by_cases ho : OppLater j' j (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) (Sum.inl p)
        · obtain ⟨hp1, hlt', k, hk, -⟩ := ho
          exact ⟨p.val, hp1, k, (sep_opp_fresh hx hb' hb hp1 hlt' hk).symm⟩
        · have he : Early j' (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) ∧ SkipsLast (Sum.inl p : Pos T) := by
            by_contra he
            exact hH (Or.inr (Or.inr ⟨⟨rfl, not_lt.1 hq⟩, fun h' => hlt h'.2, ho, he⟩))
          obtain ⟨k, hk⟩ := sep_early hT hx hb' (b' := b) (Sum.inr ⟨q', hq1', hq2'⟩) (Sum.inl p) ⟨rfl, not_lt.1 hq⟩ he.1 he.2
          exact ⟨T - 1, by omega, k, hk.symm⟩
  · have hp' := p'.isLt
    by_cases hq : q.val < T
    · by_cases hs : slc T p'.val = q.val
      · exact ⟨q.val - 1, by omega, 0, sep_run_fresh_same hT hx hb hb' j j' hq hq1 hs⟩
      · exact ⟨q.val, hq, 0, sep_run_fresh hT hx hb j j' hq hs⟩
    · by_cases hlt : q.val < p'.val
      · refine ⟨p'.val - T, by omega, 0, ?_⟩
        have := sep_run2_fresh hx hb hb' j j' (τ := p'.val - T) (not_lt.1 hq) (by omega) (by omega)
        rwa [show T + (p'.val - T) = p'.val by omega] at this
      · by_cases ho : OppLater j j' (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) (Sum.inl p')
        · obtain ⟨hp1, hlt', k, hk, -⟩ := ho
          exact ⟨p'.val, hp1, k, sep_opp_fresh hx hb hb' hp1 hlt' hk⟩
        · have he : Early j (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) ∧ SkipsLast (Sum.inl p' : Pos T) := by
            by_contra he
            exact hH (Or.inr (Or.inl ⟨⟨rfl, not_lt.1 hq⟩, fun h' => hlt h'.2, ho, he⟩))
          obtain ⟨k, hk⟩ := sep_early hT hx hb (b' := b') (Sum.inr ⟨q, hq1, hq2⟩) (Sum.inl p') ⟨rfl, not_lt.1 hq⟩ he.1 he.2
          exact ⟨T - 1, by omega, k, hk⟩
  · by_cases hq : q.val < T
    · by_cases hq' : q'.val < T
      · by_cases hqq : q.val = q'.val
        · by_cases hf : FarTags j j'
          · obtain ⟨k, hk⟩ := hf
            refine ⟨q.val, hq, k, ?_⟩
            show Sep _ k (fun σ => runVal T x b j σ q.val) (fun σ => runVal T x b' j' σ q'.val)
            rw [← hqq]; exact sep_run_columns hT hx hb hb' hq hq1 hk
          · by_cases ho : OppLater j j' (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) (Sum.inr ⟨q', hq1', hq2'⟩)
            · obtain ⟨-, hlt', k, hk, ht⟩ := ho
              exact ⟨q'.val, hq', k,
                sep_opp_run hT hx hb hb' hq' hq1' hlt' hk (ht.resolve_left Bool.false_ne_true)⟩
            · by_cases ho' : OppLater j' j (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) (Sum.inr ⟨q, hq1, hq2⟩)
              · obtain ⟨-, hlt', k, hk, ht⟩ := ho'
                exact ⟨q.val, hq, k,
                  (sep_opp_run hT hx hb' hb hq hq1 hlt' hk (ht.resolve_left Bool.false_ne_true)).symm⟩
              · exact absurd (Or.inl ⟨rfl, rfl, hq, hqq, hf, ho, ho'⟩) hH
        · rcases lt_or_gt_of_ne hqq with h | h
          · exact ⟨q'.val, hq', 0, sep_run_run hT hx hb' j j' h hq'⟩
          · exact ⟨q.val, hq, 0, (sep_run_run hT hx hb j' j h hq).symm⟩
      · by_cases ho : OppLater j' j (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) (Sum.inr ⟨q, hq1, hq2⟩)
        · obtain ⟨-, hlt', k, hk, ht⟩ := ho
          exact ⟨q.val, hq, k, (sep_opp_run hT hx hb' hb hq hq1 hlt' hk (ht.resolve_left Bool.false_ne_true)).symm⟩
        · have he : Early j' (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) ∧ SkipsLast (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) := by
            by_contra he
            exact hH (Or.inr (Or.inr ⟨⟨rfl, not_lt.1 hq'⟩, fun h' => Bool.false_ne_true h'.1, ho, he⟩))
          obtain ⟨k, hk⟩ := sep_early hT hx hb' (b' := b) (Sum.inr ⟨q', hq1', hq2'⟩) (Sum.inr ⟨q, hq1, hq2⟩) ⟨rfl, not_lt.1 hq'⟩ he.1
            he.2
          exact ⟨T - 1, by omega, k, hk.symm⟩
    · by_cases ho : OppLater j j' (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) (Sum.inr ⟨q', hq1', hq2'⟩)
      · obtain ⟨hq', hlt', k, hk, ht⟩ := ho
        exact ⟨q'.val, hq', k, sep_opp_run hT hx hb hb' hq' hq1' hlt' hk (ht.resolve_left Bool.false_ne_true)⟩
      · have he : Early j (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) ∧ SkipsLast (Sum.inr ⟨q', hq1', hq2'⟩ : Pos T) := by
          by_contra he
          exact hH (Or.inr (Or.inl ⟨⟨rfl, not_lt.1 hq⟩, fun h' => Bool.false_ne_true h'.1, ho, he⟩))
        obtain ⟨k, hk⟩ := sep_early hT hx hb (b' := b') (Sum.inr ⟨q, hq1, hq2⟩) (Sum.inr ⟨q', hq1', hq2'⟩) ⟨rfl, not_lt.1 hq⟩ he.1
          he.2
        exact ⟨T - 1, by omega, k, hk⟩

theorem word_moves {T : ℕ} {x b : ℕ → Code} (hT : T ≤ 1130) (hx : ∀ r, (x r).isNaN = false)
    (hb : ∀ r, (b r).isNaN = false) (j : ℕ) (a : Pos T) (hr : ¬ Stuck j a) :
    ∃ τ < T, ∃ k : Fin 3, ∀ σ, W T x b j a (flipS τ k σ) ≠ W T x b j a σ := by
  rcases a with p | ⟨q, hq1, hq2⟩
  · refine ⟨slc T p.val, slc_lt p.isLt, 0, fun σ e => ?_⟩
    have hm := (fresh_move hx hb j σ p.isLt).1
    change freshVal T x b j (flipS (slc T p.val) 0 σ) p.val = freshVal T x b j σ p.val at e
    rw [e, sub_self, abs_zero] at hm; norm_num at hm
  · by_cases hq : q.val < T
    · refine ⟨q.val, hq, 0, fun σ e => ?_⟩
      have hm := run_move hT hx hb j σ hq
      change runVal T x b j (flipS q.val 0 σ) q.val = runVal T x b j σ q.val at e
      rw [e, sub_self, abs_zero] at hm; norm_num at hm
    · obtain ⟨k, hk⟩ : Early j (Sum.inr ⟨q, hq1, hq2⟩ : Pos T) := by
        by_contra hc; exact hr ⟨⟨rfl, not_lt.1 hq⟩, hc⟩
      exact ⟨T - 1, by omega, k, fun σ => run_early_moves hT hx hb j σ (not_lt.1 hq) hq2 hk⟩

/-! ## On a unit -/

theorem word_eq_W (U : Unit) (i : Fin U.m) (j : Fin U.n) (a : Pos U.T) (g : U.Salt) :
    U.word (i, j, a) g = W U.T (U.row i) (U.col j) j.val a (U.rowSalt g i) := by
  cases a <;> rfl

theorem legal_facts {U : Unit} (hU : U.Legal) (i : Fin U.m) :
    U.T ≤ 1130 ∧ (∀ r, (U.row i r).isNaN = false) ∧ (∀ j r, (U.col j r).isNaN = false) ∧
    ∀ j j' : Fin U.n, j.val ≠ j'.val → ∃ k, tuple j.val k ≠ tuple j'.val k := by
  obtain ⟨hk, hn, hxq, hbq⟩ := hU
  refine ⟨by unfold Unit.T; omega, fun r => ?_, fun j r => ?_, fun j j' h => ?_⟩
  · unfold Unit.row; split_ifs
    · exact hxq _ _
    · rfl
  · unfold Unit.col; split_ifs
    · exact hbq _ _
    · rfl
  · by_contra hc
    simp only [not_exists, not_not] at hc
    exact h (h1tTupleInjective _ _ (by omega) (by omega) (funext hc))

theorem h1tOneFlip : H1TOneFlip := by
  intro U hU i
  obtain ⟨hT, hx, hb, htup⟩ := legal_facts hU i
  constructor
  · rintro ⟨j, a⟩ ⟨j', a'⟩ hne hH
    have hne' : j.val ≠ j'.val ∨ a ≠ a' := by
      by_contra hc
      simp only [not_or, not_not] at hc
      exact hne (Prod.ext (Fin.ext hc.1) hc.2)
    obtain ⟨τ, hτ, k, hsep⟩ := pair_sep hT hx (hb j) (hb j') (htup j j') a a' hne' hH
    have h := halfOfInvolution (Fin U.T → RawSlice) (psi ⟨τ, hτ⟩ k)
      (fun s => decide (U.word (i, j, a) (fun _ => s) = U.word (i, j', a') (fun _ => s))) (psi_invol _ _)
      (by
        intro s hs
        simp only [decide_eq_true_eq, decide_eq_false_iff_not] at hs ⊢
        rw [word_eq_W, word_eq_W] at hs ⊢
        rw [rowSalt_psi]
        exact hsep _ hs)
    simpa only [decide_eq_true_eq] using h
  · rintro ⟨j, a⟩ hr q
    obtain ⟨τ, hτ, k, hmove⟩ := word_moves hT hx (hb j) j.val a hr
    have h := halfOfInvolution (Fin U.T → RawSlice) (psi ⟨τ, hτ⟩ k)
      (fun s => decide (U.word (i, j, a) (fun _ => s) = q)) (psi_invol _ _)
      (by
        intro s hs
        simp only [decide_eq_true_eq, decide_eq_false_iff_not] at hs ⊢
        rw [word_eq_W] at hs ⊢
        rw [rowSalt_psi]
        intro e
        exact hmove _ (e.trans hs.symm))
    simpa only [decide_eq_true_eq] using h

end Pouw.Fp8Atom.H1T.Proofs
