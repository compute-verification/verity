import Definitions.Pouw.Fp8Atom.H1TCrossLate
import Proofs.Pouw.Fp8Atom.H1TGapStaging
import Proofs.Pouw.Fp8Atom.H1TSliceProofs
import Proofs.Pouw.Fp8Atom.H1TFlipProofs

/-!
# H-1T: `H1TCrossLate` in the exact-512 regime (proofs)

`H1TCrossLate` (`H1TGapStaging`) is a running word of one column against another column's at a position from `T − 1`
on. This file proves it in one regime, `Exact512` (its statements are in `H1TCrossLate`, which imports no proofs):
the row and both columns' real codes are 0 on the last two slices, the tag values are multiples of 32, and at the
zero salt `R_(T−3)` lies in `[2^32 + 2^18, 2^33 − 2^18]` and every block-2 word of slices `0..T−3` is a multiple of
512. There every block-2 add is exact on the 512 grid, and slices
`T − 2` and `T − 1` read only their tag fields. So from `R_(T−1)` on, a running word is a salt-dependent constant
plus `512·two512 p (q32 c σ₁) (q32 c σ₂)`: the number of 512-steps two roundings add to `512·a` (`p` the parity
of `a`) for the two slices' block-1 words `32·q32 c σ` (tag values `32·c`, tag fields `σ₁`, `σ₂`).

* **`CrossLateResidue512`** is the finite lemma over those 24 salt bits (signs and mantissas of both slices' three
  tag lanes): for `c ≠ c'` with `|c_k| ≤ 14` and either parities, the two columns' 512-steps do not differ by a
  constant. It holds at the function level for every pair (`crossLateResidue512`). Pairs whose tags differ by at
  least 2 in total are separated by the salt with mantissa 7 and signs against the difference, against all its signs
  flipped (a gap of at least 30 in `q32`, past two roundings' slack of 16). The other pairs, whose tag sums differ by
  1, are separated by signs `+` and mantissa 7 or 1 against the flip: a kernel `decide` over the 1,032 cases of the
  sum and the parities.
* **`H1TCrossLate512`** is the unit statement: on a legal unit, running words of two different columns at positions
  from `T − 1` on, both columns in the regime, are different functions of the salt (`h1tCrossLate512`).
  `crossLate_of_exact512` is `H1TCrossLate`'s conclusion for pairs in the regime, without its `¬ OppLater`.
* **`absorb_distinct`**: the regime holds for the unit of `H1TGapStaging`'s last section (`absorb_exact512`), and its
  pair `R_2257` of columns 21 and 22, equal on every salt with `absorbSalt`'s tag fields, is two different functions.

`RnMonoChainGap` and `H1TLastSliceSplit` are not needed here: block 2 adds exactly in the regime. The regime is empty
for `k ≤ 19,430`: there `|R_(T−3)| < 2^32 + 2^18`. The five theorems named above are granted, not pinned (pin when
used); the rest is lemmas.
-/

namespace Pouw.Fp8Atom.H1T.CrossLate

open Pouw.Fp8Atom Pouw.Fp8Atom.H1T Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T.Proofs

/-! ## The finite lemma -/

/-- All three tag signs flipped. -/
def negAll (σ : TagSalt) : TagSalt := (fun k => !σ.1 k, σ.2)

theorem tagMul_not (n : Bool) (m : Fin 8) : tagMul (!n) m = -tagMul n m := by
  cases n <;> simp [tagMul]

theorem q32_negAll (c : Fin 3 → ℤ) (σ : TagSalt) : q32 c (negAll σ) = -q32 c σ := by
  simp only [q32, negAll, tagMul_not]; ring

theorem rnd512_bound (p : Bool) (u : ℤ) : u - 8 ≤ 16 * rnd512 p u ∧ 16 * rnd512 p u ≤ u + 8 := by
  unfold rnd512; split_ifs <;> omega

theorem two512_bound (p : Bool) (u : ℤ) : 2 * u - 16 ≤ 16 * two512 p u u ∧ 16 * two512 p u u ≤ 2 * u + 16 := by
  unfold two512
  have h1 := rnd512_bound p u
  have h2 := rnd512_bound (p != (rnd512 p u % 2 == 1)) u
  omega

theorem sep_of_gap {p p' : Bool} {c c' : Fin 3 → ℤ} {σ : TagSalt} (h : 17 ≤ q32 c σ - q32 c' σ) :
    cross512 p p' c c' (σ, σ) ≠ cross512 p p' c c' (negAll σ, negAll σ) := by
  unfold cross512
  simp only [q32_negAll]
  have a1 := two512_bound p (q32 c σ)
  have a2 := two512_bound p' (q32 c' σ)
  have a3 := two512_bound p (-q32 c σ)
  have a4 := two512_bound p' (-q32 c' σ)
  omega

def sigMax (c c' : Fin 3 → ℤ) : TagSalt := (fun k => decide (c k < c' k), fun _ => 7)

theorem lane_max (x y : ℤ) : tagMul (decide (x < y)) 7 * x - tagMul (decide (x < y)) 7 * y = 15 * |x - y| := by
  by_cases h : x < y
  · simp [tagMul, h, abs_of_neg (sub_neg.2 h)]; ring
  · simp [tagMul, h, abs_of_nonneg (sub_nonneg.2 (not_lt.1 h))]; ring

theorem q32_sigMax (c c' : Fin 3 → ℤ) :
    q32 c (sigMax c c') - q32 c' (sigMax c c') = 15 * (|c 0 - c' 0| + |c 1 - c' 1| + |c 2 - c' 2|) := by
  have e0 := lane_max (c 0) (c' 0)
  have e1 := lane_max (c 1) (c' 1)
  have e2 := lane_max (c 2) (c' 2)
  simp only [q32, sigMax]
  linarith

def sigAll (m : Fin 8) : TagSalt := (fun _ => false, fun _ => m)

theorem q32_sigAll (c : Fin 3 → ℤ) (m : Fin 8) : q32 c (sigAll m) = (8 + m.val) * (c 0 + c 1 + c 2) := by
  simp only [q32, sigAll, tagMul]; simp; ring

def sepAt (p p' : Bool) (u u' : ℤ) : Bool :=
  two512 p u u - two512 p' u' u' != two512 p (-u) (-u) - two512 p' (-u') (-u')

theorem near_check : ∀ i : Fin 129, ∀ e p p' : Bool,
    (sepAt p p' (15 * ((i : ℤ) - 64)) (15 * ((i : ℤ) - 64 + if e then 1 else -1)) ||
      sepAt p p' (9 * ((i : ℤ) - 64)) (9 * ((i : ℤ) - 64 + if e then 1 else -1))) = true := by
  decide +kernel

theorem sep_of_sepAt {p p' : Bool} {c c' : Fin 3 → ℤ} {σ : TagSalt}
    (h : sepAt p p' (q32 c σ) (q32 c' σ) = true) :
    cross512 p p' c c' (σ, σ) ≠ cross512 p p' c c' (negAll σ, negAll σ) := by
  unfold cross512; simp only [q32_negAll]
  unfold sepAt at h; simpa using h

theorem fin3_ext {c c' : Fin 3 → ℤ} (h0 : c 0 = c' 0) (h1 : c 1 = c' 1) (h2 : c 2 = c' 2) : c = c' := by
  funext k
  match k with
  | ⟨0, _⟩ => exact h0
  | ⟨1, _⟩ => exact h1
  | ⟨2, _⟩ => exact h2

theorem crossLate512 {c c' : Fin 3 → ℤ} (hc : ∀ k, |c k| ≤ 14) (hne : c ≠ c')
    (p p' : Bool) : ∃ σ : TagSalt, cross512 p p' c c' (σ, σ) ≠ cross512 p p' c c' (negAll σ, negAll σ) := by
  have hd : 1 ≤ |c 0 - c' 0| + |c 1 - c' 1| + |c 2 - c' 2| := by
    by_contra hlt
    apply hne
    have a0 := abs_nonneg (c 0 - c' 0); have a1 := abs_nonneg (c 1 - c' 1); have a2 := abs_nonneg (c 2 - c' 2)
    apply fin3_ext
    · have : |c 0 - c' 0| = 0 := by omega
      exact sub_eq_zero.1 (abs_eq_zero.1 this)
    · have : |c 1 - c' 1| = 0 := by omega
      exact sub_eq_zero.1 (abs_eq_zero.1 this)
    · have : |c 2 - c' 2| = 0 := by omega
      exact sub_eq_zero.1 (abs_eq_zero.1 this)
  by_cases hfar : 2 ≤ |c 0 - c' 0| + |c 1 - c' 1| + |c 2 - c' 2|
  · refine ⟨sigMax c c', sep_of_gap ?_⟩
    rw [q32_sigMax]; omega
  · have hs : c' 0 + c' 1 + c' 2 = c 0 + c 1 + c 2 + 1 ∨ c' 0 + c' 1 + c' 2 = c 0 + c 1 + c 2 - 1 := by
      rcases abs_cases (c 0 - c' 0) with ⟨e0, f0⟩ | ⟨e0, f0⟩ <;>
      rcases abs_cases (c 1 - c' 1) with ⟨e1, f1⟩ | ⟨e1, f1⟩ <;>
      rcases abs_cases (c 2 - c' 2) with ⟨e2, f2⟩ | ⟨e2, f2⟩ <;> omega
    have b0 := abs_le.1 (hc 0); have b1 := abs_le.1 (hc 1); have b2 := abs_le.1 (hc 2)
    set s := c 0 + c 1 + c 2 with hs_def
    have hi : (((s + 64).toNat : ℕ) : ℤ) = s + 64 := Int.toNat_of_nonneg (by omega)
    have hlt : (s + 64).toNat < 129 := by omega
    have key : (sepAt p p' (15 * s) (15 * (c' 0 + c' 1 + c' 2)) ||
        sepAt p p' (9 * s) (9 * (c' 0 + c' 1 + c' 2))) = true := by
      rcases hs with h | h
      · have k := near_check ⟨(s + 64).toNat, hlt⟩ true p p'
        simp only [hi, add_sub_cancel_right, ite_true] at k
        rw [h]; exact k
      · have k := near_check ⟨(s + 64).toNat, hlt⟩ false p p'
        simp only [hi, add_sub_cancel_right, Bool.false_eq_true, ite_false] at k
        rw [h, sub_eq_add_neg]; exact k
    rcases Bool.or_eq_true_iff.1 key with k | k
    · refine ⟨sigAll 7, sep_of_sepAt ?_⟩
      rw [q32_sigAll, q32_sigAll]; simpa [hs_def] using k
    · refine ⟨sigAll 1, sep_of_sepAt ?_⟩
      rw [q32_sigAll, q32_sigAll]; simpa [hs_def] using k

theorem crossLateResidue512 : CrossLateResidue512 := by
  intro c c' hc hne p p'
  obtain ⟨σ, h⟩ := crossLate512 hc hne p p'
  exact ⟨_, _, h⟩

/-! ## The quiet slices -/

theorem roundEven_16 (a u : ℤ) : roundEven ((a : ℚ) + (u : ℚ) / 16) = a + rnd512 (a % 2 == 1) u := by
  have hdm := Int.mul_ediv_add_emod u 16
  have r0 : 0 ≤ u % 16 := Int.emod_nonneg u (by norm_num)
  have r1 : u % 16 < 16 := Int.emod_lt_of_pos u (by norm_num)
  have hm : (a : ℚ) + (u : ℚ) / 16 = ((a + u / 16 : ℤ) : ℚ) + ((u % 16 : ℤ) : ℚ) / 16 := by
    have : (u : ℚ) = 16 * ((u / 16 : ℤ) : ℚ) + ((u % 16 : ℤ) : ℚ) := by exact_mod_cast hdm.symm
    rw [this]; push_cast; ring
  have r0q : (0 : ℚ) ≤ ((u % 16 : ℤ) : ℚ) := by exact_mod_cast r0
  have r1q : ((u % 16 : ℤ) : ℚ) < 16 := by exact_mod_cast r1
  have hfl : ⌊(a : ℚ) + (u : ℚ) / 16⌋ = a + u / 16 := by
    rw [hm, Int.floor_eq_iff]; constructor <;> linarith
  have hfr : (a : ℚ) + (u : ℚ) / 16 - ((a + u / 16 : ℤ) : ℚ) = ((u % 16 : ℤ) : ℚ) / 16 := by rw [hm]; ring
  unfold roundEven
  simp only [hfl, hfr]
  unfold rnd512
  by_cases h1 : u % 16 < 8
  · have : ((u % 16 : ℤ) : ℚ) / 16 < 1 / 2 := by
      have : ((u % 16 : ℤ) : ℚ) < 8 := by exact_mod_cast h1
      linarith
    rw [if_pos this, if_pos h1]
  · have g8 : (8 : ℚ) ≤ ((u % 16 : ℤ) : ℚ) := by exact_mod_cast not_lt.1 h1
    have n1 : ¬ ((u % 16 : ℤ) : ℚ) / 16 < 1 / 2 := by linarith
    rw [if_neg n1, if_neg h1]
    by_cases h2 : 8 < u % 16
    · have n2 : 1 / 2 < ((u % 16 : ℤ) : ℚ) / 16 := by
        have : (8 : ℚ) < ((u % 16 : ℤ) : ℚ) := by exact_mod_cast h2
        linarith
      rw [if_pos n2, if_pos h2]; ring
    · have h8 : u % 16 = 8 := by omega
      have e8 : ¬ 1 / 2 < ((u % 16 : ℤ) : ℚ) / 16 := by rw [h8]; norm_num
      rw [if_neg e8, if_neg h2]
      rcases Int.emod_two_eq_zero_or_one a with ha | ha
      · have e : ((a + u / 16) % 2 = 0) ↔ ((u / 16 + if (a % 2 == 1) = true then 1 else 0) % 2 = 0) := by
          simp only [ha]; simp; omega
        by_cases hq : (a + u / 16) % 2 = 0
        · rw [if_pos hq, if_pos (e.1 hq)]
        · rw [if_neg hq, if_neg (fun h => hq (e.2 h))]; ring
      · have e : ((a + u / 16) % 2 = 0) ↔ ((u / 16 + if (a % 2 == 1) = true then 1 else 0) % 2 = 0) := by
          simp only [ha]; simp; omega
        by_cases hq : (a + u / 16) % 2 = 0
        · rw [if_pos hq, if_pos (e.1 hq)]
        · rw [if_neg hq, if_neg (fun h => hq (e.2 h))]; ring

theorem rnAdd_512 {a u : ℤ} (ha : 2 ^ 23 + 2 ^ 8 ≤ a) (ha' : a + 2 ^ 8 ≤ 2 ^ 24) (hu : |u| ≤ 2 ^ 11) :
    rnAdd (512 * (a : ℚ)) (32 * (u : ℚ)) = 512 * ((a + rnd512 (a % 2 == 1) u : ℤ) : ℚ) := by
  have hu' := abs_le.1 hu
  have haq : (2 : ℚ) ^ 23 + 2 ^ 8 ≤ a := by exact_mod_cast ha
  have haq' : (a : ℚ) + 2 ^ 8 ≤ 2 ^ 24 := by exact_mod_cast ha'
  have huq : -(2 : ℚ) ^ 11 ≤ u ∧ (u : ℚ) ≤ 2 ^ 11 := ⟨by exact_mod_cast hu'.1, by exact_mod_cast hu'.2⟩
  set x : ℚ := 512 * (a : ℚ) + 32 * (u : ℚ) with hx
  have hx0 : 0 < x := by rw [hx]; nlinarith
  have hb : binade x = 32 := by
    apply binade_unique hx0
    · rw [hx]; norm_num; nlinarith
    · rw [hx]; norm_num; nlinarith
  have hulp : ulp x = 512 := by
    unfold ulp; rw [abs_of_pos hx0, hb]; norm_num
  unfold rnAdd rn
  rw [← hx, if_neg (not_lt.2 hx0.le), abs_of_pos hx0, hulp]
  have : x / 512 = (a : ℚ) + (u : ℚ) / 16 := by rw [hx]; ring
  rw [this, roundEven_16]; push_cast; ring

theorem trunc14_mul32 (n : ℤ) (hn : |n| < 2 ^ 12) : trunc14 (32 * (n : ℚ)) = 32 * n := by
  rcases eq_or_ne n 0 with rfl | h0
  · simp [trunc14]
  have hy0 : (32 * (n : ℚ)) ≠ 0 := by
    have : (n : ℚ) ≠ 0 := by exact_mod_cast h0
    positivity
  have hy : 0 < |32 * (n : ℚ)| := abs_pos.2 hy0
  have hlt : |32 * (n : ℚ)| < 2 ^ (17 : ℤ) := by
    rw [abs_mul, abs_of_pos (by norm_num : (0 : ℚ) < 32)]
    have : |(n : ℚ)| < 2 ^ 12 := by rw [← Int.cast_abs]; exact_mod_cast hn
    norm_num; linarith
  have hb : binade |32 * (n : ℚ)| ≤ 16 := by
    by_contra hc
    have h1 := (binade_spec hy).1
    have : (2 : ℚ) ^ (17 : ℤ) ≤ 2 ^ binade |32 * (n : ℚ)| := zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  unfold trunc14
  rw [if_neg hy0]
  set e := binade |32 * (n : ℚ)| - 13 with he
  have hsplit : (32 * (n : ℚ)) = ((n * 2 ^ (5 - e).toNat : ℤ) : ℚ) * 2 ^ e := by
    push_cast
    have : (32 : ℚ) = 2 ^ (5 - e).toNat * 2 ^ e := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero, show (((5 - e).toNat : ℕ) : ℤ) + e = 5 by omega]; norm_num
    rw [this]; ring
  rw [hsplit, truncTo_int (two_zpow_pos _)]

theorem sliceMax_zero {xs : Fin 29 → Code} (hx : ∀ l, val (xs l) = 0) : sliceMax xs = 0 := by
  unfold sliceMax
  rcases foldl_max_mem (List.ofFn fun l => |val (xs l)|) 0 with h | h
  · exact h
  · obtain ⟨l, hl⟩ := List.mem_ofFn.1 h
    rw [← hl, hx, abs_zero]

theorem tagVal_zero (neg : Bool) (m : Fin 8) : tagVal (fOf 0) neg m = (tagMul neg m : ℚ) := by
  rw [tagVal_eq]; unfold tagMul; push_cast; ring

theorem tagMul_abs_le (n : Bool) (m : Fin 8) : |tagMul n m| ≤ 15 := by
  have hm := m.isLt
  rw [abs_le]; unfold tagMul; cases n <;> simp <;> omega

theorem q32_abs_le {c : Fin 3 → ℤ} (hc : ∀ k, |c k| ≤ 14) (σ : TagSalt) : |q32 c σ| ≤ 630 := by
  have t : ∀ k : Fin 3, |tagMul (σ.1 k) (σ.2 k) * c k| ≤ 210 := by
    intro k
    rw [abs_mul]
    have := tagMul_abs_le (σ.1 k) (σ.2 k)
    nlinarith [abs_nonneg (tagMul (σ.1 k) (σ.2 k)), abs_nonneg (c k), hc k]
  unfold q32
  have := abs_add_three (tagMul (σ.1 0) (σ.2 0) * c 0) (tagMul (σ.1 1) (σ.2 1) * c 1)
    (tagMul (σ.1 2) (σ.2 2) * c 2)
  have := t 0; have := t 1; have := t 2
  linarith

theorem tag_bound {j : ℕ} {c : Fin 3 → ℤ} (hc : ∀ k, val (tuple j k) = 32 * c k) (k : Fin 3) :
    |c k| ≤ 14 := by
  have h448 := val_abs_le (tuple_facts j k).1
  rw [hc k, abs_mul, abs_of_pos (by norm_num : (0 : ℚ) < 32), ← Int.cast_abs] at h448
  have : ((|c k| : ℤ) : ℚ) ≤ 14 := by linarith
  exact_mod_cast this

/-- A slice whose row and column real lanes are 0 writes `32·q32 c` in block 1, for tag values `32·c`. -/
theorem quiet_word {xs : Fin 29 → Code} (hx : ∀ l, val (xs l) = 0) (hxn : ∀ l, (xs l).isNaN = false)
    {b : ℕ → Code} {τ : ℕ} (hb : ∀ l : Fin 29, val (b (29 * τ + l.val)) = 0)
    (hbn : ∀ r, (b r).isNaN = false) {j : ℕ} {c : Fin 3 → ℤ} (hc : ∀ k, val (tuple j k) = 32 * c k)
    (σ : SliceSalt) :
    wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b j τ)) = 32 * q32 c (σ.neg, σ.man) := by
  have hF : floorF (sliceMax xs) = fOf 0 := by rw [sliceMax_zero hx, fOf_eq]; simp [floorF]; norm_num
  have hxb : ∀ l, |val (xs l)| < 2 ^ (0 + 5) := fun l => by rw [hx]; norm_num
  have hL := lane_facts xs hxn σ (fi := 0) (by norm_num) hF hxb
  have hC := col_facts b hbn j τ
  have hE : gridExp (formSlice xs σ).1 (colLanes b j τ) ≤ (0 : ℕ) + 13 := by
    have := gridExp_le (B := 0 + 12) (fun l => (hL l).2.2.2.2.1) (fun l => (hC l).2.2); push_cast at this ⊢; omega
  have hw : wordVal (hopperStep 0 (formSlice xs σ).1 (colLanes b j τ)) =
      trunc14 (preSum (formSlice xs σ).1 (colLanes b j τ)) := by
    unfold wordVal; rw [h1tAtomValue _ _ (fun l => (hL l).1) (fun l => (hC l).1)]; rfl
  have hcb := tag_bound hc
  have hsum : preSum (formSlice xs σ).1 (colLanes b j τ) = ((32 * q32 c (σ.neg, σ.man) : ℤ) : ℚ) := by
    unfold preSum
    set F : Fin 32 → ℚ := fun l => truncTo (2 ^ (gridExp (formSlice xs σ).1 (colLanes b j τ) - 13))
      (val ((formSlice xs σ).1 l) * val (colLanes b j τ l)) with hFdef
    have hsplit : ∑ l, F l = ∑ i : Fin 29, F (Fin.castAdd 3 i) + ∑ k : Fin 3, F (Fin.natAdd 29 k) :=
      Fin.sum_univ_add (a := 29) (b := 3) F
    rw [hsplit, Fin.sum_univ_three]
    have hreal : ∀ i : Fin 29, truncTo (2 ^ (gridExp (formSlice xs σ).1 (colLanes b j τ) - 13))
        (val ((formSlice xs σ).1 (Fin.castAdd 3 i)) * val (colLanes b j τ (Fin.castAdd 3 i))) = 0 := by
      intro i
      rw [colLanes_real b j τ _ (by simp), show (Fin.castAdd 3 i).val = i.val from rfl, hb i, mul_zero,
        truncTo_zero]
    have htag : ∀ k : Fin 3, truncTo (2 ^ (gridExp (formSlice xs σ).1 (colLanes b j τ) - 13))
        (val ((formSlice xs σ).1 (Fin.natAdd 29 k)) * val (colLanes b j τ (Fin.natAdd 29 k))) =
        (tagMul (σ.neg k) (σ.man k) : ℚ) * (32 * c k) := by
      intro k
      have e1 : (formSlice xs σ).1 (Fin.natAdd 29 k) = enc (tagVal (fOf 0) (σ.neg k) (σ.man k)) := by
        rw [← hF]; exact (formSlice_tag xs σ k).1
      have e2 : colLanes b j τ (Fin.natAdd 29 k) = tuple j k := colLanes_tag_i b j τ k
      rw [e1, e2, (tag_facts (fi := 0) (by norm_num) _ _).2.1, tag_on_grid hE, tagVal_zero, hc]
    simp only [hFdef, hreal, Finset.sum_const_zero, zero_add, htag]
    unfold q32; push_cast; ring
  rw [hw, hsum]
  have hq : |q32 c (σ.neg, σ.man)| < 2 ^ 12 := lt_of_le_of_lt (q32_abs_le hcb _) (by norm_num)
  push_cast
  exact trunc14_mul32 _ hq

theorem parity_add (a k : ℤ) : ((a + k) % 2 == 1) = ((a % 2 == 1) != (k % 2 == 1)) := by
  rcases Int.emod_two_eq_zero_or_one a with ha | ha <;> rcases Int.emod_two_eq_zero_or_one k with hk | hk
  · have : (a + k) % 2 = 0 := by omega
    simp [ha, hk, this]
  · have : (a + k) % 2 = 1 := by omega
    simp [ha, hk, this]
  · have : (a + k) % 2 = 1 := by omega
    simp [ha, hk, this]
  · have : (a + k) % 2 = 0 := by omega
    simp [ha, hk, this]

theorem rnd512_abs (p : Bool) {u : ℤ} (hu : |u| ≤ 630) : |rnd512 p u| ≤ 40 := by
  have := rnd512_bound p u
  have := abs_le.1 hu
  rw [abs_le]; constructor <;> omega

/-- Two roundings on the 512 grid from `512·a` inside `[2^32, 2^33)`. -/
theorem two_round {a u1 u2 : ℤ} (ha : 2 ^ 23 + 2 ^ 9 ≤ a) (ha' : a + 2 ^ 9 ≤ 2 ^ 24) (h1 : |u1| ≤ 630)
    (h2 : |u2| ≤ 630) :
    rnAdd (rnAdd (512 * (a : ℚ)) (32 * (u1 : ℚ))) (32 * (u2 : ℚ)) =
      512 * ((a + two512 (a % 2 == 1) u1 u2 : ℤ) : ℚ) := by
  have hk := abs_le.1 (rnd512_abs (a % 2 == 1) h1)
  rw [rnAdd_512 (by omega) (by omega) (by rw [abs_le] at h1 ⊢; omega)]
  rw [rnAdd_512 (by omega) (by omega) (by rw [abs_le] at h2 ⊢; omega)]
  unfold two512; rw [parity_add]; push_cast; ring

/-! ## The chain in the regime -/

section Chain

variable {T : ℕ} {x b : ℕ → Code} {j : ℕ}

theorem freshVal_congr_lo {σ σ' : ℕ → SliceSalt} {p : ℕ} (hp : p < T) (h : σ p = σ' p) :
    freshVal T x b j σ p = freshVal T x b j σ' p := by
  unfold freshVal freshWord; rw [if_pos hp, if_pos hp, h]

theorem freshVal_congr_hi {σ σ' : ℕ → SliceSalt} {p : ℕ} (hp : T ≤ p) (h : σ (p - T) = σ' (p - T)) :
    freshVal T x b j σ p = freshVal T x b j σ' p := by
  unfold freshVal freshWord; rw [if_neg (by omega), if_neg (by omega), h]

theorem runVal_congr {σ σ' : ℕ → SliceSalt} : ∀ {p : ℕ}, p < T → (∀ τ ≤ p, σ τ = σ' τ) →
    runVal T x b j σ p = runVal T x b j σ' p
  | 0, hp, h => by simp only [runVal]; exact freshVal_congr_lo hp (h 0 le_rfl)
  | p + 1, hp, h => by
    simp only [runVal]
    rw [runVal_congr (by omega) (fun τ hτ => h τ (by omega)), freshVal_congr_lo hp (h _ le_rfl)]

theorem abs_lt_of_rn {y : ℚ} (h : |rn y| < 2 ^ 33) : |y| < 2 ^ 33 := by
  have hr : Representable ((2 : ℚ) ^ 33) := ⟨1, 33, by norm_num, by norm_num⟩
  rw [abs_lt] at h ⊢
  constructor
  · by_contra hc
    have := rn_mono (not_lt.1 hc)
    rw [rn_neg, rn_exact hr] at this; linarith [h.1]
  · by_contra hc
    have := rn_mono (not_lt.1 hc)
    rw [rn_exact hr] at this; linarith [h.2]

theorem rnAdd_512_exact {r i : ℚ} {m n : ℤ} (hr : r = 512 * m) (hi : i = 512 * n) (h : |r + i| < 2 ^ 33) :
    rnAdd r i = r + i := by
  apply rnAdd_exact
  refine ⟨m + n, 9, ?_, ?_⟩
  · have e : r + i = 512 * ((m + n : ℤ) : ℚ) := by rw [hr, hi]; push_cast; ring
    rw [e, abs_mul, abs_of_pos (by norm_num : (0 : ℚ) < 512)] at h
    have : |((m + n : ℤ) : ℚ)| < 2 ^ 24 := by linarith
    rw [← Int.cast_abs] at this; exact_mod_cast this
  · rw [hr, hi]; push_cast; ring

/-- A block-2 step of the regime is exact and stays on the 512 grid. -/
theorem step_exact {σ : ℕ → SliceSalt}
    (h512 : ∀ q, T ≤ q → q + 3 ≤ 2 * T → ∃ n : ℤ, freshVal T x b j σ q = 512 * n)
    (hR : ∀ p, p < 2 * T → |runVal T x b j σ p| < 2 ^ 33) {p : ℕ} (hp1 : T ≤ p + 1) (hp2 : p + 1 + 3 ≤ 2 * T)
    (hm : ∃ m : ℤ, runVal T x b j σ p = 512 * m) :
    runVal T x b j σ (p + 1) = runVal T x b j σ p + freshVal T x b j σ (p + 1) ∧
      ∃ m : ℤ, runVal T x b j σ (p + 1) = 512 * m := by
  obtain ⟨m, hm⟩ := hm
  obtain ⟨n, hn⟩ := h512 (p + 1) hp1 hp2
  have hlt : |runVal T x b j σ p + freshVal T x b j σ (p + 1)| < 2 ^ 33 :=
    abs_lt_of_rn (hR (p + 1) (by omega))
  have e : runVal T x b j σ (p + 1) = runVal T x b j σ p + freshVal T x b j σ (p + 1) :=
    rnAdd_512_exact hm hn hlt
  exact ⟨e, m + n, by rw [e, hm, hn]; push_cast; ring⟩

/-- In the regime, two salts that agree on slices `0..T−3` keep the difference of their running words from `T − 1`
to `2T − 3`. -/
theorem tail_diff {σ σ' : ℕ → SliceSalt} (hag : ∀ τ, τ + 2 < T → σ τ = σ' τ)
    (h512 : ∀ q, T ≤ q → q + 3 ≤ 2 * T → ∃ n : ℤ, freshVal T x b j σ q = 512 * n)
    (h512' : ∀ q, T ≤ q → q + 3 ≤ 2 * T → ∃ n : ℤ, freshVal T x b j σ' q = 512 * n)
    (hR : ∀ p, p < 2 * T → |runVal T x b j σ p| < 2 ^ 33)
    (hR' : ∀ p, p < 2 * T → |runVal T x b j σ' p| < 2 ^ 33)
    (hm : ∃ m : ℤ, runVal T x b j σ (T - 1) = 512 * m) (hm' : ∃ m : ℤ, runVal T x b j σ' (T - 1) = 512 * m) :
    ∀ d, T - 1 + d + 3 ≤ 2 * T →
      (∃ m : ℤ, runVal T x b j σ (T - 1 + d) = 512 * m) ∧
      (∃ m : ℤ, runVal T x b j σ' (T - 1 + d) = 512 * m) ∧
      runVal T x b j σ (T - 1 + d) - runVal T x b j σ' (T - 1 + d) =
        runVal T x b j σ (T - 1) - runVal T x b j σ' (T - 1)
  | 0, _ => ⟨hm, hm', rfl⟩
  | d + 1, hd => by
    obtain ⟨a1, a2, a3⟩ := tail_diff hag h512 h512' hR hR' hm hm' d (by omega)
    have hT : 1 ≤ T := by omega
    obtain ⟨e, m⟩ := step_exact (p := T - 1 + d) h512 hR (by omega) (by omega) a1
    obtain ⟨e', m'⟩ := step_exact (p := T - 1 + d) h512' hR' (by omega) (by omega) a2
    have hf : freshVal T x b j σ (T - 1 + d + 1) = freshVal T x b j σ' (T - 1 + d + 1) :=
      freshVal_congr_hi (by omega) (hag _ (by omega))
    refine ⟨m, m', ?_⟩
    rw [show T - 1 + (d + 1) = T - 1 + d + 1 by omega, e, e', hf]
    linarith

/-- The two quiet slices of the regime: from `R_(T−3) = 512·a`, `R_(T−1)` is `512·(a + two512 …)` of the two
slices' closed-form words. -/
theorem quiet_two {σ : ℕ → SliceSalt} (hT : 3 ≤ T) (hx : ∀ r, 29 * (T - 2) ≤ r → val (x r) = 0)
    (hb : ∀ r, 29 * (T - 2) ≤ r → val (b r) = 0) (hxn : ∀ r, (x r).isNaN = false)
    (hbn : ∀ r, (b r).isNaN = false)
    {c : Fin 3 → ℤ} (hc : ∀ k, val (tuple j k) = 32 * c k) {a : ℤ} (ha : runVal T x b j σ (T - 3) = 512 * a)
    (ha1 : 2 ^ 23 + 2 ^ 9 ≤ a) (ha2 : a + 2 ^ 9 ≤ 2 ^ 24) :
    runVal T x b j σ (T - 1) = 512 * ((a + two512 (a % 2 == 1) (q32 c ((σ (T - 2)).neg, (σ (T - 2)).man))
      (q32 c ((σ (T - 1)).neg, (σ (T - 1)).man)) : ℤ) : ℚ) := by
  obtain ⟨t, rfl⟩ : ∃ t, T = t + 3 := ⟨T - 3, by omega⟩
  have hcb := tag_bound hc
  have hq : ∀ τ, t + 1 ≤ τ → τ < t + 3 →
      freshVal (t + 3) x b j σ τ = 32 * q32 c ((σ τ).neg, (σ τ).man) := by
    intro τ h1 h2
    unfold freshVal freshWord
    rw [if_pos h2]
    exact quiet_word (fun l => hx (29 * τ + l.val) (by omega)) (fun l => hxn _)
      (fun l => hb _ (by omega)) hbn hc (σ τ)
  simp only [show t + 3 - 1 = t + 1 + 1 by omega, show t + 3 - 3 = t by omega, show t + 3 - 2 = t + 1 by omega]
    at ha ⊢
  simp only [runVal]
  rw [ha, hq (t + 1) le_rfl (by omega), hq (t + 1 + 1) (by omega) (by omega)]
  have := two_round ha1 ha2 (q32_abs_le hcb ((σ (t + 1)).neg, (σ (t + 1)).man))
    (q32_abs_le hcb ((σ (t + 1 + 1)).neg, (σ (t + 1 + 1)).man))
  push_cast at this ⊢
  exact this

end Chain

/-! ## On a unit -/

/-- The raw words of a slice whose tag fields are `w` (signs `w.1`, mantissas `w.2`) and every other bit 0. -/
def rawTag (w : TagSalt) : RawSlice := fun ℓ =>
  if ℓ.val = 17 then
    ⟨(if w.1 0 then 2147483648 else 0) + (w.2 0).val * 8388608, by have := (w.2 0).isLt; split <;> omega⟩
  else if ℓ.val = 16 then
    ⟨((if w.1 1 then 32768 else 0) + (w.2 1).val * 128) + ((if w.1 2 then 2147483648 else 0) + (w.2 2).val * 8388608),
      by have := (w.2 1).isLt; have := (w.2 2).isLt; split <;> split <;> omega⟩
  else ⟨0, by norm_num⟩

theorem tag_bit {v b : ℕ} {n : Bool} (h : v / 2 ^ b % 2 = if n then 1 else 0) : bitAt v b = n := by
  rw [bitAt, Nat.shiftRight_eq_div_pow, h]; cases n <;> rfl

theorem tag_bits3 {v b : ℕ} {m : Fin 8} (h : v / 2 ^ b % 8 = m.val) : bits3 v b = m := by
  unfold bits3; ext; simp only [Nat.shiftRight_eq_div_pow, h]

theorem extract_rawTag (w : TagSalt) : ((extract (rawTag w)).neg, (extract (rawTag w)).man) = w := by
  obtain ⟨n, m⟩ := w
  have h17 : ((rawTag (n, m) 17 : Fin (2 ^ 32)) : ℕ) = (if n 0 then 2147483648 else 0) + (m 0).val * 8388608 := by
    simp [rawTag]
  have h16 : ((rawTag (n, m) 16 : Fin (2 ^ 32)) : ℕ) = ((if n 1 then 32768 else 0) + (m 1).val * 128) +
      ((if n 2 then 2147483648 else 0) + (m 2).val * 8388608) := by
    simp [rawTag]
  have m0 := (m 0).isLt; have m1 := (m 1).isLt; have m2 := (m 2).isLt
  refine Prod.ext (funext fun k => ?_) (funext fun k => ?_)
  · simp only [extract, pick3]
    match k with
    | ⟨0, _⟩ =>
      simp only [ite_true]; rw [h17]; apply tag_bit
      norm_num; split_ifs <;> omega
    | ⟨1, _⟩ =>
      simp only [show (1 : ℕ) ≠ 0 by norm_num, ite_false, ite_true]; rw [h16]; apply tag_bit
      norm_num; split_ifs <;> omega
    | ⟨2, _⟩ =>
      simp only [show (2 : ℕ) ≠ 0 by norm_num, show (2 : ℕ) ≠ 1 by norm_num, ite_false]
      rw [h16]; apply tag_bit
      norm_num; split_ifs <;> omega
  · simp only [extract, pick3]
    match k with
    | ⟨0, _⟩ =>
      simp only [ite_true]; rw [h17]; apply tag_bits3
      norm_num; split_ifs <;> omega
    | ⟨1, _⟩ =>
      simp only [show (1 : ℕ) ≠ 0 by norm_num, ite_false, ite_true]; rw [h16]; apply tag_bits3
      norm_num; split_ifs <;> omega
    | ⟨2, _⟩ =>
      simp only [show (2 : ℕ) ≠ 0 by norm_num, show (2 : ℕ) ≠ 1 by norm_num, ite_false]
      rw [h16]; apply tag_bits3
      norm_num; split_ifs <;> omega

/-- The unit's salt with tag fields `s.1` on slice `T − 2`, `s.2` on slice `T − 1`, and every other raw word 0. -/
def tagSalt (U : Unit) (s : TagSalt × TagSalt) : U.Salt := fun _ τ =>
  if τ.val + 2 = U.T then rawTag s.1 else if τ.val + 1 = U.T then rawTag s.2 else rawZero

theorem rowSalt_lo (U : Unit) (s : TagSalt × TagSalt) (i : Fin U.m) {τ : ℕ} (h : τ + 2 < U.T) :
    U.rowSalt (tagSalt U s) i τ = extract rawZero := by
  unfold Unit.rowSalt tagSalt; rw [dif_pos (by omega)]; simp only; rw [if_neg (by omega), if_neg (by omega)]

theorem rowSalt_T2 (U : Unit) (s : TagSalt × TagSalt) (i : Fin U.m) (hT : 2 ≤ U.T) :
    U.rowSalt (tagSalt U s) i (U.T - 2) = extract (rawTag s.1) := by
  unfold Unit.rowSalt tagSalt; rw [dif_pos (by omega)]; simp only; rw [if_pos (by omega)]

theorem rowSalt_T1 (U : Unit) (s : TagSalt × TagSalt) (i : Fin U.m) (hT : 2 ≤ U.T) :
    U.rowSalt (tagSalt U s) i (U.T - 1) = extract (rawTag s.2) := by
  unfold Unit.rowSalt tagSalt; rw [dif_pos (by omega)]; simp only; rw [if_neg (by omega), if_pos (by omega)]

/-- In the regime, the change of a running word from `T − 1` on between two tag salts is its two quiet slices'. -/
theorem col_gap {U : Unit} (hU : U.Legal) {i : Fin U.m} {j : Fin U.n} (hE : Exact512 U i j)
    {c : Fin 3 → ℤ} (hc : ∀ k, val (tuple j.val k) = 32 * c k) {a : ℤ}
    (ha : runVal U.T (U.row i) (U.col j) j.val (fun _ => extract rawZero) (U.T - 3) = 512 * a)
    (ha1 : 2 ^ 23 + 2 ^ 9 ≤ a) (ha2 : a + 2 ^ 9 ≤ 2 ^ 24) (s s' : TagSalt × TagSalt) {P : ℕ}
    (hP1 : U.T ≤ P + 1) (hP2 : P + 3 ≤ 2 * U.T) :
    runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) P -
        runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s') i) P =
      512 * ((two512 (a % 2 == 1) (q32 c s.1) (q32 c s.2) -
        two512 (a % 2 == 1) (q32 c s'.1) (q32 c s'.2) : ℤ) : ℚ) := by
  obtain ⟨hT3, hz, -, -, h512z⟩ := hE
  obtain ⟨-, hx, hb, -⟩ := legal_facts hU i
  have hA : ∀ s : TagSalt × TagSalt,
      runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) (U.T - 3) = 512 * a := fun s => by
    rw [← ha]; exact runVal_congr (by omega) (fun τ hτ => rowSalt_lo U s i (by omega))
  have h512 : ∀ (s : TagSalt × TagSalt) q, U.T ≤ q → q + 3 ≤ 2 * U.T →
      ∃ n : ℤ, freshVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) q = 512 * n := by
    intro s q h1 h2
    obtain ⟨n, hn⟩ := h512z q h1 h2
    exact ⟨n, by rw [← hn]; exact freshVal_congr_hi h1 (rowSalt_lo U s i (by omega))⟩
  have hR1 : ∀ s : TagSalt × TagSalt, runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) (U.T - 1) =
      512 * ((a + two512 (a % 2 == 1) (q32 c s.1) (q32 c s.2) : ℤ) : ℚ) := by
    intro s
    rw [quiet_two hT3 (fun r hr => (hz r hr).1) (fun r hr => (hz r hr).2) hx (hb j) hc (hA s) ha1 ha2,
      rowSalt_T2 U s i (by omega), rowSalt_T1 U s i (by omega), extract_rawTag, extract_rawTag]
  have hR : ∀ (s : TagSalt × TagSalt) p, p < 2 * U.T →
      |runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) p| < 2 ^ 33 :=
    fun s p hp => (h1tRange U hU i j (tagSalt U s) p hp).2.2
  obtain ⟨-, -, hd⟩ := tail_diff (fun τ hτ => by rw [rowSalt_lo U s i hτ, rowSalt_lo U s' i hτ])
    (h512 s) (h512 s') (hR s) (hR s') ⟨_, hR1 s⟩ ⟨_, hR1 s'⟩ (P - (U.T - 1)) (by omega)
  rw [show U.T - 1 + (P - (U.T - 1)) = P by omega] at hd
  rw [hd, hR1 s, hR1 s']; push_cast; ring

theorem h1tCrossLate512_of (hF : CrossLateResidue512) : H1TCrossLate512 := by
  intro U hU i ⟨j, a⟩ ⟨j', a'⟩ hjj ha ha' hP hP' hE hE' heq
  rcases a with a | ⟨p, hp1, hp2⟩
  · simp at ha
  rcases a' with a' | ⟨p', hp1', hp2'⟩
  · simp at ha'
  simp only [posIdx, Sum.elim_inr] at hP hP'
  obtain ⟨A, hA, hA1, hA2⟩ := hE.2.2.2.1
  obtain ⟨A', hA', hA1', hA2'⟩ := hE'.2.2.2.1
  choose c hc using hE.2.2.1
  choose c' hc' using hE'.2.2.1
  have hne : c ≠ c' := by
    rintro rfl
    obtain ⟨-, -, -, htup⟩ := legal_facts hU i
    obtain ⟨k, hk⟩ := htup j j' (fun h => hjj (Fin.ext h))
    have := tuple_sep _ _ k hk
    rw [hc k, hc' k, sub_self, abs_zero] at this; norm_num at this
  obtain ⟨s, s', hs⟩ := hF c c' (tag_bound hc) hne (A % 2 == 1) (A' % 2 == 1)
  have e1 := col_gap hU hE hc hA hA1 hA2 s s' hP hp2
  have e2 := col_gap hU hE' hc' hA' hA1' hA2' s s' hP' hp2'
  have g1 : runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s) i) p.val =
      runVal U.T (U.row i) (U.col j') j'.val (U.rowSalt (tagSalt U s) i) p'.val := congrFun heq (tagSalt U s)
  have g2 : runVal U.T (U.row i) (U.col j) j.val (U.rowSalt (tagSalt U s') i) p.val =
      runVal U.T (U.row i) (U.col j') j'.val (U.rowSalt (tagSalt U s') i) p'.val := congrFun heq (tagSalt U s')
  have hq : ((two512 (A % 2 == 1) (q32 c s.1) (q32 c s.2) -
        two512 (A % 2 == 1) (q32 c s'.1) (q32 c s'.2) : ℤ) : ℚ) =
      ((two512 (A' % 2 == 1) (q32 c' s.1) (q32 c' s.2) -
        two512 (A' % 2 == 1) (q32 c' s'.1) (q32 c' s'.2) : ℤ) : ℚ) := by
    linarith
  have hz := Int.cast_injective (α := ℚ) hq
  apply hs
  unfold cross512
  omega

theorem h1tCrossLate512 : H1TCrossLate512 := h1tCrossLate512_of crossLateResidue512

/-- `H1TCrossLate`'s conclusion for its pairs in the regime, without its `¬ OppLater` hypothesis. -/
theorem crossLate_of_exact512 {U : Unit} (hU : U.Legal) (i : Fin U.m) {a a' : Fin U.n × Pos U.T}
    (hj : a.1 ≠ a'.1) (ha : Run2 a.2) (ha' : a'.2.isRight = true) (hp' : U.T ≤ posIdx a'.2 + 1)
    (hE : Exact512 U i a.1) (hE' : Exact512 U i a'.1) : U.word (i, a) ≠ U.word (i, a') :=
  h1tCrossLate512 U hU i a a' hj ha.1 ha' (by have := ha.2; omega) hp' hE hE'


/-! ## The regime is not empty -/

/-- The E4M3 code of 448. -/
def code448 : Code := ⟨0x7E, by norm_num⟩

/-- The instance's operands on slices `0..1127` at the zero salt. -/
def absorbOps : (Fin 32 → Code) × (Fin 32 → Code) := formSlice (fun _ => code448) (extract rawZero)

/-- Column `j`'s lanes on slices `0..1127` of the instance. -/
def absorbCol (j : ℕ) : Fin 32 → Code := colLanes (fun _ => code448) j 0

theorem absorbUnit_T : absorbUnit.T = 1130 := by decide

theorem absorbUnit_legal : absorbUnit.Legal := by
  refine ⟨by decide, by decide, fun i l => ?_, fun l j => ?_⟩ <;> simp only [absorbUnit] <;> split <;> rfl

theorem absorb_row (i : Fin absorbUnit.m) {τ : ℕ} (h : τ ≤ 1127) :
    realLanes (absorbUnit.row i) τ = fun _ => code448 := by
  funext l
  have := l.isLt
  have hk : 29 * τ + l.val < absorbUnit.k := by show _ < 32768; omega
  unfold realLanes Unit.row
  rw [dif_pos hk]
  simp only [absorbUnit]
  rw [if_pos (by omega)]; rfl

theorem absorb_col (j : Fin absorbUnit.n) {τ : ℕ} (h : τ ≤ 1127) :
    colLanes (absorbUnit.col j) j.val τ = colLanes (fun _ => code448) j.val 0 := by
  funext l
  unfold colLanes
  split_ifs with hl
  · have hk : 29 * τ + l.val < absorbUnit.k := by show _ < 32768; omega
    unfold Unit.col
    rw [dif_pos hk]
    simp only [absorbUnit]
    rw [if_pos (by omega)]; rfl
  · rfl

theorem absorb_fresh1 (i : Fin absorbUnit.m) (j : Fin absorbUnit.n) {p : ℕ} (hp : p ≤ 1127) :
    freshVal 1130 (absorbUnit.row i) (absorbUnit.col j) j.val (fun _ => extract rawZero) p =
      wordVal (hopperStep 0 absorbOps.1 (absorbCol j.val)) := by
  unfold freshVal freshWord; rw [if_pos (by omega), absorb_row i hp, absorb_col j hp]; rfl

theorem absorb_fresh2 (i : Fin absorbUnit.m) (j : Fin absorbUnit.n) {q : ℕ} (h1 : 1130 ≤ q) (h2 : q ≤ 2257) :
    freshVal 1130 (absorbUnit.row i) (absorbUnit.col j) j.val (fun _ => extract rawZero) q =
      wordVal (hopperStep 0 absorbOps.2 (absorbCol j.val)) := by
  unfold freshVal freshWord; rw [if_neg (by omega), absorb_row i (by omega), absorb_col j (by omega)]; rfl

/-- The instance's words at the zero salt: block 1 is `2669·2^11` (column 21) or `2671·2^11` (column 22) on every
slice `0..1127`, block 2 is `512·692` or `512·684`. -/
theorem absorb_words0 :
    wordVal (hopperStep 0 absorbOps.1 (absorbCol 21)) = 2669 * 2 ^ 11 ∧
      wordVal (hopperStep 0 absorbOps.1 (absorbCol 22)) = 2671 * 2 ^ 11 ∧
      wordVal (hopperStep 0 absorbOps.2 (absorbCol 21)) = 512 * 692 ∧
      wordVal (hopperStep 0 absorbOps.2 (absorbCol 22)) = 512 * 684 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide +kernel

/-- A chain of equal increments `m·2^e` is exact while `(p + 1)·|m| < 2^24`. -/
theorem chain_exact {T : ℕ} {x b : ℕ → Code} {j : ℕ} {σ : ℕ → SliceSalt} {N : ℕ} {m : ℤ} {e : ℕ}
    (hf : ∀ p ≤ N, freshVal T x b j σ p = m * 2 ^ e) (hm : ((N + 1 : ℕ) : ℤ) * |m| < 2 ^ 24) :
    ∀ p ≤ N, runVal T x b j σ p = ((p + 1 : ℕ) : ℚ) * m * 2 ^ e
  | 0, _ => by simp only [runVal]; rw [hf 0 (by omega)]; push_cast; ring
  | p + 1, hp => by
    simp only [runVal]
    rw [chain_exact hf hm p (by omega), hf (p + 1) hp, rnAdd_exact]
    · push_cast; ring
    · refine ⟨((p + 2 : ℕ) : ℤ) * m, (e : ℤ), ?_, ?_⟩
      · rw [abs_mul, abs_of_nonneg (by positivity)]
        have : ((p + 2 : ℕ) : ℤ) ≤ ((N + 1 : ℕ) : ℤ) := by push_cast; omega
        nlinarith [abs_nonneg m]
      · rw [zpow_natCast]; push_cast; ring

/-- A column of the instance is in the regime when its block-1 word is `m·2^11` with `R_1127 = 1128·m·2^11` in range,
its block-2 word is a multiple of 512, and its tag values are multiples of 32. -/
theorem absorb_regime (j : Fin absorbUnit.n) {m n : ℤ} {c : Fin 3 → ℤ}
    (h1 : wordVal (hopperStep 0 absorbOps.1 (absorbCol j.val)) = m * 2 ^ 11) (hm0 : 0 ≤ m)
    (hm1 : 2 ^ 23 + 2 ^ 9 ≤ 4512 * m) (hm2 : 4512 * m + 2 ^ 9 ≤ 2 ^ 24)
    (h2 : wordVal (hopperStep 0 absorbOps.2 (absorbCol j.val)) = 512 * n)
    (hc : ∀ k, val (tuple j.val k) = 32 * c k) : Exact512 absorbUnit ⟨0, by decide⟩ j := by
  unfold Exact512
  rw [absorbUnit_T]
  refine ⟨by norm_num, fun r hr => ?_, fun k => ⟨c k, hc k⟩, ⟨4512 * m, ?_, hm1, hm2⟩,
    fun q hq1 hq2 => ⟨n, ?_⟩⟩
  · constructor
    · unfold Unit.row; split_ifs
      · simp only [absorbUnit]; rw [if_neg (by omega)]; decide +kernel
      · decide +kernel
    · unfold Unit.col; split_ifs
      · simp only [absorbUnit]; rw [if_neg (by omega)]; decide +kernel
      · decide +kernel
  · rw [chain_exact (N := 1127) (m := m) (e := 11) (fun p hp => by rw [absorb_fresh1 _ j hp, h1])
      (by rw [abs_of_nonneg hm0]; push_cast; omega) 1127 le_rfl]
    push_cast; ring
  · rw [absorb_fresh2 _ j hq1 (by omega), h2]

theorem absorb_exact512 : Exact512 absorbUnit ⟨0, by decide⟩ ⟨21, by decide⟩ ∧
    Exact512 absorbUnit ⟨0, by decide⟩ ⟨22, by decide⟩ := by
  obtain ⟨w1, w2, w3, w4⟩ := absorb_words0
  constructor
  · exact absorb_regime ⟨21, by decide⟩ w1 (by norm_num) (by norm_num) (by norm_num) w3 (c := pick3 13 1 1)
      (by decide +kernel)
  · exact absorb_regime ⟨22, by decide⟩ w2 (by norm_num) (by norm_num) (by norm_num) w4 (c := pick3 14 1 1)
      (by decide +kernel)

/-- The pair that `absorb_words` and `absorb_rounds` make equal at every salt with `absorbSalt`'s tag fields on slices
1128 and 1129 (`R_2257` of columns 21 and 22) is two different functions of the salt. -/
theorem absorb_distinct :
    absorbUnit.word (⟨0, by decide⟩, ⟨21, by decide⟩, .inr ⟨⟨2257, by rw [absorbUnit_T]; norm_num⟩, by
      simp only [absorbUnit_T]; norm_num⟩) ≠
    absorbUnit.word (⟨0, by decide⟩, ⟨22, by decide⟩, .inr ⟨⟨2257, by rw [absorbUnit_T]; norm_num⟩, by
      simp only [absorbUnit_T]; norm_num⟩) :=
  h1tCrossLate512 absorbUnit absorbUnit_legal _ _ _ (by decide) rfl rfl (by simp [posIdx, absorbUnit_T])
    (by simp [posIdx, absorbUnit_T]) absorb_exact512.1 absorb_exact512.2

end Pouw.Fp8Atom.H1T.CrossLate
