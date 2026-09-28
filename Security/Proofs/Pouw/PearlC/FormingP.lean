import Definitions.Pouw.PearlC.FormingPDefs
import Proofs.Pouw.PearlC.AtomErr
import Definitions.Pouw.PearlC.Sm120

/-!
# P3 over a `Pipeline`: the step's error at any width, and sm_120's atom-mix

`AtomErr.lean` proves the atom-mix's `PairErr` for Π's step `hopperE4m3K32 = ⟨[32], 14, −139⟩`. This file proves the
step's error for one group of `n ≤ 32` products at any adder width `w`, instantiates it at `w = 14` (`AtomErr`'s
statement) and at sm_120's `w = 26`, and carries the flat-pair instances to `atomMixP (sm120E4m3K32 z)` for any floor
`z ≤ −5` (`Sm120`'s step, `⟨[32], 26, z⟩`: so for `sm120E4m3`'s `−133` and anywhere in `[−150, −124]`).

* **The step at width `w`** (`pipeStep_value`, `pipeStep_err`). With the step's exponent `E` in `[w − 127, 120]`, the
  value the cast reads is two truncations toward zero: to the grid `g = 2^(E − w + 1)`, then the sum to `min w 24`
  significant bits (`truncP`), since `fromAdder` returns a wider adder to FP32's 24 bits. So the error is at most
  `(n + 1)·g + (|S| + (n + 1)·g)/2^(min w 24 − 1)`: `2^−13` relative at `w = 14`, `2^−23` at `w = 26`.
* **Instances**: `hopperStep_errP` is `AtomErr.hopperStep_err` word for word; `sm120Step_err` is `w = 26`, `E ≥ −101`.
* **Flat pairs at `w = 26`** (`atomMixP_flat_err`, `atomMixP_pairErr`): `ε = 2^−18` (`AtomErr` has `1/64`). The floor
  only enters through `2^z ≤ 1/32 ≤ 16·|E′|`.
* **The instances without `PairErr`**: `atomMixChainP_saltLive_flat` (salt-live when
  `(1 + 17·2^−18)·|x| < (15 − 17·2^−18)·c·σ`, about `14.9990·c·σ`, for `8/15 ≤ c ≤ 4`) and `rhoD_atomMixP_flat`
  (`ρ_D = 1/64` for `2667/5000 ≤ c_i ≤ 4`; the H100 needs `648/943`).
* **`saltDeadP`**: `saltDeadP_eq` (for every `p`, `saltDead` of `atomMixP p`), `saltDeadP_card_le_of_live` and
  `saltDeadP_card_le` (at most `ρ_D·m·k` salt-dead elements at sm_120).

The width-generic alignment and normalization (`alignedW_value`, `normalizeW_spec`) are proved here rather than taken
from `PeelExactProofs`' width-26 lemmas, since the step lemma is parametric in `w`.
-/

namespace Pouw.PearlC.FormingP

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Forming
  Pouw.PearlC.AtomErr

/-- At Π's step the parametric chain is §5's atom-mix. -/
theorem atomMixP_hopper : atomMixP hopperE4m3K32 = FormChain.atomMix := rfl

/-- At Π's step the parametric predicate is P3's `SaltDead.saltDead`. -/
theorem saltDeadP_hopper : saltDeadP hopperE4m3K32 = SaltDead.saltDead := rfl

/-! ## Alignment and normalization at any width -/

theorem toAdder_shiftRight (w m d : ℕ) :
    toAdder w m >>> d = ⌊(m : ℚ) * 2 ^ ((w : ℤ) - 24 - d)⌋₊ := by
  unfold toAdder
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq, ← Nat.floor_div_eq_div (K := ℚ)]
    congr 1
    push_cast
    rw [show ((w : ℤ) - 24 - d) = ((w - 24 : ℕ) : ℤ) - d by omega, zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast]
    ring
  · rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul, ← pow_add,
      ← Nat.floor_div_eq_div (K := ℚ)]
    congr 1
    push_cast
    rw [show ((w : ℤ) - 24 - d) = -((24 - w + d : ℕ) : ℤ) by omega, zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- A term aligned to exponent `E` at width `w` is its value truncated toward zero to the grid `2^(E − w + 1)`
(`H1T.Proofs.aligned_value` at `w = 14`). -/
theorem alignedW_value {w : ℕ} {E : ℤ} {t : Term} (hE : t.mag ≠ 0 → t.exp ≤ E) :
    (aligned w E t : ℚ) * 2 ^ (E - w + 1) = truncTo (2 ^ (E - w + 1)) t.value := by
  by_cases hm : t.mag = 0
  · have : t.value = 0 := by simp [Term.value, hm]
    rw [this, truncTo_zero]; simp [aligned, hm]
  · have hd : ((E - t.exp).toNat : ℤ) = E - t.exp := Int.toNat_of_nonneg (by linarith [hE hm])
    set q : ℚ := (t.mag : ℚ) * 2 ^ ((w : ℤ) - 24 - (E - t.exp).toNat) with hq
    have hv : t.value = ((if t.neg then -1 else 1) * q) * 2 ^ (E - w + 1) := by
      unfold Term.value
      rw [hq, mul_assoc, mul_assoc, mul_assoc, ← zpow_add₀ two_ne_zero, hd]
      congr 3; ring
    rw [hv, truncTo_mul (two_zpow_pos _)]
    congr 1
    have hnn : 0 ≤ q := mul_nonneg (Nat.cast_nonneg _) (two_zpow_pos _).le
    simp only [aligned, hm, ite_false, toAdder_shiftRight, ← hq]
    unfold truncZ
    cases t.neg
    · simp [hnn]
    · by_cases h0 : q = 0
      · simp [h0]
      · have : ¬ (0 ≤ -q) := not_le.2 (neg_neg_of_pos (lt_of_le_of_ne hnn (Ne.symm h0)))
        simp [this]

theorem toAdder_lt (w m : ℕ) (hm : m < 2 ^ 25) : toAdder w m < 2 ^ (w + 1) := by
  unfold toAdder
  split_ifs with h
  · rw [Nat.shiftLeft_eq]
    calc m * 2 ^ (w - 24) < 2 ^ 25 * 2 ^ (w - 24) := Nat.mul_lt_mul_of_pos_right hm (by positivity)
      _ = 2 ^ (w + 1) := by rw [← pow_add]; congr 1; omega
  · rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]
    rwa [show w + 1 + (24 - w) = 25 by omega]

theorem alignedW_natAbs_lt (w : ℕ) (E : ℤ) (t : Term) (hm : t.mag < 2 ^ 25) :
    (aligned w E t).natAbs < 2 ^ (w + 1) := by
  have h := toAdder_lt w t.mag hm
  have h' : toAdder w t.mag >>> (E - t.exp).toNat ≤ toAdder w t.mag := by
    rw [Nat.shiftRight_eq_div_pow]; exact Nat.div_le_self _ _
  unfold aligned
  dsimp only
  split_ifs
  · simp
  · rw [Int.natAbs_neg, Int.natAbs_natCast]; omega
  · rw [Int.natAbs_natCast]; omega

/-- Truncation toward zero to `p` significant bits (`trunc14` at `p = 14`). -/
def truncP (p : ℕ) (y : ℚ) : ℚ := if y = 0 then 0 else truncTo (2 ^ (binade |y| - ((p : ℤ) - 1))) y

theorem truncP_14 : truncP 14 = trunc14 := by
  funext y; simp only [truncP, trunc14]; norm_num

theorem abs_truncP_sub_le {p : ℕ} (hp : 1 ≤ p) (y : ℚ) : |truncP p y - y| ≤ |y| / 2 ^ (p - 1) := by
  rcases eq_or_ne y 0 with rfl | hy
  · simp [truncP]
  have h1 := Freeze.abs_sub_truncTo_lt (two_zpow_pos (binade |y| - ((p : ℤ) - 1))) y
  have h2 := (binade_spec (abs_pos.2 hy)).1
  have e : (2 : ℚ) ^ (binade |y| - ((p : ℤ) - 1)) = 2 ^ binade |y| / 2 ^ (p - 1) := by
    rw [zpow_sub₀ two_ne_zero, show (p : ℤ) - 1 = ((p - 1 : ℕ) : ℤ) by omega, zpow_natCast]
  have h3 : (2 : ℚ) ^ binade |y| / 2 ^ (p - 1) ≤ |y| / 2 ^ (p - 1) := div_le_div_of_nonneg_right h2 (by positivity)
  have h4 : (2 : ℚ) ^ (binade |y| - ((p : ℤ) - 1)) ≤ |y| / 2 ^ (p - 1) := by rw [e]; exact h3
  rw [truncP, ite_eq_right hy, abs_sub_comm]
  linarith

theorem floor_shiftW (w n L : ℕ) : ⌊(n : ℚ) * 2 ^ ((w : ℤ) - 1 - L)⌋₊ =
    (if w < L + 1 then n >>> (L + 1 - w) else n <<< (w - (L + 1))) := by
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, ← Nat.floor_div_eq_div (K := ℚ)]
    congr 1
    push_cast
    rw [show (w : ℤ) - 1 - L = -((L + 1 - w : ℕ) : ℤ) by omega, zpow_neg, zpow_natCast, div_eq_mul_inv]
  · rw [Nat.shiftLeft_eq, show (w : ℤ) - 1 - L = ((w - (L + 1) : ℕ) : ℤ) by omega, zpow_natCast]
    exact_mod_cast Nat.floor_natCast (n * 2 ^ (w - (L + 1)))

theorem fromAdder_floor (w N L : ℕ) :
    (fromAdder w (if w < L + 1 then N >>> (L + 1 - w) else N <<< (w - (L + 1))) : ℚ) =
      (⌊(N : ℚ) * 2 ^ (((min w 24 : ℕ) : ℤ) - 1 - L)⌋₊ : ℚ) * 2 ^ (24 - min w 24) := by
  rw [← floor_shiftW]
  unfold fromAdder
  split_ifs with h
  · rw [Nat.shiftRight_eq_div_pow, ← Nat.floor_div_natCast, min_eq_right h, Nat.sub_self, pow_zero, mul_one]
    congr 2
    push_cast
    rw [div_eq_mul_inv, mul_assoc, ← zpow_natCast, ← zpow_neg, ← zpow_add₀ two_ne_zero]
    rw [show (w : ℤ) - 1 - L + -((w - 24 : ℕ) : ℤ) = 23 - L by omega]
  · rw [Nat.shiftLeft_eq, min_eq_left (by omega)]; push_cast; ring

/-- **`normalize` at any width `w ≥ 1`**, on a nonzero sum below `2^(w+7)` at an exponent of at least `w − 127`: a
normal term, the sum on the grid `2^(E − w + 1)` truncated toward zero to `min w 24` significant bits (the adder's `w`,
then `fromAdder`'s return to FP32's 24). -/
theorem normalizeW_spec {w : ℕ} (hw : 1 ≤ w) {E total : ℤ} (h0 : total ≠ 0) (hE1 : (w : ℤ) - 127 ≤ E)
    (hT : total.natAbs < 2 ^ (w + 7)) :
    2 ^ 23 ≤ (normalize w E total).mag ∧ (normalize w E total).mag < 2 ^ 24 ∧
      -126 ≤ (normalize w E total).exp ∧ (normalize w E total).exp ≤ E + 7 ∧
      (normalize w E total).value = truncP (min w 24) ((total : ℚ) * 2 ^ (E - w + 1)) := by
  have hn : total.natAbs ≠ 0 := by omega
  have hL1 := Nat.log2_self_le hn
  have hL2 := @Nat.lt_log2_self total.natAbs
  have hL : total.natAbs.log2 < w + 7 := by
    by_contra h
    have : 2 ^ (w + 7) ≤ 2 ^ total.natAbs.log2 := Nat.pow_le_pow_right (by norm_num) (by omega)
    omega
  have hfl := fromAdder_floor w total.natAbs total.natAbs.log2
  set L := total.natAbs.log2 with hLdef
  set m0 := (if w < L + 1 then total.natAbs >>> (L + 1 - w) else total.natAbs <<< (w - (L + 1))) with hm0
  have hb1 : 2 ^ (w - 1) ≤ m0 ∧ m0 < 2 ^ w := by
    rw [hm0]; split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow]
      constructor
      · rw [Nat.le_div_iff_mul_le (by positivity), ← pow_add]; rwa [show w - 1 + (L + 1 - w) = L by omega]
      · rw [Nat.div_lt_iff_lt_mul (by positivity), ← pow_add]; rwa [show w + (L + 1 - w) = L + 1 by omega]
    · rw [Nat.shiftLeft_eq]
      constructor
      · calc 2 ^ (w - 1) = 2 ^ L * 2 ^ (w - (L + 1)) := by rw [← pow_add]; congr 1; omega
          _ ≤ _ := Nat.mul_le_mul_right _ hL1
      · calc total.natAbs * 2 ^ (w - (L + 1)) < 2 ^ (L + 1) * 2 ^ (w - (L + 1)) :=
              Nat.mul_lt_mul_of_pos_right hL2 (by positivity)
          _ = 2 ^ w := by rw [← pow_add]; congr 1; omega
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
  have hbl : bitLength total.natAbs = L + 1 := by unfold bitLength; rw [ite_eq_right hn]
  have hne : ¬ (E + ((L + 1 : ℕ) : ℤ) - (w : ℤ) < -126) := by omega
  unfold normalize
  simp only [h0, ite_false, hbl, hne, ← hm0]
  rw [ite_eq_right (by omega : fromAdder w m0 ≠ 0)]
  dsimp only
  refine ⟨hfa.1, hfa.2, by omega, by omega, ?_⟩
  -- the value
  have hy0 : (total : ℚ) * 2 ^ (E - w + 1) ≠ 0 := mul_ne_zero (by exact_mod_cast h0) (two_zpow_ne _)
  have habs : |(total : ℚ) * 2 ^ (E - w + 1)| = (total.natAbs : ℚ) * 2 ^ (E - w + 1) := by
    rw [abs_mul, abs_of_pos (two_zpow_pos _), Nat.cast_natAbs, Int.cast_abs]
  have hbin : binade |(total : ℚ) * 2 ^ (E - w + 1)| = L + (E - w + 1) := by
    rw [habs, binade_nat_mul hn]
  have hmin : 1 ≤ min w 24 := le_min hw (by norm_num)
  unfold truncP truncTo Term.value
  rw [ite_eq_right hy0, hbin, habs]
  have hdiv : (total.natAbs : ℚ) * 2 ^ (E - w + 1) / 2 ^ ((L : ℤ) + (E - w + 1) - ((min w 24 : ℕ) - 1)) =
      (total.natAbs : ℚ) * 2 ^ (((min w 24 : ℕ) : ℤ) - 1 - L) := by
    rw [mul_div_assoc, ← zpow_sub₀ two_ne_zero]; congr 2; ring
  rw [hdiv, ← Int.natCast_floor_eq_floor (mul_nonneg (Nat.cast_nonneg _) (two_zpow_pos _).le)]
  have hs : (total : ℚ) * 2 ^ (E - w + 1) < 0 ↔ total < 0 := by
    rw [mul_neg_iff]; constructor
    · rintro (⟨-, h⟩ | ⟨h, -⟩)
      · exact absurd h (not_lt.2 (two_zpow_pos _).le)
      · exact_mod_cast h
    · intro h; right; exact ⟨by exact_mod_cast h, two_zpow_pos _⟩
  have hpow : (2 : ℚ) ^ (24 - min w 24) * 2 ^ (E + ((L + 1 : ℕ) : ℤ) - (w : ℤ) - 23) =
      2 ^ ((L : ℤ) + (E - w + 1) - ((min w 24 : ℕ) - 1)) := by
    rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; omega
  have hm' : ((fromAdder w m0 : ℕ) : ℚ) = _ := hfl
  push_cast at hm' hpow ⊢
  rw [hm']
  by_cases hneg : total < 0
  · simp only [hneg, decide_true, ite_true, hs.2 hneg]
    rw [← hpow]; ring
  · simp only [hneg, decide_false, ite_false, Bool.false_eq_true, (not_congr hs).2 hneg]
    rw [← hpow]; ring

/-! ## The step of one group of `n` products at width `w` -/

/-- The step's alignment exponent on the accumulator term `tc` and `n` code pairs: the largest exponent over `tc` and
the nonzero products, at least the floor `z` (`Freeze.stepExp` at `n = 32`, `z = −139`). -/
def stepExpP (z : ℤ) {n : ℕ} (tc : Term) (a b : Fin n → Code) : ℤ :=
  maxExp z (tc :: List.ofFn fun l => product (a l) (b l))

theorem zip_ofFnN {n : ℕ} (a b : Fin n → Code) :
    List.zip (List.ofFn a) (List.ofFn b) = List.ofFn fun l => (a l, b l) := by
  apply List.ext_getElem <;> simp

/-- The step of `⟨[n], w, z⟩` on a finite accumulator term `tc` and finite codes: one group of `tc` and the `n`
products (`Freeze.hopperStep_fin` at `⟨[32], 14, −139⟩`). -/
theorem pipeStep_fin {n w : ℕ} {z : ℤ} {acc : Nat} {tc : Term} (hw : fp32Value acc = .fin tc) (a b : Fin n → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    (⟨[n], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b) = packValue
      (let t := groupSum w z (tc :: List.ofFn fun l => product (a l) (b l))
       if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t) := by
  unfold Pipeline.step
  simp only [Pipeline.fold, hw, zip_ofFnN]
  congr 1
  rw [List.take_of_length_le (by simp), List.map_ofFn]
  have hp : ((fun x : Code × Code => productTotal x.1 x.2) ∘ fun l => (a l, b l)) =
      fun l => Value.fin (product (a l) (b l)) := by
    funext l; simp [productTotal, ha l, hb l]
  rw [hp]
  unfold groupStepTotal
  have h1 : ((Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).any Value.isNaN) = false := by
    rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isNaN]
  have h2 : ∀ s, ((Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).any (Value.isInf s)) =
      false := by
    intro s; rw [List.any_eq_false]; intro v hv
    simp only [List.mem_cons, List.mem_ofFn] at hv
    rcases hv with rfl | ⟨l, rfl⟩ <;> simp [Value.isInf]
  have h3 : (Value.fin tc :: List.ofFn fun l => Value.fin (product (a l) (b l))).filterMap Value.finite? =
      tc :: List.ofFn fun l => product (a l) (b l) := by
    rw [List.filterMap_cons, List.ofFn_eq_map, List.ofFn_eq_map, List.filterMap_map]
    simp [Value.finite?, Function.comp_def]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.and_self]

/-- **The step's value at width `w`.** On a finite accumulator term `tc` and `n ≤ 32` finite code pairs, with the
step's exponent `E` in `[w − 127, 120]`, the step writes a finite word whose value is two truncations toward zero:
the accumulator and each product to the grid `g = 2^(E − w + 1)`, then their sum to `min w 24` significant bits. -/
theorem pipeStep_value {n w : ℕ} {z : ℤ} (hn : n ≤ 32) (hw1 : 1 ≤ w) {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) (a b : Fin n → Code) (ha : ∀ l, (a l).isNaN = false)
    (hb : ∀ l, (b l).isNaN = false) (hE1 : (w : ℤ) - 127 ≤ stepExpP z tc a b) (hE2 : stepExpP z tc a b ≤ 120) :
    wordQ ((⟨[n], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b)) =
      some (truncP (min w 24) (truncTo (2 ^ (stepExpP z tc a b - w + 1)) tc.value +
        ∑ l, truncTo (2 ^ (stepExpP z tc a b - w + 1)) (val (a l) * val (b l)))) := by
  obtain ⟨hm2, -⟩ := Freeze.fp32Value_fin hw
  set E := stepExpP z tc a b with hEdef
  have hcE : tc.mag ≠ 0 → tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  set total := alignedSum w E (tc :: List.ofFn fun l => product (a l) (b l)) with htdef
  rw [pipeStep_fin hw a b ha hb]
  unfold groupSum
  change wordQ (packValue (let t := normalize w E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t)) = _
  have hsum : total = aligned w E tc + ∑ l, aligned w E (product (a l) (b l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have htot : (total : ℚ) * 2 ^ (E - w + 1) =
      truncTo (2 ^ (E - w + 1)) tc.value + ∑ l, truncTo (2 ^ (E - w + 1)) (val (a l) * val (b l)) := by
    rw [hsum]; push_cast; rw [add_mul, Finset.sum_mul, alignedW_value hcE, add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [alignedW_value (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  have hT : total.natAbs < 2 ^ (w + 7) := by
    obtain ⟨B, hB⟩ : ∃ B : ℕ, B = 2 ^ (w + 1) := ⟨_, rfl⟩
    have h64 : 2 ^ (w + 7) = 64 * B := by rw [hB, show w + 7 = 6 + (w + 1) by omega, pow_add]; norm_num
    have hc : (aligned w E tc).natAbs < B := hB ▸ alignedW_natAbs_lt w E tc (by omega)
    have hp : ∀ l, (aligned w E (product (a l) (b l))).natAbs < B := fun l =>
      hB ▸ alignedW_natAbs_lt w E _ (product_mag_lt _ _)
    have h1 : |∑ l : Fin n, aligned w E (product (a l) (b l))| ≤
        ∑ l : Fin n, |aligned w E (product (a l) (b l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin n, |aligned w E (product (a l) (b l))| ≤ ∑ _l : Fin n, (B : ℤ) :=
      Finset.sum_le_sum fun l _ => by have := hp l; rw [Int.abs_eq_natAbs]; omega
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    have hnB : (n : ℤ) * B ≤ 32 * B := mul_le_mul_of_nonneg_right (by exact_mod_cast hn) (by positivity)
    rw [h64, hsum]
    rw [Int.abs_eq_natAbs] at h1
    omega
  clear_value total E
  rw [← htot]
  by_cases h0 : total = 0
  · rw [h0]
    simp [normalize, Term.zero, packValue, packTerm, Freeze.wordQ_zero, truncP]
  obtain ⟨m1, m2, e1, e3, hv⟩ := normalizeW_spec hw1 h0 hE1 hT
  have e2 : (normalize w E total).exp ≤ 127 := by omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (by omega)), packValue, wordQ_packTerm m1 m2 e1 e2, hv]

/-- **The step's error at width `w`** (`AtomErr.hopperStep_err` at `n = 32`, `w = 14`). Under `pipeStep_value`'s
hypotheses, the value the cast reads (`wordVal`) is within `(n + 1)·g + (|S| + (n + 1)·g)/2^(min w 24 − 1)` of the exact
sum `S = c + Σ_l a_l·b_l`, for the grid `g = 2^(E − w + 1)`: `(n + 1)·g` for the first truncation of the accumulator and
the `n` products, and the rest for the second, to `min w 24` significant bits. -/
theorem pipeStep_err {n w : ℕ} {z : ℤ} (hn : n ≤ 32) (hw1 : 1 ≤ w) {acc : Nat} {tc : Term}
    (hw : fp32Value acc = .fin tc) (a b : Fin n → Code) (ha : ∀ l, (a l).isNaN = false)
    (hb : ∀ l, (b l).isNaN = false) (hE1 : (w : ℤ) - 127 ≤ stepExpP z tc a b) (hE2 : stepExpP z tc a b ≤ 120) :
    |PearlC.wordVal ((⟨[n], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b)) -
        (tc.value + ∑ l, val (a l) * val (b l))| ≤
      (n + 1) * 2 ^ (stepExpP z tc a b - w + 1) +
        (|tc.value + ∑ l, val (a l) * val (b l)| + (n + 1) * 2 ^ (stepExpP z tc a b - w + 1)) /
          2 ^ (min w 24 - 1) := by
  have hv := pipeStep_value hn hw1 hw a b ha hb hE1 hE2
  set g : ℚ := 2 ^ (stepExpP z tc a b - w + 1) with hg_def
  have hg : 0 < g := two_zpow_pos _
  set S := tc.value + ∑ l, val (a l) * val (b l) with hS
  set T := truncTo g tc.value + ∑ l, truncTo g (val (a l) * val (b l)) with hT
  have hwv : PearlC.wordVal ((⟨[n], w, z⟩ : Pipeline).step acc (List.ofFn a) (List.ofFn b)) = truncP (min w 24) T := by
    unfold PearlC.wordVal; rw [hv]
  have hTS : |T - S| ≤ (n + 1) * g := by
    have e : T - S = (truncTo g tc.value - tc.value) +
        ∑ l, (truncTo g (val (a l) * val (b l)) - val (a l) * val (b l)) := by
      rw [hT, hS, Finset.sum_sub_distrib]; ring
    have h1 : |truncTo g tc.value - tc.value| ≤ g := by
      rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
    have h2 : |∑ l, (truncTo g (val (a l) * val (b l)) - val (a l) * val (b l))| ≤ n * g := by
      calc _ ≤ ∑ l, |truncTo g (val (a l) * val (b l)) - val (a l) * val (b l)| := Finset.abs_sum_le_sum_abs _ _
        _ ≤ ∑ _l : Fin n, g := Finset.sum_le_sum fun l _ => by
            rw [abs_sub_comm]; exact (Freeze.abs_sub_truncTo_lt hg _).le
        _ = n * g := by simp
    rw [e]
    obtain ⟨a1, a2⟩ := abs_le.1 h1
    obtain ⟨b1, b2⟩ := abs_le.1 h2
    exact abs_le.2 ⟨by linarith, by linarith⟩
  have hP := abs_truncP_sub_le (le_min hw1 (by norm_num) : 1 ≤ min w 24) T
  have hTa : |T| ≤ |S| + (n + 1) * g := by
    have := abs_sub_abs_le_abs_sub T S; linarith
  rw [hwv]
  obtain ⟨a1, a2⟩ := abs_le.1 hP
  obtain ⟨b1, b2⟩ := abs_le.1 hTS
  have h3 : |T| / 2 ^ (min w 24 - 1) ≤ (|S| + (n + 1) * g) / 2 ^ (min w 24 - 1) :=
    div_le_div_of_nonneg_right hTa (by positivity)
  exact abs_le.2 ⟨by linarith, by linarith⟩

/-- **The instance at `w = 14`**: `pipeStep_err` at `hopperE4m3K32` is `AtomErr.hopperStep_err`, word for word. -/
theorem hopperStep_errP {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (hE1 : -113 ≤ Freeze.stepExp tc a b)
    (hE2 : Freeze.stepExp tc a b ≤ 120) :
    |PearlC.wordVal (hopperStep w a b) - (tc.value + ∑ l, val (a l) * val (b l))| ≤
      33 * 2 ^ (Freeze.stepExp tc a b - 13) +
        (|tc.value + ∑ l, val (a l) * val (b l)| + 33 * 2 ^ (Freeze.stepExp tc a b - 13)) / 2 ^ 13 := by
  have h := pipeStep_err (n := 32) (w := 14) (z := -139) (by norm_num) (by norm_num) hw a b ha hb
    (by change _ ≤ Freeze.stepExp tc a b; push_cast; omega) hE2
  have e : Freeze.stepExp tc a b - ((14 : ℕ) : ℤ) + 1 = Freeze.stepExp tc a b - 13 := by push_cast; ring
  change |PearlC.wordVal (hopperStep w a b) - _| ≤ _ at h
  rw [show stepExpP (-139) tc a b = Freeze.stepExp tc a b from rfl, e] at h
  norm_num at h ⊢
  exact h

/-- **The instance at `w = 26`** (`sm120StepAt z`, any floor `z`): within `33·g + (|S| + 33·g)·2^−23` of `S`,
for `g = 2^(E − 25)` and `E` in `[−101, 120]`. -/
theorem sm120Step_err {z : ℤ} {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) (hE1 : -101 ≤ stepExpP z tc a b)
    (hE2 : stepExpP z tc a b ≤ 120) :
    |PearlC.wordVal (sm120StepAt z w a b) -
        (tc.value + ∑ l, val (a l) * val (b l))| ≤
      33 * 2 ^ (stepExpP z tc a b - 25) +
        (|tc.value + ∑ l, val (a l) * val (b l)| + 33 * 2 ^ (stepExpP z tc a b - 25)) / 2 ^ 23 := by
  unfold sm120StepAt sm120E4m3K32
  have h := pipeStep_err (n := 32) (w := 26) (by norm_num) (by norm_num) hw a b ha hb (by push_cast; omega) hE2
  have e : stepExpP z tc a b - ((26 : ℕ) : ℤ) + 1 = stepExpP z tc a b - 25 := by push_cast; ring
  rw [e] at h
  norm_num at h ⊢
  exact h

/-! ## sm_120's atom-mix on flat pairs (`sm120E4m3K32 z`, any floor `z ≤ −5`) -/

/-- `2^E ≤ B` for the step's exponent `E` at floor `z`, when `2^z ≤ B` and `2^e ≤ B` for the accumulator and every
nonzero product (`AtomErr.two_zpow_stepExp_le` at `z = −139`). -/
theorem two_zpow_stepExpP_le {z : ℤ} (tc : Term) (a b : Fin 32 → Code) {B : ℚ} (hz : (2 : ℚ) ^ z ≤ B)
    (hc : tc.mag ≠ 0 → (2 : ℚ) ^ tc.exp ≤ B)
    (hp : ∀ l, (product (a l) (b l)).mag ≠ 0 → (2 : ℚ) ^ (product (a l) (b l)).exp ≤ B) :
    (2 : ℚ) ^ stepExpP z tc a b ≤ B := by
  unfold stepExpP
  rcases maxExp_cases z (tc :: List.ofFn fun l => product (a l) (b l)) with e | ⟨t, ht, hm, e⟩ <;> rw [e]
  · exact hz
  · rcases List.mem_cons.1 ht with rfl | ht
    · exact hc hm
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      exact hp l hm

/-- **sm_120's atom-mix on a flat pair.** For any floor `z ≤ −5` (sm_120's `−133`, or anywhere in `[−150, −124]`),
`0 ≤ α`, `α·|x| ≤ 448` and a nonzero noise `N = flatNoise β s'`, the value the cast reads on the pair
`(flat (fun _ => false), flatF s')` is within `(α·|x| + |N|)/2^18` of `α·x + N` (`AtomErr.atomMix_flat_err` has `/64`). -/
theorem atomMixP_flat_err {z : ℤ} (hz : z ≤ -5) {α x β : ℚ} (s' : Fin 32 → Bool) (hα : 0 ≤ α)
    (hax : α * |x| ≤ 448) (hN : flatNoise β s' ≠ 0) :
    |(atomMixP (sm120E4m3K32 z)).value α x β (flat fun _ => false) (flatF s') - (α * x + flatNoise β s')| ≤
      1 / 2 ^ 18 * (α * |x| + |flatNoise β s'|) := by
  have hy : |α * x| = α * |x| := by rw [abs_mul, abs_of_nonneg hα]
  have hA := mul_nonneg hα (abs_nonneg x)
  obtain ⟨tc, hw, hc⟩ := f32Word_spec (y := α * x) (by rw [hy]; norm_num; linarith)
  have hV0 : val (lineCode β) ≠ 0 := fun h => hN (by rw [flatNoise, codeVal, ← val, h, zero_mul])
  have hK0 : ∑ t, codeVal (flatF s' t) ≠ 0 := fun h => hN (by rw [flatNoise, h, mul_zero])
  obtain ⟨hLn, hV512, hLs⟩ := val_ne_zero_spec _ hV0
  obtain ⟨a, ha_def⟩ : ∃ a : Fin 32 → Code, a = fun _ => lineCode β := ⟨_, rfl⟩
  obtain ⟨b, hb_def⟩ : ∃ b : Fin 32 → Code, b = fun t => flatF s' t := ⟨_, rfl⟩
  have hal : ∀ l, a l = lineCode β := fun l => by rw [ha_def]
  have hbl : ∀ l, b l = flatF s' l := fun l => by rw [hb_def]
  have hval : (atomMixP (sm120E4m3K32 z)).value α x β (flat fun _ => false) (flatF s') =
      PearlC.wordVal (sm120StepAt z (f32Word (α * x)) a b) := by
    have hla : lineCodes β (flat fun _ => false : Fin 32 → Code) = a := by
      rw [ha_def]; funext t; exact lineCodes_flat β t
    rw [hb_def, ← hla]; rfl
  rw [hval]
  have ha : ∀ l, (a l).isNaN = false := fun l => by rw [hal]; exact hLn
  have hb : ∀ l, (b l).isNaN = false := fun l => by rw [hbl]; exact (e4m3_flat s' l).1
  have hpm : ∀ l, (product (a l) (b l)).mag ≠ 0 := by
    intro l
    rw [hal, hbl]
    have h1 : ((lineCode β).sig * (flatF s' l).sig) <<< 17 ≠ 0 := by
      rw [Nat.shiftLeft_eq, (e4m3_flat s' l).2]
      exact Nat.mul_ne_zero (Nat.mul_ne_zero (by omega) (by norm_num)) (by positivity)
    unfold product; dsimp only; rw [ite_eq_right h1]; exact h1
  have z150 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have z126 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have z120 : (2 : ℚ) ^ (120 : ℤ) = 2 ^ 120 := by norm_num
  have hc0 : |tc.value - α * x| ≤ α * |x| / 2 ^ 24 + 1 / 2 ^ 150 := by
    rw [hy, z150] at hc
    refine le_trans hc (max_le ?_ ?_)
    · have : (0 : ℚ) < 1 / 2 ^ 150 := by norm_num
      linarith
    · have : 0 ≤ α * |x| / 2 ^ 24 := by positivity
      linarith
  have hTv : |tc.value| ≤ α * |x| + |tc.value - α * x| := by
    have := abs_sub_abs_le_abs_sub tc.value (α * x); rw [hy] at this; linarith
  have hE1 : -101 ≤ stepExpP z tc a b := by
    have h := le_maxExp z (tc :: List.ofFn fun l => product (a l) (b l)) (product (a 0) (b 0))
      (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨0, rfl⟩)) (hpm 0)
    have := (product_exp _ _ (hpm 0)).1
    unfold stepExpP; omega
  have hE2 : stepExpP z tc a b ≤ 120 := by
    refine maxExp_le z 120 _ (by omega) fun t ht htm => ?_
    rcases List.mem_cons.1 ht with rfl | ht
    · have h2 := two_zpow_acc_le hw htm
      rw [z126] at h2
      have h3 : (2 : ℚ) ^ t.exp < 2 ^ (120 : ℤ) := by rw [z120]; linarith
      exact ((zpow_lt_zpow_iff_right₀ (one_lt_two (α := ℚ))).1 h3).le
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      exact le_trans (product_exp _ _ htm).2 (by norm_num)
  have herr := sm120Step_err hw a b ha hb hE1 hE2
  have hsum : ∑ l, val (a l) * val (b l) = flatNoise β s' := by
    rw [flatNoise, codeVal, ← val, Finset.mul_sum]
    exact Finset.sum_congr rfl fun l _ => by rw [hal, hbl]; rfl
  rw [hsum] at herr
  have hNabs : |flatNoise β s'| = |val (lineCode β)| * |∑ t, codeVal (flatF s' t)| := by
    rw [flatNoise, codeVal, ← val, abs_mul]
  have hK88 := abs_sum_flat_ge s' hK0
  have hVN : 11 / 2 * |val (lineCode β)| ≤ |flatNoise β s'| := by
    rw [hNabs, mul_comm]; exact mul_le_mul_of_nonneg_left hK88 (abs_nonneg _)
  have hG : (2 : ℚ) ^ stepExpP z tc a b ≤ |tc.value| + 1 / 2 ^ 126 + 16 * |val (lineCode β)| := by
    have hV := abs_nonneg (val (lineCode β))
    have hT := abs_nonneg tc.value
    have hz2 : (2 : ℚ) ^ z ≤ 1 / 32 :=
      calc (2 : ℚ) ^ z ≤ 2 ^ (-5 : ℤ) := zpow_le_zpow_right₀ one_le_two hz
        _ = 1 / 32 := by norm_num
    refine two_zpow_stepExpP_le tc a b (by linarith) (fun hm => by
      have := two_zpow_acc_le hw hm; rw [z126] at this; linarith) fun l hm => ?_
    rw [hal, hbl] at hm ⊢
    have := product_flat_two_zpow hLn hLs s' l hm
    linarith
  have hg : (2 : ℚ) ^ (stepExpP z tc a b - 25) = 2 ^ stepExpP z tc a b / 2 ^ 25 := by
    rw [zpow_sub₀ two_ne_zero]; norm_num
  rw [hg] at herr
  have hS : |tc.value + flatNoise β s'| ≤ |tc.value| + |flatNoise β s'| := abs_add_le _ _
  have htri := abs_sub_le (PearlC.wordVal (sm120StepAt z (f32Word (α * x)) a b)) (tc.value + flatNoise β s')
    (α * x + flatNoise β s')
  have e : tc.value + flatNoise β s' - (α * x + flatNoise β s') = tc.value - α * x := by ring
  rw [e] at htri
  linarith

/-- **`PairErr` for sm_120's atom-mix at `ε = 2^−18`.** On Pearl's flat pairs `(flat (fun _ => false), flatF ¬s')` and
`(flat (fun _ => false), flatF s')`, for any floor `z ≤ −5`, with `0 ≤ α`, `α·|x| ≤ 448` and `N = flatNoise β s'`
positive (`AtomErr.atomMix_pairErr` has `ε = 1/64`). -/
theorem atomMixP_pairErr {z : ℤ} (hz : z ≤ -5) {α x β : ℚ} (s' : Fin 32 → Bool) (hα : 0 ≤ α)
    (hax : α * |x| ≤ 448) (hN : 0 < flatNoise β s') :
    PairErr (atomMixP (sm120E4m3K32 z)).value α x β (flatNoise β s') (1 / 2 ^ 18)
      (flat fun _ => false, flatF fun t => !s' t) (flat fun _ => false, flatF s') := by
  have hN' := flatNoise_not β s'
  constructor
  · have h := atomMixP_flat_err hz (β := β) (fun t => !s' t) hα hax (by rw [hN']; exact neg_ne_zero.2 hN.ne')
    rw [hN', abs_neg, abs_of_pos hN, ← sub_eq_add_neg] at h
    exact h
  · have h := atomMixP_flat_err hz (β := β) s' hα hax hN.ne'
    rw [abs_of_pos hN] at h
    exact h

/-- **Salt-liveness on sm_120's atom-mix, flat pairs** (`AtomErr.atomMixChain_saltLive_flat` at `ε = 2^−18`). With
`FlatPairs S`, the noise floor `ασ ≥ 1` and a sign pattern `s'` whose flat-pair noise is `c·ασ` with `8/15 ≤ c ≤ 4`, the
element is salt-live when `(1 + 17/2^18)·|x| < (15 − 17/2^18)·c·σ`, about `|x| < 14.9990·c·σ`. -/
theorem atomMixChainP_saltLive_flat {z : ℤ} (hz : z ≤ -5) {S : Support 32} (hS : FlatPairs S) {α x β σ c : ℚ}
    (s' : Fin 32 → Bool) (hN : flatNoise β s' = c * (α * σ)) (hα : 0 < α) (hfloor : 1 ≤ α * σ)
    (hsat : α * (|x| + 4 * σ) ≤ 448) (hc : 8 / 15 ≤ c) (hc4 : c ≤ 4)
    (hx : (1 + 17 * (1 / 2 ^ 18)) * |x| < (15 - 17 * (1 / 2 ^ 18)) * c * σ) :
    SaltLiveVal (atomMixP (sm120E4m3K32 z)).value S α x β := by
  have hax : α * |x| ≤ 448 := by
    have : α * (|x| + 4 * σ) = α * |x| + 4 * (α * σ) := by ring
    linarith
  have herr := atomMixP_pairErr hz (β := β) s' hα.le hax (by rw [hN]; exact mul_pos (by linarith) (by linarith))
  rw [hN] at herr
  exact saltLive_rel (hS _ _) (hS _ _) herr hα hfloor hsat hc hc4 (by norm_num) (by norm_num) hx

/-- **ρ_D ≤ 1/64 on sm_120's atom-mix, flat pairs** (`AtomErr.rhoD_atomMix_pi_flat` at `ε = 2^−18`). With
`FlatPairs S`, the noise floor, §5's scale and, per row, a sign pattern `s' i` whose flat-pair noise is `c_i·α_iρ_i` with
`2667/5000 ≤ c_i ≤ 4` (the H100 needs `648/943`): the liveness rule gives the forming budget with `ρ_D = 1/64`. -/
theorem rhoD_atomMixP_flat {z : ℤ} (hz : z ≤ -5) (fs : ℕ) (s : Shape) {S : Support Params.pi.r} (hS : FlatPairs S)
    {α β ρ c : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} (s' : Fin s.m → Fin 32 → Bool)
    (hN : ∀ i, flatNoise (β i) (s' i) = c i * (α i * ρ i)) (hα : ∀ i, 0 < α i) (hfloor : ∀ i, 1 ≤ α i * ρ i)
    (hsat : ∀ i l, α i * (|X i l| + 4 * ρ i) ≤ 448) (hc : ∀ i, 2667 / 5000 ≤ c i) (hc4 : ∀ i, c i ≤ 4)
    (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * s.m * s.k - rhoD * fs * s.m * s.k ≤ fs * (live (atomMixP (sm120E4m3K32 z)).value S α β X).card := by
  refine rhoD_rel_pi fs s (ε := 1 / 2 ^ 18) (c := c)
    (p := fun i => (flat fun _ => false, flatF fun t => !s' i t)) (q := fun i => (flat fun _ => false, flatF (s' i)))
    (fun i => hS _ _) (fun i => hS _ _) (fun i l => ?_) hα hfloor hsat (fun i => by linarith [hc i]) hc4
    (by norm_num) (by norm_num) hlive
  have hax : α i * |X i l| ≤ 448 := by
    have : α i * (|X i l| + 4 * ρ i) = α i * |X i l| + 4 * (α i * ρ i) := by ring
    linarith [hsat i l, hfloor i]
  have h := atomMixP_pairErr hz (β := β i) (s' i) (hα i).le hax
    (by rw [hN i]; exact mul_pos (by linarith [hc i]) (by linarith [hfloor i]))
  rw [hN i] at h
  exact h

/-! ## `saltDeadP` -/

/-- **The link:** `saltDeadP p` is `saltDead` of `atomMixP p` over `SaltDead.support` (not `SaltLiveVal`), for every
step `p`; at `hopperE4m3K32` it is `Forming.saltDead_pi` (`saltDeadP_hopper`, `atomMixP_hopper`). -/
theorem saltDeadP_eq (p : Pipeline) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) :
    saltDeadP p alpha beta = saltDead (atomMixP p).value SaltDead.support alpha beta := by
  funext k x l
  simp only [saltDeadP, saltDead, SaltLiveVal, FormChain.elem, not_exists, not_and, not_not]

open Classical in
/-- `Forming.saltDead_pi_card_le` for any step `p`: given the forming budget on `live` of `atomMixP p` over
`SaltDead.support` at the rows' own scales, at most a `ρ` share of the elements are salt-dead. -/
theorem saltDeadP_card_le_of_live {m k : ℕ} (p : Pipeline) (fs : ℕ) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ)
    (X : Matrix (Fin m) (Fin k) ℚ) {ρ : ℚ}
    (h : (fs : ℚ) * m * k - ρ * fs * m * k ≤
      fs * (live (atomMixP p).value SaltDead.support (fun i => alpha k (X i)) (fun i => beta k (X i)) X).card) :
    (fs : ℚ) * (Finset.univ.filter fun e : Fin m × Fin k => saltDeadP p alpha beta k (X e.1) e.2).card ≤
      ρ * fs * m * k := by
  have hc := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin m × Fin k)))
    (fun e => SaltLiveVal (atomMixP p).value SaltDead.support (alpha k (X e.1)) (X e.1 e.2) (beta k (X e.1)))
  rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at hc
  have hd : (Finset.univ.filter fun e : Fin m × Fin k => saltDeadP p alpha beta k (X e.1) e.2) =
      Finset.univ.filter fun e : Fin m × Fin k =>
        ¬SaltLiveVal (atomMixP p).value SaltDead.support (alpha k (X e.1)) (X e.1 e.2) (beta k (X e.1)) := by
    refine Finset.filter_congr fun e _ => ?_
    rw [saltDeadP_eq]
    exact Iff.rfl
  have hq : ((live (atomMixP p).value SaltDead.support (fun i => alpha k (X i)) (fun i => beta k (X i)) X).card +
      (Finset.univ.filter fun e : Fin m × Fin k => saltDeadP p alpha beta k (X e.1) e.2).card : ℚ) = m * k := by
    rw [hd]
    exact_mod_cast hc
  linear_combination (fs : ℚ) * hq + h

open Classical in
/-- **ρ_D for `saltDeadP` at sm_120** (`sm120E4m3K32 z`, any floor `z ≤ −5`; `sm120E4m3` at `z = −133`): under `rhoD_atomMixP_flat`'s domain
conditions at the rows' own scales, at most `ρ_D·m·k = m·k/64` elements are salt-dead, so the forming debit
`fs·#{saltDeadP ∨ flagged}` (`Forming.formDebit_eq_elems` after `saltDeadP_eq`) stays within `ρ_D` on the salt side. -/
theorem saltDeadP_card_le {z : ℤ} (hz : z ≤ -5) (fs : ℕ) (s : Shape) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ)
    {ρ c : Fin s.m → ℚ} {X : Matrix (Fin s.m) (Fin s.k) ℚ} (s' : Fin s.m → Fin 32 → Bool)
    (hN : ∀ i, flatNoise (beta s.k (X i)) (s' i) = c i * (alpha s.k (X i) * ρ i))
    (hα : ∀ i, 0 < alpha s.k (X i)) (hfloor : ∀ i, 1 ≤ alpha s.k (X i) * ρ i)
    (hsat : ∀ i l, alpha s.k (X i) * (|X i l| + 4 * ρ i) ≤ 448) (hc : ∀ i, 2667 / 5000 ≤ c i)
    (hc4 : ∀ i, c i ≤ 4) (hlive : LiveOperand Params.pi.δ ρ X) :
    (fs : ℚ) * (Finset.univ.filter fun e : Fin s.m × Fin s.k =>
      saltDeadP (sm120E4m3K32 z) alpha beta s.k (X e.1) e.2).card ≤ rhoD * fs * s.m * s.k :=
  saltDeadP_card_le_of_live _ fs alpha beta X
    (rhoD_atomMixP_flat hz fs s flatPairs_support s' hN hα hfloor hsat hc hc4 hlive)

end Pouw.PearlC.FormingP
