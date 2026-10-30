import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import Definitions.Pouw.PearlC.Defs
import Definitions.Pouw.PearlC.Skip
import Proofs.Pouw.Fp8Atom.H1TAtomProofs
import Proofs.Pouw.Fp8Atom.Fp32Proofs

/-!
# P1: freezing under promotion every `G` atoms (staging)

An atom `x` is an **identity** on the word `w` when `hopperStep w x.1 x.2 = w`. Skipping is in place: a skipped atom's
incoming word passes through, a skipped group passes its incoming total through (its promotion is skipped too), and
the groups stay where they are.

`absorbed` flags exactly the atoms whose skip alone leaves the output unchanged (`skipOne_iff`), and `jointFlags`
every atom of a set inside one group whose joint skip does (`joint_complete`); joint skips across groups escape both
(`joint_cross_group`). On a word, atom `t` is group `t / 4`'s atom `t % 4` (`getD_getD_toChunks`), so the word's flags
`Skip.wordFlags` flag every atom skippable with some set of its group's atoms (`joint_flagsComplete`, `Forming`'s
`FlagsComplete` for `WordSkippable`).
-/

namespace Pouw.PearlC.Freeze

open Pouw.Fp8Atom Pouw.Fp8Atom.Proofs Pouw.Fp8Atom.H1T Pouw.Fp8Atom.H1T.Proofs Pouw.PearlC Pouw.PearlC.Skip

/-! ## Two counterexamples to the task's wording -/

/-- Lanes `0` and `1` hold `x` and `y`, the other 30 hold `z`. -/
def lanes (x y z : Code) : Fin 32 → Code := fun i => if i.val = 0 then x else if i.val = 1 then y else z

/-- From `+0`, an identity whose exact product sum is not zero: `448² − 448² + 30·2^−18`, and the grid is `2^3`. -/
theorem first_identity_nonzero_sum :
    hopperStep 0 (lanes 0x7E 0xFE 0x01) (lanes 0x7E 0x7E 0x01) = 0 ∧
      (List.ofFn fun l => val (lanes 0x7E 0xFE 0x01 l) * val (lanes 0x7E 0x7E 0x01 l)).sum = 30 / 2 ^ 18 := by
  decide +kernel

/-- One lane `x`, the others `+0`. -/
def lane0 (x : Code) : Fin 32 → Code := lanes x 0 0

/-! ## (b1) The first atom of a group -/

/-- **(b1)** From `+0`, on finite codes, the atom is an identity exactly when the products truncated to the slice's
grid sum to zero (`preSum`), which `first_identity_nonzero_sum` shows is weaker than an exact product sum of zero. -/
theorem hopperStep_zero_iff (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    hopperStep 0 a b = 0 ↔ preSum a b = 0 := by
  constructor
  · intro h
    have hv := h1tAtomValue a b ha hb
    rw [h] at hv
    have h0 : wordQ 0 = some 0 := by simp [wordQ, fp32Value_zero, Value.finite?, Term.value, Term.zero]
    rw [h0, Option.some.injEq] at hv
    exact trunc14_eq_zero hv.symm
  · intro h
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
    have h0 : alignedSum 14 (gridExp a b) (Term.zero :: List.ofFn fun l => product (a l) (b l)) = 0 := by
      rw [h] at htot
      exact_mod_cast (mul_eq_zero.1 htot).resolve_right (two_zpow_ne _)
    rw [h0]
    simp [normalize, Term.zero, packValue, packTerm]

/-! ## (a) An identity needs domination -/

/-- The step's alignment exponent on the accumulator term `tc`: the largest exponent over `tc` and the nonzero
products, at least −139. -/
def stepExp (tc : Term) (a b : Fin 32 → Code) : ℤ := maxExp (-139) (tc :: List.ofFn fun l => product (a l) (b l))

theorem fp32Value_fin {w : Nat} {t : Term} (h : fp32Value w = .fin t) : t.mag < 2 ^ 24 ∧ t.exp ≤ 127 := by
  unfold fp32Value at h
  dsimp only at h
  have hm : w % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  have he : (w >>> 23) % 256 < 256 := Nat.mod_lt _ (by norm_num)
  split_ifs at h <;> cases h
  · exact ⟨by simp [Term.zero], by simp [Term.zero]⟩
  · dsimp only; constructor <;> omega
  · dsimp only
    exact ⟨Nat.or_lt_two_pow (by decide) (by omega), by omega⟩

/-- The step on a finite accumulator term `tc`, on finite codes: one group of `tc` and the 32 products. -/
theorem hopperStep_fin {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (a b : Fin 32 → Code)
    (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false) :
    hopperStep w a b = packValue
      (let t := groupSum 14 (-139) (tc :: List.ofFn fun l => product (a l) (b l))
       if t.mag ≠ 0 ∧ 127 < t.exp then .inf t.neg else .fin t) := by
  unfold hopperStep Pipeline.step hopperE4m3K32
  simp only [Pipeline.fold, hw, zip_ofFn]
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
    simp [Value.finite?]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.and_self]

/-- `normalize` on a nonzero sum below `2^21` at an exponent of at least −113: a normal term (it may pass 127), the
sum truncated toward zero to 14 significant bits. -/
theorem normalize_spec' {E total : ℤ} (h0 : total ≠ 0) (hE1 : -113 ≤ E) (hT : total.natAbs < 2 ^ 21) :
    2 ^ 23 ≤ (normalize 14 E total).mag ∧ (normalize 14 E total).mag < 2 ^ 24 ∧
      -126 ≤ (normalize 14 E total).exp ∧ E - 13 ≤ (normalize 14 E total).exp ∧
      (normalize 14 E total).exp ≤ E + 7 ∧
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
  refine ⟨by omega, by omega, by omega, by omega, by omega, ?_⟩
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

theorem truncZ_intCast (n : ℤ) : truncZ (n : ℚ) = n := by
  unfold truncZ
  split_ifs with h
  · have h' : 0 ≤ n := by exact_mod_cast h
    obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le h'
    rw [Int.cast_natCast, Nat.floor_natCast]
  · have h' : n < 0 := by exact_mod_cast not_le.1 h
    obtain ⟨k, hk⟩ := Int.eq_ofNat_of_zero_le (show 0 ≤ -n by omega)
    rw [← Int.cast_neg, hk, Int.cast_natCast, Nat.floor_natCast]
    omega

theorem truncTo_intMul {g : ℚ} (hg : 0 < g) (n : ℤ) : truncTo g (n * g) = n * g := by
  rw [truncTo_mul hg, truncZ_intCast]

theorem binade_term {t : Term} (h1 : 2 ^ 23 ≤ t.mag) (h2 : t.mag < 2 ^ 24) : binade |t.value| = t.exp := by
  have habs : |t.value| = (t.mag : ℚ) * 2 ^ (t.exp - 23) := by
    unfold Term.value
    split <;> simp [abs_mul, abs_of_pos (two_zpow_pos _)]
  have hn : t.mag ≠ 0 := by omega
  have hL : t.mag.log2 = 23 := (Nat.log2_eq_iff hn).2 ⟨h1, h2⟩
  rw [habs, binade_nat_mul hn, hL]; push_cast; ring

theorem binade_trunc14 {y : ℚ} (hy : y ≠ 0) : binade |trunc14 y| = binade |y| := by
  rw [trunc14_abs, trunc14_pos (abs_pos.2 hy), binade_tpos (abs_pos.2 hy)]

/-- **(a) An identity needs domination.** On finite codes and a normal accumulator `w` of value `c` and exponent at
least −113, `D = w` only if `c` lies on the step's grid `g = 2^(E − 13)` and the products truncated toward zero to
`g` sum to zero (`E = stepExp`, the largest exponent over the accumulator and the products). -/
theorem identity_needs_domination {w : Nat} {tc : Term} (hw : fp32Value w = .fin tc) (hm : 2 ^ 23 ≤ tc.mag)
    (he : -113 ≤ tc.exp) (a b : Fin 32 → Code) (ha : ∀ l, (a l).isNaN = false) (hb : ∀ l, (b l).isNaN = false)
    (hid : hopperStep w a b = w) :
    truncTo (2 ^ (stepExp tc a b - 13)) tc.value = tc.value ∧
      ∑ l, truncTo (2 ^ (stepExp tc a b - 13)) (val (a l) * val (b l)) = 0 := by
  obtain ⟨hm2, he2⟩ := fp32Value_fin hw
  set E := stepExp tc a b with hEdef
  have hg : (0 : ℚ) < 2 ^ (E - 13) := two_zpow_pos _
  have hcE : tc.exp ≤ E := le_maxExp _ _ tc List.mem_cons_self (by omega)
  have hle : ∀ l, (product (a l) (b l)).mag ≠ 0 → (product (a l) (b l)).exp ≤ E := fun l hp =>
    le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
  set total := alignedSum 14 E (tc :: List.ofFn fun l => product (a l) (b l)) with htdef
  rw [hopperStep_fin hw a b ha hb] at hid
  unfold groupSum at hid
  change packValue (let t := normalize 14 E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t) = w at hid
  clear_value total E
  have hsum : total = aligned 14 E tc + ∑ l, aligned 14 E (product (a l) (b l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have htot : (total : ℚ) * 2 ^ (E - 13) =
      truncTo (2 ^ (E - 13)) tc.value + ∑ l, truncTo (2 ^ (E - 13)) (val (a l) * val (b l)) := by
    rw [hsum]; push_cast; rw [add_mul, Finset.sum_mul, aligned_value (fun _ => hcE), add_right_inj]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [aligned_value (hle l), productExact _ _ _ _ (val_eq (ha l)) (val_eq (hb l))]
  have hT : total.natAbs < 2 ^ 21 := by
    have hc : (aligned 14 E tc).natAbs < 2 ^ 15 := aligned_natAbs_lt E tc (by omega)
    have h1 : |∑ l : Fin 32, aligned 14 E (product (a l) (b l))| ≤
        ∑ l : Fin 32, |aligned 14 E (product (a l) (b l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin 32, |aligned 14 E (product (a l) (b l))| ≤ ∑ _l : Fin 32, (2 ^ 15 : ℤ) := by
      refine Finset.sum_le_sum fun l _ => ?_
      have := aligned_natAbs_lt E (product (a l) (b l)) (product_mag_lt _ _)
      rw [Int.abs_eq_natAbs]; omega
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    rw [Int.abs_eq_natAbs] at h1
    rw [hsum]
    omega
  have hval : wordQ w = some tc.value := by simp [wordQ, hw, Value.finite?]
  by_cases h0 : total = 0
  · rw [h0] at hid
    simp [normalize, Term.zero, packValue, packTerm] at hid
    rw [← hid, fp32Value_zero] at hw
    cases hw; simp [Term.zero] at hm
  obtain ⟨m1, m2, e1, -, -, hv⟩ := normalize_spec' h0 (le_trans he hcE) hT
  dsimp only at hid
  split_ifs at hid with hinf
  · have : fp32Value w = .inf (normalize 14 E total).neg := by
      rw [← hid]; cases (normalize 14 E total).neg <;> rfl
    rw [hw] at this; cases this
  · have e2 : (normalize 14 E total).exp ≤ 127 := by
      by_contra hc; exact hinf ⟨by omega, by omega⟩
    have hq := wordQ_packTerm m1 m2 e1 e2
    simp only [packValue] at hid
    rw [hid, hval, hv, Option.some.injEq] at hq
    -- hq : tc.value = trunc14 (total * g)
    have hS0 : (total : ℚ) * 2 ^ (E - 13) ≠ 0 := mul_ne_zero (by exact_mod_cast h0) (two_zpow_ne _)
    have hbS : binade |(total : ℚ) * 2 ^ (E - 13)| = tc.exp := by
      rw [← binade_trunc14 hS0, ← hq, binade_term hm hm2]
    have hS : trunc14 ((total : ℚ) * 2 ^ (E - 13)) = (total : ℚ) * 2 ^ (E - 13) := by
      unfold trunc14
      rw [ite_eq_right hS0, hbS]
      have : (total : ℚ) * 2 ^ (E - 13) = ((total * 2 ^ (E - tc.exp).toNat : ℤ) : ℚ) * 2 ^ (tc.exp - 13) := by
        push_cast
        rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), mul_assoc, ← zpow_add₀ two_ne_zero]
        congr 2; ring
      rw [this, truncTo_intMul (two_zpow_pos _)]
    rw [hS] at hq
    have hc : truncTo (2 ^ (E - 13)) tc.value = tc.value := by rw [hq, truncTo_intMul hg]
    refine ⟨hc, ?_⟩
    rw [hc, ← hq] at htot
    linarith

theorem abs_sub_truncTo_lt {g : ℚ} (hg : 0 < g) (y : ℚ) : |y - truncTo g y| < g := by
  unfold truncTo
  have h1 := Int.floor_le (|y| / g)
  have h2 := Int.lt_floor_add_one (|y| / g)
  have e1 : (⌊|y| / g⌋ : ℚ) * g ≤ |y| := by rwa [le_div_iff₀ hg] at h1
  have e2 : |y| < (⌊|y| / g⌋ : ℚ) * g + g := by rwa [div_lt_iff₀ hg, add_mul, one_mul] at h2
  split_ifs with h
  · rw [abs_of_neg h] at e1 e2 ⊢
    simp only [neg_mul, one_mul, sub_neg_eq_add]
    rw [abs_lt]; constructor <;> linarith
  · rw [abs_of_nonneg (not_lt.1 h)] at e1 e2 ⊢
    simp only [one_mul]
    rw [abs_lt]; constructor <;> linarith

/-! ## (c) Promotion -/

/-- `rn` writes a representable value. -/
theorem rn_representable (y : ℚ) : Representable (rn y) := by
  rcases eq_or_ne y 0 with hy | hy
  · subst hy; exact ⟨0, 0, by norm_num, by simp [rn_eq, rp_zero]⟩
  have hb := sig_bounds hy
  have hf : (2 ^ 23 : ℤ) ≤ roundEven (|y| / ulp y) :=
    le_trans (Int.le_floor.2 (by exact_mod_cast hb.1)) (floor_le_roundEven _)
  have hf2 : roundEven (|y| / ulp y) ≤ 2 ^ 24 := by
    have := roundEven_le (|y| / ulp y)
    have : ⌊|y| / ulp y⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hb.2)
    omega
  set M := roundEven (|y| / ulp y)
  have hs : ∀ z : ℤ, (if y < 0 then (-1 : ℚ) else 1) * z = ((if y < 0 then -1 else 1) * z : ℤ) := by
    intro z; split <;> push_cast <;> ring
  have habs : ∀ z : ℤ, |(if y < 0 then -1 else 1) * z| = |z| := by
    intro z; split <;> simp
  have hrn : rn y = (if y < 0 then -1 else 1) * (M : ℚ) * ulp y := rfl
  rcases (show M < 2 ^ 24 ∨ M = 2 ^ 24 by omega) with h | h
  · refine ⟨(if y < 0 then -1 else 1) * M, binade |y| - 23, ?_, ?_⟩
    · rw [habs, abs_of_nonneg (by omega)]; exact h
    · rw [hrn, hs, ulp]
  · refine ⟨(if y < 0 then -1 else 1), binade |y| + 1, by split <;> norm_num, ?_⟩
    rw [hrn, h, ulp]
    rw [show (2 : ℚ) ^ (binade |y| + 1) = 2 ^ 24 * 2 ^ (binade |y| - 23) by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring]
    split <;> push_cast <;> ring

theorem moves_half_pos {r i : ℚ} (hr : Representable r) (h0 : 0 < r) (hi : ulp r < 2 * |i|) : rn (r + i) ≠ r := by
  rcases le_or_gt (ulp r) |i| with h | h
  · exact moves_pos hr h0 h
  obtain ⟨n, hn, n1, n2⟩ := pos_sig hr h0
  have hu := ulp_pos r
  have hk : (2 : ℚ) ^ binade r = 2 ^ 23 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_23]
  have hk1 : (2 : ℚ) ^ (binade r + 1) = 2 ^ 24 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_24]
  have n2' : (n : ℚ) + 1 ≤ 2 ^ 24 := by exact_mod_cast (show n + 1 ≤ 2 ^ 24 by omega)
  have n1' : (2 : ℚ) ^ 23 ≤ n := by exact_mod_cast n1
  rcases le_or_gt 0 i with hs | hs
  · rw [abs_of_nonneg hs] at hi h
    have hlo : 2 ^ binade r ≤ r + i := by rw [hk]; nlinarith
    have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith
    have hx : 0 < r + i := by linarith
    have hux : ulp (r + i) = ulp r := by rw [ulp, ulp, abs_of_pos hx, abs_of_pos h0, binade_unique hx hlo hhi]
    rw [rn_eq, ite_eq_right (not_lt.2 hx.le), one_mul, abs_of_pos hx, rp, hux]
    have q1 : 1 / 2 < i / ulp r := by rw [lt_div_iff₀ hu]; linarith
    have q2 : i / ulp r < 1 := by rw [div_lt_iff₀ hu]; linarith
    have : roundEven ((r + i) / ulp r) = n + 1 := by
      apply roundEven_near
      rw [add_div, (div_eq_iff hu.ne').2 hn, abs_lt]; push_cast; constructor <;> linarith
    rw [this]; push_cast; intro e; nlinarith
  · rw [abs_of_neg hs] at hi h
    rcases (show n = 2 ^ 23 ∨ 2 ^ 23 < n by omega) with hn1 | hn1
    · have hn1' : (n : ℚ) = 2 ^ 23 := by exact_mod_cast hn1
      have hrep : Representable (r - ulp r / 2) := by
        refine ⟨2 ^ 24 - 1, binade |r| - 24, by norm_num, ?_⟩
        have e : ulp r = 2 * 2 ^ (binade |r| - 24) := by
          rw [ulp, show binade |r| - 23 = (binade |r| - 24) + 1 by ring, zpow_add_one₀ two_ne_zero]; ring
        have hr2 : r = 2 ^ 23 * (2 * 2 ^ (binade |r| - 24)) := by rw [← e, ← hn1']; exact hn
        push_cast; linear_combination hr2 - e / 2
      have := rn_mono (show r + i ≤ r - ulp r / 2 by linarith)
      rw [rn_exact hrep] at this
      intro e; linarith
    · have n1'' : (2 : ℚ) ^ 23 + 1 ≤ n := by exact_mod_cast (show (2 : ℤ) ^ 23 + 1 ≤ n by omega)
      have hlo : 2 ^ binade r ≤ r + i := by rw [hk]; nlinarith
      have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith
      have hx : 0 < r + i := lt_of_lt_of_le (two_zpow_pos _) hlo
      have hux : ulp (r + i) = ulp r := by rw [ulp, ulp, abs_of_pos hx, abs_of_pos h0, binade_unique hx hlo hhi]
      rw [rn_eq, ite_eq_right (not_lt.2 hx.le), one_mul, abs_of_pos hx, rp, hux]
      have q1 : 1 / 2 < -i / ulp r := by rw [lt_div_iff₀ hu]; linarith
      have q2 : -i / ulp r < 1 := by rw [div_lt_iff₀ hu]; linarith
      have : roundEven ((r + i) / ulp r) = n - 1 := by
        apply roundEven_near
        rw [add_div, (div_eq_iff hu.ne').2 hn, abs_lt, neg_div] at *; push_cast; constructor <;> linarith
      rw [this]; push_cast; intro e; nlinarith

theorem roundEven_half {n : ℤ} : roundEven ((n : ℚ) + 1 / 2) = if n % 2 = 0 then n else n + 1 := by
  have hf : ⌊(n : ℚ) + 1 / 2⌋ = n := by rw [Int.floor_eq_iff]; constructor <;> linarith
  unfold roundEven
  simp only [hf, add_sub_cancel_left, lt_self_iff_false, ite_false]

theorem roundEven_half_odd {n : ℤ} (h : n % 2 = 1) : roundEven ((n : ℚ) + 1 / 2) = n + 1 := by
  rw [roundEven_half, ite_eq_right (by omega)]

theorem roundEven_half_even {n : ℤ} (h : n % 2 = 0) : roundEven ((n : ℚ) + 1 / 2) = n := by
  rw [roundEven_half, ite_eq_left h]

theorem moves_tie_pos {r i : ℚ} (hr : Representable r) (h0 : 0 < r) (hi : 2 * |i| = ulp r)
    (hodd : ¬ ∃ m : ℤ, r = m * (2 * ulp r)) : rn (r + i) ≠ r := by
  obtain ⟨n, hn, n1, n2⟩ := pos_sig hr h0
  have hu := ulp_pos r
  have hodd' : n % 2 = 1 := by
    by_contra hc
    have h2 : (n : ℚ) = 2 * ((n / 2 : ℤ) : ℚ) := by exact_mod_cast (by omega : n = 2 * (n / 2))
    exact hodd ⟨n / 2, by linear_combination hn + ulp r * h2⟩
  have hk : (2 : ℚ) ^ binade r = 2 ^ 23 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_23]
  have hk1 : (2 : ℚ) ^ (binade r + 1) = 2 ^ 24 * ulp r := by rw [ulp, abs_of_pos h0, two_pow_24]
  have n2' : (n : ℚ) + 1 ≤ 2 ^ 24 := by exact_mod_cast (show n + 1 ≤ 2 ^ 24 by omega)
  have n1'' : (2 : ℚ) ^ 23 + 1 ≤ n := by exact_mod_cast (show (2 : ℤ) ^ 23 + 1 ≤ n by omega)
  rcases le_or_gt 0 i with hs | hs
  · rw [abs_of_nonneg hs] at hi
    have hlo : 2 ^ binade r ≤ r + i := by rw [hk]; nlinarith
    have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith
    have hx : 0 < r + i := by linarith
    have hux : ulp (r + i) = ulp r := by rw [ulp, ulp, abs_of_pos hx, abs_of_pos h0, binade_unique hx hlo hhi]
    rw [rn_eq, ite_eq_right (not_lt.2 hx.le), one_mul, abs_of_pos hx, rp, hux]
    have hm : (r + i) / ulp r = (n : ℚ) + 1 / 2 := by
      rw [div_eq_iff hu.ne']; linarith
    have : roundEven ((r + i) / ulp r) = n + 1 := by rw [hm, roundEven_half_odd hodd']
    rw [this]; push_cast; intro e; nlinarith
  · rw [abs_of_neg hs] at hi
    have hlo : 2 ^ binade r ≤ r + i := by rw [hk]; nlinarith
    have hhi : r + i < 2 ^ (binade r + 1) := by rw [hk1]; nlinarith
    have hx : 0 < r + i := lt_of_lt_of_le (two_zpow_pos _) hlo
    have hux : ulp (r + i) = ulp r := by rw [ulp, ulp, abs_of_pos hx, abs_of_pos h0, binade_unique hx hlo hhi]
    rw [rn_eq, ite_eq_right (not_lt.2 hx.le), one_mul, abs_of_pos hx, rp, hux]
    have hm : (r + i) / ulp r = ((n - 1 : ℤ) : ℚ) + 1 / 2 := by
      rw [div_eq_iff hu.ne']; push_cast; linarith
    have : roundEven ((r + i) / ulp r) = n - 1 := by rw [hm, roundEven_half_even (by omega)]
    rw [this]; push_cast; intro e; nlinarith

/-- **(c) Promotion**: adding `x` to a representable total `t ≠ 0` leaves it unchanged only if `2|x| ≤ ulp t`, and at
equality only if `t`'s significand `|t| / ulp t` is even (ties to even). -/
theorem rnAdd_identity {t x : ℚ} (hr : Representable t) (h0 : t ≠ 0) (h : rnAdd t x = t) :
    2 * |x| ≤ ulp t ∧ (2 * |x| = ulp t → ∃ m : ℤ, |t| = m * (2 * ulp t)) := by
  unfold rnAdd at h
  rcases h0.lt_or_gt with hn | hp
  · have h' : rn (-t + -x) = -t := by rw [show -t + -x = -(t + x) by ring, rn_neg, h]
    have hr' := representable_neg hr
    have hp' : 0 < -t := neg_pos.2 hn
    refine ⟨not_lt.1 fun hc => moves_half_pos hr' hp' (by rwa [ulp_neg, abs_neg]) h', fun he => ?_⟩
    by_contra hc
    refine moves_tie_pos hr' hp' (by rwa [ulp_neg, abs_neg]) (fun ⟨m, hm⟩ => hc ⟨m, ?_⟩) h'
    rw [abs_of_neg hn, hm, ulp_neg]
  · refine ⟨not_lt.1 fun hc => moves_half_pos hr hp hc h, fun he => ?_⟩
    by_contra hc
    exact moves_tie_pos hr hp he (fun ⟨m, hm⟩ => hc ⟨m, by rw [abs_of_pos hp]; exact hm⟩) h

/-- (c) on `promote`: a promotion identity on a nonzero total pins the group word below half an ulp of the total. -/
theorem promote_identity {t : ℚ} {w : Nat} {x : ℚ} (hr : Representable t) (h0 : t ≠ 0) (hw : wordQ w = some x)
    (h : promote (some t) w = some t) :
    2 * |x| ≤ ulp t ∧ (2 * |x| = ulp t → ∃ m : ℤ, |t| = m * (2 * ulp t)) := by
  simp only [promote, hw, Option.some.injEq] at h
  exact rnAdd_identity hr h0 h

/-! ## Skipping in place -/

section Fold

variable {α β : Type}

/-- `ms` flags only steps `fs` flags. -/
def SubMask (ms fs : List Bool) : Prop := List.Forall₂ (fun m f => m = true → f = true) ms fs

/-- The fixed points of the run of `f` over `xs` from `w`: flag `j` says step `j` maps its incoming value to itself. -/
def fixFlags [DecidableEq β] (f : β → α → β) : β → List α → List Bool
  | _, [] => []
  | w, x :: xs => decide (f w x = w) :: fixFlags f (f w x) xs

theorem length_fixFlags [DecidableEq β] (f : β → α → β) (w : β) (xs : List α) :
    (fixFlags f w xs).length = xs.length := by
  induction xs generalizing w with
  | nil => rfl
  | cons x xs ih => simp [fixFlags, ih]

/-- **Skipping fixed points**: passing through any set of steps that are fixed points of the original run leaves the
fold unchanged. -/
theorem foldMask_fix [DecidableEq β] (f : β → α → β) (w : β) (xs : List α) (ms : List Bool)
    (h : SubMask ms (fixFlags f w xs)) : foldMask f w xs ms = xs.foldl f w := by
  induction xs generalizing w ms with
  | nil => cases h; rfl
  | cons x xs ih =>
    cases h with
    | cons hm h =>
      rename_i m ms
      rw [foldMask, List.foldl_cons]
      have : (if m then w else f w x) = f w x := by
        cases m
        · rfl
        · rw [ite_eq_left rfl]; exact (of_decide_eq_true (hm rfl)).symm
      rw [this]
      exact ih _ _ h

theorem subMask_refl (fs : List Bool) : SubMask fs fs := by
  induction fs with
  | nil => exact List.Forall₂.nil
  | cons f fs ih => exact List.Forall₂.cons id ih

end Fold

/-- A group's identity atoms: flag `j` says atom `j` of the group maps its incoming word to itself. -/
def idFlags (g : List AtomIn) : List Bool := fixFlags step 0 g

/-- The promotion identities of a chain given by its groups: flag `i` says group `i`'s promotion leaves the total
unchanged. -/
def promFlags (gs : List (List AtomIn)) : List Bool := fixFlags promoteGroup (some 0) gs

/-- `F`'s atoms in one group: all of them if the group's promotion is an identity, else its identity atoms. -/
def frozenIn (prom : Bool) (g : List AtomIn) : Nat := if prom then g.length else (idFlags g).count true

/-- **e_F** of one output word, given by its groups: the number of atoms in `F`. -/
def eF (gs : List (List AtomIn)) : Nat := ((promFlags gs).zip gs).foldr (fun p n => frozenIn p.1 p.2 + n) 0

/-- `n` flags, `true` at position `j` only. -/
def oneMask : Nat → Nat → List Bool
  | 0, _ => []
  | n + 1, 0 => true :: List.replicate n false
  | n + 1, j + 1 => false :: oneMask n j

/-- **Absorbed**, given atom `j`'s group `g`, the total `t` before the group and the later groups' words `ws`: skipping
the atom alone leaves the group's word unchanged (an identity, or a later atom of the group merges the two running
words), or the group's promotion (the two words round into one total), or the later promotions (a later promotion
rounds the two totals into one). One replay of the rest of the group and of the later promotions. -/
def absorbedAt (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (j : Nat) : Bool :=
  decide (wordIn g (oneMask g.length j) = groupWord g) ||
    decide (promote t (wordIn g (oneMask g.length j)) = promote t (groupWord g)) ||
    decide (ws.foldl promote (promote t (wordIn g (oneMask g.length j))) =
      ws.foldl promote (promote t (groupWord g)))

/-- Atom `j` of group `i` is absorbed. -/
def absorbed (gs : List (List AtomIn)) (i j : Nat) : Bool :=
  absorbedAt (gs.getD i []) (inTotal gs i) (laterWords gs i) j

theorem total_eq_foldl (gs : List (List AtomIn)) : total gs = gs.foldl promoteGroup (some 0) := by
  rw [total, List.foldl_map]; rfl

/-- The total with atoms and groups skipped in place: `am` flags each group's atoms, `gm` flags whole groups. -/
def totalSkip (gs : List (List AtomIn)) (am : List (List Bool)) (gm : List Bool) : Option ℚ :=
  foldMask (fun t p => promote t (foldMask step 0 p.1 p.2)) (some 0) (gs.zip am) gm

/-- The chain with atoms and groups skipped in place, its groups those of `chain G`. -/
def chainSkip (G : Nat) (atoms : List AtomIn) (am : List (List Bool)) (gm : List Bool) : Option ℚ :=
  totalSkip (atoms.toChunks G) am gm

/-! ## (b) Confinement: the running words of a group -/

/-- An atom of finite codes. -/
def FiniteAtom (x : AtomIn) : Prop := (∀ l, (x.1 l).isNaN = false) ∧ ∀ l, (x.2 l).isNaN = false

/-- A running word of a group from `+0` after `j` atoms: `+0`, or normal with exponent in `[−25 − 13j, 16 + 7j]`. -/
def Tame (j : Nat) (w : Nat) : Prop :=
  w = 0 ∨ ∃ tc, fp32Value w = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧ -25 - 13 * (j : ℤ) ≤ tc.exp ∧ tc.exp ≤ 16 + 7 * (j : ℤ)

theorem tame_step {j : Nat} (hj : j ≤ 6) {w : Nat} (hw : Tame j w) (x : AtomIn) (hx : FiniteAtom x) :
    Tame (j + 1) (step w x) := by
  obtain ⟨ha, hb⟩ := hx
  obtain ⟨tc, hfw, htc⟩ : ∃ tc, fp32Value w = .fin tc ∧
      (tc.mag = 0 ∨ (2 ^ 23 ≤ tc.mag ∧ -25 - 13 * (j : ℤ) ≤ tc.exp ∧ tc.exp ≤ 16 + 7 * (j : ℤ))) := by
    rcases hw with rfl | ⟨tc, h1, h2, h3, h4⟩
    · exact ⟨Term.zero, fp32Value_zero, Or.inl rfl⟩
    · exact ⟨tc, h1, Or.inr ⟨h2, h3, h4⟩⟩
  unfold step
  rw [hopperStep_fin hfw x.1 x.2 ha hb]
  unfold groupSum
  set E := maxExp (-139) (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with hEdef
  set total := alignedSum 14 E (tc :: List.ofFn fun l => product (x.1 l) (x.2 l)) with htdef
  change Tame (j + 1) (packValue (let t := normalize 14 E total
    if t.mag ≠ 0 ∧ 127 < t.exp then Value.inf t.neg else Value.fin t))
  clear_value total E
  have hsum : total = aligned 14 E tc + ∑ l, aligned 14 E (product (x.1 l) (x.2 l)) := by
    rw [htdef, alignedSum, List.map_cons, List.sum_cons, List.map_ofFn, List.sum_ofFn]
    simp only [Function.comp_apply]
  have hEle : E ≤ 16 + 7 * (j : ℤ) := by
    rw [hEdef]
    refine maxExp_le _ _ _ (by omega) fun t ht htm => ?_
    rcases List.mem_cons.1 ht with rfl | ht
    · rcases htc with h | h
      · exact absurd h htm
      · exact h.2.2
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 ht
      have := (product_exp _ _ htm).2
      omega
  have hmc : tc.mag < 2 ^ 24 := (fp32Value_fin hfw).1
  have hT : total.natAbs < 2 ^ 21 := by
    have hc : (aligned 14 E tc).natAbs < 2 ^ 15 := aligned_natAbs_lt E tc (by omega)
    have h1 : |∑ l : Fin 32, aligned 14 E (product (x.1 l) (x.2 l))| ≤
        ∑ l : Fin 32, |aligned 14 E (product (x.1 l) (x.2 l))| := Finset.abs_sum_le_sum_abs _ _
    have h2 : ∑ l : Fin 32, |aligned 14 E (product (x.1 l) (x.2 l))| ≤ ∑ _l : Fin 32, (2 ^ 15 : ℤ) := by
      refine Finset.sum_le_sum fun l _ => ?_
      have := aligned_natAbs_lt E (product (x.1 l) (x.2 l)) (product_mag_lt _ _)
      rw [Int.abs_eq_natAbs]; omega
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h2
    rw [Int.abs_eq_natAbs] at h1
    rw [hsum]
    omega
  by_cases h0 : total = 0
  · left; rw [h0]; simp [normalize, Term.zero, packValue, packTerm]
  have hEge : -25 - 13 * (j : ℤ) ≤ E := by
    by_contra hlt
    apply h0
    rw [hsum]
    have hc : aligned 14 E tc = 0 := by
      rcases htc with h | h
      · simp [aligned, h]
      · exact absurd (le_trans h.2.1 (le_maxExp _ _ tc List.mem_cons_self (by omega) |>.trans_eq hEdef.symm)) hlt
    rw [hc, zero_add]
    refine Finset.sum_eq_zero fun l _ => ?_
    by_cases hp : (product (x.1 l) (x.2 l)).mag = 0
    · simp [aligned, hp]
    · have h1 := (product_exp _ _ hp).1
      have h2 : (product (x.1 l) (x.2 l)).exp ≤ E := by
        rw [hEdef]; exact le_maxExp _ _ _ (List.mem_cons_of_mem _ (List.mem_ofFn.2 ⟨l, rfl⟩)) hp
      omega
  obtain ⟨m1, m2, e1, e3, e4, -⟩ := normalize_spec' (E := E) h0 (by omega) hT
  have e2 : (normalize 14 E total).exp ≤ 127 := by omega
  dsimp only
  rw [ite_eq_right (fun h => absurd h.2 (not_lt.2 e2))]
  right
  exact ⟨_, fp32Value_packTerm m1 m2 e1 e2, m1, by push_cast; omega, by push_cast; omega⟩

/-- The incoming word of atom `j` of a group from `+0`. -/
def inWord (g : List AtomIn) (j : Nat) : Nat := (g.take j).foldl step 0

theorem tame_inWord (g : List AtomIn) (hg : g.length ≤ 7) (hfin : ∀ x ∈ g, FiniteAtom x) :
    ∀ j ≤ g.length, Tame j (inWord g j) := by
  intro j hj
  induction j with
  | zero => left; rfl
  | succ j ih =>
    have hjl : j < g.length := by omega
    have : inWord g (j + 1) = step (inWord g j) g[j] := by
      unfold inWord
      rw [List.take_add_one, List.getElem?_eq_getElem hjl, Option.toList_some, List.foldl_append]
      rfl
    rw [this]
    exact tame_step (by omega) (ih (by omega)) _ (hfin _ (List.getElem_mem hjl))

theorem getElem_fixFlags {α β : Type} [DecidableEq β] (f : β → α → β) (w : β) (xs : List α) (j : Nat)
    (hj : j < xs.length) (hj' : j < (fixFlags f w xs).length) :
    (fixFlags f w xs)[j] = decide (f ((xs.take j).foldl f w) xs[j] = (xs.take j).foldl f w) := by
  induction xs generalizing w j with
  | nil => simp at hj
  | cons x xs ih =>
    cases j with
    | zero => rfl
    | succ j =>
      simp only [fixFlags, List.getElem_cons_succ, List.take_succ_cons, List.foldl_cons]
      exact ih _ _ (by simpa using hj) _

theorem getElem_idFlags (g : List AtomIn) (j : Nat) (h : j < (idFlags g).length) (hj : j < g.length) :
    (idFlags g)[j] = decide (step (inWord g j) g[j] = inWord g j) :=
  getElem_fixFlags step 0 g j hj h

/-- **(b) Confinement.** In a group of at most 7 atoms from `+0` (Π's `G = 4`) on finite codes, the identity flag of
atom `j` is a function of the group's first `j + 1` atoms, and an identity at atom `j` is either on the word `+0`,
where it is exactly `preSum = 0` (always so at `j = 0`), or on a normal word `w_j` that the atom's products do not
move: (a)'s domination, `truncTo g c = c` and `Σ truncTo g p = 0`, with `c` the value of `w_j`. -/
theorem identity_confined (g : List AtomIn) (hg : g.length ≤ 7) (hfin : ∀ x ∈ g, FiniteAtom x) (j : Nat)
    (hj : j < g.length) (hid : (idFlags g)[j]'(by rw [idFlags, length_fixFlags]; exact hj) = true) :
    (inWord g j = 0 ∧ preSum g[j].1 g[j].2 = 0) ∨
      ∃ tc : Term, fp32Value (inWord g j) = .fin tc ∧ 2 ^ 23 ≤ tc.mag ∧
        truncTo (2 ^ (stepExp tc g[j].1 g[j].2 - 13)) tc.value = tc.value ∧
        ∑ l, truncTo (2 ^ (stepExp tc g[j].1 g[j].2 - 13)) (val (g[j].1 l) * val (g[j].2 l)) = 0 := by
  rw [getElem_idFlags g j _ hj, decide_eq_true_iff] at hid
  obtain ⟨ha, hb⟩ := hfin _ (List.getElem_mem hj)
  rcases tame_inWord g hg hfin j hj.le with h0 | ⟨tc, h1, h2, h3, -⟩
  · left
    refine ⟨h0, (hopperStep_zero_iff _ _ ha hb).1 ?_⟩
    rw [h0] at hid; exact hid
  · right
    exact ⟨tc, h1, h2, identity_needs_domination h1 h2 (by omega) _ _ ha hb hid⟩

/-! ## The count -/

/-- A group's domination event: some atom after the first is an identity, as `0` or `1`. -/
def event (g : List AtomIn) : Nat := if (idFlags g).tail.any id then 1 else 0


/-! ## Skipping atoms of one group -/

section Masks

variable {α β : Type}

theorem mem_allMasks (m : List Bool) : m ∈ allMasks m.length := by
  induction m with
  | nil => simp [allMasks]
  | cons b m ih =>
    rw [List.length_cons, allMasks, List.mem_flatMap]
    exact ⟨m, ih, by cases b <;> simp⟩

theorem foldMask_nil (f : β → α → β) (w : β) (xs : List α) : foldMask f w xs [] = xs.foldl f w := by
  cases xs <;> rfl

theorem foldMask_replicate (f : β → α → β) (w : β) (xs : List α) :
    foldMask f w xs (List.replicate xs.length false) = xs.foldl f w := by
  induction xs generalizing w with
  | nil => rfl
  | cons x xs ih =>
    rw [List.length_cons, List.replicate_succ, foldMask, List.foldl_cons]
    exact ih _

theorem foldMask_oneMask (f : β → α → β) (w : β) (xs : List α) (j : Nat) (hj : j < xs.length) :
    foldMask f w xs (oneMask xs.length j) = (xs.drop (j + 1)).foldl f ((xs.take j).foldl f w) := by
  induction xs generalizing j w with
  | nil => simp at hj
  | cons x xs ih =>
    cases j with
    | zero =>
      rw [List.length_cons, oneMask, foldMask, foldMask_replicate]
      rfl
    | succ j =>
      rw [List.length_cons, oneMask, foldMask]
      simp only [Bool.false_eq_true, ite_false]
      rw [ih _ j (by simpa using hj)]
      rfl

theorem foldl_split (f : β → α → β) (xs : List α) (i : Nat) (hi : i < xs.length) (w : β) :
    xs.foldl f w = (xs.drop (i + 1)).foldl f (f ((xs.take i).foldl f w) xs[i]) := by
  conv_lhs => rw [← List.take_append_drop i xs, List.foldl_append, List.drop_eq_getElem_cons hi, List.foldl_cons]

theorem subMask_of_getElem (ms fs : List Bool) (hl : ms.length = fs.length)
    (h : ∀ j (h1 : j < ms.length) (h2 : j < fs.length), ms[j] = true → fs[j] = true) : SubMask ms fs := by
  induction ms generalizing fs with
  | nil =>
    cases fs with
    | nil => exact List.Forall₂.nil
    | cons _ _ => simp at hl
  | cons m ms ih =>
    cases fs with
    | nil => simp at hl
    | cons f fs =>
      exact List.Forall₂.cons (h 0 (by simp) (by simp))
        (ih fs (by simpa using hl) fun j h1 h2 => h (j + 1) (by simpa using h1) (by simpa using h2))

end Masks

/-- Masks for `totalSkip` that skip, in place, the atoms `m` flags in group `i`, and nothing else. -/
def groupMask : List (List AtomIn) → Nat → List Bool → List (List Bool)
  | [], _, _ => []
  | _ :: gs, 0, m => m :: gs.map fun g => List.replicate g.length false
  | g :: gs, i + 1, m => List.replicate g.length false :: groupMask gs i m

/-- The total with the atoms `m` flags in group `i` skipped in place. -/
def skipIn (gs : List (List AtomIn)) (i : Nat) (m : List Bool) : Option ℚ := totalSkip gs (groupMask gs i m) []

/-- The total with atom `j` of group `i` skipped alone. -/
def skipOne (gs : List (List AtomIn)) (i j : Nat) : Option ℚ := skipIn gs i (oneMask (gs.getD i []).length j)

theorem foldl_promote_words (gs : List (List AtomIn)) (t : Option ℚ) :
    (gs.map groupWord).foldl promote t = gs.foldl promoteGroup t := by
  rw [List.foldl_map]; rfl

theorem zip_replicate_foldl (gs : List (List AtomIn)) (t : Option ℚ) :
    (gs.zip (gs.map fun g => List.replicate g.length false)).foldl
      (fun t p => promote t (foldMask step 0 p.1 p.2)) t = gs.foldl promoteGroup t := by
  induction gs generalizing t with
  | nil => rfl
  | cons g gs ih =>
    simp only [List.map_cons, List.zip_cons_cons, List.foldl_cons]
    rw [foldMask_replicate]
    exact ih _

theorem skipIn_aux (gs : List (List AtomIn)) (i : Nat) (m : List Bool) (t : Option ℚ) (hi : i < gs.length) :
    (gs.zip (groupMask gs i m)).foldl (fun t p => promote t (foldMask step 0 p.1 p.2)) t =
      (gs.drop (i + 1)).foldl promoteGroup (promote ((gs.take i).foldl promoteGroup t) (wordIn gs[i] m)) := by
  induction gs generalizing i t with
  | nil => simp at hi
  | cons g gs ih =>
    cases i with
    | zero =>
      simp only [groupMask, List.zip_cons_cons, List.foldl_cons]
      exact zip_replicate_foldl gs _
    | succ i =>
      simp only [groupMask, List.zip_cons_cons, List.foldl_cons]
      rw [foldMask_replicate]
      exact ih i _ (by simpa using hi)

/-- **A skip inside one group is confined to it**: the total with atoms of group `i` skipped is the masked word
(a fold of the group alone from `+0`) promoted into the total before group `i`, then the later words promoted. -/
theorem skipIn_eq (gs : List (List AtomIn)) (i : Nat) (m : List Bool) (hi : i < gs.length) :
    skipIn gs i m = (laterWords gs i).foldl promote (promote (inTotal gs i) (wordIn gs[i] m)) := by
  rw [laterWords, foldl_promote_words, skipIn, totalSkip, foldMask_nil]
  exact skipIn_aux gs i m _ hi

theorem total_split (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) :
    total gs = (laterWords gs i).foldl promote (promote (inTotal gs i) (groupWord gs[i])) := by
  rw [laterWords, foldl_promote_words, total_eq_foldl]
  exact foldl_split _ _ i hi _

/-! ## The absorbed atoms: every atom whose skip alone leaves the output unchanged -/

/-- **Completeness and soundness of `absorbed`**: skipping atom `j` of group `i` alone, in place, leaves the output
unchanged exactly when the atom is absorbed. -/
theorem skipOne_iff (gs : List (List AtomIn)) (i j : Nat) (hi : i < gs.length) :
    skipOne gs i j = total gs ↔ absorbed gs i j = true := by
  rw [skipOne, absorbed, absorbedAt, List.getD_eq_getElem _ _ hi, skipIn_eq gs i _ hi, total_split gs i hi]
  simp only [Bool.or_eq_true, decide_eq_true_eq]
  constructor
  · exact fun h => Or.inr h
  · rintro ((h | h) | h)
    · rw [h]
    · rw [h]
    · exact h

theorem foldl_promote_none (ws : List Nat) : ws.foldl promote none = none := by
  induction ws with
  | nil => rfl
  | cons w ws ih => exact ih

/-! ## Joint skips -/

theorem length_jointFlags (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) :
    (jointFlags gs i).length = gs[i].length := by
  rw [jointFlags, jointAt, List.getD_eq_getElem _ _ hi]; simp

theorem jointFlags_getElem (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) (m : List Bool)
    (hm : m.length = gs[i].length) (h : skipIn gs i m = total gs) (j : Nat) (hj : j < (jointFlags gs i).length)
    (hmj : m.getD j false = true) : (jointFlags gs i)[j] = true := by
  rw [skipIn_eq gs i m hi, total_split gs i hi] at h
  have hk : keeps gs[i] (inTotal gs i) (laterWords gs i) m = true := by simp [keeps, h]
  simp only [jointFlags, jointAt, List.getD_eq_getElem _ _ hi, List.getElem_map, List.getElem_range]
  rw [List.any_eq_true]
  exact ⟨m, hm ▸ mem_allMasks m, by rw [Bool.and_eq_true]; exact ⟨hmj, hk⟩⟩

/-- **Completeness for one group**: every set of one group's atoms whose joint skip, in place, leaves the output
unchanged is flagged. -/
theorem joint_complete (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) (m : List Bool)
    (hm : m.length = gs[i].length) (h : skipIn gs i m = total gs) : SubMask m (jointFlags gs i) := by
  apply subMask_of_getElem _ _ (by rw [length_jointFlags gs i hi, hm])
  intro j h1 h2 hmj
  exact jointFlags_getElem gs i hi m hm h j h2 (by rw [List.getD_eq_getElem _ _ h1]; exact hmj)

theorem wordQ_zero : wordQ 0 = some 0 := by
  simp [wordQ, fp32Value_zero, Value.finite?, Term.value, Term.zero]

theorem promote_zero {t : Option ℚ} (ht : ∀ q, t = some q → Representable q) : promote t 0 = t := by
  cases t with
  | none => rfl
  | some q =>
    simp only [promote, wordQ_zero, rnAdd, add_zero]
    rw [rn_exact (ht q rfl)]

/-! ## Completeness on a word: atom `t` is group `t / 4`'s atom `t % 4` -/

section Chunks

variable {α : Type}

/-- `toChunks (n + 1)` as plain recursion: the first `n + 1` elements, then the rest in chunks. -/
def chunksOf (n : Nat) : List α → List (List α)
  | [] => []
  | x :: xs => (x :: xs.take n) :: chunksOf n (xs.drop n)
termination_by l => l.length
decreasing_by simp; omega

theorem chunksOf_cons (n : Nat) (x : α) (xs : List α) :
    chunksOf n (x :: xs) = (x :: xs.take n) :: chunksOf n (xs.drop n) := by
  rw [chunksOf]

theorem toChunks_go (n : Nat) (xs : List α) (acc₁ : Array α) (acc₂ : Array (List α)) (h1 : 0 < acc₁.size)
    (h2 : acc₁.size ≤ n + 1) :
    List.toChunks.go (n + 1) xs acc₁ acc₂ =
      acc₂.toList ++ (acc₁.toList ++ xs.take (n + 1 - acc₁.size)) :: chunksOf n (xs.drop (n + 1 - acc₁.size)) := by
  induction xs generalizing acc₁ acc₂ with
  | nil => simp [List.toChunks.go, chunksOf]
  | cons x xs ih =>
    rw [List.toChunks.go]
    by_cases h : acc₁.size = n + 1
    · simp only [show (acc₁.size == n + 1) = true by simp [h], ↓reduceIte]
      rw [ih _ _ (by simp) (by simp)]
      simp [h, chunksOf_cons]
    · simp only [show (acc₁.size == n + 1) = false by simpa using h, Bool.false_eq_true, ↓reduceIte]
      rw [ih _ _ (by simp) (by simp; omega)]
      have e : n + 1 - acc₁.size = (n - acc₁.size) + 1 := by omega
      simp [e, Array.size_push]

theorem toChunks_succ (n : Nat) (l : List α) : l.toChunks (n + 1) = chunksOf n l := by
  cases l with
  | nil => simp [List.toChunks, chunksOf]
  | cons x xs =>
    simp only [List.toChunks]
    rw [toChunks_go n xs #[x] #[] (by simp) (by simp)]
    simp [chunksOf_cons]

theorem chunksOf_eq (n : Nat) : ∀ l : List α,
    chunksOf n l = (List.range ((l.length + n) / (n + 1))).map fun i => (l.drop ((n + 1) * i)).take (n + 1)
  | [] => by simp [chunksOf, Nat.div_eq_of_lt (Nat.lt_succ_self n)]
  | x :: xs => by
    rw [chunksOf_cons, chunksOf_eq n (xs.drop n)]
    have e : (xs.length + 1 + n) / (n + 1) = (xs.length - n + n) / (n + 1) + 1 := by
      rcases Nat.lt_or_ge xs.length n with h | h
      · rw [Nat.div_eq_of_lt_le (k := 1) (by omega) (by omega), show xs.length - n = 0 by omega,
          Nat.div_eq_of_lt (by omega)]
      · rw [Nat.sub_add_cancel h, show xs.length + 1 + n = xs.length + (n + 1) by omega,
          Nat.add_div_right _ (Nat.succ_pos n)]
    simp only [List.length_cons, List.length_drop, e, List.range_succ_eq_map, List.map_cons, List.map_map]
    congr 1
    refine List.map_congr_left fun i _ => ?_
    simp only [Function.comp_apply, List.drop_drop]
    rw [show (n + 1) * (i + 1) = (n + (n + 1) * i) + 1 by ring, List.drop_succ_cons, Nat.add_comm]
termination_by l => l.length
decreasing_by simp; omega

theorem toChunks_eq {G : Nat} (hG : 0 < G) (l : List α) :
    l.toChunks G = (List.range ((l.length + G - 1) / G)).map fun i => (l.drop (G * i)).take G := by
  obtain ⟨n, rfl⟩ : ∃ n, G = n + 1 := ⟨G - 1, by omega⟩
  rw [toChunks_succ, chunksOf_eq, show l.length + (n + 1) - 1 = l.length + n by omega]

theorem length_toChunks {G : Nat} (hG : 0 < G) (l : List α) : (l.toChunks G).length = (l.length + G - 1) / G := by
  rw [toChunks_eq hG]; simp

theorem length_toChunks_of_dvd {G : Nat} (hG : 0 < G) {l : List α} (h : G ∣ l.length) :
    (l.toChunks G).length = l.length / G := by
  obtain ⟨c, hc⟩ := h
  rw [length_toChunks hG, hc, Nat.mul_div_cancel_left _ hG]
  exact Nat.div_eq_of_lt_le (by rw [Nat.mul_comm]; omega) (by rw [Nat.succ_mul, Nat.mul_comm]; omega)

theorem getElem_toChunks {G : Nat} (hG : 0 < G) (l : List α) (i : Nat) (hi : i < (l.toChunks G).length) :
    (l.toChunks G)[i] = (l.drop (G * i)).take G := by
  simp only [toChunks_eq hG, List.getElem_map, List.getElem_range]

theorem length_getElem_toChunks {G : Nat} (hG : 0 < G) (l : List α) (i : Nat) (hi : i < (l.toChunks G).length) :
    (l.toChunks G)[i].length = min G (l.length - G * i) := by
  rw [getElem_toChunks hG]; simp

theorem lt_length_toChunks {G : Nat} (hG : 0 < G) {l : List α} {t : Nat} (ht : t < l.length) :
    t / G < (l.toChunks G).length := by
  rw [length_toChunks hG, Nat.lt_iff_add_one_le, Nat.le_div_iff_mul_le hG, Nat.add_mul, Nat.one_mul]
  have := Nat.div_mul_le_self t G
  omega

theorem length_getElem_toChunks_of_dvd {G : Nat} (hG : 0 < G) {l : List α} (h : G ∣ l.length) (i : Nat)
    (hi : i < (l.toChunks G).length) : (l.toChunks G)[i].length = G := by
  rw [length_getElem_toChunks hG]
  rw [length_toChunks_of_dvd hG h, Nat.lt_div_iff_mul_lt' h] at hi
  have : G * (i + 1) ≤ l.length := by
    obtain ⟨c, hc⟩ := h
    rw [hc] at hi
    rw [hc]; exact Nat.mul_le_mul_left _ (by have := (Nat.mul_lt_mul_left hG).1 hi; omega)
  rw [Nat.mul_succ] at this
  omega

theorem mod_lt_length_getElem_toChunks {G : Nat} (hG : 0 < G) {l : List α} {t : Nat} (ht : t < l.length) :
    t % G < ((l.toChunks G)[t / G]'(lt_length_toChunks hG ht)).length := by
  rw [length_getElem_toChunks hG]
  have := Nat.div_add_mod t G
  have := Nat.mod_lt t hG
  omega

/-- The element at `t` is chunk `t / G`'s element at `t % G`. -/
theorem getElem_getElem_toChunks {G : Nat} (hG : 0 < G) {l : List α} {t : Nat} (ht : t < l.length) :
    ((l.toChunks G)[t / G]'(lt_length_toChunks hG ht))[t % G]'(mod_lt_length_getElem_toChunks hG ht) = l[t] := by
  simp only [getElem_toChunks hG, List.getElem_take, List.getElem_drop]
  congr 1
  exact Nat.div_add_mod t G

theorem getD_getD_toChunks {G : Nat} (hG : 0 < G) (l : List α) (t : Nat) (d : α) :
    ((l.toChunks G).getD (t / G) []).getD (t % G) d = l.getD t d := by
  by_cases ht : t < l.length
  · rw [List.getD_eq_getElem _ _ (lt_length_toChunks hG ht),
      List.getD_eq_getElem _ _ (mod_lt_length_getElem_toChunks hG ht), getElem_getElem_toChunks hG ht,
      List.getD_eq_getElem _ _ ht]
  · rw [List.getD_eq_default _ _ (not_lt.1 ht)]
    by_cases hi : t / G < (l.toChunks G).length
    · rw [List.getD_eq_getElem _ _ hi]
      apply List.getD_eq_default
      rw [length_getElem_toChunks hG]
      have := Nat.div_add_mod t G
      omega
    · rw [List.getD_eq_default _ _ (not_lt.1 hi)]; rfl

end Chunks

theorem length_groupMask (gs : List (List AtomIn)) (i : Nat) (m : List Bool) :
    (groupMask gs i m).length = gs.length := by
  induction gs generalizing i with
  | nil => rfl
  | cons g gs ih => cases i <;> simp [groupMask, ih]

theorem getElem_groupMask (gs : List (List AtomIn)) (i : Nat) (m : List Bool) (i' : Nat)
    (h : i' < (groupMask gs i m).length) :
    (groupMask gs i m)[i'] =
      if i' = i then m else List.replicate (gs[i']'(by rwa [length_groupMask] at h)).length false := by
  induction gs generalizing i i' with
  | nil => simp [groupMask] at h
  | cons g gs ih =>
    cases i with
    | zero => cases i' <;> simp [groupMask]
    | succ i =>
      cases i' with
      | zero => simp [groupMask]
      | succ i' => simp only [groupMask, List.getElem_cons_succ, ih, Nat.add_right_cancel_iff, List.getElem_cons_succ]

/-- Masks that flag nothing outside group `i` are `groupMask`'s. -/
theorem eq_groupMask (gs : List (List AtomIn)) (i : Nat) (hi : i < gs.length) (L : List (List Bool))
    (hl : L.length = gs.length)
    (hr : ∀ i' (h : i' < L.length), i' ≠ i → L[i'] = List.replicate (gs[i']'(hl ▸ h)).length false) :
    L = groupMask gs i (L[i]'(hl ▸ hi)) := by
  refine List.ext_getElem (by rw [length_groupMask, hl]) fun i' h1 h2 => ?_
  rw [getElem_groupMask]
  split_ifs with he
  · subst he; rfl
  · exact hr i' h1 he

/-- Atom `t` of the word over `atoms` is **skippable** in place together with some set of atoms of its group
(`t / 4`, `G = 4`): some mask `M` over the word's atoms flags `t` and only atoms of `t`'s group, and the chain with the
atoms `M` flags skipped has the chain's output. -/
def WordSkippable (atoms : List AtomIn) (t : Nat) : Prop :=
  ∃ M : List Bool, M.length = atoms.length ∧ M.getD t false = true ∧ (∀ s, M.getD s false = true → s / 4 = t / 4) ∧
    chainSkip 4 atoms (M.toChunks 4) [] = chain 4 atoms

/-- **Completeness of the joint flags on a word**: a skippable atom is flagged. -/
theorem jointFlags_complete (atoms : List AtomIn) (t : Nat) (h : WordSkippable atoms t) :
    (jointFlags (atoms.toChunks 4) (t / 4)).getD (t % 4) false = true := by
  obtain ⟨M, hM, hMt, hMs, hs⟩ := h
  have h4 : 0 < 4 := by decide
  have ht : t < atoms.length := by
    by_contra hc
    rw [List.getD_eq_default _ _ (by omega)] at hMt
    exact Bool.noConfusion hMt
  have hi := lt_length_toChunks h4 ht
  have hL : (M.toChunks 4).length = (atoms.toChunks 4).length := by rw [length_toChunks h4, length_toChunks h4, hM]
  have hc : ∀ i' (h1 : i' < (M.toChunks 4).length),
      (M.toChunks 4)[i'].length = ((atoms.toChunks 4)[i']'(hL ▸ h1)).length := by
    intro i' h1; rw [length_getElem_toChunks h4, length_getElem_toChunks h4, hM]
  have hr : ∀ i' (h1 : i' < (M.toChunks 4).length), i' ≠ t / 4 →
      (M.toChunks 4)[i'] = List.replicate ((atoms.toChunks 4)[i']'(hL ▸ h1)).length false := by
    intro i' h1 hne
    refine List.ext_getElem (by rw [List.length_replicate, hc]) fun p hp _ => ?_
    rw [List.getElem_replicate]
    have hp4 : p < 4 := by rw [length_getElem_toChunks h4] at hp; omega
    simp only [getElem_toChunks h4, List.getElem_take, List.getElem_drop]
    by_contra hf
    have := hMs (4 * i' + p) (by rw [List.getD_eq_getElem]; simpa using hf)
    omega
  have hLg := eq_groupMask _ _ hi _ hL hr
  have hskip : skipIn (atoms.toChunks 4) (t / 4) ((M.toChunks 4)[t / 4]'(hL ▸ hi)) = total (atoms.toChunks 4) := by
    rw [skipIn, ← hLg]; exact hs
  have hj : t % 4 < (jointFlags (atoms.toChunks 4) (t / 4)).length := by
    rw [length_jointFlags _ _ hi]; exact mod_lt_length_getElem_toChunks h4 ht
  have hmj : ((M.toChunks 4)[t / 4]'(hL ▸ hi)).getD (t % 4) false = true := by
    rw [← List.getD_eq_getElem _ [], getD_getD_toChunks h4]; exact hMt
  rw [List.getD_eq_getElem _ _ hj]
  exact jointFlags_getElem _ _ hi _ (hc _ _) hskip _ hj hmj

theorem wordFlags_complete (k : ℕ) (a b : Fin k → Code) (t : ℕ) (h : WordSkippable (rowAtoms a b) t) :
    wordFlags k a b t = true :=
  jointFlags_complete _ t h

/-- **The joint flags are complete** for in-group skips: `Forming.FlagsComplete (fun i j t => WordSkippable
(rowAtoms (Aq i) (Bq j)) t) (unitFlags Aq Bq)`, unfolded. -/
theorem joint_flagsComplete {m n k : ℕ} (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    ∀ i j t, WordSkippable (rowAtoms (Aq i) (Bq j)) t → unitFlags Aq Bq i j t = true :=
  fun i j t h => wordFlags_complete k (Aq i) (Bq j) t h

/-! ## The two debits: (B)'s flags are (A)'s -/

/-! ## The linear replay: the later promotions by certificates -/

/-- A certificate for the later promotions after one group: totals in `[lo, hi]` reach the output, and totals `≤ c`
or `≥ d` do not. -/
structure Cert where
  lo : ℚ
  hi : ℚ
  c : ℚ
  d : ℚ

/-- What a certificate claims, for the later words `ws` and the output `out`. -/
def Cert.Good (z : Cert) (ws : List Nat) (out : Option ℚ) : Prop :=
  (∀ x, z.lo ≤ x → x ≤ z.hi → ws.foldl promote (some x) = out) ∧
    (∀ x, x ≤ z.c → ws.foldl promote (some x) ≠ out) ∧ ∀ x, z.d ≤ x → ws.foldl promote (some x) ≠ out

/-- **The local checks**, one certificate per group: the last one brackets the output, and each earlier one's four
values, promoted with the next group's word, land where the next certificate says (four `rnAdd`s per group). -/
def certChain : List Nat → List Cert → Option ℚ → Bool
  | [], [z], some o => decide (o ≤ z.lo ∧ z.hi ≤ o ∧ z.c < o ∧ o < z.d)
  | w :: ws, z :: z' :: zs, out =>
    (match wordQ w with
      | some v => decide (z'.lo ≤ rnAdd z.lo v ∧ rnAdd z.hi v ≤ z'.hi ∧ rnAdd z.c v ≤ z'.c ∧ z'.d ≤ rnAdd z.d v)
      | none => false) && certChain ws (z' :: zs) out
  | _, _, _ => false

theorem rnAdd_mono_left {t t' : ℚ} (v : ℚ) (h : t ≤ t') : rnAdd t v ≤ rnAdd t' v := by
  unfold rnAdd; exact rn_mono (by linarith)

/-- **The local checks give the claim** (monotonicity of `rn`, one group at a time). -/
theorem certChain_good (ws : List Nat) (z : Cert) (zs : List Cert) (out : Option ℚ)
    (h : certChain ws (z :: zs) out = true) : z.Good ws out := by
  induction ws generalizing z zs with
  | nil =>
    cases zs with
    | cons _ _ => simp [certChain] at h
    | nil =>
      cases out with
      | none => simp [certChain] at h
      | some o =>
        simp only [certChain, decide_eq_true_eq] at h
        obtain ⟨h1, h2, h3, h4⟩ := h
        refine ⟨fun x hx hx' => ?_, fun x hx => ?_, fun x hx => ?_⟩
        · simp only [List.foldl_nil, Option.some.injEq]; linarith
        · simp only [List.foldl_nil, ne_eq, Option.some.injEq]; intro e; linarith
        · simp only [List.foldl_nil, ne_eq, Option.some.injEq]; intro e; linarith
  | cons w ws ih =>
    cases zs with
    | nil => simp [certChain] at h
    | cons z' zs =>
      simp only [certChain, Bool.and_eq_true] at h
      obtain ⟨hw, hrest⟩ := h
      have g := ih z' zs hrest
      cases hv : wordQ w with
      | none => rw [hv] at hw; simp at hw
      | some v =>
        rw [hv] at hw
        simp only [decide_eq_true_eq] at hw
        obtain ⟨h1, h2, h3, h4⟩ := hw
        have hs : ∀ x, (w :: ws).foldl promote (some x) = ws.foldl promote (some (rnAdd x v)) := by
          intro x; simp [List.foldl_cons, promote, hv]
        refine ⟨fun x hx hx' => ?_, fun x hx => ?_, fun x hx => ?_⟩
        · rw [hs]; exact g.1 _ (le_trans h1 (rnAdd_mono_left v hx)) (le_trans (rnAdd_mono_left v hx') h2)
        · rw [hs]; exact g.2.1 _ (le_trans (rnAdd_mono_left v hx) h3)
        · rw [hs]; exact g.2.2 _ (le_trans h4 (rnAdd_mono_left v hx))

theorem certChain_tail (w : Nat) (ws : List Nat) (z : Cert) (zs : List Cert) (out : Option ℚ)
    (h : certChain (w :: ws) (z :: zs) out = true) : certChain ws zs out = true := by
  cases zs with
  | nil => simp [certChain] at h
  | cons z' zs => simp only [certChain, Bool.and_eq_true] at h; exact h.2

theorem certChain_getElem (ws : List Nat) (zs : List Cert) (out : Option ℚ) (h : certChain ws zs out = true)
    (j : Nat) (hj : j < zs.length) : zs[j].Good (ws.drop j) out := by
  induction ws generalizing zs j with
  | nil =>
    cases zs with
    | nil => simp at hj
    | cons z zs =>
      cases j with
      | zero => exact certChain_good [] z zs out h
      | succ j =>
        cases zs with
        | nil => simp at hj
        | cons _ _ => simp [certChain] at h
  | cons w ws ih =>
    cases zs with
    | nil => simp at hj
    | cons z zs =>
      cases j with
      | zero => exact certChain_good _ z zs out h
      | succ j =>
        have := ih zs (certChain_tail w ws z zs out h) j (by simpa using hj)
        simpa using this

/-- A checked certificate list certifies each group's later promotions. -/
theorem cert_group (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) (hi : i < gs.length)
    (h : certChain ((gs.map groupWord).drop 1) zs (total gs) = true) (hz : i < zs.length) :
    zs[i].Good (laterWords gs i) ((laterWords gs i).foldl promote (promote (inTotal gs i) (groupWord gs[i]))) := by
  have g := certChain_getElem _ zs _ h i hz
  have e : ((gs.map groupWord).drop 1).drop i = laterWords gs i := by
    rw [laterWords, List.drop_drop, List.map_drop, Nat.add_comm]
  rw [e, total_split gs i hi] at g
  exact g

/-- **The fast later-promotion test**: a promoted total in the certificate's interval keeps the output, one at or past
its bounds does not, and only one strictly between falls back to the later promotions. -/
def keepsFast (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool) : Bool :=
  match promote t (wordIn g m) with
  | some x => if z.lo ≤ x ∧ x ≤ z.hi then true else if x ≤ z.c ∨ z.d ≤ x then false else keeps g t ws m
  | none => keeps g t ws m

theorem keepsFast_eq (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool)
    (hz : z.Good ws (ws.foldl promote (promote t (groupWord g)))) : keepsFast z g t ws m = keeps g t ws m := by
  unfold keepsFast
  split
  · rename_i x hx
    split_ifs with h1 h2
    · simp [keeps, hx, hz.1 x h1.1 h1.2]
    · rcases h2 with h2 | h2
      · simp [keeps, hx, hz.2.1 x h2]
      · simp [keeps, hx, hz.2.2 x h2]
    · rfl
  · rfl

/-- The joint flags of a group by the fast test. -/
def jointAtFast (z : Cert) (g : List AtomIn) (t : Option ℚ) (ws : List Nat) : List Bool :=
  (List.range g.length).map fun j => (allMasks g.length).any fun m => m.getD j false && keepsFast z g t ws m

/-- **The joint flags by the linear replay**: the certificates `zs`, one per group, however the replay builds them,
are checked once (`certChain`); on success each group's masked totals are tested against its certificate, and on
failure the test is `jointFlags`'. -/
def jointFlagsFast (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) : List Bool :=
  if h : certChain ((gs.map groupWord).drop 1) zs (total gs) = true ∧ i < zs.length then
    jointAtFast zs[i] (gs.getD i []) (inTotal gs i) (laterWords gs i)
  else jointFlags gs i

/-- **The linear replay computes debit (A)'s flags**, for any certificates. -/
theorem jointFlagsFast_eq (gs : List (List AtomIn)) (zs : List Cert) (i : Nat) (hi : i < gs.length) :
    jointFlagsFast gs zs i = jointFlags gs i := by
  unfold jointFlagsFast
  split
  · rename_i h
    have hz := cert_group gs zs i hi h.1 h.2
    rw [jointFlags, jointAtFast, jointAt, List.getD_eq_getElem _ _ hi]
    congr 1; funext j; congr 1; funext m; rw [keepsFast_eq _ _ _ _ _ hz]
  · rfl

/-! ## Examples: the mechanisms outside the group, and joint skips -/

def plusOne : AtomIn := (lane0 0x38, lane0 0x38)
def minusOne : AtomIn := (lane0 0xB8, lane0 0x38)
def sixteenth : AtomIn := (lane0 0x28, lane0 0x28)

/-- The chain `+1 ×4 | −1 ×4`, output `0`. -/
def crossEx : List AtomIn := [plusOne, plusOne, plusOne, plusOne, minusOne, minusOne, minusOne, minusOne]

/-- **Cross-group joint skips escape every group-local count**: skipping all eight atoms of `crossEx` together leaves
the output unchanged, while no nonempty set of one group's atoms does (both groups' joint flags are all `false`), no
atom is absorbed, and `e_F = 0`. -/
theorem joint_cross_group :
    chainSkip 4 crossEx [[true, true, true, true], [true, true, true, true]] [] = chain 4 crossEx ∧
      jointFlags (crossEx.toChunks 4) 0 = [false, false, false, false] ∧
      jointFlags (crossEx.toChunks 4) 1 = [false, false, false, false] ∧ eF (crossEx.toChunks 4) = 0 := by
  decide +kernel

def unitB : Matrix (Fin 1) (Fin 64) Code := fun _ _ => 0x38

end Pouw.PearlC.Freeze
