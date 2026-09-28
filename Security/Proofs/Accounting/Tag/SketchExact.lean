import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic
import Proofs.Accounting.Tag.Signs

/-!
# Exactness of integer ±1 sketches on sm_120's E4M3 tensor cores

Every value of the chain is an integer of magnitude below `2^24`, carried as `intTerm`: its sign, the exponent of its
leading bit and its 24-bit significand. One step's group (the accumulator and 32 products, `intTerm`s of integers) is
aligned to an exponent of at most `23`, so on the 26-bit adder's grid of at most `2^-2` every term is exact
(`aligned_intTerm`); the aligned sum is the integer sum scaled by a power of two, and normalising it to 24 significant
bits keeps all of it (`normalize_intTerm`). So the group returns `intTerm` of the exact sum (`groupSum_intTerm`), the
step writes its FP32 word, and the next step reads it back (`fp32Value_packTerm`).

`smallCode_product` checks the 66 products of a ±1 code and a data code by `decide`. `exact` is
`Accounting.Guarantees.Sketch.Exact`, `smallCode` is `SmallCode`, and `tensorCoreNonCollision` is
`TensorCoreNonCollision`, from `exact` and the row bound of `Signs.lean`. `fp32Value_packTerm` restates
`Pouw.Fp8Atom.fp32Value_packTerm` so this area doesn't import PoUW's proofs.
-/

namespace Accounting.TagProofs

open Accounting.Tag Pouw.Fp8Atom Finset

/-! ## Integers as terms -/

/-- An integer on the FP32 grid: zero, or its sign, the exponent of its leading bit and its 24-bit significand. -/
def intTerm (z : ℤ) : Term :=
  if z = 0 then Term.zero
  else ⟨decide (z < 0), (bitLength z.natAbs : ℤ) - 1, z.natAbs * 2 ^ (24 - bitLength z.natAbs)⟩

theorem bitLength_bounds {n : ℕ} (h : n ≠ 0) :
    1 ≤ bitLength n ∧ 2 ^ (bitLength n - 1) ≤ n ∧ n < 2 ^ bitLength n := by
  unfold bitLength
  rw [ite_eq_right h, Nat.add_sub_cancel]
  exact ⟨Nat.le_add_left _ _, Nat.log2_self_le h, Nat.lt_log2_self⟩

theorem bitLength_le {n m : ℕ} (h : n ≠ 0) (hn : n < 2 ^ m) : bitLength n ≤ m := by
  obtain ⟨h1, h2, -⟩ := bitLength_bounds h
  have : 2 ^ (bitLength n - 1) < 2 ^ m := lt_of_le_of_lt h2 hn
  have := (Nat.pow_lt_pow_iff_right (by norm_num)).1 this
  omega

theorem bitLength_mul_pow {n : ℕ} (h : n ≠ 0) (r : ℕ) : bitLength (n * 2 ^ r) = bitLength n + r := by
  obtain ⟨h1, h2, h3⟩ := bitLength_bounds h
  have h0 : n * 2 ^ r ≠ 0 := by positivity
  have hlog : (n * 2 ^ r).log2 = bitLength n - 1 + r := by
    refine (Nat.log2_eq_iff h0).2 ⟨?_, ?_⟩
    · calc 2 ^ (bitLength n - 1 + r) = 2 ^ (bitLength n - 1) * 2 ^ r := pow_add _ _ _
        _ ≤ n * 2 ^ r := Nat.mul_le_mul_right _ h2
    · calc n * 2 ^ r < 2 ^ bitLength n * 2 ^ r := Nat.mul_lt_mul_of_pos_right h3 (by positivity)
        _ = 2 ^ (bitLength n - 1 + r + 1) := by rw [← pow_add]; congr 1; omega
  rw [show bitLength (n * 2 ^ r) = (n * 2 ^ r).log2 + 1 from ite_eq_right h0, hlog]
  omega

/-- The fields of a nonzero `intTerm` below `2^24`: a normal significand, an exponent in `[0, 23]`. -/
theorem intTerm_fields {z : ℤ} (h0 : z ≠ 0) (hz : z.natAbs < 2 ^ 24) :
    2 ^ 23 ≤ (intTerm z).mag ∧ (intTerm z).mag < 2 ^ 24 ∧ 0 ≤ (intTerm z).exp ∧ (intTerm z).exp ≤ 23 := by
  have hn : z.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
  obtain ⟨h1, h2, h3⟩ := bitLength_bounds hn
  have hb := bitLength_le hn hz
  simp only [intTerm, ite_eq_right h0]
  refine ⟨?_, ?_, by omega, by omega⟩
  · calc 2 ^ 23 = 2 ^ (bitLength z.natAbs - 1) * 2 ^ (24 - bitLength z.natAbs) := by rw [← pow_add]; congr 1; omega
      _ ≤ _ := Nat.mul_le_mul_right _ h2
  · calc z.natAbs * 2 ^ (24 - bitLength z.natAbs) < 2 ^ bitLength z.natAbs * 2 ^ (24 - bitLength z.natAbs) :=
          Nat.mul_lt_mul_of_pos_right h3 (by positivity)
      _ = 2 ^ 24 := by rw [← pow_add]; congr 1; omega

theorem intTerm_mag_eq_zero {z : ℤ} : (intTerm z).mag = 0 ↔ z = 0 := by
  constructor
  · intro h
    by_contra h0
    simp only [intTerm, ite_eq_right h0] at h
    have : z.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
    simp_all
  · rintro rfl
    rfl

/-- `intTerm z` is `z`. -/
theorem intTerm_value {z : ℤ} (hz : z.natAbs < 2 ^ 24) : (intTerm z).value = z := by
  by_cases h0 : z = 0
  · subst h0
    simp [intTerm, Term.value, Term.zero]
  · have hn : z.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
    have hb := bitLength_le hn hz
    obtain ⟨h1, -, -⟩ := bitLength_bounds hn
    simp only [intTerm, ite_eq_right h0, Term.value]
    have e : ((bitLength z.natAbs : ℤ) - 1 - 23) = -((24 - bitLength z.natAbs : ℕ) : ℤ) := by omega
    rw [e, zpow_neg, zpow_natCast]
    simp only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_natAbs, Int.cast_abs]
    rw [mul_assoc, mul_assoc, mul_inv_cancel₀ (by positivity), mul_one]
    by_cases hneg : z < 0
    · have hq : (z : ℚ) < 0 := Int.cast_lt_zero.2 hneg
      simp only [hneg, decide_true, ite_true, abs_of_neg hq]
      ring
    · have hq : 0 ≤ (z : ℚ) := Int.cast_nonneg (not_lt.1 hneg)
      simp only [hneg, decide_false, Bool.false_eq_true, ite_false, abs_of_nonneg hq, one_mul]

/-! ## The group on the 26-bit adder -/

theorem foldl_maxExp_ge (ts : List Term) :
    ∀ m : ℤ, m ≤ ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) m := by
  induction ts with
  | nil => intro m; exact le_rfl
  | cons t ts ih =>
    intro m
    rw [List.foldl_cons]
    refine le_trans ?_ (ih _)
    split_ifs with h
    · exact h.2.le
    · exact le_rfl

theorem le_foldl_maxExp (ts : List Term) :
    ∀ m : ℤ, ∀ t ∈ ts, t.mag ≠ 0 → t.exp ≤ ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) m := by
  induction ts with
  | nil => intro m t ht; simp at ht
  | cons u ts ih =>
    intro m t ht hm
    rw [List.foldl_cons]
    rcases List.mem_cons.1 ht with rfl | ht
    · refine le_trans ?_ (foldl_maxExp_ge ts _)
      split_ifs with h
      · exact le_rfl
      · push Not at h
        exact h hm
    · exact ih _ t ht hm

theorem foldl_maxExp_le (ts : List Term) (B : ℤ) (hB : ∀ t ∈ ts, t.mag ≠ 0 → t.exp ≤ B) :
    ∀ m : ℤ, m ≤ B → ts.foldl (fun m t => if t.mag ≠ 0 ∧ m < t.exp then t.exp else m) m ≤ B := by
  induction ts with
  | nil => intro m hm; exact hm
  | cons u ts ih =>
    intro m hm
    rw [List.foldl_cons]
    refine ih (fun t ht => hB t (List.mem_cons_of_mem _ ht)) _ ?_
    split_ifs with h
    · exact hB u List.mem_cons_self h.1
    · exact hm

/-- On the 26-bit adder, aligned to an exponent `E ≤ 25` at or above its own, an integer's term is exact: the integer on
the grid `2^(E − 25)`. -/
theorem aligned_intTerm {E z : ℤ} (hE : E ≤ 25) (hz : z.natAbs < 2 ^ 24) (hle : z ≠ 0 → (intTerm z).exp ≤ E) :
    aligned 26 E (intTerm z) = z * 2 ^ (25 - E).toNat := by
  by_cases h0 : z = 0
  · subst h0
    simp [aligned, intTerm, Term.zero]
  · have hn : z.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
    have hb := bitLength_le hn hz
    obtain ⟨h1, -, -⟩ := bitLength_bounds hn
    have hexp := hle h0
    simp only [intTerm, ite_eq_right h0] at hexp
    have hm : z.natAbs * 2 ^ (24 - bitLength z.natAbs) ≠ 0 := by positivity
    simp only [aligned, intTerm, ite_eq_right h0, hm, ite_false, toAdder, show (24 : ℕ) ≤ 26 by norm_num, ite_true,
      Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
    set s := (E - ((bitLength z.natAbs : ℤ) - 1)).toNat with hs
    set r := (25 - E).toNat with hr
    have hsr : 24 - bitLength z.natAbs + (26 - 24) = r + s := by omega
    have e : z.natAbs * 2 ^ (24 - bitLength z.natAbs) * 2 ^ (26 - 24) = z.natAbs * 2 ^ r * 2 ^ s := by
      rw [mul_assoc, ← pow_add, hsr, pow_add, mul_assoc]
    rw [e, Nat.mul_div_cancel _ (by positivity)]
    by_cases hneg : z < 0
    · simp only [hneg, decide_true, ite_true]
      push_cast
      rw [abs_of_neg hneg]
      ring
    · simp only [hneg, decide_false, Bool.false_eq_true, ite_false]
      push_cast
      rw [abs_of_nonneg (not_lt.1 hneg)]

/-- Normalising an integer sum below `2^24` on the grid `2^(E − 25)` to 24 significant bits keeps all of it. -/
theorem normalize_intTerm {E S : ℤ} (hE : E ≤ 25) (hS : S.natAbs < 2 ^ 24) :
    normalize 26 E (S * 2 ^ (25 - E).toNat) = intTerm S := by
  by_cases h0 : S = 0
  · subst h0
    simp [Pouw.Fp8Atom.normalize, intTerm]
  · have hn : S.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
    have hb := bitLength_le hn hS
    obtain ⟨h1, -, -⟩ := bitLength_bounds hn
    set r := (25 - E).toNat with hr
    have ht0 : S * 2 ^ r ≠ 0 := mul_ne_zero h0 (by positivity)
    have habs : (S * 2 ^ r).natAbs = S.natAbs * 2 ^ r := by rw [Int.natAbs_mul, Int.natAbs_pow]; rfl
    have hbl : bitLength (S * 2 ^ r).natAbs = bitLength S.natAbs + r := by rw [habs, bitLength_mul_pow hn]
    have hneg : decide (S * 2 ^ r < 0) = decide (S < 0) := by
      have : (0 : ℤ) < 2 ^ r := by positivity
      by_cases h : S < 0
      · simp [h, mul_neg_of_neg_of_pos h this]
      · simp [h, mul_nonneg (not_lt.1 h) this.le]
    have hmag1 : (if 26 < bitLength S.natAbs + r then (S.natAbs * 2 ^ r) >>> (bitLength S.natAbs + r - 26)
        else (S.natAbs * 2 ^ r) <<< (26 - (bitLength S.natAbs + r))) = S.natAbs * 2 ^ (26 - bitLength S.natAbs) := by
      split_ifs with h
      · rw [Nat.shiftRight_eq_div_pow]
        refine Nat.div_eq_of_eq_mul_left (by positivity) ?_
        rw [mul_assoc, ← pow_add]
        congr 2
        omega
      · rw [Nat.shiftLeft_eq, mul_assoc, ← pow_add]
        congr 2
        omega
    have he : ¬ (E + ((bitLength S.natAbs + r : ℕ) : ℤ) - ((26 : ℕ) : ℤ) < -126) := by omega
    have hfrom : fromAdder 26 (S.natAbs * 2 ^ (26 - bitLength S.natAbs)) = S.natAbs * 2 ^ (24 - bitLength S.natAbs) := by
      rw [fromAdder, ite_eq_left (by norm_num), Nat.shiftRight_eq_div_pow,
        show 26 - bitLength S.natAbs = (24 - bitLength S.natAbs) + (26 - 24) by omega, pow_add, ← mul_assoc,
        Nat.mul_div_cancel _ (by positivity)]
    have hm : S.natAbs * 2 ^ (24 - bitLength S.natAbs) ≠ 0 := by positivity
    simp only [Pouw.Fp8Atom.normalize, ite_eq_right ht0, hbl]
    simp only [habs, hmag1, he, ite_false, hfrom, hm, hneg]
    unfold intTerm
    rw [ite_eq_right h0]
    congr 1
    push_cast
    omega

/-- **One group** of integer terms below `2^24`, with their sum below `2^24`, at a floor of at most `25`: the 26-bit
adder returns the exact sum. -/
theorem groupSum_intTerm {z : ℤ} (hz : z ≤ 25) (zs : List ℤ) (hzs : ∀ x ∈ zs, x.natAbs < 2 ^ 24)
    (hS : zs.sum.natAbs < 2 ^ 24) : groupSum 26 z (zs.map intTerm) = intTerm zs.sum := by
  set E := maxExp z (zs.map intTerm) with hEdef
  have hexp : ∀ t ∈ zs.map intTerm, t.mag ≠ 0 → t.exp ≤ 23 := by
    intro t ht hm
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 ht
    exact (intTerm_fields (intTerm_mag_eq_zero.not.1 hm) (hzs x hx)).2.2.2
  have hE : E ≤ 25 := foldl_maxExp_le _ 25 (fun t ht hm => (hexp t ht hm).trans (by norm_num)) z hz
  have hal : ∀ x ∈ zs, aligned 26 E (intTerm x) = x * 2 ^ (25 - E).toNat := fun x hx =>
    aligned_intTerm hE (hzs x hx) fun h0 =>
      le_foldl_maxExp _ z _ (List.mem_map_of_mem hx) (intTerm_mag_eq_zero.not.2 h0)
  have hsum : alignedSum 26 E (zs.map intTerm) = zs.sum * 2 ^ (25 - E).toNat := by
    unfold alignedSum
    rw [List.map_map, List.map_congr_left (g := fun x => x * 2 ^ (25 - E).toNat)
      (fun x hx => by simp only [Function.comp_apply]; exact hal x hx), List.sum_map_mul_right, List.map_id']
  unfold groupSum
  rw [← hEdef, hsum, normalize_intTerm hE hS]

/-! ## The step -/

/-- `pack_fp32` and `fp32_value` round trip on a normal term (`Pouw.Fp8Atom.fp32Value_packTerm`). -/
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

/-- The word of an integer below `2^24` reads back as its term. -/
theorem fp32Value_packTerm_intTerm {z : ℤ} (hz : z.natAbs < 2 ^ 24) :
    fp32Value (packTerm (intTerm z)) = .fin (intTerm z) := by
  by_cases h0 : z = 0
  · subst h0
    rfl
  · obtain ⟨h1, h2, e1, e2⟩ := intTerm_fields h0 hz
    exact fp32Value_packTerm h1 h2 (by omega) (by omega)

/-- The 66 products of a ±1 code and a data code: each exact, the term of the integer `sgn s · smallValue i`. -/
theorem smallCode_product (s : Bool) (i : Fin 33) :
    productTotal (signCode s) (smallCode i) = .fin (intTerm (sgn s * smallValue i)) := by
  have hn : ∀ (s : Bool) (i : Fin 33), ((signCode s).isNaN || (smallCode i).isNaN) = false := by decide
  have hf : ∀ (s : Bool) (i : Fin 33),
      (product (signCode s) (smallCode i)).neg = (intTerm (sgn s * smallValue i)).neg ∧
      (product (signCode s) (smallCode i)).exp = (intTerm (sgn s * smallValue i)).exp ∧
      (product (signCode s) (smallCode i)).mag = (intTerm (sgn s * smallValue i)).mag := by decide
  unfold productTotal
  simp only [hn s i, Bool.false_eq_true, ite_false]
  obtain ⟨h1, h2, h3⟩ := hf s i
  revert h1 h2 h3
  generalize product (signCode s) (smallCode i) = p
  generalize intTerm (sgn s * smallValue i) = q
  cases p
  cases q
  rintro rfl rfl rfl
  rfl

theorem term_bound (s : Bool) (i : Fin 33) : (sgn s * smallValue i).natAbs ≤ 16 := by
  revert s i
  decide

theorem zip_ofFn {α β : Type} {n : ℕ} (a : Fin n → α) (b : Fin n → β) :
    (List.ofFn a).zip (List.ofFn b) = List.ofFn fun l => (a l, b l) := by
  apply List.ext_getElem
  · simp
  · intro m h1 h2
    simp

/-- **One step** from an integer accumulator below `2^24`, with the new sum below `2^24`: the word of the exact sum. -/
theorem step_intTerm (A : ℤ) (s : Fin 32 → Bool) (x : Fin 32 → Fin 33) (hA : A.natAbs < 2 ^ 24)
    (hS : (A + ∑ l, sgn (s l) * smallValue (x l)).natAbs < 2 ^ 24) :
    Pouw.PearlC.sm120Step (packTerm (intTerm A)) (fun l => signCode (s l)) (fun l => smallCode (x l)) =
      packTerm (intTerm (A + ∑ l, sgn (s l) * smallValue (x l))) := by
  set v : Fin 32 → ℤ := fun l => sgn (s l) * smallValue (x l) with hv
  have hps : List.map (fun p => productTotal p.1 p.2)
      (List.take 32 ((List.ofFn fun l => signCode (s l)).zip (List.ofFn fun l => smallCode (x l)))) =
      (List.ofFn v).map fun z => Value.fin (intTerm z) := by
    rw [zip_ofFn, List.take_of_length_le (by simp), List.map_ofFn, List.map_ofFn]
    congr 1
    funext l
    exact smallCode_product (s l) (x l)
  have hlist : (A :: List.ofFn v).sum = A + ∑ l, v l := by rw [List.sum_cons, List.sum_ofFn]
  have hzs : ∀ z ∈ A :: List.ofFn v, z.natAbs < 2 ^ 24 := by
    intro z hz
    rcases List.mem_cons.1 hz with rfl | hz
    · exact hA
    · obtain ⟨l, rfl⟩ := List.mem_ofFn.1 hz
      exact lt_of_le_of_lt (term_bound (s l) (x l)) (by norm_num)
  have hgroup := groupSum_intTerm (z := -133) (by norm_num) (A :: List.ofFn v) hzs (by rwa [hlist])
  rw [hlist] at hgroup
  obtain hfin : ¬ ((intTerm (A + ∑ l, v l)).mag ≠ 0 ∧ 127 < (intTerm (A + ∑ l, v l)).exp) := by
    rintro ⟨hm, he⟩
    have := (intTerm_fields (intTerm_mag_eq_zero.not.1 hm) hS).2.2.2
    omega
  unfold Pouw.PearlC.sm120Step Pipeline.step Pouw.PearlC.sm120E4m3 Pouw.PearlC.sm120E4m3K32
  simp only [Pipeline.fold]
  rw [fp32Value_packTerm_intTerm hA, hps]
  unfold groupStepTotal
  have hany : ((List.ofFn v).any fun _ => false) = false := List.any_eq_false.2 fun _ _ => Bool.false_ne_true
  have hfm : List.filterMap (fun z => some (intTerm z)) (List.ofFn v) = List.map intTerm (List.ofFn v) :=
    congrFun (List.filterMap_eq_map (f := intTerm)) _
  simp only [List.any_cons, List.any_map, Value.isNaN, Value.isInf, Function.comp_def, hany, Bool.or_false,
    Bool.false_eq_true, ite_false, Bool.and_self, List.filterMap_cons, Value.finite?, List.filterMap_map, hfm]
  rw [show intTerm A :: List.map intTerm (List.ofFn v) = (A :: List.ofFn v).map intTerm from rfl, hgroup, ite_eq_right hfin]
  rfl

/-! ## The chain -/

/-- One step's sum is at most `32 · 16 = 512` in magnitude. -/
theorem stepSum_bound (s : Fin 32 → Bool) (x : Fin 32 → Fin 33) :
    (∑ l, sgn (s l) * smallValue (x l)).natAbs ≤ 512 := by
  refine (Int.natAbs_sum_le _ _).trans ?_
  calc ∑ l, (sgn (s l) * smallValue (x l)).natAbs ≤ ∑ _l : Fin 32, 16 :=
        Finset.sum_le_sum fun l _ => term_bound _ _
    _ = 512 := by simp

/-- Steps from an integer accumulator, with room for `512` per step below `2^24`: the word of the exact sum. -/
theorem foldl_steps {T : ℕ} (s : Fin T → Fin 32 → Bool) (x : Fin T → Fin 32 → Fin 33) :
    ∀ (ts : List (Fin T)) (A : ℤ), A.natAbs + 512 * ts.length < 2 ^ 24 →
      ts.foldl (fun acc t => Pouw.PearlC.sm120Step acc (fun l => signCode (s t l)) (fun l => smallCode (x t l)))
        (packTerm (intTerm A)) =
      packTerm (intTerm (A + (ts.map fun t => ∑ l, sgn (s t l) * smallValue (x t l)).sum)) := by
  intro ts
  induction ts with
  | nil => intro A _; simp
  | cons t ts ih =>
    intro A hA
    have hb := stepSum_bound (s t) (x t)
    simp only [List.length_cons] at hA
    rw [List.foldl_cons, step_intTerm A (s t) (x t) (by omega) (by omega), ih _ (by omega), List.map_cons,
      List.sum_cons, add_assoc]

theorem sketchExact : Accounting.Guarantees.Sketch.Exact := by
  intro T s x
  have hT := T.isLt
  set S := ∑ t, ∑ l, sgn (s t l) * smallValue (x t l) with hSdef
  have hS : S.natAbs < 2 ^ 24 := by
    have : S.natAbs ≤ ∑ _t : Fin T, 512 :=
      (Int.natAbs_sum_le _ _).trans (Finset.sum_le_sum fun t _ => stepSum_bound (s t) (x t))
    simp only [sum_const, card_univ, Fintype.card_fin, smul_eq_mul] at this
    omega
  have hw : sketchWord s x = packTerm (intTerm S) := by
    unfold sketchWord
    rw [show (0 : ℕ) = packTerm (intTerm 0) from rfl, foldl_steps s x _ 0 (by simp; omega), zero_add,
      ← List.ofFn_eq_map, List.sum_ofFn]
  rw [hw]
  simp [wordQ, fp32Value_packTerm_intTerm hS, Value.finite?, intTerm_value hS]

theorem smallCode_value : Accounting.Guarantees.Sketch.SmallCode := by
  intro i
  have h : ∀ i : Fin 33, (smallCode i).isNaN = false ∧ 0 ≤ (smallCode i).lsbExp + 9 ∧
      (if (smallCode i).sign then -1 else 1) * ((smallCode i).sig : ℤ) * 2 ^ ((smallCode i).lsbExp + 9).toNat =
        smallValue i * 512 := by decide
  obtain ⟨h1, h2, h3⟩ := h i
  have e : (2 : ℚ) ^ (smallCode i).lsbExp = 2 ^ ((smallCode i).lsbExp + 9).toNat / 512 := by
    rw [← zpow_natCast, Int.toNat_of_nonneg h2, zpow_add₀ two_ne_zero]
    norm_num
  have h3' := congrArg (fun z : ℤ => (z : ℚ)) h3
  simp only [decode, h1, Bool.false_eq_true, ite_false, Option.some.injEq, e]
  cases hs : (smallCode i).sign <;> simp only [hs, Bool.false_eq_true, ite_false, ite_true] at h3' ⊢ <;>
    push_cast at h3' <;> linarith

theorem tensorCoreNonCollision : Accounting.Guarantees.Sketch.TensorCoreNonCollision := by
  intro k T x y
  classical
  apply prob_uniform_le_inv_pow
  by_cases hxy : x = y
  · have : (univ.filter fun S : Fin k → Fin T → Fin 32 → Bool =>
        x ≠ y ∧ ∀ i, sketchWord (S i) x = sketchWord (S i) y) = ∅ := by
      ext S
      simp [hxy]
    rw [this, card_empty, mul_zero]
    exact Nat.zero_le _
  · obtain ⟨t0, l0, hne⟩ : ∃ t l, x t l ≠ y t l := by
      by_contra h
      push Not at h
      exact hxy (funext fun t => funext fun l => h t l)
    set d : Fin T × Fin 32 → ℤ := fun p => smallValue (x p.1 p.2) - smallValue (y p.1 p.2) with hd
    have hd0 : d (t0, l0) ≠ 0 := by
      simp only [hd, smallValue, sub_ne_zero, ne_eq, sub_left_inj, Nat.cast_inj]
      exact fun h => hne (Fin.ext h)
    obtain ⟨hf, hmv⟩ := flip_moves d (t0, l0) hd0
    set inner : (Fin T → Fin 32 → Bool) → ℤ := fun r => ∑ p, sgn (Function.uncurry r p) * d p with hinner
    set f : (Fin T → Fin 32 → Bool) → (Fin T → Fin 32 → Bool) := fun r =>
      Function.curry (Function.update (Function.uncurry r) (t0, l0) (!Function.uncurry r (t0, l0))) with hfdef
    have hfi : Function.Involutive f := by
      intro r
      have h := hf (Function.uncurry r)
      simp only at h
      simp only [hfdef, Function.uncurry_curry]
      rw [h]
      exact Function.curry_uncurry r
    have hne' : ∀ r, inner (f r) ≠ inner r := by
      intro r
      simp only [hinner, hfdef, Function.uncurry_curry]
      exact hmv _
    have hsub : (univ.filter fun S : Fin k → Fin T → Fin 32 → Bool =>
        x ≠ y ∧ ∀ i, sketchWord (S i) x = sketchWord (S i) y) ⊆
        univ.filter fun S : Fin k → Fin T → Fin 32 → Bool => ∀ i, inner (S i) = (fun _ => (0 : ℤ)) i := by
      intro S hS
      simp only [mem_filter, mem_univ, true_and] at hS ⊢
      intro i
      have h := congrArg wordQ (hS.2 i)
      rw [sketchExact T (S i) x, sketchExact T (S i) y, Option.some.injEq, Int.cast_inj] at h
      simp only [hinner, hd, Fintype.sum_prod_type, Function.uncurry_apply_pair, mul_sub, Finset.sum_sub_distrib]
      rw [h, sub_self]
    calc 2 ^ k * (univ.filter fun S : Fin k → Fin T → Fin 32 → Bool =>
          x ≠ y ∧ ∀ i, sketchWord (S i) x = sketchWord (S i) y).card
        ≤ 2 ^ k * (univ.filter fun S : Fin k → Fin T → Fin 32 → Bool =>
          ∀ i, inner (S i) = (fun _ => (0 : ℤ)) i).card := Nat.mul_le_mul_left _ (card_le_card hsub)
      _ ≤ _ := card_rows_le inner f hfi hne' _

end Accounting.TagProofs
