import Proofs.Core.Accel.Principles
import Proofs.Core.TC.Proofs.Words
import Mathlib.Tactic

/-!
# The captured float dots, proved: their product form and P1

**The product form** (`floatProductForm`). A float dot reads each operand pair only through its multiplier's output:
the pipeline's product is a function of `mulOut` (`prodOf`, `productTotal_mulOut`), and BF16's `tcDotTotal` is the
BF16 minifloat's product through the same groups (`floatDot_eq`). With the integer path's (`intProductForm`), every dot
each card's captures pin has the form (`H200.capturedProductForm`, `Sm120.capturedProductForm`).

**P1 on one group** (`p1Float_one`). Every nonzero term of a group is normal, so at least `2^exponent`, or a subnormal
product at an exponent of at most 2 (`product_ok`, `fp32Term_ok`). So on integer terms whose magnitudes sum below
`2^min(w, 24)` the group's maximum exponent `E` is at most `w - 1` (`exp_le_width`, `maxStep_le`), every term aligns
exactly to the grid `2^(E + 1 - w)` (`alignedW_exact`), the total is the terms' sum on that grid (`total_exact`),
normalising it keeps all of it (`normW_exact`), and its FP32 word reads back as the sum (`packFp32_intT`). That gives
P1 on the captured float atoms of both cards (`H200.p1Captured`, `Sm120.p1Captured`) and on E2M1's candidate pipeline
(`p1Pipelines`).
-/

namespace Verity.Accel

open Verity.TC Verity.TC.Spec

/-! ## The float dots' product form -/

/-- A float pair's product from its multiplier output (`mulOut`), at encoding `f`: `MiniFloat.productTotal`. -/
def prodOf (f : MiniFloat) (m : Cls × Cls × Bool × ℕ × ℕ) : Value :=
  if m.1 = .nan ∨ m.2.1 = .nan then .nan
  else if m.1 = .inf ∨ m.2.1 = .inf then
    if m.1 = .zero ∨ m.2.1 = .zero then .nan else .inf m.2.2.1
  else
    .fin (if m.2.2.2.1 <<< (25 - 2 * (f.manBits + 1)) = 0 then ZERO
      else ⟨m.2.2.1, (m.2.2.2.2 : ℤ) - 2 * f.bias, m.2.2.2.1 <<< (25 - 2 * (f.manBits + 1))⟩)

theorem productTotal_mulOut (f : Fmt) (h : intPair f f = false) (x y : ℕ) :
    f.mini.productTotal (f.code x) (f.code y) = prodOf f.mini (mulOut f f x y) := by
  simp only [MiniFloat.productTotal, MiniFloat.product, prodOf, mulOut, h, Bool.false_eq_true, ite_false,
    Nat.shiftRight_xor_distrib]
  push_cast
  rfl

/-- The groups of a pipeline over a list of products: each group takes the next ones. -/
def runP (width : ℕ) (z : ℤ) : List ℕ → Value → List Value → Value
  | [], acc, _ => acc
  | size :: rest, acc, ps => runP width z rest (groupStepTotal acc (ps.take size) width z) (ps.drop size)

theorem runGroupsWith_eq (prod : ℕ → ℕ → Value) (width : ℕ) (z : ℤ) :
    ∀ (gs : List ℕ) (acc : Value) (a b : List ℕ),
      runGroupsWith prod width z gs acc a b = runP width z gs acc (List.zipWith prod a b)
  | [], _, _, _ => rfl
  | size :: rest, acc, a, b => by
    rw [runGroupsWith, runP, runGroupsWith_eq prod width z rest, List.take_zipWith, List.drop_zipWith]

theorem runGroups_eq (width : ℕ) (z : ℤ) :
    ∀ (gs : List ℕ) (acc : Value) (a b : List ℕ),
      runGroups width z gs acc a b = runGroupsWith Fmt.bf16.mini.productTotal width z gs acc a b
  | [], _, _, _ => rfl
  | size :: rest, acc, a, b => by
    rw [runGroups, runGroupsWith, runGroups_eq width z rest]
    rfl

/-- Every float dot is the generic one: BF16's `tcDotTotal` is the BF16 minifloat's product through the pipeline. -/
theorem floatDot_eq (f : Fmt) (p : Pipeline) (c : Word) (a b : List ℕ) :
    floatDot f p c a b = BitVec.ofNat 32 (packValue (runP p.width p.zeroExponent p.groups (fp32Value c.toNat)
      (List.zipWith (fun x y => f.mini.productTotal (f.code x) (f.code y)) a b))) := by
  rw [← runGroupsWith_eq]
  unfold floatDot
  split_ifs with hf
  · subst hf
    rw [tcDotTotal, runGroups_eq]
    rfl
  · rfl

/-- **A float dot has the product form**: the accumulator and the pairs' multiplier outputs, through the pipeline. -/
theorem floatProductForm (f : Fmt) (p : Pipeline) (k : ℕ) (h : intPair f f = false) (hk : f.rowLen k = k) :
    ProductForm f f k (floatDot f p) := by
  refine ⟨fun c ms _ _ => BitVec.ofNat 32 (packValue (runP p.width p.zeroExponent p.groups (fp32Value c.toNat)
    (ms.map (prodOf f.mini)))), ?_⟩
  intro c a b ha hb
  rw [hk] at ha hb
  rw [List.take_of_length_le (le_of_eq ha), List.take_of_length_le (le_of_eq hb), floatDot_eq]
  simp only [List.map_zipWith, productTotal_mulOut f h]

/-- **The H200's captured dots have the product form**: the integer path's (`intProductForm`) and BF16's and FP8's. -/
theorem H200.capturedProductForm : CapturedProductForm H200.card := by
  intro fa fb sat k d hd
  change H200.captured fa fb sat k = some d at hd
  unfold H200.captured intCaptured at hd
  by_cases hi : intPair fa fb = true
  · simp only [hi, ite_true, Option.some.injEq] at hd
    subst hd
    exact intProductForm fa fb sat k hi
  · simp only [hi, Bool.false_eq_true, ite_false] at hd
    split_ifs at hd with h1 h2 h3
    · obtain ⟨rfl, rfl, rfl⟩ := h1
      cases hd
      exact floatProductForm .bf16 _ 16 rfl rfl
    · obtain ⟨rfl, rfl, rfl⟩ := h2
      cases hd
      exact floatProductForm .e4m3 _ 32 rfl rfl
    · obtain ⟨rfl, rfl, rfl⟩ := h3
      cases hd
      exact floatProductForm .e5m2 _ 32 rfl rfl

/-- **The RTX PRO 6000's captured dots have the product form.** -/
theorem Sm120.capturedProductForm : CapturedProductForm Sm120.card := by
  intro fa fb sat k d hd
  change Sm120.captured fa fb sat k = some d at hd
  unfold Sm120.captured intCaptured at hd
  by_cases hi : intPair fa fb = true
  · simp only [hi, ite_true, Option.some.injEq] at hd
    subst hd
    exact intProductForm fa fb sat k hi
  · simp only [hi, Bool.false_eq_true, ite_false] at hd
    split_ifs at hd with h1 h2 h3
    · obtain ⟨rfl, rfl, rfl⟩ := h1
      cases hd
      exact floatProductForm .bf16 _ 16 rfl rfl
    · obtain ⟨rfl, rfl, rfl⟩ := h2
      cases hd
      exact floatProductForm .e4m3 _ 32 rfl rfl
    · obtain ⟨rfl, rfl, rfl⟩ := h3
      cases hd
      exact floatProductForm .e5m2 _ 32 rfl rfl

/-! ## P1 on one group: the terms -/

theorem termVal_of_mag (t : Term) (h : t.magnitude = 0) : termVal t = 0 := by simp [termVal, h]

theorem abs_termVal (t : Term) : |termVal t| = t.magnitude * 2 ^ (t.exponent - 23) := by
  unfold termVal
  rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0 : ℚ) ≤ t.magnitude),
    abs_of_pos (by positivity : (0 : ℚ) < 2 ^ (t.exponent - 23))]
  split_ifs <;> simp

/-- A term with a normal significand is at least `2^exponent`. -/
theorem exp_lt (t : Term) (hm : 2 ^ 23 ≤ t.magnitude) (n : ℕ) (hv : |termVal t| < 2 ^ n) : t.exponent < n := by
  rw [abs_termVal] at hv
  have hm' : (2 : ℚ) ^ (23 : ℤ) ≤ t.magnitude := by norm_num; exact_mod_cast hm
  have h1 : (2 : ℚ) ^ t.exponent ≤ t.magnitude * 2 ^ (t.exponent - 23) := by
    calc (2 : ℚ) ^ t.exponent = 2 ^ (23 : ℤ) * 2 ^ (t.exponent - 23) := by
          rw [← zpow_add₀ two_ne_zero]; congr 1; ring
      _ ≤ _ := mul_le_mul_of_nonneg_right hm' (by positivity)
  have h2 : (2 : ℚ) ^ t.exponent < 2 ^ (n : ℤ) := by rw [zpow_natCast]; linarith
  exact (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℚ) < 2)).mp h2

/-- Every nonzero product is normal or has an exponent of at most 2: a subnormal operand's exponent field counts 1,
and the other's is at most `2^expBits - 1`. -/
theorem product_ok (f : MiniFloat) (he : 1 ≤ f.expBits) (hm : 2 * (f.manBits + 1) ≤ 25) (a b : ℕ) :
    (f.product a b).magnitude ≠ 0 → 2 ^ 23 ≤ (f.product a b).magnitude ∨ (f.product a b).exponent ≤ 2 := by
  intro h0
  unfold MiniFloat.product at h0 ⊢
  simp only at h0 ⊢
  generalize hea : (a >>> f.manBits) &&& (2 ^ f.expBits - 1) = ea at h0 ⊢
  generalize heb : (b >>> f.manBits) &&& (2 ^ f.expBits - 1) = eb at h0 ⊢
  have hle_a : ea ≤ 2 ^ f.expBits - 1 := hea ▸ Nat.and_le_right
  have hle_b : eb ≤ 2 ^ f.expBits - 1 := heb ▸ Nat.and_le_right
  generalize hsa : (a &&& (2 ^ f.manBits - 1)) ||| (if ea ≠ 0 then 2 ^ f.manBits else 0) = sa at h0 ⊢
  generalize hsb : (b &&& (2 ^ f.manBits - 1)) ||| (if eb ≠ 0 then 2 ^ f.manBits else 0) = sb at h0 ⊢
  by_cases hz : (sa * sb) <<< (25 - 2 * (f.manBits + 1)) = 0
  · rw [ite_eq_left hz] at h0
    exact absurd rfl h0
  rw [ite_eq_right hz]
  by_cases hab : ea ≠ 0 ∧ eb ≠ 0
  · left
    have hsa' : 2 ^ f.manBits ≤ sa := by rw [← hsa, ite_eq_left hab.1]; exact Nat.right_le_or
    have hsb' : 2 ^ f.manBits ≤ sb := by rw [← hsb, ite_eq_left hab.2]; exact Nat.right_le_or
    show 2 ^ 23 ≤ (sa * sb) <<< (25 - 2 * (f.manBits + 1))
    rw [Nat.shiftLeft_eq]
    calc 2 ^ 23 = 2 ^ f.manBits * 2 ^ f.manBits * 2 ^ (25 - 2 * (f.manBits + 1)) := by
          rw [← pow_add, ← pow_add]; congr 1; omega
      _ ≤ _ := Nat.mul_le_mul_right _ (Nat.mul_le_mul hsa' hsb')
  · right
    show ((if ea ≠ 0 then ea else 1 : ℕ) : ℤ) + ((if eb ≠ 0 then eb else 1 : ℕ) : ℤ) - 2 * f.bias ≤ 2
    have hP : ((2 ^ f.expBits : ℕ) : ℤ) = 2 * (2 : ℤ) ^ (f.expBits - 1) := by
      push_cast; rw [← pow_succ']; congr 1; omega
    have h1 : 1 ≤ 2 ^ f.expBits := Nat.one_le_two_pow
    simp only [MiniFloat.bias]
    generalize (2 : ℤ) ^ (f.expBits - 1) = Q at hP ⊢
    generalize 2 ^ f.expBits = P at hP h1 hle_a hle_b
    split_ifs <;> push_cast <;> omega

/-- An FP32 word's term is normal, zero or subnormal at exponent `-126`. -/
theorem fp32Term_ok (bits : ℕ) :
    (fp32Term bits).magnitude ≠ 0 → 2 ^ 23 ≤ (fp32Term bits).magnitude ∨ (fp32Term bits).exponent ≤ 2 := by
  intro h
  unfold fp32Term at h ⊢
  simp only at h ⊢
  split_ifs at h ⊢
  · exact absurd rfl h
  · right; simp [FP32_MIN_NORMAL_EXPONENT]
  · left; rw [← Nat.one_shiftLeft]; exact Nat.left_le_or

theorem exp_le_width (t : Term) (hok : t.magnitude ≠ 0 → 2 ^ 23 ≤ t.magnitude ∨ t.exponent ≤ 2) (w : ℕ)
    (hw : 3 ≤ w) (hv : |termVal t| < 2 ^ min w 24) (h0 : t.magnitude ≠ 0) : t.exponent ≤ (w : ℤ) - 1 := by
  rcases hok h0 with h | h
  · have := exp_lt t h _ hv
    have : ((min w 24 : ℕ) : ℤ) ≤ w := by exact_mod_cast min_le_left w 24
    omega
  · omega

/-! ## P1 on one group: the sum -/

open Verity.TC.Proofs (maxStep bitLength_spec bitLength_eq)

theorem maxStep_ge (ts : List Term) : ∀ m : ℤ, m ≤ ts.foldl maxStep m := by
  induction ts with
  | nil => intro m; exact le_rfl
  | cons t ts ih =>
    intro m
    rw [List.foldl_cons]
    refine le_trans ?_ (ih _)
    unfold maxStep
    split_ifs with h
    · exact h.2.le
    · exact le_rfl

theorem le_maxStep (ts : List Term) :
    ∀ m : ℤ, ∀ t ∈ ts, t.magnitude ≠ 0 → t.exponent ≤ ts.foldl maxStep m := by
  induction ts with
  | nil => intro m t ht; simp at ht
  | cons u ts ih =>
    intro m t ht hm
    rw [List.foldl_cons]
    rcases List.mem_cons.1 ht with rfl | ht
    · refine le_trans ?_ (maxStep_ge ts _)
      unfold maxStep
      split_ifs with h
      · exact le_rfl
      · push Not at h
        exact h hm
    · exact ih _ t ht hm

theorem maxStep_le (ts : List Term) (B : ℤ) (hB : ∀ t ∈ ts, t.magnitude ≠ 0 → t.exponent ≤ B) :
    ∀ m : ℤ, m ≤ B → ts.foldl maxStep m ≤ B := by
  induction ts with
  | nil => intro m hm; exact hm
  | cons u ts ih =>
    intro m hm
    rw [List.foldl_cons]
    refine ih (fun t ht => hB t (List.mem_cons_of_mem _ ht)) _ ?_
    unfold maxStep
    split_ifs with h
    · exact hB u List.mem_cons_self h.1
    · exact hm

/-- A term aligned to `E` on the `w`-bit adder: `group_sum`'s `aligned`. -/
def alignedW (w : ℕ) (E : ℤ) (t : Term) : ℕ :=
  let rescale : Int := (w : Int) - FP32_SIGNIFICAND_WIDTH
  (if rescale ≥ 0 then t.magnitude <<< rescale.toNat else t.magnitude >>> (-rescale).toNat) >>> (E - t.exponent).toNat

/-- `group_sum`'s total step at width `w`. -/
def totStepW (w : ℕ) (E : ℤ) (total : ℤ) (t : Term) : ℤ :=
  if t.magnitude = 0 then total
  else if t.negative then total - (alignedW w E t : ℤ) else total + (alignedW w E t : ℤ)

/-- `group_sum`'s normalisation at width `w`, from the maximum exponent `E` and the total. -/
def normW (w : ℕ) (E : ℤ) (total : ℤ) : Term :=
  let rescale : Int := (w : Int) - FP32_SIGNIFICAND_WIDTH
  if total = 0 then ZERO
  else
    let negative := decide (total < 0)
    let magnitude := total.natAbs
    let bl := bitLength magnitude
    let exponent : Int := E + bl - w
    let magnitude := if bl > w then magnitude >>> (bl - w) else magnitude <<< (w - bl)
    let magnitude' :=
      if exponent < FP32_MIN_NORMAL_EXPONENT then magnitude >>> (FP32_MIN_NORMAL_EXPONENT - exponent).toNat
      else magnitude
    let exponent' := if exponent < FP32_MIN_NORMAL_EXPONENT then FP32_MIN_NORMAL_EXPONENT else exponent
    let magnitude'' := if rescale ≥ 0 then magnitude' >>> rescale.toNat else magnitude' <<< (-rescale).toNat
    if magnitude'' = 0 then ZERO else ⟨negative, exponent', magnitude''⟩

theorem groupSum_eqW (ts : List Term) (w : ℕ) (z : ℤ) :
    groupSum ts w z = normW w (ts.foldl maxStep z) (ts.foldl (totStepW w (ts.foldl maxStep z)) 0) := rfl

theorem zpow_split (a b c : ℤ) (h : a = b + c) : (2 : ℚ) ^ a = 2 ^ b * 2 ^ c := by
  rw [h, zpow_add₀ two_ne_zero]

/-- On the `w`-bit adder, aligned to an exponent `E` at or above its own and at most `w - 1`, an integer term is
exact: its value on the grid `2^(E + 1 - w)`. -/
theorem alignedW_exact (w : ℕ) (E : ℤ) (t : Term) (hE : t.exponent ≤ E)
    (hEw : E ≤ (w : ℤ) - 1) (n : ℤ) (hn : termVal t = n) :
    (alignedW w E t : ℚ) = t.magnitude * 2 ^ (t.exponent - 23) * 2 ^ ((w : ℤ) - 1 - E) := by
  have hn' : (n.natAbs : ℚ) = t.magnitude * 2 ^ (t.exponent - 23) := by
    rw [← abs_termVal, hn, Nat.cast_natAbs, Int.cast_abs]
  have pz : ∀ k : ℕ, (2 : ℚ) ^ k = 2 ^ (k : ℤ) := fun k => (zpow_natCast _ _).symm
  have hr' : (((w : ℤ) - 1 - E).toNat : ℤ) = (w : ℤ) - 1 - E := Int.toNat_of_nonneg (by omega)
  have hd' : ((E - t.exponent).toNat : ℤ) = E - t.exponent := Int.toNat_of_nonneg (by omega)
  generalize ((w : ℤ) - 1 - E).toNat = r at hr'
  have hres : alignedW w E t = n.natAbs * 2 ^ r := by
    simp only [alignedW]
    have h24 : ((FP32_SIGNIFICAND_WIDTH : ℕ) : ℤ) = 24 := rfl
    generalize (E - t.exponent).toNat = d at hd'
    split_ifs with hw
    · have hs' : (((w : ℤ) - FP32_SIGNIFICAND_WIDTH).toNat : ℤ) = (w : ℤ) - 24 := by
        rw [Int.toNat_of_nonneg hw, h24]
      generalize ((w : ℤ) - FP32_SIGNIFICAND_WIDTH).toNat = s at hs'
      have key : t.magnitude * 2 ^ s = n.natAbs * 2 ^ r * 2 ^ d := by
        have : ((t.magnitude * 2 ^ s : ℕ) : ℚ) = ((n.natAbs * 2 ^ r * 2 ^ d : ℕ) : ℚ) := by
          push_cast
          rw [hn', pz s, pz r, pz d, zpow_split (s : ℤ) (t.exponent - 23) ((r : ℤ) + (d : ℤ)) (by omega),
            zpow_add₀ two_ne_zero]
          ring
        exact_mod_cast this
      rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, key, Nat.mul_div_cancel _ (by positivity)]
    · have hs' : ((-((w : ℤ) - FP32_SIGNIFICAND_WIDTH)).toNat : ℤ) = 24 - (w : ℤ) := by
        rw [Int.toNat_of_nonneg (by omega), h24]; ring
      generalize (-((w : ℤ) - FP32_SIGNIFICAND_WIDTH)).toNat = s at hs'
      have key : t.magnitude = n.natAbs * 2 ^ r * (2 ^ s * 2 ^ d) := by
        have : ((t.magnitude : ℕ) : ℚ) = ((n.natAbs * 2 ^ r * (2 ^ s * 2 ^ d) : ℕ) : ℚ) := by
          push_cast
          rw [hn', pz s, pz r, pz d, mul_assoc, mul_assoc, ← zpow_add₀ two_ne_zero, ← zpow_add₀ two_ne_zero,
            ← zpow_add₀ two_ne_zero, show t.exponent - 23 + ((r : ℤ) + ((s : ℤ) + (d : ℤ))) = 0 by omega,
            zpow_zero, mul_one]
        exact_mod_cast this
      rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul, ← pow_add, key, ← pow_add,
        Nat.mul_div_cancel _ (by positivity)]
  rw [hres]
  push_cast
  rw [hn', pz r, hr']

/-- The group's total: the terms' sum on the grid `2^(E + 1 - w)`. -/
theorem total_exact (w : ℕ) (E : ℤ) (hEw : E ≤ (w : ℤ) - 1) : ∀ (ts : List Term) (tot : ℤ),
    (∀ t ∈ ts, t.magnitude ≠ 0 → t.exponent ≤ E) → (∀ t ∈ ts, ∃ n : ℤ, termVal t = n) →
    ((ts.foldl (totStepW w E) tot : ℤ) : ℚ) = tot + (ts.map termVal).sum * 2 ^ ((w : ℤ) - 1 - E)
  | [], tot, _, _ => by simp
  | t :: ts, tot, hE, hn => by
    rw [List.foldl_cons, total_exact w E hEw ts _ (fun u hu => hE u (List.mem_cons_of_mem _ hu))
      (fun u hu => hn u (List.mem_cons_of_mem _ hu))]
    simp only [List.map_cons, List.sum_cons]
    have hstep : ((totStepW w E tot t : ℤ) : ℚ) = tot + termVal t * 2 ^ ((w : ℤ) - 1 - E) := by
      unfold totStepW
      by_cases h0 : t.magnitude = 0
      · rw [ite_eq_left h0, termVal_of_mag t h0]
        simp
      · rw [ite_eq_right h0]
        obtain ⟨n, hn'⟩ := hn t List.mem_cons_self
        have ha := alignedW_exact w E t (hE t List.mem_cons_self h0) hEw n hn'
        unfold termVal
        split_ifs with hneg <;> push_cast <;> rw [ha] <;> ring
    rw [hstep]
    ring

theorem bitLength_mul_pow {n : ℕ} (h : n ≠ 0) (r : ℕ) : bitLength (n * 2 ^ r) = bitLength n + r := by
  obtain ⟨h2, h3, h1⟩ := bitLength_spec n h
  refine bitLength_eq _ _ (by omega) ?_ ?_
  · calc 2 ^ (bitLength n + r - 1) = 2 ^ (bitLength n - 1) * 2 ^ r := by rw [← pow_add]; congr 1; omega
      _ ≤ n * 2 ^ r := Nat.mul_le_mul_right _ h2
  · calc n * 2 ^ r < 2 ^ bitLength n * 2 ^ r := Nat.mul_lt_mul_of_pos_right h3 (by positivity)
      _ = 2 ^ (bitLength n + r) := by rw [← pow_add]

theorem bitLength_le' {n m : ℕ} (hn : n < 2 ^ m) : bitLength n ≤ m := Verity.TC.Proofs.bitLength_le n m hn

/-- An integer on the FP32 grid: zero, or its sign, the exponent of its leading bit and its 24-bit significand. -/
def intT (N : ℤ) : Term :=
  if N = 0 then ZERO
  else ⟨decide (N < 0), (bitLength N.natAbs : ℤ) - 1, N.natAbs * 2 ^ (24 - bitLength N.natAbs)⟩

/-- Normalising an integer sum below `2^min(w, 24)` on the grid `2^(E + 1 - w)` keeps all of it. -/
theorem normW_exact (w : ℕ) (E : ℤ) (hEw : E ≤ (w : ℤ) - 1) (N : ℤ) (hN : N.natAbs < 2 ^ min w 24) :
    normW w E (N * 2 ^ ((w : ℤ) - 1 - E).toNat) = intT N := by
  by_cases h0 : N = 0
  · subst h0
    simp [normW, intT]
  have hn : N.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
  have hb := bitLength_le' hN
  have hbw : bitLength N.natAbs ≤ w := hb.trans (min_le_left _ _)
  have hb24 : bitLength N.natAbs ≤ 24 := hb.trans (min_le_right _ _)
  obtain ⟨-, -, h1⟩ := bitLength_spec _ hn
  have hr' : (((w : ℤ) - 1 - E).toNat : ℤ) = (w : ℤ) - 1 - E := Int.toNat_of_nonneg (by omega)
  generalize ((w : ℤ) - 1 - E).toNat = r at hr' ⊢
  have ht0 : N * 2 ^ r ≠ 0 := mul_ne_zero h0 (by positivity)
  have habs : (N * 2 ^ r).natAbs = N.natAbs * 2 ^ r := by rw [Int.natAbs_mul, Int.natAbs_pow]; rfl
  have hbl : bitLength (N * 2 ^ r).natAbs = bitLength N.natAbs + r := by rw [habs, bitLength_mul_pow hn]
  have hneg : decide (N * 2 ^ r < 0) = decide (N < 0) := by
    have : (0 : ℤ) < 2 ^ r := by positivity
    by_cases h : N < 0
    · simp [h, mul_neg_of_neg_of_pos h this]
    · simp [h, mul_nonneg (not_lt.1 h) this.le]
  have hmag1 : (if bitLength N.natAbs + r > w then (N.natAbs * 2 ^ r) >>> (bitLength N.natAbs + r - w)
      else (N.natAbs * 2 ^ r) <<< (w - (bitLength N.natAbs + r))) = N.natAbs * 2 ^ (w - bitLength N.natAbs) := by
    split_ifs with h
    · rw [Nat.shiftRight_eq_div_pow]
      refine Nat.div_eq_of_eq_mul_left (by positivity) ?_
      rw [mul_assoc, ← pow_add]
      congr 2
      omega
    · rw [Nat.shiftLeft_eq, mul_assoc, ← pow_add]
      congr 2
      omega
  have he : ¬ (E + ((bitLength N.natAbs + r : ℕ) : ℤ) - (w : ℤ) < FP32_MIN_NORMAL_EXPONENT) := by
    simp only [FP32_MIN_NORMAL_EXPONENT]; omega
  have hfrom : (if (w : ℤ) - FP32_SIGNIFICAND_WIDTH ≥ 0
      then (N.natAbs * 2 ^ (w - bitLength N.natAbs)) >>> ((w : ℤ) - FP32_SIGNIFICAND_WIDTH).toNat
      else (N.natAbs * 2 ^ (w - bitLength N.natAbs)) <<< (-((w : ℤ) - FP32_SIGNIFICAND_WIDTH)).toNat) =
      N.natAbs * 2 ^ (24 - bitLength N.natAbs) := by
    have h24 : ((FP32_SIGNIFICAND_WIDTH : ℕ) : ℤ) = 24 := rfl
    split_ifs with hw
    · have hs : (((w : ℤ) - FP32_SIGNIFICAND_WIDTH).toNat : ℤ) = (w : ℤ) - 24 := by
        rw [Int.toNat_of_nonneg hw, h24]
      generalize ((w : ℤ) - FP32_SIGNIFICAND_WIDTH).toNat = s at hs ⊢
      rw [Nat.shiftRight_eq_div_pow, show w - bitLength N.natAbs = (24 - bitLength N.natAbs) + s by omega, pow_add,
        ← mul_assoc, Nat.mul_div_cancel _ (by positivity)]
    · have hs : ((-((w : ℤ) - FP32_SIGNIFICAND_WIDTH)).toNat : ℤ) = 24 - (w : ℤ) := by
        rw [Int.toNat_of_nonneg (by omega), h24]; ring
      generalize (-((w : ℤ) - FP32_SIGNIFICAND_WIDTH)).toNat = s at hs ⊢
      rw [Nat.shiftLeft_eq, mul_assoc, ← pow_add]
      congr 2
      omega
  have hm : N.natAbs * 2 ^ (24 - bitLength N.natAbs) ≠ 0 := by positivity
  simp only [normW, ite_eq_right ht0, hbl]
  simp only [habs, hmag1, he, ite_false, hfrom, hm, hneg]
  unfold intT
  rw [ite_eq_right h0]
  congr 1
  push_cast
  omega

theorem intT_fields {N : ℤ} (h0 : N ≠ 0) (hN : N.natAbs < 2 ^ 24) :
    2 ^ 23 ≤ (intT N).magnitude ∧ (intT N).magnitude < 2 ^ 24 ∧
      0 ≤ (intT N).exponent ∧ (intT N).exponent ≤ 23 := by
  have hn : N.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
  obtain ⟨h2, h3, h1⟩ := bitLength_spec _ hn
  have hb := bitLength_le' hN
  simp only [intT, ite_eq_right h0]
  refine ⟨?_, ?_, by omega, by omega⟩
  · calc 2 ^ 23 = 2 ^ (bitLength N.natAbs - 1) * 2 ^ (24 - bitLength N.natAbs) := by
          rw [← pow_add]; congr 1; omega
      _ ≤ _ := Nat.mul_le_mul_right _ h2
  · calc N.natAbs * 2 ^ (24 - bitLength N.natAbs) < 2 ^ bitLength N.natAbs * 2 ^ (24 - bitLength N.natAbs) :=
          Nat.mul_lt_mul_of_pos_right h3 (by positivity)
      _ = 2 ^ 24 := by rw [← pow_add]; congr 1; omega

theorem intT_value {N : ℤ} (hN : N.natAbs < 2 ^ 24) : termVal (intT N) = N := by
  by_cases h0 : N = 0
  · subst h0
    simp [intT, termVal, ZERO]
  have hn : N.natAbs ≠ 0 := Int.natAbs_ne_zero.2 h0
  have hb := bitLength_le' hN
  obtain ⟨-, -, h1⟩ := bitLength_spec _ hn
  simp only [intT, ite_eq_right h0, termVal]
  have e : ((bitLength N.natAbs : ℤ) - 1 - 23) = -((24 - bitLength N.natAbs : ℕ) : ℤ) := by omega
  rw [e, zpow_neg, zpow_natCast]
  simp only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_natAbs, Int.cast_abs]
  rw [mul_assoc, mul_assoc, mul_inv_cancel₀ (by positivity), mul_one]
  by_cases hneg : N < 0
  · have hq : (N : ℚ) < 0 := Int.cast_lt_zero.2 hneg
    simp only [hneg, decide_true, ite_true, abs_of_neg hq]
    ring
  · have hq : 0 ≤ (N : ℚ) := Int.cast_nonneg (not_lt.1 hneg)
    simp only [hneg, decide_false, Bool.false_eq_true, ite_false, abs_of_nonneg hq, one_mul]

/-- `pack_fp32` and `fp32_term` round trip on an integer below `2^24`. -/
theorem packFp32_intT {N : ℤ} (hN : N.natAbs < 2 ^ 24) :
    packFp32 (intT N) < 2 ^ 32 ∧ fp32Term (packFp32 (intT N)) = intT N := by
  by_cases h0 : N = 0
  · subst h0
    exact ⟨by decide, rfl⟩
  obtain ⟨h1, h2, e1, e2⟩ := intT_fields h0 hN
  generalize intT N = t at h1 h2 e1 e2 ⊢
  obtain ⟨neg, exp, mag⟩ := t
  simp only at h1 h2 e1 e2
  have hm0 : mag ≠ 0 := by omega
  have hb : mag &&& (1 <<< 23) ≠ 0 := by rw [Ne, Verity.TC.Proofs.bit23 mag h2]; omega
  have hlow : mag &&& 0x7FFFFF = mag % 2 ^ 23 := by
    rw [show (0x7FFFFF : ℕ) = 2 ^ 23 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod]
  have hx : ¬ (exp > FP32_MAX_EXPONENT) := by simp only [FP32_MAX_EXPONENT]; omega
  have hef : ((exp + 127).toNat : ℤ) = exp + 127 := Int.toNat_of_nonneg (by omega)
  have hs : (if neg then 1 else 0) < 2 := by split <;> omega
  have hW : packFp32 ⟨neg, exp, mag⟩ =
      (if neg then 1 else 0) * 2 ^ 31 + (exp + 127).toNat * 2 ^ 23 + mag % 2 ^ 23 := by
    unfold packFp32
    simp only [hm0, ite_false, hx, hb, hlow]
    exact Verity.TC.Proofs.pack_fields _ _ _ hs (by omega) (Nat.mod_lt _ (by norm_num))
  rw [hW]
  generalize (exp + 127).toNat = ef at hef
  have hm2 : mag % 2 ^ 23 < 2 ^ 23 := Nat.mod_lt _ (by norm_num)
  generalize hsd : (if neg then 1 else 0) = s at hs
  have hlt : s * 2 ^ 31 + ef * 2 ^ 23 + mag % 2 ^ 23 < 2 ^ 32 := by omega
  refine ⟨hlt, ?_⟩
  obtain ⟨f1, f2, f3, -⟩ := Verity.TC.Proofs.fields _ hlt
  unfold fp32Term
  simp only [f1, f2, f3]
  have ha : (s * 2 ^ 31 + ef * 2 ^ 23 + mag % 2 ^ 23) / 2 ^ 23 % 2 ^ 8 = ef := by omega
  have hc : (s * 2 ^ 31 + ef * 2 ^ 23 + mag % 2 ^ 23) % 2 ^ 23 = mag % 2 ^ 23 := by omega
  have hd : (s * 2 ^ 31 + ef * 2 ^ 23 + mag % 2 ^ 23) / 2 ^ 31 = s := by omega
  rw [ha, hc, hd, ite_eq_right (by omega), Verity.TC.Proofs.hidden_or _ hm2,
    show 2 ^ 23 + mag % 2 ^ 23 = mag by omega]
  have hn : decide (s = 1) = neg := by cases neg <;> simp at hsd <;> simp [← hsd]
  rw [hn]
  congr 1
  omega

theorem groupStepTotal_fin (c : Term) (ts : List Term) (w : ℕ) (z : ℤ) :
    groupStepTotal (.fin c) (ts.map .fin) w z =
      if (groupSum (c :: ts) w z).magnitude ≠ 0 ∧ (groupSum (c :: ts) w z).exponent > FP32_MAX_EXPONENT
      then .inf (groupSum (c :: ts) w z).negative else .fin (groupSum (c :: ts) w z) := by
  unfold groupStepTotal
  have h1 : ((Value.fin c :: ts.map Value.fin).any Value.isNaN) = false := by
    simp [List.any_map, Function.comp_def, Value.isNaN]
  have h2 : (Value.fin c :: ts.map Value.fin).filterMap Value.infSign? = [] := by
    simp [Value.infSign?]
  have h3 : (Value.fin c :: ts.map Value.fin).filterMap Value.term? = c :: ts := by
    simp [List.filterMap_map, Function.comp_def, Value.term?]
  simp only [h1, h2, h3, Bool.false_eq_true, ite_false, List.contains_nil, and_self]

/-- **One group of integer terms** whose sum is below `2^min(w, 24)`, each nonzero one at an exponent of at most
`w - 1`, at a floor of at most `w - 1`: the `w`-bit adder returns the exact sum. -/
theorem groupSum_exact (ts : List Term) (w : ℕ) (z : ℤ) (hz : z ≤ (w : ℤ) - 1)
    (hok : ∀ t ∈ ts, t.magnitude ≠ 0 → t.exponent ≤ (w : ℤ) - 1)
    (hint : ∀ t ∈ ts, ∃ n : ℤ, termVal t = n)
    (N : ℤ) (hN : (ts.map termVal).sum = N) (hNb : N.natAbs < 2 ^ min w 24) :
    groupSum ts w z = intT N := by
  rw [groupSum_eqW]
  have hEw : ts.foldl maxStep z ≤ (w : ℤ) - 1 := maxStep_le ts _ hok z hz
  have hle : ∀ t ∈ ts, t.magnitude ≠ 0 → t.exponent ≤ ts.foldl maxStep z :=
    fun t ht h => le_maxStep ts z t ht h
  generalize ts.foldl maxStep z = E at hEw hle ⊢
  have htot := total_exact w E hEw ts 0 hle hint
  rw [hN] at htot
  have hT : ts.foldl (totStepW w E) 0 = N * 2 ^ ((w : ℤ) - 1 - E).toNat := by
    have : ((ts.foldl (totStepW w E) 0 : ℤ) : ℚ) = ((N * 2 ^ ((w : ℤ) - 1 - E).toNat : ℤ) : ℚ) := by
      rw [htot]
      push_cast
      rw [← zpow_natCast, Int.toNat_of_nonneg (by omega)]
      ring
    exact_mod_cast this
  rw [hT, normW_exact w E hEw N hNb]

theorem zipWith_congr_mem {α β γ : Type} (F G : α → β → γ) : ∀ (a : List α) (b : List β),
    (∀ x ∈ a, ∀ y ∈ b, F x y = G x y) → List.zipWith F a b = List.zipWith G a b
  | [], _, _ => by simp
  | _ :: _, [], _ => by simp
  | x :: a, y :: b, h => by
    simp only [List.zipWith_cons_cons]
    rw [h x List.mem_cons_self y List.mem_cons_self, zipWith_congr_mem F G a b
      (fun x hx y hy => h x (List.mem_cons_of_mem _ hx) y (List.mem_cons_of_mem _ hy))]

theorem zipWith_forall {α β γ : Type} (F : α → β → γ) (P : γ → Prop) (hP : ∀ x y, P (F x y)) :
    ∀ (a : List α) (b : List β), ∀ t ∈ List.zipWith F a b, P t
  | [], _, t, ht => by simp at ht
  | _ :: _, [], t, ht => by simp at ht
  | x :: a, y :: b, t, ht => by
    rw [List.zipWith_cons_cons] at ht
    rcases List.mem_cons.1 ht with rfl | ht
    · exact hP x y
    · exact zipWith_forall F P hP a b t ht

theorem sum_int (l : List ℚ) (h : ∀ p ∈ l, ∃ z : ℤ, p = z) : ∃ z : ℤ, l.sum = z := by
  induction l with
  | nil => exact ⟨0, by simp⟩
  | cons p l ih =>
    obtain ⟨z₁, h₁⟩ := h p List.mem_cons_self
    obtain ⟨z₂, h₂⟩ := ih fun q hq => h q (List.mem_cons_of_mem _ hq)
    exact ⟨z₁ + z₂, by rw [List.sum_cons, h₁, h₂]; push_cast; ring⟩

theorem abs_sum_le (l : List ℚ) : |l.sum| ≤ (l.map fun p => |p|).sum := by
  induction l with
  | nil => simp
  | cons p l ih =>
    rw [List.sum_cons, List.map_cons, List.sum_cons]
    exact (abs_add_le _ _).trans (by linarith)

theorem le_sum_abs (l : List ℚ) (p : ℚ) (hp : p ∈ l) : |p| ≤ (l.map fun p => |p|).sum :=
  List.single_le_sum (fun q hq => by obtain ⟨r, -, rfl⟩ := List.mem_map.1 hq; exact abs_nonneg r) _
    (List.mem_map_of_mem hp)

/-- **P1 on a pipeline of one group** of width at least 3 and floor at most 2, at every minifloat with an exponent
field and at most 11 fraction bits: exact below `2^min(width, 24)`. -/
theorem p1Float_one (f : Fmt) (p : Pipeline) (k : ℕ) (hg : p.groups = [k]) (hw : 3 ≤ p.width)
    (hz : p.zeroExponent ≤ 2) (he : 1 ≤ f.mini.expBits) (hm : 2 * (f.mini.manBits + 1) ≤ 25) : P1Float f p := by
  unfold P1Float P1Dot
  intro c a b ha hb _ hfin hcn hci
  dsimp only
  intro hc₀ hps hsum
  have hk : p.k = k := by simp [Pipeline.k, hg]
  rw [hk] at ha hb
  have hacc : fp32Value c.toNat = .fin (fp32Term c.toNat) := by
    unfold fp32Value
    split <;> simp_all
  have hpt : List.zipWith (fun x y => f.mini.productTotal (f.code x) (f.code y)) a b =
      (List.zipWith (fun x y => f.mini.product (f.code x) (f.code y)) a b).map .fin := by
    rw [List.map_zipWith]
    refine zipWith_congr_mem _ _ a b fun x hx y hy => ?_
    obtain ⟨-, hx1, hx2⟩ := hfin x (List.mem_append_left _ hx)
    obtain ⟨-, hy1, hy2⟩ := hfin y (List.mem_append_right _ hy)
    simp only [MiniFloat.productTotal, hx1, hy1, hx2, hy2, or_self, ite_false]
  generalize hprods : List.zipWith (fun x y => f.mini.product (f.code x) (f.code y)) a b = prods at hpt
  have hlen : prods.length = k := by rw [← hprods, List.length_zipWith, ha, hb, min_self]
  have hps' : List.zipWith (fun x y => termVal (f.mini.product (f.code x) (f.code y))) a b = prods.map termVal := by
    rw [← hprods, List.map_zipWith]
  rw [hps'] at hps hsum ⊢
  generalize hct : fp32Term c.toNat = ct at hacc hc₀ hsum ⊢
  rw [floatDot_eq, hg, hacc, hpt, runP, runP, List.take_of_length_le (by rw [List.length_map, hlen]),
    groupStepTotal_fin]
  obtain ⟨z₀, hz₀⟩ := hc₀
  obtain ⟨zs, hzs⟩ := sum_int _ hps
  have hb0 : 0 ≤ ((prods.map termVal).map fun p => |p|).sum :=
    List.sum_nonneg fun q hq => by obtain ⟨r, -, rfl⟩ := List.mem_map.1 hq; exact abs_nonneg r
  have hbound : ∀ t ∈ ct :: prods, |termVal t| < 2 ^ min p.width 24 := by
    intro t ht
    rcases List.mem_cons.1 ht with rfl | ht
    · linarith [abs_nonneg (termVal t)]
    · have := le_sum_abs _ _ (List.mem_map_of_mem (f := termVal) ht)
      linarith [abs_nonneg (termVal ct)]
  have hok : ∀ t ∈ ct :: prods, t.magnitude ≠ 0 → t.exponent ≤ (p.width : ℤ) - 1 := by
    intro t ht h0
    refine exp_le_width t ?_ p.width hw (hbound t ht) h0
    rcases List.mem_cons.1 ht with rfl | ht
    · rw [← hct]; exact fp32Term_ok _
    · rw [← hprods] at ht
      exact zipWith_forall _ (fun t => t.magnitude ≠ 0 → 2 ^ 23 ≤ t.magnitude ∨ t.exponent ≤ 2)
        (fun x y => product_ok f.mini he hm _ _) a b t ht
  have hint : ∀ t ∈ ct :: prods, ∃ n : ℤ, termVal t = n := by
    intro t ht
    rcases List.mem_cons.1 ht with rfl | ht
    · exact ⟨z₀, hz₀⟩
    · exact hps _ (List.mem_map_of_mem ht)
  have hN : ((ct :: prods).map termVal).sum = ((z₀ + zs : ℤ) : ℚ) := by
    rw [List.map_cons, List.sum_cons, hz₀, hzs]; push_cast; ring
  have hNb : (z₀ + zs).natAbs < 2 ^ min p.width 24 := by
    have h1 : |(((z₀ + zs : ℤ)) : ℚ)| < 2 ^ min p.width 24 := by
      rw [← hN, List.map_cons, List.sum_cons]
      exact (abs_add_le _ _).trans_lt (by linarith [abs_sum_le (prods.map termVal)])
    have : ((z₀ + zs).natAbs : ℚ) < ((2 ^ min p.width 24 : ℕ) : ℚ) := by
      rw [Nat.cast_natAbs, Int.cast_abs]; push_cast at h1 ⊢; exact h1
    exact_mod_cast this
  have hN24 : (z₀ + zs).natAbs < 2 ^ 24 :=
    hNb.trans_le (Nat.pow_le_pow_right (by norm_num) (min_le_right _ _))
  rw [groupSum_exact _ _ _ (by omega) hok hint _ hN hNb]
  have hfin' : ¬ ((intT (z₀ + zs)).magnitude ≠ 0 ∧ (intT (z₀ + zs)).exponent > FP32_MAX_EXPONENT) := by
    rintro ⟨h0, h1⟩
    by_cases hz0 : z₀ + zs = 0
    · simp [intT, hz0, ZERO] at h0
    · have := (intT_fields hz0 hN24).2.2.2
      simp only [FP32_MAX_EXPONENT] at h1
      omega
  obtain ⟨hlt, hrt⟩ := packFp32_intT hN24
  rw [ite_eq_right hfin', packValue, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hlt, hrt, intT_value hN24, ← hN,
    List.map_cons, List.sum_cons]

/-- **P1 on the H200's captured float atoms.** -/
theorem H200.p1Captured : H200.P1Captured :=
  ⟨p1Float_one .bf16 _ 16 rfl (by decide) (by decide) (by decide) (by decide),
    p1Float_one .e4m3 _ 32 rfl (by decide) (by decide) (by decide) (by decide),
    p1Float_one .e5m2 _ 32 rfl (by decide) (by decide) (by decide) (by decide)⟩

/-- **P1 on the RTX PRO 6000's captured float atoms.** -/
theorem Sm120.p1Captured : Sm120.P1Captured :=
  ⟨p1Float_one .bf16 _ 16 rfl (by decide) (by decide) (by decide) (by decide),
    p1Float_one .e4m3 _ 32 rfl (by decide) (by decide) (by decide) (by decide),
    p1Float_one .e5m2 _ 32 rfl (by decide) (by decide) (by decide) (by decide)⟩

/-- **P1 on E2M1's candidate pipeline.** -/
theorem p1Pipelines : P1Pipelines := p1Float_one .e2m1 _ 32 rfl (by decide) (by decide) (by decide) (by decide)

end Verity.Accel
