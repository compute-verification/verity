import Definitions.Core.TC.Spec
import Definitions.Core.TC.Relation

/-!
# One product: the relation's product rows are forced, and they are `bf16_product_total`

`prodCols Z a b` is the product row `step_witness` builds for the words `a`, `b`. `prodHolds` forces every field of a row
to it (`prodHolds_eq`), and it satisfies `prodHolds` (`prodHolds_prodCols`). `prod_value` reads the semantics' product
off it: its NaN and infinity flags, and in the finite case its sign, masked exponent and significand product.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- The class code `T_bf16` gives each class. -/
def clsCode : Cls → Int
  | .finite => 0
  | .zero => 1
  | .inf => 2
  | .nan => 3

/-- The effective exponent `e or 1` of a word. -/
def effExp (w : Nat) : Nat := if (w >>> 7) &&& 0xFF ≠ 0 then (w >>> 7) &&& 0xFF else 1

/-- The significand with the hidden bit. -/
def sig (w : Nat) : Nat := (w &&& 0x7F) ||| (if (w >>> 7) &&& 0xFF ≠ 0 then 0x80 else 0)

/-- `T_bf16`'s row for a word. -/
def bf16Row (w : Nat) : Int × Int × Int × Int :=
  ((w >>> 15 : Nat), if bf16Class w = .finite then (effExp w : Int) else 1,
    if bf16Class w = .finite then (sig w : Int) else 0, clsCode (bf16Class w))

theorem tBf16_eq (w : Nat) (hw : w < 2 ^ 16) : tBf16 (w : Int) = some (bf16Row w) := by
  have h0 : (0 : Int) ≤ (w : Int) ∧ (w : Int) < 2 ^ 16 := ⟨Int.natCast_nonneg w, by omega⟩
  unfold tBf16 bf16Row bf16Class effExp sig clsCode
  simp only [h0, and_self, ite_true, Int.toNat_natCast]
  by_cases he : (w >>> 7) &&& 0xFF = 0xFF
  · by_cases hm : w &&& 0x7F = 0 <;> simp [he, hm, CLS_NAN, CLS_INF]
  · by_cases he0 : (w >>> 7) &&& 0xFF = 0
    · by_cases hm : w &&& 0x7F = 0 <;> simp [he0, hm, CLS_ZERO, CLS_FIN]
    · simp [he, he0, CLS_FIN]

theorem sign_lt (w : Nat) (hw : w < 2 ^ 16) : w >>> 15 < 2 := by
  rw [Nat.shiftRight_eq_div_pow]; omega

theorem sig_lt (w : Nat) : sig w < 256 := by
  unfold sig
  apply Nat.or_lt_two_pow (n := 8)
  · exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)
  · split <;> decide

theorem effExp_le (w : Nat) : 1 ≤ effExp w ∧ effExp w ≤ 255 := by
  unfold effExp
  have : (w >>> 7) &&& 0xFF ≤ 0xFF := Nat.and_le_right
  split <;> omega

theorem effExp_fin (w : Nat) (h : bf16Class w = .finite) : effExp w ≤ 254 := by
  unfold bf16Class at h
  unfold effExp
  have : (w >>> 7) &&& 0xFF ≤ 0xFF := Nat.and_le_right
  by_cases he : (w >>> 7) &&& 0xFF = 0xFF
  · by_cases hm : w &&& 0x7F = 0 <;> simp [he, hm] at h
  · split <;> omega

/-- The product's sign bit, as `bf16_product` and `bf16_product_total` compute it. -/
theorem xor_sign (a b : Nat) (ha : a < 2 ^ 16) (hb : b < 2 ^ 16) :
    (((a ^^^ b) >>> 15) &&& 1 = 1) ↔ a >>> 15 ≠ b >>> 15 := by
  rw [Nat.shiftRight_xor_distrib]
  have h1 := sign_lt a ha
  have h2 := sign_lt b hb
  generalize a >>> 15 = x at *
  generalize b >>> 15 = y at *
  rcases (by omega : x = 0 ∨ x = 1) with rfl | rfl <;> rcases (by omega : y = 0 ∨ y = 1) with rfl | rfl <;> decide

/-- The row `T_pcls` gives, as a function of the classes and the product's sign. -/
def pclsRow (ca cb : Cls) (sp : Bool) : Int × Int × Int × Int × Int :=
  let s : Int := if sp then 1 else 0
  if ca = .nan ∨ cb = .nan ∨ (ca = .inf ∧ cb = .zero) ∨ (cb = .inf ∧ ca = .zero) then (1, 0, 0, 1, s)
  else if ca = .inf ∨ cb = .inf then (0, 1 - s, s, 1, s)
  else (0, 0, 0, if ca ≠ .finite ∨ cb ≠ .finite then 1 else 0, s)

theorem tPcls_eq (ca cb : Cls) (sa sb : Nat) (hsa : sa < 2) (hsb : sb < 2) :
    tPcls (clsCode ca, clsCode cb, (sa : Int), (sb : Int)) = some (pclsRow ca cb (sa ≠ sb)) := by
  rcases (by omega : sa = 0 ∨ sa = 1) with rfl | rfl <;> rcases (by omega : sb = 0 ∨ sb = 1) with rfl | rfl <;>
    cases ca <;> cases cb <;> decide

/-- The product row `step_witness` builds for the words `a`, `b` (floor `Z`). -/
def prodCols (Z : Int) (a b : Nat) : ProdCols :=
  let ra := bf16Row a
  let rb := bf16Row b
  let pc := pclsRow (bf16Class a) (bf16Class b) (a >>> 15 ≠ b >>> 15)
  let ep := ra.2.1 + rb.2.1 - 254
  { sa := ra.1, ea := ra.2.1, SA := ra.2.2.1, ca := ra.2.2.2
    sb := rb.1, eb := rb.2.1, SB := rb.2.2.1, cb := rb.2.2.2
    nan := pc.1, pinf := pc.2.1, ninf := pc.2.2.1, zero := pc.2.2.2.1, sp := pc.2.2.2.2
    P := ra.2.2.1 * rb.2.2.1, ep := ep, em := ep + pc.2.2.2.1 * (Z - ep) }

theorem prodHolds_prodCols (p : Params) (a b : Nat) (ha : a < 2 ^ 16) (hb : b < 2 ^ 16) :
    prodHolds p a b (prodCols p.Z a b) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact tBf16_eq a ha
  · exact tBf16_eq b hb
  · exact tPcls_eq _ _ _ _ (sign_lt a ha) (sign_lt b hb)
  · rfl
  · simp [LIN, prodCols]
  · simp only [MUL, prodCols]; omega

theorem prodHolds_eq (p : Params) (a b : Nat) (ha : a < 2 ^ 16) (hb : b < 2 ^ 16) (q : ProdCols)
    (h : prodHolds p a b q) : q = prodCols p.Z a b := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  unfold LOOKUP at h1 h2 h3
  unfold MUL at h4 h6
  unfold LIN at h5
  rw [tBf16_eq a ha] at h1
  rw [tBf16_eq b hb] at h2
  cases q with
  | mk sa ea SA ca sb eb SB cb nan pinf ninf zero sp P ep em =>
  simp only [bf16Row, Option.some.injEq, Prod.mk.injEq] at h1 h2
  obtain ⟨rfl, rfl, rfl, rfl⟩ := h1
  obtain ⟨rfl, rfl, rfl, rfl⟩ := h2
  simp only at h3 h4 h5 h6
  rw [tPcls_eq _ _ _ _ (sign_lt a ha) (sign_lt b hb)] at h3
  simp only [Option.some.injEq] at h3
  have hep := Int.sub_eq_zero.mp h5
  have hem := Int.sub_eq_iff_eq_add.mp h6.symm
  subst hep h4 hem
  simp only [prodCols, bf16Row, h3]
  congr 1
  exact Int.add_comm _ _

theorem sig_ne_zero (w : Nat) (h : bf16Class w = .finite) : sig w ≠ 0 := by
  unfold bf16Class at h
  unfold sig
  by_cases he : (w >>> 7) &&& 0xFF = 0
  · have hm : w &&& 0x7F ≠ 0 := by
      intro hm; simp [he, hm] at h
    simpa [he] using hm
  · simp only [he, ne_eq, not_false_eq_true, ite_true]
    have := @Nat.right_le_or (w &&& 0x7F) 0x80
    omega

theorem sig_zero (w : Nat) (h : bf16Class w = .zero) : sig w = 0 ∧ effExp w = 1 := by
  unfold bf16Class at h
  by_cases he : (w >>> 7) &&& 0xFF = 0xFF
  · by_cases hm : w &&& 0x7F = 0 <;> simp [he, hm] at h
  · by_cases hz : (w >>> 7) &&& 0xFF = 0 ∧ w &&& 0x7F = 0
    · simp [sig, effExp, hz.1, hz.2]
    · simp [he, hz] at h

/-- `bf16_product_total` by the operands' classes. -/
theorem bf16ProductTotal_eq (a b : Nat) (ha : a < 2 ^ 16) (hb : b < 2 ^ 16) :
    bf16ProductTotal a b =
      if bf16Class a = .nan ∨ bf16Class b = .nan then .nan
      else if bf16Class a = .inf ∨ bf16Class b = .inf then
        (if bf16Class a = .zero ∨ bf16Class b = .zero then .nan else .inf (a >>> 15 ≠ b >>> 15))
      else .fin (bf16Product a b) := by
  unfold bf16ProductTotal
  have hs := xor_sign a b ha hb
  simp only
  split
  · rfl
  split
  · split
    · rfl
    · congr 1; exact decide_eq_decide.mpr hs
  · rfl

/-- `bf16_product` through `sig` and `effExp`. -/
theorem bf16Product_eq (a b : Nat) :
    bf16Product a b =
      if sig a * sig b = 0 then ZERO
      else ⟨((a ^^^ b) >>> 15) &&& 1 = 1, (effExp a : Int) + (effExp b : Int) - 254, sig a * sig b * 2 ^ 9⟩ := by
  show (if (sig a * sig b) <<< 9 = 0 then ZERO
      else ⟨((a ^^^ b) >>> 15) &&& 1 = 1, (effExp a : Int) + (effExp b : Int) - 254, (sig a * sig b) <<< 9⟩) = _
  rw [Nat.shiftLeft_eq]
  by_cases h : sig a * sig b = 0
  · simp [h]
  · have h' : ¬ sig a * sig b * 2 ^ 9 = 0 := by omega
    simp only [h, h', ite_false]

def _root_.Verity.TC.Spec.Value.isInf (s : Bool) : Value → Bool
  | .inf n => n == s
  | _ => false

/-- What the finite path of the relation reads off a term: its sign, masked exponent and magnitude. -/
structure TermOK (Z : Int) (t : Term) (neg e m : Int) : Prop where
  mag : (t.magnitude : Int) = m
  exp : t.magnitude ≠ 0 → t.exponent = e
  sign : t.magnitude ≠ 0 → (t.negative ↔ neg = 1)
  zero : t.magnitude = 0 → e = Z
  neg01 : neg = 0 ∨ neg = 1

/-- Everything the group reads from one product row, against `bf16_product_total`. -/
structure ProdOK (Z : Int) (pv : Value) (c : ProdCols) : Prop where
  nan : c.nan = if pv.isNaN then 1 else 0
  pinf : c.pinf = if pv.isInf false then 1 else 0
  ninf : c.ninf = if pv.isInf true then 1 else 0
  P_nonneg : 0 ≤ c.P
  P_lt : c.P < 2 ^ 16
  em_lo : -252 ≤ c.em ∨ c.em = Z
  em_hi : c.em ≤ 254 ∨ c.em = Z
  sp01 : c.sp = 0 ∨ c.sp = 1
  special : pv.term? = none → c.P = 0 ∧ c.em = Z
  fin : ∀ t, pv = .fin t → TermOK Z t c.sp c.em (c.P * 2 ^ 9)

theorem prodOK (Z : Int) (a b : Nat) (ha : a < 2 ^ 16) (hb : b < 2 ^ 16) :
    ProdOK Z (bf16ProductTotal a b) (prodCols Z a b) := by
  rw [bf16ProductTotal_eq a b ha hb, bf16Product_eq]
  have ea := effExp_le a
  have eb := effExp_le b
  have sa := sig_lt a
  have sb := sig_lt b
  have hs := xor_sign a b ha hb
  generalize hca : bf16Class a = ca
  generalize hcb : bf16Class b = cb
  have fa : ca = .finite → sig a ≠ 0 := fun h => sig_ne_zero a (hca ▸ h)
  have ga : ca = .finite → effExp a ≤ 254 := fun h => effExp_fin a (hca ▸ h)
  have gb : cb = .finite → effExp b ≤ 254 := fun h => effExp_fin b (hcb ▸ h)
  have fb : cb = .finite → sig b ≠ 0 := fun h => sig_ne_zero b (hcb ▸ h)
  have za : ca = .zero → sig a = 0 ∧ effExp a = 1 := fun h => sig_zero a (hca ▸ h)
  have zb : cb = .zero → sig b = 0 ∧ effExp b = 1 := fun h => sig_zero b (hcb ▸ h)
  simp only [prodCols, bf16Row, pclsRow, hca, hcb]
  generalize hsg : decide (a >>> 15 ≠ b >>> 15) = sgn
  have hsg' : (((a ^^^ b) >>> 15) &&& 1 = 1) ↔ sgn = true := by rw [hs, ← hsg]; simp
  generalize sig a = A at *
  generalize sig b = B at *
  generalize effExp a = EA at *
  generalize effExp b = EB at *
  cases ca <;> cases cb <;> cases sgn <;>
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun t ht => ?_⟩ <;>
    simp_all [Value.isNaN, Value.isInf, Value.term?] <;>
    (try omega)
  all_goals first
    | (subst ht; exact ⟨by simp [ZERO], by simp [ZERO], by simp [ZERO], fun _ => by omega, by omega⟩)
    | (have hAB : A * B ≠ 0 := Nat.mul_ne_zero fa fb
       simp only [hAB, ite_false] at ht
       subst ht
       exact ⟨by simp, fun _ => rfl, fun _ => by simp, fun h => absurd h (by simp; omega), by simp⟩)
    | (have h := Nat.mul_lt_mul'' sa sb
       rw [← Int.natCast_mul]
       generalize A * B = AB at h ⊢
       omega)

end Verity.TC.Proofs
