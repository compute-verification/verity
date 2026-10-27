import Properties.Evaluate

/-!
# Proofs of the evaluator's equalities (`Properties.Evaluate`)

A copy is its definition by unfolding (`rfl`: the kernel unfolds both sides to the same term). Where the evaluator masks
an operand that Security reads unmasked, the bit fields both read are rewritten into division and remainder by powers of
two (`Bits`), which `omega` decides. Where the types are the evaluator's own, each function commutes with `toCore` by
cases on the constructors, and the lists by `List.foldl_map`, `List.filterMap_map` and induction on the groups.
-/

namespace Properties.Evaluator

/-! ## Bit fields as division and remainder -/

namespace Bits

theorem and_m32 (x : Nat) : x &&& 0xFFFFFFFF = x % 4294967296 := Nat.and_two_pow_sub_one_eq_mod x 32
theorem and_ffff (x : Nat) : x &&& 0xFFFF = x % 65536 := Nat.and_two_pow_sub_one_eq_mod x 16
theorem and_ff (x : Nat) : x &&& 0xFF = x % 256 := Nat.and_two_pow_sub_one_eq_mod x 8
theorem and_mant (x : Nat) : x &&& 0x7FFFFF = x % 8388608 := Nat.and_two_pow_sub_one_eq_mod x 23
theorem and_one (x : Nat) : x &&& 1 = x % 2 := Nat.and_two_pow_sub_one_eq_mod x 1
theorem shr16 (x : Nat) : x >>> 16 = x / 65536 := Nat.shiftRight_eq_div_pow x 16
theorem shr23 (x : Nat) : x >>> 23 = x / 8388608 := Nat.shiftRight_eq_div_pow x 23
theorem shr31 (x : Nat) : x >>> 31 = x / 2147483648 := Nat.shiftRight_eq_div_pow x 31

/-- The exponent field of an FP32 word, in place. -/
theorem and_exp (x : Nat) : x &&& 0x7F800000 = x / 8388608 % 256 * 8388608 := by
  rw [show (0x7F800000 : Nat) = (2 ^ 8 - 1) * 2 ^ 23 from rfl, show (8388608 : Nat) = 2 ^ 23 from rfl,
    show (256 : Nat) = 2 ^ 8 from rfl]
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_and, Nat.testBit_mul_two_pow, Nat.testBit_mod_two_pow, Nat.testBit_div_two_pow,
    Nat.testBit_two_pow_sub_one]
  by_cases h : 23 ≤ i
  · simp [h, show i - 23 + 23 = i by omega, Bool.and_comm]
  · simp [h]

end Bits

open Bits

/-! ## Scalar -/

namespace Scalar

theorem f32IsNan : F32IsNan := rfl
theorem order : Order := rfl
theorem f32NanPtx : F32NanPtx := rfl
theorem f32Fabs : F32Fabs := rfl
theorem f32Neg : F32Neg := rfl
theorem f32Fmaxf : F32Fmaxf := rfl
theorem i32Le : I32Le := rfl
theorem i32Add : I32Add := rfl
theorem bitAnd : BitAnd := rfl
theorem selectI32 : SelectI32 := rfl
theorem selectBf16 : SelectBf16 := rfl

end Scalar

/-! ## FP32 and BF16 -/

namespace Fp32

/-- `toCore` and `ofCore` are inverse to each other, so the `toCore` equalities say the two `Decoded` agree. -/
theorem decodedCorresponds :
    (∀ d, Decoded.ofCore (Decoded.toCore d) = d) ∧ ∀ d, Decoded.toCore (Decoded.ofCore d) = d :=
  ⟨fun d => by cases d <;> rfl, fun d => by cases d <;> rfl⟩

theorem f32Nan : F32Nan := rfl
theorem bf16NanTorch : Bf16NanTorch := rfl
theorem bf16NanF2fp : Bf16NanF2fp := rfl
theorem zeroWord : ZeroWord := rfl
theorem infWord : InfWord := rfl
theorem roundRn : RoundRn := rfl
theorem roundSum : RoundSum := rfl
theorem f32GpuNan : F32GpuNan := rfl
theorem ftzBits : FtzBits := rfl
theorem f32Gt : F32Gt := rfl
theorem f32MaxBits : F32MaxBits := rfl
theorem f32Max : F32Max := rfl
theorem guardNegInfZero : GuardNegInfZero := rfl
theorem bf16ToF32Bits : Bf16ToF32Bits := rfl
theorem f32ToBf16RnBits : F32ToBf16RnBits := rfl

/-- The bits `Verity.Fp32.decode` reads, 0 to 31, are those of the word's low 32 bits. -/
theorem decode_mask (u : Nat) : Verity.Fp32.decode (u &&& 0xFFFFFFFF) = Verity.Fp32.decode u := by
  have h1 : u % 4294967296 / 8388608 % 256 = u / 8388608 % 256 := by omega
  have h2 : u % 4294967296 % 8388608 = u % 8388608 := by omega
  have h3 : u % 4294967296 / 2147483648 % 2 = u / 2147483648 % 2 := by omega
  simp only [Verity.Fp32.decode, and_m32, and_ff, and_mant, and_one, shr23, shr31, h1, h2, h3]

theorem decode_toCore (u : Nat) : Decoded.toCore (Evaluate.Fp32.decode u) = Verity.Fp32.decode u := by
  rw [← decode_mask]
  simp only [Evaluate.Fp32.decode, Verity.Fp32.decode, apply_ite Decoded.toCore]
  rfl

theorem decode : Decode := decode_toCore

theorem mul_toCore (x y : Evaluate.Fp32.Decoded) :
    Decoded.toCore (x.mul y) = (Decoded.toCore x).mul (Decoded.toCore y) := by
  cases x <;> cases y <;>
    simp only [Evaluate.Fp32.Decoded.mul, Verity.Fp32.Decoded.mul, apply_ite Decoded.toCore] <;> rfl

theorem decodedMul : DecodedMul := mul_toCore

theorem addRn_toCore (nan d : Nat) (x y : Evaluate.Fp32.Decoded) :
    x.addRn nan d y = (Decoded.toCore x).addRn nan d (Decoded.toCore y) := by
  cases x <;> cases y <;> rfl

theorem decodedAddRn : DecodedAddRn := addRn_toCore

theorem addWith : AddWith := by
  funext a b
  simp only [Evaluate.Fp32.addWith, Verity.Fp32.f32AddV2, addRn_toCore, decode_toCore]
  rfl

theorem mulWith : MulWith := by
  funext a b
  simp only [Evaluate.Fp32.mulWith, Verity.Fp32.f32MulV2, ← decode_toCore, ← mul_toCore]
  cases Evaluate.Fp32.Decoded.mul (Evaluate.Fp32.decode a) (Evaluate.Fp32.decode b) <;> rfl

theorem divWith : DivWith := by
  funext a b
  simp only [Evaluate.Fp32.divWith, Verity.Fp32.f32DivV2, ← decode_toCore]
  cases Evaluate.Fp32.decode a <;> cases Evaluate.Fp32.decode b <;> rfl

theorem fmaWith : FmaWith := by
  funext a b c
  simp only [Evaluate.Fp32.fmaWith, Verity.Fp32.f32FmaBits, ← decode_toCore]
  rw [addRn_toCore, mul_toCore]
  cases Evaluate.Fp32.decode c <;> rfl

theorem f32AddV2 : F32AddV2 := addWith
theorem f32MulV2 : F32MulV2 := mulWith
theorem f32DivV2 : F32DivV2 := divWith
theorem f32FmaBits : F32FmaBits := fmaWith

theorem f32Fma : F32Fma := by
  funext a b c
  simp only [Evaluate.Fp32.f32Fma, Verity.Fp32.f32Fma, show Evaluate.Fp32.f32FmaBits = _ from f32FmaBits]

theorem f32FmaFtz : F32FmaFtz := by
  funext a b c
  simp only [Evaluate.Fp32.f32FmaFtz, Verity.Fp32.f32FmaFtz, show Evaluate.Fp32.f32FmaBits = _ from f32FmaBits]
  rfl

theorem f32FmaSubFtz : F32FmaSubFtz := by
  funext a b c
  simp only [Evaluate.Fp32.f32FmaSubFtz, Verity.Fp32.f32FmaSubFtz,
    show Evaluate.Fp32.f32FmaBits = _ from f32FmaBits]
  rfl

theorem f32FmaV2 : F32FmaV2 := by
  funext a b c
  simp only [Evaluate.Fp32.f32FmaV2, Verity.Fp32.f32FmaV2, show Evaluate.Fp32.f32FmaBits = _ from f32FmaBits]
  rfl

theorem f32AddFtzV2 : F32AddFtzV2 := by
  funext a b
  simp only [Evaluate.Fp32.f32AddFtzV2, Verity.Fp32.f32AddFtzV2, show Evaluate.Fp32.f32AddV2 = _ from f32AddV2]
  rfl

theorem f32SubFtzV2 : F32SubFtzV2 := by
  funext a b
  simp only [Evaluate.Fp32.f32SubFtzV2, Verity.Fp32.f32SubFtzV2, show Evaluate.Fp32.f32AddV2 = _ from f32AddV2]
  rfl

theorem f32MulFtzV2 : F32MulFtzV2 := by
  funext a b
  simp only [Evaluate.Fp32.f32MulFtzV2, Verity.Fp32.f32MulFtzV2, show Evaluate.Fp32.f32MulV2 = _ from f32MulV2]
  rfl

theorem f32FmaFtzV2 : F32FmaFtzV2 := by
  funext a b c
  simp only [Evaluate.Fp32.f32FmaFtzV2, Verity.Fp32.f32FmaFtzV2, show Evaluate.Fp32.f32FmaFtz = _ from f32FmaFtz]
  rfl

theorem f32FmaSubFtzV2 : F32FmaSubFtzV2 := by
  funext a b c
  simp only [Evaluate.Fp32.f32FmaSubFtzV2, Verity.Fp32.f32FmaSubFtzV2,
    show Evaluate.Fp32.f32FmaSubFtz = _ from f32FmaSubFtz]
  rfl

theorem bf16Add : Bf16Add := by
  funext a b
  simp only [Evaluate.Fp32.bf16Add, Verity.Fp32.bf16Add, show Evaluate.Fp32.f32AddV2 = _ from f32AddV2]
  rfl

theorem bf16AddF2fp : Bf16AddF2fp := by
  funext a b
  simp only [Evaluate.Fp32.bf16AddF2fp, Verity.Fp32.bf16AddF2fp, show Evaluate.Fp32.f32AddV2 = _ from f32AddV2]
  rfl

theorem bf16MulF32 : Bf16MulF32 := by
  funext a s
  simp only [Evaluate.Fp32.bf16MulF32, Verity.Fp32.bf16MulF32, show Evaluate.Fp32.f32MulV2 = _ from f32MulV2]
  rfl

theorem bf16MulBf16 : Bf16MulBf16 := by
  funext a b
  simp only [Evaluate.Fp32.bf16MulBf16, Verity.Fp32.bf16MulBf16, show Evaluate.Fp32.f32MulV2 = _ from f32MulV2]
  rfl

end Fp32

/-! ## Casts -/

namespace Casts

theorem bitLength : BitLength := rfl
theorem bf16ToF32 : Bf16ToF32 := rfl
theorem f32ToBf16Rn : F32ToBf16Rn := rfl
theorem e4m3Nan : E4m3Nan := rfl
theorem e4m3ToF32 : E4m3ToF32 := rfl
theorem dyadicLt : DyadicLt := rfl
theorem roundEvenDyadic : RoundEvenDyadic := rfl
theorem f32ToE4m3Sat : F32ToE4m3Sat := rfl
theorem e2m1NanCode : E2m1NanCode := rfl
theorem e2m1Up : E2m1Up := rfl
theorem f32ToE2m1Sat : F32ToE2m1Sat := rfl

theorem fp32Class_nan (w : Nat) :
    Verity.TC.Spec.fp32Class w = .nan ↔ (w >>> 23) &&& 0xFF = 0xFF ∧ w &&& 0x7FFFFF ≠ 0 := by
  unfold Verity.TC.Spec.fp32Class
  by_cases h₁ : (w >>> 23) &&& 0xFF = 0xFF <;> by_cases h₂ : w &&& 0x7FFFFF = 0 <;> simp [h₁, h₂] <;> split <;> simp

/-- Bits 16 to 31 of `s + 2^32 q` are those of `s`. -/
theorem high_half_mod (q s : Nat) : (s + 4294967296 * q) / 65536 % 65536 = s / 65536 % 65536 := by
  rw [show s + 4294967296 * q = s + 65536 * q * 65536 by omega, Nat.add_mul_div_right _ _ (by decide),
    Nat.add_mul_mod_self_left]

theorem f2fpBf16 : F2fpBf16 := by
  funext a
  have h₁ : a % 4294967296 / 8388608 % 256 = a / 8388608 % 256 := by omega
  have h₂ : a % 4294967296 % 8388608 = a % 8388608 := by omega
  have h₃ : a % 4294967296 / 65536 % 2 = a / 65536 % 2 := by omega
  have h₄ : (a % 4294967296 + 32767 + a / 65536 % 2) / 65536 % 65536 = (a + 32767 + a / 65536 % 2) / 65536 % 65536 := by
    rw [← high_half_mod (a / 4294967296) (a % 4294967296 + 32767 + a / 65536 % 2)]
    congr 2
    omega
  simp only [Evaluate.Casts.f2fpBf16, Evaluate.Fp32.f32ToBf16RnBits, Evaluate.Fp32.BF16_NAN_F2FP,
    Verity.Prims.f2fpBf16, Verity.TC.Spec.cvtRnBf16F32, fp32Class_nan, and_m32, and_exp, and_ffff, and_ff, and_mant,
    and_one, shr16, shr23, h₁, h₂, h₃, h₄]
  split <;> split <;> omega

end Casts

/-! ## The BF16 tensor-core step -/

namespace TC

/-- Each `toCore` has the inverse `ofCore`, so the `toCore` equalities say the two pipelines, terms, values and classes agree. -/
theorem typesCorrespond :
    ((∀ p, Pipeline.ofCore (Pipeline.toCore p) = p) ∧ ∀ p, Pipeline.toCore (Pipeline.ofCore p) = p) ∧
    ((∀ t, Term.ofCore (Term.toCore t) = t) ∧ ∀ t, Term.toCore (Term.ofCore t) = t) ∧
    ((∀ v, Value.ofCore (Value.toCore v) = v) ∧ ∀ v, Value.toCore (Value.ofCore v) = v) ∧
    ((∀ c, Cls.ofCore (Cls.toCore c) = c) ∧ ∀ c, Cls.toCore (Cls.ofCore c) = c) :=
  ⟨⟨fun _ => rfl, fun _ => rfl⟩, ⟨fun _ => rfl, fun _ => rfl⟩,
    ⟨fun v => by cases v <;> rfl, fun v => by cases v <;> rfl⟩, ⟨fun c => by cases c <;> rfl, fun c => by cases c <;> rfl⟩⟩

theorem bitLength : BitLength := rfl
theorem pipelineK : PipelineK := fun _ => rfl
theorem hopperBf16WgmmaK16 : HopperBf16WgmmaK16 := rfl
theorem ampereBf16M16n8k16 : AmpereBf16M16n8k16 := rfl
theorem zero : Zero := rfl
theorem packFp32 : PackFp32 := fun _ => rfl

theorem bf16Product_toCore (a b : Nat) :
    Term.toCore (Evaluate.TC.Spec.bf16Product a b) = Verity.TC.Spec.bf16Product a b := by
  simp only [Evaluate.TC.Spec.bf16Product, Verity.TC.Spec.bf16Product, apply_ite Term.toCore]
  rfl

theorem bf16Product : Bf16Product := bf16Product_toCore

theorem fp32Term_toCore (bits : Nat) : Term.toCore (Evaluate.TC.Spec.fp32Term bits) = Verity.TC.Spec.fp32Term bits := by
  simp only [Evaluate.TC.Spec.fp32Term, Verity.TC.Spec.fp32Term, apply_ite Term.toCore]
  rfl

theorem fp32Term : Fp32Term := fp32Term_toCore

theorem groupSum_toCore (terms : List Evaluate.TC.Spec.Term) (width : Nat) (zeroExponent : Int) :
    Term.toCore (Evaluate.TC.Spec.groupSum terms width zeroExponent) =
      Verity.TC.Spec.groupSum (terms.map Term.toCore) width zeroExponent := by
  simp only [Evaluate.TC.Spec.groupSum, Verity.TC.Spec.groupSum, List.foldl_map, apply_ite Term.toCore]
  rfl

theorem groupSum : GroupSum := groupSum_toCore

theorem bf16Class_toCore (w : Nat) : Cls.toCore (Evaluate.TC.Spec.bf16Class w) = Verity.TC.Spec.bf16Class w := by
  simp only [Evaluate.TC.Spec.bf16Class, Verity.TC.Spec.bf16Class, apply_ite Cls.toCore]
  rfl

theorem bf16Class : Bf16Class := bf16Class_toCore

theorem fp32Class_toCore (w : Nat) : Cls.toCore (Evaluate.TC.Spec.fp32Class w) = Verity.TC.Spec.fp32Class w := by
  simp only [Evaluate.TC.Spec.fp32Class, Verity.TC.Spec.fp32Class, apply_ite Cls.toCore]
  rfl

theorem fp32Class : Fp32Class := fp32Class_toCore

theorem bf16ProductTotal_toCore (a b : Nat) :
    Value.toCore (Evaluate.TC.Spec.bf16ProductTotal a b) = Verity.TC.Spec.bf16ProductTotal a b := by
  simp only [Evaluate.TC.Spec.bf16ProductTotal, Verity.TC.Spec.bf16ProductTotal, ← bf16Class_toCore,
    ← bf16Product_toCore]
  cases Evaluate.TC.Spec.bf16Class a <;> cases Evaluate.TC.Spec.bf16Class b <;> rfl

theorem bf16ProductTotal : Bf16ProductTotal := bf16ProductTotal_toCore

theorem fp32Value_toCore (bits : Nat) :
    Value.toCore (Evaluate.TC.Spec.fp32Value bits) = Verity.TC.Spec.fp32Value bits := by
  simp only [Evaluate.TC.Spec.fp32Value, Verity.TC.Spec.fp32Value, ← fp32Class_toCore, ← fp32Term_toCore]
  cases Evaluate.TC.Spec.fp32Class bits <;> rfl

theorem fp32Value : Fp32Value := fp32Value_toCore

theorem isNaN_toCore : Verity.TC.Spec.Value.isNaN ∘ Value.toCore = Evaluate.TC.Spec.Value.isNaN :=
  funext fun v => by cases v <;> rfl

theorem infSign_toCore : Verity.TC.Spec.Value.infSign? ∘ Value.toCore = Evaluate.TC.Spec.Value.infSign? :=
  funext fun v => by cases v <;> rfl

theorem term_toCore :
    Verity.TC.Spec.Value.term? ∘ Value.toCore = fun v => (Evaluate.TC.Spec.Value.term? v).map Term.toCore :=
  funext fun v => by cases v <;> rfl

theorem valueIsNaN : ValueIsNaN := fun v => by cases v <;> rfl
theorem valueInfSign : ValueInfSign := fun v => by cases v <;> rfl
theorem valueTerm : ValueTerm := fun v => by cases v <;> rfl

theorem groupStepTotal_toCore (acc : Evaluate.TC.Spec.Value) (products : List Evaluate.TC.Spec.Value) (width : Nat)
    (zeroExponent : Int) :
    Value.toCore (Evaluate.TC.Spec.groupStepTotal acc products width zeroExponent) =
      Verity.TC.Spec.groupStepTotal (Value.toCore acc) (products.map Value.toCore) width zeroExponent := by
  simp only [Evaluate.TC.Spec.groupStepTotal, Verity.TC.Spec.groupStepTotal,
    show Value.toCore acc :: products.map Value.toCore = (acc :: products).map Value.toCore from rfl, List.any_map,
    List.filterMap_map, isNaN_toCore, infSign_toCore, term_toCore, ← List.map_filterMap, ← groupSum_toCore,
    apply_ite Value.toCore]
  cases (acc :: products).filterMap Evaluate.TC.Spec.Value.infSign? <;> (try simp only [apply_ite Value.toCore]) <;> rfl

theorem groupStepTotal : GroupStepTotal := groupStepTotal_toCore

theorem packValue_toCore (v : Evaluate.TC.Spec.Value) :
    Evaluate.TC.Spec.packValue v = Verity.TC.Spec.packValue (Value.toCore v) := by
  cases v <;> rfl

theorem packValue : PackValue := packValue_toCore

theorem runGroups_toCore (width : Nat) (zeroExponent : Int) (groups : List Nat) (acc : Evaluate.TC.Spec.Value)
    (a b : List Nat) :
    Value.toCore (Evaluate.TC.Spec.runGroups width zeroExponent groups acc a b) =
      Verity.TC.Spec.runGroups width zeroExponent groups (Value.toCore acc) a b := by
  induction groups generalizing acc a b with
  | nil => rfl
  | cons size rest ih =>
    simp only [Evaluate.TC.Spec.runGroups, Verity.TC.Spec.runGroups, ih, groupStepTotal_toCore, List.map_zipWith,
      bf16ProductTotal_toCore]

theorem runGroups : RunGroups := runGroups_toCore

theorem tcDotTotal_toCore (p : Evaluate.TC.Spec.Pipeline) (acc : Nat) (a b : List Nat) :
    Evaluate.TC.Spec.tcDotTotal p acc a b = Verity.TC.Spec.tcDotTotal (Pipeline.toCore p) acc a b := by
  simp only [Evaluate.TC.Spec.tcDotTotal, Verity.TC.Spec.tcDotTotal, packValue_toCore, runGroups_toCore,
    fp32Value_toCore]
  rfl

theorem tcDotTotal : TcDotTotal := tcDotTotal_toCore

theorem ampereBf16Dot16 : AmpereBf16Dot16 := by
  funext acc a b
  exact tcDotTotal_toCore _ acc a b

theorem hopperBf16Dot16 : HopperBf16Dot16 := by
  funext acc a b
  exact tcDotTotal_toCore _ acc a b

end TC

/-! ## The Boolean gates on bits -/

namespace Gates

theorem andBits : AndBits := by
  intro xs
  have h : (fun b : Nat => b &&& 1 == 1) ∘ Bool.toNat = id := funext fun b => by cases b <;> rfl
  simp only [Evaluate.andBits, Array.all_map, h]
  cases xs.all id <;> rfl

theorem xor_toNat (a b : Bool) : a.toNat ^^^ b.toNat = (a ^^ b).toNat := by
  cases a <;> cases b <;> rfl

theorem foldl_xor (l : List Bool) (acc : Bool) :
    (l.map Bool.toNat).foldl (· ^^^ ·) acc.toNat = (acc ^^ l.foldr Bool.xor false).toNat := by
  induction l generalizing acc with
  | nil => simp
  | cons x l ih => simp only [List.map_cons, List.foldl_cons, List.foldr_cons, xor_toNat, ih, Bool.xor_assoc]

theorem xorBits : XorBits := by
  intro xs
  rw [Evaluate.xorBits, ← Array.foldl_toList, Array.toList_map, show (0 : Nat) = false.toNat from rfl, foldl_xor,
    Bool.false_xor]
  cases xs.toList.foldr Bool.xor false <;> rfl

theorem notBit : NotBit := fun b => by cases b <;> rfl

end Gates

end Properties.Evaluator
