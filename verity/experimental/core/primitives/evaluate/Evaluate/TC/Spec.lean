/-!
# The BF16 tensor-core steps: `verity.ml.circuits.silicon.total.tc_dot_total`

A copy of `Security/Definitions/Core/TC/Pipeline.lean` and `Spec.lean`, definition for definition: the grouped-sum
pipelines (`models.GroupSum`) and the total semantics of one BF16 step on every encoding (`term.py`, `models.group_sum`,
`total.py`). Each definition names the Security definition it must equal, to be proved where the equalities live, since code
never requires `Security/`. `groupSum` is the generic adder (a width below 24 truncates the terms onto it), which
`Evaluate.TC.Fp8Atom` copies again in the form Security's FP8 definitions give it.

Words are natural numbers below `2^32` (FP32) and `2^16` (BF16), as the walk gives them to a gate whose operands are
`Value<32>` and `Array<16, Value<16>>` words.
-/

namespace Evaluate.TC.Spec

/-- Python's `int.bit_length`. Must equal `Verity.TC.bitLength`. -/
def bitLength (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- `models.GroupSum` for a BF16 pipeline. Must equal `Verity.TC.Pipeline`. -/
structure Pipeline where
  groups : List Nat
  width : Nat
  zeroExponent : Int
deriving DecidableEq

/-- `GroupSum.k`. Must equal `Verity.TC.Pipeline.k`. -/
def Pipeline.k (p : Pipeline) : Nat := p.groups.sum

/-- `models.HOPPER_BF16_WGMMA_K16`. Must equal `Verity.TC.HOPPER_BF16_WGMMA_K16`. -/
def HOPPER_BF16_WGMMA_K16 : Pipeline := { groups := [16], width := 26, zeroExponent := -133 }

/-- `models.AMPERE_BF16_M16N8K16`. Must equal `Verity.TC.AMPERE_BF16_M16N8K16`. -/
def AMPERE_BF16_M16N8K16 : Pipeline := { groups := [8, 8], width := 25, zeroExponent := -132 }

def FP32_SIGNIFICAND_WIDTH : Nat := 24
def FP32_MIN_NORMAL_EXPONENT : Int := -126
def FP32_MAX_EXPONENT : Int := 127

/-- `term.Term`: `(-1)^negative * magnitude * 2^(exponent - 23)`. Must equal `Verity.TC.Spec.Term`. -/
structure Term where
  negative : Bool
  exponent : Int
  magnitude : Nat

/-- `term.ZERO`. Must equal `Verity.TC.Spec.ZERO`. -/
def ZERO : Term := ⟨false, 0, 0⟩

/-- `term.bf16_product` on finite operands. Must equal `Verity.TC.Spec.bf16Product`. -/
def bf16Product (a b : Nat) : Term :=
  let ea := (a >>> 7) &&& 0xFF
  let eb := (b >>> 7) &&& 0xFF
  let sa := (a &&& 0x7F) ||| (if ea ≠ 0 then 0x80 else 0)
  let sb := (b &&& 0x7F) ||| (if eb ≠ 0 then 0x80 else 0)
  let magnitude := (sa * sb) <<< 9
  if magnitude = 0 then ZERO
  else
    ⟨((a ^^^ b) >>> 15) &&& 1 = 1,
     ((if ea ≠ 0 then ea else 1 : Nat) : Int) + ((if eb ≠ 0 then eb else 1 : Nat) : Int) - 254,
     magnitude⟩

/-- `term.fp32_term` on finite words. Must equal `Verity.TC.Spec.fp32Term`. -/
def fp32Term (bits : Nat) : Term :=
  let exponentField := (bits >>> 23) &&& 0xFF
  let mantissa := bits &&& 0x7FFFFF
  let negative := bits >>> 31 = 1
  if exponentField = 0 then
    if mantissa = 0 then ZERO else ⟨negative, FP32_MIN_NORMAL_EXPONENT, mantissa⟩
  else ⟨negative, (exponentField : Int) - 127, (1 <<< 23) ||| mantissa⟩

/-- `term.pack_fp32`. Must equal `Verity.TC.Spec.packFp32`. -/
def packFp32 (t : Term) : Nat :=
  if t.magnitude = 0 then 0
  else if t.exponent > FP32_MAX_EXPONENT then ((if t.negative then 1 else 0) <<< 31) ||| 0x7F800000
  else
    let exponentField : Int := t.exponent + 127
    let exponentField := if t.magnitude &&& (1 <<< 23) = 0 then exponentField - 1 else exponentField
    ((if t.negative then 1 else 0) <<< 31) ||| (exponentField.toNat <<< 23) ||| (t.magnitude &&& 0x7FFFFF)

/-- `models.group_sum`. Must equal `Verity.TC.Spec.groupSum`. -/
def groupSum (terms : List Term) (width : Nat) (zeroExponent : Int) : Term :=
  let rescale : Int := (width : Int) - FP32_SIGNIFICAND_WIDTH
  let maxExponent := terms.foldl
    (fun mx t => if t.magnitude ≠ 0 ∧ t.exponent > mx then t.exponent else mx) zeroExponent
  let total : Int := terms.foldl
    (fun total t =>
      if t.magnitude = 0 then total
      else
        let scaled := if rescale ≥ 0 then t.magnitude <<< rescale.toNat else t.magnitude >>> (-rescale).toNat
        let aligned : Int := ((scaled >>> (maxExponent - t.exponent).toNat : Nat) : Int)
        if t.negative then total - aligned else total + aligned)
    0
  if total = 0 then ZERO
  else
    let negative := decide (total < 0)
    let magnitude := total.natAbs
    let bl := bitLength magnitude
    let exponent : Int := maxExponent + bl - width
    let magnitude := if bl > width then magnitude >>> (bl - width) else magnitude <<< (width - bl)
    let magnitude' :=
      if exponent < FP32_MIN_NORMAL_EXPONENT then magnitude >>> (FP32_MIN_NORMAL_EXPONENT - exponent).toNat
      else magnitude
    let exponent' := if exponent < FP32_MIN_NORMAL_EXPONENT then FP32_MIN_NORMAL_EXPONENT else exponent
    let magnitude'' := if rescale ≥ 0 then magnitude' >>> rescale.toNat else magnitude' <<< (-rescale).toNat
    if magnitude'' = 0 then ZERO else ⟨negative, exponent', magnitude''⟩

/-- `total.Value`. Must equal `Verity.TC.Spec.Value`. -/
inductive Value where
  | nan
  | inf (negative : Bool)
  | fin (t : Term)

/-- The classes of `bf16_class` and `fp32_class`. Must equal `Verity.TC.Spec.Cls`. -/
inductive Cls where
  | nan
  | inf
  | zero
  | finite
deriving DecidableEq

/-- `total.bf16_class`. Must equal `Verity.TC.Spec.bf16Class`. -/
def bf16Class (word : Nat) : Cls :=
  let exponent := (word >>> 7) &&& 0xFF
  let mantissa := word &&& 0x7F
  if exponent = 0xFF then (if mantissa ≠ 0 then .nan else .inf)
  else if exponent = 0 ∧ mantissa = 0 then .zero else .finite

/-- `total.fp32_class`. Must equal `Verity.TC.Spec.fp32Class`. -/
def fp32Class (word : Nat) : Cls :=
  let exponent := (word >>> 23) &&& 0xFF
  let mantissa := word &&& 0x7FFFFF
  if exponent = 0xFF then (if mantissa ≠ 0 then .nan else .inf)
  else if exponent = 0 ∧ mantissa = 0 then .zero else .finite

/-- `total.bf16_product_total`. Must equal `Verity.TC.Spec.bf16ProductTotal`. -/
def bf16ProductTotal (a b : Nat) : Value :=
  let ca := bf16Class a
  let cb := bf16Class b
  if ca = .nan ∨ cb = .nan then .nan
  else if ca = .inf ∨ cb = .inf then
    if ca = .zero ∨ cb = .zero then .nan else .inf (((a ^^^ b) >>> 15) &&& 1 = 1)
  else .fin (bf16Product a b)

/-- `total.fp32_value`. Must equal `Verity.TC.Spec.fp32Value`. -/
def fp32Value (bits : Nat) : Value :=
  match fp32Class bits with
  | .nan => .nan
  | .inf => .inf (bits >>> 31 = 1)
  | _ => .fin (fp32Term bits)

/-- Must equal `Verity.TC.Spec.Value.isNaN`. -/
def Value.isNaN : Value → Bool
  | .nan => true
  | _ => false

/-- Must equal `Verity.TC.Spec.Value.infSign?`. -/
def Value.infSign? : Value → Option Bool
  | .inf negative => some negative
  | _ => none

/-- Must equal `Verity.TC.Spec.Value.term?`. -/
def Value.term? : Value → Option Term
  | .fin t => some t
  | _ => none

/-- `total.group_step_total`. Must equal `Verity.TC.Spec.groupStepTotal`. -/
def groupStepTotal (acc : Value) (products : List Value) (width : Nat) (zeroExponent : Int) : Value :=
  let terms := acc :: products
  if terms.any Value.isNaN then .nan
  else
    let signs := terms.filterMap Value.infSign?
    if signs.contains true ∧ signs.contains false then .nan
    else
      match signs with
      | s :: _ => .inf s
      | [] =>
        let total := groupSum (terms.filterMap Value.term?) width zeroExponent
        if total.magnitude ≠ 0 ∧ total.exponent > FP32_MAX_EXPONENT then .inf total.negative else .fin total

/-- `total.pack_value`. Must equal `Verity.TC.Spec.packValue`. -/
def packValue : Value → Nat
  | .nan => 0x7FFFFFFF
  | .inf negative => ((if negative then 1 else 0) <<< 31) ||| 0x7F800000
  | .fin t => packFp32 t

/-- The loop of `tc_dot_total` over the groups. Must equal `Verity.TC.Spec.runGroups`. -/
def runGroups (width : Nat) (zeroExponent : Int) : List Nat → Value → List Nat → List Nat → Value
  | [], acc, _, _ => acc
  | size :: rest, acc, a, b =>
    runGroups width zeroExponent rest
      (groupStepTotal acc (List.zipWith bf16ProductTotal (a.take size) (b.take size)) width zeroExponent)
      (a.drop size) (b.drop size)

/-- `total.tc_dot_total`: `D = acc + sum_k a[k] * b[k]` on every encoding. Must equal `Verity.TC.Spec.tcDotTotal`. -/
def tcDotTotal (p : Pipeline) (accBits : Nat) (a b : List Nat) : Nat :=
  packValue (runGroups p.width p.zeroExponent p.groups (fp32Value accBits) a b)

end Evaluate.TC.Spec
