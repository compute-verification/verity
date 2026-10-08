import Definitions.Core.TC.Pipeline

/-!
# The step semantics: `verity.ml.tc.total.tc_dot_total` on BF16 pipelines

A transcription of `term.py` (`Term`, `bf16_product`, `fp32_term`, `pack_fp32`), `models.py` (`group_sum`) and
`total.py` (`bf16_class`, `fp32_class`, `bf16_product_total`, `fp32_value`, `group_step_total`, `pack_value`,
`tc_dot_total`), function for function and branch for branch, on the Python functions' domain: FP32 words below `2^32`,
BF16 words below `2^16`, `k` operand pairs. The Python raises outside it; the theorems take the domain as hypotheses.
Words are natural numbers and the bit operations are Python's (`>>>` is `>>`, `&&&` is `&`, `|||` is `|`, `^^^` is `^`).
-/

namespace Verity.TC.Spec

def FP32_SIGNIFICAND_WIDTH : Nat := 24
def FP32_MIN_NORMAL_EXPONENT : Int := -126
def FP32_MAX_EXPONENT : Int := 127

/-- `term.Term`: `(-1)^negative * magnitude * 2^(exponent - 23)`; `magnitude = 0` is the zero. -/
structure Term where
  negative : Bool
  exponent : Int
  magnitude : Nat

/-- `term.ZERO`. -/
def ZERO : Term := ⟨false, 0, 0⟩

/-- `term.bf16_product` on finite operands: the exact product, carried at 25 bits. -/
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

/-- `term.fp32_term` on finite words. -/
def fp32Term (bits : Nat) : Term :=
  let exponentField := (bits >>> 23) &&& 0xFF
  let mantissa := bits &&& 0x7FFFFF
  let negative := bits >>> 31 = 1
  if exponentField = 0 then
    if mantissa = 0 then ZERO else ⟨negative, FP32_MIN_NORMAL_EXPONENT, mantissa⟩
  else ⟨negative, (exponentField : Int) - 127, (1 <<< 23) ||| mantissa⟩

/-- `term.pack_fp32`: a 24-bit term as FP32 bits (zero is `+0`, overflow is `±inf`). -/
def packFp32 (t : Term) : Nat :=
  if t.magnitude = 0 then 0
  else if t.exponent > FP32_MAX_EXPONENT then ((if t.negative then 1 else 0) <<< 31) ||| 0x7F800000
  else
    let exponentField : Int := t.exponent + 127
    let exponentField := if t.magnitude &&& (1 <<< 23) = 0 then exponentField - 1 else exponentField
    ((if t.negative then 1 else 0) <<< 31) ||| (exponentField.toNat <<< 23) ||| (t.magnitude &&& 0x7FFFFF)

/-- `models.group_sum`: one grouped, truncating accumulation. -/
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

/-- `total.Value`: a NaN, a signed infinity, or a finite term. -/
inductive Value where
  | nan
  | inf (negative : Bool)
  | fin (t : Term)

/-- The classes `bf16_class` and `fp32_class` return. -/
inductive Cls where
  | nan
  | inf
  | zero
  | finite
deriving DecidableEq

/-- `total.bf16_class`. -/
def bf16Class (word : Nat) : Cls :=
  let exponent := (word >>> 7) &&& 0xFF
  let mantissa := word &&& 0x7F
  if exponent = 0xFF then (if mantissa ≠ 0 then .nan else .inf)
  else if exponent = 0 ∧ mantissa = 0 then .zero else .finite

/-- `total.fp32_class`. -/
def fp32Class (word : Nat) : Cls :=
  let exponent := (word >>> 23) &&& 0xFF
  let mantissa := word &&& 0x7FFFFF
  if exponent = 0xFF then (if mantissa ≠ 0 then .nan else .inf)
  else if exponent = 0 ∧ mantissa = 0 then .zero else .finite

/-- `total.bf16_product_total`. -/
def bf16ProductTotal (a b : Nat) : Value :=
  let ca := bf16Class a
  let cb := bf16Class b
  if ca = .nan ∨ cb = .nan then .nan
  else if ca = .inf ∨ cb = .inf then
    if ca = .zero ∨ cb = .zero then .nan else .inf (((a ^^^ b) >>> 15) &&& 1 = 1)
  else .fin (bf16Product a b)

/-- `total.fp32_value`. -/
def fp32Value (bits : Nat) : Value :=
  match fp32Class bits with
  | .nan => .nan
  | .inf => .inf (bits >>> 31 = 1)
  | _ => .fin (fp32Term bits)

def Value.isNaN : Value → Bool
  | .nan => true
  | _ => false

def Value.infSign? : Value → Option Bool
  | .inf negative => some negative
  | _ => none

def Value.term? : Value → Option Term
  | .fin t => some t
  | _ => none

/-- `total.group_step_total`: the incoming value and a group's products to the outgoing value. The Python's set of
infinity signs has two elements exactly when the list below contains both. -/
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

/-- `total.pack_value`. -/
def packValue : Value → Nat
  | .nan => 0x7FFFFFFF
  | .inf negative => ((if negative then 1 else 0) <<< 31) ||| 0x7F800000
  | .fin t => packFp32 t

/-- The loop of `tc_dot_total` over the groups: each group takes the next `size` operand pairs. -/
def runGroups (width : Nat) (zeroExponent : Int) : List Nat → Value → List Nat → List Nat → Value
  | [], acc, _, _ => acc
  | size :: rest, acc, a, b =>
    runGroups width zeroExponent rest
      (groupStepTotal acc (List.zipWith bf16ProductTotal (a.take size) (b.take size)) width zeroExponent)
      (a.drop size) (b.drop size)

/-- `total.tc_dot_total`: `D = acc + sum_k a[k] * b[k]` on every encoding, as the pipeline's silicon computes it. -/
def tcDotTotal (p : Pipeline) (accBits : Nat) (a b : List Nat) : Nat :=
  packValue (runGroups p.width p.zeroExponent p.groups (fp32Value accBits) a b)

/-- `total.cvt_rn_bf16_f32`: `cvt.rn.bf16.f32`, round to nearest even on finite words (overflow to `±inf`), `±inf` kept,
every NaN to the canonical `0x7FFF`. -/
def cvtRnBf16F32 (bits : Nat) : Nat :=
  if fp32Class bits = .nan then 0x7FFF else ((bits + 0x7FFF + ((bits >>> 16) &&& 1)) >>> 16) &&& 0xFFFF

/-- `tc_dot_total` folded over a unit's k16 steps in k order: one GEMM output coordinate's accumulator, as
`models.tc_dot_chain` folds a model's step. -/
def tcDotChainTotal (p : Pipeline) (accBits : Nat) (steps : List (List Nat × List Nat)) : Nat :=
  steps.foldl (fun acc s => tcDotTotal p acc s.1 s.2) accBits

end Verity.TC.Spec
