import Definitions.Core.TC
import Models.Core.Assumptions

/-!
# The tensor-core guarantees

The step relation (`Verity.Protocol.TC.Relation`, `verity.ml.tc.relation`) against the step semantics
(`Verity.Protocol.TC.Spec`, `verity.ml.tc.total.tc_dot_total`) on the Hopper pipeline (`HOPPER_BF16_WGMMA_K16`, whose
parameters `HOPPER_BF16_M16N8K16` shares) and the Ampere/Ada pipeline (`AMPERE_BF16_M16N8K16`, whose parameters
`ADA_BF16_M16N8K16` shares), in either alignment mode, over the integers: each step, each verification unit and the BF16
word it commits, and the range and is-zero gadgets. Under `gemm-hopper-step` and `gemm-ampere-step`
(`Verity.Assumptions`), the semantics is the device's.
-/

namespace Verity.Guarantees

open Verity.TC Verity.TC.Spec Verity.TC.Relation Verity.TC.Proofs Verity.TC.Gadgets Verity.Assumptions

/-- `Census.ISZERO`'s chunks and flags hold exactly when `x ∈ [0, 2^bits)` and `z = [x = 0]`: `ISZERO`. -/
def IsZeroGadget : Prop :=
  ∀ (x z : Int) (bits : Nat), ZeroChunks (isZeroWidths bits) x z ↔ ISZERO x bits z

/-- `Census.RANGE`'s chunks hold exactly when `x ∈ [0, 2^bits)`: `RANGE`. -/
def RangeGadget : Prop :=
  ∀ (x : Int) (bits : Nat), Chunks (rangeWidths bits) x ↔ RANGE x bits

/-- The Ada pipeline (`ADA_BF16_M16N8K16`, the RTX 4090) has the Ampere pipeline's parameters, so every guarantee about
`AMPERE_BF16_M16N8K16` holds for it. -/
def AdaEqAmpere : Prop :=
  ADA_BF16_M16N8K16 = AMPERE_BF16_M16N8K16

/-- On the Ampere pipeline, the relation accepts some witness for every input in `tc_dot_total`'s domain. -/
def AmpereStepComplete : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs AMPERE_BF16_M16N8K16 acc a b →
    ∃ w, StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w

/-- Under `gemm-ampere-step`, the output state of every witness the relation accepts for an Ampere step is the decode of
the word the device writes. -/
def AmpereStepHw : Prop :=
  ∀ (step : Nat → List Nat → List Nat → Nat), GemmAmpereStep step →
    ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs AMPERE_BF16_M16N8K16 acc a b → ∀ (w : Witness),
    StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w →
    stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups = decodeState (step acc a b) AMPERE_BF16_M16N8K16

/-- On the Ampere pipeline, the states the relation can output for the input state `decode_state(acc)` and the operand
words `a`, `b` are exactly the one the semantics gives, `decode_state(tc_dot_total(acc, a, b))`. -/
def AmpereStepIff : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs AMPERE_BF16_M16N8K16 acc a b → ∀ (out : State),
    (∃ w, StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w ∧
        stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups = out) ↔
      out = decodeState (tcDotTotal AMPERE_BF16_M16N8K16 acc a b) AMPERE_BF16_M16N8K16

/-- On the Ampere pipeline, every witness the relation accepts for the input state `decode_state(acc)` and the operand
words `a`, `b` outputs `decode_state(tc_dot_total(acc, a, b))`. -/
def AmpereStepSound : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs AMPERE_BF16_M16N8K16 acc a b → ∀ (w : Witness),
    StepHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) a b w →
    stepOut (decodeState acc AMPERE_BF16_M16N8K16) w.groups =
      decodeState (tcDotTotal AMPERE_BF16_M16N8K16 acc a b) AMPERE_BF16_M16N8K16

/-- A verification unit on the Ampere/Ada pipeline, from `zero_state` to the committed BF16 word: the relation's steps
and boundary accept only `cvt_rn_bf16_f32` of the semantics' accumulator, whose FP32 word is the accumulator itself. -/
def AmpereUnitBf16 : Prop :=
  ∀ (align : Align) (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds AMPERE_BF16_M16N8K16 align (zeroState AMPERE_BF16_M16N8K16) steps out →
    ∀ (c : PackCols), packHolds out c →
    c.word = (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1))) : Int) ∧
      c.bf16 = (cvtRnBf16F32 (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1)))) : Int)

/-- A verification unit on the Ampere pipeline: every witness chain the relation accepts from `decode_state(acc)` outputs
the decode of the semantics' accumulator after the unit's steps. -/
def AmpereUnitSound : Prop :=
  ∀ (align : Align) (acc : Nat), acc < 2 ^ 32 →
    ∀ (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds AMPERE_BF16_M16N8K16 align (decodeState acc AMPERE_BF16_M16N8K16) steps out →
    out = decodeState (tcDotChainTotal AMPERE_BF16_M16N8K16 acc (steps.map (fun s => (s.1, s.2.1))))
      AMPERE_BF16_M16N8K16

/-- `AmpereUnitSound` from the unit's constant initial state `zero_state`, which is `decode_state(+0)`. -/
def AmpereUnitSoundZero : Prop :=
  ∀ (align : Align) (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds AMPERE_BF16_M16N8K16 align (zeroState AMPERE_BF16_M16N8K16) steps out →
    out = decodeState (tcDotChainTotal AMPERE_BF16_M16N8K16 0 (steps.map (fun s => (s.1, s.2.1))))
      AMPERE_BF16_M16N8K16

/-- On the Hopper pipeline, the relation accepts some witness for every input in `tc_dot_total`'s domain. -/
def HopperStepComplete : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs HOPPER_BF16_WGMMA_K16 acc a b →
    ∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w

/-- Under `gemm-hopper-step`, the output state of every witness the relation accepts for a Hopper step is the decode of
the word the device writes. -/
def HopperStepHw : Prop :=
  ∀ (step : Nat → List Nat → List Nat → Nat), GemmHopperStep step →
    ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs HOPPER_BF16_WGMMA_K16 acc a b → ∀ (w : Witness),
    StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w →
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = decodeState (step acc a b) HOPPER_BF16_WGMMA_K16

/-- On the Hopper pipeline, the states the relation can output for the input state `decode_state(acc)` and the operand
words `a`, `b` are exactly the one the semantics gives, `decode_state(tc_dot_total(acc, a, b))`. -/
def HopperStepIff : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs HOPPER_BF16_WGMMA_K16 acc a b → ∀ (out : State),
    (∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w ∧
        stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = out) ↔
      out = decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16

/-- On the Hopper pipeline, every witness the relation accepts for the input state `decode_state(acc)` and the operand
words `a`, `b` outputs `decode_state(tc_dot_total(acc, a, b))`. -/
def HopperStepSound : Prop :=
  ∀ (align : Align) (acc : Nat) (a b : List Nat), StepInputs HOPPER_BF16_WGMMA_K16 acc a b → ∀ (w : Witness),
    StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w →
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups =
      decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16

/-- A verification unit on the Hopper pipeline, from `zero_state` to the committed BF16 word: the relation's steps and
boundary accept only `cvt_rn_bf16_f32` of the semantics' accumulator, whose FP32 word is the accumulator itself. -/
def HopperUnitBf16 : Prop :=
  ∀ (align : Align) (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds HOPPER_BF16_WGMMA_K16 align (zeroState HOPPER_BF16_WGMMA_K16) steps out →
    ∀ (c : PackCols), packHolds out c →
    c.word = (tcDotChainTotal HOPPER_BF16_WGMMA_K16 0 (steps.map (fun s => (s.1, s.2.1))) : Int) ∧
      c.bf16 = (cvtRnBf16F32 (tcDotChainTotal HOPPER_BF16_WGMMA_K16 0 (steps.map (fun s => (s.1, s.2.1)))) : Int)

/-- A verification unit on the Hopper pipeline: every witness chain the relation accepts from `decode_state(acc)` outputs
the decode of the semantics' accumulator after the unit's steps. -/
def HopperUnitSound : Prop :=
  ∀ (align : Align) (acc : Nat), acc < 2 ^ 32 →
    ∀ (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) steps out →
    out = decodeState (tcDotChainTotal HOPPER_BF16_WGMMA_K16 acc (steps.map (fun s => (s.1, s.2.1))))
      HOPPER_BF16_WGMMA_K16

/-- `HopperUnitSound` from the unit's constant initial state `zero_state`, which is `decode_state(+0)`. -/
def HopperUnitSoundZero : Prop :=
  ∀ (align : Align) (steps : List (List Nat × List Nat × Witness)),
    (∀ s ∈ steps, s.1.length = 16 ∧ s.2.1.length = 16 ∧ (∀ x ∈ s.1, x < 2 ^ 16) ∧ (∀ x ∈ s.2.1, x < 2 ^ 16)) →
    ∀ (out : State), ChainHolds HOPPER_BF16_WGMMA_K16 align (zeroState HOPPER_BF16_WGMMA_K16) steps out →
    out = decodeState (tcDotChainTotal HOPPER_BF16_WGMMA_K16 0 (steps.map (fun s => (s.1, s.2.1))))
      HOPPER_BF16_WGMMA_K16

/-- `pack_relation` is `pack_value` then `cvt_rn_bf16_f32` on the state of a value a group outputs (`stateOf`,
`Admissible`), and it accepts some columns. -/
def PackSoundComplete : Prop :=
  ∀ (Z : Int) (V : Value), Admissible V →
    (∀ c, packHolds (stateOf Z V) c → c.word = (packValue V : Int) ∧ c.bf16 = (cvtRnBf16F32 (packValue V) : Int)) ∧
      ∃ c, packHolds (stateOf Z V) c

end Verity.Guarantees
