import Verity.TC.Spec
import Verity.TC.Relation
import Verity.Assumptions

/-!
# The step relation is the step semantics

For the Hopper pipeline (`HOPPER_BF16_WGMMA_K16`, whose parameters `HOPPER_BF16_M16N8K16` shares), in either alignment
mode, over the integers:

* `hopper_step_sound`: every witness the relation accepts for the input state `decode_state(acc)` and the operand words
  `a`, `b` outputs `decode_state(tc_dot_total(acc, a, b))`;
* `hopper_step_complete`: the relation accepts some witness for every input in `tc_dot_total`'s domain;
* `hopper_step_iff`: together, the states the relation can output are exactly the one the semantics gives;
* `hopper_step_hw`: under `gemm-hopper-step`, the output state is the decode of the word the device writes.

A verification unit starts from `zero_state = decode_state(+0)` and feeds each step's output state to the next, so the
step theorem applies to every step of a unit in turn.
-/

open Verity.TC.Relation Verity.TC.Spec Verity.Assumptions

namespace Verity.TC

theorem hopper_step_sound (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (w : Witness)
    (h : StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w) :
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups =
      decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16 := by
  sorry

theorem hopper_step_complete (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) :
    ∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w := by
  sorry

theorem hopper_step_iff (align : Align) (acc : Nat) (a b : List Nat)
    (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (out : State) :
    (∃ w, StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w ∧
        stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = out) ↔
      out = decodeState (tcDotTotal HOPPER_BF16_WGMMA_K16 acc a b) HOPPER_BF16_WGMMA_K16 := by
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact hopper_step_sound align acc a b hin w hw
  · rintro rfl
    obtain ⟨w, hw⟩ := hopper_step_complete align acc a b hin
    exact ⟨w, hw, hopper_step_sound align acc a b hin w hw⟩

theorem hopper_step_hw (step : Nat → List Nat → List Nat → Nat) (hstep : GemmHopperStep step)
    (align : Align) (acc : Nat) (a b : List Nat) (hin : StepInputs HOPPER_BF16_WGMMA_K16 acc a b) (w : Witness)
    (h : StepHolds HOPPER_BF16_WGMMA_K16 align (decodeState acc HOPPER_BF16_WGMMA_K16) a b w) :
    stepOut (decodeState acc HOPPER_BF16_WGMMA_K16) w.groups = decodeState (step acc a b) HOPPER_BF16_WGMMA_K16 := by
  rw [hstep acc a b hin]
  exact hopper_step_sound align acc a b hin w h

end Verity.TC
