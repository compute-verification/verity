module

public import Definitions.Pouw.Fp8Atom.Atom
import Mathlib.Tactic.Bound.Init
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! sm_120's FP8 step `BLACKWELL_SM120_E4M3_M16N8K32`, a `Pipeline` of one group of 32 products on a 26-bit truncating adder, whose exponent floor is observable at −123, so `sm120E4m3K32` takes it as a parameter. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom

/-- `BLACKWELL_SM120_E4M3_M16N8K32` with exponent floor `floor`: one group of 32, a 26-bit adder. -/
def sm120E4m3K32 (floor : Int) : Pipeline := ⟨[32], 26, floor⟩

/-- `BLACKWELL_SM120_E4M3_M16N8K32` as `verity.ml.tc` pins it, with the floor at −133. -/
def sm120E4m3 : Pipeline := sm120E4m3K32 (-133)

def sm120StepAt (floor : Int) (acc : Nat) (a b : Fin 32 → Code) : Nat :=
  (sm120E4m3K32 floor).step acc (List.ofFn a) (List.ofFn b)

/-- **One `BLACKWELL_SM120_E4M3_M16N8K32` step**, the floor at −133. -/
def sm120Step (acc : Nat) (a b : Fin 32 → Code) : Nat := sm120E4m3.step acc (List.ofFn a) (List.ofFn b)

end Pouw.PearlC
