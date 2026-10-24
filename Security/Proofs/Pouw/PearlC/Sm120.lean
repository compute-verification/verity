import Definitions.Pouw.PearlC.Sm120

/-! The proof-only part of `Definitions.Pouw.PearlC.Sm120`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

open Pouw.Fp8Atom

/-- One sm_120 E4M3 step with exponent floor `floor`: the accumulator word and 32 code pairs to the FP32 word written. -/
def sm120StepAt (floor : Int) (acc : Nat) (a b : Fin 32 → Code) : Nat :=
  (sm120E4m3K32 floor).step acc (List.ofFn a) (List.ofFn b)

end Pouw.PearlC
