import Definitions.Pouw.Fp8Atom.Atom

/-!
# sm_120's FP8 step (trusted definitions)

`BLACKWELL_SM120_E4M3_M16N8K32` in `verity.ml.tc.models` (verity branch `cursor/vllm-sm120-fp8-probe-422d` at
`8eed14b0`): one accumulator coordinate of the RTX PRO 6000 Blackwell's `mma.sync` m16n8k32 E4M3 (SASS `QMMA.16832`),
fitted on 6,434,816 finite elements in 13 families with 0 mismatches. It is `Fp8Atom`'s grouped-sum `Pipeline` with one
group of all 32 products plus the accumulator, a 26-bit truncating adder, and an exponent floor: `⟨[32], 26, floor⟩`.
Unlike Hopper's and Ada's 14-bit FP8 adders, the 26-bit adder does not truncate a 24-bit accumulator.

The floor is unobservable at every floor up to −124 (the capture's fit covered `[−150, −124]`). Every nonzero term has
exponent at least −126: an E4M3 product at least −12, and an FP32 accumulator at least −126. So the floor is the
alignment exponent only when every term is zero, or when the accumulator is the only live term at exponent −126 or −125.
There the adder's two bits above the FP32 grid absorb an alignment of up to two places. At −123 the floor is observable:
it drops a bit of a subnormal or smallest-normal accumulator. So `sm120E4m3K32` takes the floor as a parameter, and
`sm120E4m3` pins the BF16 pipeline's −133, as `verity.ml.tc` does. The conformance vectors are `Sm120Vectors`, checked
in `Sm120Checks`: all at −133, the floor-sensitive ones also at −150 and −124, and the floor's observability at −123.
-/

namespace Pouw.PearlC

open Pouw.Fp8Atom

/-- `BLACKWELL_SM120_E4M3_M16N8K32` with exponent floor `floor`: one group of 32, a 26-bit adder. -/
def sm120E4m3K32 (floor : Int) : Pipeline := ⟨[32], 26, floor⟩

/-- `BLACKWELL_SM120_E4M3_M16N8K32` as `verity.ml.tc` pins it, with the floor at −133. -/
def sm120E4m3 : Pipeline := sm120E4m3K32 (-133)

/-- One sm_120 E4M3 step with exponent floor `floor`: the accumulator word and 32 code pairs to the FP32 word written. -/
def sm120StepAt (floor : Int) (acc : Nat) (a b : Fin 32 → Code) : Nat :=
  (sm120E4m3K32 floor).step acc (List.ofFn a) (List.ofFn b)

/-- **One `BLACKWELL_SM120_E4M3_M16N8K32` step**, the floor at −133. -/
def sm120Step (acc : Nat) (a b : Fin 32 → Code) : Nat := sm120E4m3.step acc (List.ofFn a) (List.ofFn b)

end Pouw.PearlC
