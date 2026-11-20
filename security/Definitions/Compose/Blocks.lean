module

public import Definitions.ResourceAccounting.BlockProperties

/-!
# The inference-only check on a division

The limits and the check the inference-only policy's theorem reads (`InferenceOnly.PolicyTheorem` in
`Definitions/Compose/Guarantees/Blocks.lean`).
-/

@[expose] public section

namespace InferenceOnly

open ResourceAccounting

/-- The claim is of size at most `c`. -/
def sizeAtMost (c : ℕ) : Property → Bool
  | .sizeAtMost s => decide (s ≤ c)
  | _ => false

/-- The claim is of input at most `k` bits from the block's own past, another block and the outside world. -/
def freshAtMost (k : ℕ) : Property → Bool
  | .inputAtMost [.past, .block, .world] s => decide (s ≤ k)
  | _ => false

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` and input
`k`; the free blocks' count `m` and size `freeSize`; and in the theorem's bound, the free blocks' share `share` and
the allowance inside inference blocks `allowance`. -/
structure Limits where
  n : ℕ
  c : ℕ
  k : ℕ
  m : ℕ
  freeSize : ℕ
  share : ℝ
  allowance : ℝ

/-- **The inference-only check on a division**: every block with no block inside it is an inference or a free block,
no block lies inside an inference block, and the two classes are disjoint; at most `n` inference blocks, each claimed
of size at most `c` and input at most `k` bits; at most `m` free blocks, each claimed of size at most `freeSize`. -/
def check {L : Type*} (lim : Limits) (Dv : Division L) : Bool :=
  decide (∀ i, (∀ j, Dv.parent j ≠ some i) → i ∈ Dv.inference ∨ i ∈ Dv.free) &&
    decide (∀ i ∈ Dv.inference, ∀ j, Dv.parent j ≠ some i) &&
    decide (Disjoint Dv.inference Dv.free) &&
    decide (Dv.inference.card ≤ lim.n) &&
    decide (∀ i ∈ Dv.inference, (Dv.claims i).any (sizeAtMost lim.c) = true ∧
      (Dv.claims i).any (freshAtMost lim.k) = true) &&
    decide (Dv.free.card ≤ lim.m) &&
    decide (∀ i ∈ Dv.free, (Dv.claims i).any (sizeAtMost lim.freeSize) = true)

end InferenceOnly
