module

public import Definitions.ResourceAccounting.BlockProperties

/-!
# The inference-only check on a monitoring period's runs

The limits and the check the inference-only policy's theorem reads (`InferenceOnly.PolicyTheorem` in
`Definitions/Compose/Guarantees/Blocks.lean`). The check reads each run's committed blocks and the claims on them; it is
run inside the outer proof, so the auditor sees none of its inputs.
-/

@[expose] public section

namespace InferenceOnly

open ResourceAccounting Verity Finset

/-- The claim is of size at most `c` gates. -/
def sizeAtMost (c : ℕ) : Property → Bool
  | .sizeAtMost s => decide (s ≤ c)
  | _ => false

/-- The claim is of input at most `k` bits from the block's own past, another block and the outside world. -/
def freshAtMost (k : ℕ) : Property → Bool
  | .inputAtMost [.past, .block, .world] s => decide (s ≤ k)
  | _ => false

/-- The claim is a size claim (any bound). -/
def isSize : Property → Bool
  | .sizeAtMost _ => true
  | _ => false

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` (gates) and
input `k` (bits); the free blocks' count `m` and size `freeSize` (gates); the assumed hidden hardware size `hidden`
(gates); and in the theorem's bound the free blocks' share `share` and the allowance inside inference blocks
`allowance`. -/
structure Limits where
  n : ℕ
  c : ℕ
  k : ℕ
  m : ℕ
  freeSize : ℕ
  hidden : ℕ
  share : ℝ
  allowance : ℝ

/-- **The inference-only check on a run's committed blocks**: at most `n` inference blocks, each claimed of size at most
`c` gates and input at most `k` bits; at most `m` free blocks, each claimed of size at most `freeSize` gates; and every
server-day block claimed of some size (the second cap on the same gates, read in the theorem as `min`). Coverage,
disjointness and the nesting are the partitions' own, so the check states none of them. -/
def check {L : Type*} (lim : Limits) (P : Runs L) (r : P.Run) : Bool :=
  decide ((univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n) &&
    decide (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any (sizeAtMost lim.c) = true ∧ (P.claimsIn r u).any (freshAtMost lim.k) = true) &&
    decide ((univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m) &&
    decide (∀ u, P.cls r u = Class.free → (P.claimsIn r u).any (sizeAtMost lim.freeSize) = true) &&
    decide (∀ i, P.kind r i = true → (P.claimsOut r i).any isSize = true)

end InferenceOnly
