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

/-- The claim is an input claim (any sources, any bound). -/
def isInputClaim : Property → Bool
  | .inputAtMost _ _ => true
  | _ => false

/-- **The claimed size** of a block: its first size claim's bound (`0` if none). The check reads this per free block and
sums it over a server-day block's free children, the quantity the conversion `FreeBlockAtMost` bounds. -/
def claimedSize : List Property → ℕ
  | [] => 0
  | (.sizeAtMost k) :: _ => k
  | _ :: t => claimedSize t

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
`c` gates and input at most `k` bits; at most `m` free blocks, each claimed of some size; the free blocks' claimed sizes
summed over each server-day block's free children at most `freeSize` gates (the sum the conversion `FreeBlockAtMost`
bounds, read here, not a size per free block); every server-day block claimed of some size (the second cap on the
same gates, read in the theorem as `min`); and every server-day block claimed of some input, the bound the wipe's and
network attestation's rows establish, so the policy side reads a server-day block's input. Coverage, disjointness and the
nesting are the partitions' own, so the check states none of them. -/
def check {L : Type*} (lim : Limits) (P : Runs L) (r : P.Run) : Bool :=
  decide ((univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n) &&
    decide (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any (sizeAtMost lim.c) = true ∧ (P.claimsIn r u).any (freshAtMost lim.k) = true) &&
    decide ((univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m) &&
    decide (∀ u, P.cls r u = Class.free → (P.claimsIn r u).any isSize = true) &&
    decide (∀ i, P.kind r i = true → ∑ u ∈ P.freeChildren r i, claimedSize (P.claimsIn r u) ≤ lim.freeSize) &&
    decide (∀ i, P.kind r i = true → (P.claimsOut r i).any isSize = true) &&
    decide (∀ i, P.kind r i = true → (P.claimsOut r i).any isInputClaim = true)

end InferenceOnly
