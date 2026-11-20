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

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` (gates) and
input `k` (bits); the free blocks' count `m`; `freeSize` (gates), the limit on each server-day block's committed
free-capacity claim, which pays for its free blocks' work and for the actual work of the inference blocks in the
tolerance set together; the assumed hidden hardware size `hidden` (gates); and in the theorem's bound `share`, the share
of that free quantity (the free blocks' and the tolerance set's together), and the allowance inside inference blocks
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

/-- **The inference-only check on a run's committed blocks**: at most `n` inference blocks, each with a size claim, its
committed size claim (`claimedSize`, the declared size the declared row sums over the tolerance set) at most `c` gates,
and claimed of input at most `k` bits; at most `m` free blocks; every server-day block's **committed free-capacity claim**
at most `freeSize` gates (`P.claimedFree r i ≤ lim.freeSize`, the limit on the committed number the check reads — the free
blocks' and the tolerance's work together are at most that claim, `FreeClaimHolds`); every server-day block claimed of some size (the second cap on the
same gates, read in the theorem as `min`); and every server-day block claimed of some input, the bound the wipe's and
network attestation's rows establish, so the policy side reads a server-day block's input. Coverage, disjointness and the
nesting are the partitions' own, so the check states none of them. -/
def check {L : Type*} (lim : Limits) (P : Runs L) (r : P.Run) : Bool :=
  decide ((univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n) &&
    decide (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any isSize = true ∧ claimedSize (P.claimsIn r u) ≤ lim.c ∧
        (P.claimsIn r u).any (freshAtMost lim.k) = true) &&
    decide ((univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m) &&
    decide (∀ i, P.kind r i = true → P.claimedFree r i ≤ lim.freeSize) &&
    decide (∀ i, P.kind r i = true → (P.claimsOut r i).any isSize = true) &&
    decide (∀ i, P.kind r i = true → (P.claimsOut r i).any isInputClaim = true)

end InferenceOnly
