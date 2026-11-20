module

public import Definitions.ResourceAccounting.BlockProperties

/-!
# The inference-only check on a monitoring period's runs

The limits and the check the inference-only policy's theorem reads (`InferenceOnly.PolicyTheorem` in
`Definitions/Compose/Guarantees/Blocks.lean`). The check runs at the period's end on PoUW's credited use, the attested
counts (the wipe's egress and the network attester's ingress count), the registered information (the hardware
description's compute rates), the sampling design's tolerance and the assumed covert figure, beside each run's committed
blocks and the claims on its inference blocks; it is run inside the outer proof, so the auditor sees none of its
inputs.
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

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` (gates) and
input `k` (bits); the free blocks' count `m`; `freeSize` (gates), the limit on each server-day block's capacity less
PoUW's credited use plus the tolerance, which bounds its free blocks' work and the actual work of the inference blocks
in the tolerance set together (`FreeWorkAtMost`); `cS` (gates), the limit on each server-day block's capacity, which
bounds its size; `kS` (bits), the limit on each server-day block's bound on input from its own past plus its bound on
input from the outside world; `hidden` (gates), the bound on the work at leaves no listed location holds
(`Runs.hiddenGates`), assumed: no protocol stands behind it; and in the theorem's bound `share`, the share of that free
quantity (the free blocks' and the tolerance set's together), and the allowance inside inference blocks `allowance`. The
policy, as the architecture page states it, limits a server-day block's size and unregistered input only: it has no
term for a server-day block's output, so no limit here is on output, and a server-day block's input from another block
at its location during its window is under no limit. -/
structure Limits where
  n : ℕ
  c : ℕ
  k : ℕ
  m : ℕ
  freeSize : ℕ
  cS : ℕ
  kS : ℕ
  hidden : ℕ
  share : ℝ
  allowance : ℝ

/-- **The inference-only check on a run**, eight comparisons:
1. at most `n` inference blocks;
2. each inference block has a size claim, its committed size claim (`claimedSize`, the declared size the declared row
   sums over the tolerance set) at most `c` gates, and a claim of input at most `k` bits;
3. at most `m` free blocks;
4. each server-day block's PoUW credited use at most its capacity (gates; the subtraction in 5 is of naturals, so a
   credited use above the capacity would read there as `0`);
5. each server-day block's capacity less PoUW's credited use plus the tolerance at most `freeSize` gates (the free
   blocks' and the tolerance set's work together are at most that number, `FreeWorkAtMost`);
6. each server-day block's capacity at most `cS` gates;
7. each server-day block's bound on input from its own past (`wipePastBound` of the wipe's egress) plus its bound on
   input from the outside world (the network attester's ingress count plus the covert figure, assumed) at most `kS`
   bits;
8. a unit of `inner` that is not an inference block (a free block, or a unit under hidden hardware) carries no claim.

The classes are `cls`'s, so a unit labelled inference whose window starts before its epoch's challenge tick counts among
the free blocks. No server-day block carries a claim: 6 and 7 compare numbers the run carries for it. The limit in 7 is
on the sum because the policy reads one number for a server-day block's unregistered input and the block's input from
both sources is at most the sum (`serverInput_le`). A server-day block's input from another block at its location
during its window (`.block`) is under no limit here. The policy has no term for a server-day block's output, so the check
compares none. It runs at the period's end on the audit's outputs (`credited`), the registered information (`rate`),
the sampling design's tolerance (`tol`), the epochs' challenge ticks (`salt`), the attested counts (`wipeEgress`,
`ingressCount`) and the assumed covert figure (`covert`). Coverage, disjointness and the nesting are the partitions' own,
so the check states none of them. -/
def check {L : Type*} (lim : Limits) (P : Runs L) (r : P.Run) : Bool :=
  decide ((univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n) &&
    decide (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any isSize = true ∧ claimedSize (P.claimsIn r u) ≤ lim.c ∧
        (P.claimsIn r u).any (freshAtMost lim.k) = true) &&
    decide ((univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m) &&
    decide (∀ i, P.kind r i = true → P.credited r i ≤ P.capacity r i) &&
    decide (∀ i, P.kind r i = true → P.capacity r i - P.credited r i + P.tol ≤ lim.freeSize) &&
    decide (∀ i, P.kind r i = true → P.capacity r i ≤ lim.cS) &&
    decide (∀ i, P.kind r i = true →
      wipePastBound (P.wipeEgress r i) + (P.ingressCount r i + P.covert r i) ≤ lim.kS) &&
    decide (∀ u, P.cls r u ≠ Class.inference → (P.claimsIn r u).isEmpty = true)

end InferenceOnly
