module

public import Definitions.ResourceAccounting.BlockProperties

/-!
# The inference-only check on a monitoring period's runs

The limits and the check the inference-only policy's theorem reads (`InferenceOnly.PolicyTheorem` in
`Definitions/Compose/Guarantees/Blocks.lean`). The check runs at the period's end on the audit's outputs (PoUW's credited
use) and the registered data (the hardware description's compute rates, the tolerance), beside each run's committed
blocks and the claims on them; it is run inside the outer proof, so the auditor sees none of its inputs.
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

/-- The claim is an input claim from exactly the sources `src` (any bound). -/
def isInputFrom (src : List Source) : Property → Bool
  | .inputAtMost s _ => decide (s = src)
  | _ => false

/-- **The inference-only policy's limits**, public parameters: the inference blocks' count `n`, size `c` (gates) and
input `k` (bits); the free blocks' count `m`; `freeSize` (gates), the limit on each server-day block's capacity less
PoUW's credited use plus the tolerance, which bounds its free blocks' work and the actual work of the inference blocks
in the tolerance set together (`FreeWorkAtMost`); each server-day block's size `cS` (gates) and its input from its own
past and the outside world together `kS` (bits); `hidden` (gates), the bound on the work at leaves no listed location
holds (`Runs.hiddenGates`), assumed: no protocol stands behind it; and in the theorem's bound `share`, the share of that
free quantity (the free blocks' and the tolerance set's together), and the allowance inside inference blocks
`allowance`. The policy, as the architecture page states it, limits a server-day block's size and unregistered input
only: it has no term for a server-day block's output, so no limit here is on output, and a server-day block's input from
another block at its location during its window is under no limit. -/
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

/-- **The inference-only check on a run's committed blocks**: at most `n` inference blocks, each with a size claim, its
committed size claim (`claimedSize`, the declared size the declared row sums over the tolerance set) at most `c` gates,
and claimed of input at most `k` bits; at most `m` free blocks; every server-day block's **capacity less PoUW's credited
use plus the tolerance** at most `freeSize` gates (`P.capacity r i − P.credited r i + P.tol ≤ lim.freeSize`; the free
blocks' and the tolerance set's work together are at most that number, `FreeWorkAtMost`); every server-day block with a
size claim, its committed size claim at most `cS` gates; and every server-day block with an input claim from its own
past (`[.past]`, the wipe's row) and one from the outside world (`[.world]`, network attestation's row), their committed
bounds summed at most `kS` bits. The limit is on the sum because the policy reads one number for a server-day block's
unregistered input and the block's input from both sources is at most the sum (`serverInput_le`); a limit on each would
let `2 · kS` bits in. A server-day block's input from another block at its location during its window (`.block`) is in
no row and under no limit here. The policy has no term for a server-day block's output, so the check compares none. It
runs at the period's end on the audit's outputs (`credited`) and the registered data (`rate`, `tol`). Coverage,
disjointness and the nesting are the partitions' own, so the check states none of them. -/
def check {L : Type*} (lim : Limits) (P : Runs L) (r : P.Run) : Bool :=
  decide ((univ.filter fun u => P.cls r u = Class.inference).card ≤ lim.n) &&
    decide (∀ u, P.cls r u = Class.inference →
      (P.claimsIn r u).any isSize = true ∧ claimedSize (P.claimsIn r u) ≤ lim.c ∧
        (P.claimsIn r u).any (freshAtMost lim.k) = true) &&
    decide ((univ.filter fun u => P.cls r u = Class.free).card ≤ lim.m) &&
    decide (∀ i, P.kind r i = true → P.capacity r i - P.credited r i + P.tol ≤ lim.freeSize) &&
    decide (∀ i, P.kind r i = true →
      (P.claimsOut r i).any isSize = true ∧ claimedSize (P.claimsOut r i) ≤ lim.cS) &&
    decide (∀ i, P.kind r i = true →
      (P.claimsOut r i).any (isInputFrom [.past]) = true ∧ (P.claimsOut r i).any (isInputFrom [.world]) = true ∧
        claimedInput [.past] (P.claimsOut r i) + claimedInput [.world] (P.claimsOut r i) ≤ lim.kS)

end InferenceOnly
