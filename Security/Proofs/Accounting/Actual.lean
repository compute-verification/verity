import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Windows
import Proofs.Accounting.Actual.Runtime
import Proofs.Accounting.Actual.Localization
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Actual.Markov
import Proofs.Accounting.Actual.Timing

/-!
# The actual circuit's guarantees

The model of `note:console/20261012T0645Z-draft-actual-circuit-model` (`Security/Definitions/Accounting/Actual.lean`):

* `Actual/Split.lean`: what a region reads from outside itself is what its location carried in plus what crossed its
  boundary.
* `Actual/Physical.lean`: causality at any latency makes wires run forward in time inside every location; under
  `Links`, ingress and egress are at most `β_X` a tick; compute rate at a pricing of at least one per gate bounds the
  gates.
* `Actual/Windows.lean`: `Windows`, the capacity form: the windows divide the developer's region into IUs of at most
  `Φ_X ω` gates, each reading at most `M_X + β_X ω` novel bits.
* `Actual/Runtime.lean`: `Runtime`, the corollary `t_A(P) ≥ ω (G f(K_A) / (n K_A) − 1)`, from `Windows` and B5.
* `Actual/Localization.lean`: `Localization` at `e = 0`: what a challenge reaches only through `X` reaches the rest of
  the prover only through `X`'s egress, and not at all when `X` sends it nothing of the challenge.
* `Actual/Confidence.lean`: `Composition`, protocols sound alone compose into the confidence set at `∑ δ i`, and
  `Fit`, `req_μ(P) ≤ res_μ` when `μ` adds; causality's global form from the local one.
* `Actual/Markov.lean`: `not_markov`, a locally causal strategy of feasible runs in which a location carries nothing
  across a tick and reads nothing in a window, yet what it runs there depends on the coin: `mem_X` bounds what `X`
  knows at a challenge only under causality's Markov form.
* `Actual/Timing.lean`: `zero_link`, a link of capacity 0 that carries a bit in a width-0 wire's tick: under `Links`
  widths bound information only when cross-leaf wires have width at least 1 and timing is counted.
-/
