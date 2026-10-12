import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Windows
import Proofs.Accounting.Actual.Runtime
import Proofs.Accounting.Actual.Localization
import Proofs.Accounting.Actual.Source

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
* `Actual/Source.lean`: ingress by source adds over disjoint sources, and a cover of the outside bounds ingress; under
  `Links`, ingress from a source is at most the capacity of the links joining it to `X`.
-/
