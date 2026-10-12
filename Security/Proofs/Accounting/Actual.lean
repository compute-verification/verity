import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Windows
import Proofs.Accounting.Actual.Runtime
import Proofs.Accounting.Actual.Localization
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Actual.Ticks

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
* `Actual/Localization.lean`: `Localization`, over admissible strategies: with what the rest can tell at the window's
  start and what it hears from the verifier fixed, what the rest does is a prefix-causal table keyed by what `X` sent
  it, of at most `2^e` entries; `localization_wires`, the per-run lemma on wires; memory over continuations bounds each
  run's state (`memory_of_memoryOver`).
* `Actual/Confidence.lean`: `Composition`, protocols sound alone compose into the confidence set at `∑ δ i`, and
  `Fit`, `req_μ(P) ≤ res_μ` when `μ` adds; the state over continuations is a function of what the location can tell
  (`stateOver_congr`), and causality's global form (`global_of_admissible`).
* `Actual/Ticks.lean`: a counterexample, `held_tick`: causality reads the state as seen, ticks included, and memory
  bounds its width, so a location with no memory holds a bit in the tick of a width-0 gate; memory as an alphabet
  across runs (`MemoryAlphabet`) refuses it.
-/
