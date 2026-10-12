import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Windows
import Proofs.Accounting.Actual.Runtime
import Proofs.Accounting.Actual.Localization
import Proofs.Accounting.Actual.Device
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Actual.Bridges

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
* `Actual/Localization.lean`: `Localization`, over strategies causal from every tick: with what the rest can tell at
  the window's start and what it hears from the verifier fixed, what the rest does is a prefix-causal table keyed by
  what `X` sent it, of at most `2^e` entries; `localization_wires`, the per-run lemma on wires; memory per run on the
  state bounds the circuit's `state` (`memory_of_mem`), and binds on memory accounting's `not_markov` circuit
  (`markov_mem`), and admits the honest location that holds two inputs in two cells and computes `a ∧ ¬b`
  (`and_causal`); `Recall`, B2's count: what a location does is a function of its state's content, at most `mem`
  bits; `zero_link` fails `Alphabet` at capacity 0 while meeting `Links` (`zero_link_alphabet`, `zero_links`).
* `Actual/Device.lean`: the bridge to the device model: the schedule a run induces, its chip and memory charges what
  those locations hold, fits the device (`fits`, `fits_of_feasible`) and charges what an account's gates use
  (`covers_ingress`, `covers_work`).
* `Actual/Confidence.lean`: `Composition`, protocols sound alone compose into the confidence set at `∑ δ i`, and
  `Fit`, `req_μ(P) ≤ res_μ` when `μ` adds; at tick 0 the state is the inputs (`state_zero`), and causality's global
  form (`global_of_causal`); the content is a function of the state as seen (`content_congr`), of at most `mem` bits
  (`content_length_le`).
* `Actual/Bridges.lean`: `B4`, egress factoring: across `X`'s secret inputs, what the outside does takes at most (the
  number of records) × `2^{res_out}` values, from causality, links as alphabets and the trusted devices across the
  strategy; `CM_A`'s programs carry their oracle tree (`costModel_queries`), the honest clock holds at `CM_A` from
  `t_d` on (`honestClock`), so B1 is all that is conjectured of PoUW's deadline row, and a salt-dependent gate that
  the verifier reads before `t_d` lies in the cost window (`cost_window`, `prover_cost_window`).
-/
