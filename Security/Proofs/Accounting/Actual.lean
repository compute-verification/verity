import Proofs.Accounting.Actual.Split
import Proofs.Accounting.Actual.Physical
import Proofs.Accounting.Actual.Windows
import Proofs.Accounting.Actual.Runtime
import Proofs.Accounting.Actual.Threaded
import Proofs.Accounting.Actual.Dampener
import Proofs.Accounting.Actual.Localization
import Proofs.Accounting.Actual.Device
import Proofs.Accounting.Actual.Confidence
import Proofs.Accounting.Actual.Bridges
import Proofs.Accounting.Actual.Transducer
import Proofs.Accounting.Defs.Actual.Source
import Proofs.Accounting.Actual.Sampler

/-!
# The actual circuit's guarantees

The model of `Security/Definitions/Accounting/Actual.lean`:

* `Actual/Split.lean`: what a region reads from outside itself is what its location carried in plus what crossed its
  boundary, and so in any set of gates.
* `Actual/Physical.lean`: causality at any latency makes wires run forward in time inside every location; under
  `Links`, ingress and egress are at most `β_X` a tick; compute rate at a pricing of at least one per gate bounds the
  gates; the state is its part in a set of gates plus its part outside.
* `Actual/Windows.lean`: `WindowsAt`, the capacity form at any cuts and with no memory constraint: the windows divide
  the developer's region into isolation blocks of at most `Φ_X (τ (j + 1) − τ j)` gates, each reading in a set `G` at
  most the state's part in `G` at its cut plus `β_X (τ (j + 1) − τ j)`; `Windows`, its instance at the cuts `kω` and
  `G = univ`, isolation blocks of at most `Φ_X ω` gates, each reading at most `M_X + β_X ω` novel bits.
* `Actual/Runtime.lean`: `Runtime`, the corollary `t_A(P) ≥ ω (G f(K_A) / (n K_A) − 1)`, from `Windows` and B5.
* `Actual/Threaded.lean`: `ThreadedRuntime`, the memory cap over threads: when the developer's region over `[0, t)`
  contains a threaded reference circuit, `b_a |M| ≤ n ⌈t/ω⌉ K Λ(K)` with `K = R + r ω` the state at a window's start
  plus its ingress, from `WindowsAt` pulled back through the containment and NCI's `ThreadedReads` per window;
  `ThreadedRate`, at the window `ω = R/r` in the quadratic regime, at most `reuse · n R r / (d b_w b_a)` counted
  multiplications a tick; `ThreadedRuntimeDark`, the cap through a partial containment, with every counted
  multiplication the declaration doesn't compute a dark gate.
* `Actual/Dampener.lean`: `DampenerWindowsCap`, the compute dampener's windows hypothesis
  (`Nci.Guarantees.DampenerWindows`) from `ThreadedRuntime` at a whole-tick window, when one threading counts the
  residual and the declared computation together.
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
* `Actual/Transducer.lean`: a location is a transducer over what reaches it: under causality from every tick, what `X`
  does during `[t, t + n)` is a function of its content at `t` and of its links' signals at each tick (`transducer`,
  `crossing_of_signals`), with an `(M + 1)`-bit state (`transducer_bits`), so it takes at most `2^{M + 1 + β_X n}` values
  (`acts_encard_le_links`; the `+ 1` is the content's length).
* `Actual/Sampler.lean`: the deployed uniform sampler on one word (`Accounting.Actual.wordIndex`) accepts
  `2^w − (2^w mod N)` words and gives each index at most `⌊2^w / N⌋` of them (`card_accepted`, `card_fiber_le`), with
  the probability facts PoUB's opening draw is proved with.

The wipe is PoUS's streamed-label wipe (`Pous.StreamedWipe`, its guarantees `Pous.Guarantees.StreamedWipeMeetsFamily`
and `StreamedWipeMeetsReads`), read through core's `Verity.Wipe.SurvivorBound`.
-/
