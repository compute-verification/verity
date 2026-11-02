import Definitions.Population.Dataflow
import Definitions.Population.Partition
import Definitions.Population.LongValue
import Definitions.Population.Oblivious
import Definitions.Population.AddressMap
import Definitions.Population.Necessity
import Definitions.Population.Exfiltration
import Definitions.Population.DrawPressure
import Definitions.Population.ChiPool

/-! The population sampled proofs sample from: proof blocks over committed value slots with public read and write
sets (`Dataflow`), each proof block's function and its correctness, what a checker confirms of the dataflow (single
assignment, acyclicity), cones and egress, and the value-dependent variant a machine trace is (`DynPopulation`). It
fixes which object sampled proofs' guarantee, "at most `m` proof blocks are incorrect", is about, so that each policy
follows from it.

* `Partition`: a circuit's partition as a population over its committed wires, and the orders on its proof blocks.
* `LongValue`: slots that hold `Option V`, ⊥ where no opening has the registered width, and the verifier's view of a
  run's openings.
* `Oblivious`: when a trace's read sets depend on no value (or only on the public inputs), the classical notion that
  constrains only honest runs, address closure, and the padding of a trace into a static population.
* `AddressMap`: a program's read sets without its proof blocks' functions, and the static wirings that express it.
* `Necessity`: the IOP's sampling certificate (a proof transcript before the coins, `k` samples from a distribution
  fixed before it, an always-proved set), amplification, and each policy's damage (sampled proofs' damaged egress,
  control at the public widths, a range-restricted dataflow).
* `Exfiltration`: the audit over live-coin execution phases as a game, an adaptive prover that halts at the first
  detection, with the named assumptions `DetectionPressure` and `TrustedModel`, the independent sample, and deviations
  from the honest egress.
* `DrawPressure`: the deployed sampling design, `k` of the `n` proof blocks uniformly without replacement, and the
  samples that miss an incorrect set.
* `ChiPool`: the halt `Δ` phases after the first caught phase's commitment, and what leaves through it
  (`Strategy.cumLag`, `Strategy.cumUpToLag`). -/
