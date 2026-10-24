import Definitions.Population.Dataflow
import Definitions.Population.Partition
import Definitions.Population.LongValue
import Definitions.Population.Oblivious
import Definitions.Population.AddressMap
import Definitions.Population.Necessity
import Definitions.Population.Exfiltration
import Definitions.Population.DrawPressure
import Definitions.Population.ChiPool

/-! Populations' vocabulary: units over committed value slots with public read and write sets (`Dataflow`), unit
functions and correctness, what a checker confirms of the dataflow (single assignment, acyclicity), cones and egress,
and the value-dependent variant a machine trace is (`DynPopulation`). The question it serves: which object sampled
proofs' guarantee, "at most `m` units are wrong", must be about for each policy to follow from it. `Partition`: a
circuit's partition as a population over its committed wires, and the orders on its units. `LongValue`: slots that
hold `Option V`, ⊥ where no opening has the declared width, and the verifier's view of a run's openings. `Oblivious`:
when a trace's read sets depend on no value (or only on the public inputs), the classical notion that constrains only
honest runs, address closure, and the padding of a trace into a static population. `AddressMap`: a program's read sets
without its unit functions, and the static wirings that express it. `Necessity`: the IOP's sampling certificate (a
transcript before the coins, `k` draws from a distribution fixed before it, an always-proved set), amplification, and
what each policy's damage is (integrity's damaged egress, control at the public widths, a range-restricted dataflow).
`Exfiltration`: the windowed audit as a game, an adaptive prover over live-coin windows that halts at the first
detection, with the named assumptions `DetectionPressure` and `TrustedModel`, the independent draw, and deviations from
the honest egress. `DrawPressure`: the deployed draw, `k` of the `n` units uniformly without replacement, and the draws
that miss a wrong set. `ChiPool`: the halt `Δ` windows after the first caught window's commitment, and what leaves
through it (`Strategy.cumLag`, `Strategy.cumUpToLag`). -/
