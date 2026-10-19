import Definitions.Population.Dataflow
import Definitions.Population.Partition
import Definitions.Population.LongValue
import Definitions.Population.Oblivious
import Definitions.Population.AddressMap

/-! Populations' vocabulary: units over committed value slots with public read and write sets (`Dataflow`), unit
functions and correctness, what a checker confirms of the dataflow (single assignment, acyclicity), cones and egress,
and the value-dependent variant a machine trace is (`DynPopulation`). The question it serves: which object sampled
proofs' guarantee, "at most `m` units are wrong", must be about for each policy to follow from it. `Partition`: a
circuit's partition as a population over its committed wires, and the orders on its units. `LongValue`: slots that
hold `Option V`, ⊥ where no opening has the declared width, and the verifier's view of a run's openings. `Oblivious`:
when a trace's read sets depend on no value (or only on the public inputs), the classical notion that constrains only
honest runs, address closure, and the padding of a trace into a static population. `AddressMap`: a program's read sets
without its unit functions, and the static wirings that express it. -/
