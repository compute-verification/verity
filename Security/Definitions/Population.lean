import Definitions.Population.Dataflow
import Definitions.Population.Partition

/-! Populations' vocabulary: units over committed value slots with public read and write sets (`Dataflow`), unit
functions and correctness, what a checker confirms of the dataflow (single assignment, acyclicity), cones and egress,
and the value-dependent variant a machine trace is (`DynPopulation`). The question it serves: which object sampled
proofs' guarantee, "at most `m` units are wrong", must be about for each policy to follow from it. `Partition`: a
circuit's partition as a population over its committed wires, and the orders on its units. -/
