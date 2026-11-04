import Definitions.Nci.Assumptions.Capability.Information

/-!
# Local bounds: the premises

Why splitting declared computation into small inference blocks should slow the acquisition of a new capability. Every
local bound has three ingredients, and only two of them carry premises:

* **information** (`Information`): which jointly selected values are expensive to hold. `DescriptionHard`
  (expensive to recompute), `MinEntropyHard` (unpredictable), and `SourceHard` with `Recovers` (derived from a hard
  source the outputs reveal), each over an item family.
* **structure**: how much counted computation held values support. Definitions checkable on a named circuit
  (`Nci.Capability.Incidence`, `Nci.Capability.Consumers.Upstream`, `Nci.Capability.Consumers.OverlapWeights`), no
  premise.
* **coverage**: every sufficiently cheap successful method contains enough of that computation.
  `CoversEfficientMethods`, stated over tasks and runs in `Definitions.Accounting.Assumptions.Coverage`.

The rates and deficits belong to the decoder budget, the distribution of runs, the registration, the arithmetic and the
computation evaluated; none is a universal bits-per-value constant. The smallest sufficient premise comes first.
Protocol soundness and the device model are outside these: they establish where the block restrictions apply, not that
learning is hard.
-/
