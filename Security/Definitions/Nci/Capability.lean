import Definitions.Nci.Capability.Items
import Definitions.Nci.Capability.Products
import Definitions.Nci.Capability.Consumers
import Definitions.Nci.Capability.Fresh

/-!
# Local bounds: the vocabulary

One ontology for every local bound on what a small inference block can do, on core's circuit, registration and local
description length (`Definitions.Core.Formalism`). A block has `c` gates and `k` fresh bits: it can **import**
information, spending its bits, or **reconstruct** it, spending its gates. Four modules, one per kind of object:

* `Items`: what a block can hold. `Items`, `Items.sum`, `Selects`, `selectionBits`, `holds`, `HeldPriceAtMost`.
* `Products`: the counted computation and its incidence. `completedIn`, `served`, `SlotsRead`, `Incidence`.
* `Consumers`: what a block finishes and what that depends on. `Consumers` (with `ofUses`, `ofOutputs`), `servedIn`,
  `Upstream`, `OverlapWeights`, `RequiresState`.
* `Fresh`: what else a block reads fresh, the advice and what faulty proof blocks inject. `EvaluatesAdvice`,
  `injected`, `cutReg` (with `cutCircuit`, `cutPartition`, `cutDecl`, `cutInputs`).

The premises (`Definitions.Nci.Assumptions.Capability`) are stated over items; the bounds
(`Definitions.Nci.Guarantees.Capability`) consume a budget over items and a structure over products or consumers. The
survey of both is `Security/Definitions/Nci/Capability/README.md`.
-/
