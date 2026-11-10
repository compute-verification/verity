import Definitions.InferenceOnly.Capability.Items
import Definitions.InferenceOnly.Capability.Products
import Definitions.InferenceOnly.Capability.Consumers

/-!
# Local bounds: the vocabulary

One ontology for every local bound on what a small inference block can do, on core's circuit, registration and local
description length (`Definitions.Core.Formalism`). A block has `c` gates and `k` fresh bits: it can **import**
information, spending its bits, or **reconstruct** it, spending its gates. Three modules, one per kind of object:

* `Items`: what a block can hold. `Items`, `Items.sum`, `Selects`, `selectionBits`, `holds`, `HeldPriceAtMost`.
* `Products`: the counted computation and its incidence. `completedIn`, `served`, `SlotsRead`, `Incidence`.
* `Consumers`: what a block finishes and what that depends on. `Consumers` (with `ofUses`, `ofOutputs`), `servedIn`,
  `Upstream`, `OverlapWeights`, `RequiresState`.

The premises (`Definitions.InferenceOnly.Assumptions.Capability`) are stated over items; the bounds
(`Definitions.InferenceOnly.Guarantees.Capability`) consume a budget over items and a structure over products or consumers. The
survey of both is `Security/Definitions/InferenceOnly/Capability/README.md`.
-/
