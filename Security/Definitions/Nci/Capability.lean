module

public import Definitions.Nci.Capability.Items
public import Definitions.Nci.Capability.Consumers

/-!
# Local bounds: the vocabulary

One ontology for every local bound on what a small inference block can do, on core's circuit, registration and local
description length (`Definitions.Core.Formalism`). A block has `c` gates and `k` fresh bits: it can **import**
information, spending its bits, or **reconstruct** it, spending its gates. Two modules, one per kind of object:

* `Items`: what a block can hold. `Items`, `Selects`, `selectionBits`, `holds`, `HeldPriceAtMost`.
* `Consumers`: what a block finishes and what that depends on. `Consumers` (with `ofUses`, `ofOutputs`), `servedIn`,
  `Upstream`, `OverlapWeights`, `RequiresState`.

The premises (`Definitions.Nci.Assumptions.Capability`) are stated over items; the bounds
(`Definitions.Nci.Guarantees.Capability`) consume a budget over items and a structure over consumers. The survey of
both is `Security/Definitions/Nci/Capability/README.md`.
-/

@[expose] public section
