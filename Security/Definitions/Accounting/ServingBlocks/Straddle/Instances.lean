import Definitions.Accounting.ServingBlocks.Straddle
import Definitions.Accounting.ServingBlocks.Instances

/-!
# Layouts across the certified boundary (trusted)

The instances the honest and nontrivial lemmas of `Accounting.Guarantees.ServingBlocks.Straddle` are stated at, besides
`Honest`'s dark remainder and `Counter`'s one gate (`Security/Definitions/Accounting/ServingBlocks/Instances.lean`).
`Straddle` and `Detached` are read here as one server's gates `{1, 2}` with one inference block among them:

* `Straddle.certified`, the block `{1}`: the operation gate 2 represents takes its first gate from the block and
  computes the rest, gate 2, outside it. It is a product the served inference computes, read by an uncertified
  accumulation as part of one counted operation.
* `Detached.certified`, the block `{1}`, and `Detached.represented`, the block `{2}`: the same operation with gate 2
  reading the registered gate 0 instead of gate 1, so no gate of the operation reads gate 1.
-/

namespace Accounting.ServingBlocks

namespace Straddle

/-- Gates 1 and 2 as one server's gates. -/
abbrev server : Finset (Fin d.D.N) := {1, 2}

/-- The server's one inference block, gate 1: the operation's first gate is certified, and gate 2, which reads it, is
not. -/
abbrev certified : Unit → Finset (Fin d.D.N) := fun _ => {1}

end Straddle

namespace Detached

/-- Gates 1 and 2 as one server's gates. -/
abbrev server : Finset (Fin d.D.N) := {1, 2}

/-- The server's one inference block, gate 1. -/
abbrev certified : Unit → Finset (Fin d.D.N) := fun _ => {1}

/-- The server's one inference block, gate 2, the operation's representative. -/
abbrev represented : Unit → Finset (Fin d.D.N) := fun _ => {2}

end Detached

end Accounting.ServingBlocks
