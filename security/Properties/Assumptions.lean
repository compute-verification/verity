module

public import Properties.Assumptions.Pous

/-!
# The named assumptions stated over the code

The second module a named assumption of this package is declared in, with the modules under it: this lock's
`assumptions_module` lists it beside the spec's `Definitions.Assumptions`. It holds the real assumptions whose
statements read the code's own functions, which the spec can't import. Each is a `Prop` taken as a hypothesis, never an
`axiom`, and listed by name in `lean-audit.json` with its record, like the spec's. PoUS's instantiation in the ring
model is `Properties.Assumptions.Pous`, which this module re-exports.
-/
