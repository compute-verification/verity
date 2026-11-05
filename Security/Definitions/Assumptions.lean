module

public import Definitions.Hash
public import Definitions.Core
public import Definitions.Accounting

/-!
# The named assumptions

The one module a named assumption is declared in (`assumptions_module` in `lean-audit.json`): the few `Prop`s the
guarantees take without proof because they are real assumptions (a hash's collision resistance, the auditor's devices),
each taken as a hypothesis, never an `axiom`. Every other premise a guarantee takes is open: the lock lists it under
`open` with the issue that discharges it and its due date. An area moves an assumption of its own here once it is one of
the real ones.
-/
