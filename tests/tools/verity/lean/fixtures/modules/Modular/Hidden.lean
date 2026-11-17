module

public import Modular.Defs

/-! A public statement whose proof, as every theorem's under the module system, only the private part of the module's
`.olean` holds: the `.olean` its importers read gives the theorem as an axiom. -/

namespace Modular

public theorem double_add (n : Nat) : double n = n + n := by
  unfold double
  omega

end Modular
