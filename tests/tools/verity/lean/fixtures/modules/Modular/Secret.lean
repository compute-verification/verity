module

public import Modular.Defs

/-! A public statement about `triple`, whose body this module can't unfold: its proof goes through `triple_eq`. -/

namespace Modular

public theorem triple_add (n : Nat) : triple n = n + n + n := by
  rw [triple_eq]
  omega

end Modular
