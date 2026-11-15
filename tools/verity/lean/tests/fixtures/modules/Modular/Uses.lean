module

public import Modular.Defs
import Modular.Hidden
import Modular.Secret

/-! The module controls' guarantee: its statement reads `double` and `triple`, and its proof uses the two modules above,
which this one imports privately, so its importers see neither. -/

namespace Modular

public theorem main_thm (n : Nat) : double n + n = triple n := by
  rw [double_add, triple_add]

end Modular
