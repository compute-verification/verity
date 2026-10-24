import Definitions.Core.Circuit

/-! The proof-only part of `Definitions.Core.Circuit`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Verity

open Finset

/-- `C ≼ D`: `D` implements `C`. It may compute anything else, and need not compute `C`'s outputs the way `C` does. -/
def Implements (C D : Circuit) : Prop := Nonempty (Implementation C D)

/-- `D` contains `C`: `C ≼ D`, and `D` computes every non-input gate of `C` from the same wires. -/
def Contains (C D : Circuit) : Prop := Nonempty (Containment C D)

end Verity
