import Definitions.Core.PrivatePartition

/-! The proof-only part of `Definitions.Core.PrivatePartition`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Verity

open Finset

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- `og(u)`: the number of unit `u`'s outputs. -/
def outCount (u : Fin n) : ℕ := (P.outs u).card

end Partition

end Verity
