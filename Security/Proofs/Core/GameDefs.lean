import Definitions.Core.Game

/-! The proof-only part of `Definitions.Core.Game`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Verity

open scoped ENNReal

namespace Game

variable {α β : Type}

/-- The prover sends a message of type `M`. -/
def recv (M : Type) : Game M := .send M .ret

/-- The verifier draws a uniform coin from `C`. -/
def draw (C : Type) [Fintype C] [Nonempty C] : Game C := .coin C .ret

end Game

end Verity
