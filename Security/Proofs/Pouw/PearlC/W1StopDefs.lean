import Definitions.Pouw.PearlC.W1Stop
import Definitions.Pouw.PearlC.W1

/-! The proof-only part of `Definitions.Pouw.PearlC.W1Stop`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **The priced W1 on the stopping machine.** -/
noncomputable abbrev W1.w1Stop (dv : W1.W1Device) (salt : S → List ℕ) : CostModel Q R S :=
  (W1.w1Priced dv salt : CostModel Q R S).withStop

end Pouw.PearlC
