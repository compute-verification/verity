import Definitions.Pous.Model.SegLift

/-! The proof-only part of `Definitions.Pous.Model.SegLift`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace PousDenseReCert

open Pous Pous.Dense

/-- Local parents `p` of a query in segment `s`, placed at their global positions (`none` outside segment `s`). -/
def liftPar (N B ℓ : ℕ) (s : Fin N) (p : Fin B → Option (Bits ℓ)) : Fin (N * B) → Option (Bits ℓ) :=
  fun g => if ((segIndexEquiv N B).symm g).1 = s then p ((segIndexEquiv N B).symm g).2 else none

scoped instance instNeZeroSeg14 : NeZero (448 * 2 ^ 9) := ⟨by norm_num⟩

end PousDenseReCert
