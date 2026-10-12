import Definitions.Pous.Model.Dense

/-!
# A segment's parents at their global positions (trusted)

`liftPar`, which places a segment's local parents at their global block indices. The transfer from the global keying to
the tagged one (`PousDenseReCert.segTag_auditSecure`) is stated over it. `instNeZeroSeg14` is the `NeZero` instance the
14 GB statements (`seg_meets_14`, `segTag_meets_14`) take, scoped to `PousDenseReCert`: `Pous.BandMulti` declares its
own, which the band's statements take.
-/

namespace PousDenseReCert

open Pous Pous.Dense

/-- Local parents `p` of a query in segment `s`, placed at their global positions (`none` outside segment `s`). -/
def liftPar (N B ℓ : ℕ) (s : Fin N) (p : Fin B → Option (Bits ℓ)) : Fin (N * B) → Option (Bits ℓ) :=
  fun g => if ((segIndexEquiv N B).symm g).1 = s then p ((segIndexEquiv N B).symm g).2 else none

/-- `448 * 2^9 ≠ 0`: the 14 GB store's block count. -/
scoped instance instNeZeroSeg14 : NeZero (448 * 2 ^ 9) := ⟨by norm_num⟩

end PousDenseReCert
