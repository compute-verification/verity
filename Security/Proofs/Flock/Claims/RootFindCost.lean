import Proofs.Flock.Claims.RootFindAlg

/-!
# The root finder's operation count

`rootsC_ops_le`: `rootsC n basis f` takes at most `costBound n basis.length f.length` field operations, for every `f`.
-/

namespace Claims.RootFind

open Polynomial

/-- **The root finder's operation count** is polynomial in `f`'s length, `n` and `basis`'s length. -/
theorem rootsC_ops_le {F : Type} [Field F] [DecidableEq F] (n : ℕ) (basis f : List F) :
    (rootsC n basis f).ops ≤ costBound n basis.length f.length := by
  sorry

end Claims.RootFind
