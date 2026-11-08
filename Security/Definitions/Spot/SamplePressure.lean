module

public import Definitions.Core.Draw
public import Mathlib.Data.Fintype.Powerset
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Analysis.Complex.Order
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Data.EReal.Inv

/-!
# The deployed sample: `k` of the `n` units, uniformly and without replacement

`Definitions.Spot.Exfiltration`'s game over the sample the audit makes: each window's coin is a `k`-subset of the
`n` units (`Verity.Draw.Subset n k`), every one of the `C(n, k)` subsets equally likely, and a window escapes when its
coin misses every one of the window's incorrect proof blocks.

* `uniformSample n k` is the uniform weight on the coins.
* `missing n k B` is the samples that miss the wrong set `B`: a window whose incorrect proof blocks are `B` escapes on them.
-/

@[expose] public section

namespace Population.Exfil

open Finset

/-- The uniform weight on the samples: each of the `C(n, k)` samples has weight `1 / C(n, k)`. -/
noncomputable def uniformSample (n k : ℕ) (_ : Verity.Draw.Subset n k) : ℝ := 1 / n.choose k

/-- The samples that miss the wrong set `B`. -/
def missing (n k : ℕ) (B : Finset (Fin n)) : Finset (Verity.Draw.Subset n k) := univ.filter fun s => Disjoint s.1 B

end Population.Exfil
