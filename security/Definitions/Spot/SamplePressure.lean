module

public import Definitions.Core.Draw
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Probability.Distributions.Uniform
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

* `uniformSample n k hk` is the uniform distribution on the coins, which exist when `k ≤ n`.
* `missing n k B` is the samples that miss the incorrect set `B`: a window whose incorrect proof blocks are `B` escapes on them.
-/

@[expose] public section

namespace Population.Exfil

open Finset

/-- The uniform distribution on the samples: each of the `C(n, k)` samples has probability `1 / C(n, k)`. -/
noncomputable def uniformSample (n k : ℕ) (hk : k ≤ n) : PMF (Verity.Draw.Subset n k) :=
  haveI : Nonempty (Verity.Draw.Subset n k) :=
    (exists_subset_card_eq (s := (univ : Finset (Fin n))) (by simpa using hk)).elim fun t ht => ⟨⟨t, ht.2⟩⟩
  PMF.uniformOfFintype _

/-- The samples that miss the incorrect set `B`. -/
def missing (n k : ℕ) (B : Finset (Fin n)) : Finset (Verity.Draw.Subset n k) := univ.filter fun s => Disjoint s.1 B

end Population.Exfil
