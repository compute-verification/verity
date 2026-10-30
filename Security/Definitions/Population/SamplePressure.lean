import Definitions.Population.Exfiltration

/-!
# The deployed draw: `k` of the `n` units, uniformly and without replacement

`Definitions.Population.Exfiltration`'s game over the draw the audit makes: each window's coin is a `k`-subset of the
`n` units, every one of the `C(n, k)` subsets equally likely, and a window escapes when its coin misses every one of the
window's wrong units.

* `Draw n k` is the coin space, the `k`-subsets of `Fin n`, and `uniformSample n k` its uniform weight.
* `missing n k B` is the draws that miss the wrong set `B`: a window whose wrong units are `B` escapes on them.
-/

namespace Population.Exfil

open Finset

/-- A coin of the deployed draw: `k` of the `n` units, drawn without replacement. -/
abbrev Draw (n k : ℕ) : Type := {s : Finset (Fin n) // s.card = k}

/-- The uniform weight on the draws: each of the `C(n, k)` draws has weight `1 / C(n, k)`. -/
noncomputable def uniformSample (n k : ℕ) (_ : Draw n k) : ℝ := 1 / n.choose k

/-- The draws that miss the wrong set `B`. -/
def missing (n k : ℕ) (B : Finset (Fin n)) : Finset (Draw n k) := univ.filter fun s => Disjoint s.1 B

end Population.Exfil
