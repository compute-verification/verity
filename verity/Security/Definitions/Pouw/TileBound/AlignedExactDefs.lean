import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Rat.Cast.Order

/-!
# Aligned exactness: the definitions (trusted)

For `AlignedExact.lean`: word `(i, j)`'s exactness over a window as a predicate `f i j` of row `i`'s seed and column `j`'s
seed, the one-row hypothesis `RowCompat`, and the event `HasRegion` that some `a × b` block is exact everywhere.

What `no-aligned-exact-region/sm120-unpromoted` instantiates them with:
* **`ρ` and `κ` are the raw random-oracle outputs** from which a row's and a column's noise lines are derived (E_A's and
  E_B's), which are uniform. They are not the lines: a line's image under Pearl's line rule is not uniform, and the
  counts divide by `Fintype.card ρ`. `f` applies the line rule.
* **`f i j r c`** is: word `(i, j)` is exact on every step of one fixed window, for fixed F lines, weights and
  activations, when row `i`'s noise line comes from `r` and column `j`'s from `c`.
* **The activations are the prover's**, chosen after it sees E_B and F (both per job) and before E_A (per unit) is drawn.
  So a citation needs `RowCompat` for every in-domain choice of activations, aligned-spike families included, and the
  fresh side is the rows. The transposed predicate `fun j i c r => f i j r c` would make the columns the fresh side,
  which is sound only if E_B were drawn after the activations are fixed. Pearl-C's keying doesn't give that.
-/

namespace Pouw.AlignedExact

open Finset Classical

variable {m n : ℕ} {ρ κ : Type*} [Fintype ρ] [Fintype κ]

/-- **One row against `b` columns, in the worst case over the columns' seeds:** for every row, every `b` columns and
every choice of their seeds, at most a `θ` share of row seeds make the row exact with all of them.

`∀ c` is the right quantifier, since the prover may know every column's seed when it chooses the activations. It is
also why no sampled rate establishes `RowCompat`: sampling estimates the share for typical column seeds, and this asks
for every one. So `θ` needs an argument, and measurements support it only heuristically. -/
def RowCompat (f : Fin m → Fin n → ρ → κ → Prop) (b : ℕ) (θ : ℚ) : Prop :=
  ∀ (i : Fin m) (J : Finset (Fin n)), J.card = b → ∀ c : Fin n → κ,
    ((univ.filter fun r : ρ => ∀ j ∈ J, f i j r (c j)).card : ℚ) ≤ θ * Fintype.card ρ

/-- The seed assignments with an all-exact `a × b` block. -/
def HasRegion (f : Fin m → Fin n → ρ → κ → Prop) (a b : ℕ) (x : (Fin m → ρ) × (Fin n → κ)) : Prop :=
  ∃ I : Finset (Fin m), ∃ J : Finset (Fin n), I.card = a ∧ J.card = b ∧ ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)

end Pouw.AlignedExact
