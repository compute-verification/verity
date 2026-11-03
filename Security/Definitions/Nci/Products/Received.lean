import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products

/-!
# Products of two fresh operands: partial sums received at their width

Widths as passed (`Standard.Passed`) charge a product's partial sums on the side that hands them on: a block holding a
whole MAC of `C_ij` hands `b_C` of `C_ij`'s bits to the rest of `C_ij`, or holds them as its final bits, and whichever
block reads them pays. **Partial sums received at their width** (`Standard.Received`) charge the block that holds the
MAC: there are sets `start i j` of at most `b_C` gates, where `C_ij`'s partial sum starts, and an inference block
holding a whole MAC of `C_ij` reads at least `b_C` of `C_ij`'s gates beyond `θ`, less the gates of `start i j` it holds.
Each entry a block touches then either starts its chain there or has its partial sum read at its width, from another
block or across a MAC split between blocks, and each entry's start is held once over a partition.

`chainBlocks N Γ B` is the fewest blocks `m ≥ 0` with `m Γ + B √m ≥ N`, `(2N / (√(B² + 4ΓN) + B))²`, and
`noCellBound` the bound on `ρ` that the cap without the word level (`Nci.Products.Guarantees.NoCellCap`) gives,
`chainBlocks (ni nl nj) (α + k) (β √(ni nj)) · c / |C|`, with `α = (k/3) √(k/3) / √(b_A b_B b_C)` and
`β = (k/2) / √(b_A b_B)`.
-/

namespace Nci.Products

open Finset
open Verity (Declaration)
open Nci.Formalism
open scoped ENNReal

/-- **Partial sums received at their width** at the policy's `(c, k)`, a fact of the given circuit: there are sets
`start i j` of at most `b_C` gates, where `C_ij`'s partial sum starts, such that an inference block holding a whole MAC
of `C_ij` reads at least `b_C` of `C_ij`'s gates beyond `θ`, less the gates of `start i j` it holds. -/
def Standard.Received {d : Declaration} {ni nl nj : ℕ} (S : Standard d ni nl nj) (c k bC : ℕ) : Prop :=
  ∃ start : Fin ni → Fin nj → Finset (Fin d.D.N), (∀ i j, (start i j).card ≤ bC) ∧
    ∀ I, InferenceBlock d c k I → ∀ i l j, S.mac i l j ⊆ I →
      bC ≤ (d.unregistered I ∩ S.entry i j).card + (I ∩ start i j).card

/-- **The fewest blocks** `m ≥ 0` with `m Γ + B √m ≥ N`, for `N, Γ, B ≥ 0`: `(2N / (√(B² + 4ΓN) + B))²`, the square of
the root `s ≥ 0` of `Γ s² + B s = N`. -/
noncomputable def chainBlocks (N Γ B : ℝ) : ℝ :=
  (2 * N / (√(B ^ 2 + 4 * Γ * N) + B)) ^ 2

/-- **The products kind's bound without the word level**: `chainBlocks (ni nl nj) (α + k) (β √(ni nj)) · c / |C|`, the
blocks that the cap without the word level needs for the `ni nl nj` products of `C = A · B`, priced at `c` a block,
against `|C|`. -/
noncomputable def noCellBound (d : Declaration) (ni nl nj c k bA bB bC : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal (chainBlocks ((ni * nl * nj : ℕ) : ℝ)
      ((k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + k)
      ((k : ℝ) / 2 / √((bA : ℝ) * bB) * √((ni * nj : ℕ) : ℝ))) *
    (c : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

end Nci.Products
