import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products

/-!
# Products of two fresh operands: the amortized cap and the slowdown it gives

The theorems about a product `C = A · B` by the standard algorithm with its widths as passed
(`Definitions.Nci.Products`), proved in `Nci.Products.SecurityProofs`. Each is about every partition `𝓘` of the
declaration into `m` inference blocks, each block reading at most `k` bits beyond `θ`, and charges only those reads.
`N = ni nl nj` is the number of products and `E = Σ_ij |out_ij|` the bits of `C`'s final values.

* **The fresh-products cap** (`Cap`), at the bit level: a partition may split a MAC's gates across blocks and pass any of
  their bits. For every weight `μ > 0`, `N ≤ m max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k) + μ E`. A block's whole MACs
  are at most `√(x y z)` by Loomis–Whitney, `x`, `y` and `z` the entries of `A`, `B` and `C` they touch; the block reads
  at least `b_A x + b_B y` bits for their operands, and `b_C z` is paid across the partition, by the partial sums the
  blocks hand each other and by `E`; each split MAC costs a bit read across it. AM–GM makes the bound linear in `z`
  (`Verity.Combinatorics.sum_card_le`).
* **The cap at word level** (`WordCap`): when no MAC is split, `N ≤ m k^{3/2} / (4 √(b_A b_B b_C)) + μ* E` at
  `μ* = √k / (4 √(b_A b_B b_C))`, `k^{3/2} / (4 √(b_A b_B b_C))` products a block, amortized. A block that paid `b_C` for
  each entry it touches would do at most `(k/3)^{3/2} / √(b_A b_B b_C)`; only reads are charged, so one block can do the
  products and another read their partial sums, and the amortized cap is `3√3/4` times that, attained
  (`Verity.Combinatorics.split_attains`).
* **Slowdown** (`Slowdown`), the products kind's bound on `ρ`: `ρ ≥ ((N - μ E) / |C|) / (capAt / c)`, the counting
  lemma's form (`Nci.Formalism.kindBound`) with the cap amortized, `μ E` the products the cap leaves uncharged.

Each has an honest and a nontrivial lemma (`Definitions.Nci.Guarantees.Products.Witnesses`).
-/

namespace Nci.Products.Guarantees

open Verity (Declaration Partition)
open Nci.Formalism Nci.Products
open scoped ENNReal

/-- **The fresh-products cap**: a product by the standard algorithm with its widths as passed has, in every partition
into `m` inference blocks and for every weight `μ > 0`,
`ni nl nj ≤ m max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k) + μ Σ_ij |out_ij|`. -/
def Cap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m) (μ : ℝ),
    S.Algorithm → S.Passed c k bA bB bC → P.IsCK d c k → 0 < bA → 0 < bB → 0 < bC → 0 < μ →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * capAt k bA bB bC μ + μ * S.outBits

/-- **The cap at word level**: when every MAC lies in one block,
`ni nl nj ≤ m k √k / (4 √(b_A b_B b_C)) + μ* Σ_ij |out_ij|`, `μ* = √k / (4 √(b_A b_B b_C))`. -/
def WordCap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m),
    S.Algorithm → S.Passed c k bA bB bC → P.IsCK d c k → (∀ i l j, ∃ u, S.mac i l j ⊆ P.gates u) → 0 < k →
    0 < bA → 0 < bB → 0 < bC →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * ((k : ℝ) * √(k : ℝ) / (4 * √((bA : ℝ) * bB * bC))) +
      √(k : ℝ) / (4 * √((bA : ℝ) * bB * bC)) * S.outBits

/-- **Slowdown**, the products kind's bound: every partition into `m` inference blocks has
`m c / |C| ≥ ((ni nl nj - μ Σ_ij |out_ij|) / |C|) / (capAt / c)` for every weight `μ > 0`. -/
def Slowdown : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m) (μ : ℝ),
    S.Algorithm → S.Passed c k bA bB bC → P.IsCK d c k → 0 < bA → 0 < bB → 0 < bC → 0 < μ →
    ENNReal.ofReal ((ni * nl * nj : ℕ) - μ * S.outBits) / (d.D.computed.card : ℝ≥0∞) /
      (ENNReal.ofReal (capAt k bA bB bC μ) / (c : ℝ≥0∞)) ≤ partitionSlowdown d m c

end Nci.Products.Guarantees
