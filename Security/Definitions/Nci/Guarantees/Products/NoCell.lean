import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products.Received

/-!
# Products of two fresh operands without the word level: the cap along chains with split MACs, and its bound on `ρ`

The theorems about a product `C = A · B` by the standard algorithm (`Definitions.Nci.Products`) in a partition that may
split its MACs, proved in `Nci.Products.SecurityProofs`. `N = ni nl nj` is the number of products, `m` the number of
blocks of a partition into inference blocks, `α = (k/3)^{3/2} / √(b_A b_B b_C)` and `β = (k/2) / √(b_A b_B)`.

* **The cap without the word level** (`NoCellCap`): `N ≤ m (α + k) + β √(m ni nj)`, the cap along chains over the
  MACs the blocks hold whole and at most `k` split MACs a block. A block's whole MACs pay their operands
  (`Standard.Passed`), and each entry they touch either starts its chain in the block or has its partial sum read at
  `b_C` bits (`Standard.Received`), so they are at most `α + β √s` for a block holding `s` entries' starts; each split MAC
  costs a read of one of its gates across the split (`Standard.Algorithm`), so the blocks split at most `m k`. It reads
  no premise on the partition: a MAC is whatever unit `Standard.mac` names, a GEMM gadget among them, and the `k` a
  block is `k` of those units.
* **The products kind's bound without the word level** (`NoCellBound`): `ρ ≥ chainBlocks N (α + k) (β √(ni nj)) c / |C|`
  (`noCellBound`), the fewest blocks the cap allows priced at `c` a block, a partition-level bound
  (`Nci.Formalism.PartitionBound`), so a kind's `Λ_J` in the counting lemma and the one slowdown.

The premises, beside the training formalism's: `Standard.Passed` is the operand side of `P-width`, each operand a block
uses paid at its width, with the side that hands partial sums on, which the bit-level cap reads; `Standard.Received` is
`P-width`'s read side with `P-chain`, each entry's chain started once. Neither theorem reads `P-cell`
(`Standard.Unsplit`) or the chains' order (`Standard.Chained`).

Each has an honest and a nontrivial lemma (`Definitions.Nci.Guarantees.Products.NoCellWitnesses`).
-/

namespace Nci.Products.Guarantees

open Verity (Declaration Partition)
open Nci.Formalism Nci.Products

/-- **The cap without the word level**: a product by the standard algorithm with its widths as passed and its partial
sums received at their width has, in every partition into `m` inference blocks,
`ni nl nj ≤ m ((k/3) √(k/3) / √(b_A b_B b_C) + k) + (k/2) / √(b_A b_B) · √(m ni nj)`. -/
def NoCellCap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m),
    S.Algorithm → S.Passed c k bA bB bC → S.Received c k bC → P.IsCK d c k → 0 < bA → 0 < bB → 0 < bC →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * ((k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC) + k) +
      (k : ℝ) / 2 / √((bA : ℝ) * bB) * √((m : ℝ) * (ni * nj : ℕ))

/-- **The products kind's bound without the word level**: every partition `𝓘` into inference blocks has
`|𝓘| c / |C| ≥ chainBlocks (ni nl nj) (α + k) (β √(ni nj)) · c / |C|`, a partition-level bound and so a kind's `Λ_J` in
the one slowdown (`Nci.Formalism.Guarantees.Slowdown`). -/
def NoCellBound : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC : ℕ),
    S.Algorithm → S.Passed c k bA bB bC → S.Received c k bC → 0 < bA → 0 < bB → 0 < bC →
    PartitionBound d c k (noCellBound d ni nl nj c k bA bB bC)

end Nci.Products.Guarantees
