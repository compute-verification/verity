import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products

/-!
# Products of two fresh operands: the amortized cap and the bound on `ρ` it gives

The theorems about a product `C = A · B` by the standard algorithm with its widths as passed
(`Definitions.Nci.Products`), proved in `Nci.Products.SecurityProofs`. Each is about every partition `𝓘` of the
declaration into `m` inference blocks, each block reading at most `k` bits beyond `θ`, and charges only those reads.
`N = ni nl nj` is the number of products and `E = Σ_ij |out_ij|` the bits of `C`'s final values.

* **The fresh-products cap** (`Cap`), at the bit level: a partition may split a MAC's gates across blocks and pass any
  of their bits. For every weight `μ > 0`, `N ≤ m max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k) + μ E`. A block's whole MACs
  are at most `√(x y z)` by Loomis–Whitney, `x`, `y` and `z` the entries of `A`, `B` and `C` they touch; the block reads
  at least `b_A x + b_B y` bits for their operands, and `b_C z` is paid across the partition, by the partial sums the
  blocks hand each other and by `E`; each split MAC costs a bit read across it. AM–GM makes the bound linear in `z`
  (`Verity.Combinatorics.sum_card_le`).
* **The cap at word level** (`WordCap`): when no MAC is split, `N ≤ m k^{3/2} / (4 √(b_A b_B b_C)) + μ* E` at
  `μ* = √k / (4 √(b_A b_B b_C))`, `k^{3/2} / (4 √(b_A b_B b_C))` products a block, amortized. A block that paid `b_C`
  for each entry it touches would do at most `(k/3)^{3/2} / √(b_A b_B b_C)`; only reads are charged, so one block can do
  the products and another read their partial sums, and the amortized cap is `3√3/4` times that, attained
  (`Verity.Combinatorics.split_attains`).
* **The cap along chains** (`ChainCap`): when no MAC is split and each entry's partial sum runs along its MACs in
  order, `b_C` bits from each to the next (`Standard.Chained`), `N ≤ m α + β √(m O)`, with
  `α = (k/3)^{3/2} / √(b_A b_B b_C)`, `β = (k/2) / √(b_A b_B)` and `O = ni nj` the entries of `C`. Each entry a block
  touches either starts its chain there or has its first MAC there after a MAC in another block, whose partial sum the
  block reads, so a block that starts `s` chains does at most `α + β √s` (`Verity.Combinatorics.sqrt_mul_le_chain`),
  and each chain starts in one block. It has no weight `μ` and no `E`: per block, `α` is `4 / (3√3) ≈ 0.77` of
  `WordCap`'s `k^{3/2} / (4 √(b_A b_B b_C))`, and the price is `β √(m O)`, which `α m` exceeds once `m ≥ β² O / α²`.
* **The products kind's bound** (`Bound`): `ρ ≥ ((N - μ E) / |C|) / (capAt / c)` for every `μ > 0` (`Standard.bound`),
  `μ E` the products the cap leaves uncharged. It is a partition-level bound (`Nci.Formalism.PartitionBound`), the form
  in which the counting lemma and the one slowdown (`Nci.Formalism.Guarantees.Counting`, `Slowdown` and `Security`) take
  a kind amortized over the partition: the products kind is one of their kinds `J`, its bound their `Λ_J`.

Each has an honest and a nontrivial lemma (`Definitions.Nci.Guarantees.Products.Witnesses`).
-/

namespace Nci.Products.Guarantees

open Verity (Declaration Partition)
open Nci.Formalism Nci.Products

/-- **The fresh-products cap**: a product by the standard algorithm with its widths as passed has, in every partition
into `m` inference blocks and for every weight `μ > 0`,
`ni nl nj ≤ m max (k² / (16 μ b_A b_B b_C)) ((μ + 1) k) + μ Σ_ij |out_ij|`. -/
def Cap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m) (μ : ℝ),
    S.Algorithm → S.Passed c k bA bB bC → P.IsCK d c k → 0 < bA → 0 < bB → 0 < bC → 0 < μ →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * capAt k bA bB bC μ + μ * S.outBits

/-- **The cap at word level**: when every MAC lies in one block (`Standard.Unsplit`),
`ni nl nj ≤ m k √k / (4 √(b_A b_B b_C)) + μ* Σ_ij |out_ij|`, `μ* = √k / (4 √(b_A b_B b_C))`. -/
def WordCap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m),
    S.Algorithm → S.Passed c k bA bB bC → P.IsCK d c k → S.Unsplit P → 0 < k → 0 < bA → 0 < bB → 0 < bC →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * ((k : ℝ) * √(k : ℝ) / (4 * √((bA : ℝ) * bB * bC))) +
      √(k : ℝ) / (4 * √((bA : ℝ) * bB * bC)) * S.outBits

/-- **The cap along chains**: when every MAC lies in one block (`Standard.Unsplit`) and each entry's partial sum runs
along its MACs (`Standard.Chained`), `ni nl nj ≤ m (k/3) √(k/3) / √(b_A b_B b_C) + (k/2) / √(b_A b_B) · √(m ni nj)`. -/
def ChainCap : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC m : ℕ) (P : Partition d.D m),
    S.Algorithm → S.Passed c k bA bB bC → S.Chained bC → P.IsCK d c k → S.Unsplit P → 0 < bA → 0 < bB → 0 < bC →
    ((ni * nl * nj : ℕ) : ℝ) ≤ m * ((k : ℝ) / 3 * √((k : ℝ) / 3) / √((bA : ℝ) * bB * bC)) +
      (k : ℝ) / 2 / √((bA : ℝ) * bB) * √((m : ℝ) * (ni * nj : ℕ))

/-- **The products kind's bound**: for every weight `μ > 0`, every partition `𝓘` into inference blocks has
`|𝓘| c / |C| ≥ ((ni nl nj - μ Σ_ij |out_ij|) / |C|) / (capAt / c)`, a partition-level bound and so a kind's `Λ_J` in
the one slowdown (`Nci.Formalism.Guarantees.Slowdown`). -/
def Bound : Prop :=
  ∀ (d : Declaration) (ni nl nj : ℕ) (S : Standard d ni nl nj) (c k bA bB bC : ℕ) (μ : ℝ),
    S.Algorithm → S.Passed c k bA bB bC → 0 < bA → 0 < bB → 0 < bC → 0 < μ →
    PartitionBound d c k (S.bound c k bA bB bC μ)

end Nci.Products.Guarantees
