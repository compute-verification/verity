import Definitions.Nci.Guarantees.Products
import Definitions.Nci.Products.Instances

/-!
# Products of two fresh operands: honest and nontrivial lemmas

`Cap`, `WordCap` and `Bound` each have two statements here, at `Single`
(`Security/Definitions/Nci/Products/Instances.lean`), one product of one-bit operands with `b_A = b_B = b_C = 1` and the
partition `P` of one block, gate 2; `ChainCap` has its two at `Pair`, one entry's chain of two MACs, with `P` its two
blocks, so that the second block reads a partial sum. `<Guarantee>_honest`: its premises hold together.
`<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails, its last conjunct the
bound's negation.

* The honest lemmas are at `fresh`, at `(c, k) = (1, 2)`: gate 2's block reads both operands' bits beyond `θ`, one bit
  of each cone, and holds `C`'s final bit. `P` is a partition into inference blocks there, so `Bound`'s partition-level
  bound is over at least one partition.
* The nontrivial lemmas drop widths as passed (`Standard.Passed`), at `registered`: the block reads both operands free,
  none of either cone beyond `θ`, so it does its product within `k = 0` (or `k = 1`, for `WordCap`, whose cap needs
  `k > 0`). At `μ = 1/2`, `Cap`'s bound is `1 ≤ 0 + 1/2` and `WordCap`'s `1 ≤ 1/4 + 1/4`, and `Bound` fails at `P`, the
  products kind's bound `(1/2) / 0 = ⊤` above `P`'s price `1`.
* `ChainCap`'s honest lemma is at `Pair`'s `fresh`, at `(c, k) = (1, 3)`: gate 4's block reads `a₀` and `b₀`, and gate
  5's reads `a₁`, `b₁` and gate 4's partial sum. Its nontrivial lemma drops widths as passed at `registered`, at
  `(c, k) = (1, 1)`: gate 5's block reads only the partial sum, and the bound `2 ≤ (2/3) √(1/3) + √2 / 2` fails.

`Single` and `Pair` are small faithful instances of a product of two fresh operands, accepted as the honest instances of
these structural bounds: they rest on no run and no `art:`, and a training run's circuit and its premises, which no
check reads yet, would supply the deployed ones.

The proofs are in `Security/Proofs/Nci/Products/Witnesses.lean`.
-/

namespace Nci.Products.Guarantees

open Nci.Formalism Nci.Products Nci.Products.Single

/-- **The cap's premises hold** at `Single` with its operands fresh, at `(c, k) = (1, 2)`, one-bit widths and
`μ = 1/2`: the product is by the standard algorithm, its widths are as passed and `P` is a partition into inference
blocks. -/
def Cap_honest : Prop :=
  freshProduct.Algorithm ∧ freshProduct.Passed 1 2 1 1 1 ∧ P.IsCK fresh 1 2 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 / 2

/-- **The cap is false without widths as passed**: at `Single` with its operands registered, at `(c, k) = (1, 0)` and
the rest as in `Cap_honest`, the product is by the standard algorithm and `P` is a partition into inference blocks, but
the block reads none of either cone beyond `θ`, and the bound `1 ≤ 0 + 1/2` fails. -/
def Cap_nontrivial : Prop :=
  registeredProduct.Algorithm ∧ ¬ registeredProduct.Passed 1 0 1 1 1 ∧ P.IsCK registered 1 0 ∧ (0 : ℕ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧
    ¬ ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * capAt 0 1 1 1 (1 / 2) + 1 / 2 * registeredProduct.outBits

/-- **The word-level cap's premises hold** at `Single` with its operands fresh, at `(c, k) = (1, 2)` and one-bit
widths, its one MAC in `P`'s one block. -/
def WordCap_honest : Prop :=
  freshProduct.Algorithm ∧ freshProduct.Passed 1 2 1 1 1 ∧ P.IsCK fresh 1 2 ∧ freshProduct.Unsplit P ∧ (0 : ℕ) < 2 ∧
    (0 : ℕ) < 1

/-- **The word-level cap is false without widths as passed**: at `Single` with its operands registered, at
`(c, k) = (1, 1)` and the rest as in `WordCap_honest`, the block reads none of either cone beyond `θ`, and the bound
`1 ≤ 1/4 + 1/4` fails. -/
def WordCap_nontrivial : Prop :=
  registeredProduct.Algorithm ∧ ¬ registeredProduct.Passed 1 1 1 1 1 ∧ P.IsCK registered 1 1 ∧
    registeredProduct.Unsplit P ∧ (0 : ℕ) < 1 ∧
    ¬ ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * ((1 : ℕ) * √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)))) +
      √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) * registeredProduct.outBits

/-- **The cap along chains' premises hold** at `Pair` with its operands fresh, at `(c, k) = (1, 3)` and one-bit
widths: the product is by the standard algorithm, its widths are as passed, its partial sum runs from gate 4 to gate 5,
and `P` is a partition into inference blocks that splits no MAC, its second block reading the first's partial sum. -/
def ChainCap_honest : Prop :=
  Pair.freshProduct.Algorithm ∧ Pair.freshProduct.Passed 1 3 1 1 1 ∧ Pair.freshProduct.Chained 1 ∧
    Pair.P.IsCK Pair.fresh 1 3 ∧ Pair.freshProduct.Unsplit Pair.P ∧ (0 : ℕ) < 1

/-- **The cap along chains is false without widths as passed**: at `Pair` with its operands registered, at
`(c, k) = (1, 1)` and the rest as in `ChainCap_honest`, the blocks read none of the operands' cones beyond `θ`, and the
bound `2 ≤ 2 (1/3) √(1/3) + (1/2) √2` fails. -/
def ChainCap_nontrivial : Prop :=
  Pair.registeredProduct.Algorithm ∧ ¬ Pair.registeredProduct.Passed 1 1 1 1 1 ∧ Pair.registeredProduct.Chained 1 ∧
    Pair.P.IsCK Pair.registered 1 1 ∧ Pair.registeredProduct.Unsplit Pair.P ∧ (0 : ℕ) < 1 ∧
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (2 : ℕ) * (((1 : ℕ) : ℝ) / 3 * √(((1 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) +
        ((1 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((2 : ℕ) : ℝ) * (1 * 1 : ℕ))

/-- **The products kind's bound's premises hold** at `Single` with its operands fresh, as in `Cap_honest`, and `P` is a
partition into inference blocks at `(c, k) = (1, 2)`. -/
def Bound_honest : Prop :=
  freshProduct.Algorithm ∧ freshProduct.Passed 1 2 1 1 1 ∧ P.IsCK fresh 1 2 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 / 2

/-- **The products kind's bound is false without widths as passed**: at `Single` with its operands registered, as in
`Cap_nontrivial`, the bound fails at `P`, its value `(1/2) / 0 = ⊤` above `P`'s price `1`. -/
def Bound_nontrivial : Prop :=
  registeredProduct.Algorithm ∧ ¬ registeredProduct.Passed 1 0 1 1 1 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 / 2 ∧
    ¬ PartitionBound registered 1 0 (registeredProduct.bound 1 0 1 1 1 (1 / 2))

end Nci.Products.Guarantees
