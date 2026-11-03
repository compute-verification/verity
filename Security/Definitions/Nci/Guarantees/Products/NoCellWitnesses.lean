import Definitions.Nci.Guarantees.Products.NoCell
import Definitions.Nci.Products.NoCellInstances

/-!
# Products of two fresh operands without the word level: honest and nontrivial lemmas

`NoCellCap` and `NoCellBound` each have two statements here, at `Pair`
(`Security/Definitions/Nci/Products/Instances.lean`), one entry's chain of two MACs, with `b_A = b_B = b_C = 1` and the
entry's partial sum starting at gate 4, its first MAC. `<Guarantee>_honest`: its premises hold together.
`<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails, its last conjunct the
bound's negation.

* The honest lemmas are at `fresh`, at `(c, k) = (1, 3)`, with `P` its two blocks, gate 4 and gate 5: gate 4's block
  reads `a₀` and `b₀` and holds the start, and gate 5's reads `a₁`, `b₁` and gate 4's partial sum. `P` is a partition
  into inference blocks there, so `NoCellBound`'s partition-level bound is over at least one partition.
* The nontrivial lemmas drop widths as passed (`Standard.Passed`), at `registered`, at `(c, k) = (2, 1)`, with `P1`
  (`Security/Definitions/Nci/Products/NoCellInstances.lean`) the one block holding both MACs: the partial sums are
  received, the chain starting in the block, but the operands are read free, none of their cones beyond `θ`.
  `NoCellCap`'s bound `2 ≤ (1/3) √(1/3) + 1 + 1/2` fails, and `NoCellBound` fails at `P1`, its bound
  `chainBlocks 2 ((1/3) √(1/3) + 1) (1/2) > 1` above `P1`'s price `1`.

`Pair` is a small faithful instance of a product of two fresh operands, accepted as the honest instance of these
structural bounds: it rests on no run and no `art:`, and a training run's circuit and its premises, which no check reads
yet, would supply the deployed ones.

The proofs are in `Security/Proofs/Nci/Products/NoCellWitnesses.lean`.
-/

namespace Nci.Products.Guarantees

open Nci.Formalism Nci.Products

/-- **The cap without the word level's premises hold** at `Pair` with its operands fresh, at `(c, k) = (1, 3)` and
one-bit widths: the product is by the standard algorithm, its widths are as passed, its partial sums are received with
the start at gate 4, and `P` is a partition into inference blocks. -/
def NoCellCap_honest : Prop :=
  Pair.freshProduct.Algorithm ∧ Pair.freshProduct.Passed 1 3 1 1 1 ∧ Pair.freshProduct.Received 1 3 1 ∧
    Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1

/-- **The cap without the word level is false without widths as passed**: at `Pair` with its operands registered, at
`(c, k) = (2, 1)` and one-bit widths, the product is by the standard algorithm, its partial sums are received, and `P1`
is a partition into inference blocks, but its block reads none of the operands' cones beyond `θ`, and the bound
`2 ≤ (1/3) √(1/3) + 1 + 1/2` fails. -/
def NoCellCap_nontrivial : Prop :=
  Pair.registeredProduct.Algorithm ∧ ¬ Pair.registeredProduct.Passed 2 1 1 1 1 ∧
    Pair.registeredProduct.Received 2 1 1 ∧ Pair.P1.IsCK Pair.registered 2 1 ∧ (0 : ℕ) < 1 ∧
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (1 : ℕ) * (((1 : ℕ) : ℝ) / 3 * √(((1 : ℕ) : ℝ) / 3) / √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (1 : ℕ)) +
        ((1 : ℕ) : ℝ) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) * √(((1 : ℕ) : ℝ) * (1 * 1 : ℕ))

/-- **The products kind's bound without the word level's premises hold** at `Pair` with its operands fresh, as in
`NoCellCap_honest`, and `P` is a partition into inference blocks at `(c, k) = (1, 3)`. -/
def NoCellBound_honest : Prop :=
  Pair.freshProduct.Algorithm ∧ Pair.freshProduct.Passed 1 3 1 1 1 ∧ Pair.freshProduct.Received 1 3 1 ∧
    Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1

/-- **The products kind's bound without the word level is false without widths as passed**: at `Pair` with its
operands registered, as in `NoCellCap_nontrivial`, the bound fails at `P1`, its value
`chainBlocks 2 ((1/3) √(1/3) + 1) (1/2) > 1` above `P1`'s price `1`. -/
def NoCellBound_nontrivial : Prop :=
  Pair.registeredProduct.Algorithm ∧ ¬ Pair.registeredProduct.Passed 2 1 1 1 1 ∧
    Pair.registeredProduct.Received 2 1 1 ∧ (0 : ℕ) < 1 ∧
    ¬ PartitionBound Pair.registered 2 1 (noCellBound Pair.registered 1 2 1 2 1 1 1 1)

end Nci.Products.Guarantees
