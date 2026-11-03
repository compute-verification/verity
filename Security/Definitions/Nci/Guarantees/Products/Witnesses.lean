import Definitions.Nci.Guarantees.Products
import Definitions.Nci.Products.Instances

/-!
# Products of two fresh operands: honest and nontrivial lemmas

`Cap`, `WordCap` and `Slowdown` each have two statements here, at `Single` (`Security/Definitions/Nci/Products/Instances.lean`),
one product of one-bit operands with `b_A = b_B = b_C = 1` and the partition `P` of one block, gate 2.
`<Guarantee>_honest`: its premises hold together. `<Guarantee>_nontrivial`: every premise but one holds, that one fails,
and the bound fails, its last conjunct the bound's negation.

* The honest lemmas are at `fresh`, at `(c, k) = (1, 2)`: gate 2's block reads both operands' bits beyond `θ`, one bit
  of each cone, and holds `C`'s final bit.
* The nontrivial lemmas drop widths as passed (`Standard.Passed`), at `registered`: the block reads both operands free,
  none of either cone beyond `θ`, so it does its product within `k = 0` (or `k = 1`, for `WordCap`, whose cap needs
  `k > 0`). At `μ = 1/2`, `Cap`'s bound is `1 ≤ 0 + 1/2` and `WordCap`'s `1 ≤ 1/4 + 1/4`, and `Slowdown`'s left side is
  `(1/2) / 0 = ⊤`, above `P`'s price `1`.

`Single` is a small faithful instance of a product of two fresh operands, accepted as the honest instance of these
structural bounds: it rests on no run and no `art:`, and a training run's circuit and its premises, which no check reads
yet, would supply the deployed ones.

The proofs are in `Security/Proofs/Nci/Products/Witnesses.lean`.
-/

namespace Nci.Products.Guarantees

open Nci.Formalism Nci.Products Nci.Products.Single
open scoped ENNReal

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
  freshProduct.Algorithm ∧ freshProduct.Passed 1 2 1 1 1 ∧ P.IsCK fresh 1 2 ∧
    (∀ i l j, ∃ u, freshProduct.mac i l j ⊆ P.gates u) ∧ (0 : ℕ) < 2 ∧ (0 : ℕ) < 1

/-- **The word-level cap is false without widths as passed**: at `Single` with its operands registered, at
`(c, k) = (1, 1)` and the rest as in `WordCap_honest`, the block reads none of either cone beyond `θ`, and the bound
`1 ≤ 1/4 + 1/4` fails. -/
def WordCap_nontrivial : Prop :=
  registeredProduct.Algorithm ∧ ¬ registeredProduct.Passed 1 1 1 1 1 ∧ P.IsCK registered 1 1 ∧
    (∀ i l j, ∃ u, registeredProduct.mac i l j ⊆ P.gates u) ∧ (0 : ℕ) < 1 ∧
    ¬ ((1 * 1 * 1 : ℕ) : ℝ) ≤ (1 : ℕ) * ((1 : ℕ) * √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)))) +
      √((1 : ℕ) : ℝ) / (4 * √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ))) * registeredProduct.outBits

/-- **Slowdown's premises hold** at `Single` with its operands fresh, as in `Cap_honest`. -/
def Slowdown_honest : Prop :=
  freshProduct.Algorithm ∧ freshProduct.Passed 1 2 1 1 1 ∧ P.IsCK fresh 1 2 ∧ (0 : ℕ) < 1 ∧ (0 : ℝ) < 1 / 2

/-- **Slowdown is false without widths as passed**: at `Single` with its operands registered, as in `Cap_nontrivial`,
the bound's left side `(1/2) / 0 = ⊤` is above `P`'s price `1`. -/
def Slowdown_nontrivial : Prop :=
  registeredProduct.Algorithm ∧ ¬ registeredProduct.Passed 1 0 1 1 1 ∧ P.IsCK registered 1 0 ∧ (0 : ℕ) < 1 ∧
    (0 : ℝ) < 1 / 2 ∧
    ¬ ENNReal.ofReal ((1 * 1 * 1 : ℕ) - (1 / 2 : ℝ) * registeredProduct.outBits) / (registered.D.computed.card : ℝ≥0∞) /
      (ENNReal.ofReal (capAt 0 1 1 1 (1 / 2)) / ((1 : ℕ) : ℝ≥0∞)) ≤ partitionSlowdown registered 1 1

end Nci.Products.Guarantees
