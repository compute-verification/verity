import Definitions.Nci.Guarantees.Products.Whole
import Definitions.Nci.Products.NoCellInstances
import Definitions.Nci.Products.WholeInstances

/-!
# Products made whole through chains: honest and nontrivial lemmas

`ClassBound`, `WholeCap` and `WholeBound` each have two statements here, at `Pair`'s product alone as a whole-making
system (`Pair.system`, `Security/Definitions/Nci/Products/WholeInstances.lean`): one class of chain length 2 and no value
made from an output entry, with `b = b_C = 1`, P-share at `c = 1/4`, fan-outs `A = 0`, so that the whole-making factor
is 0, and the entry's partial sum starting at gate 4. `<Guarantee>_honest`: its premises hold together.
`<Guarantee>_nontrivial`: every premise but one holds, that one fails, and the bound fails, its last conjunct the
bound's negation.

* The honest lemmas are at `freshSystem`, at `(c, k) = (1, 3)` and `μ = 3/16`, the least post-fixed point there: a
  block pays each value it has at its width. `ClassBound`'s block is gate 4, and `WholeCap`'s and `WholeBound`'s
  partition is `P`, gate 4 and gate 5.
* The nontrivial lemmas drop P-width extended (`System.WidthExtended`), at `registeredSystem`, at `(c, k) = (2, 1)` and
  `μ = 1/16`, the least post-fixed point there: the block holding both MACs, gates 4 and 5 (`P1`,
  `Security/Definitions/Nci/Products/NoCellInstances.lean`), has every value free, from `θ`. `ClassBound`'s bound
  `1 ≤ 1/16` fails there, `WholeCap`'s `2 ≤ (1/3) √(1/3) + 1 + 1/2` at `P1`, and `WholeBound` at `P1`, its bound
  `chainBlocks 2 ((1/3) √(1/3) + 1) (1/2) > 1` above `P1`'s price `1`.

`Pair` is a small faithful instance of a product of two fresh operands, accepted as the honest instance of these
structural bounds: it rests on no run and no `art:`, and a training run's circuit and its premises, which no check reads
yet, would supply the deployed ones.

The proofs are in `Security/Proofs/Nci/Products/WholeWitnesses.lean`.
-/

namespace Nci.Products.Guarantees

open Finset
open Nci.Formalism Nci.Products

/-- **The class bound's premises hold** at `Pair`'s product alone with its operands fresh, at `(c, k) = (1, 3)`,
`b = 1`, `c = 1/4`, `A = 0` and `μ = 3/16`: the product is by the standard algorithm, its four operand bits are its
values, P-share, P-width extended and P-match hold, `μ` is a post-fixed point, and gate 4 is an inference block. -/
def ClassBound_honest : Prop :=
  (∀ p, (Pair.freshSystem.prod p).Algorithm) ∧ Pair.freshSystem.Values ∧
    Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) ∧ Pair.freshSystem.WidthExtended 1 3 1 ∧
    Pair.freshSystem.Matched (fun _ _ => 0) ∧ (0 : ℕ) < 1 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧
    (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧ Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16) ∧
    InferenceBlock Pair.fresh 1 3 {4}

/-- **The class bound is false without P-width extended**: at `Pair`'s product alone with its operands registered, at
`(c, k) = (2, 1)`, `b = 1`, `c = 1/4`, `A = 0` and `μ = 1/16`, every premise but P-width extended holds, and the block
of gates 4 and 5, an inference block reading its values free from `θ`, holds the whole chain of the one output entry,
above the bound `1/16`. -/
def ClassBound_nontrivial : Prop :=
  (∀ p, (Pair.registeredSystem.prod p).Algorithm) ∧ Pair.registeredSystem.Values ∧
    Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) ∧ ¬ Pair.registeredSystem.WidthExtended 2 1 1 ∧
    Pair.registeredSystem.Matched (fun _ _ => 0) ∧ (0 : ℕ) < 1 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧
    (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) ∧
    InferenceBlock Pair.registered 2 1 {4, 5} ∧
    ¬ ((∑ p ∈ univ.filter (fun p => Pair.registeredSystem.cls p = ()),
        ((Pair.registeredSystem.prod p).wholeOut {4, 5}).card : ℕ) : ℝ) ≤ 1 / 16 * (1 : ℕ) / (1 : ℕ)

/-- **The cap with operands made whole's premises hold** at `Pair`'s product alone with its operands fresh, as in
`ClassBound_honest`, with `b_C = 1`: its partial sums are received with the start at gate 4, and `P` is a partition
into inference blocks. -/
def WholeCap_honest : Prop :=
  (∀ p, (Pair.freshSystem.prod p).Algorithm) ∧ Pair.freshSystem.Values ∧
    Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) ∧ Pair.freshSystem.WidthExtended 1 3 1 ∧
    Pair.freshSystem.Matched (fun _ _ => 0) ∧ (Pair.freshSystem.prod ()).Received 1 3 1 ∧
    Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16)

/-- **The cap with operands made whole is false without P-width extended**: at `Pair`'s product alone with its
operands registered, as in `ClassBound_nontrivial`, with `b_C = 1`, its partial sums are received and `P1` is a
partition into inference blocks, but the cap `2 ≤ (1/3) √(1/3) + 1 + 1/2` fails at `P1`, the whole-making factor 0. -/
def WholeCap_nontrivial : Prop :=
  (∀ p, (Pair.registeredSystem.prod p).Algorithm) ∧ Pair.registeredSystem.Values ∧
    Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) ∧ ¬ Pair.registeredSystem.WidthExtended 2 1 1 ∧
    Pair.registeredSystem.Matched (fun _ _ => 0) ∧ (Pair.registeredSystem.prod ()).Received 2 1 1 ∧
    Pair.P1.IsCK Pair.registered 2 1 ∧ (0 : ℕ) < 1 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) ∧
    ¬ ((1 * 2 * 1 : ℕ) : ℝ) ≤
      (1 : ℕ) * (((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 3 *
          √(((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 3) /
          √(((1 : ℕ) : ℝ) * (1 : ℕ) * (1 : ℕ)) + (1 : ℕ)) +
        ((1 : ℕ) : ℝ) * (1 + wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()) / 2 / √(((1 : ℕ) : ℝ) * (1 : ℕ)) *
          √(((1 : ℕ) : ℝ) * (1 * 1 : ℕ))

/-- **The products kind's bound with operands made whole's premises hold** at `Pair`'s product alone with its operands
fresh, as in `WholeCap_honest`, and `P` is a partition into inference blocks at `(c, k) = (1, 3)`. -/
def WholeBound_honest : Prop :=
  (∀ p, (Pair.freshSystem.prod p).Algorithm) ∧ Pair.freshSystem.Values ∧
    Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) ∧ Pair.freshSystem.WidthExtended 1 3 1 ∧
    Pair.freshSystem.Matched (fun _ _ => 0) ∧ (Pair.freshSystem.prod ()).Received 1 3 1 ∧
    Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16)

/-- **The products kind's bound with operands made whole is false without P-width extended**: at `Pair`'s product
alone with its operands registered, as in `WholeCap_nontrivial`, the bound over its one product fails at `P1`, its
value `chainBlocks 2 ((1/3) √(1/3) + 1) (1/2) > 1` above `P1`'s price `1`. -/
def WholeBound_nontrivial : Prop :=
  (∀ p, (Pair.registeredSystem.prod p).Algorithm) ∧ Pair.registeredSystem.Values ∧
    Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) ∧ ¬ Pair.registeredSystem.WidthExtended 2 1 1 ∧
    Pair.registeredSystem.Matched (fun _ _ => 0) ∧ (Pair.registeredSystem.prod ()).Received 2 1 1 ∧ (0 : ℕ) < 1 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) ∧
    ¬ PartitionBound Pair.registered 2 1
      ((univ : Finset Unit).sup fun _ => wholeBound Pair.registered 2 1 2 1 1 1
        (wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) ()))

end Nci.Products.Guarantees
