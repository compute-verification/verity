import Definitions.Nci.Guarantees.Products.Sum
import Definitions.Nci.Products.NoCellInstances
import Definitions.Nci.Products.WholeInstances

/-!
# Products made whole through chains, several in a block: honest and nontrivial lemmas

`SumCap` and `SumBound` each have two statements here, at `Pair`'s product alone as a whole-making system (`Pair.system`,
`Security/Definitions/Nci/Products/WholeInstances.lean`), as the whole-making lemmas have it
(`Definitions.Nci.Guarantees.Products.WholeWitnesses`): one class of chain length 2, P-share at `c = 1/4`, fan-outs
`A = 0`, so that the whole-making factor is 0, each MAC its own cell (`Pair.cell`, `w = 1`) and `b = b_C = 1`; with `G`
the one product, P-share on `G` at `c_G = 1/4` and `G` apart, and the bound at `t = 1`.
`<Guarantee>_honest`: its premises hold together. `<Guarantee>_nontrivial`: every premise but one holds, that one
fails, and the bound fails, its last conjunct the bound's negation.

* The honest lemmas are at `freshSystem`, at `(c, k) = (1, 3)` and `μ = 3/16`, the least post-fixed point there, with
  the partition `P`, gate 4 and gate 5.
* The nontrivial lemmas drop P-width extended (`System.WidthExtended`), at `registeredSystem`, at `(c, k) = (2, 1)` and
  `μ = 1/16`, the least post-fixed point there: the block holding both MACs, gates 4 and 5 (`P1`,
  `Security/Definitions/Nci/Products/NoCellInstances.lean`), has every value free, from `θ`. `SumCap`'s bound
  `2 ≤ 1 + √2 / 2` fails at `P1`, and `SumBound` at `P1`, its bound `7/6` above `P1`'s price `1`.

`Pair` is a small faithful instance of a product of two fresh operands, accepted as the honest instance of these
structural bounds: it rests on no run and no `art:`, and a training run's circuit and its premises, which no check reads
yet, would supply the deployed ones.

The proofs are in `Security/Proofs/Nci/Products/SumWitnesses.lean`.
-/

namespace Nci.Products.Guarantees

open Finset
open Nci.Formalism Nci.Products

/-- **The cap summed over a set's premises hold** at `Pair`'s product alone with its operands fresh, at
`(c, k) = (1, 3)`, `b = b_C = 1`, `c = 1/4`, `A = 0` and `μ = 3/16`, with `G` its one product: the class bound's
premises, P-share on `G` at `1/4`, `G` apart, each MAC its own cell, its partial sums received in those cells with the
start at gate 4, and `P` a partition into inference blocks. -/
def SumCap_honest : Prop :=
  (∀ p, (Pair.freshSystem.prod p).Wiring) ∧ Pair.freshSystem.Values ∧
    Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) ∧ Pair.freshSystem.WidthExtended 1 3 1 ∧
    Pair.freshSystem.Matched (fun _ _ => 0) ∧ Pair.freshSystem.SharedOn 1 3 (1 / 4) univ ∧
    Pair.freshSystem.Apart univ ∧ (Pair.freshSystem.prod ()).Cells Pair.cell 1 ∧
    (Pair.freshSystem.prod ()).Received Pair.cell 1 3 1 ∧ Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16)

/-- **The cap summed over a set is false without P-width extended**: at `Pair`'s product alone with its operands
registered, at `(c, k) = (2, 1)`, `b = b_C = 1`, `c = 1/4`, `A = 0` and `μ = 1/16`, with `G` its one product, every
premise but P-width extended holds and `P1` is a partition into inference blocks, but the cap `2 ≤ 1 + √2 / 2` fails
at `P1`, the whole-making factor 0. -/
def SumCap_nontrivial : Prop :=
  (∀ p, (Pair.registeredSystem.prod p).Wiring) ∧ Pair.registeredSystem.Values ∧
    Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) ∧ ¬ Pair.registeredSystem.WidthExtended 2 1 1 ∧
    Pair.registeredSystem.Matched (fun _ _ => 0) ∧ Pair.registeredSystem.SharedOn 2 1 (1 / 4) univ ∧
    Pair.registeredSystem.Apart univ ∧ (Pair.registeredSystem.prod ()).Cells Pair.cell 1 ∧
    (Pair.registeredSystem.prod ()).Received Pair.cell 2 1 1 ∧ Pair.P1.IsCK Pair.registered 2 1 ∧ (0 : ℕ) < 1 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) ∧
    ¬ ((∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
          Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q : ℕ) : ℝ) ≤
      (1 : ℕ) * ((1 : ℕ) * (1 : ℕ)) + √(1 / 4) * (((1 : ℕ) : ℝ) * (1 + ∑ j ∈ (univ : Finset Unit).image
          Pair.registeredSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) j) / (1 : ℕ)) *
        √(((1 : ℕ) : ℝ) * ((1 : ℕ) * (((1 : ℕ) : ℝ) / (1 : ℕ)) +
          (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q : ℕ)))

/-- **The bound summed over a set's premises hold** at `Pair`'s product alone with its operands fresh, as in
`SumCap_honest`, `P` is a partition into inference blocks at `(c, k) = (1, 3)`, and `t = 1` is positive. -/
def SumBound_honest : Prop :=
  (∀ p, (Pair.freshSystem.prod p).Wiring) ∧ Pair.freshSystem.Values ∧
    Pair.freshSystem.Shared 1 3 (fun _ => 1 / 4) ∧ Pair.freshSystem.WidthExtended 1 3 1 ∧
    Pair.freshSystem.Matched (fun _ _ => 0) ∧ Pair.freshSystem.SharedOn 1 3 (1 / 4) univ ∧
    Pair.freshSystem.Apart univ ∧ (Pair.freshSystem.prod ()).Cells Pair.cell 1 ∧
    (Pair.freshSystem.prod ()).Received Pair.cell 1 3 1 ∧ Pair.P.IsCK Pair.fresh 1 3 ∧ (0 : ℕ) < 1 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.freshSystem.PostFixed 3 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 3 / 16) ∧ (0 : ℝ) < 1

/-- **The bound summed over a set is false without P-width extended**: at `Pair`'s product alone with its operands
registered, as in `SumCap_nontrivial`, the bound at `t = 1` fails at `P1`, its value
`(2 - 1/4) / (1 + 1/2) = 7/6` above `P1`'s price `1`. -/
def SumBound_nontrivial : Prop :=
  (∀ p, (Pair.registeredSystem.prod p).Wiring) ∧ Pair.registeredSystem.Values ∧
    Pair.registeredSystem.Shared 2 1 (fun _ => 1 / 4) ∧ ¬ Pair.registeredSystem.WidthExtended 2 1 1 ∧
    Pair.registeredSystem.Matched (fun _ _ => 0) ∧ Pair.registeredSystem.SharedOn 2 1 (1 / 4) univ ∧
    Pair.registeredSystem.Apart univ ∧ (Pair.registeredSystem.prod ()).Cells Pair.cell 1 ∧
    (Pair.registeredSystem.prod ()).Received Pair.cell 2 1 1 ∧ (0 : ℕ) < 1 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧ (∀ _ _ : Unit, (0 : ℝ) ≤ 0) ∧
    Pair.registeredSystem.PostFixed 1 1 (fun _ => 1 / 4) (fun _ _ => 0) (fun _ => 1 / 16) ∧ (0 : ℝ) < 1 ∧
    ¬ PartitionBound Pair.registered 2 1
      (sumBound Pair.registered
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q *
          Pair.registeredSystem.K (Pair.registeredSystem.cls q) * Pair.registeredSystem.nj q)
        (∑ q ∈ (univ : Finset Unit), Pair.registeredSystem.ni q * Pair.registeredSystem.nj q) 1 2 1 1 1 (1 / 4)
        (∑ j ∈ (univ : Finset Unit).image Pair.registeredSystem.cls, wholeFactor (fun _ _ => 0) (fun _ => 1 / 16) j)
        1)

end Nci.Products.Guarantees
