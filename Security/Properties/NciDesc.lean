import Properties.Nci
import NciCheck.Desc

/-!
# The description-sized NCI check's property, stated against the Lean functions

`NciCheck.Desc.evaluate` reads θ and the owners per root node and never lays the circuit out (`NciCheck.Desc`).

* `DescAgrees`: on every instance with a verdict, the instance expanded gate by gate (`Expands`: the same program, `Γ`
  and `α`, θ as its nodes' gates and each gate's owner its root node's) exists, and on every expansion
  `NciCheck.evaluate` gives the same verdict. So `Reports` and `Witnesses` (`Properties.Nci`) hold of
  `NciCheck.Desc.evaluate`'s verdict, on the expansion. The converse is not claimed: `NciCheck.Desc.evaluate` refuses
  some instances whose expansion `NciCheck.evaluate` accepts (its module docstring, "Refusals"), and its codes may differ.
  `SecurityProofs.DescAgrees` (`Proofs.Properties.Nci.Agrees`) proves it from the three statements
  `descAgrees_of_open` (`Proofs.Properties.Nci.Novel`) takes: `Resolve.bodiesOk` proves `Resolve.BodiesOk`,
  `operandsResolve_of` `Walk.OperandsResolve` from it, and `Novel.readsExact` `Novel.ReadsExact`.
  `Properties.NciDesc.Verdict` restates `Reports`, `Witnesses` and the thread bounds of `NciCheck.Desc.evaluate` from it,
  and `Properties.NciDesc.Cover` states the cover theorem's factor of the check of record's verdict.
-/

namespace NciCheck.Properties

open Lean (Json toJson)
open Flock.Program (Prog)

/-- The gates of root node `k`: `[offsets[k], offsets[k + 1])` of the root body's. -/
def nodeGates (P : Prog) (k : ℕ) : List ℕ :=
  let offs := (P.defs[P.root]!).body.offsets
  List.range' offs[k]! (offs[k + 1]! - offs[k]!)

/-- `theta'` is θ by root node, `theta`, gate by gate: each entry's node's gates in order, in `theta`'s order. -/
def ExpandsTheta (P : Prog) (theta theta' : Array Json) : Prop :=
  theta'.toList.map natOf? = (theta.toList.flatMap fun t => ((natOf? t).map (nodeGates P)).getD []).map some

/-- `owner'` is the owners by root node, `owner`, gate by gate: one per gate, each gate's its root node's. -/
def ExpandsOwner (P : Prog) (owner owner' : Array Json) : Prop :=
  owner'.size = gates P ∧ ∀ k < owner.size, ∀ g ∈ nodeGates P k,
    (owner'[g]! = .str "input" ↔ owner[k]! = .str "input") ∧ natOf? owner'[g]! = natOf? owner[k]!

/-- `j'` is the instance `j` of `NciCheck.Desc.evaluate` expanded: the same program, `Γ` and `α`, with θ and the owners
by root node expanded gate by gate. -/
def Expands (j j' : Json) : Prop :=
  ∃ i P, Parses j i ∧ Flock.Program.decode i.program.toUTF8 = .ok P ∧ ∃ theta' owner',
    Parses j' { i with theta := theta', owner := owner' } ∧ ExpandsTheta P i.theta theta' ∧
      ExpandsOwner P i.owner owner'

/-- The verdict `out` on `j` is `evaluate`'s on `j` expanded: `j` has an expansion, and `evaluate` gives `out` on each. -/
def DescAgreed (j out : Json) : Prop :=
  (∃ j', Expands j j') ∧ ∀ j', Expands j j' → evaluate j' = .ok out

/-- **The check at the size of the description agrees with the check gate by gate.** On every instance with a verdict of
`NciCheck.Desc.evaluate`, the verdict is `DescAgreed`. -/
def DescAgrees : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → DescAgreed j out

end NciCheck.Properties
