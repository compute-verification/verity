import Flock.Annot
import Flock.AnnotExpand

/-!
# The description-level annotation check's property, stated against the Lean functions

`Flock.Annot.check Φ P D` decides a meaning `Φ` (`Flock.Annot.Meaning`: a tiling, `max_count`, `max_gates`) of the
annotation `D` (a query term with its tables) of the program `P`, on the term's per-body plans, and never lays the circuit
out. `Flock.Annot.expand P D` is the annotation laid out, each gate of `P`'s canonical layout with its labels
(`Flock.AnnotExpand`).

* `Holds Φ P n e`: `Φ` of the labelling `e` with `n` labels, gate by gate. A partition puts every computed gate in exactly
  one label and every input gate in none (`Verity.Annotation`'s `IsPartition`); a cover puts every computed gate in at least
  one; `max_count` bounds `n`, `max_gates` each label's gates. A gate is an input gate when `Prog.gate` locates it at a
  primitive with an input id (`Flock.Annot.isInputGate`).
* `CheckAgrees Φ`: on every program and annotation the check accepts, the annotation expands, and `Φ` holds of the
  expansion with the check's label count. The converse is not claimed: the check refuses some annotations whose expansion
  satisfies `Φ` (a partition is tested on the family's sets, so a gate in two sets of one label is refused; under `group`
  and a label table, `max_gates` bounds a label by the sum of its sets' sizes; `max_gates` under an affine label map is
  refused).

Nothing proves `CheckAgrees` yet. Proving it needs, on the code side, the layout lemma that `Flock.Annot.walk` lays a plan's
entries out where the descriptor's layout puts them (member `m` of node `j` of an activation at `off` is
`[off + offsets[j] + m·G, off + offsets[j] + (m + 1)·G)`, as `Prog.gate` locates gates), the invariant that every memo
entry (`St.memo`, `tiles`' state) equals a fresh evaluation, the segment argument of `tiles` (the members of a node
between two consecutive cuts are held by the same entries, and every activation of a Definition is laid out alike) and
the per-Definition flags `gateKinds` and `planHolds`. Reading `Holds` as `Verity.Annotation`'s meanings of a
`Verity.Circuit` needs `Flock.Desc`'s `program` constructor, whose `unroll` is the descriptor's lowering with its input
gates the ones `Prog.gate` classifies so, proved once: the one canonical format the Lean rules ask circuit inputs and
outputs to use. Until both land, `Flock.Annot.check` is code no soundness theorem reaches, and no verifier calls it.
-/

namespace Properties.Annot

open Flock.Annot
open Flock.Program (Prog)

/-- The gates label `l` holds in the labelling `e`. -/
def holds (e : Array (Array Nat)) (l : Nat) : List Nat := (List.range e.size).filter fun g => e[g]!.contains l

/-- Meaning `Φ` of the labelling `e` (each gate, its labels) of program `P`'s circuit with `n` labels, gate by gate. -/
def Holds (Φ : Meaning) (P : Prog) (n : Nat) (e : Array (Array Nat)) : Prop :=
  e.size = (P.defs[P.root]!).gates ∧
  (∀ l, ∀ g ∈ holds e l, l < n) ∧
  (Φ.tiling = some .partition → ∀ g < e.size,
    (isInputGate P g = .ok true → e[g]! = #[]) ∧ (isInputGate P g = .ok false → e[g]!.size = 1)) ∧
  (Φ.tiling = some .cover → ∀ g < e.size, isInputGate P g = .ok false → 0 < e[g]!.size) ∧
  (∀ m, Φ.maxCount = some m → n ≤ m) ∧
  (∀ c, Φ.maxGates = some c → ∀ l < n, (holds e l).length ≤ c)

/-- **The check at the size of the description agrees with the meaning gate by gate.** Whenever `Flock.Annot.check Φ`
accepts annotation `D` of program `P`, `D` expands, and `Φ` holds of the expansion with the verdict's label count. -/
def CheckAgrees (Φ : Meaning) : Prop :=
  ∀ (P : Prog) (D : Desc) (v : Verdict), check Φ P D = .ok v → v.accept = true →
    ∃ e, expand P D = .ok e ∧ Holds Φ P v.count e

end Properties.Annot
