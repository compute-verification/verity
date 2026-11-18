module

public import Properties.InferenceOnly.Record

/-!
# Every unit's rows are its own input gates or an earlier root node's outputs

`OwnRows` asks every leaf of a reader row to be an input gate of the row's unit. A root whose node reads an earlier
non-input root node's output rows (an edge) has reader rows whose leaves are that node's computed gates.
`OwnOrEdgeRows` widens `OwnRows` by those rows: the edges are a wiring's summary `E` beside the record's, whose
`writerOf` names each edge's writer, and a leaf of a row `E` has a writer for is the gate at the same leaf of that
writer's output row (`WriterLeaf`), a unit of a template whose range comes before the reader's. Where `E` has no wire
it is `OwnRows`.
-/

@[expose] public section

namespace InferenceOnlyVerifier.Properties.Record

open Flock.Program
open Flock.TemplateQuery (TRange)

/-- **Every unit's rows are its own input gates, or an earlier root node's outputs through an edge of `E`**: a leaf
of a reader row with no edge is an input gate, and a leaf of a row with an edge is the gate at the same leaf and bits
of the writer's output row, the writer's template's range before the reader's; no gate is a leaf of two rows with no
edge, or two leaves of one. -/
def OwnOrEdgeRows (P : Prog) (rs : Array TRange) (W : Spot.Wires.Wiring) (E : Spot.Wires.Summary) : Prop :=
  (∀ r l g lo hi, RowLeaf P rs W r l g lo hi →
      (E.writerOf r.tid r.unit r.port r.row = none ∧
        ∃ p ops sc, P.gate g = .ok (p, ops, sc) ∧ Flock.Program.isInputId (P.defs[p]!).key = true) ∨
      (WriterLeaf P rs E r l g lo hi ∧
        ∃ wt wu out i i', E.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) ∧ i' < i ∧ i < rs.size ∧
          rs[i]!.template = r.tid ∧ rs[i']!.template = wt)) ∧
    ∀ r l lo hi r' l' lo' hi' g, E.writerOf r.tid r.unit r.port r.row = none →
      E.writerOf r'.tid r'.unit r'.port r'.row = none →
      RowLeaf P rs W r l g lo hi → RowLeaf P rs W r' l' g lo' hi' → r = r' ∧ l = l'

end InferenceOnlyVerifier.Properties.Record
