import Difftest
import Flock.Annot
import Flock.AnnotExpand

/-!
# The annotation check's port: `Flock.Annot` behind `Difftest.serve`

`annot-difftest` serves the description-level annotation check and its per-gate reference to the difftest harness, so the
Python mirror (`verity.ml.circuits.query_ast` and its codec) is held to them (`verity_difftest.annot`). `program` is the
descriptor's text, `annotation` is `{"query", "tables"}` (`Annot.Desc.ofJson`) and `meaning` is
`{"tiling", "max_gates", "max_count"}` (`Annot.Meaning.ofJson`):

* `annot.term`: a query document read and written back (`Annot.Term.ofJson`, `toJson`): the codec's canonical form;
* `annot.check`: `Annot.check`'s verdict `{accept, count, code?}` on `{program, annotation, meaning}`;
* `annot.reference`: the same meaning decided gate by gate (`Annot.reference`);
* `annot.sets`: the family's sets in order, each its root-gate intervals `[[a, b], …]`, and the gates each label holds,
  on `{program, annotation}` (`Annot.sets`, `Annot.labelled`).

A refusal is a reject whose reason starts with its code (`term`, `table`, `plan`, `fuel`, `meaning-unsupported`).
-/

namespace AnnotDifftest

open Lean (Json toJson)
open Flock.Annot

def program (j : Json) : Except String Flock.Program.Prog := do
  let .ok (.str text) := j.getObjVal? "program" | throw "term: program is the descriptor's text"
  Flock.Program.decode text.toUTF8

def desc (j : Json) : Except String Desc :=
  Desc.ofJson ((j.getObjVal? "annotation").toOption.getD Json.null)

def meaning (j : Json) : Except String Meaning :=
  Meaning.ofJson ((j.getObjVal? "meaning").toOption.getD (Json.mkObj []))

def intervals (xs : Array (Nat × Nat)) : Json := Json.arr (xs.map fun (a, b) => toJson #[a, b])

def term : Difftest.Fn := fun j => return (← Term.ofJson Flock.Steps.FUEL j).toJson

def check : Difftest.Fn := fun j => do
  return (← Flock.Annot.check (← meaning j) (← program j) (← desc j)).toJson

def reference : Difftest.Fn := fun j => do
  return (← Flock.Annot.reference (← meaning j) (← program j) (← desc j)).toJson

def sets : Difftest.Fn := fun j => do
  let P ← program j
  let F ← eval P (← desc j)
  let ss ← Flock.Annot.sets P F
  let ls ← labelled P F
  return Json.mkObj [("sets", Json.arr (ss.map (intervals ·.gates))), ("labels", Json.arr (ls.map intervals))]

def table := Difftest.table [("annot.term", term), ("annot.check", check), ("annot.reference", reference),
  ("annot.sets", sets)]

end AnnotDifftest
