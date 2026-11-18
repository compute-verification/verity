module

public import Proofs.Properties.InferenceOnly.RecordForm

/-!
# A shared run's rows are held

`SharedHeld`: on every record `read` accepts, every row with a leaf of a shared run (`Properties.Record.SharedLeaf`)
is held as a registered read of one parameter, the same for every row of that leaf: `Properties.Record.Served`'s
clause on shared leaves, for the record check's form (`form`, whose `held` is `Record.Case.heldAt`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Lean (Json)
open Flock.TemplateQuery (TRange)

/-- **Every row with a leaf of a shared run is held as a registered read of one parameter**, on every record `read`
accepts, over the ranges the template query gives its wiring's templates: the clause of `Properties.Record.Served`
on shared leaves. Open. The check's `sharedHeld` stage (`Record.finish`) proves it: a leaf of a shared run is a
reference of one argument of a root node, a part of which reads the run (`Resolve.LeafAt`, `arg_leaf`, the runs
being equal or disjoint, `runsOf`), so that argument is one of `sharedOf`'s; each of its rows is filled, unit by
unit, by a registered read of the row's parameter, θ (`heldFrom`, `sharedDecide`), which every row of the run's
input node shares; and `heldAt` gives that read. -/
def SharedHeld : Prop :=
  ∀ (j tr : Json) (c : Record.Case) (rs : Array TRange), Record.read j tr = .ok c →
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P) (c.W.templates.map (·.1)).toArray = .ok rs →
    ∀ g, Properties.Record.SharedLeaf c.f.P g → ∃ p, ∀ r l lo hi,
      Properties.Record.RowLeaf c.f.P rs c.W r l g lo hi → c.heldAt r = some (.read p)

end InferenceOnlyVerifier.Record.SecurityProofs
