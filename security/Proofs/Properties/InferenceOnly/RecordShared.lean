module

public import Proofs.Properties.InferenceOnly.RecordForm

/-!
# A shared run's rows are held

`SharedHeld`: on every record `read` accepts, every row with a leaf of a shared run (`Properties.Record.SharedLeaf`)
is held as a registered read of one parameter, or as a table read of one table's root or domain id
(`Properties.Record.SharesRun`), the same for every row of that leaf: `Properties.Record.Served`'s clause on shared
leaves, for the record check's form (`form`, whose `held` is `Record.Case.heldAt`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.Record.SecurityProofs

open Lean (Json)
open Flock.TemplateQuery (TRange)

/-- **Every row with a leaf of a shared run is held as a registered read of one parameter, or as a table read of one
table's root or domain id**, on every record `read` accepts, over the ranges the template query gives its wiring's
templates: the clause of `Properties.Record.Served` on shared leaves. Open. The check's `sharedHeld` stage
(`Record.finish`) proves it: a leaf of a shared run is a reference of one argument of a root node, a part of which
reads the run (`Resolve.LeafAt`, `arg_leaf`, the runs being equal or disjoint, `runsOf`), so that argument is one of
`sharedOf`'s; each of its rows is filled, unit by unit, by a registered read of the row's parameter, θ (`heldFrom`),
or by one table read of a θ table (`tableFrom`), whose source every row of the run's input node shares
(`sharedDecide`'s marks); and `heldAt` gives that reason. -/
def SharedHeld : Prop :=
  ∀ (j tr : Json) (c : Record.Case) (rs : Array TRange), Record.read j tr = .ok c →
    Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P) (c.W.templates.map (·.1)).toArray = .ok rs →
    ∀ g, Properties.Record.SharedLeaf c.f.P g → ∃ h : {h : Spot.RecordText.Held // Properties.Record.SharesRun h},
      ∀ r l lo hi, Properties.Record.RowLeaf c.f.P rs c.W r l g lo hi → c.heldAt r = some h.1

/-- **A digest as a port's words** (`interface.digest_words`): the 64 bytes `d` spells in hex as eight big-endian
64-bit words, each its four 16-bit words least significant first (`Flock.unhex`); none unless `d` spells 64 bytes. -/
def digestWords (d : String) : Option (Array Nat) := do
  let bs ← (_root_.Flock.unhex d).toOption
  if bs.size != 64 then none
  pure ((List.range 32).map fun k =>
    let x := (List.range 8).foldl (fun a b => 256 * a + (bs[8 * (k / 4) + b]!).toNat) 0
    x / 65536 ^ (k % 4) % 65536).toArray

/-- **A table read's row is its table's registration.** For every record the check accepts, each input row `r` it
holds as a table read, `.table name p root`, is held in its proof block's session (`heldRow`) to the words of the root
(`root`), else the domain id, of the log's first record's last registration of `p`. Open: no code compares a table
read's row with the registration. It is what crediting a table read's row as θ (`Record.kindOf`) means, as
`Properties.Record.PinnedOpens` is for an opened row. -/
def TableReadHeld (heldRow : Record.Case → Spot.RecordText.Row → Option (Array Nat)) : Prop :=
  ∀ (j tr : Json) (c : Record.Case), Record.read j tr = .ok c →
    ∀ r name p root, c.heldAt r = some (.table name p root) →
      ∃ e d w, Theta.lastWith c.f.T.log[0]! "registered" "value" p = some e ∧
        Theta.strAt e (if root then "root" else "domain") = some d ∧ digestWords d = some w ∧ heldRow c r = some w

end InferenceOnlyVerifier.Record.SecurityProofs
