module

public import Flock.TemplateQuery
public import PouwCheck.Json
public import Spot.Json

/-!
# The root nodes credit reads as inputs

`credit.py`'s readers of a root's nodes (`layout_of`, `live_of`, `leaf_of`) skip a node as an input only as
`program.is_input_entry` reads it: its callee a primitive under an input id. Each of the Lean's asks
`Flock.TemplateQuery.Desc.isInputNode`, the template query's own reading: over the decoded program (`nonInputNodes`),
or over a descriptor's JSON (`inputsOf`, `inputNode`).
-/

@[expose] public section

namespace PouwCheck.Credit

open Lean (Json)

/-- A descriptor's callees and definitions as `Flock.TemplateQuery.Desc.isInputNode` reads them, for a reader of the
descriptor's JSON (`program.is_input_entry`'s reading): each definition's key and whether its form is `primitive`. It
holds nothing else of the descriptor. -/
def inputsOf (callees : Array String) (defs : Json) : Flock.TemplateQuery.Desc where
  callees := callees
  defs := match defs with
    | .obj kvs => kvs.toArray.map fun (id, dj) =>
      (id, ({ prim := Spot.jsonStrIs (get dj "form") "primitive", nodes := #[] } : Flock.TemplateQuery.DDef))
    | _ => #[]
  root := ""
  sha512 := ByteArray.empty

/-- Whether a node whose callee is `d`'s callee `f` is an input node (`Flock.TemplateQuery.Desc.isInputNode`: the
callee a primitive whose id is an input id). A composite under an input id is a node like any other. -/
def inputNode (d : Flock.TemplateQuery.Desc) (f : Nat) : Bool := d.isInputNode { form := "", fn := f, n := 1 }

/-- The root's nodes of `P` other than its input nodes (`inputNode`, over `Flock.TemplateQuery.Desc.ofProg P`), in the
root's order. -/
def nonInputNodes (P : Flock.Program.Prog) : Array Flock.Program.Node :=
  let d := Flock.TemplateQuery.Desc.ofProg P
  (P.defs.getD P.root default).body.nodes.filter fun x => !inputNode d x.fn

end PouwCheck.Credit
