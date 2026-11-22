module

public import Flock.TemplateQuery
meta import Flock.TemplateQuery

/-!
The verifier's own program, from its descriptor's bytes (`verity-ir/descriptor/v2`): Flock's decoding of the whole program
(`Flock.Program.Prog`), whose root Calls `Q_word` and `Q_call` cut, and the template queries' reading of it
(`Flock.TemplateQuery.Desc.ofProg`). Bytes `Flock.Program.decode` refuses are refused.
-/

@[expose] public section

namespace Spot

open Flock (V)

/-- `json` is the descriptor `prog` decodes, `Docket.ungatedOpeners`'s reading; null outside `Program.parse`. -/
structure Program where
  desc : Flock.TemplateQuery.Desc
  prog : Flock.Program.Prog
  json : Lean.Json := .null
  deriving Inhabited

/-- The program its descriptor's bytes describe. -/
def Program.parse (bytes : ByteArray) : V Program := do
  let prog ← Flock.Program.decode bytes
  if !(prog.defs[prog.root]!).body.recut.isEmpty then
    throw ("recut-root: the root reads an input node at a width other than its registered one; " ++
      "Spot registers no such Program")
  let json := match Flock.Program.readJson bytes with | some (.ok j) => j | _ => .null
  return { desc := .ofProg prog, prog, json }

/-- `partition_object.program_sha512`: SHA-512 of the canonical descriptor without `annotations`. -/
def Program.sha512 (p : Program) : ByteArray := p.desc.sha512

private def recutRootDesc : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]],[\"Array\",4,[\"Value\",8]]]}"
private def plainRootDesc : String :=
  "{\"boolean\":{},\"callees\":[\"Input16_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":1}],\"params\":[],\"ret\":{\"affine\":[\"n\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",16]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}},\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",16],[\"Array\",2,[\"Value\",16]]]}"

private def parseVerdict (s : String) : String :=
  match Program.parse s.toUTF8 with
  | .ok p => s!"ok recut={(p.prog.defs[p.prog.root]!).body.recut.toList}"
  | .error e => e

#guard parseVerdict recutRootDesc ==
  "recut-root: the root reads an input node at a width other than its registered one; Spot registers no such Program"
#guard parseVerdict plainRootDesc == "ok recut=[]"

end Spot
