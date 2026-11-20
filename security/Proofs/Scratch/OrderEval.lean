module

public import Spot.OneStageZkStatement
public import Spot.Slots
meta import Spot.OneStageZkStatement
meta import Spot.Slots

/-! Scratch, not for merge: a root call node that reads a later root node's leaf through a narrower parameter, on
`decode`, the partition's order check, the door and `restrict`. `C8`: T(a : Value<8>) reads node 1, an Input16;
`C16`: T(a : Value<16>) reads node 1, an Input32; `H16`: the honest order, node 0 the Input16 and T(a : Value<16>)
reading it. Run: `git apply security/Proofs/Scratch/lakefile.patch && verity lean try security/Proofs/Scratch/OrderEval.lean`. -/

@[expose] public section

namespace Scratch.OrderEval

open Lean (Json)

def desc (w wIn : Nat) (forward : Bool) : String :=
  let input := "{\"args\":[],\"fn\":0,\"form\":\"primitive\",\"out\":1}"
  let call (n : Nat) := "{\"args\":[{\"refs\":{\"affine\":[\"n\"," ++ toString n ++ ",0,1]},\"type\":0}],\"fn\":1,\"form\":\"call\",\"out\":0}"
  let nodes := if forward then s!"[{call 1},{input}]" else s!"[{input},{call 0}]"
  let callAt := if forward then 0 else 1
  "{\"annotations\":{},\"boolean\":{},\"callees\":[\"Input" ++ toString wIn ++ "_v1\",\"T_v1\"],\"definitions\":{" ++
  "\"Input" ++ toString wIn ++ "_v1\":{\"form\":\"primitive\",\"id\":\"Input" ++ toString wIn ++
    "_v1\",\"params\":[],\"ret\":[\"Value\"," ++ toString wIn ++ "]}," ++
  "\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":" ++ nodes ++
    ",\"params\":[],\"ret\":{\"affine\":[\"n\"," ++ toString callAt ++ ",0,1]},\"ret_type\":[\"Value\"," ++ toString w ++
    "]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}," ++
  "\"T_v1\":{\"bindings\":{},\"body\":{\"nodes\":[],\"params\":[[\"a\",[\"Value\"," ++ toString w ++
    "]]],\"ret\":{\"affine\":[\"p\",0,0,1]},\"ret_type\":[\"Value\"," ++ toString w ++ "]},\"form\":\"composite\",\"id\":\"T_v1\"}}," ++
  "\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\"," ++
    toString w ++ "],[\"Value\"," ++ toString wIn ++ "]]}"

/-- T(a : Array 16 (Value 1)) = map Not a, at root node `callAt`, reading node `1 - callAt`, a batch of 16 `Input<wIn>`. -/
def descBits (wIn : Nat) (forward : Bool) (N : Nat := 16) : String :=
  let n := toString N
  let callAt := if forward then 0 else 1
  let input := "{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":" ++ n ++ ",\"out\":3}"
  let call := "{\"args\":[{\"refs\":{\"affine\":[\"n\"," ++ toString (1 - callAt) ++ ",0," ++ n ++ "]},\"type\":1}],\"fn\":2,\"form\":\"call\",\"out\":1}"
  let nodes := if forward then s!"[{call},{input}]" else s!"[{input},{call}]"
  let inp := "Input" ++ toString wIn ++ "_v1"
  "{\"annotations\":{},\"boolean\":{},\"callees\":[\"" ++ inp ++ "\",\"Not_v1\",\"T_v1\"],\"definitions\":{" ++
  "\"" ++ inp ++ "\":{\"form\":\"primitive\",\"id\":\"" ++ inp ++ "\",\"params\":[],\"ret\":[\"Value\"," ++ toString wIn ++ "]}," ++
  "\"Not_v1\":{\"form\":\"primitive\",\"id\":\"Not_v1\",\"params\":[[\"a\",[\"Value\",1]]],\"ret\":[\"Value\",1]}," ++
  "\"Root[P_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":" ++ nodes ++
    ",\"params\":[],\"ret\":{\"affine\":[\"n\"," ++ toString callAt ++ ",0," ++ n ++ "]},\"ret_type\":[\"Array\"," ++ n ++ ",[\"Value\",1]]},\"form\":\"composite\",\"id\":\"Root[P_v1]_v1\"}," ++
  "\"T_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0," ++ n ++ "]},\"type\":1}],\"axes\":[0],\"fn\":1,\"form\":\"batch\",\"n\":" ++ n ++ ",\"out\":1}]," ++
    "\"params\":[[\"a\",[\"Array\"," ++ n ++ ",[\"Value\",1]]]],\"ret\":{\"affine\":[\"n\",0,0," ++ n ++ "]},\"ret_type\":[\"Array\"," ++ n ++ ",[\"Value\",1]]},\"form\":\"composite\",\"id\":\"T_v1\"}}," ++
  "\"program_fn\":\"P_v1\",\"root\":\"Root[P_v1]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",1],[\"Array\"," ++ n ++ ",[\"Value\",1]],[\"Value\"," ++
    toString wIn ++ "],[\"Array\"," ++ n ++ ",[\"Value\"," ++ toString wIn ++ "]]]}"

def B16 : ByteArray := (descBits 16 true).toUTF8
def HB1 : ByteArray := (descBits 1 false).toUTF8
def B32 : ByteArray := (descBits 16 true 2048).toUTF8
def HB32 : ByteArray := (descBits 1 false 2048).toUTF8
def B16back : ByteArray := (descBits 16 false).toUTF8

def C8 : ByteArray := (desc 8 16 true).toUTF8
def C16 : ByteArray := (desc 16 32 true).toUTF8
def H16 : ByteArray := (desc 16 16 false).toUTF8

def query : Json :=
  Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", "T_v1")]),
    ("version", Lean.toJson (0 : Nat))]

def partitionOf (C : ByteArray) : Json :=
  let hex := match Flock.TemplateQuery.Desc.parse C with
    | .ok d => Flock.hexOf d.sha512
    | .error _ => String.ofList (List.replicate 128 '0')
  Json.mkObj [("blocks", Json.arr #[Json.arr #[Json.arr #[Lean.toJson (0 : Nat), Lean.toJson (1 : Nat)]]]),
    ("format", Json.str Flock.HmRow.PARTITION_V1), ("program", Json.str hex), ("query", query)]

def say {α : Type} (f : α → String) : Except String α → String
  | .ok a => s!"ok {f a}"
  | .error e => s!"refused: {e}"

def report (name : String) (C : ByteArray) : IO Unit := do
  let part := partitionOf C
  IO.println s!"ORDER {name} decode: {say (fun _ => "") (Flock.Program.decode C)}"
  match Flock.Program.decode C with
  | .error _ => pure ()
  | .ok P =>
    IO.println s!"ORDER {name} orderViolation: {(P.orderViolation.map (·.compress)).getD "none"}"
    IO.println s!"ORDER {name} root recut: {(P.defs[P.root]!).body.recut.toList}"
    let sp : Spot.Program := { desc := .ofProg P, prog := P }
    IO.println s!"ORDER {name} Spot.Program.parse: {say (fun _ => "") (Spot.Program.parse C)}"
    IO.println s!"ORDER {name} Partition.evaluate (the order check): {say (fun rs => toString (rs.map fun r => (r.node, r.template, r.first, r.count))) (Spot.Partition.evaluate sp query)}"
    IO.println s!"ORDER {name} Slots.popOk at 1 (popHeld): {Spot.Slots.popOk sp query 1}"
    IO.println s!"ORDER {name} TemplateQuery.Desc.parse+evaluate (restrict's blocks road): {say (fun rs => toString (rs.map fun r => (r.template, r.first, r.count))) (do let d ← Flock.TemplateQuery.Desc.parse C; let ts ← (Flock.TemplateQuery.templatesOf "Q_template_instance" 0 (Json.mkObj [("template", "T_v1")])).getD (throw "no"); Flock.TemplateQuery.evaluate d ts)}"
    IO.println s!"ORDER {name} HmRow.blocksOf: {say (fun b => toString b) (Flock.HmRow.blocksOf part C)}"
    IO.println s!"ORDER {name} hasEdge: {Spot.OneStageZk.hasEdge P}"
    IO.println s!"ORDER {name} cutsOf: {say (fun c => toString c) (Spot.OneStageZk.cutsOf (.ok P) part)}"
    IO.println s!"ORDER {name} unitAt 0: {say (fun km => toString km) (Spot.BlockFront.unitAt P 0)}"
    let x := (P.defs[P.root]!).body.nodes[if (P.defs[P.root]!).body.nodes[0]!.form == .call then 0 else 1]!
    IO.println s!"ORDER {name} operandsW (gate, width), first 3 and count: {say (fun o => toString (o.toList.take 3, o.size)) (Spot.BlockFront.operandsW P x 0)}"
    IO.println s!"ORDER {name} admittedBlock: {say (fun _ => "") (Spot.OneStageZk.admittedBlock P #[0] "T_v1")}"
    match Spot.Restrict.restrict P #[0] with
    | .error e => IO.println s!"ORDER {name} Restrict.restrict: refused: {e}"
    | .ok S =>
      IO.println s!"ORDER {name} Restrict.restrict: ok, S's root node widths, first 3 each {((S.defs[S.root]!).body.nodes.map (·.out.widths.toList.take 3)).toList}"
      IO.println s!"ORDER {name} admitted: {say (fun a => s!"key {a.key} ins {a.ins} fields {a.fields}") (Spot.OneStageZk.admitted part [] S)}"
    match Spot.OneStageZk.restrict C part 0 with
    | .ok b => IO.println s!"ORDER {name} restrict (C_S's file): ok, {b.size} bytes: {String.ofList (((String.fromUTF8? b).getD "?").toList.take 3000)}"
    | .error e => IO.println s!"ORDER {name} restrict (C_S's file): refused: {e}"

#eval report "C8" C8
#eval report "C16" C16
#eval report "H16" H16
#eval report "B16" B16
#eval report "B16back" B16back
#eval report "HB1" HB1
#eval report "B32" B32
#eval report "HB32" HB32

end Scratch.OrderEval
