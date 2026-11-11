module

public import Flock.TemplateQuery
public import Flock.JsonParse
public import Spot.Wires
public import Lean.Data.Json
meta import Flock.Program
meta import Flock.TemplateQuery
meta import Flock.JsonParse
meta import Spot.Wires
meta import Lean.Data.Json

/-!
# Evaluations beside the instances against the circuit record's properties

A scratch file, never merged: what the compiled readers give on small texts. Nothing here is a proof. `tie` is what
`NciCheck.Properties.Record.Served` reads of a record's text (a copy of its `wiresOf`), `direct` is
`Spot.Wires.Wiring.ofJson` on the whole text, and `showProg` is what `Flock.Program.decode` and
`Flock.TemplateQuery.evaluate` give on a descriptor, gate by gate and unit by unit.
-/

@[expose] public section

namespace NciCheck.Properties.Record.Probe

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn)

/-- A copy of `NciCheck.Properties.Record.wiresOf`. -/
def wiresOf (doc : Json) : Json :=
  Json.mkObj [("templates", Json.mkObj []), ("cuts", (doc.getObjVal? "cuts").toOption.getD (Json.mkObj [])),
    ("wires", (doc.getObjVal? "wires").toOption.getD Json.null)]

def showWiring (W : Spot.Wires.Wiring) : String :=
  s!"templates {W.templates.map (·.1)}, cuts {repr W.cuts}, wires {repr W.wires}, order {repr W.order}"

/-- What `Served` reads of the text `s`. -/
def tie (s : String) : String :=
  match Flock.JsonParse.parse s with
  | .error e => "parse refuses: " ++ e
  | .ok doc =>
    match Spot.Wires.Wiring.ofJson (wiresOf doc) with
    | .error e => "ofJson refuses: " ++ e
    | .ok W => showWiring W

/-- What `Spot.Wires.Wiring.ofJson` reads of the whole text `s`. -/
def direct (s : String) : String :=
  match Flock.JsonParse.parse s with
  | .error e => "parse refuses: " ++ e
  | .ok doc =>
    match Spot.Wires.Wiring.ofJson doc with
    | .error e => "ofJson refuses: " ++ e
    | .ok W => showWiring W

/-- Lean's own parser on `s`, printed compactly. -/
def core (s : String) : String :=
  match Lean.Json.parse s with
  | .error e => "refuses: " ++ e
  | .ok j => j.compress

def wireA : String :=
  "{\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0,\"radix\":[1],\"base\":0,\"step\":[0]}"

def wireACut : String :=
  "{\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0,\"radix\":[1],\"base\":0,\"step\":[0]," ++
  "\"cut\":[0,1]}"

def wireAExp : String :=
  "{\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0e0,\"radix\":[1e0],\"base\":-0," ++
  "\"step\":[0]}"

def wireAEsc : String :=
  "{\"re\\u0061der\":\"\\u0042\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0,\"radix\":[1],\"base\":0," ++
  "\"step\":[0]}"

def wireATwice : String :=
  "{\"reader\":\"Z\",\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0,\"radix\":[1]," ++
  "\"base\":0,\"step\":[0]}"

def wireAFloat : String :=
  "{\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0.0,\"radix\":[1],\"base\":0,\"step\":[0]}"

def wireAExtra : String :=
  "{\"reader\":\"B\",\"port\":\"x\",\"writer\":\"A\",\"out\":\"o0\",\"at\":0,\"radix\":[1],\"base\":0,\"step\":[0]," ++
  "\"note\":1}"

/-- The texts, each with a name. -/
def texts : List (String × String) := [
  ("wires empty", "{\"wires\":[]}"),
  ("one wire", "{\"wires\":[" ++ wireA ++ "]}"),
  ("one wire, cut [0,1] stated", "{\"wires\":[" ++ wireACut ++ "]}"),
  ("one wire, numbers as 0e0, 1e0, -0", "{\"wires\":[" ++ wireAExp ++ "]}"),
  ("one wire, a key and a value escaped", "{\"wires\":[" ++ wireAEsc ++ "]}"),
  ("one wire, reader given twice (Z then B)", "{\"wires\":[" ++ wireATwice ++ "]}"),
  ("one wire, at 0.0", "{\"wires\":[" ++ wireAFloat ++ "]}"),
  ("one wire, a key more", "{\"wires\":[" ++ wireAExtra ++ "]}"),
  ("wires twice: one wire, then none", "{\"wires\":[" ++ wireA ++ "],\"wires\":[]}"),
  ("wires twice: none, then one wire", "{\"wires\":[],\"wires\":[" ++ wireA ++ "]}"),
  ("cuts twice: x 2, then none", "{\"cuts\":{\"B\":{\"x\":2}},\"cuts\":{},\"wires\":[]}"),
  ("cuts: one template twice (x 2, then x 4)", "{\"cuts\":{\"B\":{\"x\":2},\"B\":{\"x\":4}},\"wires\":[]}"),
  ("cuts stated", "{\"cuts\":{\"B\":{\"x\":2}},\"wires\":[]}"),
  ("cuts 1e1", "{\"cuts\":{\"B\":{\"x\":1e1}},\"wires\":[]}"),
  ("cuts 0", "{\"cuts\":{\"B\":{\"x\":0}},\"wires\":[]}"),
  ("cuts null", "{\"cuts\":null,\"wires\":[]}"),
  ("cuts a list", "{\"cuts\":[],\"wires\":[]}"),
  ("cuts a number", "{\"cuts\":3,\"wires\":[]}"),
  ("cuts: a template's cuts null", "{\"cuts\":{\"B\":null},\"wires\":[]}"),
  ("no wires key", "{\"cuts\":{}}"),
  ("wires null", "{\"wires\":null}"),
  ("wires an object", "{\"wires\":{}}"),
  ("the text a list", "[]"),
  ("the text a string", "\"wires\""),
  ("the text empty", ""),
  ("text after the object", "{\"wires\":[]} {\"wires\":[" ++ wireA ++ "]}"),
  ("templates and order stated",
    "{\"order\":[[\"T\",0,1]],\"templates\":{\"T\":{\"outs\":{},\"rows\":{},\"units\":1}},\"wires\":[]}"),
  ("an order that is no list", "{\"order\":7,\"wires\":[]}"),
  ("templates malformed", "{\"templates\":7,\"wires\":[]}"),
  ("blocks and registered stated",
    "{\"blocks\":{\"inference\":{\"entries\":[],\"slots\":[]}},\"registered\":{\"params\":[\"w\"]},\"wires\":[]}")]

#eval do
  for (name, s) in texts do
    IO.println s!"TIE {name}: {tie s}"
    IO.println s!"    whole text: {direct s}"
    IO.println s!"    Lean's parser: {core s}"

/-! ### Programs -/

def programEmpty : String :=
  "{\"annotations\":{},\"callees\":[],\"definitions\":{\"Root_v1\":{\"bindings\":{},\"body\":{\"nodes\"" ++
  ":[],\"params\":[],\"ret\":{\"concat\":[]},\"ret_type\":[\"Tuple\",[]]},\"form\":\"composite\",\"id\"" ++
  ":\"Root_v1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1" ++
  "\",\"types\":[]}"

def programOne : String :=
  "{\"annotations\":{},\"callees\":[\"And16_v1\",\"Input16_v1\",\"Copy_v1\"],\"definitions\":{\"And16_v" ++
  "1\":{\"form\":\"primitive\",\"id\":\"And16_v1\",\"params\":[[\"a\",[\"Value\",16]],[\"b\",[\"Value\"" ++
  ",16]]],\"ret\":[\"Value\",16]},\"Copy_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs" ++
  "\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0" ++
  ",\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",16]]],\"ret\":{\"affine\":[\"n\",0" ++
  ",0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"Copy_v1\"},\"Input16_v1\":{\"for" ++
  "m\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root_v1\":{\"binding" ++
  "s\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":1,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"re" ++
  "fs\":{\"affine\":[\"n\",0,0,1]},\"type\":0}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"" ++
  "ret\":{\"affine\":[\"n\",1,0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"Root_v" ++
  "1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types" ++
  "\":[[\"Value\",16]]}"

def programTwo : String :=
  "{\"annotations\":{},\"callees\":[\"And16_v1\",\"Input16_v1\",\"Copy_v1\"],\"definitions\":{\"And16_v" ++
  "1\":{\"form\":\"primitive\",\"id\":\"And16_v1\",\"params\":[[\"a\",[\"Value\",16]],[\"b\",[\"Value\"" ++
  ",16]]],\"ret\":[\"Value\",16]},\"Copy_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs" ++
  "\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0" ++
  ",\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",16]]],\"ret\":{\"affine\":[\"n\",0" ++
  ",0,1]},\"ret_type\":[\"Value\",16]},\"form\":\"composite\",\"id\":\"Copy_v1\"},\"Input16_v1\":{\"for" ++
  "m\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Root_v1\":{\"binding" ++
  "s\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":1,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"re" ++
  "fs\":{\"affine\":[\"n\",0,0,1]},\"type\":0}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[],\"fn" ++
  "\":1,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",2,0,1]},\"type\":0}]," ++
  "\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",1,0,1]}" ++
  ",{\"affine\":[\"n\",3,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Value\",16],[\"Value\",16]]]},\"form\":\"c" ++
  "omposite\",\"id\":\"Root_v1\"}},\"program_fn\":\"Root_v1\",\"root\":\"Root_v1\",\"schema\":\"verity-" ++
  "ir/descriptor/v1\",\"types\":[[\"Value\",16]]}"

def showTy : Ty → String
  | .value w => s!"Value {w}"
  | .array n _ => s!"Array {n}"
  | .tuple fs => s!"Tuple of {fs.size}"
  | .record fs => s!"Record of {fs.size}"

def showRef (r : Ref) : String := (if r.p then "p" else "n") ++ toString r.i ++ "." ++ toString r.j

def showSeq (s : Flock.Program.Seq) : String :=
  "[" ++ ",".intercalate ((List.range s.count).map fun k => showRef (s.get k)) ++ "]"

partial def showScope : Scope → String
  | .top d off _ => s!"top(def {d}, off {off})"
  | .child p d off node m => s!"child({showScope p}, def {d}, off {off}, node {node}, member {m})"

def showNode (x : Flock.Program.Node) : String :=
  s!"({x.form.name} fn {x.fn}, n {x.n}, out {showTy x.out}, args {x.args.map fun a => showSeq a.seq})"

def showDefn (d : Defn) : String :=
  s!"{d.key}: id {d.id}, prim {d.prim}, params {d.params.map fun p => (p.1, showTy p.2)}, ret {showTy d.ret}, " ++
  s!"gates {d.gates}, paramLeaves {d.paramLeaves}, outLeaves {d.outLeaves}, body params " ++
  s!"{d.body.params.map fun p => (p.1, showTy p.2)}, body retType {showTy d.body.retType}, offsets " ++
  s!"{d.body.offsets}, body ret {showSeq d.body.ret}, nodes {d.body.nodes.map showNode}"

def showV (x : Flock.V Nat) : String :=
  match x with
  | .ok g => s!"gate {g}"
  | .error e => "refused: " ++ e

/-- What the decoder, the template query, the gate walk and the reference walk give on the descriptor `s`. -/
def showProg (s : String) (ts : Array String) : IO Unit := do
  match Flock.Program.decode s.toUTF8 with
  | .error e => IO.println ("  decode refuses: " ++ e)
  | .ok P =>
    let root := P.defs[P.root]!
    IO.println s!"  decodes: root {P.root}, gates {root.gates}, programFn {P.programFn}"
    for d in P.defs do
      IO.println ("  definition " ++ showDefn d)
    match Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts with
    | .error e => IO.println ("  evaluate refuses: " ++ e)
    | .ok rs => IO.println s!"  evaluate: ranges {rs.map fun r => (r.node, r.template, r.first, r.count)}"
    for g in [0:root.gates] do
      match P.gate g with
      | .error e => IO.println s!"  gate {g}: refused: {e}"
      | .ok (p, ops, sc) =>
        IO.println (s!"  gate {g}: definition {p} ({(P.defs[p]!).key}), input " ++
          s!"{Flock.Program.isInputId (P.defs[p]!).key}, ret {showTy (P.defs[p]!).ret}, operands {ops}, " ++
          s!"scope {showScope sc}")
    for k in [0:root.body.nodes.size] do
      let x := root.body.nodes[k]!
      if x.form == .prim then continue
      for m in [0:x.members] do
        let sc := P.child P.rootScope k m
        let d := P.defs[sc.d]!
        IO.println s!"  root node {k} member {m}: scope {showScope sc}"
        for i in [0:d.params.size] do
          for j in [0:d.params[i]!.2.leaves] do
            IO.println s!"    parameter {d.params[i]!.1} leaf {j}: {showV (Flock.Program.resolve P sc ⟨true, i, j⟩)}"
        for l in [0:d.outLeaves] do
          IO.println s!"    return leaf {l}: {showV (Flock.Program.resolve P sc (d.body.ret.get l))}"

#eval do
  IO.println "PROGRAM empty, no template"
  showProg programEmpty #[]
  IO.println "PROGRAM one unit, template Copy_v1"
  showProg programOne #["Copy_v1"]
  IO.println "PROGRAM one unit, no template listed"
  showProg programOne #[]
  IO.println "PROGRAM two units, template Copy_v1"
  showProg programTwo #["Copy_v1"]

/-- The facts about the decoder's output on `programOne` that the one-unit witness takes, each as a Boolean. -/
def oneUnitFacts (P : Flock.Program.Prog) : List (String × Bool) :=
  let sc := P.child P.rootScope 1 0
  let d := P.defs[sc.d]!
  let isValue16 : Ty → Bool := fun t => match t with
    | .value 16 => true
    | _ => false
  [("ranges", match Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) #["Copy_v1"] with
      | .ok rs => rs.size == 1 && rs[0]!.node == 1 && rs[0]!.template == "Copy_v1" && rs[0]!.first == 0 &&
          rs[0]!.count == 1
      | .error _ => false),
    ("rootGates", (P.defs[P.root]!).gates == 2),
    ("gate0", match P.gate 0 with
      | .ok (p, ops, _) => ops == #[] && isValue16 (P.defs[p]!).ret && Flock.Program.isInputId (P.defs[p]!).key
      | .error _ => false),
    ("gate1", match P.gate 1 with
      | .ok (p, ops, _) => ops == #[0, 0] && isValue16 (P.defs[p]!).ret &&
          !Flock.Program.isInputId (P.defs[p]!).key
      | .error _ => false),
    ("off", sc.off == 1),
    ("unitGates", d.gates == 1),
    ("params", d.params.size == 1 && d.params[0]!.1 == "x" && isValue16 d.params[0]!.2),
    ("ret", isValue16 d.ret),
    ("port", match Flock.Program.resolve P sc ⟨true, 0, 0⟩ with
      | .ok g => g == 0
      | .error _ => false)]

#eval do
  match Flock.Program.decode programOne.toUTF8 with
  | .error e => IO.println ("ONEUNIT decode refuses: " ++ e)
  | .ok P => IO.println s!"ONEUNIT {oneUnitFacts P}"

/-! ### Wirings -/

def copyTemplate (units : Nat) : String × Spot.Wires.Template :=
  ("Copy_v1", { units := units, rows := [("x", { words := 1, outer := 1 })], outs := [("o0", 1)] })

/-- Unit 1's row `x` reads unit 0's output `o0`. -/
def wire01 : Spot.Wires.Wire :=
  { reader := "Copy_v1", port := "x", writer := "Copy_v1", out := "o0", start := 1, radix := [1], base := 0,
    step := [0], cut0 := 0, cuts := 1 }

def showCheck (W : Spot.Wires.Wiring) (rows : List (String × Nat × String × Nat)) : String :=
  match Spot.Wires.check W with
  | .error r => s!"refused: {r.code.name}, wire {r.wire}"
  | .ok S => s!"passes: wires {S.wires.length}, rowWords {S.rowWords}, writers " ++
      toString (rows.map fun (t, u, p, r) => S.writerOf t u p r)

#eval do
  IO.println ("CHECK no template, no wire, no order: " ++ showCheck ⟨[], [], [], none⟩ [])
  IO.println ("CHECK no template, no wire, order []: " ++ showCheck ⟨[], [], [], some []⟩ [])
  IO.println ("CHECK one unit, no wire, no order: " ++
    showCheck ⟨[copyTemplate 1], [], [], none⟩ [("Copy_v1", 0, "x", 0)])
  IO.println ("CHECK one unit, no wire, order [(Copy_v1, 0, 1)]: " ++
    showCheck ⟨[copyTemplate 1], [], [], some [("Copy_v1", 0, 1)]⟩ [("Copy_v1", 0, "x", 0)])
  IO.println ("CHECK two units, wire 0 to 1, no order: " ++
    showCheck ⟨[copyTemplate 2], [], [wire01], none⟩ [("Copy_v1", 0, "x", 0), ("Copy_v1", 1, "x", 0)])
  IO.println ("CHECK two units, wire 0 to 1, order of the two units: " ++
    showCheck ⟨[copyTemplate 2], [], [wire01], some [("Copy_v1", 0, 1), ("Copy_v1", 1, 1)]⟩
      [("Copy_v1", 0, "x", 0), ("Copy_v1", 1, "x", 0)])
  IO.println ("CHECK two units, wire 0 to 1, the order reversed: " ++
    showCheck ⟨[copyTemplate 2], [], [wire01], some [("Copy_v1", 1, 1), ("Copy_v1", 0, 1)]⟩
      [("Copy_v1", 0, "x", 0), ("Copy_v1", 1, "x", 0)])
  IO.println ("TIE of the two units' record: " ++ tie ("{\"wires\":[{\"reader\":\"Copy_v1\",\"port\":\"x\"," ++
    "\"writer\":\"Copy_v1\",\"out\":\"o0\",\"at\":1,\"radix\":[1],\"base\":0,\"step\":[0]}]}"))

end NciCheck.Properties.Record.Probe
