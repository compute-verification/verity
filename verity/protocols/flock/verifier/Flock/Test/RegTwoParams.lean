module

public import Flock.Test.RegComplete
meta import Flock.Test.RegComplete

/-!
# Two parameters of one name

`desc "w"` is a program whose template `RegisteredTestTwoParams_v0` has two parameters named `w`, the first reading the
program's registered input `A` and the second its registered input `B`; `desc "v"` names the second `v`
(`verity/protocols/spot/verifier/tests/test_registered.py`'s `_two_params`, whose descriptors these are, each pinned by
its SHA-512 in both). `Program.decode` refuses `desc "w"`, as the Python decoder and `Builder` do, so
`Registered.coversParam`, which matches a port to a parameter by name, never meets one name for two parameters.

Flock's registered reads in `flock-verify verify`'s order (`Registered.ofJson` and `paramsOf`, then `checkProgram`,
`namesOnce`, `check`, `checkComplete` and `checkCover`) refuse a statement of `desc "w"` at `checkCover`'s decode. For
`desc "v"`, a file whose ports `w` and `u` both open `A`'s registered root is refused while no port reads parameter `v`,
and accepted once `u` names it: no check of Flock's reads which registered input a parameter reads, and the verifier's
file, built from its own program (`registered.reads`), states each port's value and root.
-/

@[expose] public section

namespace Flock.Test.RegTwoParams

open Lean (Json)
open Flock.Test.RegFormats (dom bcs leaves root v2 v2Port)

/-- `_two_params(second)`'s descriptor: its template's parameters are `w` and `second`. -/
def desc (second : String) : ByteArray :=
  ("{\"annotations\":{},\"callees\":[\"RegisteredTestXor8_v1\",\"RegisteredInput8_v1\",\"RegisteredTestTwoParams_v0\"],\"definitions\":{\"RegisteredInput8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredInput8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"RegisteredTestTwoParams_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,1,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,1,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"n\",1,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"w\",[\"Array\",2,[\"Value\",8]]],[\"" ++
    second ++
    "\",[\"Array\",2,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"RegisteredTestTwoParams_v0\"},\"RegisteredTestXor8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredTestXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Root[RegisteredTestTwoParamsProgram_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2]},\"type\":2},{\"refs\":{\"affine\":[\"n\",1,0,2]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,2,2]},\"type\":2},{\"refs\":{\"affine\":[\"n\",1,2,2]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,1]},{\"affine\":[\"n\",3,0,1]}]},\"ret_type\":[\"Array\",2,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[RegisteredTestTwoParamsProgram_v0]_v1\"}},\"program_fn\":\"RegisteredTestTwoParamsProgram_v0\",\"root\":\"Root[RegisteredTestTwoParamsProgram_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",2,[\"Value\",8]]]}").toUTF8

def T : String := "RegisteredTestTwoParams_v0"

def prog (second : String) : Program.Prog := (Program.decode (desc second)).toOption.get!

def names (second : String) : Array String := (Registered.regParams (prog second) T).map (·.1)

-- the descriptors are the Python test's
#guard hexOf (Sha512.hash (desc "w")) ==
  "802171939d9baed267cc2b5516306111e61a937c6a5ad3892c0c032d44c287b278b2456e0f6e79f31d9797c9da6d86d1c67817396fa1f12ad1bb48055021b6fb"
#guard hexOf (Sha512.hash (desc "v")) ==
  "352f7f13f7e80ef597b363b4fee73e5b2cd0a6c44cd4aae4a39337dbb87be66dcd137737f06e3319226a2faee21a86c9a0921e027960d737e16164fbe5baa000"

def twice : String := s!"definition '{T}': parameters: a name twice"

-- `desc "w"` is refused; `desc "v"` decodes, each of its template's two instances reading a registered input at both
-- its parameters, and a port of parameter `w` alone leaves its `v` uncovered
#guard match Program.decode (desc "w") with | .error e => e == twice | .ok _ => false
#guard (Program.decode (desc "v")).toBool
#guard names "v" == #["w", "v", "w", "v"]
#guard (Registered.coverMiss (prog "v") T #["w"]).map (·.1) == some "v"

/-- A template query of `T`, whose instances are the program's two root calls. -/
def q : Json := Json.mkObj [("query", Json.mkObj [("name", Json.str "Q_template_instance"),
  ("version", Lean.toJson (0 : Nat)), ("params", Json.mkObj [("template", Json.str T)])])]

/-- The statement's circuit: META names `T` alone, and its ports are `w` (the first parameter's rows) and `u` (the
second's), each a bit row of 512 bits. -/
def circ : Circuit :=
  { (default : Circuit) with
    metaJ := Json.mkObj [("program_digests", Json.mkObj [(T, Json.str "00")])]
    ports := #[v2Port, { v2Port with name := "u" }] }

/-- Two instances, instance `i` reading table row `i` of each port, both tables rows of `A` opened at its root, row `r`
at position `r` through the other row's leaf. -/
def pub : Public :=
  let entry := Json.mkObj [("value", Json.str "A"), ("positions", Lean.toJson [0, 1]),
    ("paths", Json.arr #[Json.str (hexOf (leaves v2)[1]!), Json.str (hexOf (leaves v2)[0]!)])]
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [0, 1])]),
      ("registered", Json.mkObj [("w", entry), ("u", entry)])],
    n := 2, digests := bcs ++ bcs, outputs := #[], sha := .empty,
    tables := some { counts := #[2, 2], starts := #[0, 2], refs := #[0, 0, 1, 1] } }

/-- A port of `A` at its registered root, unit `u` reading row `u`, reading the template parameter `param` when it names
one (else the port's own name). -/
def port (param : Option String := none) : Json :=
  Json.mkObj ([("value", Json.str "A"), ("root", Json.str (hexOf (root v2))), ("leaves", Lean.toJson (2 : Nat)),
    ("domain", Json.str (hexOf dom)), ("reads", Lean.toJson [some 0, some 1]), ("words", Lean.toJson (32 : Nat)),
    ("row_schema", Json.str v2)] ++ (param.map fun p => [(Registered.PARAM_KEY, Json.str p)]).getD [])

/-- The verifier's file: ports `w` and `u`, both of `A`; `u` reads the parameter `second` when given. -/
def file (second : Option String := none) : Json :=
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", port), ("u", port second)])]

/-- `flock-verify verify`'s registered reads of the verifier's program `program` and file `f` on the statement (`circ`,
`pub`, its units derived), in its order. -/
def setup (program : ByteArray) (f : Json) : Except String Unit := do
  let own ← Registered.ofJson f
  let params ← Registered.paramsOf f
  Registered.checkProgram (some program) true
  Registered.namesOnce circ
  Registered.check circ pub (some own) true
  Registered.checkComplete circ own
  Registered.checkCover circ (some program) (some q) (some own) params

def why (program : ByteArray) (f : Json) : String :=
  match setup program f with
  | .error e => e
  | .ok () => ""

-- `desc "w"`: refused, on a file naming no port of its second parameter
#guard (Registered.paramsOf (file none)).toOption == some #["u", "w"]
#guard why (desc "w") (file none) == twice
-- `desc "v"`: the same file is refused, its `v` read by no port; a file whose `u` reads `v` is accepted, at `A`'s root too
#guard why (desc "v") (file none) ==
  "registered reads (cover): parameter v of RegisteredTestTwoParams_v0 reads a registered input of the verifier's program, and no port of the verifier's registered reads reads it: its rows would be committed afresh"
#guard why (desc "v") (file (some "v")) == ""

end Flock.Test.RegTwoParams
