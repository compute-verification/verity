module

public import Flock.Test.RegComplete
meta import Flock.Test.RegComplete

/-!
# Two parameters of one name

`desc "w"` is a program whose template `RegisteredTestTwoParams_v0` has two parameters named `w`, the first reading the
program's registered input `A` and the second its registered input `B`; `desc "v"` names the second `v`
(`verity/protocols/proof/spot/verifier/tests/test_registered.py`'s `_two_params`, whose descriptors these are, each
pinned by its SHA-512 in both). `Program.decode` refuses `desc "w"`, as the Python decoder and `Builder` do, so
`Registered.coversParam`, which matches a port to a parameter by name, never meets one name for two parameters.

Flock's registered reads in `verity verify proof`'s order (`Registered.ofJson` and `paramsOf`, then `checkProgram`,
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
  ("{\"annotations\":{},\"boolean\":{\"RegisteredTestXor8_v1\":\"StandIn[RegisteredTestXor8]_v1\"},\"callees\":[\"RegisteredTestXor8_v1\",\"RegisteredInput8_v1\",\"RegisteredTestTwoParams_v0\",\"Const1[0x0]_v1\"],\"definitions\":{\"Const1[0x0]_v1\":{\"form\":\"primitive\",\"id\":\"Const1[0x0]_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"RegisteredInput8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredInput8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"RegisteredTestTwoParams_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,1,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",1,1,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"n\",1,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"w\",[\"Array\",2,[\"Value\",8]]],[\"" ++
    second ++
    "\",[\"Array\",2,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",8]},\"form\":\"composite\",\"id\":\"RegisteredTestTwoParams_v0\"},\"RegisteredTestXor8_v1\":{\"form\":\"primitive\",\"id\":\"RegisteredTestXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Root[RegisteredTestTwoParamsProgram_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2]},\"type\":2},{\"refs\":{\"affine\":[\"n\",1,0,2]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":0},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,2,2]},\"type\":2},{\"refs\":{\"affine\":[\"n\",1,2,2]},\"type\":2}],\"fn\":2,\"form\":\"call\",\"out\":0}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,1]},{\"affine\":[\"n\",3,0,1]}]},\"ret_type\":[\"Array\",2,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"Root[RegisteredTestTwoParamsProgram_v0]_v1\"},\"StandIn[RegisteredTestXor8]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"fn\":3,\"form\":\"primitive\",\"out\":3}],\"params\":[[\"a\",[\"Array\",8,[\"Value\",1]]],[\"b\",[\"Array\",8,[\"Value\",1]]]],\"ret\":{\"explicit\":[[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0],[\"n\",0,0]]},\"ret_type\":[\"Array\",8,[\"Value\",1]]},\"form\":\"composite\",\"id\":\"StandIn[RegisteredTestXor8]_v1\"}},\"program_fn\":\"RegisteredTestTwoParamsProgram_v0\",\"root\":\"Root[RegisteredTestTwoParamsProgram_v0]_v1\",\"schema\":\"verity-ir/descriptor/v2\",\"types\":[[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",2,[\"Value\",8]],[\"Value\",1]]}").toUTF8

def T : String := "RegisteredTestTwoParams_v0"

def prog (second : String) : Program.Prog := (Program.decode (desc second)).toOption.get!

def names (second : String) : Array String := (Registered.regParams (prog second) T).map (·.1)

-- the descriptors are the Python test's, each word primitive given a stand-in Boolean body (one constant bit)
#guard hexOf (Sha512.hash (desc "w")) ==
  "2d45995f25f30f4b13548360550c0d76cbfb36ffa8a8da60c135ce3f41310cac98e3a1c2bdfa9ddabe7c402506b661f099b092f4b51e9c1bcbd4c51c6a37d81b"
#guard hexOf (Sha512.hash (desc "v")) ==
  "80e298ae740db1a2d1b6b0d31ed34bf2cc918678bae9d485b79fcd5bc96a943303b964a20019da3ddfff474fad07be17743f6c04e8a3db37b41f0f5098afdc6b"

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
    ("domain", Json.str (hexOf dom)), ("words", Lean.toJson (32 : Nat)),
    ("row_schema", Json.str v2)] ++ (param.map fun p => [(Registered.PARAM_KEY, Json.str p)]).getD [])

/-- The verifier's file: ports `w` and `u`, both of `A`; `u` reads the parameter `second` when given. -/
def file (second : Option String := none) : Json :=
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", port), ("u", port second)])]

/-- `verity verify proof`'s registered reads of the verifier's program `program` and file `f` on the statement (`circ`,
`pub`, its units derived), in its order. -/
def setup (program : ByteArray) (f : Json) : Except String Unit := do
  let own ← Registered.ofJson f
  let params ← Registered.paramsOf f
  Registered.checkProgram (some program) true
  Registered.namesOnce circ
  Registered.check circ pub (some (Flock.Test.RegFormats.derivedReads own)) true
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
