module

public import Spot.Docket
public import Spot.Admitted
meta import Spot.Docket
meta import Spot.Admitted

/-!
`Spot.Docket.check`'s codes, each on the honest record of a small verifier changed in one place: the record the verifier's
own copies give (`honest`), so that the check returns no code, and per code the one change that refuses by it alone.
-/

@[expose] public section

namespace Spot.Test.Docket

open Lean (Json)
open Flock (V)
open Spot.Docket

/-- `E3` (`Spot.Admitted`) with a second call of `E_v1` on the same `x` and `y`: two proof blocks read the run Inputs. -/
def E3S : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",3,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,3]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":3,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,3]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2},{"args":[{"refs":{"affine":["n",0,0,3]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"concat":[{"affine":["n",2,0,4]},{"affine":["n",3,0,4]}]},"ret_type":["Array",8,["Value",1]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",3,["Value",1]],["Tuple",[["Array",3,["Value",1]],["Value",1]]],["Array",8,["Value",1]]]}"#

def program (d : String) : Program := (Program.parse d.toUTF8).toOption.getD default

/-- The query listing `E_v1`. -/
def QE : Json :=
  Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", "E_v1")]), ("version", (0 : Nat))]

/-- `E_v1`'s program function's parameters, `x : Array 3 (Value 1)` and `y : Value 1`. -/
def params (p : Program) : Array Registered.Param :=
  let bit := Json.arr #["Value", (1 : Nat)]
  (Registered.params p.desc (Json.arr #[Json.arr #["x", Json.arr #["Array", (3 : Nat), bit]], Json.arr #["y", bit]])).toOption.getD #[]

/-- A v2 verifier of the descriptor `d`: `x` registered in rows of one word, one port, a uniform sample of one. -/
def verifier (d : String := Spot.OneStageZk.E3) : Verifier :=
  let p := program d
  { program := p, query := QE, samplingDesign := Json.mkObj [("law", "subset"), ("k", (1 : Nat))], workTable := none,
    scheme := "s", ports := #[("o", Json.mkObj [("leaves", (4 : Nat)), ("domain", "d")])], leafLayer := none,
    anchors := #[], log := #[], partition := none, registered := #[("x", 1)], params := params p }

/-- A root, lowercase hex of 64 bytes. -/
def ROOT : String := String.ofList (List.replicate 128 'a')

/-- Another root. -/
def OTHER : String := String.ofList (List.replicate 128 'b')

/-- The record `v`'s own copies give: the check returns no code on it. -/
def honest (v : Verifier) : V Json := do
  let program := Flock.hexOf v.program.sha512
  let part := v.partition.getD (Partition.build program v.query)
  let some pd := Partition.digest part | throw "no partition digest"
  let previous := match v.log.back? with
    | some e => (digestHex e).toOption.map Json.str |>.getD .null
    | none => .null
  let base : List (String × Json) :=
    [("format", if v.v2 then FORMAT_V2 else FORMAT), ("protocol", PROTOCOL), ("program", program),
     ("query", ← queryDigest (getD part "query")), ("partition", pd),
     ("population", Lean.toJson (← Partition.units v.program v.query)),
     ("law", ← Draw.derive v.program v.query v.samplingDesign v.workTable), ("scheme", v.scheme),
     ("roots", Json.arr (v.ports.map fun (p, w) => Json.mkObj [("port", p), ("root", ROOT),
       ("leaves", getD w "leaves"), ("domain", getD w "domain")])),
     ("leaf_layer", optStr v.leafLayer), ("anchors", Json.arr v.anchors),
     ("window", Json.mkObj [("kind", "served")]), ("previous", previous)]
  if !v.v2 then return Json.mkObj base
  let es ← ownEntries v
  return Json.mkObj (base ++ [("registered", Json.arr (es.map fun e => Json.mkObj [("value", e.value), ("root", ROOT),
    ("leaves", Lean.toJson e.leaves), ("words", Lean.toJson e.words), ("domain", e.domain)]))])

/-- The record `j` with key `k` set to `x`, or removed for none. -/
def put (j : Json) (k : String) (x : Option Json) : Json := match j with
  | .obj kvs => Json.mkObj ((kvs.toArray.filter (·.1 != k)).toList ++ (x.map fun x => [(k, x)]).getD [])
  | _ => j

/-- The record's first registered entry and first root, each with key `k` set to `x`. -/
def putEntry (j : Json) (list k : String) (x : Json) : Json := match getD j list with
  | .arr es => put j list (some (Json.arr (es.modify 0 fun e => put e k (some x))))
  | _ => j

/-- The check's codes on the honest record of `v`, changed by `f`. -/
def codes (v : Verifier) (f : Json → Json := id) : Array String :=
  match honest v with
  | .ok r => check (f r) v
  | .error e => #[s!"fixture: {e}"]

/-- The record with an entry for the certified value `c` of domain `d`, under `ROOT`. -/
def withCertified (c d : String) (j : Json) : Json := match getD j "registered" with
  | .arr es => put j "registered" (some (Json.arr (es.push (Json.mkObj [("value", c), ("root", ROOT),
      ("leaves", (1 : Nat)), ("words", (1 : Nat)), ("domain", d)]))))
  | _ => j

def V3 : Verifier := verifier

#guard codes V3 == #[]
#guard codes V3 (put · "previous" none) == #["keys"]
#guard codes V3 (put · "protocol" (some "verity/sampled-proofs/one-stage/v0")) == #["R1-protocol"]
#guard codes V3 (put · "law" (some (Json.mkObj [("law", "subset"), ("k", (2 : Nat))]))) == #["R1-law"]
#guard codes V3 (put · "window" (some (Json.mkObj [("kind", "rehearsal")]))) == #["window"]
#guard codes V3 (put · "program" (some OTHER)) == #["R2-program"]
-- a partition with no digest gives no honest record of its own, so the record is the unchanged verifier's
#guard (honest V3).toOption.map (check · { V3 with partition := some (Json.mkObj [("format", "verity/partition/v0"),
  ("program", Flock.hexOf V3.program.sha512), ("query", QE)]) }) == some #["R2-partition"]
#guard codes V3 (put · "query" (some OTHER)) == #["R2-query"]
#guard codes V3 (put · "population" (some (2 : Nat))) == #["R2-population"]
#guard codes V3 (put · "scheme" (some "t")) == #["R3-scheme"]
#guard codes V3 (put · "roots" (some (Json.arr #[]))) == #["R3-roots"]
#guard codes { V3 with ports := V3.ports.push ("q", Json.mkObj [("leaves", (4 : Nat)), ("domain", "d")]) }
  (put · "roots" (some (Json.arr #[Json.mkObj [("port", "o"), ("root", ROOT), ("leaves", (4 : Nat)), ("domain", "d")]]))) ==
  #["R3-ports"]
#guard codes V3 (putEntry · "roots" "root" "zz") == #["R3-root"]
#guard codes V3 (putEntry · "roots" "leaves" (5 : Nat)) == #["R3-leaves"]
#guard codes V3 (putEntry · "roots" "domain" "e") == #["R3-domain"]
#guard codes V3 (put · "leaf_layer" (some "l")) == #["R4-leaf-layer"]
#guard codes V3 (put · "anchors" (some (Json.arr #[(1 : Nat)]))) == #["R5-anchors"]
#guard codes V3 (put · "previous" (some OTHER)) == #["R6-previous"]
-- the log holds the honest record itself, so the window is the log's and the record's previous is its digest
#guard codes { V3 with log := #[(honest V3).toOption.getD .null] } == #["R6-window"]
#guard codes { V3 with outputs := #[("x", "00", .port "p")] } == #["R7-output"]
#guard codes V3 (put · "registered" (some .null)) == #["R7-registered"]
#guard codes V3 (put · "registered" (some (Json.arr #[]))) == #["R7-values"]
#guard codes V3 (putEntry · "registered" "root" "zz") == #["R7-root"]
#guard codes V3 (putEntry · "registered" "leaves" (4 : Nat)) == #["R7-leaves"]
#guard codes V3 (putEntry · "registered" "words" (2 : Nat)) == #["R7-words"]
#guard codes V3 (putEntry · "registered" "domain" "e") == #["R7-domain"]
-- 16-bit words are named by leaving `word_bits` out
#guard codes V3 (putEntry · "registered" "word_bits" (16 : Nat)) == #["R7-word_bits"]
#guard codes { V3 with pinned := #[("x", OTHER)] } == #["R7-held"]
#guard codes { V3 with supplied := #[("z", #[])] } == #["R7-supplied"]
#guard codes { V3 with certified := #[({ value := "c", leaves := 1, words := 1, wordBits := 16, domain := "d" }, OTHER)] }
  (withCertified "c" "d") == #["R7-certified"]
#guard codes { V3 with tied := #["nope"] } == #["R8-tied"]
#guard codes (verifier E3S) == #["R8-shared-input"]
-- a member's units past the population of one: `bounds` refuses it
#guard codes { V3 with members := #[(#[(0, 2)], 1)] } == #["link-points"]

end Spot.Test.Docket
