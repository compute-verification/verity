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

/-- The descriptor `d` with `x`'s node a registered input and `y`'s a run Input. -/
def regX (d : String) : String :=
  ((d.replace r#""callees":["Input1_v1","Not_v1","E_v1"]"# r#""callees":["RegisteredInput1_v1","Not_v1","E_v1","Input1_v1"]"#).replace
    r#""Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]}"#
    (r#""Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"# ++
      r#""RegisteredInput1_v1":{"form":"primitive","id":"RegisteredInput1_v1","params":[],"ret":["Value",1]}"#)).replace
    r#"{"args":[],"fn":0,"form":"primitive","out":0}"# r#"{"args":[],"fn":3,"form":"primitive","out":0}"#

/-- A root, lowercase hex of 64 bytes. -/
def ROOT : String := String.ofList (List.replicate 128 'a')

/-- A v2 verifier of the descriptor `d`: `x` registered in rows of one word and pinned at `ROOT`, one port, a uniform
sample of one. -/
def verifier (d : String := regX Spot.OneStageZk.E3) : Verifier :=
  let p := program d
  { program := p, query := QE, samplingDesign := Json.mkObj [("law", "subset"), ("k", (1 : Nat))], workTable := none,
    scheme := "s", ports := #[("o", Json.mkObj [("leaves", (4 : Nat)), ("domain", "d")])], leafLayer := none,
    anchors := #[], log := #[], partition := none, registered := #[("x", 1)], params := params p,
    pinned := #[("x", ROOT)] }

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
  let es ← expectedEntries v
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
-- `x` is pinned, so a root of its that isn't one is also not its pin
#guard codes V3 (putEntry · "registered" "root" "zz") == #["R7-root", "R7-held"]
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
-- two proof blocks reading a run Input: the variant with no members refuses it; the per-member variant holds its
-- readers to the node's one tree
#guard codes (verifier (regX E3S)) == #["R8-shared-input"]
#guard !(codes { verifier (regX E3S) with members := #[(#[(0, 1)], 1), (#[(1, 2)], 1)] }).contains "R8-shared-input"
-- a registered value at a run Input node: refused by name
#guard codes (verifier Spot.OneStageZk.E3) == #["R7-run-input"]
#guard codes (verifier E3S) == #["R7-run-input", "R8-shared-input"]
-- a member's units past the population of one: `bounds` refuses it
#guard codes { V3 with members := #[(#[(0, 2)], 1)] } == #["link-points"]

/-! R8's hold on the program's registered inputs (`weightsCodes`): `E3S` with both inputs bound at registration, each read
by both calls, its leaves one bit each. -/

/-- `E3S` with `x` and `y` registered inputs. -/
def E3R : String := E3S.replace "Input1_v1" "RegisteredInput1_v1"

/-- A v1 verifier of `d` that expects nothing registered, its program function's parameters given. -/
def weighed (d : String) : Verifier := let v := verifier d; { v with registered := #[] }

/-- The verifier's own entry for `E3R`'s parameter `value` at one word a row (`Registered.entries`). -/
def ownEntry (value : String) : Registered.Entry :=
  match Registered.entries (program E3R).desc (params (program E3R)) #[(value, 1)] with
  | .ok #[e] => e
  | _ => { value, leaves := 0, words := 0, wordBits := 0, domain := "" }

/-- A pin of `value` at its own entry's shape, `more` leaves past it, its domain at the registration's program digest
`ROOT` unless `domain` names one. -/
def pin (value : String) (more : Nat := 0) (domain : Option String := none) : String × Json :=
  let e := ownEntry value
  let leaves := e.leaves + more
  let d := domain.getD ((Registered.domain ROOT value e.words leaves e.wordBits).toOption.map Flock.hexOf |>.getD "")
  let kvs : List (String × Json) := [("value", Json.str value), ("root", Json.str ROOT), ("leaves", Lean.toJson leaves),
    ("words", Lean.toJson e.words), ("domain", Json.str d)]
  (value, Json.mkObj (kvs ++ if e.wordBits == Registered.WORD_BITS then [] else [(Registered.WIDTH_KEY, Lean.toJson e.wordBits)]))

/-- The registration of record of `pins`, with the keys `more` beside `program` and `values`, and its SHA-512. -/
def ofRecord (pins : List (String × Json)) (more : List (String × Json) := []) : ByteArray × String :=
  let b := (Json.compress (Json.mkObj ([("program", Json.str ROOT), ("values", Json.mkObj pins)] ++ more))).toUTF8
  (b, Flock.hexOf (Flock.Sha512.hash b))

def PINS : List (String × Json) := [pin "x", pin "y"]

/-- The check's codes on the honest record of `v`. -/
def weighedCodes (v : Verifier) : Array String :=
  match honest v with
  | .ok r => check r v
  | .error e => #[s!"fixture: {e}"]

#guard Partition.inputReads (weighed E3R).program.prog ((Partition.templateRanges? (program E3R) QE).getD #[])
  #[("x", 0, 3), ("y", 3, 1)] == #[("E_v1", "x", "x"), ("E_v1", "y", "y")]
-- a registered input two proof blocks read passes with the registration and its digest; the
-- pins it accepts are no entry R7 reads, and the record stays v1
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord PINS) } == #[]
#guard ((weightsOf { weighed E3R with weightsOfRecord := some (ofRecord PINS) }).2.map fun (e, r) =>
  (e.value, e.leaves, r)) == #[("x", (ownEntry "x").leaves, ROOT), ("y", (ownEntry "y").leaves, ROOT)]
-- without the registration, or at another digest, nothing fixes its root
#guard weighedCodes (weighed E3R) == #["R8-unpinned"]
#guard weighedCodes { weighed E3R with weightsOfRecord := some ((ofRecord PINS).1, OTHER) } == #["R8-weights-digest"]
-- a marked input with no pin is refused
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord [pin "x"]) } == #["R8-unpinned"]
-- a pin of a parameter the program doesn't mark registered excuses nothing
#guard weighedCodes { weighed E3S with weightsOfRecord := some (ofRecord PINS) } == #["R8-shared-input"]
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord [pin "x", pin "y" 0 (some OTHER)]) } ==
  #["R8-weights-domain", "R8-unpinned"]
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord [pin "x" 1, pin "y"]) } == #["R8-weights-shape"]
-- the document is exactly `{program, values}`: a `reads` key, or any other, is another form
def READS : List (String × Json) := [("reads", Json.arr #[Json.arr #["E_v1", "x", "x"]])]
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord PINS READS) } == #["R8-weights-form"]
#guard weighedCodes { weighed E3R with weightsOfRecord := some (ofRecord PINS [("reads", Json.arr #[])]) } ==
  #["R8-weights-form"]
-- what the members' check holds reads to (`readPins`): with marked inputs, none without a registration or with one R8
-- refuses; with one R8 accepts, the marked inputs, their gates, the ranges reading them and R8's pins
#guard (readPins (weighed E3R)).isNone
#guard (readPins { weighed E3R with weightsOfRecord := some (ofRecord [pin "x"]) }).isNone
#guard (readPins { weighed E3R with weightsOfRecord := some (ofRecord PINS) }).map (fun rp =>
  (rp.loose, rp.held, rp.ranges.map (·.template), rp.pins.size)) == some (#["x", "y"], #[("x", 0, 3), ("y", 3, 1)],
    #["E_v1", "E_v1"], 2)
-- nothing marked, or every marked input expected: nothing held
#guard (readPins (weighed E3S)).map (fun rp => rp.loose.size + rp.ranges.size + rp.pins.size) == some 0
#guard (readPins { verifier E3R with registered := #[("x", 1), ("y", 1)] }).map
  (fun rp => rp.loose.size + rp.ranges.size + rp.pins.size) == some 0
-- expected by name alone excuses nothing: R8-unpinned unless R8 pins it or the verifier holds its root itself
#guard weightsCodes { verifier E3R with registered := #[("x", 1), ("y", 1)] } == #["R8-unpinned"]
#guard weightsCodes { verifier E3R with registered := #[("x", 1), ("y", 1)], weightsOfRecord := some (ofRecord PINS) } ==
  #[]
#guard weightsCodes { verifier E3R with registered := #[("x", 1), ("y", 1)], pinned := #[("x", ROOT), ("y", ROOT)] } ==
  #[]
#guard ((readPins { verifier E3R with registered := #[("x", 1), ("y", 1)], pinned := #[("x", ROOT)] }).map
  fun rp => rp.roots.map fun (e, r) => (e.value, r)) == some #[("x", ROOT)]
-- a registered input under a query that is no template query that applies
#guard weightsCodes { weighed E3R with query := Json.mkObj [("name", "Q_all")] } == #["R8-weights-query"]
-- why `readPins` is none: no template query applying, else no registration R8 accepts
#guard readPinsWhy { { weighed E3R with query := Json.mkObj [("name", "Q_all")] } with
  weightsOfRecord := some (ofRecord PINS) } == WEIGHTS_QUERY_WHY
#guard readPinsWhy (weighed E3R) == WEIGHTS_UNREGISTERED_WHY

/-! R7's hold on C's supplied anchors (`anchorUnpinned`): a mark of `registered.root/v1` or `registered.domain/v1` of a
value V is refused unless the verifier's `pinned` or the weights' registration of record pins V. -/

def anchorMark (x : String) (fn : String := "registered.root/v1") : Supplied.Mark :=
  { fn, args := #[("value", .str x)], layout := "digest", bits := 512 }

#guard codes { V3 with marks := #[("a", anchorMark "w")] } == #["R7-anchor-unpinned"]
#guard codes { V3 with marks := #[("a", anchorMark "w" "registered.domain/v1")] } == #["R7-anchor-unpinned"]
#guard anchorUnpinned { V3 with marks := #[("a", anchorMark "w"), ("b", anchorMark "w" "registered.domain/v1")] } ==
  #["w"]
#guard anchorUnpinned { V3 with marks := #[("a", anchorMark "w")], pinned := #[("w", ROOT)] } == #[]
#guard anchorUnpinned { V3 with marks := #[("a", anchorMark "x")], weightsOfRecord := some (ofRecord PINS) } == #[]
#guard anchorUnpinned { V3 with marks := #[("a", anchorMark "w")], weightsOfRecord := some (ofRecord PINS) } == #["w"]
-- a registration of record R8 refuses pins nothing
def V3X : Verifier := { V3 with pinned := #[] }
#guard anchorUnpinned { V3X with marks := #[("a", anchorMark "x")], weightsOfRecord := some ((ofRecord PINS).1, OTHER) } ==
  #["x"]
-- another function's mark is no anchor
#guard anchorUnpinned { V3 with marks := #[("a", { anchorMark "w" with fn := "record.root/v1" })] } == #[]

/-! R7's hold on a pinned value's record entry (`offPin`): the record's entry of a value the registration of record pins
is that pin in root, leaf count, words and domain, and an anchor's value it pins has one; any other is refused. -/

def VX : Verifier := { V3 with marks := #[("a", anchorMark "x")], weightsOfRecord := some (ofRecord PINS) }
def recOf (es : Array Json) : Json := Json.mkObj [("registered", Json.arr es)]
def pinX : Json := (pin "x").2

#guard offPin (recOf #[pinX]) VX == #[]
#guard offPin (recOf #[]) VX == #["x"]
#guard offPin (recOf #[put pinX "root" (some (Json.str OTHER))]) VX == #["x"]
#guard offPin (recOf #[put pinX "domain" (some (Json.str OTHER))]) VX == #["x"]
#guard offPin (recOf #[put pinX "leaves" (some (Lean.toJson (99 : Nat)))]) VX == #["x"]
#guard offPin (recOf #[put pinX "words" (some (Lean.toJson (99 : Nat)))]) VX == #["x"]
-- the last entry of the value is the one read
#guard offPin (recOf #[pinX, put pinX "root" (some (Json.str OTHER))]) VX == #["x"]
-- a pinned value no anchor names may have no entry; a value the registration does not pin is not this check's
#guard offPin (recOf #[]) { VX with marks := #[] } == #[]
#guard offPin (recOf #[put pinX "root" (some (Json.str OTHER))]) { VX with marks := #[] } == #["x"]
#guard offPin (recOf #[]) { VX with weightsOfRecord := none } == #[]
#guard ANCHOR_FUNCTIONS.all fun fn => offPin (recOf #[]) { VX with marks := #[("a", anchorMark "x" fn)] } == #["x"]
-- the whole check names it: an entry equal to the pin but in its domain is refused `R7-pin`, the pin itself is not
#guard (codes VX (put · "registered" (some (Json.arr #[put pinX "domain" (some (Json.str OTHER))])))).contains "R7-pin"
#guard !(codes VX (put · "registered" (some (Json.arr #[pinX])))).contains "R7-pin"

/-! R7's hold on a digest anchor's width (`anchorWidth`): the record's entry of an anchor input is one row of its mark's
bits / 16 words, the width C's node gives, whatever the verifier's `registered` names. -/

def anchorEntry (n : String) (leaves words : Nat) : Json :=
  Json.mkObj [("value", .str n), ("root", .str ROOT), ("leaves", Lean.toJson leaves), ("words", Lean.toJson words),
    ("domain", .str OTHER)]
def MW : Array (String × Supplied.Mark) := #[("x/root", anchorMark "x"), ("x/domain", anchorMark "x" "registered.domain/v1")]
def VW : Verifier := { VX with marks := MW }

#guard anchorWidth (recOf #[anchorEntry "x/root" 1 32, anchorEntry "x/domain" 1 32]) MW == #[]
-- the same 512 bits in two rows of 16 words, or one row of 16: refused, each anchor by its input's name
#guard anchorWidth (recOf #[anchorEntry "x/root" 2 16, anchorEntry "x/domain" 2 16]) MW == #["x/root", "x/domain"]
#guard anchorWidth (recOf #[anchorEntry "x/root" 1 16]) MW == #["x/root"]
#guard anchorWidth (recOf #[anchorEntry "x/root" 1 32, anchorEntry "x/root" 2 16]) MW == #["x/root"]
-- no entry is `R7-values`' and `R7-pin`'s; another layout or function is not an anchor's width
#guard anchorWidth (recOf #[]) MW == #[]
#guard anchorWidth (recOf #[anchorEntry "x/root" 2 16]) #[("x/root", { anchorMark "x" with layout := "le" })] == #[]
#guard anchorWidth (recOf #[anchorEntry "x/root" 2 16]) #[("x/root", { anchorMark "x" with fn := "record.root/v1" })] == #[]
-- the width is the mark's bits: 256 bits make one row of 16 words
#guard anchorWidth (recOf #[anchorEntry "x/root" 1 16]) #[("x/root", { anchorMark "x" with bits := 256 })] == #[]
-- the whole check names it
#guard (codes VW (put · "registered" (some (Json.arr #[anchorEntry "x/root" 2 16])))).contains "R7-anchor-width"
#guard !(codes VW (put · "registered" (some (Json.arr #[anchorEntry "x/root" 1 32])))).contains "R7-anchor-width"

/-! R7's refusal of a supplied input with no source (`unsourced`, `R7-supplied-source`): `x`'s node a supplied input
and no source for it in the parameters' `supplied`. -/

def supX : String := (regX Spot.OneStageZk.E3).replace "RegisteredInput1_v1" "SuppliedInput1_v1"

#guard unsourced (verifier supX) == #["x"]
#guard unsourced { verifier supX with supplied := #[("x", #[])] } == #[]
#guard unsourced V3 == #[]
#guard (codes { verifier supX with pinned := #[] }).contains "R7-supplied-source"
-- C marks no supplied input here, and a record that registers nothing is refused all the same
#guard (verifier supX).marks.isEmpty
#guard (codes { verifier supX with pinned := #[], registered := #[] }).contains "R7-supplied-source"
#guard !(codes { verifier supX with pinned := #[], registered := #[], supplied := #[("x", #[])] }).contains
  "R7-supplied-source"
-- a table input's mark is no source: with no held table its rows have none, with the table pinned they do
#guard unsourced { verifier supX with supplied := #[("x", #[])], marks := #[("x", anchorMark "t")] } == #["x"]
def supT : Verifier := { verifier supX with supplied := #[("x", #[])], marks := #[("x", anchorMark "y")] }
#guard unsourced { supT with weightsOfRecord := some (ofRecord PINS) } == #[]

/-! A table named by a table input of C (`tableEntries`): its pin is R7's expected entry and the table inputs' source
(`heldTables`), never the record's entry. `t` is no parameter of C, pinned by the registration of record and named by
C's mark. -/

/-- A pin of the table `value`, four leaves of one word, its domain at the registration's program digest `ROOT`. -/
def tablePin (value : String) : String × Json :=
  let d := (Registered.domain ROOT value 1 4 Registered.WORD_BITS).toOption.map Flock.hexOf |>.getD ""
  (value, Json.mkObj [("value", Json.str value), ("root", Json.str ROOT), ("leaves", (4 : Nat)), ("words", (1 : Nat)),
    ("domain", Json.str d)])

def TDOMAIN : String := match getD (tablePin "t").2 "domain" with | .str d => d | _ => ""
def VT : Verifier := { V3 with marks := #[("a", anchorMark "t")], weightsOfRecord := some (ofRecord [tablePin "t"]) }

/-- The record's entries of `value`, each with key `k` set to `x`, or the entries without `value` for none. -/
def putValue (value k : String) (x : Option Json) (j : Json) : Json := match getD j "registered" with
  | .arr es => put j "registered" (some (Json.arr (match x with
      | some x => es.map fun e => if getD e "value" == Json.str value then put e k (some x) else e
      | none => es.filter fun e => getD e "value" != Json.str value)))
  | _ => j

/-- The record with `e` added to its entries. -/
def addEntry (e : Json) (j : Json) : Json := match getD j "registered" with
  | .arr es => put j "registered" (some (Json.arr (es.push e)))
  | _ => j

#guard TDOMAIN.length == 128
#guard ((tableEntries VT).map fun e => (e.value, e.leaves, e.words, e.domain)) == #[("t", 4, 1, TDOMAIN)]
#guard heldTables VT == #[("t", ROOT, TDOMAIN)]
-- a pin no mark names, a mark of a value nothing pins, a value already expected: no entry
#guard (tableEntries { VT with marks := #[] }).isEmpty
#guard (tableEntries { VT with marks := #[("a", anchorMark "z")] }).isEmpty
#guard (tableEntries { VT with marks := #[("a", anchorMark "x")], weightsOfRecord := some (ofRecord [tablePin "x"]) }).isEmpty
-- the table registered at its pin passes; left out, or at another root or domain, it is refused
#guard codes VT == #[]
#guard codes { VT with marks := #[("a", anchorMark "t"), ("b", anchorMark "t" "registered.domain/v1")] } == #[]
#guard (codes VT (putValue "t" "root" none)).contains "R7-values"
#guard (codes VT (putValue "t" "root" (some (Json.str OTHER)))).contains "R7-pin"
#guard (codes VT (putValue "t" "domain" (some (Json.str OTHER)))).contains "R7-pin"
-- with no pin of it the table input is refused
#guard (codes { VT with weightsOfRecord := none }).contains "R7-anchor-unpinned"
-- a pinned table that no mark of C names, registered in the record: refused, the pin widens nothing
#guard (codes { VT with marks := #[] } (addEntry (tablePin "t").2)).contains "R7-values"

end Spot.Test.Docket
