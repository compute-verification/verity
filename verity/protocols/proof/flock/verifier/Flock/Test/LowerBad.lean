module

public import Flock.Lower
public import Flock.Test.Lower
meta import Flock.Lower
meta import Flock.Test.Lower

/-!
`Lower.lower` on a one-gate Definition built as a `Prog`, not decoded: refused where the program is one `Program.decode`
refuses (a return shorter than its type, a reference past its source's leaves), and on the honest Definition the
template computes it (`¬a`).
-/

@[expose] public section

namespace Flock.Test.LowerBad

open Flock Flock.Program Flock.Test.Lower

def notDef : Defn :=
  { key := "Not_v1", id := "Not_v1", prim := true, params := #[("a", .value 1)], ret := .value 1, gates := 1, paramLeaves := 1, outLeaves := 1 }

def node (r : Part) : Node := { form := .prim, fn := 0, args := #[{ ty := .value 1, seq := Seq.ofParts #[r] }], out := .value 1 }

def bad (outLeaves : Nat) (r : Part) (ret : Part) : Defn :=
  { key := "Bad_v1", id := "Bad_v1", prim := false, params := #[("a", .value 1)], ret := .array outLeaves (.value 1),
    body := { params := #[("a", .value 1)], retType := .array outLeaves (.value 1), nodes := #[node r],
              ret := Seq.ofParts #[ret], offsets := #[] }, gates := 1, paramLeaves := 1, outLeaves }

def prog (b : Defn) : Prog := { defs := #[notDef, b], index := {}, root := 1, programFn := "", sha512 := ByteArray.empty }

-- a return of 1 leaf under a type of 2
#guard (Lower.lower (prog (bad 2 (.affine true 0 0 1) (.affine false 0 0 1))) 1 #[16] #[16]).toOption.isNone
-- a parameter reference past the parameters' leaves
#guard match Lower.lower (prog (bad 1 (.affine true 0 3 1) (.affine false 0 0 1))) 1 #[16] #[16] with
  | .error e => e.startsWith Lower.PAST
  | .ok _ => false
-- a node reference past its leaves
#guard match Lower.lower (prog (bad 1 (.affine true 0 0 1) (.affine false 0 1 1))) 1 #[16] #[16] with
  | .error e => e.startsWith Lower.PAST
  | .ok _ => false
-- the same Definition, honest: `¬a`
#guard match Lower.lower (prog (bad 1 (.affine true 0 0 1) (.affine false 0 0 1))) 1 #[16] #[16] with
  | .ok t => (List.range 2).all fun x => (outputs t 1 (bits x 1)).toOption == some #[x == 0]
  | .error _ => false
-- the same Definition reading an input primitive, the one primitive the lift passes that is no gate (`gateOf?`)
#guard
  let b := bad 1 (.affine true 0 0 1) (.affine false 0 0 1)
  let input := { notDef with key := "Input1_v1", id := "Input1_v1", params := #[], gates := 0, paramLeaves := 0 }
  let b := { b with body := { b.body with nodes := #[{ node (.affine true 0 0 1) with args := #[] }] } }
  match Lower.lower { prog b with defs := #[input, b] } 1 #[16] #[16] with
  | .error e => e == "Input1_v1: not a Boolean gate (PROTOCOL.md §9)"
  | .ok _ => false

/-! ## Two scans `Work.scan` refuses

`SCAN_DESC`'s adder with one Definition changed after decoding, so these reach `Lower.lower` only built by hand:
`Program.decode` refuses both first. -/

/-- `SCAN_DESC`'s `ExampleScanAdd3_v1` lowered, Definition `id` changed by `f`. -/
def scanWith (id : String) (f : Defn → Defn) : V Tmpl.Tmpl := do
  let P ← Program.decode SCAN_DESC.toUTF8
  let some i := P.index[id]? | throw s!"no {id}"
  let some d := P.index["ExampleScanAdd3_v1"]? | throw "no ExampleScanAdd3_v1"
  Lower.lower { P with defs := P.defs.modify i f } d #[16] #[16]

/-- The adder's scan (its node 1) with no arguments: no init for its carry. -/
def noInit (d : Defn) : Defn :=
  { d with body := { d.body with nodes := d.body.nodes.modify 1 fun x => { x with args := #[] } } }

/-- The step's carry three leaves wide; it still returns two. -/
def wideCarry (d : Defn) : Defn :=
  let wide := fun (ps : Array (String × Ty)) => ps.modify 0 fun p => (p.1, .array 3 (.value 1))
  { d with params := wide d.params, body := { d.body with params := wide d.body.params }, paramLeaves := d.paramLeaves + 2 }

#guard match scanWith "ExampleScanAdd3_v1" noInit with
  | .error e => e.endsWith "ExampleAddStep_v1: a scan without its carry's init"
  | .ok _ => false

#guard match scanWith "ExampleAddStep_v1" wideCarry with
  | .error e => e.endsWith "ExampleAddStep_v1: a scan whose callee returns fewer leaves than its carry"
  | .ok _ => false

/-! ## Word-typed leaves `Lower.WIDE` refuses

`Lower.lower` on Definitions with a word-typed leaf that `Lift.lift` leaves as they are, since they reach no word
primitive: `C_v1 (a : Value 32) (b : Value 1)`, returning `Not b` (`a` unused) or `(a, Not b)` (`a` passed through),
called once or as a batch of two, and `D_v1 a b = (W32 a, C_v1 a b)`, which reaches a word primitive and is lifted while
its callee `C_v1` is not. Each is refused by name (`Lower.WIDE`): read as one column, a word leaf leaves the rows reading
`a`'s bit 1 for `b` and its other bits free, so a wrong output satisfies them.
-/
def UNUSED : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":1}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",1]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

def PASS : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,1]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

def UNUSEDBATCH : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2,"out":3},{"args":[],"axes":[],"fn":1,"form":"batch","n":2,"out":4},{"args":[{"refs":{"affine":["n",0,0,2]},"type":3},{"refs":{"affine":["n",1,0,2]},"type":4}],"axes":[0,0],"fn":3,"form":"batch","n":2,"out":4}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Array",2,["Value",1]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

def PASSBATCH : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,1]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2,"out":3},{"args":[],"axes":[],"fn":1,"form":"batch","n":2,"out":4},{"args":[{"refs":{"affine":["n",0,0,2]},"type":3},{"refs":{"affine":["n",1,0,2]},"type":4}],"axes":[0,0],"fn":3,"form":"batch","n":2,"out":5}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Array",2,["Tuple",[["Value",32],["Value",1]]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#

def INSIDE : String := r#"{"annotations":{},"boolean":{"W32_v1":"WB_v1"},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1","W32_v1","D_v1","WB_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"D_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":4,"form":"primitive","out":0},{"args":[{"refs":{"affine":["p",0,0,1]},"type":0},{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":3,"form":"call","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["n",1,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"D_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[DProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":5,"form":"call","out":3}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"Root[DProg_v1]_v1"},"W32_v1":{"form":"primitive","id":"W32_v1","params":[["a",["Value",32]]],"ret":["Value",32]},"WB_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,32]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":32,"out":2}],"params":[["x",["Array",32,["Value",1]]]],"ret":{"affine":["n",0,0,32]},"ret_type":["Array",32,["Value",1]]},"form":"composite","id":"WB_v1"}},"program_fn":"DProg_v1","root":"Root[DProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Array",32,["Value",1]],["Tuple",[["Value",32],["Value",1]]]]}"#

/-- Definition `id` of descriptor `d` lowered at `nIn` input and `nOut` output words of 16 bits, as the door's block
circuit lowers its template. -/
def lowerAt (d id : String) (nIn nOut : Nat) : V Tmpl.Tmpl := do
  let P ← Program.decode d.toUTF8
  let some i := P.index[id]? | throw s!"no {id}"
  Lower.lower P i (Array.replicate nIn 16) (Array.replicate nOut 16)

def refusedWide (t : V Tmpl.Tmpl) : Bool := match t with
  | .error e => e == s!"C_v1: {Lower.WIDE}"
  | .ok _ => false

#guard refusedWide (lowerAt UNUSED "C_v1" 3 1)
#guard refusedWide (lowerAt PASS "C_v1" 3 3)
#guard refusedWide (lowerAt UNUSEDBATCH "C_v1" 3 1)
#guard refusedWide (lowerAt PASSBATCH "C_v1" 3 3)
-- lifted whole where it reaches a word primitive: `D_v1` on 33 bits, its callee `C_v1` left on words, and refused there
#guard match Program.decode INSIDE.toUTF8 with
  | .ok P => match P.index["D_v1"]? with
    | some i => match Lift.lift P i with
      | .ok (Q, j) => (Q.defs[j]!).paramLeaves == 33 &&
          ((Q.defs[j]!).body.nodes[1]?.map fun x => (Q.defs[x.fn]!).key) == some "C_v1"
      | .error _ => false
    | none => false
  | .error _ => false
#guard refusedWide (lowerAt INSIDE "D_v1" 3 3)

end Flock.Test.LowerBad
