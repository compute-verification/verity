module

public import Flock.Lower
public import Flock.TmplAt
meta import Flock.Lower
meta import Flock.TmplAt

/-!
`Lower.lower` on PROTOCOL.md §9's Boolean program (`tests/verity/circuits/boolean_vectors.json`'s `program`, its
descriptor copied here): `ExampleAdd2_v1`, a 2-bit adder of a batch of `Xor2_v1` over an axis, two calls of a full adder,
a constant, a `Not_v1` and an `Xor3_v1`. Its template reads back through `Tmpl.parse` to the text it was written as, is
read as a templated unit (`TmplUnit.read`), holds a Definition for the full adder, one for the batched gate and the
root's, and on each of the 16 inputs its unrolled rows give the vectors' word: `x + y + 8`, bit 3 the `Not_v1` of 0.
-/

@[expose] public section

namespace Flock.Test.Lower

open Flock

def DESC : String := r#"{"annotations":{},"boolean":{},"callees":["Xor2_v1","And2_v1","Const1[0x0]_v1","ExampleFullAdder_v1","Not_v1","Xor3_v1","Input1_v1","ExampleAdd2_v1"],"definitions":{"And2_v1":{"form":"primitive","id":"And2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]},"Const1[0x0]_v1":{"form":"primitive","id":"Const1[0x0]_v1","params":[],"ret":["Value",1]},"ExampleAdd2_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"concat":[{"affine":["p",0,0,2]},{"affine":["p",1,0,2]}]},"type":2}],"axes":[1],"fn":0,"form":"batch","n":2,"out":0},{"args":[],"fn":2,"form":"primitive","out":1},{"args":[{"refs":{"affine":["p",0,0,1]},"type":1},{"refs":{"affine":["p",1,0,1]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":0},{"args":[{"refs":{"affine":["p",0,1,1]},"type":1},{"refs":{"affine":["p",1,1,1]},"type":1},{"refs":{"affine":["n",2,1,1]},"type":1}],"fn":3,"form":"call","out":0},{"args":[{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":4,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",2,0,1]},{"affine":["n",3,0,1]},{"affine":["n",3,1,1]}]},"type":3}],"fn":5,"form":"primitive","out":1}],"params":[["x",["Array",2,["Value",1]]],["y",["Array",2,["Value",1]]]],"ret":{"concat":[{"alias":[5,0]},{"affine":["n",4,0,1]}]},"ret_type":["Array",4,["Value",1]]},"form":"composite","id":"ExampleAdd2_v1"},"ExampleFullAdder_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"concat":[{"affine":["p",0,0,1]},{"affine":["p",1,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",2,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1},{"args":[{"refs":{"alias":[0,0]},"type":0}],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"alias":[1,0]},"type":0}],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",2,0,1]},{"affine":["n",3,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1}],"params":[["a",["Value",1]],["b",["Value",1]],["c",["Value",1]]],"ret":{"concat":[{"affine":["n",1,0,1]},{"affine":["n",4,0,1]}]},"ret_type":["Array",2,["Value",1]]},"form":"composite","id":"ExampleFullAdder_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[ExampleAdd2Program_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":6,"form":"batch","n":2,"out":0},{"args":[],"axes":[],"fn":6,"form":"batch","n":2,"out":0},{"args":[{"refs":{"affine":["n",0,0,2]},"type":0},{"refs":{"affine":["n",1,0,2]},"type":0}],"fn":7,"form":"call","out":4}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Array",4,["Value",1]]},"form":"composite","id":"Root[ExampleAdd2Program_v1]_v1"},"Xor2_v1":{"form":"primitive","id":"Xor2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]},"Xor3_v1":{"form":"primitive","id":"Xor3_v1","params":[["x",["Array",3,["Value",1]]]],"ret":["Value",1]}},"program_fn":"ExampleAdd2Program_v1","root":"Root[ExampleAdd2Program_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Array",2,["Value",1]],["Value",1],["Array",2,["Array",2,["Value",1]]],["Array",3,["Value",1]],["Array",4,["Value",1]]]}"#

/-- The vectors' words: `x`, `y`, and the program's return. -/
def WORDS : List (Nat × Nat × Nat) := [(0, 0, 8), (0, 1, 9), (0, 2, 10), (0, 3, 11), (1, 0, 9), (1, 1, 10), (1, 2, 11), (1, 3, 12), (2, 0, 10), (2, 1, 11), (2, 2, 12), (2, 3, 13), (3, 0, 11), (3, 1, 12), (3, 2, 13), (3, 3, 14)]

def prog : V Program.Prog := Program.decode DESC.toUTF8

def tmpl : V Tmpl.Tmpl := do
  let P ← prog
  let some d := P.index["ExampleAdd2_v1"]? | throw "no ExampleAdd2_v1"
  Lower.lower P d #[16] #[16]

/-! The unrolled rows' values, the test's reference. -/

/-- The unrolled template's columns on input bits `x`: the input rows `x`, every other written column `(A·z)(B·z)` in
column order (each reads earlier columns and the constant, the last column), the constant 1, and an unwritten column 0. A reference: no verifier runs it. -/
def solve (t : Tmpl.Tmpl) (x : Array Bool) : V (Array Bool) := do
  let acc ← Tmpl.Tmpl.rows t
  let k := t.constPos
  let sum := fun (z : Array Bool) (r : Array Nat) => r.foldl (fun b c => b != (c == k || z.getD c false)) false
  return (List.range acc.ra.size).foldl (fun z c =>
    z.push (if c < x.size then x[c]! else if c == k then true
      else if acc.set.getD c false then sum z acc.ra[c]! && sum z acc.rb[c]! else false)) #[]

/-- The output rows' bits on input bits `x`. -/
def outputs (t : Tmpl.Tmpl) (n : Nat) (x : Array Bool) : V (Array Bool) := do
  let z ← solve t x
  return (Array.range n).map fun k => z.getD (Lower.outBit t k) false

def bits (v n : Nat) : Array Bool := (Array.range n).map v.testBit

def word (bs : Array Bool) : Nat := bs.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- `render` of what `Tmpl.parse` reads back is the text. -/
def roundTrip : Bool := match tmpl with
  | .ok t => match Tmpl.parse (Lower.renderLines t) with
    | .ok t' => Lower.render t' == Lower.render t
    | .error _ => false
  | .error _ => false

#guard roundTrip

#guard match tmpl with
  | .ok t => match TmplUnit.read t .lenient with
    | .ok _ => true
    | .error _ => false
  | .error _ => false

-- the template's Definitions: the batched `Xor2_v1`, the full adder, `ExampleAdd2_v1`, the root
#guard match tmpl with
  | .ok t => t.defs.size == 4 && t.root == 3 && t.defs.all (·.calls.size ≤ 3)
  | .error _ => false

-- every input against the vectors
#guard match tmpl with
  | .ok t => WORDS.all fun (x, y, o) => match outputs t 4 (bits x 2 ++ bits y 2) with
    | .ok bs => word bs == o
    | .error _ => false
  | .error _ => false

/-! ## A scan

`ExampleScanAdd3_v1`: a 3-bit ripple adder as a `scan` of `ExampleAddStep_v1` (carry in, one bit of each word, returns the
carry and the sum bit) over `x` and `y`, its carry in a constant 0; it returns the three sum bits, then the carry out:
`x + y`. Its template is the step's Definition, the adder's (the scan as two `CALL`s: the first iteration on the constant,
then two more, each reading the previous one's carry) and the root's. -/

def SCAN_DESC : String := r#"{"annotations":{},"boolean":{},"callees":["Xor2_v1","And2_v1","Const1[0x0]_v1","ExampleAddStep_v1","Input1_v1","ExampleScanAdd3_v1"],"definitions":{"And2_v1":{"form":"primitive","id":"And2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]},"Const1[0x0]_v1":{"form":"primitive","id":"Const1[0x0]_v1","params":[],"ret":["Value",1]},"ExampleAddStep_v1":{"bindings":{},"form":"composite","id":"ExampleAddStep_v1","body":{"params":[["c",["Value",1]],["a",["Value",1]],["b",["Value",1]]],"nodes":[{"args":[{"refs":{"concat":[{"affine":["p",1,0,1]},{"affine":["p",2,0,1]}]},"type":1}],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",0,0,1]}]},"type":1}],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"concat":[{"affine":["p",1,0,1]},{"affine":["p",2,0,1]}]},"type":1}],"fn":1,"form":"primitive","out":0},{"args":[{"refs":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",0,0,1]}]},"type":1}],"fn":1,"form":"primitive","out":0},{"args":[{"refs":{"concat":[{"affine":["n",2,0,1]},{"affine":["n",3,0,1]}]},"type":1}],"fn":0,"form":"primitive","out":0}],"ret":{"concat":[{"affine":["n",4,0,1]},{"affine":["n",1,0,1]}]},"ret_type":["Tuple",[["Value",1],["Value",1]]]}},"ExampleScanAdd3_v1":{"bindings":{},"form":"composite","id":"ExampleScanAdd3_v1","body":{"params":[["x",["Array",3,["Value",1]]],["y",["Array",3,["Value",1]]]],"nodes":[{"args":[],"fn":2,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["p",0,0,3]},"type":2},{"refs":{"affine":["p",1,0,3]},"type":2}],"fn":3,"form":"scan","n":3,"n_xs":2,"out":4}],"ret":{"concat":[{"affine":["n",1,1,3]},{"affine":["n",1,0,1]}]},"ret_type":["Array",4,["Value",1]]}},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Root[ExampleScanAdd3Program_v1]_v1":{"bindings":{},"form":"composite","id":"Root[ExampleScanAdd3Program_v1]_v1","body":{"params":[],"nodes":[{"args":[],"axes":[],"fn":4,"form":"batch","n":3,"out":2},{"args":[],"axes":[],"fn":4,"form":"batch","n":3,"out":2},{"args":[{"refs":{"affine":["n",0,0,3]},"type":2},{"refs":{"affine":["n",1,0,3]},"type":2}],"fn":5,"form":"call","out":3}],"ret":{"affine":["n",2,0,4]},"ret_type":["Array",4,["Value",1]]}},"Xor2_v1":{"form":"primitive","id":"Xor2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]}},"program_fn":"ExampleScanAdd3Program_v1","root":"Root[ExampleScanAdd3Program_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2,["Value",1]],["Array",3,["Value",1]],["Array",4,["Value",1]],["Tuple",[["Value",1],["Array",3,["Value",1]]]],["Tuple",[["Value",1],["Value",1]]]]}"#

def scanTmpl : V Tmpl.Tmpl := do
  let P ← Program.decode SCAN_DESC.toUTF8
  let some d := P.index["ExampleScanAdd3_v1"]? | throw "no ExampleScanAdd3_v1"
  Lower.lower P d #[16] #[16]

#guard match scanTmpl with
  | .ok t => match Tmpl.parse (Lower.renderLines t) with
    | .ok t' => Lower.render t' == Lower.render t
    | .error _ => false
  | .error _ => false

#guard match scanTmpl with
  | .ok t => match TmplUnit.read t .lenient with
    | .ok _ => true
    | .error _ => false
  | .error _ => false

-- the step, the adder with the scan's two calls, the root
#guard match scanTmpl with
  | .ok t => t.defs.size == 3 && t.root == 2 && (t.defs[1]!).calls.size == 2 &&
      (t.defs[1]!).calls.map (·.count) == #[1, 2]
  | .error _ => false

-- every input: `x + y`
#guard match scanTmpl with
  | .ok t => (List.range 8).all fun x => (List.range 8).all fun y =>
      match outputs t 4 (bits x 3 ++ bits y 3) with
      | .ok bs => word bs == x + y
      | .error _ => false
  | .error _ => false

/-! ## A batch over listed references

`Perm_v1`: a batch of four `Xor2_v1` over axis 1 of `Array<2, Array<4, Value<1>>>`, member `m` reading references `m`
and `4 + m` of its argument, as a hand-written Boolean body's levels list theirs: all explicit, or affine and explicit
parts mixed. Its outputs on all 16 inputs are the XORs its references name, and its template Definition's `CALL`s are
the pieces, within `Lower.callBound`. -/

def xorDef : Program.Defn :=
  { key := "Xor2_v1", id := "Xor2_v1", prim := true, params := #[("x", .array 2 (.value 1))], ret := .value 1, gates := 1,
    paramLeaves := 2, outLeaves := 1 }

def pairsTy : Program.Ty := .array 2 (.array 4 (.value 1))

def permDef (ps : Array Program.Part) : Program.Defn :=
  { key := "Perm_v1", id := "Perm_v1", prim := false, params := #[("a", .array 4 (.value 1))], ret := .array 4 (.value 1),
    body := { params := #[("a", .array 4 (.value 1))], retType := .array 4 (.value 1),
              nodes := #[{ form := .batch, fn := 0, n := 4, axes := #[some 1], out := .array 4 (.value 1),
                           args := #[{ ty := pairsTy, seq := Program.Seq.ofParts ps, dims := pairsTy.axisDims 1 }] }],
              ret := Program.Seq.ofParts #[.affine false 0 0 4], offsets := #[] },
    gates := 4, paramLeaves := 4, outLeaves := 4 }

def permTmpl (ps : Array Program.Part) : V Tmpl.Tmpl :=
  Lower.lower { defs := #[xorDef, permDef ps], index := {}, root := 1, programFn := "", sha512 := ByteArray.empty } 1
    #[16] #[16]

def pa (j : Nat) : Program.Ref := ⟨true, 0, j⟩

def MIXED : Array Program.Part := #[.affine true 0 0 2, .explicit #[pa 3, pa 2], .affine true 0 0 4]
def LISTED : Array Program.Part := #[.explicit #[pa 1, pa 2, pa 3, pa 0, pa 0, pa 0, pa 1, pa 2]]

/-- Every input: member `m`'s output is the XOR of the bits its references `m` and `4 + m` name; and the template
Definition's `CALL`s number `calls`, within the bound. -/
def permOk (ps : Array Program.Part) (calls : Nat) : Bool := match permTmpl ps with
  | .ok t =>
    let refs := (Program.Seq.ofParts ps).toArray
    (t.defs.getD (t.root - 1) default).calls.size == calls && calls ≤ Lower.callBound (permDef ps) &&
    (List.range 16).all fun x => match outputs t 4 (bits x 4) with
      | .ok bs => (List.range 4).all fun m => bs[m]! == (x.testBit refs[m]!.j != x.testBit refs[4 + m]!.j)
      | .error _ => false
  | .error _ => false

-- the mixed argument: members 0 and 1 one stepping `CALL`, members 2 and 3 one each
#guard permOk MIXED 3
#guard permOk LISTED 4
-- a strided part is refused by name
#guard match permTmpl #[.strided true 0 0 4 0 1 8] with
  | .error e => e == "a batch argument stepping through strided or grid references"
  | .ok _ => false

-- a template the reader refuses is refused under the self-check's own name, not as a refusal of the input
#guard match tmpl with
  | .ok t => match Lower.selfCheck { t with root := t.defs.size } with
    | .error e => e.startsWith Lower.SELF_CHECK
    | .ok _ => false
  | .error _ => false

end Flock.Test.Lower
