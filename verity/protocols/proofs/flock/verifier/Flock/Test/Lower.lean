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

def DESC : String := r#"{"annotations":{},"callees":["Xor2_v1","And2_v1","Const1[0x0]_v1","ExampleFullAdder_v1","Not_v1","Xor3_v1","Input1_v1","ExampleAdd2_v1"],"definitions":{"And2_v1":{"form":"primitive","id":"And2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]},"Const1[0x0]_v1":{"form":"primitive","id":"Const1[0x0]_v1","params":[],"ret":["Value",1]},"ExampleAdd2_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"concat":[{"affine":["p",0,0,2]},{"affine":["p",1,0,2]}]},"type":2}],"axes":[1],"fn":0,"form":"batch","n":2,"out":0},{"args":[],"fn":2,"form":"primitive","out":1},{"args":[{"refs":{"affine":["p",0,0,1]},"type":1},{"refs":{"affine":["p",1,0,1]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":0},{"args":[{"refs":{"affine":["p",0,1,1]},"type":1},{"refs":{"affine":["p",1,1,1]},"type":1},{"refs":{"affine":["n",2,1,1]},"type":1}],"fn":3,"form":"call","out":0},{"args":[{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":4,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",2,0,1]},{"affine":["n",3,0,1]},{"affine":["n",3,1,1]}]},"type":3}],"fn":5,"form":"primitive","out":1}],"params":[["x",["Array",2,["Value",1]]],["y",["Array",2,["Value",1]]]],"ret":{"concat":[{"alias":[5,0]},{"affine":["n",4,0,1]}]},"ret_type":["Array",4,["Value",1]]},"form":"composite","id":"ExampleAdd2_v1"},"ExampleFullAdder_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"concat":[{"affine":["p",0,0,1]},{"affine":["p",1,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",2,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1},{"args":[{"refs":{"alias":[0,0]},"type":0}],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"alias":[1,0]},"type":0}],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"concat":[{"affine":["n",2,0,1]},{"affine":["n",3,0,1]}]},"type":0}],"fn":0,"form":"primitive","out":1}],"params":[["a",["Value",1]],["b",["Value",1]],["c",["Value",1]]],"ret":{"concat":[{"affine":["n",1,0,1]},{"affine":["n",4,0,1]}]},"ret_type":["Array",2,["Value",1]]},"form":"composite","id":"ExampleFullAdder_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[ExampleAdd2Program_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":6,"form":"batch","n":2,"out":0},{"args":[],"axes":[],"fn":6,"form":"batch","n":2,"out":0},{"args":[{"refs":{"affine":["n",0,0,2]},"type":0},{"refs":{"affine":["n",1,0,2]},"type":0}],"fn":7,"form":"call","out":4}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Array",4,["Value",1]]},"form":"composite","id":"Root[ExampleAdd2Program_v1]_v1"},"Xor2_v1":{"form":"primitive","id":"Xor2_v1","params":[["x",["Array",2,["Value",1]]]],"ret":["Value",1]},"Xor3_v1":{"form":"primitive","id":"Xor3_v1","params":[["x",["Array",3,["Value",1]]]],"ret":["Value",1]}},"program_fn":"ExampleAdd2Program_v1","root":"Root[ExampleAdd2Program_v1]_v1","schema":"verity-ir/descriptor/v1","types":[["Array",2,["Value",1]],["Value",1],["Array",2,["Array",2,["Value",1]]],["Array",3,["Value",1]],["Array",4,["Value",1]]]}"#

/-- The vectors' words: `x`, `y`, and the program's return. -/
def WORDS : List (Nat × Nat × Nat) := [(0, 0, 8), (0, 1, 9), (0, 2, 10), (0, 3, 11), (1, 0, 9), (1, 1, 10), (1, 2, 11), (1, 3, 12), (2, 0, 10), (2, 1, 11), (2, 2, 12), (2, 3, 13), (3, 0, 11), (3, 1, 12), (3, 2, 13), (3, 3, 14)]

def prog : V Program.Prog := Program.decode DESC.toUTF8

def tmpl : V Tmpl.Tmpl := do
  let P ← prog
  let some d := P.index["ExampleAdd2_v1"]? | throw "no ExampleAdd2_v1"
  Lower.lower P d #[16] #[16]

/-! The unrolled rows' values, the test's reference. -/

/-- The unrolled template's columns on input bits `x`: the input rows `x`, every other written column `(A·z)(B·z)` in
column order (each reads earlier columns), the constant 1, and an unwritten column 0. A reference: no verifier runs it. -/
def solve (t : Tmpl.Tmpl) (x : Array Bool) : V (Array Bool) := do
  let acc ← Tmpl.Tmpl.rows t
  let k := t.constPos
  let sum := fun (z : Array Bool) (r : Array Nat) => r.foldl (fun b c => b != z.getD c false) false
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

end Flock.Test.Lower
