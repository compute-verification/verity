module

public import Spot.Restrict
public import Spot.BlockFront
public import Spot.Edges
public import Flock.Test.Compose
public import Properties.Flock.Circuit.Chain
meta import Spot.Restrict
meta import Spot.BlockFront
meta import Spot.Edges
meta import Flock.Test.Compose
meta import Properties.Flock.Circuit.Chain

/-! Scratch, not for merge: a block whose template's return has a first field of 3 (or 2047) bits, through
`decode`, `front`, `restrict`, `portedOf`, `lift`, `lower` and `compose`, and the lowered rows on an honest and a wrong
output. Run: `git apply security/Proofs/Scratch/lakefile.patch && verity lean try security/Proofs/Scratch/FieldEval.lean`. -/

@[expose] public section

namespace Scratch.FieldEval

open Flock Flock.Program

def D_field : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",3,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,3]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":3,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,3]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Tuple",[["Array",3,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",3,["Value",1]],["Tuple",[["Array",3,["Value",1]],["Value",1]]]]}"#

def D_fieldbig : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2047,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,2047]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",2047,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2047,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,2047]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2048]},"ret_type":["Tuple",[["Array",2047,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2047,["Value",1]],["Tuple",[["Array",2047,["Value",1]],["Value",1]]]]}"#

/-- (x, Not y) at x of 2048 bits: the 1-bit `y` and `Not y` are each the LAST parameter and return field. -/
def D_last : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2048,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,2048]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",2048,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2048,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,2048]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2049]},"ret_type":["Tuple",[["Array",2048,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2048,["Value",1]],["Tuple",[["Array",2048,["Value",1]],["Value",1]]]]}"#

/-- E(x : Array 2047 (Value 1)) = x with its last bit negated: one parameter and one return field, alone. -/
def D_one : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,2046,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2047,["Value",1]]]],"ret":{"concat":[{"affine":["p",0,0,2046]},{"affine":["n",0,0,1]}]},"ret_type":["Array",2047,["Value",1]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2047,"out":1},{"args":[{"refs":{"affine":["n",0,0,2047]},"type":1}],"fn":2,"form":"call","out":1}],"params":[],"ret":{"affine":["n",1,0,2047]},"ret_type":["Array",2047,["Value",1]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2047,["Value",1]]]}"#

def show? {α} (f : α → String) : V α → String
  | .ok a => "ok " ++ f a
  | .error e => "REFUSED: " ++ e

def ports (ps : Array Spot.BlockFront.Port) : String := toString (ps.toList.map fun p => (p.gate, p.width))

/-- Every written column of `t`'s unrolled rows equals its row's `(A·z)(B·z)`, and the constant is 1. -/
def rowsOk (t : Tmpl.Tmpl) (z : Array Bool) : V Bool := do
  let acc ← Tmpl.Tmpl.rows t
  let k := t.constPos
  let sum := fun (r : Array Nat) => r.foldl (fun b c => b != (c == k || z.getD c false)) false
  return z.getD k false && (List.range acc.ra.size).all fun c =>
    !(acc.set.getD c false) || z.getD c false == (sum acc.ra[c]! && sum acc.rb[c]!)

/-- `solve` on the input bits `x`, then the output port bits overwritten by `o` (bit `k` at `outBit t k`). -/
def assign (t : Tmpl.Tmpl) (x o : Array Bool) : V (Array Bool) := do
  let z ← Test.Lower.solve t x
  let z := z ++ Array.replicate (t.constPos + 2) false
  return (List.range o.size).foldl (fun z k => z.set! (Lower.outBit t k) o[k]!) z

def bitsOf (v n : Nat) : Array Bool := Test.Lower.bits v n

/-- The door's path step by step on `d`, units `us`. -/
def path (d : String) (us : Array Nat) : List String :=
  match decode d.toUTF8 with
  | .error e => ["decode REFUSED: " ++ e]
  | .ok P =>
    let fr := Spot.BlockFront.front P us
    let S := Spot.Restrict.restrict P us
    let Q := Spot.Restrict.portedOf P us
    let L := Q >>= fun Q => Lift.lift Q Q.root
    let lowChain := do
      let f ← fr; let Q ← Q
      Lower.lower Q Q.root (f.ins.map (·.width)) (f.outs.map (·.width))
    let key := S >>= fun S => do
      let some r := S.defs[S.root]? | throw "no root"
      let ks := (List.range r.body.nodes.size).filter (fun k => !Spot.Edges.isInputNode S k) |>.map fun k =>
        (S.defs[(r.body.nodes[k]!).fn]!).key
      pure ks
    ["decode ok: defs " ++ toString (P.defs.toList.map (·.key)),
     "front: " ++ show? (fun f => s!"ranges {f.ranges} ins {ports f.ins} outs {ports f.outs}") fr,
     "restrict: " ++ show? (fun S => s!"root nodes {((S.defs[S.root]?).map (·.body.nodes.size))}") S,
     "restrict's non-input callees: " ++ show? toString key,
     "portedOf: " ++ show? (fun Q => s!"root paramLeaves {(Q.defs[Q.root]!).paramLeaves} outLeaves {(Q.defs[Q.root]!).outLeaves}") Q,
     "lift: " ++ show? (fun (L, e) => s!"root {(L.defs[e]!).key} paramLeaves {(L.defs[e]!).paramLeaves} outLeaves {(L.defs[e]!).outLeaves} params {(L.defs[e]!).params.toList.map (·.2.leaves)}") L,
     "lower (Chain's form: portedOf's root at the front's widths): " ++ show? (fun t => s!"inGroups {t.inGroups.toList.map (·.bits)} outGroups {t.outGroups.toList.map (·.bits)}") lowChain]

/-- `lowerBlock`'s lowering (`Spot.OneStageZkStatement.lowerBlock`, no cuts): the template `C_v1` of `restrict P us`,
lowered at its parameters' bits in 16-bit words and its return fields in whole words, and composed. -/
def realLower (d key : String) (us : Array Nat) : V (Tmpl.Tmpl × Compose.Block) := do
  let P ← decode d.toUTF8
  let S ← Spot.Restrict.restrict P us
  let some i := S.index[key]? | throw s!"no {key} in S"
  let fn := S.defs[i]!
  let bits := fun (t : Ty) => t.widths.foldl (· + ·) 0
  let ins := fn.params.map fun (n, t) => (n, bits t)
  let outs := (fieldsOf fn.ret).toArray.mapIdx fun j f => (s!"o{j}", (bits f + 15) / 16)
  let nIn := ins.foldl (fun a p => a + (p.2 + 15) / 16) 0
  let nOut := outs.foldl (· + ·.2) 0
  let t ← Lower.lower S i (Array.replicate nIn 16) (Array.replicate nOut 16)
  pure (t, { name := s!"unit/{key}", body := .tmpl t "iv/hybrid/4", ins, outs, programDigests := [(key, "".pushn 'a' 128)] })

def realReport (d key : String) (us : Array Nat) (x : Array Bool) (oks : List (String × Array Bool)) : List String :=
  match realLower d key us with
  | .error e => ["lowerBlock's lower REFUSED: " ++ e]
  | .ok (t, b) =>
    let composed := match Test.Compose.read b with
      | .ok c => match Test.Compose.agrees b c with
        | .ok true => "compose + door's reader: ok, reads back as the block"
        | .ok false => "compose + door's reader: read, NOT the block"
        | .error e => "agrees REFUSED: " ++ e
      | .error e => "compose/reader REFUSED: " ++ e
    [s!"lowerBlock: ins {b.ins} outs {b.outs} inGroups {t.inGroups.toList.map (·.bits)} outGroups {t.outGroups.toList.map (·.bits)}",
     composed] ++ oks.map fun (name, o) => match assign t x o >>= rowsOk t with
      | .ok r => s!"{name}: every row satisfied = {r}"
      | .error e => s!"{name}: rows REFUSED {e}"

#eval path D_field #[0]
-- x = 5 (3 bits, word 0), y = 0 (word 1): C's (o0, o1) = (5, Not y = 1)
#eval realReport D_field "E_v1" #[0] (bitsOf 5 16 ++ bitsOf 0 16)
  [("honest (o0, o1) = (5, 1)", bitsOf 5 16 ++ bitsOf 1 16), ("WRONG (o0, o1) = (5, 0)", bitsOf 5 16 ++ bitsOf 0 16),
   ("WRONG (o0, o1) = (13, 0): Not y at o0's bit 3", bitsOf 13 16 ++ bitsOf 0 16), ("(o0, o1) = (13, 1)", bitsOf 13 16 ++ bitsOf 1 16)]
-- y = 1: C's (o0, o1) = (5, 0)
#eval realReport D_field "E_v1" #[0] (bitsOf 5 16 ++ bitsOf 1 16)
  [("honest (o0, o1) = (5, 0)", bitsOf 5 16 ++ bitsOf 0 16), ("WRONG (o0, o1) = (5, 1)", bitsOf 5 16 ++ bitsOf 1 16)]
-- 2047 bits, so it composes: x = 5, y = 0; C's (o0, o1) = (5, 1)
#eval realReport D_fieldbig "E_v1" #[0] (bitsOf 5 2048 ++ bitsOf 0 16)
  [("honest (o0, o1) = (5, 1)", bitsOf 5 2048 ++ bitsOf 1 16), ("WRONG (o0, o1) = (5, 0)", bitsOf 5 2048 ++ bitsOf 0 16),
   ("WRONG (o0, o1) = (5 + 2^2047, 0): Not y at o0's bit 2047", bitsOf (5 + 2^2047) 2048 ++ bitsOf 0 16),
   ("(o0, o1) = (5 + 2^2047, 1)", bitsOf (5 + 2^2047) 2048 ++ bitsOf 1 16)]

/-- The lowered rows of `d`'s `key` against the honest output `o` on input `x`: whether it satisfies; each output bit
flipped alone (those NOT rejected listed, padding included: index `i` is the port's bit `i`); and each input bit at
`inPad` set to 1 with `o` kept (those still satisfied are padding the rows leave free). -/
def flipReport (d key : String) (x o : Array Bool) (inPad : List Nat) : List String :=
  match realLower d key #[0] with
  | .error e => ["lowerBlock's lower REFUSED: " ++ e]
  | .ok (t, b) =>
    let composed := match Test.Compose.read b with
      | .ok c => match Test.Compose.agrees b c with
        | .ok true => "compose + door's reader: ok, reads back as the block"
        | .ok false => "compose + door's reader: read, NOT the block"
        | .error e => "agrees REFUSED: " ++ e
      | .error e => "compose/reader REFUSED: " ++ e
    match Tmpl.Tmpl.rows t with
    | .error e => ["rows REFUSED: " ++ e]
    | .ok acc =>
      let k := t.constPos
      let sum := fun (z : Array Bool) (r : Array Nat) => r.foldl (fun b c => b != (c == k || z.getD c false)) false
      let ok := fun (z : Array Bool) => z.getD k false && (List.range acc.ra.size).all fun c =>
        !(acc.set.getD c false) || z.getD c false == (sum z acc.ra[c]! && sum z acc.rb[c]!)
      let put := fun (x : Array Bool) => match Test.Lower.solve t x with
        | .ok z => (List.range o.size).foldl (fun z i => z.set! (Lower.outBit t i) o[i]!)
            (z ++ Array.replicate (t.constPos + 2) false)
        | .error _ => #[]
      let zh := put x
      let notRej := (List.range o.size).filter fun i => ok (zh.set! (Lower.outBit t i) (!o[i]!))
      let freeIn := inPad.filter fun i => ok (put (x.set! i true))
      [s!"lowerBlock: ins {b.ins} outs {b.outs}", composed, s!"honest satisfies every row: {ok zh}",
       s!"output port bits {o.size}; flipped alone and NOT rejected: {notRej}",
       s!"input padding bits {inPad} set to 1, rows still satisfied (free): {freeIn}"]

-- the last field: x = 5 (2048 bits), y = 0 (word 128); C's (o0, o1) = (5, 1); o1's padding is the port's bits 2049..2063
#eval flipReport D_last "E_v1" (bitsOf 5 2048 ++ bitsOf 0 16) (bitsOf 5 2048 ++ bitsOf 1 16) (List.range' 2049 15)
-- y = 1: C's (o0, o1) = (5, 0)
#eval flipReport D_last "E_v1" (bitsOf 5 2048 ++ bitsOf 1 16) (bitsOf 5 2048 ++ bitsOf 0 16) (List.range' 2049 15)
-- one field alone: x = 5 + 2^2046 (2047 bits); C's output 5 (bit 2046 negated); the port's bit 2047 is padding
#eval flipReport D_one "E_v1" (bitsOf (5 + 2^2046) 2048) (bitsOf 5 2048) [2047]

end Scratch.FieldEval
