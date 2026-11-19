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

/-! Scratch, not for merge: a block whose callee has a word parameter and reaches no word primitive, through
`decode`, `front`, `restrict`, `portedOf`, `lift`, `lower` and `compose`, and the lowered rows on an honest and a wrong
output. Run: `git apply security/Proofs/Scratch/lakefile.patch && verity lean try security/Proofs/Scratch/WordPassEval.lean`. -/

@[expose] public section

namespace Scratch.WordPassEval

open Flock Flock.Program

def D_pass : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,1]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#
def D_unused : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":1}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",1]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#
def D_passbatch : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,1]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Value",32],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2,"out":3},{"args":[],"axes":[],"fn":1,"form":"batch","n":2,"out":4},{"args":[{"refs":{"affine":["n",0,0,2]},"type":3},{"refs":{"affine":["n",1,0,2]},"type":4}],"axes":[0,0],"fn":3,"form":"batch","n":2,"out":5}],"params":[],"ret":{"affine":["n",2,0,4]},"ret_type":["Array",2,["Tuple",[["Value",32],["Value",1]]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#
def D_unusedbatch : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Value",32]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2,"out":3},{"args":[],"axes":[],"fn":1,"form":"batch","n":2,"out":4},{"args":[{"refs":{"affine":["n",0,0,2]},"type":3},{"refs":{"affine":["n",1,0,2]},"type":4}],"axes":[0,0],"fn":3,"form":"batch","n":2,"out":4}],"params":[],"ret":{"affine":["n",2,0,2]},"ret_type":["Array",2,["Value",1]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#
def D_prim : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","TAdd32_v1"],"definitions":{"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"fn":0,"form":"primitive","out":0},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,1]},"type":0},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":3,"form":"call","out":0}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",32]},"form":"composite","id":"Root[CProg_v1]_v1"},"TAdd32_v1":{"form":"primitive","id":"TAdd32_v1","params":[["a",["Value",32]],["b",["Value",32]]],"ret":["Value",32]}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Tuple",[["Value",32],["Value",1]]],["Array",2,["Value",32]],["Array",2,["Value",1]],["Array",2,["Tuple",[["Value",32],["Value",1]]]]]}"#
def D_bigunused : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Array",64,["Value",32]]],["b",["Value",1]]],"ret":{"affine":["n",0,0,1]},"ret_type":["Value",1]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":64,"out":2},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,64]},"type":2},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":1}],"params":[],"ret":{"affine":["n",2,0,1]},"ret_type":["Value",1]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Array",64,["Value",32]]]}"#
def D_bigpass : String := r#"{"annotations":{},"boolean":{},"callees":["Input32_v1","Input1_v1","Not_v1","C_v1"],"definitions":{"C_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":1}],"fn":2,"form":"primitive","out":1}],"params":[["a",["Array",64,["Value",32]]],["b",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,64]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",64,["Value",32]],["Value",1]]]},"form":"composite","id":"C_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Input32_v1":{"form":"primitive","id":"Input32_v1","params":[],"ret":["Value",32]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[CProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":64,"out":2},{"args":[],"fn":1,"form":"primitive","out":1},{"args":[{"refs":{"affine":["n",0,0,64]},"type":2},{"refs":{"affine":["n",1,0,1]},"type":1}],"fn":3,"form":"call","out":3}],"params":[],"ret":{"affine":["n",2,0,65]},"ret_type":["Tuple",[["Array",64,["Value",32]],["Value",1]]]},"form":"composite","id":"Root[CProg_v1]_v1"}},"program_fn":"CProg_v1","root":"Root[CProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",32],["Value",1],["Array",64,["Value",32]],["Tuple",[["Array",64,["Value",32]],["Value",1]]]]}"#

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

#eval path D_pass #[0]
#eval path D_unused #[0]
#eval path D_passbatch #[0]
#eval path D_unusedbatch #[0]
#eval path D_prim #[0]

/-- `lowerBlock`'s lowering (`Spot.OneStageZkStatement.lowerBlock`, no cuts): the template `C_v1` of `restrict P us`,
lowered at its parameters' bits in 16-bit words and its return fields in whole words, and composed. -/
def realLower (d : String) (us : Array Nat) : V (Tmpl.Tmpl × Compose.Block) := do
  let P ← decode d.toUTF8
  let S ← Spot.Restrict.restrict P us
  let some i := S.index["C_v1"]? | throw "no C_v1 in S"
  let fn := S.defs[i]!
  let bits := fun (t : Ty) => t.widths.foldl (· + ·) 0
  let ins := fn.params.map fun (n, t) => (n, bits t)
  let outs := (fieldsOf fn.ret).toArray.mapIdx fun j f => (s!"o{j}", (bits f + 15) / 16)
  let nIn := ins.foldl (fun a p => a + (p.2 + 15) / 16) 0
  let nOut := outs.foldl (· + ·.2) 0
  let t ← Lower.lower S i (Array.replicate nIn 16) (Array.replicate nOut 16)
  pure (t, { name := "unit/C_v1", body := .tmpl t "iv/hybrid/4", ins, outs, programDigests := [("C_v1", "".pushn 'a' 128)] })

def realReport (d : String) (us : Array Nat) (x : Array Bool) (oks : List (String × Array Bool)) : List String :=
  match realLower d us with
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

-- a = 1 (bit 0 set, bit 1 clear), b = 1: C's o0 is `Not b` = 0
#eval realReport D_unused #[0] (bitsOf 1 32 ++ bitsOf 1 16)
  [("honest o0 = Not b = 0", bitsOf 0 16), ("WRONG o0 = 1", bitsOf 1 16)]
-- pass-through: C's (o0, o1) = (a, Not b) = (1, 0)
#eval realReport D_pass #[0] (bitsOf 1 32 ++ bitsOf 1 16)
  [("honest (o0, o1) = (1, 0)", bitsOf 1 32 ++ bitsOf 0 16), ("WRONG (o0, o1) = (3, 1)", bitsOf 3 32 ++ bitsOf 1 16),
   ("WRONG (o0, o1) = (3, 0)", bitsOf 3 32 ++ bitsOf 0 16)]
-- a = 2 (bit 1 set), b = 0: unused, C's o0 = 1; honest
#eval realReport D_unused #[0] (bitsOf 2 32 ++ bitsOf 0 16)
  [("honest o0 = Not b = 1", bitsOf 1 16), ("WRONG o0 = 0", bitsOf 0 16)]

#eval path D_bigunused #[0]
#eval path D_bigpass #[0]
-- a[0] = 1 (bit 0 set, bit 1 clear), the other words 0, b = 1: C's o0 = Not b = 0
#eval realReport D_bigunused #[0] (bitsOf 1 2048 ++ bitsOf 1 16)
  [("honest o0 = Not b = 0", bitsOf 0 16), ("WRONG o0 = 1", bitsOf 1 16)]
#eval realReport D_bigpass #[0] (bitsOf 1 2048 ++ bitsOf 1 16)
  [("honest (o0, o1) = (a, Not b) = (1, 0)", bitsOf 1 2048 ++ bitsOf 0 16),
   ("WRONG (o0, o1) = (1 + 2^64, 1)", bitsOf (1 + 2^64) 2048 ++ bitsOf 1 16)]

end Scratch.WordPassEval
