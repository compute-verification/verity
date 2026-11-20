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

/-! Scratch, not for merge: a return field and a parameter not whole u16 words that are the LAST (and one alone),
through `lowerBlock`'s lowering, the rows against the honest output, each output bit flipped, and the padding columns.
Compose is left out (`FieldEval` composes the same size class). Run: `git apply security/Proofs/Scratch/lakefile.patch &&
verity lean try security/Proofs/Scratch/LastFieldEval.lean`. -/

@[expose] public section

namespace Scratch.LastFieldEval

open Flock Flock.Program

/-- (x, Not y) at x of 2048 bits: the 1-bit `y` and `Not y` are each the LAST parameter and return field. -/
def D_last : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",1,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2048,["Value",1]]],["y",["Value",1]]],"ret":{"concat":[{"affine":["p",0,0,2048]},{"affine":["n",0,0,1]}]},"ret_type":["Tuple",[["Array",2048,["Value",1]],["Value",1]]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2048,"out":1},{"args":[],"fn":0,"form":"primitive","out":0},{"args":[{"refs":{"affine":["n",0,0,2048]},"type":1},{"refs":{"affine":["n",1,0,1]},"type":0}],"fn":2,"form":"call","out":2}],"params":[],"ret":{"affine":["n",2,0,2049]},"ret_type":["Tuple",[["Array",2048,["Value",1]],["Value",1]]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2048,["Value",1]],["Tuple",[["Array",2048,["Value",1]],["Value",1]]]]}"#

/-- E(x : Array 2047 (Value 1)) = x with its last bit negated: one parameter and one return field, alone. -/
def D_one : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","E_v1"],"definitions":{"E_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,2046,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2047,["Value",1]]]],"ret":{"concat":[{"affine":["p",0,0,2046]},{"affine":["n",0,0,1]}]},"ret_type":["Array",2047,["Value",1]]},"form":"composite","id":"E_v1"},"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[EProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2047,"out":1},{"args":[{"refs":{"affine":["n",0,0,2047]},"type":1}],"fn":2,"form":"call","out":1}],"params":[],"ret":{"affine":["n",1,0,2047]},"ret_type":["Array",2047,["Value",1]]},"form":"composite","id":"Root[EProg_v1]_v1"}},"program_fn":"EProg_v1","root":"Root[EProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2047,["Value",1]]]}"#

def bitsOf (v n : Nat) : Array Bool := Test.Lower.bits v n

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

/-- The lowered rows of `d`'s `key` against the honest output `o` on input `x`: whether it satisfies; each output bit
each output bit of `flips` flipped alone (those NOT rejected listed; index `i` is the port's bit `i`, padding included); and each input bit at
`inPad` set to 1 with `o` kept (those still satisfied are padding the rows leave free). -/
def flipReport (d key : String) (x o : Array Bool) (flips inPad : List Nat) : List String :=
  match realLower d key #[0] with
  | .error e => ["lowerBlock's lower REFUSED: " ++ e]
  | .ok (t, b) =>
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
      let notRej := flips.filter fun i => ok (zh.set! (Lower.outBit t i) (!o[i]!))
      let freeIn := inPad.filter fun i => ok (put (x.set! i true))
      [s!"lowerBlock: ins {b.ins} outs {b.outs}", s!"honest satisfies every row: {ok zh}",
       s!"output port bits {o.size}; flipped alone: {flips}; NOT rejected: {notRej}",
       s!"input padding bits {inPad} set to 1, rows still satisfied (free): {freeIn}"]

-- the last field: x = 5 (2048 bits), y = 0 (word 128); C's (o0, o1) = (5, 1); o1's padding is the port's bits 2049..2063
#eval flipReport D_last "E_v1" (bitsOf 5 2048 ++ bitsOf 0 16) (bitsOf 5 2048 ++ bitsOf 1 16) (List.range 16 ++ List.range' 2032 32) (List.range' 2049 15)
-- y = 1: C's (o0, o1) = (5, 0)
#eval flipReport D_last "E_v1" (bitsOf 5 2048 ++ bitsOf 1 16) (bitsOf 5 2048 ++ bitsOf 0 16) (List.range 16 ++ List.range' 2032 32) (List.range' 2049 15)
-- one field alone: x = 5 + 2^2046 (2047 bits); C's output 5 (bit 2046 negated); the port's bit 2047 is padding
#eval flipReport D_one "E_v1" (bitsOf (5 + 2^2046) 2048) (bitsOf 5 2048) (List.range 16 ++ List.range' 2032 16) [2047]

end Scratch.LastFieldEval
