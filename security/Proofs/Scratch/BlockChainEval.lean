module

public import Spot.OneStageZk
public import Spot.Restrict
public import Spot.BlockFront
public import Flock.Test.Lower
meta import Spot.OneStageZk
meta import Spot.Restrict
meta import Spot.BlockFront
meta import Flock.Test.Lower

/-! Scratch, not for merge: a block of two units where the second reads the first's output, through `front`,
`restrict`, `admitted`, `blockTmpl`, `lowerBlock` and C's wiring; and the rows of the one template at a wrong read.
It reads `Spot.OneStageZk.admitted` and `blockTmpl`: run it on a tree holding them, with the lakefile's roots holding
`Proofs.Scratch.BlockChainEval` (`lakefile.patch`), by `verity lean try security/Proofs/Scratch/BlockChainEval.lean`. -/

@[expose] public section

namespace Scratch.BlockChainEval

open Flock Flock.Program Spot.OneStageZk Lean

/-- T(x : Array 2048 (Value 1)) = (Not x[0]) ++ x[1:2048]; root: u0 = T(input), u1 = T(u0). -/
def D_calls : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","T_v1"],"definitions":{"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[ChainProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":2048,"out":1},{"args":[{"refs":{"affine":["n",0,0,2048]},"type":1}],"fn":2,"form":"call","out":1},{"args":[{"refs":{"affine":["n",1,0,2048]},"type":1}],"fn":2,"form":"call","out":1}],"params":[],"ret":{"affine":["n",2,0,2048]},"ret_type":["Array",2048,["Value",1]]},"form":"composite","id":"Root[ChainProg_v1]_v1"},"T_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2048,["Value",1]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",0,1,2047]}]},"ret_type":["Array",2048,["Value",1]]},"form":"composite","id":"T_v1"}},"program_fn":"ChainProg_v1","root":"Root[ChainProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2048,["Value",1]]]}"#
/-- The same with two batches of two members: node 2's members read node 1's. -/
def D_batches : String := r#"{"annotations":{},"boolean":{},"callees":["Input1_v1","Not_v1","T_v1"],"definitions":{"Input1_v1":{"form":"primitive","id":"Input1_v1","params":[],"ret":["Value",1]},"Not_v1":{"form":"primitive","id":"Not_v1","params":[["a",["Value",1]]],"ret":["Value",1]},"Root[ChainProg_v1]_v1":{"bindings":{},"body":{"nodes":[{"args":[],"axes":[],"fn":0,"form":"batch","n":4096,"out":3},{"args":[{"refs":{"affine":["n",0,0,4096]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":2,"out":2},{"args":[{"refs":{"affine":["n",1,0,4096]},"type":2}],"axes":[0],"fn":2,"form":"batch","n":2,"out":2}],"params":[],"ret":{"affine":["n",2,0,4096]},"ret_type":["Array",2,["Array",2048,["Value",1]]]},"form":"composite","id":"Root[ChainProg_v1]_v1"},"T_v1":{"bindings":{},"body":{"nodes":[{"args":[{"refs":{"affine":["p",0,0,1]},"type":0}],"fn":1,"form":"primitive","out":0}],"params":[["x",["Array",2048,["Value",1]]]],"ret":{"concat":[{"affine":["n",0,0,1]},{"affine":["p",0,1,2047]}]},"ret_type":["Array",2048,["Value",1]]},"form":"composite","id":"T_v1"}},"program_fn":"ChainProg_v1","root":"Root[ChainProg_v1]_v1","schema":"verity-ir/descriptor/v2","types":[["Value",1],["Array",2048,["Value",1]],["Array",2,["Array",2048,["Value",1]]],["Array",4096,["Value",1]]]}"#

def query : Json := Json.mkObj [("name", "Q_template_instance"), ("version", (0 : Nat)),
  ("params", Json.mkObj [("template", "T_v1")])]

def cut (s : String) (n : Nat) : String := (s.take n).toString

def refsOf (s : Seq) : String := cut (toString (s.toArray.toList.map fun r => (if r.p then "p" else "n", r.i, r.j))) 200

def report (d : String) (units : Array Nat) (file : Bool := false) : List String := Id.run do
  let mut out : List String := []
  match (do
    let P ← decode d.toUTF8
    let fr ← Spot.BlockFront.front P units
    pure (P, fr)) with
  | .error e => return ["decode/front REFUSED: " ++ e]
  | .ok (P, fr) =>
    out := out ++ [s!"front: ranges {fr.ranges} ins {fr.ins.toList.map (fun p => (p.gate, p.width))} outs {fr.outs.toList.map (fun p => (p.gate, p.width))}"]
    match Spot.Restrict.restrict P units with
    | .error e => return out ++ ["restrict REFUSED: " ++ e]
    | .ok S =>
      let root := S.defs[S.root]!
      out := out ++ (List.range root.body.nodes.size).map fun k =>
        let x := root.body.nodes[k]!
        s!"S root node {k}: {(S.defs[x.fn]!).key} {x.form.name} n={x.n} args {x.args.toList.map (refsOf ·.seq)}"
      match admitted (Json.mkObj [("query", query)]) [] S with
      | .error e => return out ++ ["admitted REFUSED: " ++ e]
      | .ok a =>
        out := out ++ [s!"admitted: key {a.key} d {a.d} ins {a.ins} fields {a.fields}"]
        match (do
            let some ts := Flock.TemplateQuery.templatesOf "Q_template_instance" 0 (query.getObjValD "params")
              | throw "no template query"
            Flock.TemplateQuery.evaluate (.ofProg P) (← ts)) with
        | .error e => out := out ++ ["partition REFUSED: " ++ e]
        | .ok rs =>
          match derivedWiring P rs with
          | .error e => out := out ++ ["wiring REFUSED: " ++ e]
          | .ok W => out := out ++ [s!"C's wiring: {W.wires.length} wire(s): " ++ cut (reprStr W.wires) 600]
        match blockTmpl S a with
        | .error e => return out ++ ["blockTmpl REFUSED: " ++ e]
        | .ok t =>
          out := out ++ [s!"blockTmpl: one template, paramLeaves in {(List.range a.ins.size).length} row(s), constPos {t.constPos}"]
          match Tmpl.Tmpl.rows t with
          | .error e => return out ++ ["rows REFUSED: " ++ e]
          | .ok acc =>
            let k := t.constPos
            let sum := fun (z : Array Bool) (r : Array Nat) => r.foldl (fun b c => b != (c == k || z.getD c false)) false
            let ok := fun (z : Array Bool) => z.getD k false && (List.range acc.ra.size).all fun c =>
              !(acc.set.getD c false) || z.getD c false == (sum z acc.ra[c]! && sum z acc.rb[c]!)
            let outv := fun (z : Array Bool) => (List.range 16).map fun i => z.getD (Lower.outBit t i) false
            let bits := fun (v : Nat) => Test.Lower.bits v 2048
            -- x = 0: C's u0 = 1, C's u1 = T(1) = 0. The copy of u1 reads a WRONG 0 for u0's output.
            let row := fun (x : Array Bool) => match Test.Lower.solve t x with
              | .ok z => s!"satisfies {ok z}, output {outv z}"
              | .error e => "solve: " ++ e
            out := out ++ [s!"copy u0 at x = 0 (right): {row (bits 0)}",
                           s!"copy u1 at its input = 1 (right, u0's output): {row (bits 1)}",
                           s!"copy u1 at its input = 0 (WRONG; u0's output is 1): {row (bits 0)}"]
  if !file then return out
  match Spot.OneStageZk.lowerBlock (Json.parse d |>.toOption.getD Json.null) (Json.mkObj [("query", query)]) []
      ((decode d.toUTF8 >>= fun P => Spot.Restrict.restrict P units).toOption.getD default) with
  | .error e => out := out ++ ["lowerBlock REFUSED: " ++ e]
  | .ok f =>
    let s := String.fromUTF8? f |>.getD ""
    out := out ++ ["lowerBlock file, first 2 lines: " ++ cut ((s.splitOn "\n").take 2 |> String.intercalate " | ") 900]
  return out

#eval report D_calls #[0, 1] (file := true)
#eval report D_batches #[0, 1, 2, 3]
#eval report D_batches #[0, 2]

end Scratch.BlockChainEval
