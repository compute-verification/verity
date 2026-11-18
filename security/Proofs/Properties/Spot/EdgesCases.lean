module

public import Properties.Spot.Edges
meta import Properties.Spot.Edges
meta import Spot.Edges
meta import Spot.Wires
meta import Flock.Program
meta import Flock.TemplateQuery

/-!
# `EdgesExact`'s reads evaluated on concrete wires

Each case is a Program the circuit format decodes, its templates' ranges and its wiring (JSON, as the edges difftest's
toys write them). `edgesOf` either refuses it, by the refusal named, or returns wires; for each wire, every reader row it
indexes is read whole at the witnesses its own lookups give (`WireReadsWith`, the second clause of `EdgesExact`):
`checkedReads` counts those reads and is `none` when any clause fails. The quantiser at K = 3 reads its scale at word 2,
behind an int8 field of 24 bits, which the statement's earlier clause (every field before the read one whole words)
refused (`oldClause`).
-/

namespace Proofs.Properties.Spot.EdgesCases

open _root_.Spot
open Properties.Spot.Edges
open Flock.Program (Prog fieldsOf)
open Flock.TemplateQuery (TRange)
open Lean (Json)

/-- A case: the Program's bytes, its ranges `[[node, template, first, count], ...]` and its wiring, as JSON text. -/
structure Case where
  program : String
  ranges : String
  wiring : String

instance (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (u row t k i x w l rb wb mk mw j f : ℕ) :
    Decidable (WireReadsWith P rs W e u row t k i x w l rb wb mk mw j f) := by
  unfold WireReadsWith; dsimp only; infer_instance

/-- The case decoded (`Flock.Program.decode`, as `EdgesOfExact` reads it) and `edgesOf`'s answer. -/
def run (c : Case) : Except String (Prog × Array TRange × Wires.Wiring × Except String (List Wires.Wire)) := do
  let P ← (Flock.Program.decode c.program.toUTF8).mapError toString
  let .arr rj ← Json.parse c.ranges | throw "ranges"
  let rs ← rj.mapM fun r => match r with
    | .arr #[k, .str t, f, n] => match k.getNat?, f.getNat?, n.getNat? with
      | .ok k, .ok f, .ok n => pure ({ node := k, template := t, first := f, count := n } : TRange)
      | _, _, _ => throw "range"
    | _ => throw "range"
  let W ← Wires.Wiring.ofJson (← Json.parse c.wiring)
  return (P, rs, W, Edges.edgesOf P rs W)

/-- `edgesOf`'s refusal on the case by its name (without where it arose), or `"wires"`. -/
def answer (c : Case) : String :=
  match run c with
  | .ok (_, _, _, .ok _) => "wires"
  | .ok (_, _, _, .error e) => (e.splitOn " ").headD e
  | .error e => s!"undecoded: {e}"

/-- The reads of wire `e` (reader node `k`'s argument 0, writer node `w`) at every row it indexes, each at the witnesses
the wire's lookups give: their count, or `none` when `WireReadsWith` fails at one. -/
def wireReads (P : Prog) (rs : Array TRange) (W : Wires.Wiring) (e : Wires.Wire) (k w : ℕ) : Option ℕ := do
  let b := (P.defs[P.root]!).body
  let some (_, rb) := (Edges.firstUnits rs)[k]? | none
  let some (_, wb) := (Edges.firstUnits rs)[w]? | none
  let wd := P.defs[(b.nodes[w]!).fn]!
  let len := rowLen P W e.reader k 0
  let mut n := 0
  for mk in List.range (b.nodes[k]!).members do
    for row in List.range (W.cutsOf e.reader e.port) do
      let some j := e.indexAt (rb + mk) row | continue
      let mw := (e.writerAt j - wb).toNat
      let f := ((List.range (fieldsOf wd.ret).length).find? (Edges.fieldWord wd.ret · = e.offsetAt j)).getD 0
      for t in List.range len do
        let x := memberIndex ((b.nodes[k]!).args[0]!) mk (row * len + t)
        let l := mw * wd.outLeaves + Edges.fieldOffset wd.ret f + t
        if !decide (WireReadsWith P rs W e (rb + mk) row t k 0 x w l rb wb mk mw j f) then none
        n := n + 1
  return n

/-- Every wire's reads (`wireReads`), the reader and writer nodes by the ranges: their count, `none` when one fails. -/
def checkedReads (c : Case) : Option ℕ := do
  let .ok (P, rs, W, .ok ws) := run c | none
  let node (tid : String) := (rs.find? (·.template == tid)).map (·.node)
  ws.foldlM (fun n e => do
    let k ← node e.reader
    let w ← node e.writer
    return n + (← wireReads P rs W e k w)) 0

/-- The clause the statement held before the layout: every return field before field `f` of the case's writer
whole u16 words. -/
def oldClause (c : Case) (f : ℕ) : Bool :=
  match run c with
  | .ok (P, rs, _, _) =>
    let b := (P.defs[P.root]!).body
    let w := (rs.getD 0 default).node
    let ret := (P.defs[(b.nodes[w]!).fn]!).ret
    decide (∀ g < f, Bits.bitsOf (fieldsOf ret)[g]! % 16 = 0)
  | .error _ => false

/-- quantiser K = 3 read at its scale: `(Array 3 (Value 8), Value 32)`, the scale at word 2. -/
def quantOdd : Case :=
  ⟨"{\"annotations\":{},\"callees\":[\"CoreEdgesLow8_v1\",\"CoreEdgesXor32_v1\",\"Input32_v1\",\"CoreEdgesCaseQ3W_v0\",\"CoreEdgesCaseQ3R_v0\"],\"definitions\":{\"CoreEdgesCaseQ3R_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"affine\":[\"n\",0,0,1]},\"ret_type\":[\"Value\",32]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseQ3R_v0\"},\"CoreEdgesCaseQ3W_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"strided\":[\"n\",0,0,1,0,0,3]},{\"affine\":[\"n\",1,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseQ3W_v0\"},\"CoreEdgesLow8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesLow8_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",8]},\"CoreEdgesXor32_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor32_v1\",\"params\":[[\"a\",[\"Value\",32]],[\"b\",[\"Value\",32]]],\"ret\":[\"Value\",32]},\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Root[CoreEdgesCaseQ3_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":2},{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":2},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":2}],\"axes\":[0],\"fn\":3,\"form\":\"batch\",\"n\":4,\"out\":3},{\"args\":[{\"refs\":{\"strided\":[\"n\",2,3,1,4,1,4]},\"type\":2}],\"axes\":[0],\"fn\":4,\"form\":\"batch\",\"n\":4,\"out\":2}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,16]},{\"affine\":[\"n\",3,0,4]}]},\"ret_type\":[\"Tuple\",[[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]],[\"Array\",4,[\"Value\",32]]]]},\"form\":\"composite\",\"id\":\"Root[CoreEdgesCaseQ3_v0]_v1\"}},\"program_fn\":\"CoreEdgesCaseQ3_v0\",\"root\":\"Root[CoreEdgesCaseQ3_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Value\",8],[\"Array\",4,[\"Value\",32]],[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]]]}",
   "[[2,\"CoreEdgesCaseQ3W_v0\",0,4],[3,\"CoreEdgesCaseQ3R_v0\",4,4]]",
   "{\"cuts\":{},\"templates\":{\"CoreEdgesCaseQ3R_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4},\"CoreEdgesCaseQ3W_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4}},\"wires\":[]}"⟩

/-- quantiser K = 4 read at its int8 field: one row of 32 bits. -/
def int8Even : Case :=
  ⟨"{\"annotations\":{},\"callees\":[\"CoreEdgesLow8_v1\",\"CoreEdgesXor32_v1\",\"CoreEdgesXor8_v1\",\"Input32_v1\",\"Input8_v1\",\"CoreEdgesCaseI4W_v0\",\"CoreEdgesCaseI4R_v0\"],\"definitions\":{\"CoreEdgesCaseI4R_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,4]},\"type\":2},{\"refs\":{\"affine\":[\"p\",0,0,4]},\"type\":2}],\"axes\":[0,0],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":2}],\"params\":[[\"x\",[\"Array\",4,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseI4R_v0\"},\"CoreEdgesCaseI4W_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"strided\":[\"n\",0,0,1,0,0,4]},{\"affine\":[\"n\",1,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseI4W_v0\"},\"CoreEdgesLow8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesLow8_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",8]},\"CoreEdgesXor32_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor32_v1\",\"params\":[[\"a\",[\"Value\",32]],[\"b\",[\"Value\",32]]],\"ret\":[\"Value\",32]},\"CoreEdgesXor8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Root[CoreEdgesCaseI4_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":3,\"form\":\"batch\",\"n\":4,\"out\":3},{\"args\":[],\"axes\":[],\"fn\":4,\"form\":\"batch\",\"n\":16,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":3}],\"axes\":[0],\"fn\":5,\"form\":\"batch\",\"n\":4,\"out\":5},{\"args\":[{\"refs\":{\"strided\":[\"n\",2,0,4,5,1,16]},\"type\":6}],\"axes\":[0],\"fn\":6,\"form\":\"batch\",\"n\":4,\"out\":6}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,20]},{\"affine\":[\"n\",3,0,16]}]},\"ret_type\":[\"Tuple\",[[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]],[\"Array\",4,[\"Array\",4,[\"Value\",8]]]]]},\"form\":\"composite\",\"id\":\"Root[CoreEdgesCaseI4_v0]_v1\"}},\"program_fn\":\"CoreEdgesCaseI4_v0\",\"root\":\"Root[CoreEdgesCaseI4_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",4,[\"Value\",32]],[\"Array\",16,[\"Value\",8]],[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]],[\"Array\",4,[\"Array\",4,[\"Value\",8]]]]}",
   "[[2,\"CoreEdgesCaseI4W_v0\",0,4],[3,\"CoreEdgesCaseI4R_v0\",4,4]]",
   "{\"cuts\":{},\"templates\":{\"CoreEdgesCaseI4R_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4},\"CoreEdgesCaseI4W_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4}},\"wires\":[]}"⟩

/-- quantiser K = 3 read at its int8 field: one row of 24 bits. -/
def int8Odd : Case :=
  ⟨"{\"annotations\":{},\"callees\":[\"CoreEdgesLow8_v1\",\"CoreEdgesXor32_v1\",\"CoreEdgesXor8_v1\",\"Input32_v1\",\"Input8_v1\",\"CoreEdgesCaseI3W_v0\",\"CoreEdgesCaseI3R_v0\"],\"definitions\":{\"CoreEdgesCaseI3R_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,3]},\"type\":2},{\"refs\":{\"affine\":[\"p\",0,0,3]},\"type\":2}],\"axes\":[0,0],\"fn\":2,\"form\":\"batch\",\"n\":3,\"out\":2}],\"params\":[[\"x\",[\"Array\",3,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,3]},\"ret_type\":[\"Array\",3,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseI3R_v0\"},\"CoreEdgesCaseI3W_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0},{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":1,\"form\":\"primitive\",\"out\":0}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"strided\":[\"n\",0,0,1,0,0,3]},{\"affine\":[\"n\",1,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseI3W_v0\"},\"CoreEdgesLow8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesLow8_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",8]},\"CoreEdgesXor32_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor32_v1\",\"params\":[[\"a\",[\"Value\",32]],[\"b\",[\"Value\",32]]],\"ret\":[\"Value\",32]},\"CoreEdgesXor8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Root[CoreEdgesCaseI3_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":3,\"form\":\"batch\",\"n\":4,\"out\":3},{\"args\":[],\"axes\":[],\"fn\":4,\"form\":\"batch\",\"n\":12,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":3}],\"axes\":[0],\"fn\":5,\"form\":\"batch\",\"n\":4,\"out\":5},{\"args\":[{\"refs\":{\"strided\":[\"n\",2,0,3,4,1,12]},\"type\":6}],\"axes\":[0],\"fn\":6,\"form\":\"batch\",\"n\":4,\"out\":6}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,16]},{\"affine\":[\"n\",3,0,12]}]},\"ret_type\":[\"Tuple\",[[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]],[\"Array\",4,[\"Array\",3,[\"Value\",8]]]]]},\"form\":\"composite\",\"id\":\"Root[CoreEdgesCaseI3_v0]_v1\"}},\"program_fn\":\"CoreEdgesCaseI3_v0\",\"root\":\"Root[CoreEdgesCaseI3_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Value\",8],[\"Array\",3,[\"Value\",8]],[\"Array\",4,[\"Value\",32]],[\"Array\",12,[\"Value\",8]],[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8],[\"Value\",8]]],[\"Value\",32]]]],[\"Array\",4,[\"Array\",3,[\"Value\",8]]]]}",
   "[[2,\"CoreEdgesCaseI3W_v0\",0,4],[3,\"CoreEdgesCaseI3R_v0\",4,4]]",
   "{\"cuts\":{},\"templates\":{\"CoreEdgesCaseI3R_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4},\"CoreEdgesCaseI3W_v0\":{\"outs\":{\"out\":4},\"rows\":{\"x\":[2,1]},\"units\":4}},\"wires\":[]}"⟩

/-- two fields of two 8-bit leaves read as a port cut into 2 rows of 16 bits. -/
def cutPairs : Case :=
  ⟨"{\"annotations\":{},\"callees\":[\"CoreEdgesLow8_v1\",\"CoreEdgesXor8_v1\",\"Input32_v1\",\"Input8_v1\",\"CoreEdgesCaseC2W_v0\",\"CoreEdgesCaseC2R_v0\"],\"definitions\":{\"CoreEdgesCaseC2R_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,4]},\"type\":2},{\"refs\":{\"affine\":[\"p\",0,0,4]},\"type\":2}],\"axes\":[0,0],\"fn\":1,\"form\":\"batch\",\"n\":4,\"out\":2}],\"params\":[[\"x\",[\"Array\",4,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,4]},\"ret_type\":[\"Array\",4,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseC2R_v0\"},\"CoreEdgesCaseC2W_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"strided\":[\"n\",0,0,1,0,0,2]},{\"strided\":[\"n\",1,0,1,0,0,2]}]},\"ret_type\":[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8]]],[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseC2W_v0\"},\"CoreEdgesLow8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesLow8_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",8]},\"CoreEdgesXor8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Root[CoreEdgesCaseC2_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":3},{\"args\":[],\"axes\":[],\"fn\":3,\"form\":\"batch\",\"n\":16,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":3}],\"axes\":[0],\"fn\":4,\"form\":\"batch\",\"n\":4,\"out\":5},{\"args\":[{\"refs\":{\"strided\":[\"n\",2,0,4,4,1,16]},\"type\":6}],\"axes\":[0],\"fn\":5,\"form\":\"batch\",\"n\":4,\"out\":6}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,16]},{\"affine\":[\"n\",3,0,16]}]},\"ret_type\":[\"Tuple\",[[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8]]],[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]]]],[\"Array\",4,[\"Array\",4,[\"Value\",8]]]]]},\"form\":\"composite\",\"id\":\"Root[CoreEdgesCaseC2_v0]_v1\"}},\"program_fn\":\"CoreEdgesCaseC2_v0\",\"root\":\"Root[CoreEdgesCaseC2_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Value\",8],[\"Array\",4,[\"Value\",8]],[\"Array\",4,[\"Value\",32]],[\"Array\",16,[\"Value\",8]],[\"Array\",4,[\"Tuple\",[[\"Tuple\",[[\"Value\",8],[\"Value\",8]]],[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]]]],[\"Array\",4,[\"Array\",4,[\"Value\",8]]]]}",
   "[[2,\"CoreEdgesCaseC2W_v0\",0,4],[3,\"CoreEdgesCaseC2R_v0\",4,4]]",
   "{\"cuts\":{\"CoreEdgesCaseC2R_v0\":{\"x\":2}},\"templates\":{\"CoreEdgesCaseC2R_v0\":{\"outs\":{\"out\":2},\"rows\":{\"x\":[1,2]},\"units\":4},\"CoreEdgesCaseC2W_v0\":{\"outs\":{\"out\":2},\"rows\":{\"x\":[1,2]},\"units\":4}},\"wires\":[]}"⟩

/-- two 8-bit fields read as a port cut into 2 rows of 8 bits. -/
def cutBytes : Case :=
  ⟨"{\"annotations\":{},\"callees\":[\"CoreEdgesLow8_v1\",\"CoreEdgesXor8_v1\",\"Input32_v1\",\"Input8_v1\",\"CoreEdgesCaseC1W_v0\",\"CoreEdgesCaseC1R_v0\"],\"definitions\":{\"CoreEdgesCaseC1R_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,2]},\"type\":2},{\"refs\":{\"affine\":[\"p\",0,0,2]},\"type\":2}],\"axes\":[0,0],\"fn\":1,\"form\":\"batch\",\"n\":2,\"out\":2}],\"params\":[[\"x\",[\"Array\",2,[\"Value\",8]]]],\"ret\":{\"affine\":[\"n\",0,0,2]},\"ret_type\":[\"Array\",2,[\"Value\",8]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseC1R_v0\"},\"CoreEdgesCaseC1W_v0\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"p\",0,0,1]},\"type\":0}],\"fn\":0,\"form\":\"primitive\",\"out\":1}],\"params\":[[\"x\",[\"Value\",32]]],\"ret\":{\"concat\":[{\"affine\":[\"n\",0,0,1]},{\"affine\":[\"n\",1,0,1]}]},\"ret_type\":[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]},\"form\":\"composite\",\"id\":\"CoreEdgesCaseC1W_v0\"},\"CoreEdgesLow8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesLow8_v1\",\"params\":[[\"a\",[\"Value\",32]]],\"ret\":[\"Value\",8]},\"CoreEdgesXor8_v1\":{\"form\":\"primitive\",\"id\":\"CoreEdgesXor8_v1\",\"params\":[[\"a\",[\"Value\",8]],[\"b\",[\"Value\",8]]],\"ret\":[\"Value\",8]},\"Input32_v1\":{\"form\":\"primitive\",\"id\":\"Input32_v1\",\"params\":[],\"ret\":[\"Value\",32]},\"Input8_v1\":{\"form\":\"primitive\",\"id\":\"Input8_v1\",\"params\":[],\"ret\":[\"Value\",8]},\"Root[CoreEdgesCaseC1_v0]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":2,\"form\":\"batch\",\"n\":4,\"out\":3},{\"args\":[],\"axes\":[],\"fn\":3,\"form\":\"batch\",\"n\":8,\"out\":4},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,4]},\"type\":3}],\"axes\":[0],\"fn\":4,\"form\":\"batch\",\"n\":4,\"out\":5},{\"args\":[{\"refs\":{\"strided\":[\"n\",2,0,2,2,1,8]},\"type\":6}],\"axes\":[0],\"fn\":5,\"form\":\"batch\",\"n\":4,\"out\":6}],\"params\":[],\"ret\":{\"concat\":[{\"affine\":[\"n\",2,0,8]},{\"affine\":[\"n\",3,0,8]}]},\"ret_type\":[\"Tuple\",[[\"Array\",4,[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]],[\"Array\",4,[\"Array\",2,[\"Value\",8]]]]]},\"form\":\"composite\",\"id\":\"Root[CoreEdgesCaseC1_v0]_v1\"}},\"program_fn\":\"CoreEdgesCaseC1_v0\",\"root\":\"Root[CoreEdgesCaseC1_v0]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Value\",32],[\"Value\",8],[\"Array\",2,[\"Value\",8]],[\"Array\",4,[\"Value\",32]],[\"Array\",8,[\"Value\",8]],[\"Array\",4,[\"Tuple\",[[\"Value\",8],[\"Value\",8]]]],[\"Array\",4,[\"Array\",2,[\"Value\",8]]]]}",
   "[[2,\"CoreEdgesCaseC1W_v0\",0,4],[3,\"CoreEdgesCaseC1R_v0\",4,4]]",
   "{\"cuts\":{\"CoreEdgesCaseC1R_v0\":{\"x\":2}},\"templates\":{\"CoreEdgesCaseC1R_v0\":{\"outs\":{\"out\":2},\"rows\":{\"x\":[1,2]},\"units\":4},\"CoreEdgesCaseC1W_v0\":{\"outs\":{\"out\":2},\"rows\":{\"x\":[1,2]},\"units\":4}},\"wires\":[]}"⟩

#guard answer quantOdd == "wires"
#guard checkedReads quantOdd == some 4
#guard oldClause quantOdd 1 == false
#guard answer int8Even == "wires"
#guard checkedReads int8Even == some 16
#guard answer int8Odd == "edge-word"
#guard answer cutPairs == "wires"
#guard checkedReads cutPairs == some 16
#guard answer cutBytes == "edge-word"

end Proofs.Properties.Spot.EdgesCases
