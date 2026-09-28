import VStar.PyJson
import VStar.RecOpen

/-!
# Units and their `flock-ir-unit/v2` text

A unit is a circuit built over `forms` (`Flock.HmNets.M`) with its inputs first, one 16-bit port per word (`in0 ..`), and
its outputs' forms, 16 per word (`out0 ..`), as `rec_outer.lowering` and `rec_residuals.lowering` build theirs. Its layout is
`ir_lower.layout`'s with one input group and one output group (`Flock.HmNets.layout`): the input rows, every AND's row in
creation order, the output group's copy rows, the constant. `ir_lower.layout` lays out only the live ANDs (those an output
reads, through any rows); every AND of these units is live, and `laid` refuses a unit with a dead one rather than lay it
out differently.

`text` is `ir_lower.netlist`: a header line `flock-ir-unit/v2 name rows const` and each side's groups, one line
`|A| A.. |B| B..` per row, then (a lowering's text) `LEAVES` and the leaf maps as `json.dumps` writes them (`pyDumps`). A circuit file's sections
are the same text without `LEAVES`.
-/

namespace VStar

open Flock.HmNets
open Lean (Json)

/-- A unit: its name, its input and output ports' words, the built circuit and its outputs' forms. -/
structure UnitC where
  name : String
  inPorts : List Nat
  outPorts : List Nat
  st : St
  outs : Array Lin

def UnitC.inWords (u : UnitC) : Nat := u.inPorts.foldl (· + ·) 0
def UnitC.outWords (u : UnitC) : Nat := u.outPorts.foldl (· + ·) 0

/-- `rec_outer.lowering(LANES, H)`'s unit: `rec_open.unit_circuit`. -/
def recOpenUnit (LANES H : Nat) : Except String UnitC := do
  checkRecOpen LANES H
  let pw := recOpenPorts LANES H
  let build : M (List Lin × List Lin) := do
    let flat ← input (16 * pw.foldl (· + ·) 0)
    match slices flat pw with
    | [row, aux, acc, coef, dirs] => recOpen LANES H row aux acc coef dirs
    | _ => pure ([], [])
  let r := build.run St.empty
  pure { name := s!"rec-open/l{LANES}-h{H}", inPorts := pw, outPorts := [OUT_WORDS, ACC_WORDS], st := r.2,
         outs := (r.1.1 ++ r.1.2).toArray }

/-- An AND no output reads, through any rows, if there is one: `ir_lower.layout` would leave it out. The rows are newest
first and each reads older bits only, so one pass marks every live bit. -/
def deadRow (st : St) (outs : Array Lin) : Option Nat := Id.run do
  let mut live := Array.replicate st.next false
  for f in outs do
    for v in f.s do live := live.set! v true
  for r in st.rows do
    if live.getD r.out false then
      for v in r.a.s do live := live.set! v true
      for v in r.b.s do live := live.set! v true
  for r in st.rows do
    if !live.getD r.out false then return some r.out
  return none

/-- The unit's layout, ports of 16 bits. -/
def UnitC.laid (u : UnitC) : Except String Laid := do
  if u.inWords == 0 then throw s!"{u.name}: a unit of no input words (ir_lower.layout lays out no group for it)"
  if let some v := deadRow u.st u.outs then
    throw s!"{u.name}: AND {v} reaches no output; ir_lower.layout leaves such rows out, which this layout does not"
  pure (layout u.st (Array.replicate u.inWords 16) (Array.replicate u.outWords 16) u.outs)

/-- `Lowering.leaves` of a unit that is its instance: `in` and `out` every leaf in order, each port `[words, 16]`. -/
def UnitC.leaves (u : UnitC) : Json :=
  let idx := fun (n : Nat) => Json.arr #[Json.arr ((Array.range n).map Lean.toJson)]
  let ports := fun (ws : List Nat) => Json.arr (ws.toArray.map fun w => Json.arr #[Lean.toJson w, Lean.toJson (16 : Nat)])
  Json.mkObj [("in", idx u.inWords), ("in_ports", ports u.inPorts), ("out", idx u.outWords), ("out_ports", ports u.outPorts)]

/-! ## The text -/

def FORMAT : String := "flock-ir-unit/v2"

/-- The header: `FORMAT name rows const`, then each side's `n_groups (col words n_ports bits..)..`. -/
def header (name : String) (l : Laid) : String :=
  let side := fun (gs : Array Flock.Group) =>
    toString gs.size :: gs.toList.flatMap fun g => ([g.col, g.words, g.bits.size] ++ g.bits.toList).map toString
  " ".intercalate ([FORMAT, name, toString l.useful, toString l.constPos] ++ side l.inGroups ++ side l.outGroups)

/-- A row's line: `|A| A.. |B| B..`. -/
def rowLine (a b : Array Nat) : String := Id.run do
  let mut s := toString a.size
  for x in a do s := (s.push ' ') ++ toString x
  s := (s.push ' ') ++ toString b.size
  for x in b do s := (s.push ' ') ++ toString x
  return s

/-- The text's lines, each with its newline, to `put`. -/
def emitNet {m : Type → Type} [Monad m] (put : String → m Unit) (name : String) (l : Laid) (leaves : Option String) :
    m Unit := do
  put (header name l ++ "\n")
  for i in [0:l.useful] do
    put (rowLine (l.ra.getD i #[]) (l.rb.getD i #[]) ++ "\n")
  if let some j := leaves then put ("LEAVES " ++ j ++ "\n")

/-- The text's lines: the header's and the rows', the newlines not counted after the last. -/
def netLines (l : Laid) : Nat := 1 + l.useful

end VStar
