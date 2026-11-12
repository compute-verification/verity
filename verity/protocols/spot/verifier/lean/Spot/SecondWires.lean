module

public import Spot.SecondStage
public import Spot.Bits

/-!
# Three conditions on a two-stage design's wires, decided (`Spot.SecondWires`)

Round 2's check of the wires (`SecondStage.wiresOk`) holds each wire to the parts, to its rows' units and to the links.
It does not decide three conditions the wiring of the members' circuits also rests on. This module decides them, each
at the size of what it reads; no check calls it.

* `rowsUnder`: at the rows of each port a wire names, each run's last unit reads a row under the rows' `per`
  (`row + (count - 1)·step < per`), so every unit the run holds reads one of its writer's units of its own replay
  block. It reads the wires' `template` and `port` and the rows (`Own.wires`, `Own.rows`): one lookup of a port's rows
  a wire and one comparison a run, a run being one entry however many units it holds.
* `namesOnce`: no two parts have one template name. It reads the parts' `template`: one comparison a pair of parts.
* `wiresWide`: for each wire, every bit under the `bits` of each port of the wire's name in the reader's circuit file
  is one the writer's circuit file writes on the wire's output row (`covers`). It reads, beside the wires, each
  template's circuit file by its name (`cs`), which round 2's check does not hold: of the reader's file its ports'
  `name` and `bits`; of the writer's its output ports and rows (`outRows`), its instances a unit (`upv`), one unit's
  returned leaves (`leavesOut`) and three numbers of its unit net (`inWords`, `constPos` and its output group's `col`).
  A wire costs one pass over those leaves' bits (`writes`), each looked up among the output rows, into a bitmap of the
  reader port's bits (`marks`), and one pass over the bitmap. It reads no unit of a batch, since the leaves are one
  unit's and the same in every unit, and no row of the unit net: a leaf's bit `t` is on a computed row exactly when its
  column `col·WORD + 16 j + t` is at or past the net's inputs, `min (inWords·WORD) constPos`, and before its constant.
  Its work follows the lists the file holds, never a number in it: `writes` passes over the instances whose leaves the
  file lists, at most `upv`, and `covers` marks a bitmap only where the port is no wider than the bits listed.
* `decided` is the three together, and `failures` each one's refusal: `second-wire-rows`, `second-part-names` and
  `second-wire-widths`, with its reason.

Each is stated on the wires and the rows themselves (`rowsUnderAt`, `wiresWideAt`, `decidedAt`, `failuresAt`) and read
at what the verifier holds of the run, its wires, none where none is derived, and its rows.

`piece` is the difftest's `second_wires`, which holds `second_wires`' Python to these.
-/

@[expose] public section

namespace Spot.SecondWires

open Lean (Json)
open Spot.SecondStage (Own Part Wire Rows Run)

/-! ## The rows and the names -/

/-- **Each run's last row is one of its replay block's rows**, at wires `ws` and rows `rows`: at the rows of each
port a wire names, the first of that template and port, each run's last unit reads a row under `per`
(`row + (count - 1)·step < per`), so every unit the run holds does. A wire whose port has no rows passes:
`SecondStage.wiresOk` refuses it. -/
def rowsUnderAt (ws : Array Wire) (rows : Array Rows) : Bool :=
  ws.all fun w => (rows.find? fun r => r.template == w.template && r.port == w.port).all fun r =>
    r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)

/-- `rowsUnderAt` at what the verifier holds of the run: its wires and its rows (`SecondStage.rowsOf`). -/
def rowsUnder (own : Own) : Bool := rowsUnderAt (own.wires.getD #[]) own.rows

/-- **Each name is one template's**: no two parts have one template name. -/
def namesOnce (ps : Array Part) : Bool := decide (ps.toList.map (·.template)).Nodup

/-! ## What a circuit file writes on an output row -/

/-- **A circuit file's output rows**, in order: each one's name, the output word it starts at and its words
(`Flock.HmOut.outGroup`). -/
def outRows (c : Flock.Circuit) : List (String × Nat × Nat) :=
  (List.range c.outRowPorts.size).map fun o =>
    ((c.ports.getD (Flock.HmOut.nIn c + o) default).name, (Flock.HmOut.outGroup c o).2.2.1,
      (Flock.HmOut.outGroup c o).2.2.2)

/-- **The bits of its output row `port` that circuit file `c` writes**, a unit: bit `16 (x - lo) + t` for bit `t` of a
returned leaf `j` of an instance `k`, where the leaf's output word `x` is one of the row's, from the row's first word
`lo`, and the bit's column `col·WORD + 16 j + t` is a computed row of the unit net: at or past its inputs
(`min (inWords·WORD) constPos`) and before its constant. One entry a leaf's bit, the leaf looked up among the output
rows; the same in every unit. The instances are those under `upv` whose leaves the file lists: one past the list
returns no leaf. -/
def writes (c : Flock.Circuit) (port : String) : List Nat :=
  let rows := outRows c
  let cp := c.unit.constPos
  let nIn := min (c.unit.inWords * Flock.WORD) cp
  let base := (c.unit.outGroups.getD 0 default).col * Flock.WORD
  (List.range (min c.upv c.leavesOut.size)).flatMap fun k =>
    let ls := c.leavesOut.getD k #[]
    (List.range ls.size).flatMap fun j =>
      match rows.find? fun r => decide (r.2.1 ≤ ls.getD j 0) && decide (ls.getD j 0 < r.2.1 + r.2.2) with
      | some r =>
        if r.1 == port then
          (List.range 16).filterMap fun t =>
            if nIn ≤ base + (16 * j + t) ∧ base + (16 * j + t) < cp then some (16 * (ls.getD j 0 - r.2.1) + t)
            else none
        else []
      | none => []

/-- **A bitmap of `n` bits with those of `l` set**; a bit of `l` at or past `n` sets none. -/
def marks (n : Nat) (l : List Nat) : Array Bool :=
  l.foldl (fun a i => a.setIfInBounds i true) (Array.replicate n false)

/-- **Circuit file `c` writes every bit under `bits` of its output row `port`** (`writes`, marked once): no more bits
than it lists, which a list that holds each needs, and every bit of the bitmap set. -/
def covers (c : Flock.Circuit) (port : String) (bits : Nat) : Bool :=
  let l := writes c port
  decide (bits ≤ l.length) && (marks bits l).all id

/-- **A reader's port is no wider than what its writer writes**, at wires `ws`: for each wire, every bit under the
`bits` of each port of the wire's name in the reader's circuit file is one the writer's circuit file writes on the
wire's output row (`covers`). `cs` names each template's circuit file, the first of a name its file; a wire whose
reader or writer it names no file for fails. -/
def wiresWideAt (cs : Array (String × Flock.Circuit)) (ws : Array Wire) : Bool :=
  ws.all fun w =>
    match cs.find? (·.1 == w.template), cs.find? (·.1 == w.writer) with
    | some r, some x => r.2.ports.all fun p => p.name != w.port || covers x.2 w.wport p.bits
    | _, _ => false

/-- `wiresWideAt` at the wires the verifier holds of the run. -/
def wiresWide (cs : Array (String × Flock.Circuit)) (own : Own) : Bool := wiresWideAt cs (own.wires.getD #[])

/-! ## The three together -/

/-- **The three conditions**, over parts `ps`, the templates' circuit files `cs`, wires `ws` and rows `rows`. -/
def decidedAt (ps : Array Part) (cs : Array (String × Flock.Circuit)) (ws : Array Wire) (rows : Array Rows) : Bool :=
  rowsUnderAt ws rows && namesOnce ps && wiresWideAt cs ws

/-- **The three conditions** at what the verifier holds of the run: `rowsUnder`, `namesOnce` and `wiresWide`. -/
def decided (ps : Array Part) (cs : Array (String × Flock.Circuit)) (own : Own) : Bool :=
  decidedAt ps cs (own.wires.getD #[]) own.rows

def ROWS_WHY : String :=
  "a wire's rows give a unit of a run a row that is not one of its writer's units of the unit's replay block"

def NAMES_WHY : String := "two of the interior program's parts have one template name"

def WIDTHS_WHY : String :=
  "a wire's reader port has a bit its writer's circuit file does not write on the wire's output row, or the " ++
  "reader's or the writer's circuit file is not given"

/-- `second-wire-rows` unless `rowsUnderAt`, `second-part-names` unless `namesOnce` and `second-wire-widths` unless
`wiresWideAt`, each with its reason; none exactly when `decidedAt`. -/
def failuresAt (ps : Array Part) (cs : Array (String × Flock.Circuit)) (ws : Array Wire) (rows : Array Rows) :
    List (String × Json) :=
  (if rowsUnderAt ws rows then [] else [("second-wire-rows", Json.str ROWS_WHY)]) ++
    (if namesOnce ps then [] else [("second-part-names", Json.str NAMES_WHY)]) ++
    if wiresWideAt cs ws then [] else [("second-wire-widths", Json.str WIDTHS_WHY)]

/-- `failuresAt` at what the verifier holds of the run; none exactly when `decided`. -/
def failures (ps : Array Part) (cs : Array (String × Flock.Circuit)) (own : Own) : List (String × Json) :=
  failuresAt ps cs (own.wires.getD #[]) own.rows

/-! ## The difftest's piece -/

/-- The layout fields `wiresWide` reads of a circuit file, as a circuit file: `{"ports": [[name, bits], ...]` (the
input rows, then the output rows), `"out_ports": [[name, words], ...]`, `"out_row_ports": [count, ...]`, `"upv"`,
`"leaves_out": [[word, ...], ...]`, `"in_words"`, `"const_pos"`, `"out_col"}`; every other field is the default's. -/
def circuitOf? (j : Json) : Option Flock.Circuit := do
  guard (keys? j == some #["const_pos", "in_words", "leaves_out", "out_col", "out_ports", "out_row_ports", "ports",
    "upv"])
  let pair (x : Json) : Option (String × Nat) := match x with
    | .arr #[.str s, n] => do pure (s, ← nat? n)
    | _ => none
  let list (x : Json) : Option (Array Json) := match x with
    | .arr xs => some xs
    | _ => none
  let ports ← (← list (getD j "ports")).mapM pair
  let outPorts ← (← list (getD j "out_ports")).mapM pair
  let outRowPorts ← (← list (getD j "out_row_ports")).mapM nat?
  let leavesOut ← (← list (getD j "leaves_out")).mapM fun ls => do (← list ls).mapM nat?
  let upv ← nat? (getD j "upv")
  let inWords ← nat? (getD j "in_words")
  let constPos ← nat? (getD j "const_pos")
  let col ← nat? (getD j "out_col")
  let unit : Flock.Net := { (default : Flock.Net) with inWords, constPos, outGroups := #[⟨col, 0, #[]⟩] }
  pure { (default : Flock.Circuit) with
    ports := ports.map fun (name, bits) => { (default : Flock.Port) with name, bits }
    outPorts, outRowPorts, upv, leavesOut, nets := #[("unit", unit)], unitNet := 0 }

/-- The `second_wires` piece: `{"parts": [[template, lo, n], ...], "wires": <SecondStage.wiresOf?'s, null for none>,
"rows": <SecondStage.rowsOf?'s>, "circuits": [[template, <circuitOf?'s>], ...]}` to each decider's answer at those
wires, no wire for null, and rows, the three together and the failures' codes. -/
def piece (j : Json) : Except String Json := do
  let .arr ps := getD j "parts" | throw "parts is not a list"
  let some parts := ps.mapM (fun p => match p with
      | .arr #[.str t, lo, n] => do pure (Part.mk t (← nat? lo) (← nat? n))
      | _ => none : Json → Option Part)
    | throw "parts is not a list of [template, lo, n], a string and two naturals"
  let ws ← match getD j "wires" with
    | .null => pure #[]
    | ws => match SecondStage.wiresOf? ws with
      | some ws => pure ws
      | none => throw "wires is null or a list of {template, port, writer, wport}, strings"
  let some rows := SecondStage.rowsOf? (getD j "rows")
    | throw "rows is not a list of {template, port, per, runs}, runs [unit, count, row, step] naturals"
  let .arr cj := getD j "circuits" | throw "circuits is not a list"
  let some cs := cj.mapM (fun c => match c with
      | .arr #[.str t, l] => do pure (t, ← circuitOf? l)
      | _ => none : Json → Option (String × Flock.Circuit))
    | throw "circuits is not a list of [template, layout], a layout its eight fields, naturals and lists of them"
  return Json.mkObj [("rows_under", .bool (rowsUnderAt ws rows)), ("names_once", .bool (namesOnce parts)),
    ("wires_wide", .bool (wiresWideAt cs ws)), ("decided", .bool (decidedAt parts cs ws rows)),
    ("codes", .arr ((failuresAt parts cs ws rows).map fun x => Json.str x.1).toArray)]

end Spot.SecondWires
