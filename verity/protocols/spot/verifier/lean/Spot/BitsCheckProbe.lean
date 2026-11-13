module

public import Spot.Bits
public import Flock.ProvedScope
public import Flock.Zk
public import Flock.HmRow
public import Flock.Program
public import Lean.Data.Json

/-!
# A scratch probe: `Spot.Bits.check` on a round's circuit files

`bits-check-probe PROGRAM CIRCUIT...`: each circuit file read as `verify --zk` reads a session's statement circuit
(`ProvedScope.circuitOf` at `Zk.tags Tags.circuit`, then `HmOut.unrolled`), held to its own template (META
`program_digests`' key, `HmRow.queryTemplate`) of the Program `PROGRAM` by `Spot.Bits.check`. Beside the verdict it
prints, for each file, what `Spot.Bits.unitReads` and `readsWhy` say of each parameter's bits, what each returned output
bit's column is (`outsOf`: a computed row or not; the row by its own sides; and, copies followed, a constant, one input
bit, an XOR or an AND), and each step's time.

One line a file, `BITSCHECK {json}`, and a last line `BITSCHECK-RESULT {json}`. Exit 0 when `check` accepts every
file, 1 when it refuses one, 2 when a file doesn't parse. Scratch: nothing calls it and it lands nowhere.
-/

@[expose] public section

namespace Spot.BitsCheckProbe

open Lean (Json)

def timed {α : Type} (act : Unit → α) : IO (α × Nat) := do
  let t0 ← IO.monoMsNow
  let x ← IO.lazyPure act
  let t1 ← IO.monoMsNow
  return (x, t1 - t0)

/-- Each parameter of `d`: its name, its bits, how many of them no row of the unit reads, and the first such bit, by the
columns `Spot.Bits.readsWhy` reads them at. -/
def unreadOf (c : Flock.Circuit) (d : Flock.Program.Defn) (reads : Array Bool) :
    Array (String × Nat × Nat × Option Nat) := Id.run do
  let pc := Flock.portCols c.unit.inGroups
  let mut P := 0
  let mut out := #[]
  for (nm, t) in d.params do
    let n := Spot.Bits.bitsOf t
    let mut unread := 0
    let mut first : Option Nat := none
    for β in [0:n] do
      unless reads.getD ((pc.getD (P + β / 16) (0, 0)).1 + β % 16) false do
        unread := unread + 1
        if first.isNone then first := some β
    out := out.push (nm, n, unread, first)
    P := P + Spot.Bits.wordsOf t
  return out

/-- Each stacked input column's key among `B` (`Spot.Bits.keyIdx`, in one pass): the first input column with the same
source, none at the zero. -/
def keysOf (c : Flock.Circuit) (B : Nat) : Array (Option Nat) := Id.run do
  let ix := Spot.Bits.Index.of c
  let mut seen : Std.HashMap Spot.Bits.Src Nat := {}
  let mut out : Array (Option Nat) := Array.mkEmpty B
  for x in [0:B] do
    let s := Spot.Bits.srcKey c ix x
    if s == .zero then out := out.push none
    else match seen[s]? with
      | some b => out := out.push (some b)
      | none =>
        seen := seen.insert s x
        out := out.push (some x)
  return out

/-- The stacked rows a probe reads: the unit net's rows, how many times they stack, and each stacked input's key. -/
structure View where
  R0 : Spot.Bits.Rows
  upv : Nat
  keys : Array (Option Nat)

def View.nIn (v : View) : Nat := v.upv * v.R0.nIn
def View.nComp (v : View) : Nat := v.upv * (v.R0.nComp + 1)

/-- Whether the stacked column `col` is true by construction: the shared constant, or an instance's constant row, which
reads only the shared constant. -/
def View.isOne (v : View) (col : Nat) : Bool :=
  v.nIn + v.nComp ≤ col || (v.nIn ≤ col && (col - v.nIn) % (v.R0.nComp + 1) == 0)

/-- What the XOR of the stacked columns `L` is, as the key program wires them: its constant part (a column true by
construction, met an odd number of times) and the gates it leaves free, a gate met twice cancelled. An input column is
its key's gate, and no gate at the zero. -/
def xorOf (v : View) (L : List Nat) : Bool × List Nat :=
  L.foldl (fun (acc : Bool × List Nat) col =>
    if v.isOne col then (!acc.1, acc.2)
    else match (if col < v.nIn then v.keys.getD col none else some col) with
      | none => acc
      | some x => if acc.2.contains x then (acc.1, acc.2.erase x) else (acc.1, x :: acc.2)) (false, [])

/-- What a stacked column computes, as far as copies and constants go: a constant, a free input column (negated or
not), an XOR of two or more columns, an AND of two XORs that are no constants, or a chain of copies past the fuel. -/
inductive Kind
  | const (b : Bool)
  | input (col : Nat) (neg : Bool)
  | linear
  | and
  | deep
  deriving Inhabited

def Kind.name : Kind → String
  | .const false => "const0"
  | .const true => "const1"
  | .input _ false => "input"
  | .input _ true => "input_negated"
  | .linear => "xor"
  | .and => "and"
  | .deep => "deep"

/-- Stacked row `i` as `(⊕ a) ∧ (⊕ b)`: each side's constant part and free gates. -/
def sides (v : View) (i : Nat) : (Bool × List Nat) × (Bool × List Nat) :=
  (xorOf v (v.R0.stackRow v.upv v.R0.a i), xorOf v (v.R0.stackRow v.upv v.R0.b i))

/-- What a row with these sides is: a constant, the gate `x` or its negation (a copy), an XOR of several gates, or an
AND. -/
inductive Shape
  | const (b : Bool)
  | copy (x : Nat) (neg : Bool)
  | linear
  | and

def shapeOf : (Bool × List Nat) × (Bool × List Nat) → Shape
  | ((ca, fa), (cb, fb)) =>
    if (fa.isEmpty && !ca) || (fb.isEmpty && !cb) then .const false
    else if fa.isEmpty && fb.isEmpty then .const true
    else if fa.isEmpty then (match fb with
      | [x] => .copy x cb
      | _ => .linear)
    else if fb.isEmpty then (match fa with
      | [x] => .copy x ca
      | _ => .linear)
    else if fa.length == fb.length && fa.all fb.contains then
      (if ca != cb then .const false else match fa with
        | [x] => .copy x ca
        | _ => .linear)
    else .and

/-- The stacked column `col` followed through copy rows: columns below `nIn` are inputs, then the stacked rows, then the
shared constant. `neg` is the negation gathered on the way. -/
def kindOf (v : View) : Nat → Nat → Bool → Kind
  | 0, _, _ => .deep
  | fuel + 1, col, neg =>
    if col < v.nIn then (match v.keys.getD col none with
      | none => .const neg
      | some b => .input b neg)
    else if v.isOne col then .const (!neg)
    else match shapeOf (sides v (col - v.nIn)) with
      | .const b => .const (b != neg)
      | .copy x n => kindOf v fuel x (neg != n)
      | .linear => .linear
      | .and => .and

/-- What a row is by its own two sides, before any copy is followed. -/
def ownOf (v : View) (col : Nat) : String :=
  match shapeOf (sides v (col - v.nIn)) with
  | .const _ => "const"
  | .copy x _ => if x < v.nIn then "copy_of_input" else "copy_of_row"
  | .linear => "xor"
  | .and => "and"

/-- **Every returned output bit of one unit, by what its column is**: for each output row port, how many of its bits
sit on a computed row of the unit net (the bits `Spot.Bits.outGate` gives a gate, a row's AND in the lowered program),
how many on an input column or the constant column (no gate), what each such row is by its own sides (`own`), and what
it is once copies are followed (`resolved`): a constant, one free input bit, an XOR or an AND. For a bit that resolves
to an input, the row port the input is read from (`from`), whether it is the same bit of that port (`from_same_bit`).
Counters are keyed `port|what|...`; `first_bit` is the first bit of each; `runs` lists each port's bits in order as
runs `[what, first bit, length]`, an input as `input:PORT=` (the same bit of `PORT`) or `input:PORT~` (another). -/
def outsOf (c : Flock.Circuit) : Json := Id.run do
  let R0 := Spot.Bits.Rows.ofNet c.unit
  let ix := Spot.Bits.Index.of c
  let v : View := { R0, upv := c.upv, keys := keysOf c (c.upv * R0.nIn) }
  let mut n : Std.HashMap String Nat := {}
  let mut first : Std.HashMap String Nat := {}
  let mut runs : Std.HashMap String (Array (String × Nat × Nat)) := {}
  for k in [0:c.upv] do
    for j in [0:(c.leavesOut.getD k #[]).size] do
      for t in [0:16] do
        let (port, bit) := match Spot.Bits.outRowBit c 0 k j t with
          | some r => (r.row, r.bit)
          | none => ("(no output row)", 16 * j + t)
        let bump := fun (m : Std.HashMap String Nat) (key : String) => m.insert key (m.getD key 0 + 1)
        let note := fun (m : Std.HashMap String Nat) (key : String) => if m.contains key then m else m.insert key bit
        n := bump n s!"{port}|bits"
        let oc := Spot.Bits.outCol c j t
        let mut what := ""
        if oc < R0.nIn then
          what := "no_gate:input_column"
          n := bump n s!"{port}|no_gate|input_column"
          first := note first s!"{port}|no_gate|input_column"
        else if R0.nIn + R0.nComp ≤ oc then
          what := "no_gate:constant_column"
          n := bump n s!"{port}|no_gate|constant_column"
          first := note first s!"{port}|no_gate|constant_column"
        else
          let col := R0.stackCol c.upv k oc
          n := bump n s!"{port}|gate"
          -- `outGate` builds the unit's rows at each call: asked of one bit in 4096, beside this probe's own test
          if (16 * j + t) % 4096 == 0 then
            n := bump n s!"{port}|outGate_asked"
            if (Spot.Bits.outGate c 1 0 k j t).isSome then n := bump n s!"{port}|outGate_some"
          let own := ownOf v col
          n := bump n s!"{port}|own|{own}"
          first := note first s!"{port}|own|{own}"
          let kind := kindOf v 64 col false
          what := kind.name
          n := bump n s!"{port}|resolved|{kind.name}"
          first := note first s!"{port}|resolved|{kind.name}"
          match kind with
          | .input x _ =>
            let (from_, same) := match Spot.Bits.srcPos c 0 (Spot.Bits.srcKey c ix x) with
              | some r => (r.row, r.bit == bit)
              | none => ("(no row port)", false)
            what := s!"{kind.name}:{from_}{if same then "=" else "~"}"
            n := bump n s!"{port}|from|{from_}"
            first := note first s!"{port}|from|{from_}"
            if same then n := bump n s!"{port}|from_same_bit|{from_}"
          | _ => pure ()
        let rs := runs.getD port #[]
        runs := runs.insert port <| match rs.back? with
          | some (w, s, len) => if w == what && s + len == bit then rs.pop.push (w, s, len + 1) else rs.push (what, bit, 1)
          | none => rs.push (what, bit, 1)
  let obj := fun (m : Std.HashMap String Nat) =>
    Json.mkObj ((m.toList.mergeSort fun a b => decide (a.1 ≤ b.1)).map fun (k, v) => (k, Lean.toJson v))
  let runsJ := Json.mkObj ((runs.toList.mergeSort fun a b => decide (a.1 ≤ b.1)).map fun (port, rs) =>
    (port, Json.mkObj [("runs", Lean.toJson rs.size), ("first", Json.arr ((rs.extract 0 400).map fun (w, s, len) =>
      Json.arr #[.str w, Lean.toJson s, Lean.toJson len]))]))
  return Json.mkObj [("counts", obj n), ("first_bit", obj first), ("runs", runsJ)]

def optStr : Option String → Json
  | some s => .str s
  | none => .null

def optNat : Option Nat → Json
  | some n => Lean.toJson n
  | none => .null

/-- One file's line: `(json, accepted, parsed)`. -/
def one (P : Flock.Program.Prog) (path : String) : IO (Json × Bool × Bool) := do
  let bytes ← IO.FS.readBinFile path
  let tags := Flock.Zk.tags Flock.Tags.circuit (Flock.Zk.maskLogOf bytes) 1
  let t0 ← IO.monoMsNow
  let parsed ← IO.lazyPure fun _ => Flock.ProvedScope.circuitOf tags bytes []
  let t1 ← IO.monoMsNow
  match parsed with
  | .error e =>
    return (Json.mkObj [("file", .str path), ("bytes", Lean.toJson bytes.size), ("parse", .str e)], false, false)
  | .ok c0 =>
    let c := Flock.HmOut.unrolled c0
    let tmpl := (Flock.HmRow.queryTemplate c).map (·.1)
    let key := tmpl.getD ""
    let d? := P.defs.find? (·.key == key)
    let root := P.defs.getD P.root default
    let calls := root.body.nodes.toList.filterMap fun x =>
      if (P.defs.getD x.fn default).key == key then some (Json.mkObj [("form", .str x.form.name), ("n", Lean.toJson x.n)])
      else none
    let (why, checkMs) ← timed fun _ => Spot.Bits.check c P key
    let (reads, readsMs) ← timed fun _ => Spot.Bits.unitReads c
    let (outs, outsMs) ← timed fun _ => outsOf c
    let R := Spot.Bits.Rows.ofNet c.unit
    let params := match d? with
      | some d => (unreadOf c d reads).toList.map fun (x : String × Nat × Nat × Option Nat) =>
          Json.mkObj [("name", .str x.1), ("bits", Lean.toJson x.2.1), ("words", Lean.toJson ((x.2.1 + 15) / 16)),
            ("unread", Lean.toJson x.2.2.1), ("first_unread", optNat x.2.2.2),
            ("readsWhy", optStr (Spot.Bits.readsWhy c d reads [x.1]))]
      | none => []
    let ret := match d? with
      | some d => (Spot.Bits.fieldsOf d.ret).map fun t =>
          Json.mkObj [("bits", Lean.toJson (Spot.Bits.bitsOf t)), ("words", Lean.toJson (Spot.Bits.wordsOf t))]
      | none => []
    let padding := match d? with
      | some d => optStr (Spot.Bits.readsWhy c d reads [])
      | none => .null
    let j := Json.mkObj [
      ("file", .str path), ("bytes", Lean.toJson bytes.size), ("name", .str c.name),
      ("templated", .bool c0.tmpl.isSome), ("template", optStr tmpl),
      ("definition_of_program", .bool d?.isSome), ("root_calls", Json.arr calls.toArray),
      ("program_sha512_meta", (c.metaJ.getObjVal? "program_sha512").toOption.getD .null),
      ("upv", Lean.toJson c.upv), ("vus_per_block", Lean.toJson c.g), ("k_log", Lean.toJson c.kLog),
      ("nets", Lean.toJson c.nets.size),
      ("ports", Json.arr (c.ports.map fun p => Json.arr #[.str p.name, Lean.toJson p.words])),
      ("unit_in_columns", Lean.toJson R.nIn), ("unit_rows", Lean.toJson R.nComp),
      ("unit_in_ports", Lean.toJson (Flock.portCols c.unit.inGroups).size),
      ("check", optStr why), ("accepted", .bool why.isNone),
      ("params", Json.arr params.toArray), ("return_fields", Json.arr ret.toArray),
      ("reads_padding", padding),
      ("outputs", outs),
      ("read_columns", Lean.toJson (reads.foldl (fun n b => if b then n + 1 else n) 0)),
      ("parse_ms", Lean.toJson (t1 - t0)), ("check_ms", Lean.toJson checkMs), ("unit_reads_ms", Lean.toJson readsMs),
      ("outputs_ms", Lean.toJson outsMs)]
    return (j, why.isNone, true)

def run (args : List String) : IO UInt32 := do
  match args with
  | program :: files =>
    let pbytes ← IO.FS.readBinFile program
    match Flock.Program.decode pbytes with
    | .error e =>
      IO.println s!"BITSCHECK-RESULT {(Json.mkObj [("program", .str e)]).compress}"
      return 2
    | .ok P =>
      let root := P.defs.getD P.root default
      let nodes := root.body.nodes.toList.map fun x =>
        Json.mkObj [("form", .str x.form.name), ("n", Lean.toJson x.n), ("fn", .str (P.defs.getD x.fn default).key)]
      IO.println s!"BITSCHECK-PROGRAM {(Json.mkObj [("sha512", .str (Flock.hexOf P.sha512)),
        ("definitions", Lean.toJson P.defs.size), ("root", .str root.key),
        ("root_nodes", Json.arr nodes.toArray)]).compress}"
      let mut accepted := 0
      let mut refused := 0
      let mut unparsed := 0
      for f in files do
        let (j, ok, parsed) ← one P f
        IO.println s!"BITSCHECK {j.compress}"
        (← IO.getStdout).flush
        if !parsed then unparsed := unparsed + 1
        else if ok then accepted := accepted + 1
        else refused := refused + 1
      IO.println s!"BITSCHECK-RESULT {(Json.mkObj [("files", Lean.toJson files.length),
        ("accepted", Lean.toJson accepted), ("refused", Lean.toJson refused),
        ("unparsed", Lean.toJson unparsed)]).compress}"
      return if unparsed > 0 then 2 else if refused > 0 then 1 else 0
  | [] =>
    IO.eprintln "usage: bits-check-probe PROGRAM CIRCUIT..."
    return 2

end Spot.BitsCheckProbe

def main (args : List String) : IO UInt32 := Spot.BitsCheckProbe.run args
