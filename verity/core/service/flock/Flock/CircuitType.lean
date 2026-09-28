/-
Circuit types (Project doc `constant-api-public.md` §2.2; the reference is `verity_flock.circuit_types`): each distinct
sub-circuit once, as its own ANDs over linear forms, its table reads and calls into its parts. The content is canonical JSON,
and the digest is SHA-512 of `"verity/circuit-type/v1" ‖ 0 ‖ canon(content)`.

`WellFormed` states the format's rules once, as named clauses over a type and the types it calls; `check` decides it and
names the first clause that fails. The two parsers decide the same predicate (`circuit_type_agree.py` holds the Python
reference to it), and the private track's registration predicate proves it on the committed circuit. Two of the design's
parser checks hold by construction here: a call binds exactly its callee's inputs, and a caller reads only wires, so never a
bit inside a callee.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Hash
import Flock.Library

namespace Flock

open Lean (Json)

/-- A linear form: the XOR of the wires `ws` and the constant `c`. -/
structure Form where
  ws : Array Nat
  c : Nat
  deriving Inhabited, BEq, Repr

/-- An item appends wires: an AND of two forms (one wire), a call of `calls[k]` on its input forms (the callee's output
bits), a read of `tables[t]` at its index forms (the table's value bits). Any other op code is kept as `bad`, never valid. -/
inductive Item where
  | and (a b : Form)
  | call (k : Nat) (args : Array Form)
  | read (t : Nat) (idx : Array Form)
  | bad (op : Nat)
  deriving Inhabited, Repr

structure CircuitType where
  label : String
  ins : Array (Nat × Nat)
  outs : Array (Nat × Nat)
  calls : Array String
  tables : Array (String × Nat × Nat)
  items : Array Item
  outputs : Array Form
  deriving Inhabited, Repr

namespace CircuitType

def FORMAT : String := "verity/circuit-type/v1"

def portBits (ps : Array (Nat × Nat)) : Nat := ps.foldl (fun a p => a + p.1 * p.2) 0
def inBits (t : CircuitType) : Nat := portBits t.ins
def outBits (t : CircuitType) : Nat := portBits t.outs

/-! ## Content -/

private def ctx : String := "circuit type"

def formOfJson (j : Json) : V Form := do
  let a ← J.asArr j s!"{ctx}: a form"
  if a.size != 2 then throw s!"{ctx}: a form is not [wires, constant]"
  let ws ← (← J.asArr a[0]! s!"{ctx}: a form's wires").mapM (J.asNat · s!"{ctx}: a wire")
  return ⟨ws, ← J.asNat a[1]! s!"{ctx}: a form's constant"⟩

def itemOfJson (j : Json) : V Item := do
  let a ← J.asArr j s!"{ctx}: an item"
  if a.size == 0 then throw s!"{ctx}: an empty item"
  let op ← J.asNat a[0]! s!"{ctx}: an item's op"
  if op == 0 then
    if a.size != 3 then throw s!"{ctx}: an AND is not [0, form, form]"
    return .and (← formOfJson a[1]!) (← formOfJson a[2]!)
  if op == 3 || op == 4 then
    if a.size != 3 then throw s!"{ctx}: a call or read is not [op, index, forms]"
    let k ← J.asNat a[1]! s!"{ctx}: a call's or read's index"
    let fs ← (← J.asArr a[2]! s!"{ctx}: a call's or read's forms").mapM formOfJson
    return if op == 3 then .call k fs else .read k fs
  return .bad op

private def portsOfJson (a : Array Json) (what : String) : V (Array (Nat × Nat)) := a.mapM fun p => do
  let q ← J.asArr p s!"{ctx}: {what}"
  if q.size != 2 then throw s!"{ctx}: {what} is not [count, width]"
  return (← J.asNat q[0]! s!"{ctx}: {what}", ← J.asNat q[1]! s!"{ctx}: {what}")

def ofJson (j : Json) : V CircuitType := do
  if (← J.str j "format" ctx) != FORMAT then throw s!"{ctx}: not {FORMAT}"
  let tables ← (← J.arr j "tables" ctx).mapM fun e => do
    let q ← J.asArr e s!"{ctx}: a table"
    if q.size != 3 then throw s!"{ctx}: a table is not [sha512, index_bits, value_bits]"
    return (← J.asStr q[0]! s!"{ctx}: a table", ← J.asNat q[1]! s!"{ctx}: a table", ← J.asNat q[2]! s!"{ctx}: a table")
  return { label := ← J.str j "label" ctx, ins := ← portsOfJson (← J.arr j "in" ctx) "an input port",
           outs := ← portsOfJson (← J.arr j "out" ctx) "an output port",
           calls := ← (← J.arr j "calls" ctx).mapM (J.asStr · s!"{ctx}: a call"), tables,
           items := ← (← J.arr j "items" ctx).mapM itemOfJson, outputs := ← (← J.arr j "outputs" ctx).mapM formOfJson }

def formJson (f : Form) : Json := Json.arr #[Json.arr (f.ws.map Lean.toJson), Lean.toJson f.c]

def itemJson : Item → Json
  | .and a b => Json.arr #[Lean.toJson (0 : Nat), formJson a, formJson b]
  | .call k fs => Json.arr #[Lean.toJson (3 : Nat), Lean.toJson k, Json.arr (fs.map formJson)]
  | .read k fs => Json.arr #[Lean.toJson (4 : Nat), Lean.toJson k, Json.arr (fs.map formJson)]
  | .bad op => Json.arr #[Lean.toJson op]

/-- The content, as `verity_flock.circuit_types.CircuitType.content` writes it. -/
def toJson (t : CircuitType) : Json :=
  let ports := fun (ps : Array (Nat × Nat)) => Json.arr (ps.map fun p => Json.arr #[Lean.toJson p.1, Lean.toJson p.2])
  Json.mkObj [("format", Json.str FORMAT), ("label", Json.str t.label), ("in", ports t.ins), ("out", ports t.outs),
              ("calls", Json.arr (t.calls.map Json.str)),
              ("tables", Json.arr (t.tables.map fun e => Json.arr #[Json.str e.1, Lean.toJson e.2.1, Lean.toJson e.2.2])),
              ("items", Json.arr (t.items.map itemJson)), ("outputs", Json.arr (t.outputs.map formJson))]

/-- `SHA-512("verity/circuit-type/v1" ‖ 0 ‖ canon(content))`, as lowercase hex. -/
def digest (t : CircuitType) : V String := do
  let s ← canon t.toJson
  return hexOf (Sha512.hash (FORMAT.toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8))

/-! ## The rules -/

def isHex128 (s : String) : Bool := s.length == 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

def distinct (a : Array String) : Bool :=
  (List.range a.size).all fun i => (List.range i).all fun j => a[j]! != a[i]!

/-- A form over `n` wires: constant 0 or 1, and ascending (so distinct) wires below `n`. -/
def formOk (n : Nat) (f : Form) : Bool :=
  f.c ≤ 1 && f.ws.all (· < n) && (List.range (f.ws.size - 1)).all fun i => f.ws[i]! < f.ws[i + 1]!

def itemForms : Item → Array Form
  | .and a b => #[a, b]
  | .call _ fs => fs
  | .read _ fs => fs
  | .bad _ => #[]

/-- The wires an item appends, given the types it may call; `none` for a bad op, a call or read of a missing entry, or a
callee not given. -/
def width (callee : String → Option CircuitType) (t : CircuitType) : Item → Option Nat
  | .and _ _ => some 1
  | .call k _ => (t.calls[k]?).bind callee |>.map outBits
  | .read k _ => (t.tables[k]?).map fun e => e.2.2
  | .bad _ => none

/-- Every item's first wire, and the wires in all: `in_bits`, then the running sum of the widths. -/
def spans (callee : String → Option CircuitType) (t : CircuitType) : Option (Array Nat × Nat) :=
  t.items.foldl (fun acc it => acc.bind fun (s, n) => (width callee t it).map fun w => (s.push n, n + w))
    (some (#[], t.inBits))

def labelOk (t : CircuitType) : Bool := t.label == LIBRARY || t.label == PROGRAM

def portsOk (t : CircuitType) : Bool := (t.ins ++ t.outs).all fun p => p.2 ≥ 1

def callsOk (t : CircuitType) : Bool := t.calls.all isHex128 && distinct t.calls

/-- Each table is `[sha512, index_bits, value_bits]` within 1..32, listed once; a library table has the library's shape;
a library type reads only library tables. -/
def tablesOk (t : CircuitType) : Bool :=
  distinct (t.tables.map (·.1)) && t.tables.all fun (s, ib, vb) =>
    isHex128 s && 1 ≤ ib && ib ≤ 32 && 1 ≤ vb && vb ≤ 32 &&
      match libraryTable? s with
      | some lt => lt.indexBits == ib && lt.valueBits == vb
      | none => t.label != LIBRARY

def resolves (callee : String → Option CircuitType) (t : CircuitType) : Bool := (spans callee t).isSome

/-- Each item reads only earlier wires; a call passes exactly its callee's input bits, and a library type calls only library
types; a read passes exactly its table's index bits. -/
def itemsOk (callee : String → Option CircuitType) (t : CircuitType) : Bool :=
  match spans callee t with
  | none => false
  | some (starts, _) => (List.range t.items.size).all fun i =>
    let it := t.items[i]!
    (itemForms it).all (formOk starts[i]!) &&
      match it with
      | .and _ _ => true
      | .call k fs => match (t.calls[k]?).bind callee with
        | some u => fs.size == u.inBits && (t.label != LIBRARY || u.label == LIBRARY)
        | none => false
      | .read k fs => match t.tables[k]? with
        | some e => fs.size == e.2.1
        | none => false
      | .bad _ => false

/-- The calls and tables in order of their first use. -/
def firstUses (t : CircuitType) : Array Nat × Array Nat :=
  t.items.foldl (fun (cs, ts) it => match it with
    | .call k _ => (if cs.contains k then cs else cs.push k, ts)
    | .read k _ => (cs, if ts.contains k then ts else ts.push k)
    | _ => (cs, ts)) (#[], #[])

def firstUseOk (t : CircuitType) : Bool :=
  (firstUses t).1 == (Array.range t.calls.size) && (firstUses t).2 == (Array.range t.tables.size)

def outputsOk (callee : String → Option CircuitType) (t : CircuitType) : Bool :=
  match spans callee t with
  | none => false
  | some (_, n) => t.outputs.size == t.outBits && t.outputs.all (formOk n)

/-- No dead item: walking back from the outputs, every item makes a wire something later reads. -/
def liveOk (callee : String → Option CircuitType) (t : CircuitType) : Bool :=
  match spans callee t with
  | none => false
  | some (starts, n) => Id.run do
    let mut used := Array.replicate n false
    for f in t.outputs do
      for w in f.ws do used := used.set! w true
    let mut ok := true
    for j in [0:t.items.size] do
      let i := t.items.size - 1 - j
      let lo := starts[i]!
      let w := (width callee t t.items[i]!).getD 0
      if !(List.range w).any (fun d => used[lo + d]!) then ok := false
      for f in itemForms t.items[i]! do
        for x in f.ws do used := used.set! x true
    return ok

/-- **The format's rules**, given the types a type calls (`callee`, already well formed). -/
def WellFormed (callee : String → Option CircuitType) (t : CircuitType) : Prop :=
  labelOk t = true ∧ portsOk t = true ∧ callsOk t = true ∧ tablesOk t = true ∧ resolves callee t = true ∧
    itemsOk callee t = true ∧ firstUseOk t = true ∧ outputsOk callee t = true ∧ liveOk callee t = true

instance instDecidableWellFormed (callee : String → Option CircuitType) (t : CircuitType) : Decidable (WellFormed callee t) := by
  unfold WellFormed; infer_instance

/-- The first rule `t` breaks, by name. -/
def failure (callee : String → Option CircuitType) (t : CircuitType) : String :=
  if !labelOk t then "label" else if !portsOk t then "ports" else if !callsOk t then "calls"
  else if !tablesOk t then "tables" else if !resolves callee t then "resolves" else if !itemsOk callee t then "items"
  else if !firstUseOk t then "first-use" else if !outputsOk callee t then "outputs" else "dead"

/-- Decide `WellFormed`: `.ok ()` exactly when it holds (`check_ok`), else the first rule that fails. -/
def check (callee : String → Option CircuitType) (t : CircuitType) : Except String Unit :=
  if WellFormed callee t then .ok () else .error (failure callee t)

theorem check_ok {callee : String → Option CircuitType} {t : CircuitType} (h : check callee t = .ok ()) :
    WellFormed callee t := by
  unfold check at h
  split at h
  · assumption
  · cases h

/-- `flock-verify circuit-type FILE...`: each FILE is `{"types": [content, ...]}`, callees before callers. Prints one line
`CIRCUITTYPE {"file", "index", "digest", "ok", "error"}` per type. A type is a callee of later ones only once accepted. -/
def cmd (paths : List String) : IO UInt32 := do
  for path in paths do
    let text ← try IO.FS.readFile path catch _ => pure ""
    let base := fun (i : Nat) => [("file", Json.str path), ("index", Lean.toJson i)]
    match Json.parse text >>= fun j => j.getObjValAs? (Array Json) "types" with
    | .error e =>
      IO.println s!"CIRCUITTYPE {(Json.mkObj [("file", Json.str path), ("ok", Json.bool false), ("error", Json.str s!"not a types file: {e}")]).compress}"
    | .ok ts =>
      let mut known : Array (String × CircuitType) := #[]
      for i in [0:ts.size] do
        match ofJson ts[i]! with
        | .error e => IO.println s!"CIRCUITTYPE {(Json.mkObj (base i ++ [("ok", Json.bool false), ("error", Json.str e)])).compress}"
        | .ok t =>
          match t.digest with
          | .error e => IO.println s!"CIRCUITTYPE {(Json.mkObj (base i ++ [("ok", Json.bool false), ("error", Json.str e)])).compress}"
          | .ok d =>
            let callee := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
            match check callee t with
            | .ok () =>
              known := known.push (d, t)
              IO.println s!"CIRCUITTYPE {(Json.mkObj (base i ++ [("digest", Json.str d), ("ok", Json.bool true)])).compress}"
            | .error e =>
              IO.println s!"CIRCUITTYPE {(Json.mkObj (base i ++ [("digest", Json.str d), ("ok", Json.bool false), ("error", Json.str e)])).compress}"
  return 0

end CircuitType

end Flock
