/-
`deriveAll` (Project doc `verified-lowering-design.md` §1.2; S3a): every layout of a statement derived once, callees
first, and the unit's rows in unit coordinates, as the prover's mirror (`verity_flock.derive`) does and its vectors pin
(`backends/flock/tests/derive_vectors.json`).

* An inner layout's own region: its input groups with exported bits left out, its items in order (one row per AND, an
  inlined callee's rows at its item), its output copy rows, its constant. Each placed callee's rows sit at `q · 2^s`,
  its non-exported inputs as binding copies `form · [callee constant]` and its constant `[c] · [c]` of the caller's.
* The unit's layout: its own region likewise; each placed callee in the block range after it, aligned, in item order;
  its bindings and every part's constant (to `one`) in Δ.
* An input bit is exported to a placed call when its form is one input wire the caller reads nowhere else; the callee's
  input row is then that wire's column.

Reads come with S3b. `Flock.Derive.derive` is the flat case.
-/
import Flock.Derive
import Flock.Layout

namespace Flock.DeriveAll

open Lean (Json)

def WORD : Nat := 128

/-- A form over columns: ascending distinct columns and its constant. -/
abbrev CF := List Nat × Nat

def xorCols : List Nat → List Nat → List Nat
  | [], m => m
  | l, [] => l
  | a :: l, b :: m =>
    if a < b then a :: xorCols l (b :: m) else if b < a then b :: xorCols (a :: l) m else xorCols l m

def xorCF (a b : CF) : CF := (xorCols a.1 b.1, (a.2 + b.2) % 2)

/-- A form in the caller's wires, as columns: the XOR of its wires' forms, and its constant. -/
def formOf (wire : Array CF) (f : Form) : CF := f.ws.foldl (fun acc w => xorCF acc (wire.getD w ([], 0))) ([], f.c % 2)

/-- A row side: columns, and whether it holds the constant (resolved once the constant's column is known). -/
abbrev Ent := List Nat × Bool

def ent (f : CF) : Ent := (f.1, f.2 == 1)

def roundUp (r size : Nat) : Nat := (r + size - 1) / size * size

def span (n : Nat) : Nat := WORD * Derive.ceilPow2 (max 1 ((n + WORD - 1) / WORD))

/-- A logical item: an own row, or a placed entry's segment (its layout, base and exported bits). -/
inductive Ref where
  | row (c : Nat)
  | seg (j base : Nat) (exported : List Nat)
  deriving Inhabited, Repr

/-- A layout's rows, in its own coordinates (the unit's: unit coordinates). -/
structure D where
  rows : Array (List Nat × List Nat) := #[]
  const : Nat := 0
  inGroups : Array (Nat × Nat × Array Nat) := #[]
  outGroups : Array (Nat × Nat × Array Nat) := #[]
  inCols : Array Nat := #[]
  outCols : Array Nat := #[]
  items : Array Ref := #[]
  own : Nat := 0
  size : Nat := 0
  parts : Array (Nat × Nat) := #[]
  /-- The unit's Δ: a row and its source (`none`: `one`). -/
  delta : Array (Nat × Option (List Nat)) := #[]
  deriving Inhabited

/-- The builder of one layout's own region. -/
structure Build where
  rows : Array (Ent × Ent) := #[]
  items : Array Ref := #[]

def Build.row (b : Build) (a c : Ent) : Build × Nat := ({ b with rows := b.rows.push (a, c) }, b.rows.size)

def Build.zeros (b : Build) (upTo : Nat) : Build :=
  (List.range (upTo - b.rows.size)).foldl (fun b _ => (b.row ([], false) ([], false)).1) b

structure Ctx where
  types : String → Option CircuitType
  ls : Array Layout
  info : String → Option TableInfo
  done : Array D

/-- A placed entry: its item, layout, base, argument forms and exported bits. -/
structure Inst where
  item : Nat
  j : Nat
  base : Nat
  exported : List Nat
  forms : Array CF := #[]
  deriving Inhabited

/-- Lay `t`'s items out under `l` into `b`, the wires given; a placed call's entry comes from `insts`. Returns the
builder, the wires and the placed entries with their argument forms. Inline calls recurse (`fuel` bounds the depth). -/
partial def items (cx : Ctx) (t : CircuitType) (l : Layout) (b0 : Build) (wire0 : Array CF) (insts0 : Array Inst) :
    Except String (Build × Array CF × Array Inst) := do
  let mut b := b0
  let mut wire := wire0
  let mut insts := insts0
  let mut c := 0
  for i in [0:t.items.size] do
    match t.items[i]! with
    | .and fa fb =>
      let (b', col) := b.row (ent (formOf wire fa)) (ent (formOf wire fb))
      b := { b' with items := b'.items.push (.row col) }
      wire := wire.push ([col], 0)
    | .call k args =>
      let e := l.calls.getD c default
      c := c + 1
      let fs := args.map (formOf wire)
      match e.offset with
      | none =>
        let some u := (t.calls[k]?).bind cx.types | throw s!"derive: a callee of item {i} is not held"
        let (b', w', _) ← items cx u (cx.ls.getD e.layout default) b fs #[]
        b := b'
        let outs := u.outputs.map (formOf w')
        wire := wire ++ outs
      | some _ =>
        let some n := insts.findIdx? (·.item == i) | throw s!"derive: item {i}'s placed entry is missing"
        let inst := insts[n]!
        insts := insts.set! n { inst with forms := fs }
        b := { b with items := b.items.push (.seg inst.j inst.base inst.exported) }
        let sub := cx.done.getD inst.j default
        wire := wire ++ sub.outCols.map fun col => ([inst.base + col], 0)
    | .read _ _ => throw "derive: reads come with S3b"
    | .bad op => throw s!"derive: item {i} has op {op}"
  return (b, wire, insts)

/-- The placed entries of layout `l` of type `t`: in an inner layout at `q · 2^s`, at the unit's in the block ranges after
its own region (`start`), aligned, in item order; each with its exported bits. -/
def placements (cx : Ctx) (t : CircuitType) (l : Layout) (unitLevel : Bool) (start : Nat) :
    Except String (Array Inst × Array (Nat × Nat)) := do
  let u := Layout.uses t
  let mut insts : Array Inst := #[]
  let mut exports : Array (Nat × Nat) := #[]
  let mut cursor := start
  let mut c := 0
  for i in [0:t.items.size] do
    match t.items[i]! with
    | .call _ args =>
      let e := l.calls.getD c default
      c := c + 1
      match e.offset with
      | none => pure ()
      | some q =>
        let s := (cx.ls.getD e.layout default).rangeLog.getD 0
        let base := if unitLevel then roundUp cursor (2 ^ s) else q * 2 ^ s
        if unitLevel then cursor := base + 2 ^ s
        let sub := cx.done.getD e.layout default
        let mut ex : List Nat := []
        for bit in [0:args.size] do
          let f := args[bit]!
          if f.c == 0 && f.ws.size == 1 && f.ws[0]! < t.inBits && u.getD f.ws[0]! 0 == 1 then
            ex := ex ++ [bit]
            exports := exports.push (f.ws[0]!, base + sub.inCols.getD bit 0)
        insts := insts.push { item := i, j := e.layout, base, exported := ex }
    | .read _ _ => throw "derive: reads come with S3b"
    | _ => pure ()
  return (insts, exports)

def resolve (k : Nat) (e : Ent) : List Nat := if e.2 then (e.1 ++ [k]).mergeSort else e.1

/-- **Layout `j`'s rows** (`unitLevel`: the unit's). -/
def deriveOne (cx : Ctx) (j : Nat) (unitLevel : Bool) : Except String D := do
  let l := cx.ls.getD j default
  match l.own with
  | .gen .. => throw "derive: generated reads come with S3b"
  | .derive =>
  let some t := l.type.bind cx.types | throw s!"derive: layout {j}'s type is not held"
  let ex := Layout.exported t l
  let bits := Layout.portBits t.ins
  let gs := Layout.groups bits l.inSplit
  let cuts := #[0] ++ l.inSplit ++ #[bits.size]
  let spans := ((List.range (cuts.size - 1)).filterMap fun i =>
    if cuts[i + 1]! > cuts[i]! then some (cuts[i]!, cuts[i + 1]!) else none).toArray
  let mut b : Build := {}
  let mut inCols : Array Nat := Array.replicate t.inBits 0
  let mut inGroups : Array (Nat × Nat × Array Nat) := #[]
  for g in [0:gs.size] do
    let (first, count) := gs[g]!
    let kept := (List.range count).filterMap fun d => if ex.contains (first + d) then none else some (first + d)
    let size := span kept.length
    let start := roundUp b.rows.size size
    b := b.zeros start
    for (w, i) in kept.zipIdx do inCols := inCols.set! w (start + i)
    let (pa, pb) := spans.getD g (0, 0)
    let s0 := (bits.toList.take pa).sum
    let perPort := ((List.range (pb - pa)).foldl (fun (acc : Array Nat × Nat) p =>
      let wb := bits.getD (pa + p) 0
      (acc.1.push (((List.range wb).filter fun d => !ex.contains (acc.2 + d)).length), acc.2 + wb)) (#[], s0)).1
    inGroups := inGroups.push (start / WORD, size / WORD, perPort)
    for c in [start:start + size] do
      b := (if c - start < kept.length then b.row ([c], false) ([c], false) else b.row ([], false) ([], false)).1
  let mut wire : Array CF := (Array.range t.inBits).map fun w => ([inCols[w]!], 0)
  let startPlaced := Layout.ownSize cx.types cx.ls cx.info j
  let (insts0, exports) ← placements cx t l unitLevel startPlaced
  for (w, col) in exports do
    inCols := inCols.set! w col
    wire := wire.set! w ([col], 0)
  let (b1, wire1, insts) ← items cx t l b wire insts0
  b := b1
  let outs := t.outputs.map (formOf wire1)
  let obits := Layout.portBits t.outs
  let mut outCols : Array Nat := #[]
  for (first, count) in Layout.groups obits l.outSplit do
    let size := span count
    let start := roundUp b.rows.size size
    b := b.zeros start
    for o in [first:first + count] do
      let (b', col) := b.row (ent (outs.getD o default)) ([], true)
      b := b'
      outCols := outCols.push col
    b := b.zeros (start + size)
  let ocuts := #[0] ++ l.outSplit ++ #[obits.size]
  let mut outGroups : Array (Nat × Nat × Array Nat) := #[]
  let mut oi := 0
  for i in [0:ocuts.size - 1] do
    if ocuts[i + 1]! > ocuts[i]! then
      let pb := (obits.toList.drop ocuts[i]!).take (ocuts[i + 1]! - ocuts[i]!)
      let n := pb.sum
      outGroups := outGroups.push (outCols.getD oi 0 / WORD, span n / WORD, pb.toArray)
      oi := oi + n
  let (b2, const) := b.row ([], true) ([], true)
  b := b2
  let own := b.rows.size
  -- the placed entries' rows at their bases; bindings in an inner layout, Δ at the unit's
  let ends := insts.map fun inst => inst.base + 2 ^ ((cx.ls.getD inst.j default).rangeLog.getD 0)
  let size := if unitLevel then own else ends.foldl max own
  let total := ends.foldl max own
  let mut rows : Array (List Nat × List Nat) := Array.replicate total ([], [])
  for r in [0:own] do
    let (a, c) := b.rows[r]!
    rows := rows.set! r (resolve const a, resolve const c)
  let mut delta : Array (Nat × Option (List Nat)) := if unitLevel then #[(const, none)] else #[]
  for inst in insts do
    let sub := cx.done.getD inst.j default
    for col in [0:sub.rows.size] do
      let (a, c) := sub.rows[col]!
      if !(a.isEmpty && c.isEmpty) then rows := rows.set! (inst.base + col) (a.map (· + inst.base), c.map (· + inst.base))
    let cb := inst.base + sub.const
    if unitLevel then delta := delta.push (cb, none) else rows := rows.set! cb ([const], [const])
    for bit in [0:inst.forms.size] do
      if !inst.exported.contains bit then
        let row := inst.base + sub.inCols.getD bit 0
        let f := inst.forms[bit]!
        let src := resolve const (ent f)
        if unitLevel then delta := delta.push (row, some src) else rows := rows.set! row (src, [cb])
  return { rows, const, inGroups, outGroups, inCols, outCols, items := b.items, own, size,
           parts := if unitLevel then insts.map fun inst => (inst.j, inst.base) else #[], delta }

/-- **Every layout derived**, callees first, the unit's last. -/
def deriveAll (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo)
    (unit : Nat) : Except String (Array D) := do
  let mut done : Array D := #[]
  for j in [0:ls.size] do
    let cx : Ctx := { types, ls, info, done }
    let d ← if j == unit then deriveOne cx j true else deriveOne cx j false
    done := done.push d
  return done

/-! ## The texts the vectors pin -/

def line (xs : List String) : String := " ".intercalate xs

def sideStr (l : List Nat) : List String := toString l.length :: l.map toString

/-- A layout's rows as `derive.netlist` writes them: the header, then rows `0 .. size - 1`. -/
def netlist (d : D) (name : String) : String :=
  let side := fun (gs : Array (Nat × Nat × Array Nat)) =>
    toString gs.size :: gs.toList.flatMap fun (w, ws, bits) => ([w, ws, bits.size] ++ bits.toList).map toString
  let head := line (["flock-ir-unit/v2", name, toString d.size, toString d.const] ++ side d.inGroups ++ side d.outGroups)
  let rows := (List.range d.size).map fun c => let (a, b) := d.rows.getD c ([], []); line (sideStr a ++ sideStr b)
  "\n".intercalate (head :: rows) ++ "\n"

/-- The unit's parts and Δ: `PART digest base`, then `DELTA row |src| src..` (`one` as `one`). -/
def deltaText (ls : Array Layout) (digests : Array String) (u : D) : String :=
  let parts := u.parts.toList.map fun (j, base) => line ["PART", digests.getD j "", toString base]
  let ds := u.delta.toList.map fun (c, src) => match src with
    | none => line ["DELTA", toString c, "1", "one"]
    | some s => line (["DELTA", toString c] ++ sideStr s)
  let _ := ls
  "\n".intercalate (parts ++ ds) ++ "\n"

/-- A placed entry's segment in logical order: its constant, its binding copies, its items, its output copies. -/
partial def segment (done : Array D) (d : D) (base : Nat) (exported : List Nat) (top : Bool) : List Nat :=
  let head := [base + d.const] ++ (if top then [] else
    ((List.range d.inCols.size).filter (!exported.contains ·)).map fun i => base + d.inCols[i]!)
  let body := d.items.toList.flatMap fun r => match r with
    | .row c => [base + c]
    | .seg j b ex => segment done (done.getD j default) (base + b) ex false
  head ++ body ++ d.outCols.toList.map (base + ·)

/-- The logical list: the unit's inputs, its segment, then `one`; each row as the block holds it (Δ applied). -/
def logicalText (done : Array D) (u : D) : String :=
  let order := u.inCols.toList ++ segment done u 0 [] true
  let deltaAt := fun (c : Nat) => u.delta.find? (·.1 == c)
  let rowStr := fun (c : Nat) => match deltaAt c with
    | some (_, none) => line [toString c, "1", "one", "1", "one"]
    | some (_, some s) => line ([toString c] ++ sideStr s ++ sideStr s)
    | none => let (a, b) := u.rows.getD c ([], []); line ([toString c] ++ sideStr a ++ sideStr b)
  "\n".intercalate (order.map rowStr ++ [line ["one", "1", "one", "1", "one"]]) ++ "\n"

/-! ## Archives (`flock-verify layout`'s format) -/

/-- An archive's types, tables, layouts and unit: each type and layout checked as `Layout.checkFile` checks it. -/
def load (j : Json) : V ((String → Option CircuitType) × Array Layout × (String → Option TableInfo) × Nat ×
    Array String) := do
  let mut known : Array (String × CircuitType) := #[]
  for c in ← J.arr j "types" "archive" do
    let t ← CircuitType.ofJson c
    let d ← t.digest
    let callee := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
    match CircuitType.check callee t with
    | .ok () => known := known.push (d, t)
    | .error e => throw s!"type {d.take 16}: {e}"
  let mut tabs : Array (String × TableInfo) := #[]
  match (← J.get j "tables" "archive") with
  | Json.obj kvs =>
    for (sha, spec) in kvs.toArray do
      let ws ← (← J.arr spec "words" "archive").mapM (J.asNat · "archive: a table word")
      tabs := tabs.push (sha, Layout.tableInfo ws)
  | _ => throw "archive: tables is not an object"
  let ls ← (← J.arr j "layouts" "archive").mapM Layout.ofJson
  let types := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
  let info := fun (x : String) => (tabs.find? (·.1 == x)).map (·.2)
  let unit ← J.nat j "unit_layout" "archive"
  let _ ← Layout.check types ls unit info
  return (types, ls, info, unit, ← ls.mapM Layout.digest)

/-- The part of an archive's derivation that `part` names: `unit`, `delta`, `logical` or a layout's digest. -/
def archivePart (j : Json) (part : String) : V String := do
  let (types, ls, info, unit, ds) ← load j
  let done ← deriveAll types ls info unit
  let u := done.getD unit default
  if part == "unit" then return netlist u ((ds.getD unit "").take 16).toString
  if part == "delta" then return deltaText ls ds u
  if part == "logical" then return logicalText done u
  match ds.findIdx? (· == part) with
  | some i => return netlist (done.getD i default) (part.take 16).toString
  | none => throw s!"archive: no layout {part.take 16}"

end Flock.DeriveAll
