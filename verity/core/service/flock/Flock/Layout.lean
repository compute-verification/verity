/-
Layouts (Project doc `constant-api-public.md` §2.2, settled with the soundness lane; the reference is
`verity_flock.layouts`): how a statement places its circuit types. The verifier derives every row from the types and these
records, so a statement carries no rows. This file parses the records and states their rules once, as `WellFormed`: named
decidable clauses over the layouts, beside `CircuitType.WellFormed`. `check` decides it and returns each layout's
own-region size, the count `derive`'s v2 rule makes; `check_ok` says acceptance implies `WellFormed`.

- **An inner layout's own region fills** `[0, ownSize)`:
  - its input port groups, each an aligned power-of-two run of 128-bit words (one word at least), exported bits left out;
  - one row per AND item, the rows of inlined callees and of inline reads at their items;
  - its output port groups, aligned the same way;
  - its constant.

  Each placed callee sits at an aligned offset at or above `ownSize`, inside `2^rangeLog`.
- **The unit's layout has no range:** the block places its own region and each layout it places. Its entries' offsets are
  the slots `0, 1, …` among its placed entries of one layout, in item order.
- **A placed call's input bit is exported** when its form is one input wire of the caller that nothing else there reads.
- **An inlined call names its callee's layout,** which places nothing.
- **A read is tied to its type's table entry** `[sha512, index_bits, value_bits]`: the held table has exactly
  `2^index_bits` words, and a placed read's generated layout has that index and value width.
- **A read's rows are `table/v2`'s,** with product rows for its low `k` value bits: a library table's up to its highest
  varying bit (the rest are constant, and wired), a program table's all, so no shape depends on a program table's values.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Hash
import Flock.CircuitType

namespace Flock

open Lean (Json)

/-- A layout's own rows: derived by the v2 rule from its type, or a generated `table/v2` read. -/
inductive Own where
  | derive
  | gen (table : String) (inBits valueBits loBits : Nat)
  deriving Inhabited, Repr, BEq

/-- A call entry: its item, the callee's layout, and its offset (`none`: inlined). -/
structure CallE where
  item : Nat
  layout : Nat
  offset : Option Nat
  deriving Inhabited, Repr

/-- A read entry: its item, and either a generated layout and offset or, inline, its low decoder width. -/
structure ReadE where
  item : Nat
  placed : Option (Nat × Nat)
  lo : Nat
  deriving Inhabited, Repr

structure Layout where
  type : Option String
  rangeLog : Option Nat
  inSplit : Array Nat
  outSplit : Array Nat
  own : Own
  calls : Array CallE
  reads : Array ReadE
  deriving Inhabited, Repr

/-- A held table's facts: its word count, and the OR over its words of each word XOR the first. -/
structure TableInfo where
  size : Nat
  diff : Nat
  deriving Inhabited, Repr

namespace Layout

def FORMAT : String := "verity/layout/v1"

private def ctx : String := "layout"

private def natArr (j : Json) (what : String) : V (Array Nat) := do
  (← J.asArr j s!"{ctx}: {what}").mapM (J.asNat · s!"{ctx}: {what}")

private def hasOnly (j : Json) (keys : List String) : Bool :=
  match j with
  | Json.obj kvs => kvs.toArray.all fun (k, _) => keys.contains k
  | _ => false

def ofJson (j : Json) : V Layout := do
  if !hasOnly j ["sha512", "type", "range_log", "in_split", "out_split", "own", "calls", "reads"] then
    throw s!"{ctx}: unknown keys"
  let ownJ ← J.get j "own" ctx
  let own ← if hasOnly ownJ ["derive"] && J.has ownJ "derive" then do
      if (← J.str ownJ "derive" ctx) != "v2" then throw s!"{ctx}: own is not derive v2"
      pure Own.derive
    else if hasOnly ownJ ["gen", "table", "in_bits", "value_bits", "lo_bits"] then do
      if (← J.str ownJ "gen" ctx) != "table/v2" then throw s!"{ctx}: own is not gen table/v2"
      pure (Own.gen (← J.str ownJ "table" ctx) (← J.nat ownJ "in_bits" ctx) (← J.nat ownJ "value_bits" ctx)
        (← J.nat ownJ "lo_bits" ctx))
    else throw s!"{ctx}: own is neither derive nor gen"
  let calls ← (← J.arr j "calls" ctx).mapM fun e => do
    if hasOnly e ["item", "layout", "offset"] && J.has e "offset" then
      pure ({ item := ← J.nat e "item" ctx, layout := ← J.nat e "layout" ctx, offset := some (← J.nat e "offset" ctx) } : CallE)
    else if hasOnly e ["item", "layout", "inline"] && J.has e "inline" then do
      if !(← J.bool e "inline" ctx) then throw s!"{ctx}: a call's inline is not true"
      pure ({ item := ← J.nat e "item" ctx, layout := ← J.nat e "layout" ctx, offset := none } : CallE)
    else throw s!"{ctx}: a call entry is neither placed nor inline"
  let reads ← (← J.arr j "reads" ctx).mapM fun e => do
    if hasOnly e ["item", "layout", "offset"] && J.has e "offset" then
      pure ({ item := ← J.nat e "item" ctx, placed := some (← J.nat e "layout" ctx, ← J.nat e "offset" ctx), lo := 0 } : ReadE)
    else if hasOnly e ["item", "inline"] then do
      let inl ← J.get e "inline" ctx
      if !hasOnly inl ["lo_bits"] then throw s!"{ctx}: a read's inline holds more than lo_bits"
      pure ({ item := ← J.nat e "item" ctx, placed := none, lo := ← J.nat inl "lo_bits" ctx } : ReadE)
    else throw s!"{ctx}: a read entry is neither placed nor inline"
  let ty ← if J.has j "type" then some <$> J.str j "type" ctx else pure none
  let rl ← if J.has j "range_log" then some <$> J.nat j "range_log" ctx else pure none
  return { type := ty, rangeLog := rl, inSplit := ← natArr (← J.get j "in_split" ctx) "in_split",
           outSplit := ← natArr (← J.get j "out_split" ctx) "out_split", own, calls, reads }

/-- The record, as `verity_flock.layouts.Layout.record` writes it (without `sha512`). -/
def record (l : Layout) : Json :=
  let own := match l.own with
    | .derive => Json.mkObj [("derive", Json.str "v2")]
    | .gen t n vb lo => Json.mkObj [("gen", Json.str "table/v2"), ("table", Json.str t), ("in_bits", Lean.toJson n),
                                    ("value_bits", Lean.toJson vb), ("lo_bits", Lean.toJson lo)]
  let calls := l.calls.map fun e => match e.offset with
    | some q => Json.mkObj [("item", Lean.toJson e.item), ("layout", Lean.toJson e.layout), ("offset", Lean.toJson q)]
    | none => Json.mkObj [("item", Lean.toJson e.item), ("layout", Lean.toJson e.layout), ("inline", Json.bool true)]
  let reads := l.reads.map fun e => match e.placed with
    | some (j, q) => Json.mkObj [("item", Lean.toJson e.item), ("layout", Lean.toJson j), ("offset", Lean.toJson q)]
    | none => Json.mkObj [("item", Lean.toJson e.item), ("inline", Json.mkObj [("lo_bits", Lean.toJson e.lo)])]
  Json.mkObj ([("in_split", Lean.toJson l.inSplit), ("out_split", Lean.toJson l.outSplit), ("own", own),
               ("calls", Json.arr calls), ("reads", Json.arr reads)] ++
    (match l.type with | some t => [("type", Json.str t)] | none => []) ++
    (match l.rangeLog with | some s => [("range_log", Lean.toJson s)] | none => []))

/-- `SHA-512("verity/layout/v1" ‖ 0 ‖ canon(record))`, as lowercase hex. -/
def digest (l : Layout) : V String := do
  let s ← canon l.record
  return hexOf (Sha512.hash (FORMAT.toUTF8 ++ ByteArray.mk #[0] ++ s.toUTF8))

/-! ## Counting derive's own rows -/

def WORD : Nat := 128

def ceilPow2 (n : Nat) : Nat := 2 ^ (Nat.log2 (2 * n - 1))

/-- The rows of a one-hot decoder of `m` bits (blocks of two or more bits commit their minterms). -/
def decodeRows : Nat → Nat
  | 0 => 0
  | 1 => 0
  | m + 2 => 2 ^ (m + 2) + decodeRows ((m + 2) / 2) + decodeRows ((m + 2) - (m + 2) / 2)
decreasing_by all_goals omega

def readRowsInline (n lo k : Nat) : Nat := decodeRows lo + decodeRows (n - lo) + k * 2 ^ (n - lo) + k

def readRowsPlaced (n lo k : Nat) : Nat :=
  let body := WORD + decodeRows lo + decodeRows (n - lo) + k * 2 ^ (n - lo)
  (body + WORD - 1) / WORD * WORD + WORD + 1

/-- A held table's facts, from its words. -/
def tableInfo (words : Array Nat) : TableInfo :=
  let w0 := words.getD 0 0
  ⟨words.size, words.foldl (fun d w => d ||| (w ^^^ w0)) 0⟩

/-- The value bits a read gives product rows: a library table's up to its highest varying bit, a program table's all. -/
def kOf (sha : String) (valueBits : Nat) (ti : TableInfo) : Nat :=
  if (libraryTable? sha).isSome then
    let d := ti.diff &&& (2 ^ valueBits - 1)
    if d == 0 then 0 else Nat.log2 d + 1
  else valueBits

/-- Each group's (first bit, bit count) over ports of the given bit widths, split at port indices. -/
def groups (bits : Array Nat) (split : Array Nat) : Array (Nat × Nat) := Id.run do
  let cuts := #[0] ++ split ++ #[bits.size]
  let mut starts : Array Nat := #[0]
  for b in bits do starts := starts.push (starts.back! + b)
  let mut out := #[]
  for i in [0:cuts.size - 1] do
    let a := cuts[i]!
    let b := cuts[i + 1]!
    if b > a then out := out.push (starts[a]!, starts[b]! - starts[a]!)
  return out

def portBits (ps : Array (Nat × Nat)) : Array Nat := ps.map fun p => p.1 * p.2

/-- How many times each wire is read by the type's forms and outputs. -/
def uses (t : CircuitType) : Array Nat := Id.run do
  let n := t.inBits + t.items.size * 64 + 64
  let mut u := Array.replicate n 0
  for it in t.items do
    for f in CircuitType.itemForms it do
      for w in f.ws do u := if w < u.size then u.modify w (· + 1) else u
  for f in t.outputs do
    for w in f.ws do u := if w < u.size then u.modify w (· + 1) else u
  return u

/-- The caller's input wires exported to a placed call: a whole single-wire argument that nothing else reads. -/
def exported (t : CircuitType) (l : Layout) : Array Nat := Id.run do
  let u := uses t
  let mut out := #[]
  let mut c := 0
  for it in t.items do
    match it with
    | .call _ fs =>
      let placed := (l.calls.getD c default).offset.isSome
      c := c + 1
      if placed then
        for f in fs do
          if f.c == 0 && f.ws.size == 1 && f.ws[0]! < t.inBits && u.getD f.ws[0]! 0 == 1 then out := out.push f.ws[0]!
    | _ => pure ()
  return out

/-- The own-region rows of a layout's items: ANDs, inlined callees' items (by their layouts) and inline reads. `fuel`
bounds the inlining depth (layouts are listed callees first, so the number of layouts is enough). -/
def itemsRows (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo) :
    Nat → CircuitType → Layout → Nat
  | 0, _, _ => 0
  | fuel + 1, t, l => Id.run do
    let mut n := 0
    let mut c := 0
    let mut r := 0
    for it in t.items do
      match it with
      | .and _ _ => n := n + 1
      | .call _ _ =>
        let e := l.calls.getD c default
        c := c + 1
        if e.offset.isNone then
          let sub := ls.getD e.layout default
          match sub.type.bind types with
          | some u => n := n + itemsRows types ls info fuel u sub
          | none => pure ()
      | .read k _ =>
        let e := l.reads.getD r default
        r := r + 1
        if e.placed.isNone then
          let (sha, ib, vb) := t.tables.getD k ("", 0, 0)
          n := n + readRowsInline ib e.lo (kOf sha vb ((info sha).getD default))
      | .bad _ => pure ()
    return n

/-- Layout `j`'s own-region size under derive's v2 rule (a generated read: all its rows). -/
def ownSize (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo) (j : Nat) : Nat :=
  let l := ls.getD j default
  match l.own with
  | .gen sha n vb lo => readRowsPlaced n lo (kOf sha vb ((info sha).getD default))
  | .derive =>
    match l.type.bind types with
    | none => 0
    | some t => Id.run do
      let ex := exported t l
      let mut rows := 0
      for (first, count) in groups (portBits t.ins) l.inSplit do
        let nbits := ((List.range count).filter fun d => !ex.contains (first + d)).length
        let span := WORD * ceilPow2 (max 1 ((nbits + WORD - 1) / WORD))
        rows := (rows + span - 1) / span * span + span
      rows := rows + itemsRows types ls info ls.size t l
      for (_, count) in groups (portBits t.outs) l.outSplit do
        let span := WORD * ceilPow2 (max 1 ((count + WORD - 1) / WORD))
        rows := (rows + span - 1) / span * span + span
      return rows + 1

/-- Layout `j` places nothing: derived own rows, every call inline (by layouts that place nothing) and every read inline. -/
def placesNothing (ls : Array Layout) : Nat → Nat → Bool
  | 0, _ => false
  | fuel + 1, j =>
    let l := ls.getD j default
    l.own == .derive && l.calls.all (fun e => e.offset.isNone && e.layout < j && placesNothing ls fuel e.layout) &&
      l.reads.all (·.placed.isNone)

/-! ## The rules -/

def increasing (xs : Array Nat) (lo hi : Nat) : Bool :=
  xs.all (fun x => lo ≤ x && x < hi) && (List.range (xs.size - 1)).all fun i => xs[i]! < xs[i + 1]!

/-- The layouts an entry names: every call's, and every placed read's. -/
def named (l : Layout) : Array Nat := l.calls.map (·.layout) ++ l.reads.filterMap (·.placed.map (·.1))

/-- Every layout is reached from the unit's. Entries name layouts listed before them, so one downward sweep marks every
reached layout: mark the unit's, then each marked layout's entries below it. -/
def reachOk (ls : Array Layout) (unit : Nat) : Bool :=
  let marked := (List.range ls.size).reverse.foldl (fun (m : Array Bool) j =>
      if m.getD j false then (named ls[j]!).foldl (fun m jj => if jj < j then m.set! jj true else m) m else m)
    ((Array.replicate ls.size false).set! unit true)
  unit < ls.size && marked.all id

/-- A range on every layout but the unit's, in 7..40. -/
def rangeOk (ls : Array Layout) (unit j : Nat) : Bool :=
  match (ls.getD j default).rangeLog with
  | none => j == unit
  | some s => j != unit && 7 ≤ s && s ≤ 40

/-- A generated read: no type, entries or port groups, not the unit's; widths in 1..32 and a split in range; a
library table in the library's shape; a held table of exactly `2^in_bits` words; its rows inside its range. -/
def genOk (ls : Array Layout) (info : String → Option TableInfo) (unit j : Nat) : Bool :=
  let l := ls.getD j default
  match l.own with
  | .derive => true
  | .gen sha n vb lo =>
    l.type.isNone && j != unit && l.calls.isEmpty && l.reads.isEmpty && l.inSplit.isEmpty && l.outSplit.isEmpty &&
      CircuitType.isHex128 sha && 1 ≤ n && n ≤ 32 && 1 ≤ vb && vb ≤ 32 && 2 ≤ lo && lo < n &&
      (match libraryTable? sha with
       | some lt => lt.indexBits == n && lt.valueBits == vb
       | none => true) &&
      match info sha with
      | some ti => ti.size == 2 ^ n && readRowsPlaced n lo (kOf sha vb ti) ≤ 2 ^ (l.rangeLog.getD 0)
      | none => false

def callItems (t : CircuitType) : Array Nat :=
  (Array.range t.items.size).filter fun i => match t.items[i]! with | .call _ _ => true | _ => false

def readItems (t : CircuitType) : Array Nat :=
  (Array.range t.items.size).filter fun i => match t.items[i]! with | .read _ _ => true | _ => false

/-- A derived layout's entries: its type held; port groups increasing inside the ports; one entry per call and read item,
naming it; each call's layout listed before it and of the callee's type, and an inlined one placing nothing; each read's
table held with exactly `2^index_bits` words for the type's entry, and either inline with its split in range, or placed
at a generated read listed before it of that table, index width and value width. -/
def entriesOk (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo) (j : Nat) : Bool :=
  let l := ls.getD j default
  match l.own with
  | .gen _ _ _ _ => true
  | .derive =>
    match l.type.bind types with
    | none => false
    | some t =>
      let ci := callItems t
      let ri := readItems t
      increasing l.inSplit 1 t.ins.size && increasing l.outSplit 1 t.outs.size &&
        l.calls.size == ci.size && l.reads.size == ri.size &&
        (List.range ci.size).all (fun c =>
          let e := l.calls[c]!
          let callee := match t.items[ci[c]!]! with | .call k _ => t.calls.getD k "" | _ => ""
          let sub := ls.getD e.layout default
          e.item == ci[c]! && e.layout < j && sub.own == .derive && sub.type == some callee &&
            (e.offset.isSome || placesNothing ls ls.size e.layout)) &&
        (List.range ri.size).all (fun r =>
          let e := l.reads[r]!
          let (sha, ib, vb) := match t.items[ri[r]!]! with | .read k _ => t.tables.getD k ("", 0, 0) | _ => ("", 0, 0)
          e.item == ri[r]! && ((info sha).map (·.size) == some (2 ^ ib)) &&
            match e.placed with
            | none => 2 ≤ e.lo && e.lo < ib
            | some (jj, _) => jj < j && match (ls.getD jj default).own with
              | .gen s n v _ => s == sha && n == ib && v == vb
              | .derive => false)

/-- The placed entries of a layout, as (layout, offset), calls then reads in item order. -/
def placedOf (l : Layout) : Array (Nat × Nat) :=
  l.calls.filterMap (fun e => e.offset.map (e.layout, ·)) ++ l.reads.filterMap (·.placed)

/-- The unit's placed entries of one layout take slots `0, 1, …` in item order. -/
def slotsOk (p : Array (Nat × Nat)) : Bool :=
  (List.range p.size).all fun i => p[i]!.2 == ((List.range i).filter fun h => p[h]!.1 == p[i]!.1).length

/-- An inner derived layout: its own region inside its range; each placed range aligned, clear of the own region and inside
the range; no two overlapping. The unit's: its slots. -/
def geometryOk (types : String → Option CircuitType) (ls : Array Layout) (info : String → Option TableInfo) (unit j : Nat) :
    Bool :=
  let l := ls.getD j default
  match l.own with
  | .gen _ _ _ _ => true
  | .derive =>
    let p := placedOf l
    if j == unit then slotsOk p else
    let size := ownSize types ls info j
    let R := 2 ^ (l.rangeLog.getD 0)
    let spans := p.map fun (jj, q) => let s := 2 ^ ((ls.getD jj default).rangeLog.getD 0); (q * s, (q + 1) * s)
    size ≤ R && spans.all (fun sp => size ≤ sp.1 && sp.2 ≤ R) &&
      (List.range spans.size).all fun a => (List.range a).all fun b =>
        !(spans[a]!.1 < spans[b]!.2 && spans[b]!.1 < spans[a]!.2)

/-- **The layouts' rules**, over held types (already well formed) and held tables. -/
def WellFormed (types : String → Option CircuitType) (ls : Array Layout) (unit : Nat) (info : String → Option TableInfo) :
    Prop :=
  reachOk ls unit = true ∧
    ∀ j, j < ls.size → rangeOk ls unit j = true ∧ genOk ls info unit j = true ∧ entriesOk types ls info j = true ∧
      geometryOk types ls info unit j = true

instance instDecidableWellFormed (types : String → Option CircuitType) (ls : Array Layout) (unit : Nat) (info : String → Option TableInfo) :
    Decidable (WellFormed types ls unit info) := by
  unfold WellFormed; infer_instance

/-- The first rule broken, by name. -/
def failure (types : String → Option CircuitType) (ls : Array Layout) (unit : Nat) (info : String → Option TableInfo) :
    String :=
  if !reachOk ls unit then "reach" else
  match (List.range ls.size).find? (fun j => !(rangeOk ls unit j && genOk ls info unit j && entriesOk types ls info j &&
      geometryOk types ls info unit j)) with
  | some j => if !rangeOk ls unit j then s!"layout {j}: range" else if !genOk ls info unit j then s!"layout {j}: gen"
      else if !entriesOk types ls info j then s!"layout {j}: entries" else s!"layout {j}: geometry"
  | none => "none"

/-- Decide `WellFormed`: each layout's own-region size when it holds (`check_ok`), else the first rule broken. -/
def check (types : String → Option CircuitType) (ls : Array Layout) (unit : Nat) (info : String → Option TableInfo) :
    Except String (Array Nat) :=
  if WellFormed types ls unit info then .ok ((Array.range ls.size).map (ownSize types ls info))
  else .error (failure types ls unit info)

theorem check_ok {types : String → Option CircuitType} {ls : Array Layout} {unit : Nat} {info : String → Option TableInfo}
    {sizes : Array Nat} (h : check types ls unit info = .ok sizes) : WellFormed types ls unit info := by
  unfold check at h
  split at h
  · assumption
  · cases h

private def wordsBytes (ws : Array Nat) : ByteArray :=
  ws.foldl (fun b w => b ++ le32 w.toUInt32) ByteArray.empty

/-- One archive: `{"types": [content, ...], "tables": {sha512: {"words": [...]}}, "layouts": [record, ...],
"unit_layout": k}`. Types are checked callees first, tables against their SHA-512, then the layouts. -/
def checkFile (j : Json) : V (Array String × Array Nat) := do
  let mut known : Array (String × CircuitType) := #[]
  for c in ← J.arr j "types" ctx do
    let t ← CircuitType.ofJson c
    let d ← t.digest
    let callee := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
    match CircuitType.check callee t with
    | .ok () => known := known.push (d, t)
    | .error e => throw s!"type {d.take 16}: {e}"
  let mut tabs : Array (String × TableInfo) := #[]
  match (← J.get j "tables" ctx) with
  | Json.obj kvs =>
    for (sha, spec) in kvs.toArray do
      let ws ← (← J.arr spec "words" ctx).mapM (J.asNat · s!"{ctx}: a table word")
      if hexOf (Sha512.hash (wordsBytes ws)) != sha then throw s!"table {sha.take 16}: its words are not its SHA-512"
      tabs := tabs.push (sha, tableInfo ws)
  | _ => throw s!"{ctx}: tables is not an object"
  let ls ← (← J.arr j "layouts" ctx).mapM fun r => do
    let l ← ofJson r
    if (← J.str r "sha512" ctx) != (← l.digest) then throw s!"{ctx}: sha512 is not the record's digest"
    pure l
  let types := fun (x : String) => (known.find? (·.1 == x)).map (·.2)
  let info := fun (x : String) => (tabs.find? (·.1 == x)).map (·.2)
  let sizes ← check types ls (← J.nat j "unit_layout" ctx) info
  return (← ls.mapM digest, sizes)

/-- `flock-verify layout FILE...`: one line `LAYOUT {"file", "ok", "digests", "sizes" | "error"}` per archive. -/
def cmd (paths : List String) : IO UInt32 := do
  for path in paths do
    let text ← try IO.FS.readFile path catch _ => pure ""
    let line := match Json.parse text >>= checkFile with
      | .ok (ds, sizes) => Json.mkObj [("file", Json.str path), ("ok", Json.bool true), ("digests", Lean.toJson ds),
                                       ("sizes", Lean.toJson sizes)]
      | .error e => Json.mkObj [("file", Json.str path), ("ok", Json.bool false), ("error", Json.str e)]
    IO.println s!"LAYOUT {line.compress}"
  return 0

end Layout

end Flock
