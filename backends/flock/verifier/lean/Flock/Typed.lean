/-
Item 1e (Project doc `verified-lowering-design.md`): the typed statement `verity/flock-circuit/types`
(`verity_flock.typed_statement`, #225), whose unit is a circuit type and a layout instead of rows. No statement carries the
unit's rows: the verifier derives them and places them as the old statement's `unit` slots. It folds rows only through
`DeriveAll.deriveChecked` (#247): `deriveAll`'s rows, used once every layout's pass the item-by-item check and the unit's
are in logical order. `flock-rows --part` prints unchecked rows, so nothing here reads them.

`expand` checks META's new keys and returns the file with the derived rows as its `CIRCUIT unit` section, first, where the
old statement had it. `HmRow.parseTyped` reads that as it reads any statement, and keeps the typed file's own digest and
class. The keys:
* `types`: `{digest: content}`, each the digest of its content and checked by `CircuitType.check` (#194);
* `layouts`: #200's records, each its `sha512`'s record, checked by `Layout.check`;
* `unit_layout`: the unit's layout, derived (`own` is `derive`);
* `unit_type`: that layout's type;
* `policy`: `calls/v1`;
* `unit_in`: the unit's constant-bound inputs, none in this version.

A flat class (no placed parts, Δ or reads) is one `CIRCUIT unit` section. A template is its block (`Template`): the root,
each placed layout's net, and Δ. A placed read's layout is generated (`table/v2`); its product rows' B side is the table,
which the verifier holds (`hold`): the statement names each table by the SHA-512 of its words and never carries them.
-/
import Flock.Comp
import Flock.DeriveCheck
import Flock.Layout
import Flock.Net
import Flock.Tags

namespace Flock.Typed

open Lean (Json)

private def ctx : String := "typed statement META"

def POLICY : String := "calls/v1"

/-- The typed file's lines, its circuit's name and its META. Refuses a file that carries a unit's rows itself. -/
def split (tags : Tags) (bytes : ByteArray) : V (Array String × String × Json) := do
  let text ← match String.fromUTF8? bytes with
    | some s => pure s
    | none => throw "circuit: not UTF-8"
  let lines0 := (text.splitOn "\n").toArray
  let lines := if lines0.back? == some "" then lines0.pop else lines0
  let head := (lines.getD 0 "").trimAsciiEnd.toString
  if !head.startsWith (tags.fileFormat ++ " ") then throw s!"circuit: not a {tags.fileFormat} file"
  let name := (head.drop (tags.fileFormat.length + 1)).toString
  let metaLine := (lines.getD 1 "").trimAsciiEnd.toString
  if !metaLine.startsWith "META " then throw "circuit: no META line"
  let mj ← match Json.parse (metaLine.drop 5).toString with
    | .ok j => pure j
    | .error e => throw s!"META: {e}"
  let mut at_ := 2
  while at_ < lines.size do
    let h := tokens lines[at_]!
    if h.size != 3 || h[0]! != tags.blockKeyword then throw s!"circuit: expected a {tags.blockKeyword} line"
    if h[1]! == "unit" || h[1]! == "root" then
      throw "circuit: a typed statement carries no unit rows; the verifier derives them"
    let some n := h[2]!.toNat? | throw s!"circuit: {tags.blockKeyword} line count"
    at_ := at_ + 1 + n
  return (lines, name, mj)

/-- A table a type reads, as the verifier holds it: its SHA-512 name, its raw `u32` LE words, and their `TableInfo`. -/
structure Held where
  sha512 : String
  bytes : ByteArray
  words : Array Nat
  info : TableInfo
  deriving Inhabited

/-- The tables `types` read, held. Each must be a library table (`LIBRARY_TABLES`, by its SHA-512), given in `tables` under
the library's name, and be `2^index_bits` words whose SHA-512 is the name: checked before any use. -/
def hold (types : Array (String × CircuitType)) (tables : List (String × ByteArray)) : V (Array Held) := do
  let mut out : Array Held := #[]
  for (_, t) in types do
    for (sha, _, _) in t.tables do
      if out.any (·.sha512 == sha) then continue
      let some lt := libraryTable? sha
        | throw s!"META: table {sha.take 16}…: not a library table, the only tables this verifier holds"
      let some (_, bytes) := tables.find? (·.1 == lt.name) | throw s!"table {lt.name}: not given (the verifier's copy, --tables)"
      if bytes.size != 4 * 2 ^ lt.indexBits || hexOf (Sha512.hash bytes) != sha then
        throw s!"table {lt.name}: its words are not the table its SHA-512 names"
      let words := (Array.range (bytes.size / 4)).map fun i => rd32 bytes (4 * i)
      out := out.push ⟨sha, bytes, words, Layout.tableInfo words⟩
  return out

/-- META's types and layouts, checked, and their rows through `DeriveAll.deriveChecked` only: every layout's rows, the
layouts, the unit's layout, its type, and the tables held. -/
def derivation (mj : Json) (tables : List (String × ByteArray)) :
    V (Array DeriveAll.D × Array Layout × Nat × CircuitType × Array Held) := do
  let kvs ← match mj.getObjVal? "types" with
    | .ok (Json.obj kvs) => pure kvs.toArray
    | _ => throw "META: types is not an object"
  let mut types : Array (String × CircuitType) := #[]
  for (d, c) in kvs do
    let t ← CircuitType.ofJson c
    if (← t.digest) != d then throw s!"META: type {d.take 16}… is not its content's digest"
    types := types.push (d, t)
  let typeOf := fun (x : String) => (types.find? (·.1 == x)).map (·.2)
  for (d, t) in types do
    if let .error e := CircuitType.check typeOf t then throw s!"META: type {d.take 16}… breaks the rule '{e}'"
  let held ← hold types tables
  let info := fun (x : String) => (held.find? (·.sha512 == x)).map (·.info)
  let words := fun (x : String) => (held.find? (·.sha512 == x)).map (·.words)
  let ls ← (← J.arr mj "layouts" ctx).mapM fun r => do
    let l ← Layout.ofJson r
    if (← J.str r "sha512" ctx) != (← l.digest) then throw s!"{ctx}: a layout's sha512 is not its record's digest"
    pure l
  let unit ← J.nat mj "unit_layout" ctx
  if let .error e := Layout.check typeOf ls unit info then throw s!"META: the layouts break the rule '{e}'"
  let some ul := ls[unit]? | throw "META: unit_layout names no layout"
  let some ut := ul.type | throw "META: the unit's layout names no type"
  if (← J.str mj "unit_type" ctx) != ut then throw "META: unit_type is not the unit's layout's type"
  if ul.own != .derive then throw "META: the unit's layout is not derived"
  if (← J.str mj "policy" ctx) != POLICY then throw s!"META: policy is not {POLICY}"
  if !(← J.arr mj "unit_in" ctx).isEmpty then
    throw "META: unit_in binds the unit's inputs to constants, which this verifier does not derive yet"
  let some t := typeOf ut | throw "META: unit_type names no type"
  let done ← match DeriveAll.deriveChecked typeOf ls info words unit with
    | .ok done => pure done
    | .error e => throw s!"META: the unit's rows: {e}"
  return (done, ls, unit, t, held)

/-- A flat class's unit: no placed parts, no reads, no Δ but its constant's copy of the block's `one` (which `HmRow.delta`
pins for every unit slot). -/
def isFlat (u : DeriveAll.D) : Bool :=
  u.parts.isEmpty && u.reads.isEmpty && u.delta.all (fun e => e.1 == u.const && e.2.isNone)

/-- A flat class's file with its unit's rows `u` inserted as its `CIRCUIT unit` section, first, where the old statement had
it; META and the other sections unchanged. -/
def expandWith (tags : Tags) (lines : Array String) (name : String) (u : DeriveAll.D) : ByteArray :=
  let rows := ((DeriveAll.netlist u name).splitOn "\n").toArray.pop
  let out := #[lines[0]!, lines[1]!, s!"{tags.blockKeyword} unit {rows.size}"] ++ rows ++ lines.extract 2 lines.size
  ("\n".intercalate out.toList ++ "\n").toUTF8

/-- A template's block as its types derive it (#273's `Instance`), for `HmRow.parse`: the root net (the unit's own region,
its entries at part columns left out, which are Δ's), each placed layout's net by digest, and the block's data in unit
coordinates. -/
structure Template where
  root : Net
  /-- each placed layout, in layout order: its digest, its net, its range's log size -/
  layouts : Array (String × Net × Nat)
  /-- per part, in item order: its layout's digest and its base -/
  parts : Array (String × Nat)
  own : Nat
  ins : Array (Nat × Nat)
  outs : Array (Nat × Nat)
  inCols : Array Nat
  bind : Array (Nat × Array Nat)
  cross : Array (Nat × Array Nat × Array Nat)
  deriving Inhabited

def netOf (text : String) : V Net := Net.parse ((text.splitOn "\n").toArray.pop)

/-- A placed layout's net: a derived layout's rows as they are; a generated read's (`table/v2`) with its product rows' B
side read from the held table, at its `READ` record's columns and value bits (`k`, not the table's). -/
def placedNet (l : Layout) (d : DeriveAll.D) (nm : String) (held : Array Held) : V Net := do
  match l.own with
  | .gen sha n _ lo =>
    let r ← match d.reads with
      | #[r] => pure r
      | _ => throw s!"layout {nm}: a generated layout that is not one read"
    if r.table != sha || r.n != n || r.lo != lo then throw s!"layout {nm}: its read is not its record's"
    let some h := held.find? (·.sha512 == sha) | throw s!"layout {nm}: its table is not held"
    let net ← netOf (DeriveAll.netlist { d with reads := #[] } nm)
    return { net with lookup := some ⟨h.bytes, r.n, r.lo, r.k, r.prod0, r.loTop⟩ }
  | .derive =>
    if !d.reads.isEmpty then throw s!"layout {nm}: a placed call that reads a table, which this verifier does not fold yet"
    netOf (DeriveAll.netlist d nm)

/-- A template's block from its derivation. -/
def templateOf (name : String) (done : Array DeriveAll.D) (ls : Array Layout) (unit : Nat) (t : CircuitType)
    (held : Array Held) : V Template := do
  let u := done.getD unit default
  let own := u.size
  if u.reads.any (·.prod0 < own) then
    throw "META: the instance type reads a table in its own region, which this verifier's template reading does not fold yet"
  let inOwn := fun (xs : List Nat) => xs.filter (· < own)
  let root ← netOf (DeriveAll.netlist { u with rows := u.rows.map fun (a, b) => (inOwn a, inOwn b) } name)
  let digests ← ls.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let layouts ← placed.toArray.mapM fun j => do
    let d := digests.getD j ""
    let l := ls.getD j default
    pure (d, ← placedNet l (done.getD j default) (d.take 16).toString held, l.rangeLog.getD 0)
  let cross := (Array.range own).filterMap fun r =>
    let (a, b) := u.rows.getD r ([], [])
    let (xa, xb) := (a.filter (· ≥ own), b.filter (· ≥ own))
    if xa.isEmpty && xb.isEmpty then none else some (r, xa.toArray, xb.toArray)
  return { root, layouts, parts := u.parts.map fun (j, b) => (digests.getD j "", b), own, ins := t.ins, outs := t.outs,
           inCols := u.inCols, bind := u.delta.filterMap fun (r, src) => src.map (r, ·.toArray), cross }

/-- **A typed file, read** (item 1e): its rows derived once, through `deriveChecked`. A flat class is its file with the
unit's rows as its `CIRCUIT unit` section (`.inl`); a template is its block (`.inr`), which `HmRow.parse` places. -/
def read (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V (ByteArray ⊕ Template) := do
  let (lines, name, mj) ← split tags bytes
  let (done, ls, unit, t, held) ← derivation mj tables
  let u := done.getD unit default
  if isFlat u then return .inl (expandWith tags lines name u)
  return .inr (← templateOf name done ls unit t held)

/-- A template's rows as the verifier derives them, in the writer's `rows/` formats (`typed-rows`): `root.txt` (the unit's
own rows, their entries at part columns included), each placed layout's `<digest>.txt` (a read's with its `READ` line),
and `delta.txt`. -/
def rows (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) : V (Array (String × String)) := do
  let (_, name, mj) ← split tags bytes
  let (done, ls, unit, _, _) ← derivation mj tables
  let u := done.getD unit default
  let digests ← ls.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let files := placed.toArray.map fun j =>
    let d := digests.getD j ""
    (s!"{d}.txt", DeriveAll.netlist (done.getD j default) (d.take 16).toString)
  return #[("root.txt", DeriveAll.netlist u name)] ++ files ++ #[("delta.txt", DeriveAll.deltaText ls digests u)]

/-- A flat class's file with its unit's rows derived (`typed-expand`). -/
def expand (tags : Tags) (bytes : ByteArray) : V ByteArray := do
  match ← read tags bytes with
  | .inl b => pure b
  | .inr _ => throw "META: the unit has placed parts: a template, whose block `HmRow.parse` places, not a flat class"

end Flock.Typed
