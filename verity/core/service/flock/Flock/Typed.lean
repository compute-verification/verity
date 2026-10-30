/-
Item 1e (Project doc `verified-lowering-design.md`): the typed statement `verity/flock-circuit/types`
(`verity_flock.typed_statement`), whose unit is a circuit type and a layout instead of rows. No statement carries the
unit's rows: the verifier derives them and places them as the old statement's `unit` slots. It folds rows only through
`DeriveAll.deriveChecked`: `deriveAll`'s rows, used once every layout's pass the item-by-item check and the unit's
are in logical order. `flock-rows --part` prints unchecked rows, so nothing here reads them.

`expand` checks META's new keys and returns the file with the derived rows as its `CIRCUIT unit` section, first, where the
old statement had it. `HmRow.parseTyped` reads that as it reads any statement, and keeps the typed file's own digest and
class. The keys:
* `types`: `{digest: content}`, each the digest of its content and checked by `CircuitType.check`;
* `layouts`: layout records, each its `sha512`'s record, checked by `Layout.check`;
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

/-- The layouts' types by digest. -/
def typeIn (types : Array (String × CircuitType)) (x : String) : Option CircuitType := (types.find? (·.1 == x)).map (·.2)

/-- The held tables' `TableInfo` by SHA-512. -/
def infoIn (held : Array Held) (x : String) : Option TableInfo := (held.find? (·.sha512 == x)).map (·.info)

/-- The held tables' words by SHA-512. -/
def wordsIn (held : Array Held) (x : String) : Option (Array Nat) := (held.find? (·.sha512 == x)).map (·.words)

/-- **A typed statement's unit class,** as the verifier derives it: META's checked types and layouts, the tables it
holds, the unit's layout, and `deriveChecked`'s rows on exactly those, with the run that returned them (`hd`). An accepted
typed statement carries it (`Circuit.cls`), and every net the verifier folds for it is built from `done` (`netOfD`). -/
structure Class where
  types : Array (String × CircuitType)
  ls : Array Layout
  held : Array Held
  unit : Nat
  done : Array DeriveAll.D
  hd : DeriveAll.deriveChecked (typeIn types) ls (infoIn held) (wordsIn held) unit = .ok done

/-- META's types and layouts, checked, and the class's rows through `DeriveAll.deriveChecked` only; with the unit's
type. -/
def classOf (mj : Json) (tables : List (String × ByteArray)) : V (Class × CircuitType) := do
  let kvs ← match mj.getObjVal? "types" with
    | .ok (Json.obj kvs) => pure kvs.toArray
    | _ => throw "META: types is not an object"
  let mut types : Array (String × CircuitType) := #[]
  for (d, c) in kvs do
    let t ← CircuitType.ofJson c
    if (← t.digest) != d then throw s!"META: type {d.take 16}… is not its content's digest"
    types := types.push (d, t)
  for (d, t) in types do
    if let .error e := CircuitType.check (typeIn types) t then throw s!"META: type {d.take 16}… breaks the rule '{e}'"
  let held ← hold types tables
  let ls ← (← J.arr mj "layouts" ctx).mapM fun r => do
    let l ← Layout.ofJson r
    if (← J.str r "sha512" ctx) != (← l.digest) then throw s!"{ctx}: a layout's sha512 is not its record's digest"
    pure l
  let unit ← J.nat mj "unit_layout" ctx
  if let .error e := Layout.check (typeIn types) ls unit (infoIn held) then throw s!"META: the layouts break the rule '{e}'"
  let some ul := ls[unit]? | throw "META: unit_layout names no layout"
  let some ut := ul.type | throw "META: the unit's layout names no type"
  if (← J.str mj "unit_type" ctx) != ut then throw "META: unit_type is not the unit's layout's type"
  if ul.own != .derive then throw "META: the unit's layout is not derived"
  if (← J.str mj "policy" ctx) != POLICY then throw s!"META: policy is not {POLICY}"
  if !(← J.arr mj "unit_in" ctx).isEmpty then
    throw "META: unit_in binds the unit's inputs to constants, which this verifier does not derive yet"
  let some t := typeIn types ut | throw "META: unit_type names no type"
  match h : DeriveAll.deriveChecked (typeIn types) ls (infoIn held) (wordsIn held) unit with
  | .ok done => return (⟨types, ls, held, unit, done, h⟩, t)
  | .error e => throw s!"META: the unit's rows: {e}"

/-- `d`'s rows' A columns, one array per row below its size, in order. -/
def rowsA (d : DeriveAll.D) : Array (Array Nat) := (Array.range d.size).map fun c => (d.rows.getD c ([], [])).1.toArray

/-- `d`'s rows' B columns, one array per row below its size, in order. -/
def rowsB (d : DeriveAll.D) : Array (Array Nat) := (Array.range d.size).map fun c => (d.rows.getD c ([], [])).2.toArray

/-- A layout's net, built straight from `derive`'s rows (no text): its size, constant and port groups, and each row's A
and B columns, as `DeriveAll.netlist` would print them and `Net.parse` read them back. -/
def netOfD (d : DeriveAll.D) (name : String) : V Net :=
  let grp := fun (gs : Array (Nat × Nat × Array Nat)) => gs.map fun (col, words, bits) => ({ col, words, bits } : Group)
  Net.ofRows name d.size d.const (grp d.inGroups) (grp d.outGroups) (rowsA d) (rowsB d)

/-- **The net holds the derived rows:** a net built from `d` has `d`'s rows as its A and B, at its slot size, and `d`'s
size and constant; and the net's checks hold of it: the input rows end before the constant, each is a self row or empty,
the constant row is `[const]·[const]`, and the row order passed. -/
theorem netOfD_ok {d : DeriveAll.D} {name : String} {n : Net} (h : netOfD d name = .ok n) :
    n.unitLog = max 7 (log2ceil d.size) ∧ Sparse.ofRows (rowsA d) (2 ^ n.unitLog) = .ok n.a ∧
      Sparse.ofRows (rowsB d) (2 ^ n.unitLog) = .ok n.b ∧ n.useful = d.size ∧ n.constPos = d.const ∧
      n.inWords * WORD ≤ d.const ∧ (rowsA d).getD d.const #[] = #[d.const] ∧ (rowsB d).getD d.const #[] = #[d.const] ∧
      Net.checkOrder "flock-ir-unit/v2" (n.inWords * WORD) d.const (rowsA d) (rowsB d) = .ok () ∧
      ∀ i < n.inWords * WORD, (((rowsA d).getD i #[] == #[i] && (rowsB d).getD i #[] == #[i]) ||
        (((rowsA d).getD i #[]).isEmpty && ((rowsB d).getD i #[]).isEmpty)) = true := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, h8, h9, h10, h11, h12, -, -⟩ := Net.ofRows_ok h
  exact ⟨h1, h2, h3, h4, h5, h8, h9, h10, h11, h12⟩

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

/-- A template's block as its types derive it (the typed statement's `Instance`), for `HmRow.parse`: the root net (the unit's own region,
its entries at part columns left out, which are Δ's), each placed layout's net by digest, and the block's data in unit
coordinates. -/
structure Template where
  root : Net
  /-- each placed layout, in layout order: its digest, its net, its range's log size -/
  layouts : Array (String × Net × Nat)
  /-- per part, in item order: its layout's position among `layouts` and its base -/
  parts : Array (Nat × Nat)
  own : Nat
  ins : Array (Nat × Nat)
  outs : Array (Nat × Nat)
  inCols : Array Nat
  bind : Array (Nat × Array Nat)
  cross : Array (Nat × Array Nat × Array Nat)
  deriving Inhabited

/-- A placed layout's net: a derived layout's rows as they are; a generated read's (`table/v2`) with its product rows' B
side read from the held table, at its `READ` record's columns and value bits (`k`, not the table's). -/
def placedNet (l : Layout) (d : DeriveAll.D) (nm : String) (held : Array Held) : V Net := do
  match l.own with
  | .gen sha n _ lo =>
    let r ← match d.reads.toList with
      | [r] => pure r
      | _ => throw s!"layout {nm}: a generated layout that is not one read"
    if r.table != sha || r.n != n || r.lo != lo then throw s!"layout {nm}: its read is not its record's"
    let some h := held.find? (·.sha512 == sha) | throw s!"layout {nm}: its table is not held"
    let net ← netOfD d nm
    return { net with lookup := some ⟨h.bytes, r.n, r.lo, r.k, r.prod0, r.loTop⟩ }
  | .derive =>
    if !d.reads.isEmpty then throw s!"layout {nm}: a placed call that reads a table, which this verifier does not fold yet"
    netOfD d nm

/-- The template root's rows: the unit's own, with their entries at part columns (Δ's cross entries) left out. -/
def rootOf (u : DeriveAll.D) : DeriveAll.D :=
  { u with rows := u.rows.map fun (a, b) => (a.filter (· < u.size), b.filter (· < u.size)) }

/-- The unit's own rows' entries at part columns: `(row, A's, B's)` for each row that has any. -/
def crossOf (u : DeriveAll.D) : Array (Nat × Array Nat × Array Nat) :=
  (Array.range u.size).filterMap fun r =>
    let (a, b) := u.rows.getD r ([], [])
    let (xa, xb) := (a.filter (· ≥ u.size), b.filter (· ≥ u.size))
    if xa.isEmpty && xb.isEmpty then none else some (r, xa.toArray, xb.toArray)

/-- Placed layout `j`'s entry in the block: its digest, its net, and its range's log size. -/
def placedEntry (digests : List String) (ls : Array Layout) (done : Array DeriveAll.D) (held : Array Held) (j : Nat) :
    V (String × Net × Nat) := do
  let d := digests.getD j ""
  let l := ls.getD j default
  pure (d, ← placedNet l (done.getD j default) (d.take 16).toString held, l.rangeLog.getD 0)

/-- A template's block from its derivation. -/
def templateOf (name : String) (done : Array DeriveAll.D) (ls : Array Layout) (unit : Nat) (t : CircuitType)
    (held : Array Held) : V Template := do
  let u := done.getD unit default
  if u.reads.any (·.prod0 < u.size) then
    throw "META: the instance type reads a table in its own region, which this verifier's template reading does not fold yet"
  if u.parts.any (·.1 ≥ ls.size) then throw "META: a part places a layout the file does not have"
  let root ← netOfD (rootOf u) name
  let digests ← ls.toList.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let layouts ← placed.mapM (placedEntry digests ls done held)
  return { root, layouts := layouts.toArray, parts := u.parts.map fun (j, b) => (placed.idxOf j, b), own := u.size,
           ins := t.ins, outs := t.outs, inCols := u.inCols, bind := u.delta.filterMap fun (r, src) => src.map (r, ·.toArray),
           cross := crossOf u }

/-- `List.mapM` in `V`: an `ok` result has one entry per input, each the function's `ok` value there. -/
theorem mapM_ok {α β : Type} {f : α → V β} : ∀ {xs : List α} {ys : List β}, xs.mapM f = .ok ys →
    ys.length = xs.length ∧ ∀ i (hx : i < xs.length) (hy : i < ys.length), f xs[i] = .ok ys[i]
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; exact ⟨rfl, fun i hx => absurd hx (by simp)⟩
  | x :: xs, ys, h => by
    simp only [List.mapM_cons, bind, Except.bind, pure, Except.pure] at h
    split at h
    · simp at h
    rename_i y hy
    split at h
    · simp at h
    rename_i ys' hys
    simp only [Except.ok.injEq] at h
    subst h
    obtain ⟨hl, hi⟩ := mapM_ok hys
    refine ⟨by simp [hl], fun i hx hy' => ?_⟩
    cases i with
    | zero => simpa using hy
    | succ i => simpa using hi i (by simpa using hx) (by simpa using hy')

/-- A placed layout's entry: its digest, its range's log size, and its net, `placedNet` of its own derivation. -/
theorem placedEntry_ok {digests : List String} {ls : Array Layout} {done : Array DeriveAll.D} {held : Array Held}
    {j : Nat} {e : String × Net × Nat} (h : placedEntry digests ls done held j = .ok e) :
    e.1 = digests.getD j "" ∧ e.2.2 = (ls.getD j default).rangeLog.getD 0 ∧
      placedNet (ls.getD j default) (done.getD j default) ((digests.getD j "").take 16).toString held = .ok e.2.1 := by
  unfold placedEntry at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  rename_i n hn
  simp only [Except.ok.injEq] at h
  subst h
  exact ⟨rfl, rfl, hn⟩

/-- **A placed layout's net is its derivation's** (`netOfD`), with no lookup; a generated read's adds the one its `READ`
record and held table give (the product rows' table-direct side). -/
theorem placedNet_spec {l : Layout} {d : DeriveAll.D} {nm : String} {held : Array Held} {n : Net}
    (h : placedNet l d nm held = .ok n) :
    ∃ n0, netOfD d nm = .ok n0 ∧ { n with lookup := none } = n0 ∧
      (l.own = .derive → n.lookup = none) ∧
      ∀ sha nb vb lo, l.own = .gen sha nb vb lo → ∃ r hd, d.reads.toList = [r] ∧ held.find? (·.sha512 == sha) = some hd ∧
        n.lookup = some ⟨hd.bytes, r.n, r.lo, r.k, r.prod0, r.loTop⟩ := by
  have hnone : ∀ {n0 : Net}, netOfD d nm = .ok n0 → n0.lookup = none := by
    intro n0 h0; unfold netOfD Net.ofRows at h0
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h0
    repeat' (first | split at h0 | (injection h0 with h0; subst h0))
    all_goals first | exact absurd h0 (by simp) | rfl
  unfold placedNet at h
  split at h
  · rename_i sha nb vb lo hown
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    split at h
    · rename_i r hr
      split at h
      · simp at h
      split at h
      · rename_i hd hhd
        split at h
        · simp at h
        rename_i n0 hn0
        simp only [Except.ok.injEq] at h
        subst h
        have h0 := hnone hn0
        refine ⟨n0, hn0, ?_, by simp [hown], fun sha' nb' vb' lo' h' => ?_⟩
        · cases n0; simp_all
        · rw [hown] at h'; cases h'; exact ⟨r, hd, hr, hhd, rfl⟩
      · simp at h
    · simp at h
  · rename_i hown
    simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    split at h
    · simp at h
    have h0 := hnone h
    refine ⟨n, h, by cases n; simp_all, fun _ => h0, fun _ _ _ _ h' => by simp [hown] at h'⟩

/-- **A template, as `templateOf` builds it** (for audit-lean's `TemplateLayout`): its root is `netOfD` of the unit's own
rows, its cross entries and bound rows are the unit's, its `i`-th placed layout is `placedEntry` of the `i`-th layout the
unit places, and each part names its layout by position among them (every part's layout is placed). -/
theorem templateOf_spec {name : String} {done : Array DeriveAll.D} {ls : Array Layout} {unit : Nat} {ty : CircuitType}
    {held : Array Held} {t : Template} (h : templateOf name done ls unit ty held = .ok t) :
    let u := done.getD unit default
    let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
    ∃ digests : List String, ls.toList.mapM (·.digest) = .ok digests ∧
      netOfD (rootOf u) name = .ok t.root ∧ t.own = u.size ∧ t.inCols = u.inCols ∧ t.ins = ty.ins ∧ t.outs = ty.outs ∧
      t.bind = u.delta.filterMap (fun (r, src) => src.map (r, ·.toArray)) ∧ t.cross = crossOf u ∧
      t.parts = u.parts.map (fun (j, b) => (placed.idxOf j, b)) ∧ (∀ p ∈ u.parts.toList, p.1 ∈ placed) ∧
      t.layouts.size = placed.length ∧
      ∀ i (hi : i < placed.length) (ht : i < t.layouts.size),
        placedEntry digests ls done held placed[i] = .ok t.layouts[i] := by
  intro u placed
  unfold templateOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · simp at h
  split at h
  · simp at h
  rename_i hparts
  split at h
  · simp at h
  rename_i root hroot
  split at h
  · simp at h
  rename_i digests hdig
  split at h
  · simp at h
  rename_i lays hlays
  simp only [Except.ok.injEq] at h
  subst h
  obtain ⟨hlen, hi⟩ := mapM_ok hlays
  have hmem : ∀ p ∈ u.parts.toList, p.1 ∈ placed := by
    intro p hp
    have hp' := Array.mem_toList_iff.1 hp
    have hlt : p.1 < ls.size := Nat.lt_of_not_le fun hge => hparts (Array.any_eq_true'.2 ⟨p, hp', by simpa using hge⟩)
    exact List.mem_filter.2 ⟨List.mem_range.2 hlt, Array.any_eq_true'.2 ⟨p, hp', by simp⟩⟩
  refine ⟨digests, hdig, hroot, rfl, rfl, rfl, rfl, rfl, rfl, rfl, hmem, by rw [List.size_toArray]; exact hlen,
    fun i hx ht => ?_⟩
  rw [List.getElem_toArray]
  exact hi i hx (by rw [List.size_toArray] at ht; exact ht)

/-- A flat unit's net (`netOfD`), whose computed columns (the unit's order after its inputs: its constant, items and
outputs) lie at or past the net's input rows, and whose input columns are the net's input port bits, in order.
`deriveOne` lays the input groups out first and puts input bit `d` of each group at its start plus `d`, so no honest
unit is refused; `Net.ofRows` alone would accept an empty row (an item whose forms cancel) below the input rows, and
`netOfD` takes the unit's input groups as given. -/
def flatNet (done : Array DeriveAll.D) (u : DeriveAll.D) (name : String) : V Net := do
  let net ← netOfD u name
  if ((DeriveAll.order done u).drop u.inCols.size).any (· < net.inWords * Flock.WORD) then
    throw "a flat unit's computed column lies among its input rows"
  if u.inCols.toList != (portCols net.inGroups).toList.flatMap fun p => List.range' p.1 p.2 then
    throw "a flat unit's input columns are not its net's input port bits"
  return net

/-- `flatNet` is `netOfD`, with every computed column at or past the input rows, and the input columns the net's input
port bits in order. -/
theorem flatNet_ok {done : Array DeriveAll.D} {u : DeriveAll.D} {name : String} {net : Net}
    (h : flatNet done u name = .ok net) :
    netOfD u name = .ok net ∧ (∀ c ∈ (DeriveAll.order done u).drop u.inCols.size, net.inWords * Flock.WORD ≤ c) ∧
      u.inCols.toList = (portCols net.inGroups).toList.flatMap fun p => List.range' p.1 p.2 := by
  unfold flatNet at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · simp at h
  rename_i n hn
  split at h
  · simp at h
  rename_i hany
  split at h
  · simp at h
  rename_i hcols
  simp only [Except.ok.injEq] at h
  subst h
  refine ⟨hn, fun c hc => Nat.le_of_not_lt fun hlt => hany ?_, by simpa using hcols⟩
  exact List.any_eq_true.2 ⟨c, hc, by simpa using hlt⟩

/-- **A typed file, read** (item 1e): its class, its rows derived once through `deriveChecked`, and the nets built from
them. A flat class is its unit's net (`.inl`), which `HmRow.parse` places first, where the old statement's `CIRCUIT unit`
section was; a template is its block (`.inr`). -/
def read (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray) := []) : V (Class × (Net ⊕ Template)) := do
  let (_, name, mj) ← split tags bytes
  let (k, t) ← classOf mj tables
  let u := k.done.getD k.unit default
  if isFlat u then return (k, .inl (← flatNet k.done u name))
  return (k, .inr (← templateOf name k.done k.ls k.unit t k.held))

/-- A flat class read: its unit's net is built from the class's own unit rows, the unit's computed columns lie at or past
the net's input rows, and its input columns are the net's input port bits in order (`flatNet`). -/
theorem read_flat {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {k : Class} {net : Net}
    (h : read tags bytes tables = .ok (k, .inl net)) :
    ∃ name, netOfD (k.done.getD k.unit default) name = .ok net ∧
      (∀ c ∈ (DeriveAll.order k.done (k.done.getD k.unit default)).drop (k.done.getD k.unit default).inCols.size,
        net.inWords * Flock.WORD ≤ c) ∧
      (k.done.getD k.unit default).inCols.toList = (portCols net.inGroups).toList.flatMap fun p => List.range' p.1 p.2 := by
  unfold read at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  rename_i s _
  split at h
  · simp at h
  rename_i kt _
  split at h
  · split at h
    · simp at h
    · rename_i v hv
      simp only [Except.ok.injEq, Prod.mk.injEq, Sum.inl.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨s.2.1, flatNet_ok hv⟩
  · split at h <;> simp at h

/-- A template read: its block is `templateOf` on the class's own derivation. -/
theorem read_template {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {k : Class}
    {t : Template} (h : read tags bytes tables = .ok (k, .inr t)) :
    ∃ name ty, templateOf name k.done k.ls k.unit ty k.held = .ok t := by
  unfold read at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  split at h
  · simp at h
  rename_i s _
  split at h
  · simp at h
  rename_i kt _
  split at h
  · split at h <;> simp at h
  · split at h
    · simp at h
    · rename_i v hv
      simp only [Except.ok.injEq, Prod.mk.injEq, Sum.inr.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨s.2.1, kt.2, hv⟩

/-- A template's rows as the verifier derives them, in the writer's `rows/` formats (`typed-rows`): `root.txt` (the unit's
own rows, their entries at part columns included), each placed layout's `<digest>.txt` (a read's with its `READ` line),
and `delta.txt`. -/
def rows (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) : V (Array (String × String)) := do
  let (_, name, mj) ← split tags bytes
  let (k, _) ← classOf mj tables
  let (done, ls) := (k.done, k.ls)
  let u := done.getD k.unit default
  let digests ← ls.mapM (·.digest)
  let placed := (List.range ls.size).filter fun j => u.parts.any (·.1 == j)
  let files := placed.toArray.map fun j =>
    let d := digests.getD j ""
    (s!"{d}.txt", DeriveAll.netlist (done.getD j default) (d.take 16).toString)
  return #[("root.txt", DeriveAll.netlist u name)] ++ files ++ #[("delta.txt", DeriveAll.deltaText ls digests u)]

/-- A flat class's file with its unit's rows derived, as text (`typed-expand --out`, for the tests that compare it with the
old statement byte for byte). -/
def expand (tags : Tags) (bytes : ByteArray) : V ByteArray := do
  let (lines, name, mj) ← split tags bytes
  let (k, _) ← classOf mj []
  let u := k.done.getD k.unit default
  if !isFlat u then throw "META: the unit has placed parts: a template, whose block `HmRow.parse` places, not a flat class"
  return expandWith tags lines name u

end Flock.Typed
