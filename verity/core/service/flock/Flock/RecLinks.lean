/-
The link table of a recursive session's inner population (`verity/rec-links/v1`), and the reference check its rows are held
to. Every row of the inner population is committed under its own salt, so no two commitments show that their rows are
equal, and which unit reads which is private. A link says that input row `port` of unit `reader` holds, word for word,
output row `out` of unit `writer`.

The table holds one link a row of `ROW_BYTES` bytes: `KIND_LINK`, `reader`, `port`, `writer` and `out`, each a u32
little-endian, then zeros. The links come first, then all-zero filler rows, `grid L` rows in all, `L` the link count and
`grid` partition v2's public grid (`GRID_BITS` significant bits). The prover commits the rows as advice before any coin
(`verity.ml.commitments.advice`: `hm96-sha512/row/v2` hiding leaves in one `frame-v3-sha512` tree), so what is public of
the table is `grid L` and the root.

`check` is the reference, at the inner population's shape (its unit count and each input and output row's words) and its
rows' values. It decodes every row (`row-size`, `row-kind`, `row-nonzero`), holds the links before the fillers (`order`),
at least one link (`empty`) and `grid L` rows (`count`), then each link in order to the shape (`range`, `width`), to no
input row read twice (`twice`) and to its reader row's words being its writer row's (`unequal`). `check_sound`: whatever it
accepts, its table is its links then fillers, and every link holds. No statement the verifier accepts reaches it: it is
what V*'s link unit is held to, and `verity_flock.rec_links` mirrors it for the prover (`flock-rec-links`, the difftest).
Its witnesses: `check_sound_honest`, `check` accepting the prover's table of one link that holds, and
`check_sound_nontrivial`, its conclusion failing at a table a check that accepts everything accepts.

A recursive statement carries the table's root and row count as `--rec-links`, which `verify` and `statement` refuse
(`refuseLinks`) until a guarantee covers it.
-/

namespace Flock
namespace RecLinks

def FORMAT : String := "verity/rec-links/v1"
def ROW_BYTES : Nat := 128
def GRID_BITS : Nat := 4
def KIND_LINK : Nat := 1
/-- The bytes a link's fields take: its kind and four u32s. -/
def FIELD_BYTES : Nat := 20

/-- Python's `int.bit_length`. -/
def bitLength (x : Nat) : Nat := if x = 0 then 0 else Nat.log2 x + 1

/-- The least `y ≥ x` with at most `bits` significant bits (partition v2's `grid`): `x` below `2 ^ bits`, else `x` rounded
up to a multiple of `2 ^ (bitLength x - bits)`. -/
def grid (x : Nat) (bits : Nat := GRID_BITS) : Nat :=
  let e := bitLength x - bits
  (x + 2 ^ e - 1) / 2 ^ e * 2 ^ e

/-- One entry of the link table: input row `port` of unit `reader` holds output row `out` of unit `writer`. -/
structure LinkEntry where
  reader : Nat
  port : Nat
  writer : Nat
  out : Nat
  deriving Repr, DecidableEq

/-- The inner population's public shape: its units, and each input row's and each output row's words. -/
structure Shape where
  units : Nat
  ins : Array Nat
  outs : Array Nat
  deriving Repr

def u32At (row : ByteArray) (off : Nat) : Nat :=
  (row.get! off).toNat + 256 * (row.get! (off + 1)).toNat + 65536 * (row.get! (off + 2)).toNat +
    16777216 * (row.get! (off + 3)).toNat

/-- A table row: a link, a filler (none), or the refusal. -/
def decode (row : ByteArray) : Except String (Option LinkEntry) :=
  if row.size != ROW_BYTES then .error s!"row-size: a row of {row.size} bytes, not {ROW_BYTES}"
  else
    let kind := u32At row 0
    if kind == 0 then
      if (List.range ROW_BYTES).all fun i => row.get! i == 0 then .ok none
      else .error "row-nonzero: a filler row is not all zero"
    else if kind == KIND_LINK then
      if (List.range (ROW_BYTES - FIELD_BYTES)).all fun i => row.get! (FIELD_BYTES + i) == 0 then
        .ok (some ⟨u32At row 4, u32At row 8, u32At row 12, u32At row 16⟩)
      else .error "row-nonzero: a link row's bytes past its fields are not zero"
    else .error s!"row-kind: a row of kind {kind}, neither a link ({KIND_LINK}) nor a filler (0)"

def le32 (x : Nat) : ByteArray := ⟨#[(x % 256).toUInt8, (x / 256 % 256).toUInt8, (x / 65536 % 256).toUInt8,
  (x / 16777216 % 256).toUInt8]⟩

/-- A link's row (each field below `2 ^ 32`). -/
def encode (l : LinkEntry) : ByteArray :=
  le32 KIND_LINK ++ le32 l.reader ++ le32 l.port ++ le32 l.writer ++ le32 l.out ++
    ⟨Array.replicate (ROW_BYTES - FIELD_BYTES) 0⟩

/-- The prover's table: the links' rows, then fillers up to `grid L`. -/
def table (ls : List LinkEntry) : Array ByteArray :=
  (ls.map encode).toArray ++ Array.replicate (grid ls.length - ls.length) ⟨Array.replicate ROW_BYTES 0⟩

/-- Every row decoded, in order; the first refusal. -/
def decodeAll : List ByteArray → Except String (List (Option LinkEntry))
  | [] => .ok []
  | r :: rs =>
    match decode r with
    | .error e => .error e
    | .ok x =>
      match decodeAll rs with
      | .error e => .error e
      | .ok xs => .ok (x :: xs)

/-- The links come first: no link after a filler. -/
def ordered : List (Option LinkEntry) → Bool
  | [] => true
  | some _ :: xs => ordered xs
  | none :: xs => xs.all Option.isNone

/-- What a link holds at the shape and the rows' values: its units and rows exist, its two rows have the same words, and
its reader row's words are its writer row's. -/
def Holds (sh : Shape) (inRow outRow : Nat → Nat → Array Nat) (l : LinkEntry) : Prop :=
  l.reader < sh.units ∧ l.writer < sh.units ∧ l.port < sh.ins.size ∧ l.out < sh.outs.size ∧
    sh.ins[l.port]! = sh.outs[l.out]! ∧ inRow l.reader l.port = outRow l.writer l.out

/-- Link `k`'s refusal, `seen` the input rows the links before it read. -/
def linkErr (sh : Shape) (inRow outRow : Nat → Nat → Array Nat) (seen : List (Nat × Nat)) (k : Nat) (l : LinkEntry) :
    Option String :=
  if ¬ (l.reader < sh.units ∧ l.writer < sh.units ∧ l.port < sh.ins.size ∧ l.out < sh.outs.size) then
    some s!"range: link {k} reads input row {l.port} of unit {l.reader} as output row {l.out} of unit {l.writer}, outside the population's {sh.units} units, {sh.ins.size} input and {sh.outs.size} output rows"
  else if sh.ins[l.port]! ≠ sh.outs[l.out]! then
    some s!"width: link {k}'s input row {l.port} has {sh.ins[l.port]!} words, its output row {l.out} {sh.outs[l.out]!}"
  else if (l.reader, l.port) ∈ seen then
    some s!"twice: link {k} reads input row {l.port} of unit {l.reader}, which an earlier link reads"
  else if inRow l.reader l.port ≠ outRow l.writer l.out then
    some s!"unequal: link {k}: input row {l.port} of unit {l.reader} is not output row {l.out} of unit {l.writer}"
  else none

/-- Each link in order, `k` the first's index. -/
def links (sh : Shape) (inRow outRow : Nat → Nat → Array Nat) : List (Nat × Nat) → Nat → List LinkEntry → Except String Unit
  | _, _, [] => .ok ()
  | seen, k, l :: ls =>
    match linkErr sh inRow outRow seen k l with
    | some e => .error e
    | none => links sh inRow outRow ((l.reader, l.port) :: seen) (k + 1) ls

/-- **The reference check** of a table's rows at the shape and the rows' values (the module docstring): the link count. -/
def check (sh : Shape) (inRow outRow : Nat → Nat → Array Nat) (rows : Array ByteArray) : Except String Nat :=
  match decodeAll rows.toList with
  | .error e => .error e
  | .ok xs =>
    let ls := xs.filterMap id
    if !ordered xs then .error "order: a link row after a filler row"
    else if ls.isEmpty then .error "empty: a table without links is no table"
    else if rows.size != grid ls.length then
      .error s!"count: {rows.size} rows, not the {ls.length} links' grid {grid ls.length}"
    else
      match links sh inRow outRow [] 0 ls with
      | .error e => .error e
      | .ok () => .ok ls.length

/-! ## What `check` accepts -/

theorem linkErr_none {sh : Shape} {inRow outRow : Nat → Nat → Array Nat} {seen : List (Nat × Nat)} {k : Nat} {l : LinkEntry}
    (h : linkErr sh inRow outRow seen k l = none) : Holds sh inRow outRow l ∧ (l.reader, l.port) ∉ seen := by
  unfold linkErr at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · rename_i h2
      split at h
      · cases h
      · rename_i h3
        split at h
        · cases h
        · rename_i h4
          have h1 := Decidable.not_not.1 h1
          exact ⟨⟨h1.1, h1.2.1, h1.2.2.1, h1.2.2.2, Decidable.not_not.1 h2, Decidable.not_not.1 h4⟩, h3⟩

theorem links_sound {sh : Shape} {inRow outRow : Nat → Nat → Array Nat} :
    ∀ {seen : List (Nat × Nat)} {k : Nat} {ls : List LinkEntry}, links sh inRow outRow seen k ls = .ok () →
      (∀ l ∈ ls, Holds sh inRow outRow l) ∧ (ls.map fun l => (l.reader, l.port)).Nodup ∧
        ∀ l ∈ ls, (l.reader, l.port) ∉ seen
  | _, _, [], _ => ⟨by simp, by simp, by simp⟩
  | seen, k, l :: ls, h => by
    unfold links at h
    cases he : linkErr sh inRow outRow seen k l with
    | some e => rw [he] at h; cases h
    | none =>
      rw [he] at h
      obtain ⟨hh, hs⟩ := linkErr_none he
      obtain ⟨ih1, ih2, ih3⟩ := links_sound h
      refine ⟨?_, ?_, ?_⟩
      · intro x hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hh
        · exact ih1 x hx
      · refine List.nodup_cons.2 ⟨fun hm => ?_, ih2⟩
        obtain ⟨x, hx, hxe⟩ := List.mem_map.1 hm
        exact ih3 x hx (hxe ▸ List.mem_cons_self)
      · intro x hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact hs
        · exact fun hm => ih3 x hx (List.mem_cons_of_mem _ hm)

theorem ordered_split : ∀ {xs : List (Option LinkEntry)}, ordered xs = true →
    xs = (xs.filterMap id).map some ++ List.replicate (xs.length - (xs.filterMap id).length) none
  | [], _ => rfl
  | some l :: xs, h => by
    have ih := ordered_split (xs := xs) h
    rw [show List.filterMap id (some l :: xs) = l :: List.filterMap id xs from rfl]
    simp only [List.map_cons, List.length_cons, List.cons_append, Nat.add_sub_add_right]
    exact congrArg _ ih
  | none :: xs, h => by
    have hn : ∀ x ∈ xs, x = none := fun x hx => Option.isNone_iff_eq_none.1 (List.all_eq_true.1 h x hx)
    have hf : xs.filterMap id = [] := List.filterMap_eq_nil_iff.2 fun x hx => by simp [hn x hx]
    rw [show List.filterMap id (none :: xs) = List.filterMap id xs from rfl, hf]
    simp only [List.map_nil, List.nil_append, List.length_nil, List.length_cons, Nat.sub_zero]
    rw [List.replicate_succ]
    exact congrArg _ (List.eq_replicate_iff.2 ⟨rfl, hn⟩)

/-- **What `check` accepts**: the table decodes to its `L > 0` links then fillers, `grid L` rows in all; every link holds at
the shape and the rows' values (`Holds`: its reader row's words are its writer row's); and no input row is read by two
links. -/
theorem check_sound {sh : Shape} {inRow outRow : Nat → Nat → Array Nat} {rows : Array ByteArray} {L : Nat}
    (h : check sh inRow outRow rows = .ok L) :
    ∃ ls : List LinkEntry, decodeAll rows.toList = .ok (ls.map some ++ List.replicate (rows.size - L) none) ∧
      L = ls.length ∧ 0 < L ∧ rows.size = grid L ∧ (∀ l ∈ ls, Holds sh inRow outRow l) ∧
      (ls.map fun l => (l.reader, l.port)).Nodup := by
  unfold check at h
  cases hd : decodeAll rows.toList with
  | error e => rw [hd] at h; cases h
  | ok xs =>
    rw [hd] at h
    simp only at h
    split at h
    · cases h
    · rename_i ho
      have ho : ordered xs = true := by simpa using ho
      split at h
      · cases h
      · rename_i he
        split at h
        · cases h
        · rename_i hc
          have hc' : rows.size = grid (xs.filterMap id).length := by simpa using hc
          cases hl : links sh inRow outRow [] 0 (xs.filterMap id) with
          | error e => rw [hl] at h; cases h
          | ok u =>
            rw [hl] at h
            cases h
            obtain ⟨h1, h2, -⟩ := links_sound hl
            have hlen : xs.length = rows.size := by
              have := congrArg List.length (ordered_split ho)
              have hx : ∀ {ys : List ByteArray} {zs : List (Option LinkEntry)}, decodeAll ys = .ok zs → zs.length = ys.length
                := by
                intro ys
                induction ys with
                | nil => intro zs hz; cases hz; rfl
                | cons r rs ih =>
                  intro zs hz
                  unfold decodeAll at hz
                  cases hr : decode r with
                  | error e => rw [hr] at hz; cases hz
                  | ok x =>
                    rw [hr] at hz
                    cases hrs : decodeAll rs with
                    | error e => rw [hrs] at hz; cases hz
                    | ok ws => rw [hrs] at hz; cases hz; simp [ih hrs]
              simpa using hx hd
            refine ⟨xs.filterMap id, ?_, rfl, ?_, hc', h1, h2⟩
            · rw [← hlen, ← ordered_split ho]
            · cases hxs : xs.filterMap id with
              | nil => simp [hxs] at he
              | cons _ _ => simp

def NO_LINKS : String :=
  "--rec-links is refused: no guarantee yet covers a recursive statement's link table (its root and row count)"

/-- **`--rec-links` is refused**, given a value or not, before any file is read: a statement carrying the link table's
root and row count claims that every link holds in the inner population, and no verdict may claim that before V*'s link
statement and its binding are proved. -/
def refuseLinks (args : List String) : Except String Unit :=
  if args.contains "--rec-links" then .error NO_LINKS else .ok ()

/-! ## `check_sound`'s witnesses -/

/-- The honest instance: one unit with one one-word input row and one one-word output row, its output row read into its
input row, both rows `#[0]`. -/
def honestShape : Shape := ⟨1, #[1], #[1]⟩

def honestRow (_ _ : Nat) : Array Nat := #[0]

def honestLink : LinkEntry := ⟨0, 0, 0, 0⟩

/-- **`check_sound`'s honest lemma**: its hypothesis holds at an honest table, the prover's (`table`) of one link that
holds at the honest instance, which `check` accepts with its one link. -/
theorem check_sound_honest : check honestShape honestRow honestRow (table [honestLink]) = .ok 1 := by
  set_option maxRecDepth 100000 in rfl

/-- **`check_sound`'s nontrivial lemma**: its conclusion fails at a one-row table whose row is empty, for one link,
whatever links are named: the row decodes to no link (`row-size`). -/
theorem check_sound_nontrivial : ∀ ls : List LinkEntry,
    ¬ (decodeAll (#[ByteArray.empty] : Array ByteArray).toList =
          .ok (ls.map some ++ List.replicate ((#[ByteArray.empty] : Array ByteArray).size - 1) none) ∧
        1 = ls.length ∧ 0 < 1 ∧ (#[ByteArray.empty] : Array ByteArray).size = grid 1 ∧
        (∀ l ∈ ls, Holds honestShape honestRow honestRow l) ∧ (ls.map fun l => (l.reader, l.port)).Nodup) := by
  rintro ls ⟨h, -⟩
  simp [decodeAll, decode, ROW_BYTES] at h

end RecLinks
end Flock
