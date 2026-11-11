module

public import Proofs.Flock.SpotFlock.Frame

/-!
# The frame reads its ports off the run's roots

Under SPOT[Flock]'s two-stage code an opening is checked under the run's roots: the statement's (`decode`'s `roots`)
or a replay block's (`irootsOf`: its place among the replay blocks and their number, then the same). They are the
record's (`portRoots`): whether it names its ports by template, then four messages for each of its ports and registered
values, its name behind its kind, its root, its domain id and its leaf count. The frame reads its ports off them, so a
frame fixed before the run checks each opening at the port, and the row, the run committed it under.

* `entriesOf`, `namedOf`, `placeOf`, `qOf` read them back (`entriesOf_portRoots`, `namedOf_portRoots`, and the same
  at a replay block's place, `entriesOf_cons`, `namedOf_cons`, `placeOf_cons`, `qOf_cons`): the entries, the leaf count
  through `msgNat` (`msgNat_natMsg`), the naming, and the replay block's place and the number of replay blocks, `0` at
  the statement's roots.
* `Slot`: where gate `g`'s row is, all public: the entry it names (`SlotKey`: template `t`'s port `p` of the record,
  named as the record names its ports, `SecondStage.portOf`; a port by its name; or a registered value), the port's row
  schema, and the row: unit `pos + place·stride` of a tree of `units + q·stride` units (`unitsAt`), at the unit's first
  leaf (`leafAt`), and the port's leaf count the units' where the slot is `sized` (a published tree's, one row a unit),
  else the entry's (`leavesAt`); leaf `pos` where the slot counts no units.
* `keyAt roots k`, `entryAt roots s`: the first entry named as the key, or the slot, names it (`keyOf`, at the roots'
  naming).
* `atOf slot roots g`: that entry's root, at the slot's leaf; `ownAtOf slot cap roots g`: its port at the slot
  (`portAt`), where it is at most `cap` levels deep. `frameAt` sets both on a frame, at its own ports' depth
  (`depth_ownAtOf` gives `Code.ownAt_depth`).
* `port_frameAt`: where the slot has a port `T`'s schema and leaf count, and the entry it names has `T`'s root and
  domain id, at most the frame's own depth, the frame checks gate `g`'s opening at a port with `T`'s root, leaf count,
  domain id and schema, at the slot's leaf, the first with that root: either branch of `FrameIs` at `T` (the registered
  read, or the published tree's port), but for the position and the path. `entryAt_irootsOf` and `entryAt_decode` read
  the entry off the record; `unitsAt_irootsOf`, `leafAt_irootsOf` and `leafAt_decode` the row, which moves with the
  replay block's place under its roots and is `pos` under the statement's.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.NetworkCertifier.Bytes (bs)
open Lean (Json)

/-! ## The entries, read back -/

/-- **A number read back** from its bytes, little-endian: `natMsg`'s inverse (`msgNat_natMsg`). -/
def msgNat (m : Msg) : ℕ := ((List.range m.length).map fun i => (m.getD i 0).toNat * 256 ^ i).sum

theorem sum_digits (n : ℕ) :
    ∀ k, ((List.range k).map fun i => n / 256 ^ i % 256 * 256 ^ i).sum = n % 256 ^ k
  | 0 => by simp [Nat.mod_one]
  | k + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_digits n k, Nat.mod_pow_succ]
    simp [Nat.mul_comm]

theorem msgNat_natMsg (n : ℕ) : msgNat (natMsg n) = n := by
  have hlen : (natMsg n).length = n.log2 / 8 + 1 := by simp [natMsg]
  have h : ∀ i ∈ List.range (natMsg n).length,
      ((natMsg n).getD i 0).toNat * 256 ^ i = n / 256 ^ i % 256 * 256 ^ i := by
    intro i hi
    rw [hlen] at hi
    have hi' : i < n.log2 / 8 + 1 := List.mem_range.1 hi
    simp [natMsg, List.getD_eq_getElem?_getD, hi']
  have hk : n < 256 ^ (n.log2 / 8 + 1) := by
    have h2 : 2 ^ (n.log2 + 1) ≤ 256 ^ (n.log2 / 8 + 1) := by
      rw [show (256 : ℕ) = 2 ^ 8 by norm_num, ← Nat.pow_mul]
      exact Nat.pow_le_pow_right (by norm_num) (by omega)
    exact Nat.lt_of_lt_of_le Nat.lt_log2_self h2
  rw [msgNat, List.map_congr_left h, hlen, sum_digits n, Nat.mod_eq_of_lt hk]

/-- **An entry read back**: its name behind its kind (`0` a port, `1` a registered value), its root and domain id, and
its leaf count. -/
structure Entry where
  name : Msg
  root : Msg
  domain : Msg
  leaves : ℕ

/-- **Entries read back**, four messages at a time, the leaf count through `msgNat`; a shorter tail reads as none. -/
def entries : List Msg → List Entry
  | n :: r :: d :: l :: rest => ⟨n, r, d, msgNat l⟩ :: entries rest
  | _ => []

/-- **The roots' entries**: after the replay block's place and the number of replay blocks where the roots have them,
and the record's naming, which make them one or three past a multiple of four. -/
def entriesOf (roots : List Msg) : List Entry := entries (roots.drop (roots.length % 4))

/-- **Whether the roots' record names its ports by template**: the message before its entries. -/
def namedOf (roots : List Msg) : Bool := (roots.drop (roots.length % 4 - 1)).head? == some [1]

/-- **The replay block's place among the replay blocks** the roots state before the record's naming, `0` where they
state none (the statement's). -/
def placeOf (roots : List Msg) : ℕ := msgNat ((roots.take (roots.length % 4 - 1)).getD 0 [])

/-- **The number of replay blocks** the roots state before the record's naming, `0` where they state none. -/
def qOf (roots : List Msg) : ℕ := msgNat ((roots.take (roots.length % 4 - 1)).getD 1 [])

/-- **A record's entry as the frame reads it**: its name behind its kind, root, domain id and leaf count. -/
def entryOf (e : UInt8 × String × Json) : Entry :=
  ⟨e.1 :: e.2.1.toUTF8.data.toList, hexMsg (Spot.getD e.2.2 "root"), hexMsg (Spot.getD e.2.2 "domain"),
    (Spot.nat? (Spot.getD e.2.2 "leaves")).getD 0⟩

/-- **A record's entries as the frame reads them**: its ports', then its registered values' (`tagged`). -/
def recEntries (rec : Json) : List Entry := (tagged rec).map entryOf

theorem entries_flatMap (es : List (UInt8 × String × Json)) : entries (es.flatMap portEntry) = es.map entryOf := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons, List.map_cons, portEntry, List.cons_append, List.nil_append]
    rw [entries, ih, entryOf, msgNat_natMsg]

theorem length_flatMap_portEntry (es : List (UInt8 × String × Json)) :
    (es.flatMap portEntry).length = 4 * es.length := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons, List.length_append, ih, List.length_cons, portEntry, List.length_nil]
    omega

theorem length_portRoots (rec : Json) : (portRoots rec).length = 4 * (tagged rec).length + 1 := by
  rw [portRoots, List.length_cons, length_flatMap_portEntry]

/-- **A statement's roots read back**: the record's entries. -/
theorem entriesOf_portRoots (rec : Json) : entriesOf (portRoots rec) = recEntries rec := by
  have hl : (portRoots rec).length % 4 = 1 := by
    rw [length_portRoots]
    omega
  rw [entriesOf, hl, portRoots, List.drop_one, List.tail_cons, entries_flatMap, recEntries]

/-- **A replay block's roots read back**: after its place and the number of replay blocks, the round-1 record's
entries. -/
theorem entriesOf_cons (i n : Msg) (rec : Json) : entriesOf (i :: n :: portRoots rec) = recEntries rec := by
  have hl : (i :: n :: portRoots rec).length % 4 = 3 := by
    rw [List.length_cons, List.length_cons, length_portRoots]
    omega
  rw [entriesOf, hl, portRoots, List.drop_succ_cons, List.drop_succ_cons, List.drop_one, List.tail_cons,
    entries_flatMap, recEntries]

theorem namedOf_portRoots (rec : Json) : namedOf (portRoots rec) = Spot.SecondStage.byTemplate rec := by
  have hl : (portRoots rec).length % 4 - 1 = 0 := by
    rw [length_portRoots]
    omega
  rw [namedOf, hl, List.drop_zero, portRoots]
  cases Spot.SecondStage.byTemplate rec <;> rfl

theorem namedOf_cons (i n : Msg) (rec : Json) :
    namedOf (i :: n :: portRoots rec) = Spot.SecondStage.byTemplate rec := by
  have hl : (i :: n :: portRoots rec).length % 4 - 1 = 2 := by
    rw [List.length_cons, List.length_cons, length_portRoots]
    omega
  rw [namedOf, hl, List.drop_succ_cons, List.drop_one, List.tail_cons, portRoots]
  cases Spot.SecondStage.byTemplate rec <;> rfl

theorem placeOf_portRoots (rec : Json) : placeOf (portRoots rec) = 0 := by
  have hl : (portRoots rec).length % 4 - 1 = 0 := by
    rw [length_portRoots]
    omega
  rw [placeOf, hl, List.take_zero]
  rfl

theorem qOf_portRoots (rec : Json) : qOf (portRoots rec) = 0 := by
  have hl : (portRoots rec).length % 4 - 1 = 0 := by
    rw [length_portRoots]
    omega
  rw [qOf, hl, List.take_zero]
  rfl

theorem placeOf_cons (a b : ℕ) (rec : Json) : placeOf (natMsg a :: natMsg b :: portRoots rec) = a := by
  have hl : (natMsg a :: natMsg b :: portRoots rec).length % 4 - 1 = 2 := by
    rw [List.length_cons, List.length_cons, length_portRoots]
    omega
  rw [placeOf, hl]
  exact msgNat_natMsg a

theorem qOf_cons (a b : ℕ) (rec : Json) : qOf (natMsg a :: natMsg b :: portRoots rec) = b := by
  have hl : (natMsg a :: natMsg b :: portRoots rec).length % 4 - 1 = 2 := by
    rw [List.length_cons, List.length_cons, length_portRoots]
    omega
  rw [qOf, hl]
  exact msgNat_natMsg b

theorem entriesOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    entriesOf (irootsOf m₁ ρ) = recEntries rec := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact entriesOf_cons _ _ rec

theorem namedOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    namedOf (irootsOf m₁ ρ) = Spot.SecondStage.byTemplate rec := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact namedOf_cons _ _ rec

theorem placeOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    placeOf (irootsOf m₁ ρ) = (blocksOf rec).idxOf ρ := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact placeOf_cons _ _ rec

theorem qOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    qOf (irootsOf m₁ ρ) = (blocksOf rec).length := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact qOf_cons _ _ rec

theorem decode_eq {x : Msg} {st : Roots} (hx : decode x = some st) :
    ∃ rec, jsonMsg x = some rec ∧ st = ⟨rec, portRoots rec⟩ := by
  unfold decode at hx
  cases hj : jsonMsg x with
  | none => simp [hj] at hx
  | some rec =>
    simp only [hj, Option.map_some, Option.some.injEq] at hx
    exact ⟨rec, rfl, hx.symm⟩

theorem entriesOf_decode {x : Msg} {st : Roots} (hx : decode x = some st) :
    entriesOf st.roots = recEntries st.record := by
  obtain ⟨rec, -, rfl⟩ := decode_eq hx
  exact entriesOf_portRoots rec

theorem namedOf_decode {x : Msg} {st : Roots} (hx : decode x = some st) :
    namedOf st.roots = Spot.SecondStage.byTemplate st.record := by
  obtain ⟨rec, -, rfl⟩ := decode_eq hx
  exact namedOf_portRoots rec

theorem placeOf_decode {x : Msg} {st : Roots} (hx : decode x = some st) : placeOf st.roots = 0 := by
  obtain ⟨rec, -, rfl⟩ := decode_eq hx
  exact placeOf_portRoots rec

theorem qOf_decode {x : Msg} {st : Roots} (hx : decode x = some st) : qOf st.roots = 0 := by
  obtain ⟨rec, -, rfl⟩ := decode_eq hx
  exact qOf_portRoots rec

/-! ## The frame's ports -/

/-- **The entry a slot names**: template `t`'s port `p` of the record, named as the record names its ports
(`SecondStage.portOf`); a port by its name; or a registered value. -/
inductive SlotKey where
  | interior (t p : String)
  | port (v : String)
  | reg (v : String)

/-- **The name, behind its kind, of the entry a slot names**, at a record that names its ports by template where
`named`. -/
def keyOf (named : Bool) : SlotKey → Msg
  | .interior t p => 0 :: (Spot.SecondStage.portOf named t p).toUTF8.data.toList
  | .port v => 0 :: v.toUTF8.data.toList
  | .reg v => 1 :: v.toUTF8.data.toList

/-- **Where a gate's row is**, all public: the entry its port is, the port's row schema, and the row: unit
`pos + place·stride` of a tree of `units + q·stride` units, `q` replay blocks, at the place of the replay block whose
roots it is read under, `0` and `0` under the statement's; or leaf `pos` where the slot counts no units. Where the slot
is `sized` (a published tree's, one row a unit) the port's leaf count is its units'. -/
structure Slot where
  key : SlotKey
  schema : String
  pos : ℕ
  stride : ℕ := 0
  units : ℕ := 0
  sized : Bool := false

/-- **The first of the roots' entries named `name`.** -/
def entryNamed (roots : List Msg) (name : Msg) : Option Entry :=
  (entriesOf roots).find? fun e => decide (e.name = name)

/-- **The entry a key names** under the roots, at their naming. -/
def keyAt (roots : List Msg) (k : SlotKey) : Option Entry := entryNamed roots (keyOf (namedOf roots) k)

/-- **The entry a slot names** under the roots (`keyAt`). -/
def entryAt (roots : List Msg) (s : Slot) : Option Entry := keyAt roots s.key

/-- **The units a slot's tree holds** under the roots: its own, and `stride` for each replay block. -/
def unitsAt (roots : List Msg) (s : Slot) : ℕ := s.units + qOf roots * s.stride

/-- **A port's leaf count at a slot**: its units' where the slot is sized, else the entry's. -/
def leavesAt (roots : List Msg) (s : Slot) (e : Entry) : ℕ := if s.sized then unitsAt roots s else e.leaves

/-- **The leaf a slot opens** under the roots: its unit's first, `leaves / units` leaves a unit; leaf `pos` where the
slot counts no units. -/
def leafAt (roots : List Msg) (s : Slot) (e : Entry) : ℕ :=
  if unitsAt roots s = 0 then s.pos else (s.pos + placeOf roots * s.stride) * (leavesAt roots s e / unitsAt roots s)

/-- **An entry's port at a slot**: the entry's root and domain id, the slot's schema, and its leaf count at the slot
(`leavesAt`); no name, value or reads, which the deployed check doesn't read. -/
def portAt (roots : List Msg) (s : Slot) (e : Entry) : _root_.Flock.Registered.Port :=
  { name := "", value := "", root := bytes e.root, leaves := leavesAt roots s e, words := none, wordBits := 0,
    domain := bytes e.domain, readsRel := #[], schema := s.schema }

/-- **Where the frame checks gate `g`'s opening**: at the root of the entry its slot names, at the slot's leaf. -/
def atOf (slot : ℕ → Option Slot) (roots : List Msg) (g : ℕ) : Option (Msg × ℕ) :=
  (slot g).bind fun s => (entryAt roots s).map fun e => (e.root, leafAt roots s e)

/-- **The ports gate `g`'s opening reads**: that entry's port at the slot, where it is at most `cap` levels deep; none
otherwise. -/
def ownAtOf (slot : ℕ → Option Slot) (cap : ℕ) (roots : List Msg) (g : ℕ) : Array _root_.Flock.Registered.Port :=
  match slot g with
  | some s =>
    match entryAt roots s with
    | some e => if _root_.Flock.log2ceil (portAt roots s e).leaves ≤ cap then #[portAt roots s e] else #[]
    | none => #[]
  | none => #[]

theorem depth_ownAtOf (slot : ℕ → Option Slot) (cap : ℕ) (roots : List Msg) (g : ℕ) :
    depth (ownAtOf slot cap roots g) ≤ cap := by
  unfold ownAtOf
  split
  · split
    · split
      · rename_i h
        simpa [depth] using h
      · simp [depth]
    · simp [depth]
  · simp [depth]

/-- **The frame at a slot map**: `c` with its ports read off the roots (`ownAtOf`, at `c`'s own ports' depth) and each
gate's opening checked at its slot (`atOf`). -/
def frameAt (slot : ℕ → Option Slot) (c : Code) : Code :=
  { c with
    ownAt := ownAtOf slot (depth c.own)
    ownAt_depth := fun roots g => depth_ownAtOf slot (depth c.own) roots g
    at_ := atOf slot }

variable {slot : ℕ → Option Slot} {c : Code} {roots : List Msg} {g : ℕ} {s : Slot} {e : Entry}

theorem at_frameAt (hs : slot g = some s) (he : entryAt roots s = some e) :
    (frameAt slot c).at_ roots g = some (bs (portAt roots s e).root, leafAt roots s e) := by
  simp [frameAt, atOf, hs, he, portAt, bs_bytes]

theorem find_frameAt (hs : slot g = some s) (he : entryAt roots s = some e)
    (hd : _root_.Flock.log2ceil (portAt roots s e).leaves ≤ depth c.own) :
    ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs (portAt roots s e).root)) =
      some (portAt roots s e) := by
  simp [frameAt, ownAtOf, hs, he, hd]

/-- **The frame checks at a port as `FrameIs` asks**: where gate `g`'s slot has port `T`'s schema and leaf count
(`leavesAt`), and the entry it names has `T`'s root and domain id, at most the frame's own depth, the frame checks the
opening at a port with `T`'s root, leaf count, domain id and schema, at the slot's leaf, the first with that root. `T`
is the registered read in `FrameIs`'s registered branch and the published tree's port in its roots branch. -/
theorem port_frameAt {T : _root_.Flock.Registered.Port} (hs : slot g = some s) (he : entryAt roots s = some e)
    (hr : bytes e.root = T.root) (hl : leavesAt roots s e = T.leaves) (hd : bytes e.domain = T.domain)
    (hsc : s.schema = T.schema) (hcap : _root_.Flock.log2ceil T.leaves ≤ depth c.own) :
    ∃ P : _root_.Flock.Registered.Port, P.root = T.root ∧ P.leaves = T.leaves ∧ P.domain = T.domain ∧
      P.schema = T.schema ∧ (frameAt slot c).at_ roots g = some (bs P.root, leafAt roots s e) ∧
      ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs P.root)) = some P := by
  refine ⟨portAt roots s e, hr, hl, hd, hsc, at_frameAt hs he, find_frameAt hs he ?_⟩
  show _root_.Flock.log2ceil (leavesAt roots s e) ≤ _
  rw [hl]
  exact hcap

/-- **The entry a slot names under a replay block's roots** is the first of the round-1 record's so named, at its
naming. -/
theorem entryAt_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) (s : Slot) :
    entryAt (irootsOf m₁ ρ) s =
      (recEntries rec).find? fun e => decide (e.name = keyOf (Spot.SecondStage.byTemplate rec) s.key) := by
  rw [entryAt, keyAt, entryNamed, entriesOf_irootsOf hm, namedOf_irootsOf hm]

/-- **The entry a slot names under the statement's roots** is the first of the boundary record's so named, at its
naming. -/
theorem entryAt_decode {x : Msg} {st : Roots} (hx : decode x = some st) (s : Slot) :
    entryAt st.roots s =
      (recEntries st.record).find? fun e => decide (e.name = keyOf (Spot.SecondStage.byTemplate st.record) s.key) := by
  rw [entryAt, keyAt, entryNamed, entriesOf_decode hx, namedOf_decode hx]

/-- **A slot's units under a replay block's roots**: `units + q·stride`, at the round-1 record's `q` replay blocks. -/
theorem unitsAt_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) (s : Slot) :
    unitsAt (irootsOf m₁ ρ) s = s.units + (blocksOf rec).length * s.stride := by
  rw [unitsAt, qOf_irootsOf hm]

/-- **A sized slot's leaf under a replay block's roots** is its unit `pos + i·stride`, at the replay block's place `i`
among the round-1 record's, where it counts any units. -/
theorem leafAt_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) {s : Slot} (hs : s.sized = true)
    (hu : 0 < unitsAt (irootsOf m₁ ρ) s) (e : Entry) :
    leafAt (irootsOf m₁ ρ) s e = s.pos + (blocksOf rec).idxOf ρ * s.stride := by
  rw [leafAt, ite_eq_right (Nat.pos_iff_ne_zero.1 hu), leavesAt, ite_eq_left hs, Nat.div_self hu, Nat.mul_one,
    placeOf_irootsOf hm]

/-- **A slot's leaf under the statement's roots** is `pos`, where it counts no units of its own: the statement's roots
count no replay blocks. -/
theorem leafAt_decode {x : Msg} {st : Roots} (hx : decode x = some st) {s : Slot} (hu : s.units = 0) (e : Entry) :
    leafAt st.roots s e = s.pos := by
  have h0 : unitsAt st.roots s = 0 := by rw [unitsAt, hu, qOf_decode hx, Nat.zero_mul, Nat.add_zero]
  rw [leafAt, ite_eq_left h0]

end SpotFlock
