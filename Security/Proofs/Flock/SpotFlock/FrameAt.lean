module

public import Proofs.Flock.SpotFlock.Frame

/-!
# The frame reads its ports off the run's roots

Under SPOT[Flock]'s two-stage code an opening is checked under the run's roots: the statement's (`decode`'s `roots`)
or a replay block's (`irootsOf`: its place among the replay blocks, then the same). They are the record's
(`portRoots`): whether it names its ports by template, then four messages for each of its ports and registered values,
its name behind its kind, its root, its domain id and its leaf count. The frame reads its ports off them, so a frame
fixed before the run checks each opening at the port the run committed it under.

* `entriesOf`, `namedOf` read them back (`entriesOf_portRoots`, `namedOf_portRoots`, and the same at a replay block's
  place, `entriesOf_cons`, `namedOf_cons`): the entries, the leaf count through `msgNat` (`msgNat_natMsg`), and the
  naming.
* `Slot`: where gate `g`'s row is, all public: the entry it names (`SlotKey`: template `t`'s port `p` of the record,
  named as the record names its ports, `SecondStage.portOf`; a port by its name; or a registered value), the port's row
  schema, the row's leaf position, and the port's leaf count where the slot fixes one (a published tree's).
* `keyAt roots k`, `entryAt roots s`: the first entry named as the key, or the slot, names it (`keyOf`, at the roots'
  naming).
* `atOf slot roots g`: that entry's root, at the slot's position; `ownAtOf slot cap roots g`: its port at the slot
  (`portAt`), where it is at most `cap` levels deep. `frameAt` sets both on a frame, at its own ports' depth
  (`depth_ownAtOf` gives `Code.ownAt_depth`).
* `port_frameAt`: where the slot has a port `T`'s schema, and `T`'s leaf count where it fixes one, and the entry it
  names has `T`'s root and domain id, and `T`'s leaf count where the slot fixes none, at most the frame's own depth, the
  frame checks gate `g`'s opening at a port with `T`'s root, leaf count, domain id and schema, at the slot's position,
  the first with that root: either branch of `FrameIs` at `T` (the registered read, or the published tree's port), but
  for the position and the path. `entryAt_irootsOf` and `entryAt_decode` read the entry off the record.
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

/-- **The roots' entries**: after the replay block's place where the roots have one, and the record's naming, which
make them one or two past a multiple of four. -/
def entriesOf (roots : List Msg) : List Entry := entries (roots.drop (roots.length % 4))

/-- **Whether the roots' record names its ports by template**: the message before its entries. -/
def namedOf (roots : List Msg) : Bool := (roots.drop (roots.length % 4 - 1)).head? == some [1]

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

/-- **A replay block's roots read back**: after its place, the round-1 record's entries. -/
theorem entriesOf_cons (i : Msg) (rec : Json) : entriesOf (i :: portRoots rec) = recEntries rec := by
  have hl : (i :: portRoots rec).length % 4 = 2 := by
    rw [List.length_cons, length_portRoots]
    omega
  rw [entriesOf, hl, portRoots, List.drop_succ_cons, List.drop_one, List.tail_cons, entries_flatMap, recEntries]

theorem namedOf_portRoots (rec : Json) : namedOf (portRoots rec) = Spot.SecondStage.byTemplate rec := by
  have hl : (portRoots rec).length % 4 - 1 = 0 := by
    rw [length_portRoots]
    omega
  rw [namedOf, hl, List.drop_zero, portRoots]
  cases Spot.SecondStage.byTemplate rec <;> rfl

theorem namedOf_cons (i : Msg) (rec : Json) : namedOf (i :: portRoots rec) = Spot.SecondStage.byTemplate rec := by
  have hl : (i :: portRoots rec).length % 4 - 1 = 1 := by
    rw [List.length_cons, length_portRoots]
    omega
  rw [namedOf, hl, List.drop_one, List.tail_cons, portRoots]
  cases Spot.SecondStage.byTemplate rec <;> rfl

theorem entriesOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    entriesOf (irootsOf m₁ ρ) = recEntries rec := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact entriesOf_cons _ rec

theorem namedOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    namedOf (irootsOf m₁ ρ) = Spot.SecondStage.byTemplate rec := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact namedOf_cons _ rec

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

/-- **Where a gate's row is**, all public: the entry its port is, the port's row schema, the row's leaf position, and
the port's leaf count where the slot fixes one. -/
structure Slot where
  key : SlotKey
  schema : String
  pos : ℕ
  leaves : Option ℕ := none

/-- **The first of the roots' entries named `name`.** -/
def entryNamed (roots : List Msg) (name : Msg) : Option Entry :=
  (entriesOf roots).find? fun e => decide (e.name = name)

/-- **The entry a key names** under the roots, at their naming. -/
def keyAt (roots : List Msg) (k : SlotKey) : Option Entry := entryNamed roots (keyOf (namedOf roots) k)

/-- **The entry a slot names** under the roots (`keyAt`). -/
def entryAt (roots : List Msg) (s : Slot) : Option Entry := keyAt roots s.key

/-- **An entry's port at a slot**: the entry's root and domain id, the slot's schema, and the slot's leaf count where it
fixes one, else the entry's; no name, value or reads, which the deployed check doesn't read. -/
def portAt (s : Slot) (e : Entry) : _root_.Flock.Registered.Port :=
  { name := "", value := "", root := bytes e.root, leaves := s.leaves.getD e.leaves, words := none, wordBits := 0,
    domain := bytes e.domain, readsRel := #[], schema := s.schema }

/-- **Where the frame checks gate `g`'s opening**: at the root of the entry its slot names, at the slot's position. -/
def atOf (slot : ℕ → Option Slot) (roots : List Msg) (g : ℕ) : Option (Msg × ℕ) :=
  (slot g).bind fun s => (entryAt roots s).map fun e => (e.root, s.pos)

/-- **The ports gate `g`'s opening reads**: that entry's port at the slot, where it is at most `cap` levels deep; none
otherwise. -/
def ownAtOf (slot : ℕ → Option Slot) (cap : ℕ) (roots : List Msg) (g : ℕ) : Array _root_.Flock.Registered.Port :=
  match slot g with
  | some s =>
    match entryAt roots s with
    | some e => if _root_.Flock.log2ceil (portAt s e).leaves ≤ cap then #[portAt s e] else #[]
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
    (frameAt slot c).at_ roots g = some (bs (portAt s e).root, s.pos) := by
  simp [frameAt, atOf, hs, he, portAt, bs_bytes]

theorem find_frameAt (hs : slot g = some s) (he : entryAt roots s = some e)
    (hd : _root_.Flock.log2ceil (portAt s e).leaves ≤ depth c.own) :
    ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs (portAt s e).root)) =
      some (portAt s e) := by
  simp [frameAt, ownAtOf, hs, he, hd]

/-- **The frame checks at a port as `FrameIs` asks**: where gate `g`'s slot has port `T`'s schema, and `T`'s leaf count
where it fixes one, and the entry it names has `T`'s root and domain id, and `T`'s leaf count where the slot fixes
none, at most the frame's own depth, the frame checks the opening at a port with `T`'s root, leaf count, domain id and
schema, at the slot's position, the first with that root. `T` is the registered read in `FrameIs`'s registered branch
and the published tree's port in its roots branch. -/
theorem port_frameAt {T : _root_.Flock.Registered.Port} (hs : slot g = some s) (he : entryAt roots s = some e)
    (hr : bytes e.root = T.root) (hl : s.leaves.getD e.leaves = T.leaves) (hd : bytes e.domain = T.domain)
    (hsc : s.schema = T.schema) (hcap : _root_.Flock.log2ceil T.leaves ≤ depth c.own) :
    ∃ P : _root_.Flock.Registered.Port, P.root = T.root ∧ P.leaves = T.leaves ∧ P.domain = T.domain ∧
      P.schema = T.schema ∧ (frameAt slot c).at_ roots g = some (bs P.root, s.pos) ∧
      ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs P.root)) = some P := by
  refine ⟨portAt s e, hr, hl, hd, hsc, at_frameAt hs he, find_frameAt hs he ?_⟩
  show _root_.Flock.log2ceil (s.leaves.getD e.leaves) ≤ _
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

end SpotFlock
