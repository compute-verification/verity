module

public import Proofs.Flock.SpotFlock.Frame

/-!
# The frame reads its ports off the run's roots

Under SPOT[Flock]'s two-stage code an opening is checked under the run's roots: the statement's (`decode`'s `roots`)
or a replay block's (`irootsOf`: its place among the replay blocks, then the same). Their entries are the record's
ports' (`portEntry`): four messages a port, its name, root, domain id and leaf count. The frame reads its ports off
them, so a frame fixed before the run checks each opening at the port the run committed it under.

* `entriesOf` reads the entries back (`entriesOf_portRoots`, `entriesOf_irootsOf`): four messages at a time, after the
  replay block's place where there is one, the leaf count through `decNat` (`decNat_encNat`, below `2^64`).
* `Slot`: where gate `g`'s row is, all public: its port's name, the port's row schema and the row's leaf position.
* `atOf slot roots g`: the root of the first entry named like the slot's port, at the slot's position.
* `ownAtOf slot cap roots g`: that entry's port at the slot's schema (`portOf`), where it is at most `cap` levels deep;
  `frameAt` sets both on a frame, at its own ports' depth (`depth_ownAtOf` gives `Code.ownAt_depth`).
* `at_frameAt`, `find_frameAt`: there the frame checks gate `g`'s opening at the entry's root and the slot's position,
  at the entry's port, the first with that root.
* `port_frameAt`: so where the slot has a port `T`'s schema and the entry `T`'s root, leaf count and domain, the frame
  meets either branch of `FrameIs` at `T` (the registered read, or the published tree's port), but for the position
  and the path. `entryNamed_irootsOf` and `entryNamed_decode` read the entry off the record's ports.

What `FrameIs` then asks of a frame `frameAt slot c` is three facts: the reveal's openings (`c`'s `reveal`, `place`,
`bits`, `salt` and `π`) are the session rows' at the layout's columns; each gate's slot names its port, that port's
schema and the row's position (the session header's, or the drawn unit's row of the published tree); and each port's
entry in an accepted run's record has its registered read's root, leaf count and domain (round 2's `second-read`) or
its published tree's, at most `c`'s own depth.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.NetworkCertifier.Bytes (bs)
open Lean (Json)

/-! ## The entries, read back -/

/-- **A number read back** from its first eight bytes, little-endian: `FlockDeploy.encNat`'s inverse below `2^64`. -/
def decNat (m : Msg) : ℕ := ((List.range 8).map fun i => (m.getD i 0).toNat * 256 ^ i).sum

theorem sum_digits (n : ℕ) :
    ∀ k, ((List.range k).map fun i => n / 256 ^ i % 256 * 256 ^ i).sum = n % 256 ^ k
  | 0 => by simp [Nat.mod_one]
  | k + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_digits n k, Nat.mod_pow_succ]
    simp [Nat.mul_comm]

theorem decNat_encNat (n : ℕ) : decNat (FlockDeploy.encNat n) = n % 2 ^ 64 := by
  have h : ∀ i ∈ List.range 8,
      ((FlockDeploy.encNat n).getD i 0).toNat * 256 ^ i = n / 256 ^ i % 256 * 256 ^ i := by
    intro i hi
    have hi8 : i < 8 := List.mem_range.1 hi
    simp [FlockDeploy.encNat, List.getD_eq_getElem?_getD, hi8]
  rw [decNat, List.map_congr_left h, sum_digits n 8]
  norm_num

/-- **A port's entry read back**: its name, root and domain id, and its leaf count. -/
structure Entry where
  name : Msg
  root : Msg
  domain : Msg
  leaves : ℕ

/-- **Entries read back**, four messages at a time, the leaf count through `decNat`; a shorter tail reads as none. -/
def entries : List Msg → List Entry
  | n :: r :: d :: l :: rest => ⟨n, r, d, decNat l⟩ :: entries rest
  | _ => []

/-- **The roots' entries**: after the replay block's place where the roots have one, which makes them one past a
multiple of four. -/
def entriesOf (roots : List Msg) : List Entry := entries (if roots.length % 4 = 1 then roots.tail else roots)

/-- **A record's port's entry as the frame reads it**: its name, root and domain id, and its leaf count below
`2^64`. -/
def entryOf (e : String × Json) : Entry :=
  ⟨e.1.toUTF8.data.toList, hexMsg (Spot.getD e.2 "root"), hexMsg (Spot.getD e.2 "domain"),
    (Spot.nat? (Spot.getD e.2 "leaves")).getD 0 % 2 ^ 64⟩

theorem entries_flatMap (es : List (String × Json)) : entries (es.flatMap portEntry) = es.map entryOf := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons, List.map_cons, portEntry, List.cons_append, List.nil_append]
    rw [entries, ih, entryOf, decNat_encNat]

theorem length_flatMap_portEntry (es : List (String × Json)) : (es.flatMap portEntry).length = 4 * es.length := by
  induction es with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons, List.length_append, ih, List.length_cons, portEntry, List.length_nil]
    omega

/-- **A statement's roots read back**: each of its record's ports' entries. -/
theorem entriesOf_portRoots (rec : Json) :
    entriesOf (portRoots rec) = (Spot.Record.byPort rec).toList.map entryOf := by
  have hl : ¬(portRoots rec).length % 4 = 1 := by
    rw [portRoots, length_flatMap_portEntry]
    omega
  simp only [entriesOf, hl, ↓reduceIte]
  rw [portRoots, entries_flatMap]

/-- **A replay block's roots read back**: after its place, each of the round-1 record's ports' entries. -/
theorem entriesOf_cons (i : Msg) (rec : Json) :
    entriesOf (i :: portRoots rec) = (Spot.Record.byPort rec).toList.map entryOf := by
  have hl : (i :: portRoots rec).length % 4 = 1 := by
    rw [List.length_cons, portRoots, length_flatMap_portEntry]
    omega
  simp only [entriesOf, hl, ↓reduceIte, List.tail_cons]
  rw [portRoots, entries_flatMap]

theorem entriesOf_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) :
    entriesOf (irootsOf m₁ ρ) = (Spot.Record.byPort rec).toList.map entryOf := by
  simp only [irootsOf, hm, Option.map_some, Option.getD_some]
  exact entriesOf_cons _ rec

theorem entriesOf_decode {x : Msg} {st : Roots} (hx : decode x = some st) :
    entriesOf st.roots = (Spot.Record.byPort st.record).toList.map entryOf := by
  unfold decode at hx
  cases hj : jsonMsg x with
  | none => simp [hj] at hx
  | some rec =>
    simp only [hj, Option.map_some, Option.some.injEq] at hx
    subst hx
    exact entriesOf_portRoots rec

/-! ## The frame's ports -/

/-- **Where a gate's row is**, all public: its port's name, the port's row schema and the row's leaf position. -/
structure Slot where
  port : Msg
  schema : String
  pos : ℕ

/-- **The first of the roots' entries named `name`.** -/
def entryNamed (roots : List Msg) (name : Msg) : Option Entry :=
  (entriesOf roots).find? fun e => decide (e.name = name)

/-- **An entry's port** at a schema: its root, domain id and leaf count; no name, value or reads, which the deployed
check doesn't read. -/
def portOf (schema : String) (e : Entry) : _root_.Flock.Registered.Port :=
  { name := "", value := "", root := bytes e.root, leaves := e.leaves, words := none, wordBits := 0,
    domain := bytes e.domain, readsRel := #[], schema := schema }

/-- **Where the frame checks gate `g`'s opening**: at the root of the first entry named like its slot's port, at the
slot's position. -/
def atOf (slot : ℕ → Option Slot) (roots : List Msg) (g : ℕ) : Option (Msg × ℕ) :=
  (slot g).bind fun s => (entryNamed roots s.port).map fun e => (e.root, s.pos)

/-- **The ports gate `g`'s opening reads**: that entry's port at the slot's schema, where it is at most `cap` levels
deep; none otherwise. -/
def ownAtOf (slot : ℕ → Option Slot) (cap : ℕ) (roots : List Msg) (g : ℕ) : Array _root_.Flock.Registered.Port :=
  match slot g with
  | some s =>
    match entryNamed roots s.port with
    | some e => if _root_.Flock.log2ceil e.leaves ≤ cap then #[portOf s.schema e] else #[]
    | none => #[]
  | none => #[]

theorem depth_ownAtOf (slot : ℕ → Option Slot) (cap : ℕ) (roots : List Msg) (g : ℕ) :
    depth (ownAtOf slot cap roots g) ≤ cap := by
  unfold ownAtOf
  split
  · split
    · split
      · rename_i h
        simpa [depth, portOf] using h
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

theorem at_frameAt (hs : slot g = some s) (he : entryNamed roots s.port = some e) :
    (frameAt slot c).at_ roots g = some (bs (portOf s.schema e).root, s.pos) := by
  simp [frameAt, atOf, hs, he, portOf, bs_bytes]

theorem find_frameAt (hs : slot g = some s) (he : entryNamed roots s.port = some e)
    (hd : _root_.Flock.log2ceil e.leaves ≤ depth c.own) :
    ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs (portOf s.schema e).root)) =
      some (portOf s.schema e) := by
  simp [frameAt, ownAtOf, hs, he, hd]

/-- **The frame checks at a port as `FrameIs` asks**: where gate `g`'s slot has port `T`'s schema and the roots' entry
named like its port has `T`'s root, leaf count and domain, at most the frame's own depth, the frame checks the opening
at a port with `T`'s root, leaf count, domain and schema, at the slot's position, the first with that root. `T` is the
registered read in `FrameIs`'s registered branch and the published tree's port in its roots branch. -/
theorem port_frameAt {T : _root_.Flock.Registered.Port} (hs : slot g = some s) (he : entryNamed roots s.port = some e)
    (hr : bytes e.root = T.root) (hl : e.leaves = T.leaves) (hd : bytes e.domain = T.domain)
    (hsc : s.schema = T.schema) (hcap : _root_.Flock.log2ceil e.leaves ≤ depth c.own) :
    ∃ P : _root_.Flock.Registered.Port, P.root = T.root ∧ P.leaves = T.leaves ∧ P.domain = T.domain ∧
      P.schema = T.schema ∧ (frameAt slot c).at_ roots g = some (bs P.root, s.pos) ∧
      ((frameAt slot c).ownAt roots g).find? (fun P' => decide (bs P'.root = bs P.root)) = some P :=
  ⟨portOf s.schema e, hr, hl, hd, hsc, at_frameAt hs he, find_frameAt hs he hcap⟩

/-- **A replay block's entry named `name`** is the first of the round-1 record's ports' so named. -/
theorem entryNamed_irootsOf {m₁ : Msg} {rec : Json} (hm : jsonMsg m₁ = some rec) (ρ : ℕ) (name : Msg) :
    entryNamed (irootsOf m₁ ρ) name =
      ((Spot.Record.byPort rec).toList.map entryOf).find? fun e => decide (e.name = name) := by
  rw [entryNamed, entriesOf_irootsOf hm]

/-- **The statement's entry named `name`** is the first of the boundary record's ports' so named. -/
theorem entryNamed_decode {x : Msg} {st : Roots} (hx : decode x = some st) (name : Msg) :
    entryNamed st.roots name =
      ((Spot.Record.byPort st.record).toList.map entryOf).find? fun e => decide (e.name = name) := by
  rw [entryNamed, entriesOf_decode hx]

end SpotFlock
