/-
Level 3: registered reads (PROTOCOL.md §16.10). `check_ok`: when `Registered.check` accepts, each of the verifier's
registered ports has the registered rows' width, and every instance's row of it climbs, from its frame-v3-sha512 leaf at
the position the verifier's program reads for the instance's unit, to the registered root the verifier holds.
`complete_opens`: when the complete check (`checkComplete`) accepts too, every port of the circuit is such a port.
`opensLeaf_binding`: two openings the verifier accepts at
one position under one registered root, with different leaves, give an explicit SHA-512 collision (`openPair`), and
`opens_binding` the same of two commit strings (`bcPair`). No random oracle. `checkDescription_ok` (PROTOCOL.md §16.16):
when `Registered.checkDescription` accepts an inner statement, each of its registered rows opens the registered root at
its own index, every instance reads one of them, and every one of them is read.
-/
import Flock.Registered
import Proofs.Flock.Verifier.Hash

namespace Flock.Registered

/-! ### Acceptance -/

theorem except_bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem throw_ne_ok {α : Type} (e : String) (a : α) : (throw e : V α) ≠ .ok a := nofun

theorem forM_ok {ε α : Type} {f : α → Except ε Unit} : ∀ {l : List α}, l.forM f = .ok () → ∀ x ∈ l, f x = .ok ()
  | [], _, x, hx => absurd hx List.not_mem_nil
  | a :: as, h, x, hx => by
    simp only [List.forM] at h
    obtain ⟨u, hu, hrest⟩ := except_bind_ok h
    cases hx with
    | head => exact hu
    | tail _ hx => exact forM_ok hrest x hx

/-- A port `checkPort` accepts is the circuit's port `p`, of the registration's leaf schema, whose rows are the
registered rows, and every instance's row of its table opens the registered root at the position the verifier's program
reads for the instance's unit. -/
theorem checkPort_ok {c : Circuit} {pub : Public} {t : Tables} {indices : Array Nat} {reg : Port}
    (h : checkPort c pub t indices reg = .ok ()) :
    ∃ p, p < c.ports.size ∧ c.ports[p]!.name = reg.name ∧ portSchema c.ports[p]! = reg.schema ∧
      reg.fits c.ports[p]! = true ∧ ∀ i < pub.n,
      ref t c.ports.size p i < t.counts.getD p 0 ∧ ∃ pos path, reg.reads[indices.getD i 0]? = some (some pos) ∧
        opens reg (tableRow pub t p (ref t c.ports.size p i)) pos path = true := by
  unfold checkPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
    · rename_i hsch
      split at h
      · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
      · rename_i hfit
        obtain ⟨⟨positions, paths⟩, _, h⟩ := except_bind_ok h
        simp only at h
        split at h
        · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
        · split at h
          · exact absurd h (throw_ne_ok _ _)
          · rename_i hrow
            split at h
            · rename_i hinst
              have hpn := List.find?_some hp
              refine ⟨p, List.mem_range.mp (List.mem_of_find?_eq_some hp), by simpa using hpn, by simpa using hsch,
                by simpa using hfit, fun i hi => ?_⟩
              have hi' := List.find?_eq_none.mp hinst i (List.mem_range.mpr hi)
              simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, not_or, Classical.not_not] at hi'
              obtain ⟨hr, hreads⟩ := hi'
              have ho := List.find?_eq_none.mp hrow _ (List.mem_range.mpr hr)
              exact ⟨hr, _, _, (reg.readAt_eq _).symm.trans hreads, by simpa using ho⟩
            · split at h
              · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
              · split at h <;> exact absurd h (throw_ne_ok _ _)
  · exact absurd h (throw_ne_ok _ _)

/-- **`Registered.check` accepting.** If the verifier holds registered roots (`own`) and `check` accepts the statement, the
units were derived from the verifier's own program, and for each of the verifier's registered ports `reg` (the circuit's
port `p`, of `reg`'s leaf schema, whose rows are `reg`'s registered rows: `Port.fits`) and each instance `i`, the
verifier's program reads a position `pos` of `reg`'s value in instance `i`'s unit (`indices`: the units the public
file's header names, which the partition check derived), and instance `i`'s row (`Public.digest`, its `b ‖ c`) climbs
from its frame-v3-sha512 leaf at `pos` (its hm96 leaf, framed at `pos` under `reg`'s domain and schema) through a path
of one 64-byte sibling per level to `reg`'s root. -/
theorem check_ok {c : Circuit} {pub : Public} {own : Array Port} {derived : Bool}
    (h : check c pub (some own) derived = .ok ()) :
    derived = true ∧ ∃ indices, unitIndices pub = .ok indices ∧
      ∀ reg ∈ own, ∃ p, p < c.ports.size ∧ c.ports[p]!.name = reg.name ∧ portSchema c.ports[p]! = reg.schema ∧
        reg.fits c.ports[p]! = true ∧
        ∀ i < pub.n, ∃ pos path, reg.reads[indices.getD i 0]? = some (some pos) ∧ pos < reg.leaves ∧
          path.size = 64 * log2ceil reg.leaves ∧
          climb reg.domain (leafAs reg.schema reg.domain pos (Hm96.Default512.treeLeaf (pub.digest i p c.ports.size)))
            pos (log2ceil reg.leaves) path = reg.root := by
  unfold check at h
  simp only at h
  split at h
  · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
  · rename_i hd
    refine ⟨by simpa using hd, ?_⟩
    split at h
    · obtain ⟨_, _, h⟩ := except_bind_ok h
      split at h
      · rename_i t ht
        obtain ⟨indices, hidx, h⟩ := except_bind_ok h
        refine ⟨indices, hidx, fun reg hreg => ?_⟩
        obtain ⟨p, hp, hname, hsch, hfit, hall⟩ := checkPort_ok (forM_ok h reg (by simpa using hreg))
        refine ⟨p, hp, hname, hsch, hfit, fun i hi => ?_⟩
        obtain ⟨_, pos, path, hr, ho⟩ := hall i hi
        simp only [opens, opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at ho
        exact ⟨pos, path, hr, ho.1.1, ho.1.2, by simpa only [Public.digest, ht, tableRow, ref] using ho.2⟩
      · exact absurd h (throw_ne_ok _ _)
    · exact absurd h (throw_ne_ok _ _)

/-- **Every port of a complete statement opens at its registered root.** If `check` accepts the statement and the
complete check does too (`checkComplete`, which flock-verify runs when the `--registered` file sets `complete`), and the
circuit names each port once, then every port `q` of the circuit, each input port and each output row (so whatever an
output's cone reads), is a port `reg` of `own` of its name, `reg`'s schema and rows, and every instance's row of it
climbs, from its frame-v3-sha512 leaf at the position the verifier's program reads for the instance's unit, to `reg`'s
root. -/
theorem complete_opens {c : Circuit} {pub : Public} {own : Array Port} {derived : Bool}
    (h : check c pub (some own) derived = .ok ()) (hc : checkComplete c own = .ok ())
    (hN : ∀ p < c.ports.size, ∀ q < c.ports.size, c.ports[p]!.name = c.ports[q]!.name → p = q) :
    ∃ indices, unitIndices pub = .ok indices ∧
      ∀ q < c.ports.size, ∃ reg ∈ own, reg.name = c.ports[q]!.name ∧ portSchema c.ports[q]! = reg.schema ∧
        reg.fits c.ports[q]! = true ∧
        ∀ i < pub.n, ∃ pos path, reg.reads[indices.getD i 0]? = some (some pos) ∧ pos < reg.leaves ∧
          path.size = 64 * log2ceil reg.leaves ∧
          climb reg.domain (leafAs reg.schema reg.domain pos (Hm96.Default512.treeLeaf (pub.digest i q c.ports.size)))
            pos (log2ceil reg.leaves) path = reg.root := by
  obtain ⟨-, indices, hidx, hall⟩ := check_ok h
  refine ⟨indices, hidx, fun q hq => ?_⟩
  obtain ⟨reg, hreg, hname⟩ :=
    checkComplete_ok hc c.ports[q]! (by rw [getElem!_pos c.ports q hq]; exact Array.getElem_mem hq)
  obtain ⟨p, hp, hpn, hs, hf, hr⟩ := hall reg hreg
  obtain rfl : p = q := hN p hp q hq (hpn.trans hname)
  exact ⟨reg, hreg, hname, hs, hf, hr⟩

/-! ### The extractor: two accepted openings at one position give an explicit SHA-512 collision

As in `FlockProofs.Merkle`, the extractor is computable (`climbPair`, `openPair`) and the theorem is that its output
collides; a proposition "SHA-512 has a collision" follows (`opensLeaf_collision`) and says nothing on its own. -/

/-- Two SHA-512 inputs that collide: different, with one digest. -/
def Collides (x : ByteArray × ByteArray) : Prop := x.1 ≠ x.2 ∧ Sha512.hash x.1 = Sha512.hash x.2

/-- One part of a frame-v3 frame: its 8-byte big-endian length, then its bytes. -/
def step (acc p : ByteArray) : ByteArray := acc ++ be64 p.size.toUInt64 ++ p

/-- What `FV3.hash tag parts H` hashes. -/
def frame (tag : String) (parts : List ByteArray) : ByteArray :=
  parts.foldl step (FV3.FRAME ++ be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8)

theorem hash_frame (tag : String) (parts : List ByteArray) (H : ByteArray → ByteArray) :
    FV3.hash tag parts H = H (frame tag parts) := rfl

/-- What `node` hashes. -/
def nodeIn (dom : ByteArray) (d idx : Nat) (l r : ByteArray) : ByteArray := frame "node" [dom, FV3.uint d, FV3.uint idx, l, r]

/-- What `leafAs schema` hashes. -/
def leafInAs (schema : String) (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  frame "leaf" [dom, FV3.uint pos, FV3.uint pos, schema.toUTF8, payload]

/-- What `rowLeaf` hashes. -/
def leafIn (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  frame "leaf" [dom, FV3.uint pos, FV3.uint pos, HmRow.ROW_SCHEMA.toUTF8, payload]

theorem node_eq (dom : ByteArray) (d idx : Nat) (l r : ByteArray) :
    node dom d idx l r = Sha512.hash (nodeIn dom d idx l r) := rfl

theorem rowLeaf_eq (dom : ByteArray) (pos : Nat) (payload : ByteArray) :
    rowLeaf dom pos payload = Sha512.hash (leafIn dom pos payload) := rfl

theorem leafAs_eq (schema : String) (dom : ByteArray) (pos : Nat) (payload : ByteArray) :
    leafAs schema dom pos payload = Sha512.hash (leafInAs schema dom pos payload) := rfl

/-- `rowLeaf` is the leaf at `hm96-sha512/row/v1`. -/
theorem leafAs_row : leafAs HmRow.ROW_SCHEMA = rowLeaf := rfl

theorem leafInAs_row : leafInAs HmRow.ROW_SCHEMA = leafIn := rfl

/-- `opensLeaf` at a port of `hm96-sha512/row/v1` rows climbs from `rowLeaf`. -/
theorem opensLeaf_row {reg : Port} (hs : reg.schema = HmRow.ROW_SCHEMA) (payload : ByteArray) (pos : Nat)
    (path : ByteArray) :
    opensLeaf reg payload pos path = (decide (pos < reg.leaves) && decide (path.size = 64 * log2ceil reg.leaves) &&
      decide (climb reg.domain (rowLeaf reg.domain pos payload) pos (log2ceil reg.leaves) path = reg.root)) := by
  rw [opensLeaf, hs, leafAs_row]

theorem step_size (acc p : ByteArray) : (step acc p).size = acc.size + (be64 p.size.toUInt64).size + p.size := by
  simp only [step, ByteArray.size_append]

/-- A frame's last part: two frames one part longer than prefixes of one size, with last parts of one size, are one only
when the prefixes are and the parts are. -/
theorem step_inj {acc acc' p p' : ByteArray} (ha : acc.size = acc'.size) (hp : p.size = p'.size)
    (h : step acc p = step acc' p') : acc = acc' ∧ p = p' := by
  simp only [step] at h
  rw [hp] at h
  have h1 := ByteArray.append_inj_left h (by simp only [ByteArray.size_append, ha])
  refine ⟨(ByteArray.append_left_inj _).mp h1, ?_⟩
  rw [h1] at h
  exact (ByteArray.append_right_inj _).mp h

/-- A node's frame binds its two children, of one size each. -/
theorem nodeIn_inj {dom : ByteArray} {d idx : Nat} {l r l' r' : ByteArray} (hl : l.size = l'.size) (hr : r.size = r'.size)
    (h : nodeIn dom d idx l r = nodeIn dom d idx l' r') : l = l' ∧ r = r' := by
  simp only [nodeIn, frame, List.foldl] at h
  have h2 := step_inj (by simp only [step_size, hl]) hr h
  exact ⟨(step_inj rfl hl h2.1).2, h2.2⟩

/-- A leaf's frame binds its payload, of one size. -/
theorem leafInAs_inj {schema : String} {dom : ByteArray} {pos : Nat} {x y : ByteArray} (hs : x.size = y.size)
    (h : leafInAs schema dom pos x = leafInAs schema dom pos y) : x = y := by
  simp only [leafInAs, frame, List.foldl] at h
  exact (step_inj rfl hs h).2

theorem leafIn_inj {dom : ByteArray} {pos : Nat} {x y : ByteArray} (hs : x.size = y.size)
    (h : leafIn dom pos x = leafIn dom pos y) : x = y :=
  leafInAs_inj (schema := HmRow.ROW_SCHEMA) hs h

/-- **The climb's extractor:** walking two sibling lists up in step from two digests at index `idx` of level `d`, the
first node whose two frames hash to one digest; the two digests themselves when the lists run out. -/
def climbPair (dom : ByteArray) : ByteArray → ByteArray → Nat → Nat → List ByteArray → List ByteArray →
    ByteArray × ByteArray
  | a, b, idx, d, s :: rest, t :: rest2 =>
    let x := if idx % 2 = 0 then nodeIn dom d (idx / 2) a s else nodeIn dom d (idx / 2) s a
    let y := if idx % 2 = 0 then nodeIn dom d (idx / 2) b t else nodeIn dom d (idx / 2) t b
    if Sha512.hash x = Sha512.hash y then (x, y)
    else climbPair dom (Sha512.hash x) (Sha512.hash y) (idx / 2) (d + 1) rest rest2
  | a, b, _, _, _, _ => (a, b)

/-- `climbPair`'s output collides: climbing from two different 64-byte digests to one root along sibling lists of one
length, every sibling 64 bytes. -/
theorem climbPair_collides (dom : ByteArray) :
    ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx d : Nat), s1.length = s2.length →
      (∀ x ∈ s1, x.size = 64) → (∀ x ∈ s2, x.size = 64) → a.size = 64 → b.size = 64 → a ≠ b →
      climbL dom a idx d s1 = climbL dom b idx d s2 → Collides (climbPair dom a b idx d s1 s2) := by
  intro s1
  induction s1 with
  | nil =>
    intro s2 a b idx d hl _ _ _ _ hab h
    cases s2 with
    | nil => exact absurd h hab
    | cons _ _ => simp at hl
  | cons s rest ih =>
    intro s2 a b idx d hl hw1 hw2 ha hb hab h
    cases s2 with
    | nil => simp at hl
    | cons t rest2 =>
      have hl' : rest.length = rest2.length := by simpa using hl
      have hsz := hw1 s (by simp)
      have htz := hw2 t (by simp)
      have hr1 : ∀ x ∈ rest, x.size = 64 := fun x hx => hw1 x (by simp [hx])
      have hr2 : ∀ x ∈ rest2, x.size = 64 := fun x hx => hw2 x (by simp [hx])
      simp only [climbL, node_eq] at h
      by_cases hp : idx % 2 = 0
      · simp only [hp, ↓reduceIte] at h
        by_cases he : Sha512.hash (nodeIn dom d (idx / 2) a s) = Sha512.hash (nodeIn dom d (idx / 2) b t)
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ⟨fun hh => hab (nodeIn_inj (by rw [ha, hb]) (by rw [hsz, htz]) hh).1, he⟩
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ih rest2 _ _ (idx / 2) (d + 1) hl' hr1 hr2 (Sha512.hash_size _) (Sha512.hash_size _) he h
      · simp only [hp, ↓reduceIte] at h
        by_cases he : Sha512.hash (nodeIn dom d (idx / 2) s a) = Sha512.hash (nodeIn dom d (idx / 2) t b)
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ⟨fun hh => hab (nodeIn_inj (by rw [hsz, htz]) (by rw [ha, hb]) hh).2, he⟩
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ih rest2 _ _ (idx / 2) (d + 1) hl' hr1 hr2 (Sha512.hash_size _) (Sha512.hash_size _) he h

theorem sibs_length (path : ByteArray) (depth : Nat) : (sibs path depth).length = depth := by
  simp [sibs]

/-- A path of `depth` 64-byte siblings splits into 64-byte siblings. -/
theorem sibs_size {path : ByteArray} {depth : Nat} (h : path.size = 64 * depth) : ∀ x ∈ sibs path depth, x.size = 64 := by
  intro x hx
  simp only [sibs, List.mem_map, List.mem_range] at hx
  obtain ⟨d, hd, rfl⟩ := hx
  simp only [ByteArray.size_extract, h]
  omega

/-- **The opening's extractor:** for leaves `x` and `y` opened through `path` and `path'` at one position `pos` under
`reg`'s root, the two leaves' frames (under `reg`'s schema) if their frame-v3 leaves are one, else the climb's pair. -/
def openPair (reg : Port) (pos : Nat) (x path y path' : ByteArray) : ByteArray × ByteArray :=
  let lx := leafInAs reg.schema reg.domain pos x
  let ly := leafInAs reg.schema reg.domain pos y
  if Sha512.hash lx = Sha512.hash ly then (lx, ly)
  else climbPair reg.domain (Sha512.hash lx) (Sha512.hash ly) pos 0 (sibs path (log2ceil reg.leaves))
    (sibs path' (log2ceil reg.leaves))

/-- **Registered-path binding.** Two openings the verifier accepts (`opensLeaf`) at one position `pos` under one registered
root, of two different 64-byte hm96 leaves `x ≠ y` of the port's schema: `openPair` computes two different SHA-512
inputs with one digest from them. The frame binds the payload's position, the node's level and index, and each child's
length, so the pair is a collision of two leaves' frames or of two nodes' frames. -/
theorem opensLeaf_binding (reg : Port) (pos : Nat) (x path y path' : ByteArray) (hx : x.size = 64) (hy : y.size = 64)
    (hne : x ≠ y) (h1 : opensLeaf reg x pos path = true) (h2 : opensLeaf reg y pos path' = true) :
    Collides (openPair reg pos x path y path') := by
  simp only [opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq : climbL reg.domain (leafAs reg.schema reg.domain pos x) pos 0 (sibs path (log2ceil reg.leaves)) =
      climbL reg.domain (leafAs reg.schema reg.domain pos y) pos 0 (sibs path' (log2ceil reg.leaves)) :=
    h1.2.trans h2.2.symm
  rw [leafAs_eq, leafAs_eq] at heq
  by_cases hl : Sha512.hash (leafInAs reg.schema reg.domain pos x) = Sha512.hash (leafInAs reg.schema reg.domain pos y)
  · simp only [openPair, hl, ↓reduceIte]
    exact ⟨fun hh => hne (leafInAs_inj (by rw [hx, hy]) hh), hl⟩
  · simp only [openPair, hl, ↓reduceIte]
    exact climbPair_collides reg.domain _ _ _ _ pos 0 (by rw [sibs_length, sibs_length]) (sibs_size h1.1.2)
      (sibs_size h2.1.2) (Sha512.hash_size _) (Sha512.hash_size _) hl heq

/-- The proposition `opensLeaf_binding` implies (satisfiable by counting on its own; `opensLeaf_binding` is the
reduction). -/
theorem opensLeaf_collision (reg : Port) (pos : Nat) (x path y path' : ByteArray) (hx : x.size = 64) (hy : y.size = 64)
    (hne : x ≠ y) (h1 : opensLeaf reg x pos path = true) (h2 : opensLeaf reg y pos path' = true) : Sha512.Collision :=
  let h := opensLeaf_binding reg pos x path y path' hx hy hne h1 h2
  ⟨_, _, h.1, h.2⟩

/-- An opening's position is among the registered value's leaves. -/
theorem opens_lt {reg : Port} {bc path : ByteArray} {pos : Nat} (h : opens reg bc pos path = true) :
    pos < reg.leaves := by
  simp only [opens, opensLeaf, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.1

/-- No commit strings open a registered root at every position: the positions past its leaves open nothing. -/
theorem not_opens_all (reg : Port) (bc path : Nat → ByteArray) : ¬ ∀ i, opens reg (bc i) i (path i) = true :=
  fun h => Nat.lt_irrefl _ (opens_lt (h reg.leaves))

/-- **A commit string's extractor:** for commit strings `bc` and `bc'` opened through `path` and `path'` at one position
`pos` under `reg`'s root, the two tree leaves' inputs if their tree leaves are one, else the opening's pair
(`openPair`) of the two tree leaves. -/
def bcPair (reg : Port) (pos : Nat) (bc path bc' path' : ByteArray) : ByteArray × ByteArray :=
  if Hm96.Default512.treeLeaf bc = Hm96.Default512.treeLeaf bc' then
    (Hm96.Default512.leafPrefix ++ bc, Hm96.Default512.leafPrefix ++ bc')
  else openPair reg pos (Hm96.Default512.treeLeaf bc) path (Hm96.Default512.treeLeaf bc') path'

/-- **A registered root binds each position's commit string.** Two different commit strings the verifier opens at one
position under one registered root (`opens`): `bcPair` computes two different SHA-512 inputs with one digest from them,
the tree leaves' inputs (one leaf prefix, so they differ) or the opening's (`opensLeaf_binding`). -/
theorem opens_binding (reg : Port) (pos : Nat) (bc path bc' path' : ByteArray) (hne : bc ≠ bc')
    (h1 : opens reg bc pos path = true) (h2 : opens reg bc' pos path' = true) :
    Collides (bcPair reg pos bc path bc' path') := by
  unfold bcPair
  split
  · rename_i hl
    exact ⟨fun e => hne ((ByteArray.append_right_inj _).mp e), hl⟩
  · rename_i hl
    exact opensLeaf_binding reg pos _ path _ path' (Sha512.hash_size _) (Sha512.hash_size _) hl h1 h2

/-! ### A recursive session's inner statement: its description (PROTOCOL.md §16.16)

`checkDescription_ok`: when `Registered.checkDescription` accepts, each of the verifier's registered ports `reg` is the
circuit's port `p`, whose table is `reg`'s registration in order: table row `r` (`rowsOf c pub reg r`) opens `reg`'s root
at `r` through the path the header gives it (`pathOf`) for every `r < reg.leaves`, every instance's row of the port is
one of them, and each of them is some instance's row. -/

theorem portOf_some {c : Circuit} {name : String} {p : Nat} (h : portOf c name = some p) :
    p < c.ports.size ∧ c.ports[p]!.name = name := by
  unfold portOf at h
  exact ⟨List.mem_range.mp (List.mem_of_find?_eq_some h), by simpa using List.find?_some h⟩

/-- A port `checkDescriptionPort` accepts is the circuit's port `p`, of `reg`'s width, whose table has `reg.leaves` rows,
row `r` opening `reg`'s root at `r` through its path; every instance's ref names one of them, and each is some instance's
ref. -/
theorem checkDescriptionPort_ok {c : Circuit} {pub : Public} {t : Tables} {reg : Port}
    (h : checkDescriptionPort c pub t reg = .ok ()) :
    ∃ p positions paths, portOf c reg.name = some p ∧ reg.fits c.ports[p]! = true ∧
      entry pub reg = .ok (positions, paths) ∧ t.counts.getD p 0 = reg.leaves ∧
      (∀ r < reg.leaves, opens reg (tableRow pub t p r) r (paths.getD r ByteArray.empty) = true) ∧
      (∀ i < pub.n, ref t c.ports.size p i < reg.leaves) ∧
      (∀ r < reg.leaves, ∃ i < pub.n, ref t c.ports.size p i = r) := by
  unfold checkDescriptionPort at h
  simp only at h
  split at h
  · rename_i p hp
    split at h
    · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
    · split at h
      · exact absurd (except_bind_ok h) fun ⟨_, h1, _⟩ => throw_ne_ok _ _ h1
      · rename_i hfit
        obtain ⟨⟨positions, paths⟩, hent, h⟩ := except_bind_ok h
        simp only at h
        split at h
        · exact absurd h (throw_ne_ok _ _)
        · rename_i hrows
          split at h
          · exact absurd h (throw_ne_ok _ _)
          · split at h
            · exact absurd h (throw_ne_ok _ _)
            · rename_i hpos
              split at h
              · exact absurd h (throw_ne_ok _ _)
              · rename_i hrow
                split at h
                · exact absurd h (throw_ne_ok _ _)
                · rename_i href
                  split at h
                  · exact absurd h (throw_ne_ok _ _)
                  · rename_i hun
                    have hr : t.counts.getD p 0 = reg.leaves := by simpa using hrows
                    refine ⟨p, positions, paths, hp, by simpa using hfit, hent, hr, fun r hr' => ?_, fun i hi => ?_,
                      fun r hr' => ?_⟩
                    · have hpr := List.find?_eq_none.mp hpos r (List.mem_range.mpr (hr ▸ hr'))
                      have ho := List.find?_eq_none.mp hrow r (List.mem_range.mpr (hr ▸ hr'))
                      simp only [bne_iff_ne, ne_eq, Classical.not_not] at hpr
                      simpa [hpr] using ho
                    · have hi' := List.find?_eq_none.mp href i (List.mem_range.mpr hi)
                      simp only [Bool.not_eq_true', decide_eq_false_iff_not, Nat.not_lt] at hi'
                      omega
                    · have hu := List.find?_eq_none.mp hun r (List.mem_range.mpr (hr ▸ hr'))
                      simpa using hu
  · exact absurd h (throw_ne_ok _ _)

/-- `checkDescription` accepting: the statement is a shared-row file, with tables `t`, and `checkDescriptionPort`
accepts each of the verifier's registered ports. -/
theorem checkDescription_tables {c : Circuit} {pub : Public} {own : Array Port}
    (h : checkDescription c pub own = .ok ()) :
    ∃ t, pub.tables = some t ∧ ∀ reg ∈ own, checkDescriptionPort c pub t reg = .ok () := by
  unfold checkDescription at h
  simp only at h
  split at h
  · obtain ⟨_, _, h⟩ := except_bind_ok h
    split at h
    · rename_i t ht
      exact ⟨t, ht, fun reg hreg => forM_ok h reg (by simpa using hreg)⟩
    · exact absurd h (throw_ne_ok _ _)
  · exact absurd h (throw_ne_ok _ _)

/-- **`Registered.checkDescription` accepting.** For each of the verifier's registered ports `reg` (the circuit's port
`p`, whose rows of u16 words are `reg`'s registered rows: `Port.fits`), every registered row `r < reg.leaves` of the
statement (`rowsOf`, its `b ‖ c`) opens `reg`'s root at `r` through the path the header gives it (`pathOf`); every
instance's row of the port (`Public.digest`) is the registered row its ref names; and every registered row is some
instance's. -/
theorem checkDescription_ok {c : Circuit} {pub : Public} {own : Array Port} (h : checkDescription c pub own = .ok ())
    {reg : Port} (hreg : reg ∈ own) :
    ∃ t p, pub.tables = some t ∧ portOf c reg.name = some p ∧ p < c.ports.size ∧ reg.fits c.ports[p]! = true ∧
      (∀ r < reg.leaves, opens reg (rowsOf c pub reg r) r (pathOf pub reg r) = true) ∧
      (∀ i < pub.n, ref t c.ports.size p i < reg.leaves ∧
        pub.digest i p c.ports.size = rowsOf c pub reg (ref t c.ports.size p i)) ∧
      (∀ r < reg.leaves, ∃ i < pub.n, ref t c.ports.size p i = r) := by
  obtain ⟨t, ht, hall⟩ := checkDescription_tables h
  obtain ⟨p, positions, paths, hp, hfit, hent, -, hopen, href, hread⟩ := checkDescriptionPort_ok (hall reg hreg)
  have hrow : ∀ r, rowsOf c pub reg r = tableRow pub t p r := fun r => by simp only [rowsOf, hp, ht]
  have hpath : ∀ r, pathOf pub reg r = paths.getD r ByteArray.empty := fun r => by simp only [pathOf, hent]
  refine ⟨t, p, ht, hp, (portOf_some hp).1, hfit, fun r hr => ?_, fun i hi => ⟨href i hi, ?_⟩, hread⟩
  · rw [hrow, hpath]
    exact hopen r hr
  · rw [hrow]
    simp only [Public.digest, ht, tableRow, ref]

end Flock.Registered
