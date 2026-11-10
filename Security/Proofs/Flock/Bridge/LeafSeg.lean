module

public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Flock.Claims.Sampling.Cost
public import Mathlib.Tactic.Tauto

/-!
# The row-seg leaf binds (hm96-sha512/row-seg/v1), and the deployed commitment with it

`commitDepL` hashes a segmented port's row as `sha512/row-seg/v1`'s outer message `Q(ℓ) ‖ d₀ ‖ … ‖ d_{s-1}`, whose
digests `d_j` are SHA-512's own: that message is not injective in the row, so `hm96Leaf_binds` (whose row must place
the value) doesn't apply there. Here the leaf asks the oracle for each segment's digest, then hashes the outer message
of its answers as hm96 does:

* `queryAll qs k`: the queries `qs` in order, then `k` on their answers (`queryAll_eval`).
* `hm96Leaf_pair`: `hm96Leaf_binds` at two openings whose rows differ, at any row: it reads each opening's row only.
* `hm96SegsLeaf segs outer salt`: at an opening `o`, the queries `segs o`, then hm96's leaf of `outer o ds` at their
  answers `ds`, refusing an answer that isn't 64 bytes. **It binds at every `H`** (`hm96SegsLeaf_binds`) where the
  outer message splits into its answers and two openings at one position, of different values, differ in a segment:
  one leaf of two openings is a collision of hm96's queries where their outer messages differ, and otherwise one answer
  to two different segment queries. Its calls are the segments' and hm96's three (`hm96SegsLeaf_calls_le`).
* `segLeaf place bits salt`: `sha512/row-seg/v1`'s, the segments `RowSeg.segMsg` of the row's bytes
  (`SamplingDeploy.rowBits`) and the outer message `Q(ℓ) ‖ d₀ ‖ …`. It binds (`segLeaf_binds`), and at SHA-512 it is
  hm96's leaf of `RowSeg.enc`, the message `commitDepL` hashes (`segLeaf_sha512`).
* `rowLeaf C rows c place bits salt`: each position's port's leaf, `segLeaf` at a segmented port (`segAt`) and hm96's
  of `rowDep` elsewhere. It binds at every circuit file (`rowLeaf_binds`), and at SHA-512 it is `commitDepL`'s leaf
  (`rowLeaf_sha512`), in at most `RowSeg.count ℓ + 3` calls (`rowLeaf_calls_le`).
* `commitRowL C rows c L roots`: `commitDepL` with `rowLeaf`. It binds at every `H` with no hypothesis on the ports
  (`commitRowL_binds`), and it opens at SHA-512 exactly what `commitDepL` opens (`commitRowL_sha512`).
* `Binding_of_rowLeaf`: **the claims' `Binding`** for a commitment of `rowLeaf` leaves under the registered path check.
-/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling

/-- **Queries in order**: ask `qs`, then run `k` on their answers. -/
def queryAll {α : Type} : List Msg → (List Msg → OC α) → OC α
  | [], k => k []
  | q :: qs, k => .query q fun a => queryAll qs fun as => k (a :: as)

theorem queryAll_eval {α : Type} (H : Msg → Msg) : ∀ (qs : List Msg) (k : List Msg → OC α),
    (queryAll qs k).eval H = (((k (qs.map H)).eval H).1, qs.map (fun q => (q, H q)) ++ ((k (qs.map H)).eval H).2)
  | [], k => by simp [queryAll]
  | q :: qs, k => by simp [queryAll, OC.eval, queryAll_eval H qs]

theorem hm96Leaf_congr {row row' : Opening Bool → Msg} (salt : Opening Bool → Msg) {o : Opening Bool}
    (h : row o = row' o) : hm96Leaf row salt o = hm96Leaf row' salt o := by
  unfold hm96Leaf
  rw [h]

/-- **hm96's leaf binds two openings whose rows differ**: `hm96Leaf_binds` at the row that is `row o` at `o`'s value
and `row o'` at the other, which places the value. -/
theorem hm96Leaf_pair (H : Msg → Msg) {row salt : Opening Bool → Msg} {o o' : Opening Bool} {l : Msg}
    (hpos : o.pos = o'.pos) (hne : o.val ≠ o'.val) (hrow : row o ≠ row o')
    (h : ((hm96Leaf row salt o).eval H).1 = some l) (h' : ((hm96Leaf row salt o').eval H).1 = some l) :
    LogColl (((hm96Leaf row salt o).eval H).2 ++ ((hm96Leaf row salt o').eval H).2) := by
  have hb : ∀ a b c : Bool, a ≠ c → b ≠ a → b = c := by decide
  have hr : RowPlaces fun x => if x.val = o.val then row o else row o' := by
    intro x x' _ hx
    dsimp only
    by_cases hxo : x.val = o.val
    · rw [ite_eq_left hxo, ite_eq_right fun e => hx (hxo.trans e.symm)]
      exact hrow
    · rw [ite_eq_right hxo, ite_eq_left (hb _ _ _ hxo (Ne.symm hx))]
      exact Ne.symm hrow
  have e := hm96Leaf_congr (row := fun x => if x.val = o.val then row o else row o') (row' := row) salt (o := o)
    (ite_eq_left rfl)
  have e' := hm96Leaf_congr (row := fun x => if x.val = o.val then row o else row o') (row' := row) salt (o := o')
    (ite_eq_right fun e => hne e.symm)
  have hc := hm96Leaf_binds H hr salt o o' l hpos hne (by rw [e]; exact h) (by rw [e']; exact h')
  rwa [e, e'] at hc

/-- **hm96 over segment digests**: the segments' queries `segs o`, then hm96's leaf of the outer message `outer o ds`
at their answers `ds`, refusing an answer that isn't 64 bytes, which SHA-512's never are. -/
def hm96SegsLeaf (segs : Opening Bool → List Msg) (outer : Opening Bool → List Msg → Msg) (salt : Opening Bool → Msg)
    (o : Opening Bool) : OC (Option Msg) :=
  queryAll (segs o) fun ds => if ∀ d ∈ ds, d.length = 64 then hm96Leaf (fun x => outer x ds) salt o else .ret none

/-- **hm96 over segment digests binds, for every `H`**, where the outer message splits into its 64-byte answers
(`hsplit`) and openings at one position, of different values, differ in a segment (`hdiff`): their outer messages
differ, and hm96's queries collide (`hm96Leaf_pair`), or they agree, and so do the answers to two different segment
queries. -/
theorem hm96SegsLeaf_binds (H : Msg → Msg) {segs : Opening Bool → List Msg} {outer : Opening Bool → List Msg → Msg}
    (hsplit : ∀ o o' : Opening Bool, o.pos = o'.pos → ∀ ds ds' : List Msg, ds.length = (segs o).length →
      ds'.length = (segs o').length → (∀ d ∈ ds, d.length = 64) → (∀ d ∈ ds', d.length = 64) →
      outer o ds = outer o' ds' → ds = ds')
    (hdiff : ∀ o o' : Opening Bool, o.pos = o'.pos → o.val ≠ o'.val →
      ∃ j, ∃ (hj : j < (segs o).length) (hj' : j < (segs o').length), (segs o)[j] ≠ (segs o')[j])
    (salt : Opening Bool → Msg) : LeafBinds H (hm96SegsLeaf segs outer salt) := by
  intro o o' l hpos hne h h'
  simp only [hm96SegsLeaf, queryAll_eval] at h h' ⊢
  have hd : ∀ d ∈ (segs o).map H, d.length = 64 := by
    by_contra hd
    rw [ite_eq_right hd] at h
    simp [OC.eval] at h
  have hd' : ∀ d ∈ (segs o').map H, d.length = 64 := by
    by_contra hd
    rw [ite_eq_right hd] at h'
    simp [OC.eval] at h'
  rw [ite_eq_left hd] at h ⊢
  rw [ite_eq_left hd'] at h' ⊢
  by_cases hr : outer o ((segs o).map H) = outer o' ((segs o').map H)
  · have hm := hsplit o o' hpos _ _ (List.length_map _) (List.length_map _) hd hd' hr
    obtain ⟨j, hj, hj', hs⟩ := hdiff o o' hpos hne
    refine ⟨((segs o)[j], H (segs o)[j]), ?_, ((segs o')[j], H (segs o')[j]), ?_, hs, ?_⟩
    · exact List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨_, List.getElem_mem hj, rfl⟩))
    · exact List.mem_append_right _ (List.mem_append_left _ (List.mem_map.2 ⟨_, List.getElem_mem hj', rfl⟩))
    · have e := List.getElem_of_eq hm (by rw [List.length_map]; exact hj)
      rw [List.getElem_map, List.getElem_map] at e
      exact e
  · rw [show hm96Leaf (fun x => outer x ((segs o).map H)) salt o =
        hm96Leaf (fun x => outer x ((segs x).map H)) salt o from rfl] at h ⊢
    rw [show hm96Leaf (fun x => outer x ((segs o').map H)) salt o' =
        hm96Leaf (fun x => outer x ((segs x).map H)) salt o' from rfl] at h' ⊢
    exact (hm96Leaf_pair H hpos hne hr h h').mono fun q hq => by
      simp only [List.mem_append] at hq ⊢
      tauto

/-- **Its calls**: one per segment, then hm96's three. -/
theorem hm96SegsLeaf_calls_le (H : Msg → Msg) (segs : Opening Bool → List Msg) (outer : Opening Bool → List Msg → Msg)
    (salt : Opening Bool → Msg) (o : Opening Bool) :
    ((hm96SegsLeaf segs outer salt o).eval H).2.length ≤ (segs o).length + 3 := by
  simp only [hm96SegsLeaf, queryAll_eval, List.length_append, List.length_map]
  split
  · rw [hm96Leaf_log]
    simp
  · simp [OC.eval]

/-- Lists of one length of 64-byte digests with one concatenation are one list. -/
theorem flatten_inj64 {ds ds' : List Msg} (hl : ds.length = ds'.length) (h : ∀ d ∈ ds, d.length = 64)
    (h' : ∀ d ∈ ds', d.length = 64) (he : ds.flatten = ds'.flatten) : ds = ds' := by
  refine List.eq_iff_flatten_eq.2 ⟨he, ?_⟩
  rw [List.map_congr_left (g := fun _ => 64) h, List.map_congr_left (g := fun _ => 64) h', List.map_const',
    List.map_const', hl]

/-! ### `sha512/row-seg/v1` -/

/-- **A row's segment messages**: `RowSeg.segMsg` of the bytes of the opening's row (`SamplingDeploy.rowBits`), one
per segment of its width. -/
def segsOf (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (o : Opening Bool) : List Msg :=
  (List.range (FlockSoundness.Binding.RowSeg.count (place o.pos).1)).map
    (FlockSoundness.Binding.RowSeg.segMsg (place o.pos).1 (Flock.HmRow.packBits (SamplingDeploy.rowBits place bits o)))

/-- **The outer message at segment digests `ds`**: `Q(ℓ) ‖ d₀ ‖ …` at the row's width `ℓ`. -/
def outerOf (place : ℕ → (w : ℕ) × Fin w) (o : Opening Bool) (ds : List Msg) : Msg :=
  (Flock.HmRow.rowPrefixSeg (place o.pos).1).data.toList ++ ds.flatten

/-- **`hm96-sha512/row-seg/v1`'s leaf as an oracle program**: each segment's digest, then hm96 over the outer message. -/
def segLeaf (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg) :
    Opening Bool → OC (Option Msg) :=
  hm96SegsLeaf (segsOf place bits) (outerOf place) salt

/-- **The row-seg leaf binds, for every `H`**: one prefix at one position, so one outer message is one list of
digests; two rows at one position, of different values, differ in a segment's bytes (`RowSeg.exists_segBytes_ne`),
so in that segment's message. -/
theorem segLeaf_binds (H : Msg → Msg) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool)
    (salt : Opening Bool → Msg) : LeafBinds H (segLeaf place bits salt) := by
  refine hm96SegsLeaf_binds H ?_ ?_ salt
  · intro o o' hpos ds ds' hl hl' hd hd' he
    simp only [outerOf, hpos] at he
    simp only [segsOf, List.length_map, List.length_range, hpos] at hl hl'
    exact flatten_inj64 (hl.trans hl'.symm) hd hd' (List.append_cancel_left he)
  · rintro ⟨p, v, d⟩ ⟨p', v', d'⟩ hp hv
    simp only at hp hv
    subst hp
    have hlen : ∀ o : Opening Bool,
        (Flock.HmRow.packBits (SamplingDeploy.rowBits place bits o)).length = ((place o.pos).1 + 7) / 8 := fun o => by
      rw [FlockLevel3.RowV2.packBits_length, SamplingDeploy.rowBits, List.length_ofFn]
    have hk : Flock.HmRow.packBits (SamplingDeploy.rowBits place bits ⟨p, v, d⟩) ≠
        Flock.HmRow.packBits (SamplingDeploy.rowBits place bits ⟨p, v', d'⟩) := fun e => by
      have hb := FlockLevel3.RowV2.packBits_inj (by simp only [SamplingDeploy.rowBits, List.length_ofFn]) e
      simp only [SamplingDeploy.rowBits] at hb
      exact hv (by simpa using congrFun (List.ofFn_inj.1 hb) (place p).2)
    obtain ⟨j, hj, hs⟩ := FlockSoundness.Binding.RowSeg.exists_segBytes_ne (hlen ⟨p, v, d⟩) (hlen ⟨p, v', d'⟩) hk
    refine ⟨j, by simpa [segsOf] using hj, by simpa [segsOf] using hj, ?_⟩
    simp only [segsOf, List.getElem_map, List.getElem_range, FlockSoundness.Binding.RowSeg.segMsg]
    exact fun e => hs (List.append_cancel_left e)

/-- **At SHA-512 the row-seg leaf is hm96's leaf of `RowSeg.enc`**: every answer is 64 bytes, and the outer message of
the segments' SHA-512 digests is `RowSeg.enc`, the message `rowMsg` hashes at a segmented port. -/
theorem segLeaf_sha512 (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg)
    (o : Opening Bool) :
    ((segLeaf place bits salt o).eval sha512Msg).1 =
      ((hm96Leaf (fun x => FlockSoundness.Binding.RowSeg.enc (place x.pos).1
        (Flock.HmRow.packBits (SamplingDeploy.rowBits place bits x))) salt o).eval sha512Msg).1 := by
  have e : ∀ m, sha512Msg m = FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 m) := fun m => by
    unfold sha512Msg bytes
    rw [FlockSoundness.Binding.digBytes, FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.H512_val]
    rfl
  have hd : ∀ d ∈ (segsOf place bits o).map sha512Msg, d.length = 64 := fun d hd => by
    obtain ⟨m, -, rfl⟩ := List.mem_map.1 hd
    rw [e]
    exact FlockSoundness.Refine.hL512 _
  have ho : outerOf place o ((segsOf place bits o).map sha512Msg) =
      FlockSoundness.Binding.RowSeg.enc (place o.pos).1 (Flock.HmRow.packBits (SamplingDeploy.rowBits place bits o)) := by
    simp only [outerOf, segsOf, FlockSoundness.Binding.RowSeg.enc, FlockSoundness.Binding.RowSeg.digests, List.map_map,
      Function.comp_def, e, List.flatMap_def]
  simp only [segLeaf, hm96SegsLeaf, queryAll_eval]
  rw [ite_eq_left hd]
  simp only [hm96Leaf_fst, hm96Outer, ho]

/-! ### Each port's leaf -/

/-- Whether claims position `i`'s row is at a segmented port. -/
def segAt (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (i : ℕ) : Bool :=
  match portOfPos C rows c i with
  | some p => p.segs.isSome
  | none => false

/-- **An opening's leaf in its port's format**: the row-seg leaf at a segmented port, hm96's of `rowDep` elsewhere. -/
def rowLeaf (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg) (o : Opening Bool) :
    OC (Option Msg) :=
  if segAt C rows c o.pos then segLeaf place bits salt o else hm96Leaf (rowDep C rows c place bits) salt o

/-- **`rowDep` places the value at an unsegmented port**: `rowDep_places` at one position. -/
theorem rowDep_ne (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) {o o' : Opening Bool}
    (hs : ¬ segAt C rows c o.pos = true) (hpos : o.pos = o'.pos) (hne : o.val ≠ o'.val) :
    rowDep C rows c place bits o ≠ rowDep C rows c place bits o' := by
  obtain ⟨i, v, d⟩ := o
  obtain ⟨i', v', d'⟩ := o'
  simp only at hpos hne hs
  subst hpos
  intro h
  unfold rowDep at h
  dsimp only at h
  cases hp : portOfPos C rows c i with
  | none =>
    simp only [hp] at h
    exact SamplingDeploy.row_places place bits ⟨i, v, d⟩ ⟨i, v', d'⟩ rfl hne h
  | some p =>
    simp only [hp] at h
    have hps : p.segs.isSome = false := by simpa [segAt, hp] using hs
    have hb := rowMsg_inj hps (by simp [SamplingDeploy.rowBits]) h
    simp only [SamplingDeploy.rowBits] at hb
    exact hne (by simpa using congrFun (List.ofFn_inj.1 hb) (place i).2)

/-- **At a segmented port `rowDep` is `RowSeg.enc`** of the row's bytes. -/
theorem rowDep_seg (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) {o : Opening Bool} (hs : segAt C rows c o.pos = true) :
    rowDep C rows c place bits o =
      FlockSoundness.Binding.RowSeg.enc (place o.pos).1 (Flock.HmRow.packBits (SamplingDeploy.rowBits place bits o)) := by
  cases hp : portOfPos C rows c o.pos with
  | none => simp [segAt, hp] at hs
  | some p =>
    have hps : p.segs.isSome = true := by simpa [segAt, hp] using hs
    simp only [rowDep, hp, rowMsg, ite_eq_left hps, SamplingDeploy.rowBits, List.length_ofFn]

/-- **`rowLeaf` binds, for every `H`, at every circuit file**: one position is one port, whose leaf binds, the row-seg
leaf (`segLeaf_binds`) or hm96's of a row that places the value there (`hm96Leaf_pair`, `rowDep_ne`). -/
theorem rowLeaf_binds (H : Msg → Msg) (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg) :
    LeafBinds H (rowLeaf C rows c place bits salt) := by
  intro o o' l hpos hne h h'
  unfold rowLeaf at h h' ⊢
  rw [← hpos] at h' ⊢
  by_cases hs : segAt C rows c o.pos = true
  · simp only [ite_eq_left hs] at h h' ⊢
    exact segLeaf_binds H place bits salt o o' l hpos hne h h'
  · simp only [ite_eq_right hs] at h h' ⊢
    exact hm96Leaf_pair H hpos hne (rowDep_ne C rows c place bits hs hpos hne) h h'

/-- **At SHA-512 `rowLeaf` is `commitDepL`'s leaf**, hm96's of `rowDep`. -/
theorem rowLeaf_sha512 (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg) (o : Opening Bool) :
    ((rowLeaf C rows c place bits salt o).eval sha512Msg).1 =
      ((hm96Leaf (rowDep C rows c place bits) salt o).eval sha512Msg).1 := by
  unfold rowLeaf
  by_cases hs : segAt C rows c o.pos = true
  · rw [ite_eq_left hs, segLeaf_sha512]
    simp only [hm96Leaf_fst, hm96Outer, rowDep_seg C rows c place bits hs]
  · rw [ite_eq_right hs]

/-- **`rowLeaf`'s calls**: at most a segment's each and hm96's three. -/
theorem rowLeaf_calls_le (H : Msg → Msg) (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool) (salt : Opening Bool → Msg)
    (o : Opening Bool) :
    ((rowLeaf C rows c place bits salt o).eval H).2.length ≤ FlockSoundness.Binding.RowSeg.count (place o.pos).1 + 3 := by
  unfold rowLeaf
  split
  · have hc := hm96SegsLeaf_calls_le H (segsOf place bits) (outerOf place) salt o
    simp only [segsOf, List.length_map, List.length_range] at hc
    exact hc
  · rw [hm96Leaf_log]
    simp

/-! ### The deployed commitment -/

theorem pathCommit_opens_congr {H : Msg → Msg} {at_ : List Msg → ℕ → Option (Msg × ℕ)}
    {leaf leaf' : Opening Bool → OC (Option Msg)} {π : Opening Bool → Msg} {path : Msg → ℕ → Msg → Msg → OC Bool}
    {layer layer' : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg}
    (h : ∀ o, ((leaf o).eval H).1 = ((leaf' o).eval H).1) (roots : List Msg) (o : Opening Bool) :
    (pathCommit at_ leaf π path layer).Opens H roots o = (pathCommit at_ leaf' π path layer').Opens H roots o := by
  unfold CommitmentScheme.Opens
  simp only [pathCommit]
  cases at_ roots o.pos with
  | none => rfl
  | some ri => simp only [ocBind_eval, h o]

/-- **The deployed trace commitment with each port's own leaf**: `commitDepL` with `rowLeaf`. -/
noncomputable def commitRowL (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (L : LeafMap) (roots : List Msg) : CommitmentScheme Bool :=
  pathCommit (atOf C rows) (rowLeaf C rows c (placeOf C rows) (bitsOf L.d) saltOf) (pathOfL C rows L)
    (regPath (portsOfL c L roots)) fun _ _ _ => []

/-- **`commitRowL` binds, at every `H`**, at every circuit file, segmented ports included. -/
theorem commitRowL_binds (H : Msg → Msg) (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt)
    (c : Flock.Circuit) (L : LeafMap) (roots : List Msg) : (commitRowL C rows c L roots).Binds H :=
  pathCommit_binds (rowLeaf_binds H C rows c _ _ _) (regPath_binds H _)

/-- **At SHA-512 `commitRowL` opens what `commitDepL` opens.** -/
theorem commitRowL_sha512 (C : Verity.Boolean.Circuit) (rows : Fin C.N → Option RowAt) (c : Flock.Circuit)
    (L : LeafMap) (roots rs : List Msg) (o : Opening Bool) :
    (commitRowL C rows c L roots).Opens sha512Msg rs o = (commitDepL C rows c L roots).Opens sha512Msg rs o := by
  unfold commitRowL commitDepL
  exact pathCommit_opens_congr (rowLeaf_sha512 C rows c (placeOf C rows) (bitsOf L.d) saltOf) rs o

/-- **Sampling: binding, at `rowLeaf` leaves under the registered path check**, at every circuit file. -/
theorem Binding_of_rowLeaf {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T)
    (own : Array Flock.Registered.Port) (at_ : List Msg → ℕ → Option (Msg × ℕ)) (C : Verity.Boolean.Circuit)
    (rows : Fin C.N → Option RowAt) (c : Flock.Circuit) (place : ℕ → (w : ℕ) × Fin w) (bits : Msg → ℕ → Bool)
    (salt π : Opening Bool → Msg) (layer : (Msg → Msg) → List Msg → List (Opening Bool) → List Msg)
    (hD : D.commit = pathCommit at_ (rowLeaf C rows c place bits salt) π (regPath own) layer) :
    Claims.Sampling.Binding D :=
  Binding_of_regPath D own at_ _ π layer hD fun H => rowLeaf_binds H C rows c place bits salt

end Flock.Bridge
