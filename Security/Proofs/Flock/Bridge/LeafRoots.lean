module

public import Proofs.Flock.Bridge.PortRoots
public import Proofs.Flock.Bridge.LeafMap
import all Init.Data.String.Defs

/-!
# Two ports with one root at a leaf map and different domains are a SHA-512 collision

`PortRoots`' finder at a leaf map (`LeafMap`), whose ports may have different leaf counts (a `shared_rows` file's
tables), so trees of different depths. Two top frames at different depths need not differ from the back, since a node
frame's depth part (`FV3.uint`) has no fixed width, so the finder walks the two ports' left spines down from their roots.

* `spineL c L q k`: the frame port `q`'s tree hashes to the first member of its level `k` (`hash_spineL`): above level 0
  the node over the level below's first two members, at level 0 the first leaf's frame, or with no leaf the lone pad's.
  The root is SHA-512 of the spine's frame at the tree's depth `depthL` (`root_spineL`).
* Two equal node frames of two spines have one left child (`spine_child`), and two equal frames at level 0 of either
  spine have one domain (`spine_bottom`), since frames of different kinds (pad, leaf, node) differ at their tags' first
  bytes (`frame_ne_of_head`). So two ports with one root and different domains hash two different frames of their spines
  to one digest at some level (`dom_of_spine`, `spineColl_ne_none`).
* `rootsFinderL`: no coins; it recomputes every port's registered root (`rootsCostL` SHA-512 calls) and outputs, at the
  first pair of ports with one root and different domains (`collPairL`), the spines' frames at the first level where
  they collide (`spineColl`, `rootsCollL`): a SHA-512 collision (`rootsCollL_collides`), and none only where
  `RootFixesDomainL` holds (`fixesL_of_none`). So a probability `p ≤ 1` bounded at every registration with
  `RootFixesDomainL` is bounded at every registration with `ecr/sha-512`'s term at that finder added
  (`le_add_ecrBoundL`).
-/

@[expose] public section

namespace Flock.Bridge

open Claims Claims.Sampling
open Flock.SecurityProofs
open Properties.NetworkCertifier.Bytes (bs bs_inj be32_size)
open scoped ENNReal

/-! ## Frames of different kinds -/

theorem foldl_step_data : ∀ (ps : List ByteArray) (a : ByteArray),
    ∃ l : List UInt8, (ps.foldl Registered.step a).data.toList = a.data.toList ++ l
  | [], a => ⟨[], by simp⟩
  | p :: ps, a => by
    obtain ⟨l, hl⟩ := foldl_step_data ps (Registered.step a p)
    exact ⟨(Flock.be64 p.size.toUInt64).data.toList ++ p.data.toList ++ l, by
      rw [List.foldl_cons, hl]
      simp [Registered.step, ByteArray.data_append]⟩

/-- **Frames whose tags open with different bytes differ**, whatever their parts. -/
theorem frame_ne_of_head {t t' : String} {a a' : UInt8} {r r' : List UInt8} (ht : t.toUTF8.data.toList = a :: r)
    (ht' : t'.toUTF8.data.toList = a' :: r') (ha : a ≠ a') (ps ps' : List ByteArray) :
    Registered.frame t ps ≠ Registered.frame t' ps' := by
  intro h
  obtain ⟨l, hl⟩ := foldl_step_data ps (Flock.FV3.FRAME ++ Flock.be32 t.toUTF8.size.toUInt32 ++ t.toUTF8)
  obtain ⟨l', hl'⟩ := foldl_step_data ps' (Flock.FV3.FRAME ++ Flock.be32 t'.toUTF8.size.toUInt32 ++ t'.toUTF8)
  have e := congrArg (fun b : ByteArray => b.data.toList) h
  simp only [Registered.frame] at e
  rw [hl, hl'] at e
  simp only [ByteArray.data_append, Array.toList_append, ht, ht', List.append_assoc] at e
  have e2 := (List.append_inj (List.append_cancel_left e) (by
    simp only [Array.length_toList]
    exact (be32_size _).trans (be32_size _).symm)).2
  simp only [List.cons_append, List.cons.injEq] at e2
  exact ha e2.1

theorem pad_bytes : "pad".toUTF8.data.toList = [112, 97, 100] := by decide

theorem leaf_bytes : "leaf".toUTF8.data.toList = [108, 101, 97, 102] := by decide

theorem node_bytes : "node".toUTF8.data.toList = [110, 111, 100, 101] := by decide

theorem pad_ne_leaf (ps ps' : List ByteArray) : Registered.frame "pad" ps ≠ Registered.frame "leaf" ps' :=
  frame_ne_of_head pad_bytes leaf_bytes (by decide) ps ps'

theorem pad_ne_node (ps ps' : List ByteArray) : Registered.frame "pad" ps ≠ Registered.frame "node" ps' :=
  frame_ne_of_head pad_bytes node_bytes (by decide) ps ps'

theorem leaf_ne_node (ps ps' : List ByteArray) : Registered.frame "leaf" ps ≠ Registered.frame "node" ps' :=
  frame_ne_of_head leaf_bytes node_bytes (by decide) ps ps'

/-! ## A port's left spine -/

/-- Port `q`'s tree depth at `L`: `log2ceil` of its leaf count. -/
def depthL (L : LeafMap) (q : ℕ) : ℕ := Flock.log2ceil (L.cnt q)

/-- **The frame port `q`'s tree hashes to the first member of level `k`**: above level 0 the node over level `k - 1`'s
first two members; at level 0 the first leaf's frame, or with no leaf the lone pad's. -/
def spineL (c : Flock.Circuit) (L : LeafMap) (q k : ℕ) : ByteArray :=
  if k = 0 then
    if L.cnt q = 0 then Registered.frame "pad" [domOf c q, Flock.FV3.uint 0]
    else Registered.leafInAs (schemaOf c q) (domOf c q) 0 (Flock.Hm96.Default512.treeLeaf (L.dig q 0))
  else Registered.nodeIn (domOf c q) (k - 1) 0 (TopRoot.lev (domOf c q) (leavesL c L q) (k - 1))[0]!
    (TopRoot.lev (domOf c q) (leavesL c L q) (k - 1))[1]!

theorem leavesL_size (c : Flock.Circuit) (L : LeafMap) (q : ℕ) : (leavesL c L q).size = L.cnt q := by
  simp [leavesL]

theorem treeLeaf_size (x : ByteArray) : (Flock.Hm96.Default512.treeLeaf x).size = 64 := by
  unfold Flock.Hm96.Default512.treeLeaf
  exact Flock.Sha512.hash_size _

theorem leavesL_width (c : Flock.Circuit) (L : LeafMap) (q : ℕ) : ∀ x ∈ leavesL c L q, x.size = 64 := by
  intro x hx
  simp only [leavesL, Array.mem_map] at hx
  obtain ⟨_, -, rfl⟩ := hx
  exact Flock.Sha512.hash_size _

theorem lev_sizeL (c : Flock.Circuit) (L : LeafMap) (q : ℕ) {k : ℕ} (hk : k ≤ depthL L q) :
    (TopRoot.lev (domOf c q) (leavesL c L q) k).size = 2 ^ (depthL L q - k) := by
  have h := TopRoot.lev_size (domOf c q) (leavesL c L q) k (by rw [leavesL_size]; exact hk)
  rw [leavesL_size] at h
  exact h

theorem lev_widthL (c : Flock.Circuit) (L : LeafMap) (q k : ℕ) {i : ℕ}
    (hi : i < (TopRoot.lev (domOf c q) (leavesL c L q) k).size) :
    ((TopRoot.lev (domOf c q) (leavesL c L q) k)[i]!).size = 64 :=
  TopRoot.lev_width _ _ (leavesL_width c L q) k _ (TopRoot.get!_mem hi)

/-- **Level `k`'s first member is SHA-512 of the spine's frame there**, at every level up to the root's. -/
theorem hash_spineL (c : Flock.Circuit) (L : LeafMap) (q : ℕ) {k : ℕ} (hk : k ≤ depthL L q) :
    Flock.Sha512.hash (spineL c L q k) = (TopRoot.lev (domOf c q) (leavesL c L q) k)[0]! := by
  cases k with
  | zero =>
    by_cases h0 : L.cnt q = 0
    · simp [spineL, h0, TopRoot.lev, TopRoot.base, leavesL, Flock.nextPow2, Flock.log2ceil, Registered.hash_frame]
    · have hc : 0 < (leavesL c L q).size := by rw [leavesL_size]; omega
      have hb : 0 < (TopRoot.base (domOf c q) (leavesL c L q)).size := by
        simp only [TopRoot.base, Array.size_append]
        omega
      rw [show TopRoot.lev (domOf c q) (leavesL c L q) 0 = TopRoot.base (domOf c q) (leavesL c L q) from rfl,
        getElem!_pos _ 0 hb]
      simp only [TopRoot.base]
      rw [Array.getElem_append_left hc]
      simp [spineL, h0, leavesL, Registered.leafAs_eq]
  | succ k =>
    have hs : 0 < (TopRoot.lev (domOf c q) (leavesL c L q) k).size / 2 := by
      rw [lev_sizeL c L q (by omega)]
      have h1 : 1 < 2 ^ (depthL L q - k) := Nat.one_lt_two_pow (by omega)
      omega
    rw [TopRoot.lev, TopRoot.next_get _ _ _ hs]
    simp only [spineL, ite_eq_right (Nat.succ_ne_zero k), Nat.add_sub_cancel, Nat.mul_zero, Nat.zero_add]

/-- **A port's root is SHA-512 of its spine's frame at its tree's depth.** -/
theorem root_spineL (c : Flock.Circuit) (L : LeafMap) (q : ℕ) :
    Flock.FV3.root (domOf c q) (leavesL c L q) Flock.Sha512.hash = Flock.Sha512.hash (spineL c L q (depthL L q)) := by
  rw [TopRoot.root_eq, hash_spineL c L q le_rfl, leavesL_size]
  rfl

theorem rootsOfL_getD (c : Flock.Circuit) (L : LeafMap) {q : ℕ} (hq : q < c.ports.size) :
    (rootsOfL c L).getD q [] = bs (Flock.FV3.root (domOf c q) (leavesL c L q) Flock.Sha512.hash) := by
  simp [rootsOfL, List.getD_eq_getElem?_getD, hq]

/-- **Two ports with one root hash their spines' frames at their depths to one digest.** -/
theorem spine_of_roots {c : Flock.Circuit} {L : LeafMap} {q' q : ℕ} (hq' : q' < c.ports.size) (hq : q < c.ports.size)
    (hr : (rootsOfL c L).getD q' [] = (rootsOfL c L).getD q []) :
    Flock.Sha512.hash (spineL c L q' (depthL L q')) = Flock.Sha512.hash (spineL c L q (depthL L q)) := by
  rw [rootsOfL_getD c L hq', rootsOfL_getD c L hq] at hr
  rw [← root_spineL, ← root_spineL]
  exact bs_inj hr

/-- **Two equal node frames of two spines have one left child.** -/
theorem spine_child {c : Flock.Circuit} {L : LeafMap} {q' q k' k : ℕ} (hk' : 1 ≤ k') (hk'D : k' ≤ depthL L q')
    (hk : 1 ≤ k) (hkD : k ≤ depthL L q) (h : spineL c L q' k' = spineL c L q k) :
    Flock.Sha512.hash (spineL c L q' (k' - 1)) = Flock.Sha512.hash (spineL c L q (k - 1)) := by
  rw [hash_spineL c L q' (by omega), hash_spineL c L q (by omega)]
  have h2' : 1 < (TopRoot.lev (domOf c q') (leavesL c L q') (k' - 1)).size := by
    rw [lev_sizeL c L q' (by omega)]
    exact Nat.one_lt_two_pow (by omega)
  have h2 : 1 < (TopRoot.lev (domOf c q) (leavesL c L q) (k - 1)).size := by
    rw [lev_sizeL c L q (by omega)]
    exact Nat.one_lt_two_pow (by omega)
  simp only [spineL, ite_eq_right (show k' ≠ 0 by omega), ite_eq_right (show k ≠ 0 by omega), Registered.nodeIn,
    Registered.frame, List.foldl] at h
  obtain ⟨h1, -⟩ := PortRoots.step_inj' (by rw [lev_widthL c L q' _ h2', lev_widthL c L q _ h2]) h
  exact (PortRoots.step_inj' (by rw [lev_widthL c L q' _ (by omega), lev_widthL c L q _ (by omega)]) h1).2

/-- **A frame at level 0 of one spine equal to a frame of another has its domain.** -/
theorem spine_bottom0 {c : Flock.Circuit} {L : LeafMap} {q' q k : ℕ} (h : spineL c L q' 0 = spineL c L q k) :
    domOf c q' = domOf c q := by
  unfold spineL at h
  rw [ite_eq_left rfl] at h
  by_cases hk : k = 0
  · rw [ite_eq_left hk] at h
    split_ifs at h with hc' hc
    · exact PortRoots.frame_head_inj h (.cons rfl .nil)
    · exact absurd h (pad_ne_leaf _ _)
    · exact absurd h.symm (pad_ne_leaf _ _)
    · have hs : schemaOf c q' = schemaOf c q := by
        have h' := h
        simp only [Registered.leafInAs, Registered.frame, List.foldl] at h'
        exact PortRoots.step_schema (PortRoots.schemaOf_mem c q') (PortRoots.schemaOf_mem c q)
          (PortRoots.step_inj' (by rw [treeLeaf_size, treeLeaf_size]) h').1
      rw [hs] at h
      exact PortRoots.frame_head_inj h
        (.cons rfl (.cons rfl (.cons rfl (.cons (by rw [treeLeaf_size, treeLeaf_size]) .nil))))
  · rw [ite_eq_right hk] at h
    split_ifs at h
    · exact absurd h (pad_ne_node _ _)
    · exact absurd h (leaf_ne_node _ _)

/-- **Two equal frames of two spines, one of them at level 0, have one domain.** -/
theorem spine_bottom {c : Flock.Circuit} {L : LeafMap} {q' q k' k : ℕ} (h0 : k' = 0 ∨ k = 0)
    (h : spineL c L q' k' = spineL c L q k) : domOf c q' = domOf c q := by
  rcases h0 with rfl | rfl
  · exact spine_bottom0 h
  · exact (spine_bottom0 h.symm).symm

/-- **Two spines from one digest under different domains collide at some level**: if at no level `i`, from the roots
down to the shallower tree's leaves, the spines hash two different frames to one digest, they have one domain. -/
theorem dom_of_spine {c : Flock.Circuit} {L : LeafMap} {q' q : ℕ}
    (hn : ∀ i, i ≤ min (depthL L q') (depthL L q) →
      ¬ (spineL c L q' (depthL L q' - i) ≠ spineL c L q (depthL L q - i) ∧
        Flock.Sha512.hash (spineL c L q' (depthL L q' - i)) = Flock.Sha512.hash (spineL c L q (depthL L q - i))))
    (h0 : Flock.Sha512.hash (spineL c L q' (depthL L q')) = Flock.Sha512.hash (spineL c L q (depthL L q))) :
    domOf c q' = domOf c q := by
  have hall : ∀ i, i ≤ min (depthL L q') (depthL L q) →
      spineL c L q' (depthL L q' - i) = spineL c L q (depthL L q - i) := by
    intro i
    induction i with
    | zero =>
      intro hi
      by_contra hne
      exact hn 0 hi ⟨hne, by simpa using h0⟩
    | succ i ih =>
      intro hi
      have hm := Nat.le_min.mp hi
      have hh := spine_child (k' := depthL L q' - i) (k := depthL L q - i) (by omega) (by omega) (by omega)
        (by omega) (ih (by omega))
      rw [show depthL L q' - i - 1 = depthL L q' - (i + 1) by omega,
        show depthL L q - i - 1 = depthL L q - (i + 1) by omega] at hh
      by_contra hne
      exact hn (i + 1) hi ⟨hne, hh⟩
  refine spine_bottom ?_ (hall _ le_rfl)
  rcases le_total (depthL L q') (depthL L q) with hl | hl
  · left
    rw [Nat.min_eq_left hl, Nat.sub_self]
  · right
    rw [Nat.min_eq_right hl, Nat.sub_self]

/-! ## The finder, and the term in `ecr/sha-512` -/

/-- **The first level, from the roots down, at which ports `q'` and `q`'s spines hash two different frames to one
digest.** -/
def spineColl (c : Flock.Circuit) (L : LeafMap) (q' q : ℕ) : Option ℕ :=
  (List.range (min (depthL L q') (depthL L q) + 1)).find? fun i =>
    decide (spineL c L q' (depthL L q' - i) ≠ spineL c L q (depthL L q - i) ∧
      Flock.Sha512.hash (spineL c L q' (depthL L q' - i)) = Flock.Sha512.hash (spineL c L q (depthL L q - i)))

theorem spineColl_ne_none {c : Flock.Circuit} {L : LeafMap} {q' q : ℕ} (hd : domOf c q' ≠ domOf c q)
    (h0 : Flock.Sha512.hash (spineL c L q' (depthL L q')) = Flock.Sha512.hash (spineL c L q (depthL L q))) :
    spineColl c L q' q ≠ none := by
  intro hn
  refine hd (dom_of_spine (fun i hi => ?_) h0)
  have h := List.find?_eq_none.mp hn i (List.mem_range.mpr (by omega))
  simpa only [decide_eq_true_eq] using h

/-- The first pair of ports, earlier first, with one registered root at `L` and different domains. -/
def collPairL (c : Flock.Circuit) (L : LeafMap) : Option (ℕ × ℕ) :=
  (PortRoots.pairs c.ports.size).find? fun p =>
    decide ((rootsOfL c L).getD p.1 [] = (rootsOfL c L).getD p.2 [] ∧ domOf c p.1 ≠ domOf c p.2)

/-- **The collision a registration's roots at a leaf map expose**: at `collPairL`'s two ports, their spines' frames
at `spineColl`'s level. -/
def rootsCollL (c : Flock.Circuit) (L : LeafMap) : Option (Msg × Msg) :=
  (collPairL c L).bind fun p => (spineColl c L p.1 p.2).map fun i =>
    (bs (spineL c L p.1 (depthL L p.1 - i)), bs (spineL c L p.2 (depthL L p.2 - i)))

/-- **SHA-512 calls to recompute every port's root at `L`**: per port `q`, `L.cnt q` hm96 tree leaves, as many
frame-v3 leaves, and at most `2 · nextPow2 (L.cnt q)` pads and nodes. The spines' frames are among them. -/
def rootsCostL (c : Flock.Circuit) (L : LeafMap) : ℕ :=
  ∑ q ∈ Finset.range c.ports.size, (2 * L.cnt q + 2 * Flock.nextPow2 (L.cnt q))

/-- **The roots finder at a leaf map**: no coins; it recomputes every port's registered root and outputs
`rootsCollL`. -/
def rootsFinderL (c : Flock.Circuit) (L : LeafMap) : Claims.Finder where
  S := Unit
  out _ := rootsCollL c L
  cost _ := rootsCostL c L

theorem rootsCollL_collides {c : Flock.Circuit} {L : LeafMap} {x y : Msg} (h : rootsCollL c L = some (x, y)) :
    Claims.Collision sha512Msg x y := by
  obtain ⟨⟨q', q⟩, -, hp⟩ := Option.bind_eq_some_iff.mp h
  obtain ⟨i, hi, he⟩ := Option.map_eq_some_iff.mp hp
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
  have hpr := List.find?_some hi
  simp only [decide_eq_true_eq] at hpr
  refine ⟨fun e => hpr.1 (bs_inj e), ?_⟩
  simp only [sha512Msg, bytes_bs, hpr.2]

theorem fixesL_of_none {c : Flock.Circuit} {L : LeafMap} (h : rootsCollL c L = none) : RootFixesDomainL c L := by
  intro q q' hlt hq he
  by_contra hd
  cases hc : collPairL c L with
  | none =>
    have hp := List.find?_eq_none.mp hc (q', q) (PortRoots.mem_pairs.mpr ⟨hlt, hq⟩)
    simp only [decide_eq_true_eq, not_and, not_not] at hp
    exact hd (hp he)
  | some p =>
    have hp := List.find?_some hc
    have hmem := PortRoots.mem_pairs.mp (List.mem_of_find?_eq_some hc)
    simp only [decide_eq_true_eq] at hp
    simp only [rootsCollL, hc, Option.bind_some, Option.map_eq_none_iff] at h
    exact spineColl_ne_none hp.2 (spine_of_roots (hmem.1.trans hmem.2) hmem.2 hp.1) h

theorem ex_costL (c : Flock.Circuit) (L : LeafMap) :
    Claims.Ex (fun s => ((rootsFinderL c L).cost s : ℝ)) = rootsCostL c L := by
  have hc : (Fintype.card (rootsFinderL c L).S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  unfold Claims.Ex
  change (∑ _s : (rootsFinderL c L).S, (rootsCostL c L : ℝ)) / _ = _
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm, mul_div_assoc, div_self hc, mul_one]

/-- **A registration whose roots at `L` expose a collision has `ecr/sha-512`'s term at least 1** at the roots
finder. -/
theorem one_le_ecrBoundL {c : Flock.Circuit} {L : LeafMap} {u v : Msg} (hc : rootsCollL c L = some (u, v)) :
    1 ≤ Claims.ecrBound sha512Msg (rootsFinderL c L) (rootsCostL c L) := by
  have hPr : (rootsFinderL c L).Pr sha512Msg = 1 :=
    PortRoots.finderPr_one fun _ => ⟨u, v, hc, rootsCollL_collides hc⟩
  unfold Claims.ecrBound
  split_ifs with h
  · rw [ENNReal.one_le_ofReal]
    have h1 := h.1
    unfold Claims.ExpectedCR at h1
    rw [hPr, ex_costL] at h1
    exact h1
  · exact le_top

/-- **The roots guard at a leaf map costs one `ecr/sha-512` term**: a probability `p ≤ 1` that is at most `x` at every
registration whose roots at `L` fix their domains (`RootFixesDomainL`) is at most `x` plus `ecr/sha-512`'s term at the
roots finder, at every registration. -/
theorem le_add_ecrBoundL {c : Flock.Circuit} {L : LeafMap} {p x : ℝ≥0∞} (hp : p ≤ 1)
    (h : RootFixesDomainL c L → p ≤ x) :
    p ≤ x + Claims.ecrBound sha512Msg (rootsFinderL c L) (rootsCostL c L) := by
  cases hc : rootsCollL c L with
  | none => exact (h (fixesL_of_none hc)).trans le_self_add
  | some z => exact hp.trans ((one_le_ecrBoundL (u := z.1) (v := z.2) (by rw [hc])).trans le_add_self)

end Flock.Bridge
