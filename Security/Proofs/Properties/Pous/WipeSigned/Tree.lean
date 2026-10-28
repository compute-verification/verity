import Properties.Pous.WipeSigned
import Proofs.Properties.Pous.MemoryChallenger.Tree

/-!
# The wipe's records tree binds its values

`Tree`'s argument at the wipe's record schema: the tree's levels are the same functions of its leaves (`Tree.lev`), and
only the leaves' schema differs, so under no collision among the strings two trees hash (`wipeTreeStrings`) one root
gives one level below it, down to the leaves, and one leaf one value (`tree_bind`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.WipeTree

open MemoryChallenger Flock Flock.FV3.Sha512
open Tree (CF lev levels_eq levels_size levels_get root_eq lev_size lev_width get!_mem hash_size frame_last frame_two
  node_hash leaf_hash next_get isEmpty_of_size)

/-- The leaves the wipe's records tree is built over. -/
def leavesOf (d : Domain) (vs : Array ByteArray) : Array ByteArray :=
  (Array.range vs.size).map fun r => d.leaf r r StreamedWipe.RECORD_SCHEMA vs[r]!

theorem leavesOf_size (d : Domain) (vs : Array ByteArray) : (leavesOf d vs).size = vs.size := by simp [leavesOf]

theorem leavesOf_width (d : Domain) (vs : Array ByteArray) : ∀ x ∈ leavesOf d vs, x.size = 64 := by
  intro x hx
  simp only [leavesOf, Array.mem_map] at hx
  obtain ⟨r, -, rfl⟩ := hx
  exact hash_size _

theorem treeStrings_eq (d : Domain) (vs : Array ByteArray) : wipeTreeStrings d vs =
    (List.range vs.size).map (fun r =>
      FV3.input "leaf" [d.id, FV3.uint r, FV3.uint r, StreamedWipe.RECORD_SCHEMA.toUTF8, vs[r]!]) ++
    (List.range ((d.levels (leavesOf d vs)).size - 1)).flatMap fun l =>
      (List.range ((d.levels (leavesOf d vs))[l]!.size / 2)).map fun i =>
        FV3.input "node" [d.id, FV3.uint l, FV3.uint i, ((d.levels (leavesOf d vs))[l]!)[2 * i]!,
          ((d.levels (leavesOf d vs))[l]!)[2 * i + 1]!] := rfl

theorem leavesOf_isEmpty {d : Domain} {vs : Array ByteArray} (hn : vs.isEmpty = false) :
    (leavesOf d vs).isEmpty = false := by
  cases h : (leavesOf d vs).isEmpty
  · rfl
  · rw [Array.isEmpty_iff_size_eq_zero, leavesOf_size] at h
    rw [Array.isEmpty_iff_size_eq_zero.mpr h] at hn
    cases hn

theorem leaf_mem {d : Domain} {vs : Array ByteArray} {r : Nat} (hr : r < vs.size) :
    FV3.input "leaf" [d.id, FV3.uint r, FV3.uint r, StreamedWipe.RECORD_SCHEMA.toUTF8, vs[r]!] ∈
      wipeTreeStrings d vs := by
  simp only [treeStrings_eq, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl ⟨r, hr, rfl⟩

theorem node_mem {d : Domain} {vs : Array ByteArray} (hn : vs.isEmpty = false) {l i : Nat}
    (hl : l < log2ceil vs.size) (hi : i < (lev d (leavesOf d vs) l).size / 2) :
    FV3.input "node" [d.id, FV3.uint l, FV3.uint i, (lev d (leavesOf d vs) l)[2 * i]!,
      (lev d (leavesOf d vs) l)[2 * i + 1]!] ∈ wipeTreeStrings d vs := by
  have hn' := leavesOf_isEmpty (d := d) hn
  have hs := leavesOf_size d vs
  simp only [treeStrings_eq, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  refine Or.inr ⟨l, ?_, i, ?_, ?_⟩
  · rw [levels_size _ _ hn', hs]; omega
  · rw [levels_get _ _ hn' (by rw [hs]; omega)]; exact hi
  · rw [levels_get _ _ hn' (by rw [hs]; omega)]

/-- Under no SHA-512 collision among the strings both trees hash, trees over equally many values with one root agree
level by level, from the root down. -/
theorem lev_bind {d : Domain} {vs vs' : Array ByteArray} {S : List ByteArray} (hn : vs.size = vs'.size)
    (h0 : vs.size ≠ 0) (hS : ∀ x ∈ wipeTreeStrings d vs, x ∈ S) (hS' : ∀ x ∈ wipeTreeStrings d vs', x ∈ S)
    (cf : CF S) (hr : d.root (leavesOf d vs) = d.root (leavesOf d vs')) :
    ∀ k ≤ log2ceil vs.size,
      lev d (leavesOf d vs) (log2ceil vs.size - k) = lev d (leavesOf d vs') (log2ceil vs.size - k) := by
  have hL := leavesOf_size d vs
  have hL' := leavesOf_size d vs'
  have hne := isEmpty_of_size h0
  have hne' := isEmpty_of_size (hn ▸ h0)
  have size_l : ∀ l ≤ log2ceil vs.size, (lev d (leavesOf d vs) l).size = 2 ^ (log2ceil vs.size - l) := fun l hl => by
    rw [lev_size d _ l (by rw [hL]; exact hl), hL]
  have size_l' : ∀ l ≤ log2ceil vs.size, (lev d (leavesOf d vs') l).size = 2 ^ (log2ceil vs.size - l) := fun l hl => by
    rw [lev_size d _ l (by rw [hL', ← hn]; exact hl), hL', ← hn]
  intro k
  induction k with
  | zero =>
    intro _
    rw [root_eq _ _ (leavesOf_isEmpty hne), root_eq _ _ (leavesOf_isEmpty hne'), hL, hL', ← hn] at hr
    have s1 := size_l (log2ceil vs.size) (Nat.le_refl _)
    have s2 := size_l' (log2ceil vs.size) (Nat.le_refl _)
    simp only [Nat.sub_self, Nat.pow_zero] at s1 s2
    simp only [Nat.sub_zero]
    apply Array.ext (by rw [s1, s2])
    intro i hi _
    have : i = 0 := by omega
    subst this
    rwa [getElem!_pos _ 0 (by omega), getElem!_pos _ 0 (by omega)] at hr
  | succ k ih =>
    intro hk
    have ih := ih (by omega)
    rw [show log2ceil vs.size - k = (log2ceil vs.size - (k + 1)) + 1 by omega] at ih
    generalize hl : log2ceil vs.size - (k + 1) = l at ih ⊢
    have hlD : l < log2ceil vs.size := by omega
    have s1 := size_l l (by omega)
    have s2 := size_l' l (by omega)
    rw [show log2ceil vs.size - l = k + 1 by omega, Nat.pow_succ] at s1 s2
    apply Array.ext (by rw [s1, s2])
    intro j hj hj'
    have hi : j / 2 < (lev d (leavesOf d vs) l).size / 2 := by rw [s1]; omega
    have hi' : j / 2 < (lev d (leavesOf d vs') l).size / 2 := by rw [s2]; omega
    have e := congrArg (·[j / 2]!) ih
    simp only [lev] at e
    rw [next_get _ _ _ hi, next_get _ _ _ hi', node_hash, node_hash] at e
    have e := cf _ (hS _ (node_mem hne hlD hi)) _ (hS' _ (node_mem hne' (by rw [← hn]; exact hlD) hi')) e
    have w := lev_width d _ (leavesOf_width d vs) l
    have w' := lev_width d _ (leavesOf_width d vs') l
    have ha : ((lev d (leavesOf d vs) l)[2 * (j / 2)]!).size = ((lev d (leavesOf d vs') l)[2 * (j / 2)]!).size := by
      rw [w _ (get!_mem (by omega)), w' _ (get!_mem (by omega))]
    obtain ⟨ea, eb⟩ := frame_two (ps := [d.id, FV3.uint l, FV3.uint (j / 2)]) ha e
    rw [← getElem!_pos _ j hj, ← getElem!_pos _ j hj']
    rcases Nat.even_or_odd' j with ⟨i, rfl | rfl⟩
    · rw [show 2 * i / 2 = i by omega] at ea; exact ea
    · rw [show (2 * i + 1) / 2 = i by omega] at eb; exact eb

/-- **The wipe's records tree binds its values**: under no SHA-512 collision among the strings both trees hash, two
lists of equally many values with one root are one. -/
theorem tree_bind {d : Domain} {vs vs' : Array ByteArray} {S : List ByteArray} (hn : vs.size = vs'.size)
    (hS : ∀ x ∈ wipeTreeStrings d vs, x ∈ S) (hS' : ∀ x ∈ wipeTreeStrings d vs', x ∈ S) (cf : CF S)
    (hr : d.root (leavesOf d vs) = d.root (leavesOf d vs')) : vs = vs' := by
  by_cases h0 : vs.size = 0
  · obtain rfl := Array.size_eq_zero_iff.mp h0
    obtain rfl := Array.size_eq_zero_iff.mp (show vs'.size = 0 by rw [← hn]; exact h0)
    rfl
  have e := lev_bind hn h0 hS hS' cf hr (log2ceil vs.size) (Nat.le_refl _)
  simp only [Nat.sub_self, lev, Tree.base] at e
  have eL := (Array.append_inj e (by rw [leavesOf_size, leavesOf_size, hn])).1
  apply Array.ext hn
  intro r hr hr'
  have e := congrArg (·[r]!) eL
  rw [getElem!_pos _ r (by rw [leavesOf_size]; exact hr), getElem!_pos _ r (by rw [leavesOf_size]; exact hr')] at e
  simp only [leavesOf, Array.getElem_map, Array.getElem_range] at e
  rw [leaf_hash, leaf_hash] at e
  have e := frame_last (ps := [d.id, FV3.uint r, FV3.uint r, StreamedWipe.RECORD_SCHEMA.toUTF8])
    (cf _ (hS _ (leaf_mem hr)) _ (hS' _ (leaf_mem hr')) e)
  rwa [getElem!_pos _ r hr, getElem!_pos _ r hr'] at e

theorem root_width (d : Domain) (vs : Array ByteArray) : (d.root (leavesOf d vs)).size = 64 := by
  by_cases h0 : vs.size = 0
  · have : leavesOf d vs = #[] := Array.size_eq_zero_iff.mp (by rw [leavesOf_size]; exact h0)
    rw [this]
    exact hash_size _
  have hne := leavesOf_isEmpty (d := d) (isEmpty_of_size h0)
  rw [root_eq _ _ hne]
  refine lev_width d _ (leavesOf_width d vs) _ _ (get!_mem ?_)
  rw [lev_size _ _ _ (Nat.le_refl _), Nat.sub_self]
  decide

end MemoryChallenger.Properties.SecurityProofs.WipeTree
