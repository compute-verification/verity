import Properties.Pous.MemoryChallenger
import Proofs.Properties.NetworkCertifier.Bytes

/-!
# The records tree binds its values

`Flock.FV3.Domain.levels` is the leaves padded to `nextPow2` of their count (`base`), then `log2ceil` levels
each of the nodes over the pairs below (`next`, `lev`, `levels_eq`). Every node and leaf is SHA-512 of a frame whose
parts are length-prefixed, so under no collision among the strings two trees hash (`treeStrings`, `CF`) one root gives
one level below it, down to the leaves, and one leaf one value (`tree_bind`).
-/

namespace MemoryChallenger.Properties.SecurityProofs.Tree

open MemoryChallenger Flock Flock.FV3.Sha512
open Properties.NetworkCertifier.Bytes (bs bs_append bs_length bs_inj split_bytes be64_size)

/-- No two distinct strings of `S` share a SHA-512 digest. -/
def CF (S : List ByteArray) : Prop := ∀ x ∈ S, ∀ y ∈ S, Sha512.hash x = Sha512.hash y → x = y

/-- One level up: the nodes over each pair. -/
def next (d : Domain) (lvl : Nat) (level : Array ByteArray) : Array ByteArray :=
  (Array.range (level.size / 2)).map fun i => d.node lvl i level[2 * i]! level[2 * i + 1]!

/-- The leaves padded to `nextPow2` of their count. -/
def base (d : Domain) (leaves : Array ByteArray) : Array ByteArray :=
  leaves ++ (Array.range (nextPow2 leaves.size - leaves.size)).map fun k => d.pad (leaves.size + k)

/-- Level `l` of the tree. -/
def lev (d : Domain) (leaves : Array ByteArray) : Nat → Array ByteArray
  | 0 => base d leaves
  | l + 1 => next d l (lev d leaves l)

theorem fold_eq (d : Domain) (leaves : Array ByteArray) : ∀ (k s : Nat),
    (List.range' s k).foldl (fun st lvl => (next d lvl st.1, st.2.push (next d lvl st.1)))
        (lev d leaves s, ((List.range (s + 1)).map (lev d leaves)).toArray) =
      (lev d leaves (s + k), ((List.range (s + k + 1)).map (lev d leaves)).toArray)
  | 0, s => by simp
  | k + 1, s => by
    rw [List.range'_succ, List.foldl_cons]
    have : (next d s (lev d leaves s), ((List.range (s + 1)).map (lev d leaves)).toArray.push (next d s (lev d leaves s))) =
        (lev d leaves (s + 1), ((List.range (s + 1 + 1)).map (lev d leaves)).toArray) := by
      rw [List.range_succ (n := s + 1), List.map_append]
      simp [lev]
    rw [this, fold_eq d leaves k (s + 1), show s + 1 + k = s + (k + 1) by omega]

theorem levels_eq (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    d.levels leaves = ((List.range (log2ceil leaves.size + 1)).map (lev d leaves)).toArray := by
  unfold FV3.Domain.levels
  simp only [Id.run, h, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  have hs : [:log2ceil leaves.size].size = log2ceil leaves.size := by simp [Std.Legacy.Range.size]
  rw [hs]
  have := congrArg Prod.snd (fold_eq d leaves (log2ceil leaves.size) 0)
  simp only [Nat.zero_add] at this
  exact this

theorem le_pow_log2ceil (n : Nat) : n ≤ 2 ^ log2ceil n := by
  unfold log2ceil
  split
  · simp; omega
  · have := @Nat.lt_log2_self (n - 1)
    omega

theorem base_size (d : Domain) (leaves : Array ByteArray) : (base d leaves).size = 2 ^ log2ceil leaves.size := by
  have := le_pow_log2ceil leaves.size
  simp [base, nextPow2]
  omega

theorem next_size (d : Domain) (l : Nat) (a : Array ByteArray) : (next d l a).size = a.size / 2 := by
  simp [next]

theorem lev_size (d : Domain) (leaves : Array ByteArray) :
    ∀ l ≤ log2ceil leaves.size, (lev d leaves l).size = 2 ^ (log2ceil leaves.size - l)
  | 0, _ => by simp [lev, base_size]
  | l + 1, h => by
    rw [lev, next_size, lev_size d leaves l (by omega),
      show log2ceil leaves.size - l = (log2ceil leaves.size - (l + 1)) + 1 by omega, Nat.pow_succ]
    simp

theorem get!_mem {a : Array ByteArray} {i : Nat} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]
  exact Array.getElem_mem h

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := Properties.NetworkCertifier.Bytes.hash_size m

theorem lev_width (d : Domain) (leaves : Array ByteArray) (h : ∀ x ∈ leaves, x.size = 64) :
    ∀ l, ∀ x ∈ lev d leaves l, x.size = 64
  | 0, x, hx => by
    simp only [lev, base, Array.mem_append, Array.mem_map] at hx
    rcases hx with hx | ⟨k, -, rfl⟩
    · exact h x hx
    · exact hash_size _
  | l + 1, x, hx => by
    simp only [lev, next, Array.mem_map] at hx
    obtain ⟨i, -, rfl⟩ := hx
    exact hash_size _

theorem levels_size (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    (d.levels leaves).size = log2ceil leaves.size + 1 := by
  simp [levels_eq d leaves h]

theorem levels_get (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) {l : Nat}
    (hl : l ≤ log2ceil leaves.size) : (d.levels leaves)[l]! = lev d leaves l := by
  rw [getElem!_pos _ l (by rw [levels_size d leaves h]; omega)]
  simp [levels_eq d leaves h]

theorem root_eq (d : Domain) (leaves : Array ByteArray) (h : leaves.isEmpty = false) :
    d.root leaves = (lev d leaves (log2ceil leaves.size))[0]! := by
  unfold FV3.Domain.root
  rw [levels_eq d leaves h, List.back?_toArray, List.getLast?_map, List.getLast?_range]
  simp

theorem frame_snoc (tag : String) (ps : List ByteArray) (v : ByteArray) :
    bs (FV3.input tag (ps ++ [v])) = bs (FV3.input tag ps) ++ (bs (be64 v.size.toUInt64) ++ bs v) := by
  simp [FV3.input, List.foldl_append, bs_append]

/-- A frame's last part, after parts that agree, is determined. -/
theorem frame_last {tag : String} {ps : List ByteArray} {v v' : ByteArray}
    (h : FV3.input tag (ps ++ [v]) = FV3.input tag (ps ++ [v'])) : v = v' := by
  have e := congrArg bs h
  rw [frame_snoc, frame_snoc] at e
  exact bs_inj (split_bytes (List.append_cancel_left e) (by rw [be64_size, be64_size])).2

/-- A frame's two last parts, after parts that agree, are determined when the first of them has one width. -/
theorem frame_two {tag : String} {ps : List ByteArray} {a b a' b' : ByteArray} (ha : a.size = a'.size)
    (h : FV3.input tag (ps ++ [a, b]) = FV3.input tag (ps ++ [a', b'])) : a = a' ∧ b = b' := by
  have e := congrArg bs h
  rw [show ps ++ [a, b] = (ps ++ [a]) ++ [b] by simp, show ps ++ [a', b'] = (ps ++ [a']) ++ [b'] by simp,
    frame_snoc, frame_snoc, frame_snoc, frame_snoc] at e
  simp only [List.append_assoc] at e
  have e := List.append_cancel_left e
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  obtain ⟨ha', e⟩ := split_bytes e ha
  obtain ⟨-, e⟩ := split_bytes e (by rw [be64_size, be64_size])
  exact ⟨ha', bs_inj e⟩

theorem node_hash (d : Domain) (l i : Nat) (a b : ByteArray) :
    d.node l i a b = Sha512.hash (FV3.input "node" ([d.id, FV3.uint l, FV3.uint i] ++ [a, b])) := rfl

theorem leaf_hash (d : Domain) (r : Nat) (schema : String) (v : ByteArray) :
    d.leaf r r schema v = Sha512.hash (FV3.input "leaf" ([d.id, FV3.uint r, FV3.uint r, schema.toUTF8] ++ [v])) := rfl

/-- The leaves the records tree is built over. -/
def leavesOf (d : Domain) (vs : Array ByteArray) : Array ByteArray :=
  (Array.range vs.size).map fun r => d.leaf r r RECORD_SCHEMA vs[r]!

theorem leavesOf_size (d : Domain) (vs : Array ByteArray) : (leavesOf d vs).size = vs.size := by simp [leavesOf]

theorem leavesOf_width (d : Domain) (vs : Array ByteArray) : ∀ x ∈ leavesOf d vs, x.size = 64 := by
  intro x hx
  simp only [leavesOf, Array.mem_map] at hx
  obtain ⟨r, -, rfl⟩ := hx
  exact hash_size _

theorem treeStrings_eq (d : Domain) (vs : Array ByteArray) : treeStrings d vs =
    (List.range vs.size).map (fun r =>
      FV3.input "leaf" [d.id, FV3.uint r, FV3.uint r, RECORD_SCHEMA.toUTF8, vs[r]!]) ++
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
    FV3.input "leaf" [d.id, FV3.uint r, FV3.uint r, RECORD_SCHEMA.toUTF8, vs[r]!] ∈ treeStrings d vs := by
  simp only [treeStrings_eq, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl ⟨r, hr, rfl⟩

theorem node_mem {d : Domain} {vs : Array ByteArray} (hn : vs.isEmpty = false) {l i : Nat}
    (hl : l < log2ceil vs.size) (hi : i < (lev d (leavesOf d vs) l).size / 2) :
    FV3.input "node" [d.id, FV3.uint l, FV3.uint i, (lev d (leavesOf d vs) l)[2 * i]!,
      (lev d (leavesOf d vs) l)[2 * i + 1]!] ∈ treeStrings d vs := by
  have hn' := leavesOf_isEmpty (d := d) hn
  have hs := leavesOf_size d vs
  simp only [treeStrings_eq, List.mem_append, List.mem_flatMap, List.mem_map, List.mem_range]
  refine Or.inr ⟨l, ?_, i, ?_, ?_⟩
  · rw [levels_size _ _ hn', hs]; omega
  · rw [levels_get _ _ hn' (by rw [hs]; omega)]; exact hi
  · rw [levels_get _ _ hn' (by rw [hs]; omega)]

theorem next_get (d : Domain) (l : Nat) (a : Array ByteArray) {i : Nat} (hi : i < a.size / 2) :
    (next d l a)[i]! = d.node l i a[2 * i]! a[2 * i + 1]! := by
  rw [getElem!_pos _ i (by rw [next_size]; exact hi)]
  simp [next]

theorem isEmpty_of_size {vs : Array ByteArray} (h : vs.size ≠ 0) : vs.isEmpty = false := by
  cases he : vs.isEmpty
  · rfl
  · exact absurd (Array.isEmpty_iff_size_eq_zero.mp he) h

/-- Under no SHA-512 collision among the strings both trees hash, trees over equally many values with one root agree
level by level, from the root down. -/
theorem lev_bind {d : Domain} {vs vs' : Array ByteArray} {S : List ByteArray} (hn : vs.size = vs'.size)
    (h0 : vs.size ≠ 0) (hS : ∀ x ∈ treeStrings d vs, x ∈ S) (hS' : ∀ x ∈ treeStrings d vs', x ∈ S) (cf : CF S)
    (hr : d.root (leavesOf d vs) = d.root (leavesOf d vs')) :
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

/-- **The records tree binds its values**: under no SHA-512 collision among the strings both trees hash, two lists of
equally many values with one root are one. -/
theorem tree_bind {d : Domain} {vs vs' : Array ByteArray} {S : List ByteArray} (hn : vs.size = vs'.size)
    (hS : ∀ x ∈ treeStrings d vs, x ∈ S) (hS' : ∀ x ∈ treeStrings d vs', x ∈ S) (cf : CF S)
    (hr : d.root (leavesOf d vs) = d.root (leavesOf d vs')) : vs = vs' := by
  by_cases h0 : vs.size = 0
  · obtain rfl := Array.size_eq_zero_iff.mp h0
    obtain rfl := Array.size_eq_zero_iff.mp (show vs'.size = 0 by rw [← hn]; exact h0)
    rfl
  have e := lev_bind hn h0 hS hS' cf hr (log2ceil vs.size) (Nat.le_refl _)
  simp only [Nat.sub_self, lev, base] at e
  have eL := (Array.append_inj e (by rw [leavesOf_size, leavesOf_size, hn])).1
  apply Array.ext hn
  intro r hr hr'
  have e := congrArg (·[r]!) eL
  rw [getElem!_pos _ r (by rw [leavesOf_size]; exact hr), getElem!_pos _ r (by rw [leavesOf_size]; exact hr')] at e
  simp only [leavesOf, Array.getElem_map, Array.getElem_range] at e
  rw [leaf_hash, leaf_hash] at e
  have e := frame_last (ps := [d.id, FV3.uint r, FV3.uint r, RECORD_SCHEMA.toUTF8])
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

end MemoryChallenger.Properties.SecurityProofs.Tree
