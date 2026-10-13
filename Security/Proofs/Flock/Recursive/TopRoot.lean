import Proofs.Flock.Verifier.Registered
import Proofs.Flock.Soundness.Refine.Paths

/-!
# The registered top root binds its commit strings

`RecVStar.topEntry` commits the level-zero top as a frame-v3-sha512 tree whose
leaves are registered-row frames around hm96 tree leaves.  This file follows
that computation level by level.  If two unequal arrays of commit strings give
one root, one of the hm96 leaf inputs, row frames, or node frames hashed while
recomputing those roots is an explicit SHA-512 collision.
-/

namespace Flock.SecurityProofs

open FlockSoundness

/-- The leaves in `RecVStar.topEntry`'s registered `stacked` root. -/
def topLeaves (dom : ByteArray) (bcs : Array ByteArray) : Array ByteArray :=
  bcs.mapIdx fun i bc => Flock.Registered.rowLeaf dom i (Flock.Hm96.Default512.treeLeaf bc)

namespace TopRoot

open Flock

/-- No two distinct members of `S` have one SHA-512 digest. -/
def CF (S : List ByteArray) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, Sha512.hash x = Sha512.hash y → x = y

/-- One frame-v3 level above `level`. -/
def next (dom : ByteArray) (depth : Nat) (level : Array ByteArray) : Array ByteArray :=
  (Array.range (level.size / 2)).map fun i =>
    Sha512.hash (Registered.nodeIn dom depth i level[2 * i]! level[2 * i + 1]!)

/-- The supplied leaves, padded as `FV3.root` pads them. -/
def base (dom : ByteArray) (leaves : Array ByteArray) : Array ByteArray :=
  leaves ++ (Array.range (nextPow2 leaves.size - leaves.size)).map fun k =>
    FV3.hash "pad" [dom, FV3.uint (leaves.size + k)] Sha512.hash

/-- Level `l` in the computation of `FV3.root`. -/
def lev (dom : ByteArray) (leaves : Array ByteArray) : Nat → Array ByteArray
  | 0 => base dom leaves
  | l + 1 => next dom l (lev dom leaves l)

theorem fold_eq (dom : ByteArray) (leaves : Array ByteArray) : ∀ (k s : Nat),
    (List.range' s k).foldl (fun level depth => next dom depth level) (lev dom leaves s) =
      lev dom leaves (s + k)
  | 0, s => by simp
  | k + 1, s => by
    rw [List.range'_succ, List.foldl_cons, show next dom s (lev dom leaves s) = lev dom leaves (s + 1) by
      rw [show s + 1 = Nat.succ s by omega, lev]]
    rw [fold_eq dom leaves k (s + 1), show s + 1 + k = s + (k + 1) by omega]

theorem le_nextPow2 (n : Nat) : n ≤ nextPow2 n := by
  unfold nextPow2 log2ceil
  split
  · rw [Nat.pow_zero]
    omega
  · have := Nat.lt_log2_self (n := n - 1)
    omega

theorem log2ceil_nextPow2 (n : Nat) : log2ceil (nextPow2 n) = log2ceil n := by
  unfold nextPow2 log2ceil
  split
  · simp
  · rename_i hn
    simp only [Nat.pow_succ, Nat.reducePow, Nat.mul_le_one, Nat.reduceLeDiff, false_and, ↓reduceIte]
    rw [show 2 * 2 ^ Nat.log2 (n - 1) - 1 = 2 ^ (Nat.log2 (n - 1) + 1) - 1 by
      rw [Nat.pow_succ, Nat.mul_comm]]
    rw [Nat.log2_pred_pow]

theorem base_size (dom : ByteArray) (leaves : Array ByteArray) :
    (base dom leaves).size = nextPow2 leaves.size := by
  have h := le_nextPow2 leaves.size
  simp [base]
  omega

theorem next_size (dom : ByteArray) (l : Nat) (a : Array ByteArray) :
    (next dom l a).size = a.size / 2 := by
  simp [next]

theorem lev_size (dom : ByteArray) (leaves : Array ByteArray) :
    ∀ l ≤ log2ceil leaves.size, (lev dom leaves l).size = 2 ^ (log2ceil leaves.size - l)
  | 0, _ => by simp [lev, base_size, nextPow2]
  | l + 1, h => by
    rw [lev, next_size, lev_size dom leaves l (by omega),
      show log2ceil leaves.size - l = (log2ceil leaves.size - (l + 1)) + 1 by omega, Nat.pow_succ]
    simp

theorem root_eq (dom : ByteArray) (leaves : Array ByteArray) :
    FV3.root dom leaves Sha512.hash = (lev dom leaves (log2ceil leaves.size))[0]! := by
  unfold FV3.root
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  rw [log2ceil_nextPow2]
  change
    ((List.range (log2ceil leaves.size)).foldl
      (fun level depth => next dom depth level) (base dom leaves)).getD 0 ByteArray.empty =
      (lev dom leaves (log2ceil leaves.size))[0]!
  rw [show List.range (log2ceil leaves.size) = List.range' 0 (log2ceil leaves.size) by simp,
    show base dom leaves = lev dom leaves 0 by rfl, fold_eq]
  rfl

theorem get!_mem {a : Array ByteArray} {i : Nat} (h : i < a.size) : a[i]! ∈ a := by
  rw [getElem!_pos a i h]
  exact Array.getElem_mem h

theorem lev_width (dom : ByteArray) (leaves : Array ByteArray) (h : ∀ x ∈ leaves, x.size = 64) :
    ∀ l, ∀ x ∈ lev dom leaves l, x.size = 64
  | 0, x, hx => by
    simp only [lev, base, Array.mem_append, Array.mem_map] at hx
    rcases hx with hx | ⟨k, -, rfl⟩
    · exact h x hx
    · exact Sha512.hash_size _
  | l + 1, x, hx => by
    simp only [lev, next, Array.mem_map] at hx
    obtain ⟨i, -, rfl⟩ := hx
    exact Sha512.hash_size _

/-- The node-frame inputs hashed while computing a root. -/
def rootStrings (dom : ByteArray) (leaves : Array ByteArray) : List ByteArray :=
  (List.range (log2ceil leaves.size)).flatMap fun l =>
    (List.range ((lev dom leaves l).size / 2)).map fun i =>
      Registered.nodeIn dom l i (lev dom leaves l)[2 * i]! (lev dom leaves l)[2 * i + 1]!

theorem node_mem {dom : ByteArray} {leaves : Array ByteArray} {l i : Nat}
    (hl : l < log2ceil leaves.size) (hi : i < (lev dom leaves l).size / 2) :
    Registered.nodeIn dom l i (lev dom leaves l)[2 * i]! (lev dom leaves l)[2 * i + 1]! ∈
      rootStrings dom leaves := by
  simp only [rootStrings, List.mem_flatMap, List.mem_range, List.mem_map]
  exact ⟨l, hl, i, hi, rfl⟩

theorem next_get (dom : ByteArray) (l : Nat) (a : Array ByteArray) {i : Nat} (hi : i < a.size / 2) :
    (next dom l a)[i]! = Sha512.hash (Registered.nodeIn dom l i a[2 * i]! a[2 * i + 1]!) := by
  rw [getElem!_pos _ i (by rw [next_size]; exact hi)]
  simp [next]

/-- Equal-size, nonempty frame-v3 roots bind their supplied leaves unless two
node-frame inputs from the two computations collide. -/
theorem fv3Root_bind {dom : ByteArray} {leaves leaves' : Array ByteArray} {S : List ByteArray}
    (hsz : leaves.size = leaves'.size) (hpos : 0 < leaves.size)
    (hw : ∀ x ∈ leaves, x.size = 64) (hw' : ∀ x ∈ leaves', x.size = 64)
    (hS : ∀ x ∈ rootStrings dom leaves, x ∈ S) (hS' : ∀ x ∈ rootStrings dom leaves', x ∈ S)
    (cf : CF S)
    (hr : FV3.root dom leaves Sha512.hash = FV3.root dom leaves' Sha512.hash) :
    leaves = leaves' := by
  have size_l : ∀ l ≤ log2ceil leaves.size,
      (lev dom leaves l).size = 2 ^ (log2ceil leaves.size - l) :=
    fun l hl => lev_size dom leaves l hl
  have size_l' : ∀ l ≤ log2ceil leaves.size,
      (lev dom leaves' l).size = 2 ^ (log2ceil leaves.size - l) := fun l hl => by
    rw [lev_size dom leaves' l (by rw [← hsz]; exact hl), ← hsz]
  have levels_eq : ∀ k ≤ log2ceil leaves.size,
      lev dom leaves (log2ceil leaves.size - k) =
        lev dom leaves' (log2ceil leaves.size - k) := by
    intro k
    induction k with
    | zero =>
      intro _
      rw [root_eq, root_eq, hsz] at hr
      have s1 := size_l (log2ceil leaves.size) (Nat.le_refl _)
      have s2 := size_l' (log2ceil leaves.size) (Nat.le_refl _)
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
      rw [show log2ceil leaves.size - k = (log2ceil leaves.size - (k + 1)) + 1 by omega] at ih
      generalize hl : log2ceil leaves.size - (k + 1) = l at ih ⊢
      have hlD : l < log2ceil leaves.size := by omega
      have s1 := size_l l (by omega)
      have s2 := size_l' l (by omega)
      rw [show log2ceil leaves.size - l = k + 1 by omega, Nat.pow_succ] at s1 s2
      apply Array.ext (by rw [s1, s2])
      intro j hj hj'
      have hi : j / 2 < (lev dom leaves l).size / 2 := by rw [s1]; omega
      have hi' : j / 2 < (lev dom leaves' l).size / 2 := by rw [s2]; omega
      have e := congrArg (·[j / 2]!) ih
      simp only [lev] at e
      rw [next_get _ _ _ hi, next_get _ _ _ hi'] at e
      have e := cf _ (hS _ (node_mem hlD hi)) _ (hS' _ (node_mem (by rw [← hsz]; exact hlD) hi')) e
      have w := lev_width dom leaves hw l
      have w' := lev_width dom leaves' hw' l
      have ha : ((lev dom leaves l)[2 * (j / 2)]!).size =
          ((lev dom leaves' l)[2 * (j / 2)]!).size := by
        rw [w _ (get!_mem (by omega)), w' _ (get!_mem (by omega))]
      obtain ⟨ea, eb⟩ := Registered.nodeIn_inj ha
        (by rw [w _ (get!_mem (by omega)), w' _ (get!_mem (by omega))]) e
      rw [← getElem!_pos _ j hj, ← getElem!_pos _ j hj']
      rcases Nat.even_or_odd' j with ⟨i, rfl | rfl⟩
      · rw [show 2 * i / 2 = i by omega] at ea
        exact ea
      · rw [show (2 * i + 1) / 2 = i by omega] at eb
        exact eb
  have e := levels_eq (log2ceil leaves.size) (Nat.le_refl _)
  simp only [Nat.sub_self, lev, base] at e
  exact (Array.append_inj e (by rw [hsz])).1

/-- Every SHA-512 input used to turn `bcs` into its registered top root:
the hm96 leaf inputs, the registered-row frames, and the tree's node frames. -/
def topStrings (dom : ByteArray) (bcs : Array ByteArray) : List ByteArray :=
  (List.range bcs.size).map (fun i => Hm96.Default512.leafPrefix ++ bcs[i]!) ++
  (List.range bcs.size).map (fun i =>
    Registered.leafIn dom i (Hm96.Default512.treeLeaf bcs[i]!)) ++
  rootStrings dom (topLeaves dom bcs)

theorem hm_mem {dom : ByteArray} {bcs : Array ByteArray} {i : Nat} (hi : i < bcs.size) :
    Hm96.Default512.leafPrefix ++ bcs[i]! ∈ topStrings dom bcs := by
  simp only [topStrings, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inl ⟨i, hi, rfl⟩

theorem row_mem {dom : ByteArray} {bcs : Array ByteArray} {i : Nat} (hi : i < bcs.size) :
    Registered.leafIn dom i (Hm96.Default512.treeLeaf bcs[i]!) ∈ topStrings dom bcs := by
  simp only [topStrings, List.mem_append, List.mem_map, List.mem_range]
  exact Or.inr (Or.inl ⟨i, hi, rfl⟩)

theorem root_mem {dom : ByteArray} {bcs : Array ByteArray} {x : ByteArray}
    (hx : x ∈ rootStrings dom (topLeaves dom bcs)) : x ∈ topStrings dom bcs := by
  simp only [topStrings, List.mem_append]
  exact Or.inr (Or.inr hx)

theorem topLeaves_size (dom : ByteArray) (bcs : Array ByteArray) :
    (topLeaves dom bcs).size = bcs.size := by
  simp [topLeaves]

theorem topLeaves_width (dom : ByteArray) (bcs : Array ByteArray) :
    ∀ x ∈ topLeaves dom bcs, x.size = 64 := by
  intro x hx
  simp only [topLeaves, Array.mem_mapIdx] at hx
  obtain ⟨i, hi, rfl⟩ := hx
  exact Sha512.hash_size _

/-- Under no collision among either recomputation's inputs, equal top roots
bind the arrays of commit strings. -/
theorem top_bind {dom : ByteArray} {bcs bcs' : Array ByteArray}
    (hsz : bcs.size = bcs'.size) (hpos : 0 < bcs.size)
    (cf : CF (topStrings dom bcs ++ topStrings dom bcs'))
    (hroot : FV3.root dom (topLeaves dom bcs) Sha512.hash =
      FV3.root dom (topLeaves dom bcs') Sha512.hash) :
    bcs = bcs' := by
  have hleaves := fv3Root_bind
    (dom := dom) (leaves := topLeaves dom bcs) (leaves' := topLeaves dom bcs')
    (S := topStrings dom bcs ++ topStrings dom bcs')
    (by simp [topLeaves_size, hsz]) (by simpa [topLeaves_size] using hpos)
    (topLeaves_width dom bcs) (topLeaves_width dom bcs')
    (fun x hx => List.mem_append_left _ (root_mem hx))
    (fun x hx => List.mem_append_right _ (root_mem hx)) cf hroot
  apply Array.ext hsz
  intro i hi hi'
  have e := congrArg (·[i]!) hleaves
  simp only [topLeaves, getElem!_pos _ i hi, getElem!_pos _ i hi', Array.getElem_mapIdx] at e
  rw [Registered.rowLeaf_eq, Registered.rowLeaf_eq] at e
  have erow := cf _ (List.mem_append_left _ (row_mem hi)) _
    (List.mem_append_right _ (row_mem hi')) e
  have eleaf := Registered.leafIn_inj
    (by rw [Sha512.hash_size, Sha512.hash_size]) erow
  have ehm := cf _ (List.mem_append_left _ (hm_mem hi)) _
    (List.mem_append_right _ (hm_mem hi')) eleaf
  have ebc := ByteArray.append_right_inj Hm96.Default512.leafPrefix ehm
  rwa [getElem!_pos _ i hi, getElem!_pos _ i hi'] at ebc

end TopRoot

open TopRoot

/-- Two unequal level-zero top arrays with one `topEntry` root expose a
collision among SHA-512 inputs used by those two root computations. -/
theorem topRoot_collision {dom : ByteArray} {bcs bcs' : Array ByteArray}
    (hsz : bcs.size = bcs'.size) (hpos : 0 < bcs.size) (hne : bcs ≠ bcs')
    (hroot : Flock.FV3.root dom (topLeaves dom bcs) Flock.Sha512.hash =
      Flock.FV3.root dom (topLeaves dom bcs') Flock.Sha512.hash) :
    Merkle.Collision Refine.H512 := by
  classical
  let S := topStrings dom bcs ++ topStrings dom bcs'
  by_contra noCollision
  have cf : CF S := by
    intro x hx y hy he
    by_contra hxy
    apply noCollision
    refine ⟨x.data.toList, y.data.toList, ?_, ?_⟩
    · intro h
      exact hxy (ByteArray.ext (Array.ext' h))
    · apply Subtype.ext
      simpa only [Refine.H512, Refine.bytes_of_list] using he
  exact hne (top_bind hsz hpos cf hroot)

end Flock.SecurityProofs
