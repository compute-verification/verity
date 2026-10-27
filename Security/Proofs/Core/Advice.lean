import Mathlib.Basic.Countable.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.List.OfFn
import Definitions.Core.Advice

/-!
# The advice tree binds its rows, up to a collision among the strings it hashes

Two advice trees with one root have the same rows, or `H` maps two distinct strings, one hashed in each tree, to one
digest (`root_bind`). Down from the root, equal digests of different strings are such a collision; equal strings give
equal parts, since the framing is separated and digests have one length. At the leaves, equal hm96 leaves give equal salt
digests, so equal salts (or a collision), so equal masks, so equal row digests, so equal rows (or a collision). Two trees
of different depths collide at the root, whose messages differ in level or kind.
-/

namespace Verity.Advice

variable {H : List UInt8 → Digest} {F : Framing}

/-- `H` collides between a string of `S` and a different one of `T`. -/
def CollidesIn (H : List UInt8 → Digest) (S T : Finset (List UInt8)) : Prop :=
  ∃ x ∈ S, ∃ y ∈ T, x ≠ y ∧ H x = H y

theorem CollidesIn.mono {S S' T T' : Finset (List UInt8)} (hS : S ⊆ S') (hT : T ⊆ T') :
    CollidesIn H S T → CollidesIn H S' T'
  | ⟨x, hx, y, hy, hne, he⟩ => ⟨x, hS hx, y, hT hy, hne, he⟩

/-- One hash: equal digests of `x ∈ S` and `y ∈ T` are equal strings or a collision. -/
theorem eq_or_collides {S T : Finset (List UInt8)} {x y : List UInt8} (hx : x ∈ S) (hy : y ∈ T) (h : H x = H y) :
    x = y ∨ CollidesIn H S T := by
  by_cases hxy : x = y
  · exact Or.inl hxy
  · exact Or.inr ⟨x, hx, y, hy, hxy, h⟩

theorem xor_inj {x x' m : Digest} (h : (fun i => x i ^^^ m i : Digest) = fun i => x' i ^^^ m i) : x = x' :=
  funext fun i => (UInt8.xor_left_inj (m i)).1 (congrFun h i)

theorem bytes_inj {d e : Digest} (h : d.bytes = e.bytes) : d = e := List.ofFn_injective h

theorem digests_inj {b c b' c' : Digest} (h : b.bytes ++ c.bytes = b'.bytes ++ c'.bytes) : b = b' ∧ c = c' := by
  obtain ⟨hb, hc⟩ := List.append_inj h (by simp [Digest.bytes])
  exact ⟨bytes_inj hb, bytes_inj hc⟩

/-- Position `i` holds the same in both trees: a row in both or in neither, and then the same row. -/
def Agree (rows₀ rows₁ : List (List UInt8)) (i : ℕ) : Prop :=
  (i < rows₀.length ↔ i < rows₁.length) ∧ ∀ (h₀ : i < rows₀.length) (h₁ : i < rows₁.length), rows₀[i] = rows₁[i]

variable {rows₀ rows₁ : List (List UInt8)} {salt₀ salt₁ : ℕ → List UInt8}

/-- **hm96's leaf binds its row**, up to a collision among the leaf's four inputs. -/
theorem row_bind {S T : Finset (List UInt8)} {i : ℕ} {row₀ row₁ y₀ y₁ : List UInt8} (hF : F.Separated)
    (hS : rowInput F row₀ ∈ S ∧ saltInput F y₀ ∈ S ∧ hmInput H F row₀ y₀ ∈ S ∧ leafInput H F i row₀ y₀ ∈ S)
    (hT : rowInput F row₁ ∈ T ∧ saltInput F y₁ ∈ T ∧ hmInput H F row₁ y₁ ∈ T ∧ leafInput H F i row₁ y₁ ∈ T)
    (h : H (leafInput H F i row₀ y₀) = H (leafInput H F i row₁ y₁)) : row₀ = row₁ ∨ CollidesIn H S T := by
  obtain ⟨hr₀, hs₀, hm₀, hl₀⟩ := hS
  obtain ⟨hr₁, hs₁, hm₁, hl₁⟩ := hT
  rcases eq_or_collides hl₀ hl₁ h with h | hc
  · rcases eq_or_collides hm₀ hm₁ (hF.leaf_inj i h) with h | hc
    · obtain ⟨hb, hc⟩ := digests_inj (List.append_cancel_left h)
      rcases eq_or_collides hs₀ hs₁ hc with hy | hc
      · have hy : y₀ = y₁ := List.append_cancel_left hy
        subst hy
        rcases eq_or_collides hr₀ hr₁ (xor_inj hb) with hx | hc
        · exact Or.inl (List.append_cancel_left hx)
        · exact Or.inr hc
      · exact Or.inr hc
    · exact Or.inr hc
  · exact Or.inr hc

theorem leaf_mem {a b c d : List UInt8} : d ∈ ({a, b, c, d} : Finset (List UInt8)) :=
  Finset.mem_insert_of_mem (Finset.mem_insert_of_mem (Finset.mem_insert_of_mem (Finset.mem_singleton_self _)))

/-- **A leaf binds what it holds**, up to a collision among its inputs: a row in both trees or in neither, and then the
same row. -/
theorem leaf_bind (hF : F.Separated) {i : ℕ} (h : leafAt H F rows₀ salt₀ i = leafAt H F rows₁ salt₁ i) :
    Agree rows₀ rows₁ i ∨ CollidesIn H (inputsAt H F rows₀ salt₀ 0 i) (inputsAt H F rows₁ salt₁ 0 i) := by
  unfold leafAt at h
  unfold inputsAt
  by_cases h₀ : i < rows₀.length <;> by_cases h₁ : i < rows₁.length <;> simp only [h₀, h₁, dite_true, dite_false] at h ⊢
  · exact (row_bind hF (by simp only [Finset.mem_insert, Finset.mem_singleton, true_or, or_true, and_self])
      (by simp only [Finset.mem_insert, Finset.mem_singleton, true_or, or_true, and_self]) h).imp
        (fun hr => ⟨by simp [h₀, h₁], fun _ _ => hr⟩) id
  · exact Or.inr ⟨_, leaf_mem, _, Finset.mem_singleton_self _, hF.leaf_ne_pad _ _ _, h⟩
  · exact Or.inr ⟨_, Finset.mem_singleton_self _, _, leaf_mem, fun e => hF.leaf_ne_pad _ _ _ e.symm, h⟩
  · exact Or.inl ⟨by simp [h₀, h₁], fun h => absurd h h₀⟩

theorem node_mem (rows : List (List UInt8)) (salt : ℕ → List UInt8) (k j : ℕ) :
    F.node k j (nodeAt H F rows salt k (2 * j)) (nodeAt H F rows salt k (2 * j + 1)) ∈
      inputsAt H F rows salt (k + 1) j :=
  Finset.mem_insert_self _ _

theorem left_sub (rows : List (List UInt8)) (salt : ℕ → List UInt8) (k j : ℕ) :
    inputsAt H F rows salt k (2 * j) ⊆ inputsAt H F rows salt (k + 1) j :=
  Finset.subset_union_left.trans (Finset.subset_insert _ _)

theorem right_sub (rows : List (List UInt8)) (salt : ℕ → List UInt8) (k j : ℕ) :
    inputsAt H F rows salt k (2 * j + 1) ⊆ inputsAt H F rows salt (k + 1) j :=
  Finset.subset_union_right.trans (Finset.subset_insert _ _)

theorem div_two_pow_succ {i k j : ℕ} (h : i / 2 ^ (k + 1) = j) : i / 2 ^ k = 2 * j ∨ i / 2 ^ k = 2 * j + 1 := by
  rw [Nat.pow_succ, ← Nat.div_div_eq_div_mul] at h
  omega

/-- **A subtree binds its leaves**: two nodes of one height and index with one digest agree at every position below them,
up to a collision among the strings hashed below them. -/
theorem nodeAt_bind (hF : F.Separated) : ∀ k j, nodeAt H F rows₀ salt₀ k j = nodeAt H F rows₁ salt₁ k j →
    (∀ i, i / 2 ^ k = j → Agree rows₀ rows₁ i) ∨
      CollidesIn H (inputsAt H F rows₀ salt₀ k j) (inputsAt H F rows₁ salt₁ k j)
  | 0, j, h => by
    rcases leaf_bind hF h with ha | hc
    · exact Or.inl fun i hi => by simp only [pow_zero, Nat.div_one] at hi; exact hi ▸ ha
    · exact Or.inr hc
  | k + 1, j, h => by
    rcases eq_or_collides (node_mem rows₀ salt₀ k j) (node_mem rows₁ salt₁ k j) h with he | hc
    · obtain ⟨-, ha, hb⟩ := hF.node_inj he
      rcases nodeAt_bind hF k (2 * j) ha with ha | hc
      · rcases nodeAt_bind hF k (2 * j + 1) hb with hb | hc
        · refine Or.inl fun i hi => ?_
          rcases div_two_pow_succ hi with hi | hi
          · exact ha i hi
          · exact hb i hi
        · exact Or.inr (hc.mono (right_sub _ _ _ _) (right_sub _ _ _ _))
      · exact Or.inr (hc.mono (left_sub _ _ _ _) (left_sub _ _ _ _))
    · exact Or.inr hc

/-- The string whose digest is the node at height `k` and index `j`. -/
def topAt (H : List UInt8 → Digest) (F : Framing) (rows : List (List UInt8)) (salt : ℕ → List UInt8) :
    ℕ → ℕ → List UInt8
  | 0, j => if h : j < rows.length then leafInput H F j rows[j] (salt j) else F.pad j
  | k + 1, j => F.node k j (nodeAt H F rows salt k (2 * j)) (nodeAt H F rows salt k (2 * j + 1))

theorem nodeAt_eq (rows : List (List UInt8)) (salt : ℕ → List UInt8) :
    ∀ k j, nodeAt H F rows salt k j = H (topAt H F rows salt k j)
  | 0, j => by rw [nodeAt, leafAt, topAt]; split_ifs <;> rfl
  | _ + 1, _ => rfl

theorem topAt_mem (rows : List (List UInt8)) (salt : ℕ → List UInt8) :
    ∀ k j, topAt H F rows salt k j ∈ inputsAt H F rows salt k j
  | 0, j => by
    rw [topAt, inputsAt]
    split_ifs <;> simp only [Finset.mem_insert, Finset.mem_singleton, or_true]
  | k + 1, j => node_mem rows salt k j

/-- Nodes of different heights are hashed from different strings: a node's message names its level, and a leaf's or a
pad's is no node's. -/
theorem topAt_ne (hF : F.Separated) {k k' : ℕ} (hk : k ≠ k') (j j' : ℕ) :
    topAt H F rows₀ salt₀ k j ≠ topAt H F rows₁ salt₁ k' j' := by
  match k, k' with
  | 0, 0 => exact absurd rfl hk
  | 0, _ + 1 =>
    rw [topAt, topAt]
    split_ifs
    · exact hF.leaf_ne_node _ _ _ _ _ _
    · exact hF.pad_ne_node _ _ _ _ _
  | _ + 1, 0 =>
    rw [topAt, topAt]
    split_ifs
    · exact fun e => hF.leaf_ne_node _ _ _ _ _ _ e.symm
    · exact fun e => hF.pad_ne_node _ _ _ _ _ e.symm
  | _ + 1, _ + 1 => exact fun e => hk (congrArg (· + 1) (hF.node_inj e).1)

theorem length_le_depth (count : ℕ) : count ≤ 2 ^ depth count := by
  have := Nat.lt_size_self (count - 1)
  unfold depth
  omega

/-- **The advice tree binds its rows**: two trees with one root have the same rows, or `H` collides between a string one
hashes and a different one the other does. -/
theorem root_bind (hF : F.Separated) (h : root H F rows₀ salt₀ = root H F rows₁ salt₁) :
    rows₀ = rows₁ ∨ CollidesIn H (hashInputs H F rows₀ salt₀) (hashInputs H F rows₁ salt₁) := by
  unfold root at h
  unfold hashInputs
  by_cases hd : depth rows₀.length = depth rows₁.length
  · have hl₀ := length_le_depth rows₀.length
    have hl₁ := length_le_depth rows₁.length
    rw [← hd] at h hl₁ ⊢
    rcases nodeAt_bind hF _ 0 h with ha | hc
    · have ha : ∀ i < 2 ^ depth rows₀.length, Agree rows₀ rows₁ i := fun i hi => ha i (Nat.div_eq_of_lt hi)
      have hlen : rows₀.length = rows₁.length := by
        rcases Nat.lt_trichotomy rows₀.length rows₁.length with hlt | heq | hlt
        · exact absurd ((ha _ (by omega)).1.mpr hlt) (lt_irrefl _)
        · exact heq
        · exact absurd ((ha _ (by omega)).1.mp hlt) (lt_irrefl _)
      exact Or.inl (List.ext_getElem hlen fun i h₀ h₁ => (ha i (by omega)).2 h₀ h₁)
    · exact Or.inr hc
  · rw [nodeAt_eq, nodeAt_eq] at h
    exact Or.inr ⟨_, topAt_mem _ _ _ _, _, topAt_mem _ _ _ _, topAt_ne hF hd 0 0, h⟩

/-! ## The premises can hold: a separated framing, and rows that decode -/

/-- A framing that is separated: a kind byte, then the digests, then a level or rank in unary. -/
def Framing.unary : Framing where
  rowPrefix := []
  saltPrefix := []
  leafPrefix := []
  mask _ := fun _ => 0
  leaf i t := 0 :: (t.bytes ++ List.replicate i 0)
  pad i := 1 :: List.replicate i 0
  node k _ a b := 2 :: (a.bytes ++ (b.bytes ++ List.replicate k 0))

theorem Framing.unary_separated : Framing.unary.Separated where
  leaf_inj i t t' h := by
    simp only [Framing.unary, List.cons.injEq, true_and] at h
    exact bytes_inj (List.append_inj h (by simp [Digest.bytes])).1
  node_inj {k j a b k' j' a' b'} h := by
    simp only [Framing.unary, List.cons.injEq, true_and] at h
    obtain ⟨ha, h⟩ := List.append_inj h (by simp [Digest.bytes])
    obtain ⟨hb, h⟩ := List.append_inj h (by simp [Digest.bytes])
    exact ⟨by simpa using congrArg List.length h, bytes_inj ha, bytes_inj hb⟩
  leaf_ne_pad _ _ _ := by simp [Framing.unary]
  leaf_ne_node _ _ _ _ _ _ := by simp [Framing.unary]
  pad_ne_node _ _ _ _ _ := by simp [Framing.unary]

theorem partition_unit_injective {C : Circuit} {n : ℕ} :
    Function.Injective fun P : Partition C n => P.unit := by
  rintro ⟨u, hu⟩ ⟨u', hu'⟩ h
  cases h
  rfl

/-- Rows that decode, for every circuit: a partition's index in a counting of all partitions, in unary. -/
theorem Rows.exists_decodes (C : Circuit) : ∃ A : Rows C, A.Decodes := by
  have : Countable ((n : ℕ) × Partition C n) :=
    have : ∀ n, Countable (Partition C n) := fun _ => partition_unit_injective.countable
    inferInstance
  obtain ⟨f, hf⟩ := Countable.exists_injective_nat ((n : ℕ) × Partition C n)
  let rows : (n : ℕ) → Partition C n → List (List UInt8) := fun n P => [List.replicate (f ⟨n, P⟩) 0]
  have hrows : Function.Injective fun x : (n : ℕ) × Partition C n => rows x.1 x.2 := fun x y h =>
    hf (by simpa [rows] using h)
  exact ⟨⟨rows, Function.partialInv fun x : (n : ℕ) × Partition C n => rows x.1 x.2⟩,
    fun n P => Function.partialInv_left hrows ⟨n, P⟩⟩

end Verity.Advice
