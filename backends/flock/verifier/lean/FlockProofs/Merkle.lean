/-
Level 3, L3-M (PROTOCOL.md §14, §17.4 stage 1b): Merkle binding. Two openings the verifier accepts for one position of one
cap, with different rows, give an explicit collision of the scheme's hash: two different digest-width node inputs, or two
different rows, with one digest. Parametric in the scheme; for each scheme the verifier knows, such a collision is a SHA-256
or SHA-512 collision (`MerkleScheme.collision_hash`). No random oracle.
-/
import FlockProofs.Hash
import FlockProofs.Encode

namespace Flock

/-- A scheme's digests have its width: every node and leaf it computes is `digestLen` bytes. -/
def MerkleScheme.Sized (ms : MerkleScheme) : Prop :=
  (∀ l r, (ms.node l r).size = ms.digestLen) ∧ ∀ row y, (ms.leaf row y).size = ms.digestLen

theorem MerkleScheme.sha256_sized (id : String) : (MerkleScheme.sha256 id).Sized :=
  ⟨fun _ _ => Sha256.hash_size _, fun _ _ => Sha256.hash_size _⟩

theorem MerkleScheme.sha512_sized (id : String) : (MerkleScheme.sha512 id).Sized :=
  ⟨fun _ _ => Sha512.hash_size _, fun _ _ => Sha512.hash_size _⟩

theorem MerkleScheme.hm96Sha512_sized : MerkleScheme.hm96Sha512.Sized :=
  ⟨fun _ _ => Sha512.hash_size _, fun _ _ => Hm96.Default512.leaf_size _ _⟩

theorem MerkleScheme.all_sized : ∀ ms ∈ MerkleScheme.all, ms.Sized := by
  intro ms h
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl
  · exact MerkleScheme.sha256_sized _
  · exact MerkleScheme.sha256_sized _
  · exact MerkleScheme.sha512_sized _
  · exact MerkleScheme.hm96Sha512_sized

/-- A collision of a Merkle scheme's hash: two different pairs of digest-width children with one node, or two different rows
with one leaf under some salts.
- The children must have the digest's width. A node hashes `l ‖ r` without length framing, so one byte string split two
  ways would otherwise be a collision of nothing. The verifier only ever hashes digest-width children: its siblings are
  checked to be `digestLen` bytes, and its running digests are hash outputs.
- The rows must differ. The salts are the prover's (`opened_salts`), and an unsalted scheme's leaf ignores its salt, so one
  row under two salts would otherwise be a collision of nothing. -/
def MerkleScheme.Collision (ms : MerkleScheme) : Prop :=
  (∃ a b a' b' : ByteArray, a.size = ms.digestLen ∧ b.size = ms.digestLen ∧ a'.size = ms.digestLen ∧
    b'.size = ms.digestLen ∧ (a, b) ≠ (a', b') ∧ ms.node a b = ms.node a' b') ∨
  (∃ (r r' : Array F128) (y y' : ByteArray), r ≠ r' ∧ ms.leaf r y = ms.leaf r' y')

/-- Climbing from two different digests to one digest along paths of one length, every digest and sibling of the scheme's
width, is a node collision. -/
theorem climb_binding (ms : MerkleScheme) (hs : ms.Sized) :
    ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx : Nat), s1.length = s2.length →
      (∀ x ∈ s1, x.size = ms.digestLen) → (∀ x ∈ s2, x.size = ms.digestLen) → a.size = ms.digestLen →
      b.size = ms.digestLen → a ≠ b → climb ms a idx s1 = climb ms b idx s2 → ms.Collision := by
  intro s1
  induction s1 with
  | nil =>
    intro s2 a b idx hl _ _ _ _ hab h
    cases s2 with
    | nil => exact absurd h hab
    | cons _ _ => simp at hl
  | cons s rest ih =>
    intro s2 a b idx hl hw1 hw2 ha hb hab h
    cases s2 with
    | nil => simp at hl
    | cons t rest2 =>
      have hl' : rest.length = rest2.length := by simpa using hl
      have hsz := hw1 s (by simp)
      have htz := hw2 t (by simp)
      have hr1 : ∀ x ∈ rest, x.size = ms.digestLen := fun x hx => hw1 x (by simp [hx])
      have hr2 : ∀ x ∈ rest2, x.size = ms.digestLen := fun x hx => hw2 x (by simp [hx])
      by_cases hp : idx % 2 = 0
      · have h' : climb ms (ms.node a s) (idx / 2) rest = climb ms (ms.node b t) (idx / 2) rest2 := by
          simpa [climb, hp] using h
        by_cases he : ms.node a s = ms.node b t
        · exact Or.inl ⟨a, s, b, t, ha, hsz, hb, htz, fun hh => hab (Prod.mk.inj hh).1, he⟩
        · exact ih rest2 _ _ (idx / 2) hl' hr1 hr2 (hs.1 _ _) (hs.1 _ _) he h'
      · have h' : climb ms (ms.node s a) (idx / 2) rest = climb ms (ms.node t b) (idx / 2) rest2 := by
          simpa [climb, hp] using h
        by_cases he : ms.node s a = ms.node t b
        · exact Or.inl ⟨s, a, t, b, hsz, ha, htz, hb, fun hh => hab (Prod.mk.inj hh).2, he⟩
        · exact ih rest2 _ _ (idx / 2) hl' hr1 hr2 (hs.1 _ _) (hs.1 _ _) he h'

/-- Two accepted openings of one position against one cap, with different rows and siblings of the digest's width, are a
collision. -/
theorem opens_binding (ms : MerkleScheme) (hs : ms.Sized) (cap : Array ByteArray) (c d lanes pos : Nat)
    (r1 r2 : Array F128) (y1 y2 : ByteArray) (s1 s2 : List ByteArray) (hw1 : ∀ x ∈ s1, x.size = ms.digestLen)
    (hw2 : ∀ x ∈ s2, x.size = ms.digestLen) (h1 : opens ms cap c d lanes pos r1 y1 s1 = true)
    (h2 : opens ms cap c d lanes pos r2 y2 s2 = true) (hr : r1 ≠ r2) : ms.Collision := by
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq : climb ms (ms.leaf r1 y1) pos s1 = climb ms (ms.leaf r2 y2) pos s2 := h1.2.trans h2.2.symm
  by_cases hl : ms.leaf r1 y1 = ms.leaf r2 y2
  · exact Or.inr ⟨r1, r2, y1, y2, hr, hl⟩
  · exact climb_binding ms hs s1 s2 _ _ pos (by rw [h1.1.1.2, h2.1.1.2]) hw1 hw2 (hs.2 _ _) (hs.2 _ _) hl heq

/-- An accepted Merkle check opens every query. -/
theorem merkleCheck_opens (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows : Array (Array F128)) (siblings salts : Array ByteArray) (lanes : Nat)
    (h : merkleCheck ms cap c d positions rows siblings salts lanes = .ok ()) (j : Nat) (hj : j < positions.size) :
    opens ms cap c d lanes positions[j]! rows[j]! (salts.getD j ByteArray.empty) (sibsOf siblings (d - c) j) = true := by
  unfold merkleCheck at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
    · split at h
      · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      · rename_i hn
        have := List.find?_eq_none.mp hn j (List.mem_range.mpr hj)
        simpa using this

/-- An accepted Merkle check's siblings all have the digest's width. -/
theorem merkleCheck_sibs (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows : Array (Array F128)) (siblings salts : Array ByteArray) (lanes : Nat)
    (h : merkleCheck ms cap c d positions rows siblings salts lanes = .ok ()) (j : Nat) :
    ∀ x ∈ sibsOf siblings (d - c) j, x.size = ms.digestLen := by
  unfold merkleCheck at h
  split at h
  · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
  · split at h
    · exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
    · rename_i hw
      intro x hx
      simp only [sibsOf, Array.mem_toList_iff] at hx
      obtain ⟨k, hk, rfl⟩ := Array.mem_extract_iff_getElem.mp hx
      simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, bne_iff_ne, ne_eq, Decidable.not_not] at hw
      exact hw.2 _ (by omega)

/-- **Merkle binding (L3-M).** Two Merkle checks the verifier accepts against one cap at the same positions, whose opened
rows differ at some query, give a collision of the scheme's hash, for a scheme whose digests have its width
(`MerkleScheme.all_sized`: every scheme the verifier knows). -/
theorem merkle_binding (ms : MerkleScheme) (hs : ms.Sized) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows1 rows2 : Array (Array F128)) (sib1 sib2 salt1 salt2 : Array ByteArray) (lanes : Nat)
    (h1 : merkleCheck ms cap c d positions rows1 sib1 salt1 lanes = .ok ())
    (h2 : merkleCheck ms cap c d positions rows2 sib2 salt2 lanes = .ok ())
    (j : Nat) (hj : j < positions.size) (hr : rows1[j]! ≠ rows2[j]!) : ms.Collision :=
  opens_binding ms hs cap c d lanes positions[j]! _ _ _ _ _ _
    (merkleCheck_sibs ms cap c d positions rows1 sib1 salt1 lanes h1 j)
    (merkleCheck_sibs ms cap c d positions rows2 sib2 salt2 lanes h2 j)
    (merkleCheck_opens ms cap c d positions rows1 sib1 salt1 lanes h1 j hj)
    (merkleCheck_opens ms cap c d positions rows2 sib2 salt2 lanes h2 j hj) hr

/-! ### What a collision is, for each scheme the verifier knows -/

/-- Two different pairs of `w`-byte children are two different concatenations. -/
theorem append_ne_of_pair_ne {a b a' b' : ByteArray} {w : Nat} (ha : a.size = w) (ha' : a'.size = w)
    (hne : (a, b) ≠ (a', b')) : a ++ b ≠ a' ++ b' := by
  intro h
  have h1 := ByteArray.append_inj_left h (by rw [ha, ha'])
  subst h1
  exact hne (by rw [(ByteArray.append_right_inj _).mp h])

theorem MerkleScheme.sha256_collision (id : String) (h : (MerkleScheme.sha256 id).Collision) : Sha256.Collision := by
  rcases h with ⟨a, b, a', b', ha, _, ha', _, hne, he⟩ | ⟨r, r', _, _, hr, he⟩
  · exact ⟨_, _, append_ne_of_pair_ne ha ha' hne, he⟩
  · exact ⟨_, _, fun h => hr (rowBytes_inj h), he⟩

theorem MerkleScheme.sha512_collision (id : String) (h : (MerkleScheme.sha512 id).Collision) : Sha512.Collision := by
  rcases h with ⟨a, b, a', b', ha, _, ha', _, hne, he⟩ | ⟨r, r', _, _, hr, he⟩
  · exact ⟨_, _, append_ne_of_pair_ne ha ha' hne, he⟩
  · exact ⟨_, _, fun h => hr (rowBytes_inj h), he⟩

/-- For `hm96-sha512/v1`, a leaf collision of two different rows is a collision of the outer hash, the salt digest or the
row hash (`Hm96.Default512.leaf_binding`, `rowBytes_inj`). -/
theorem MerkleScheme.hm96Sha512_collision (h : MerkleScheme.hm96Sha512.Collision) : Sha512.Collision := by
  rcases h with ⟨a, b, a', b', ha, _, ha', _, hne, he⟩ | ⟨r, r', y, y', hr, he⟩
  · exact ⟨_, _, append_ne_of_pair_ne ha ha' hne, he⟩
  · by_cases hx : Sha512.hash (MerkleScheme.rowBytes r) = Sha512.hash (MerkleScheme.rowBytes r')
    · exact ⟨_, _, fun h => hr (rowBytes_inj h), hx⟩
    · exact Hm96.Default512.leaf_binding _ _ y y' (Sha512.hash_size _) (Sha512.hash_size _)
        (fun hh => hx (Prod.mk.inj hh).1) he

/-- **A collision of any scheme the verifier knows is a SHA-256 or SHA-512 collision**, so `merkle_binding` reduces the
verifier's Merkle checks to the hashes' collision resistance. -/
theorem MerkleScheme.collision_hash : ∀ ms ∈ MerkleScheme.all, ms.Collision → Sha256.Collision ∨ Sha512.Collision := by
  intro ms h hc
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl
  · exact Or.inl (MerkleScheme.sha256_collision _ hc)
  · exact Or.inl (MerkleScheme.sha256_collision _ hc)
  · exact Or.inr (MerkleScheme.sha512_collision _ hc)
  · exact Or.inr (MerkleScheme.hm96Sha512_collision hc)

end Flock
