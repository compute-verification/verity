/-
Level 3, L3-M (PROTOCOL.md §14, §17.4 stage 1b): Merkle binding. Two openings the verifier accepts for one position of one
cap, with different rows, give an explicit collision of the scheme's hash: two different digest-width node inputs, or two
different rows, with one digest. Parametric in the scheme; for each scheme the verifier knows, such a collision is a SHA-256
or SHA-512 collision (`MerkleScheme.collision_hash`). No random oracle.
-/
import Proofs.Flock.Verifier.Hash
import Proofs.Flock.Verifier.Encode

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

/-! ### The extractor: two accepted openings give an explicit colliding pair

A fixed hash with 32- or 64-byte digests has collisions by counting, so a proposition "a collision exists" is provable
outright and says nothing about the verifier. What the verifier's checks give is a reduction: computable functions from two
accepted openings to an explicit pair of inputs (`climbPair`, `opensPair`, `merklePair`, then `MerklePair.shaInputs` /
`MerklePair.hm96Inputs` for the hash's own inputs), and theorems that the pair collides. The propositions follow as
corollaries (`merkle_binding_collision`, `MerkleScheme.collision_hash`). -/

/-- Two hash inputs an extractor names: two pairs of children of one node, or two rows (with their salts) of one leaf. -/
inductive MerklePair where
  | node (a b a' b' : ByteArray)
  | leaf (r r' : Array F128) (y y' : ByteArray)
  deriving Inhabited

/-- The pair in hex, for tests. -/
def MerklePair.render : MerklePair → String
  | .node a b a' b' => s!"node {hexOf a} {hexOf b} | {hexOf a'} {hexOf b'}"
  | .leaf r r' y y' => s!"leaf {hexOf (MerkleScheme.rowBytes r)} {hexOf y} | {hexOf (MerkleScheme.rowBytes r')} {hexOf y'}"

/-- The pair collides under `ms`: two different pairs of digest-width children with one node, or two different rows with
one leaf under their salts.
- The children must have the digest's width. A node hashes `l ‖ r` without length framing, so one byte string split two
  ways would otherwise be a collision of nothing. The verifier only ever hashes digest-width children: its siblings are
  checked to be `digestLen` bytes, and its running digests are hash outputs.
- The rows must differ. The salts are the prover's (`opened_salts`), and an unsalted scheme's leaf ignores its salt. -/
def MerklePair.Collides (ms : MerkleScheme) : MerklePair → Prop
  | .node a b a' b' => a.size = ms.digestLen ∧ b.size = ms.digestLen ∧ a'.size = ms.digestLen ∧ b'.size = ms.digestLen ∧
      (a, b) ≠ (a', b') ∧ ms.node a b = ms.node a' b'
  | .leaf r r' y y' => r ≠ r' ∧ ms.leaf r y = ms.leaf r' y'

/-- A collision of a Merkle scheme's hash, as a proposition: some pair collides. -/
def MerkleScheme.HasCollision (ms : MerkleScheme) : Prop := ∃ p : MerklePair, p.Collides ms

/-- **The climb's extractor:** walking two paths up in step, the first node whose two parents are one while its children
differ. -/
def climbPair (ms : MerkleScheme) : List ByteArray → List ByteArray → ByteArray → ByteArray → Nat → MerklePair
  | s :: rest, t :: rest2, a, b, idx =>
    if idx % 2 = 0 then
      if ms.node a s = ms.node b t then .node a s b t else climbPair ms rest rest2 (ms.node a s) (ms.node b t) (idx / 2)
    else
      if ms.node s a = ms.node t b then .node s a t b else climbPair ms rest rest2 (ms.node s a) (ms.node t b) (idx / 2)
  | _, _, a, b, _ => .node a b a b

/-- `climbPair`'s output collides: climbing from two different digests to one digest along paths of one length, every digest
and sibling of the scheme's width. -/
theorem climb_binding (ms : MerkleScheme) (hs : ms.Sized) :
    ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx : Nat), s1.length = s2.length →
      (∀ x ∈ s1, x.size = ms.digestLen) → (∀ x ∈ s2, x.size = ms.digestLen) → a.size = ms.digestLen →
      b.size = ms.digestLen → a ≠ b → climb ms a idx s1 = climb ms b idx s2 → (climbPair ms s1 s2 a b idx).Collides ms := by
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
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ⟨ha, hsz, hb, htz, fun hh => hab (Prod.mk.inj hh).1, he⟩
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ih rest2 _ _ (idx / 2) hl' hr1 hr2 (hs.1 _ _) (hs.1 _ _) he h'
      · have h' : climb ms (ms.node s a) (idx / 2) rest = climb ms (ms.node t b) (idx / 2) rest2 := by
          simpa [climb, hp] using h
        by_cases he : ms.node s a = ms.node t b
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ⟨hsz, ha, htz, hb, fun hh => hab (Prod.mk.inj hh).2, he⟩
        · simp only [climbPair, hp, he, ↓reduceIte]
          exact ih rest2 _ _ (idx / 2) hl' hr1 hr2 (hs.1 _ _) (hs.1 _ _) he h'

/-- **The opening's extractor:** the two rows if their leaves are one, else the climb's pair. -/
def opensPair (ms : MerkleScheme) (pos : Nat) (r1 : Array F128) (y1 : ByteArray) (s1 : List ByteArray) (r2 : Array F128)
    (y2 : ByteArray) (s2 : List ByteArray) : MerklePair :=
  if ms.leaf r1 y1 = ms.leaf r2 y2 then .leaf r1 r2 y1 y2 else climbPair ms s1 s2 (ms.leaf r1 y1) (ms.leaf r2 y2) pos

/-- `opensPair`'s output collides: two accepted openings of one position against one cap, with different rows and siblings
of the digest's width. -/
theorem opens_binding (ms : MerkleScheme) (hs : ms.Sized) (cap : Array ByteArray) (c d lanes pos : Nat)
    (r1 r2 : Array F128) (y1 y2 : ByteArray) (s1 s2 : List ByteArray) (hw1 : ∀ x ∈ s1, x.size = ms.digestLen)
    (hw2 : ∀ x ∈ s2, x.size = ms.digestLen) (h1 : opens ms cap c d lanes pos r1 y1 s1 = true)
    (h2 : opens ms cap c d lanes pos r2 y2 s2 = true) (hr : r1 ≠ r2) : (opensPair ms pos r1 y1 s1 r2 y2 s2).Collides ms := by
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq : climb ms (ms.leaf r1 y1) pos s1 = climb ms (ms.leaf r2 y2) pos s2 := h1.2.trans h2.2.symm
  by_cases hl : ms.leaf r1 y1 = ms.leaf r2 y2
  · simp only [opensPair, hl, ↓reduceIte]
    exact ⟨hr, hl⟩
  · simp only [opensPair, hl, ↓reduceIte]
    exact climb_binding ms hs s1 s2 _ _ pos (by rw [h1.1.1.2, h2.1.1.2]) hw1 hw2 (hs.2 _ _) (hs.2 _ _) hl heq

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

/-- **The Merkle check's extractor:** query `j`'s two openings, as `opensPair` reads them. -/
def merklePair (ms : MerkleScheme) (c d : Nat) (positions : Array Nat) (rows1 rows2 : Array (Array F128))
    (sib1 sib2 salt1 salt2 : Array ByteArray) (j : Nat) : MerklePair :=
  opensPair ms positions[j]! rows1[j]! (salt1.getD j ByteArray.empty) (sibsOf sib1 (d - c) j) rows2[j]!
    (salt2.getD j ByteArray.empty) (sibsOf sib2 (d - c) j)

/-- **Merkle binding (L3-M).** Two Merkle checks the verifier accepts against one cap at the same positions, whose opened
rows differ at query `j`: `merklePair` computes a colliding pair of the scheme's hash from them, for a scheme whose digests
have its width (`MerkleScheme.all_sized`: every scheme the verifier knows). -/
theorem merkle_binding (ms : MerkleScheme) (hs : ms.Sized) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows1 rows2 : Array (Array F128)) (sib1 sib2 salt1 salt2 : Array ByteArray) (lanes : Nat)
    (h1 : merkleCheck ms cap c d positions rows1 sib1 salt1 lanes = .ok ())
    (h2 : merkleCheck ms cap c d positions rows2 sib2 salt2 lanes = .ok ())
    (j : Nat) (hj : j < positions.size) (hr : rows1[j]! ≠ rows2[j]!) :
    (merklePair ms c d positions rows1 rows2 sib1 sib2 salt1 salt2 j).Collides ms :=
  opens_binding ms hs cap c d lanes positions[j]! _ _ _ _ _ _
    (merkleCheck_sibs ms cap c d positions rows1 sib1 salt1 lanes h1 j)
    (merkleCheck_sibs ms cap c d positions rows2 sib2 salt2 lanes h2 j)
    (merkleCheck_opens ms cap c d positions rows1 sib1 salt1 lanes h1 j hj)
    (merkleCheck_opens ms cap c d positions rows2 sib2 salt2 lanes h2 j hj) hr

/-- The proposition `merkle_binding` implies (satisfiable by counting on its own; `merkle_binding` is the reduction). -/
theorem merkle_binding_collision (ms : MerkleScheme) (hs : ms.Sized) (cap : Array ByteArray) (c d : Nat)
    (positions : Array Nat) (rows1 rows2 : Array (Array F128)) (sib1 sib2 salt1 salt2 : Array ByteArray) (lanes : Nat)
    (h1 : merkleCheck ms cap c d positions rows1 sib1 salt1 lanes = .ok ())
    (h2 : merkleCheck ms cap c d positions rows2 sib2 salt2 lanes = .ok ())
    (j : Nat) (hj : j < positions.size) (hr : rows1[j]! ≠ rows2[j]!) : ms.HasCollision :=
  ⟨_, merkle_binding ms hs cap c d positions rows1 rows2 sib1 sib2 salt1 salt2 lanes h1 h2 j hj hr⟩

/-! ### From a colliding pair to the hash's own inputs, for each scheme the verifier knows -/

/-- Two different pairs of `w`-byte children are two different concatenations. -/
theorem append_ne_of_pair_ne {a b a' b' : ByteArray} {w : Nat} (ha : a.size = w) (ha' : a'.size = w)
    (hne : (a, b) ≠ (a', b')) : a ++ b ≠ a' ++ b' := by
  intro h
  have h1 := ByteArray.append_inj_left h (by rw [ha, ha'])
  subst h1
  exact hne (by rw [(ByteArray.append_right_inj _).mp h])

/-- **An unsalted scheme's inputs:** a node's two concatenated children, or a leaf's two row encodings. -/
def MerklePair.shaInputs : MerklePair → ByteArray × ByteArray
  | .node a b a' b' => (a ++ b, a' ++ b')
  | .leaf r r' _ _ => (MerkleScheme.rowBytes r, MerkleScheme.rowBytes r')

theorem MerklePair.sha256_inputs (id : String) (p : MerklePair) (h : p.Collides (MerkleScheme.sha256 id)) :
    p.shaInputs.1 ≠ p.shaInputs.2 ∧ Sha256.hash p.shaInputs.1 = Sha256.hash p.shaInputs.2 := by
  cases p with
  | node a b a' b' => exact ⟨append_ne_of_pair_ne h.1 h.2.2.1 h.2.2.2.2.1, h.2.2.2.2.2⟩
  | leaf r r' y y' => exact ⟨fun hh => h.1 (rowBytes_inj hh), h.2⟩

theorem MerklePair.sha512_inputs (id : String) (p : MerklePair) (h : p.Collides (MerkleScheme.sha512 id)) :
    p.shaInputs.1 ≠ p.shaInputs.2 ∧ Sha512.hash p.shaInputs.1 = Sha512.hash p.shaInputs.2 := by
  cases p with
  | node a b a' b' => exact ⟨append_ne_of_pair_ne h.1 h.2.2.1 h.2.2.2.2.1, h.2.2.2.2.2⟩
  | leaf r r' y y' => exact ⟨fun hh => h.1 (rowBytes_inj hh), h.2⟩

/-- **`hm96-sha512/v1`'s inputs:** a node's two concatenated children; for a leaf, the two row encodings when their row
hashes are one, else `Hm96.leafPair`'s (the outer inputs, or the salt digest's). -/
def MerklePair.hm96Inputs : MerklePair → ByteArray × ByteArray
  | .node a b a' b' => (a ++ b, a' ++ b')
  | .leaf r r' y y' =>
    let x := Sha512.hash (MerkleScheme.rowBytes r)
    let x' := Sha512.hash (MerkleScheme.rowBytes r')
    if x = x' then (MerkleScheme.rowBytes r, MerkleScheme.rowBytes r')
    else Hm96.leafPair Hm96.Default512.leafPrefix Hm96.Default512.saltPrefix Hm96.Default512.mask x x' y y'

theorem MerklePair.hm96_inputs (p : MerklePair) (h : p.Collides MerkleScheme.hm96Sha512) :
    p.hm96Inputs.1 ≠ p.hm96Inputs.2 ∧ Sha512.hash p.hm96Inputs.1 = Sha512.hash p.hm96Inputs.2 := by
  cases p with
  | node a b a' b' => exact ⟨append_ne_of_pair_ne h.1 h.2.2.1 h.2.2.2.2.1, h.2.2.2.2.2⟩
  | leaf r r' y y' =>
    simp only [MerklePair.hm96Inputs]
    split
    · rename_i hx
      exact ⟨fun hh => h.1 (rowBytes_inj hh), hx⟩
    · rename_i hx
      exact Hm96.leafPair_collides _ _ _ _ _ y y' (Sha512.hash_size _) (Sha512.hash_size _)
        (fun hh => hx (Prod.mk.inj hh).1) h.2

theorem MerkleScheme.sha256_collision (id : String) (h : (MerkleScheme.sha256 id).HasCollision) : Sha256.HasCollision :=
  let ⟨p, hp⟩ := h
  ⟨_, _, (p.sha256_inputs id hp).1, (p.sha256_inputs id hp).2⟩

theorem MerkleScheme.sha512_collision (id : String) (h : (MerkleScheme.sha512 id).HasCollision) : Sha512.HasCollision :=
  let ⟨p, hp⟩ := h
  ⟨_, _, (p.sha512_inputs id hp).1, (p.sha512_inputs id hp).2⟩

theorem MerkleScheme.hm96Sha512_collision (h : MerkleScheme.hm96Sha512.HasCollision) : Sha512.HasCollision :=
  let ⟨p, hp⟩ := h
  ⟨_, _, (p.hm96_inputs hp).1, (p.hm96_inputs hp).2⟩

/-- A collision of any scheme the verifier knows is a SHA-256 or SHA-512 collision (the corollary of the extractors
`MerklePair.shaInputs` and `MerklePair.hm96Inputs`). -/
theorem MerkleScheme.collision_hash : ∀ ms ∈ MerkleScheme.all, ms.HasCollision → Sha256.HasCollision ∨ Sha512.HasCollision := by
  intro ms h hc
  simp only [MerkleScheme.all, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl
  · exact Or.inl (MerkleScheme.sha256_collision _ hc)
  · exact Or.inl (MerkleScheme.sha256_collision _ hc)
  · exact Or.inr (MerkleScheme.sha512_collision _ hc)
  · exact Or.inr (MerkleScheme.hm96Sha512_collision hc)

end Flock
