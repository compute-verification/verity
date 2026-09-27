/-
Level 3, L3-M (PROTOCOL.md §14, §17.4 stage 1b): Merkle binding. Two openings the verifier accepts for one position of one
cap, with different rows, give an explicit collision of the scheme's hash: two different node inputs, or two different rows,
with one digest. Parametric in the scheme (SHA-256 or SHA-512 trees alike); no random oracle.
-/
import Flock.Merkle

namespace Flock

/-- A collision of a Merkle scheme's hash. -/
def MerkleScheme.Collision (ms : MerkleScheme) : Prop :=
  (∃ a b a' b' : ByteArray, (a, b) ≠ (a', b') ∧ ms.node a b = ms.node a' b') ∨
  (∃ r r' : Array F128, r ≠ r' ∧ ms.leaf r = ms.leaf r')

/-- Climbing from two different digests to one digest along paths of one length is a node collision. -/
theorem climb_binding (ms : MerkleScheme) :
    ∀ (s1 s2 : List ByteArray) (a b : ByteArray) (idx : Nat), s1.length = s2.length → a ≠ b →
      climb ms a idx s1 = climb ms b idx s2 → ms.Collision := by
  intro s1
  induction s1 with
  | nil =>
    intro s2 a b idx hl hab h
    cases s2 with
    | nil => exact absurd h hab
    | cons _ _ => simp at hl
  | cons s rest ih =>
    intro s2 a b idx hl hab h
    cases s2 with
    | nil => simp at hl
    | cons t rest2 =>
      have hl' : rest.length = rest2.length := by simpa using hl
      by_cases hp : idx % 2 = 0
      · have h' : climb ms (ms.node a s) (idx / 2) rest = climb ms (ms.node b t) (idx / 2) rest2 := by
          simpa [climb, hp] using h
        by_cases he : ms.node a s = ms.node b t
        · exact Or.inl ⟨a, s, b, t, fun hh => hab (Prod.mk.inj hh).1, he⟩
        · exact ih rest2 _ _ (idx / 2) hl' he h'
      · have h' : climb ms (ms.node s a) (idx / 2) rest = climb ms (ms.node t b) (idx / 2) rest2 := by
          simpa [climb, hp] using h
        by_cases he : ms.node s a = ms.node t b
        · exact Or.inl ⟨s, a, t, b, fun hh => hab (Prod.mk.inj hh).2, he⟩
        · exact ih rest2 _ _ (idx / 2) hl' he h'

/-- Two accepted openings of one position against one cap, with different rows, are a collision. -/
theorem opens_binding (ms : MerkleScheme) (cap : Array ByteArray) (c d lanes pos : Nat) (r1 r2 : Array F128)
    (s1 s2 : List ByteArray) (h1 : opens ms cap c d lanes pos r1 s1 = true) (h2 : opens ms cap c d lanes pos r2 s2 = true)
    (hr : r1 ≠ r2) : ms.Collision := by
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h1 h2
  have heq : climb ms (ms.leaf r1) pos s1 = climb ms (ms.leaf r2) pos s2 := h1.2.trans h2.2.symm
  by_cases hl : ms.leaf r1 = ms.leaf r2
  · exact Or.inr ⟨r1, r2, hr, hl⟩
  · exact climb_binding ms s1 s2 _ _ pos (by rw [h1.1.2, h2.1.2]) hl heq

/-- An accepted Merkle check opens every query. -/
theorem merkleCheck_opens (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows : Array (Array F128)) (siblings : Array ByteArray) (lanes : Nat)
    (h : merkleCheck ms cap c d positions rows siblings lanes = .ok ()) (j : Nat) (hj : j < positions.size) :
    opens ms cap c d lanes positions[j]! rows[j]! (sibsOf siblings (d - c) j) = true := by
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

/-- **Merkle binding (L3-M).** Two Merkle checks the verifier accepts against one cap at the same positions, whose opened
rows differ at some query, give a collision of the scheme's hash. -/
theorem merkle_binding (ms : MerkleScheme) (cap : Array ByteArray) (c d : Nat) (positions : Array Nat)
    (rows1 rows2 : Array (Array F128)) (sib1 sib2 : Array ByteArray) (lanes : Nat)
    (h1 : merkleCheck ms cap c d positions rows1 sib1 lanes = .ok ())
    (h2 : merkleCheck ms cap c d positions rows2 sib2 lanes = .ok ())
    (j : Nat) (hj : j < positions.size) (hr : rows1[j]! ≠ rows2[j]!) : ms.Collision :=
  opens_binding ms cap c d lanes positions[j]! _ _ _ _
    (merkleCheck_opens ms cap c d positions rows1 sib1 lanes h1 j hj)
    (merkleCheck_opens ms cap c d positions rows2 sib2 lanes h2 j hj) hr

end Flock
