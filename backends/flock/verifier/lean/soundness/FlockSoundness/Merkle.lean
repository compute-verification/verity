import Mathlib.Data.List.Basic
import Mathlib.Logic.Function.Basic

/-!
# Merkle trees (PROTOCOL.md §14) and their binding

A level's commitment is a Merkle cap: the nodes at depth `c` of a binary tree over the columns.
Leaf `q` is `H(encode(column q))`; a node is `H(left ‖ right)`; an opening of position `pos` is the
column and the `d − c` siblings from the leaf up to the cap entry `pos >> (d − c)`. The hash is
one function on byte strings (SHA-512 in the target, §4.3 of `ASSUMPTIONS.md`), so every collision
below is a pair of distinct byte strings with the same hash.

**Binding** (`opening_binding`): two openings of one position that both verify against the same
cap but carry different columns yield an explicit collision of `H`. This is the step both
reductions to collision resistance use: the straight-line one (a prover that knows its table)
and the rewinding one (every prover, `Rewinding`). The development states it for our tree shape
directly; VCVio proves the same fact for its inductive trees
(`InductiveMerkleTree.getPutativeRootWithHash_binding_collision`).
-/

namespace FlockSoundness.Merkle

variable {Col D : Type}

/-- A collision of a hash on byte strings. -/
def Collision (H : List UInt8 → D) : Prop := ∃ x y : List UInt8, x ≠ y ∧ H x = H y

/-- The byte encodings a tree uses: a column (its elements, 16 bytes each) and a digest. Both are
injective, and all digests have one length, so a node's input determines its two children. -/
structure Enc (Col D : Type) where
  col : Col → List UInt8
  dig : D → List UInt8
  col_inj : Function.Injective col
  dig_inj : Function.Injective dig
  dig_len : ∀ a b : D, (dig a).length = (dig b).length

variable (H : List UInt8 → D) (E : Enc Col D)

/-- One step up the tree: `H(acc ‖ sib)` at an even index, `H(sib ‖ acc)` at an odd one. -/
def up (idx : ℕ) (acc sib : D) : D :=
  if idx % 2 = 0 then H (E.dig acc ++ E.dig sib) else H (E.dig sib ++ E.dig acc)

/-- Walk from a node up through the siblings (§14). -/
def walk : ℕ → D → List D → D
  | _, acc, [] => acc
  | idx, acc, sib :: rest => walk (idx / 2) (up H E idx acc sib) rest

/-- The cap entry an opening of `pos` reaches, starting from the leaf `H(encode column)`. -/
def reach (pos : ℕ) (column : Col) (sibs : List D) : D := walk H E pos (H (E.col column)) sibs

/-- An opening verifies: it reaches the cap entry `pos >> (d − c)` (§14). -/
def Verifies (cap : ℕ → D) (depth : ℕ) (pos : ℕ) (column : Col) (sibs : List D) : Prop :=
  sibs.length = depth ∧ reach H E pos column sibs = cap (pos / 2 ^ depth)

theorem append_inj {a b a' b' : D} (h : E.dig a ++ E.dig b = E.dig a' ++ E.dig b') :
    a = a' ∧ b = b' := by
  have hl : (E.dig a).length = (E.dig a').length := E.dig_len a a'
  obtain ⟨h1, h2⟩ := List.append_inj h hl
  exact ⟨E.dig_inj h1, E.dig_inj h2⟩

/-- Two walks from different nodes that meet give a collision. -/
theorem walk_binding : ∀ (sibs sibs' : List D) (idx : ℕ) (a a' : D),
    sibs.length = sibs'.length → a ≠ a' → walk H E idx a sibs = walk H E idx a' sibs' →
      Collision H
  | [], [], _, a, a', _, hne, heq => absurd heq hne
  | [], _ :: _, _, _, _, hlen, _, _ => by simp at hlen
  | _ :: _, [], _, _, _, hlen, _, _ => by simp at hlen
  | s :: rest, s' :: rest', idx, a, a', hlen, hne, heq => by
      simp only [walk] at heq
      by_cases hup : up H E idx a s = up H E idx a' s'
      · -- the step itself collides: different inputs (the accumulators differ), equal hashes
        unfold up at hup
        split_ifs at hup with he
        · refine ⟨_, _, fun h => hne (append_inj E h).1, hup⟩
        · refine ⟨_, _, fun h => hne (append_inj E h).2, hup⟩
      · exact walk_binding rest rest' (idx / 2) _ _ (by simpa using hlen) hup heq

/-- **Merkle binding.** Two openings of one position that verify against one cap with different
columns yield a collision of `H`. -/
theorem opening_binding (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D)
    (hc : c ≠ c') (h : Verifies H E cap depth pos c sibs) (h' : Verifies H E cap depth pos c' sibs') :
    Collision H := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reach H E pos c sibs = reach H E pos c' sibs' := hr.trans hr'.symm
  by_cases hleaf : H (E.col c) = H (E.col c')
  · exact ⟨_, _, fun h => hc (E.col_inj h), hleaf⟩
  · exact walk_binding H E sibs sibs' pos _ _ (hl.trans hl'.symm) hleaf heq

end FlockSoundness.Merkle
