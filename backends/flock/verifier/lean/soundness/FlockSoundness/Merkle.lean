import Mathlib.Data.List.Basic
import Mathlib.Logic.Function.Basic
import FlockSoundness.Protocol.Merkle

/-!
# Merkle trees (PROTOCOL.md §14) and their binding

A level's commitment is a Merkle cap: the nodes at depth `c` of a binary tree over the columns.
Leaf `q` is `H(encode(column q))`; a node is `H(left ‖ right)`; an opening of position `pos` is the
column and the `d − c` siblings from the leaf up to the cap entry `pos >> (d − c)`. The hash is
one function on byte strings (SHA-512 in the target, `ASSUMPTIONS.md`), so every collision
below is a pair of distinct byte strings with the same hash.

**Binding** (`opening_binding`): two openings of one position that both verify against the same
cap but carry different columns yield an explicit collision of `H`. This is the step both
reductions to collision resistance use: the straight-line one (a prover that knows its table)
and the rewinding one (every prover, `Rewinding`). The development states it for our tree shape
directly; VCVio proves the same fact for its inductive trees
(`InductiveMerkleTree.getPutativeRootWithHash_binding_collision`).

**Salted leaves.** A leaf scheme (`Leaf`) gives each column a leaf under a salt, and binds it: two columns with one
leaf give a collision. `VerifiesL` and `opening_bindingL` are `Verifies` and `opening_binding` for any leaf scheme,
and `Leaf.plain` is the unsalted leaf above. hm96's leaf (`Leaf.hm96`, `hm96Leaf_bind`) binds its column with no
assumption beyond `H`'s collisions.
-/

namespace FlockSoundness.Merkle

variable {Col D : Type}

variable (H : List UInt8 → D) (E : Enc Col D)

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

/-! ## Leaf schemes: salted leaves (hm96) -/

/-- The unsalted leaf, `H(encode column)`. -/
def Leaf.plain : Leaf Col Unit D H where
  leaf c _ := H (E.col c)
  bind _ _ _ _ hc h := ⟨_, _, fun e => hc (E.col_inj e), h⟩

variable {S : Type}

theorem verifiesL_plain (cap : ℕ → D) (depth pos : ℕ) (column : Col) (sibs : List D) :
    VerifiesL H E (Leaf.plain H E) cap depth pos column () sibs ↔ Verifies H E cap depth pos column sibs :=
  Iff.rfl

/-- **Merkle binding, for any leaf scheme.** Two openings of one position that verify against one cap with different
columns, under any salts, yield a collision of `H`. -/
theorem opening_bindingL (Lf : Leaf Col S D H) (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S)
    (sibs sibs' : List D) (hc : c ≠ c') (h : VerifiesL H E Lf cap depth pos c s sibs)
    (h' : VerifiesL H E Lf cap depth pos c' s' sibs') : Collision H := by
  obtain ⟨hl, hr⟩ := h
  obtain ⟨hl', hr'⟩ := h'
  have heq : reachL H E Lf pos c s sibs = reachL H E Lf pos c' s' sibs' := hr.trans hr'.symm
  by_cases hleaf : Lf.leaf c s = Lf.leaf c' s'
  · exact Lf.bind c c' s s' hc hleaf
  · exact walk_binding H E sibs sibs' pos _ _ (hl.trans hl'.symm) hleaf heq

end FlockSoundness.Merkle
