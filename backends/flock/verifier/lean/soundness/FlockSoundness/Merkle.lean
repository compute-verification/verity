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

end FlockSoundness.Merkle
