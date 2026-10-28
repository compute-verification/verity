import Mathlib.Data.List.Basic
import Mathlib.Logic.Function.Basic
import Definitions.Hash.Basic

/-!
# Merkle trees (PROTOCOL.md §14)

A level's commitment is a Merkle cap: the nodes at depth `c` of a binary tree over the columns.
Leaf `q` is `H(encode(column q))`; a node is `H(left ‖ right)`; an opening of position `pos` is the
column and the `d − c` siblings from the leaf up to the cap entry `pos >> (d − c)`. The hash is
one function on byte strings (SHA-512 in the target, `ASSUMPTIONS.md`), so every collision
is a pair of distinct byte strings with the same hash (`Hash.HasCollision`).

**Salted leaves.** A leaf scheme (`Leaf`) gives each column a leaf under a salt, and binds it: two columns with one
leaf give a collision. `VerifiesL` is `Verifies` for any leaf scheme, and `Leaf.plain` is the unsalted leaf above.
hm96's leaf (`Leaf.hm96`) binds its column with no assumption beyond `H`'s collisions (`hm96Leaf_bind`, its proof
field). The binding of a whole opening is in `Proofs.Flock.Soundness.Merkle`.
-/

namespace FlockSoundness.Merkle

variable {Col D : Type}

/-- A collision of a hash on byte strings: `Hash.HasCollision`. -/
abbrev Collision (H : List UInt8 → D) : Prop := Hash.HasCollision H

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

/-! ## Leaf schemes: salted leaves (hm96) -/

/-- **A leaf scheme**: a column's leaf under a salt of type `S`, such that two columns with one leaf, under any salts,
give a collision of `H`. The unsalted leaf `H(encode column)` is `Leaf.plain`, with `S = Unit`. -/
structure Leaf (Col S D : Type) (H : List UInt8 → D) where
  leaf : Col → S → D
  bind : ∀ c c' s s', c ≠ c' → leaf c s = leaf c' s' → Collision H

/-- The unsalted leaf, `H(encode column)`. -/
def Leaf.plain : Leaf Col Unit D H where
  leaf c _ := H (E.col c)
  bind _ _ _ _ hc h := ⟨_, _, fun e => hc (E.col_inj e), h⟩

/-- hm96's leaf parameters: the leaf and salt prefixes, and a salt's mask `y ↦ M·y`, `L` bytes long. -/
structure Hm96Key where
  leafPrefix : List UInt8
  saltPrefix : List UInt8
  mask : List UInt8 → List UInt8
  L : ℕ
  mask_len : ∀ y, (mask y).length = L

/-- Bytewise xor. -/
def xorBytes (a b : List UInt8) : List UInt8 := List.zipWith (· ^^^ ·) a b

theorem xorBytes_inj : ∀ {a a' m : List UInt8}, a.length = m.length → a'.length = m.length →
    xorBytes a m = xorBytes a' m → a = a'
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, h, h', _ => by simp at h; rw [← h] at h'; simp at h'
  | _ :: _, [], _, h, h', _ => by simp at h'; rw [← h'] at h; simp at h
  | x :: as, x' :: as', [], h, _, _ => by simp at h
  | x :: as, x' :: as', k :: m, h, h', he => by
    simp only [xorBytes, List.zipWith_cons_cons, List.cons.injEq] at he
    rw [(UInt8.xor_left_inj k).1 he.1, xorBytes_inj (by simpa using h) (by simpa using h') he.2]

variable {H E} in
/-- hm96's leaf of column `c` under salt `y`: `H(LP ‖ b ‖ c)`, with `x = H(encode column)`, `b = x ⊕ M·y` and
`c = H(SP ‖ y)`. -/
def hm96Leaf (K : Hm96Key) (c : Col) (y : List UInt8) : D :=
  H (K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y))))

/-- **hm96's leaf binds its column**, with no assumption beyond `H`'s collisions: equal leaves collide in the outer hash
when `b ‖ c` differ, in the salt hash when the salts differ, and in the inner hash when the salts agree (then the masks
agree, so the `x`s do). It uses only the fixed lengths, and that `x ↦ x ⊕ M·y` is injective for a fixed `y`. -/
theorem hm96Leaf_bind (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) (c c' : Col) (y y' : List UInt8)
    (hc : c ≠ c') (h : hm96Leaf (H := H) (E := E) K c y = hm96Leaf (H := H) (E := E) K c' y') : Collision H := by
  by_cases ho : K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y))) =
      K.leafPrefix ++ (xorBytes (E.dig (H (E.col c'))) (K.mask y') ++ E.dig (H (K.saltPrefix ++ y')))
  · have hbl : (xorBytes (E.dig (H (E.col c))) (K.mask y)).length =
        (xorBytes (E.dig (H (E.col c'))) (K.mask y')).length := by
      simp only [xorBytes, List.length_zipWith, hL, K.mask_len]
    obtain ⟨hb, hs⟩ := List.append_inj (List.append_cancel_left ho) hbl
    by_cases hy : y = y'
    · subst hy
      have hx := xorBytes_inj (by rw [hL, K.mask_len]) (by rw [hL, K.mask_len]) hb
      exact ⟨_, _, fun e => hc (E.col_inj e), E.dig_inj hx⟩
    · exact ⟨_, _, fun e => hy (List.append_cancel_left e), E.dig_inj hs⟩
  · exact ⟨_, _, ho, h⟩

/-- hm96's salted leaf scheme (`hm96Leaf`), for digests of the mask's length. -/
def Leaf.hm96 (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) : Leaf Col (List UInt8) D H where
  leaf := hm96Leaf (H := H) (E := E) K
  bind := hm96Leaf_bind H E K hL

variable {S : Type}

/-- The cap entry an opening of `pos` reaches, starting from the leaf of `column` under `salt`. -/
def reachL (Lf : Leaf Col S D H) (pos : ℕ) (column : Col) (salt : S) (sibs : List D) : D :=
  walk H E pos (Lf.leaf column salt) sibs

/-- An opening verifies under a leaf scheme: from its column's leaf under its salt, it reaches the cap entry
`pos >> (d − c)`. -/
def VerifiesL (Lf : Leaf Col S D H) (cap : ℕ → D) (depth pos : ℕ) (column : Col) (salt : S) (sibs : List D) : Prop :=
  sibs.length = depth ∧ reachL H E Lf pos column salt sibs = cap (pos / 2 ^ depth)

end FlockSoundness.Merkle
