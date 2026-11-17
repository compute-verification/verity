module

public import Definitions.Hash.Basic
public import Mathlib.Tactic.Attr.Core
import Mathlib.Tactic.Push

/-! Merkle caps and their openings (`Flock.Merkle`), and leaf schemes that bind a column under a salt (`Leaf`). -/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}

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

/-- The bytes one step up the tree hashes: `acc ‖ sib` at an even index, `sib ‖ acc` at an odd one. -/
def upIn (idx : ℕ) (acc sib : D) : List UInt8 :=
  if idx % 2 = 0 then E.dig acc ++ E.dig sib else E.dig sib ++ E.dig acc

/-- One step up the tree: `H` of the step's input (`upIn`). -/
def up (idx : ℕ) (acc sib : D) : D := H (upIn E idx acc sib)

/-- Walk from a node up through the siblings (`Flock.Merkle`). -/
def walk : ℕ → D → List D → D
  | _, acc, [] => acc
  | idx, acc, sib :: rest => walk (idx / 2) (up H E idx acc sib) rest

def reach (pos : ℕ) (column : Col) (sibs : List D) : D := walk H E pos (H (E.col column)) sibs

def Verifies (cap : ℕ → D) (depth : ℕ) (pos : ℕ) (column : Col) (sibs : List D) : Prop :=
  sibs.length = depth ∧ reach H E pos column sibs = cap (pos / 2 ^ depth)

/-- **A leaf scheme**: a column's leaf under a salt of type `S`, and for two columns and salts the two byte strings
`pair` that collide under `H` when the columns differ and the leaves agree. The pair is a function of the two columns
and salts alone, so a collision finder can output it. The unsalted leaf `H(encode column)` is `Leaf.plain`, with
`S = Unit`. -/
structure Leaf (Col S D : Type) (H : List UInt8 → D) where
  leaf : Col → S → D
  pair : Col → S → Col → S → List UInt8 × List UInt8
  pair_spec : ∀ c c' s s', c ≠ c' → leaf c s = leaf c' s' → Hash.Collision H (pair c s c' s').1 (pair c s c' s').2

/-- The unsalted leaf, `H(encode column)`. Its pair is the two columns' encodings. -/
def Leaf.plain : Leaf Col Unit D H where
  leaf c _ := H (E.col c)
  pair c _ c' _ := (E.col c, E.col c')
  pair_spec _ _ _ _ hc h := ⟨fun e => hc (E.col_inj e), h⟩

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

/-- The input of the outer hash of hm96's leaf: `LP ‖ b ‖ c`. -/
def hm96LeafIn (K : Hm96Key) (c : Col) (y : List UInt8) : List UInt8 :=
  K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y)))

/-- **The pair of hm96's leaf**, for two columns and salts: the two outer inputs when they differ; else the two salt
hashes' inputs when the salts differ; else the two columns' encodings. -/
def hm96LeafPair (K : Hm96Key) (c : Col) (y : List UInt8) (c' : Col) (y' : List UInt8) : List UInt8 × List UInt8 :=
  if hm96LeafIn H E K c y = hm96LeafIn H E K c' y' then
    if y = y' then (E.col c, E.col c') else (K.saltPrefix ++ y, K.saltPrefix ++ y')
  else (hm96LeafIn H E K c y, hm96LeafIn H E K c' y')

/-- hm96's leaf binds its column with no assumption beyond `H`'s collisions. It uses only the fixed lengths, and that
`x ↦ x ⊕ M·y` is injective for a fixed `y`. -/
theorem hm96LeafPair_spec (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) (c c' : Col) (y y' : List UInt8)
    (hc : c ≠ c') (h : hm96Leaf (H := H) (E := E) K c y = hm96Leaf (H := H) (E := E) K c' y') :
    Hash.Collision H (hm96LeafPair H E K c y c' y').1 (hm96LeafPair H E K c y c' y').2 := by
  have hbl : (xorBytes (E.dig (H (E.col c))) (K.mask y)).length =
      (xorBytes (E.dig (H (E.col c'))) (K.mask y')).length := by
    simp only [xorBytes, List.length_zipWith, hL, K.mask_len]
  unfold hm96LeafPair
  by_cases ho : hm96LeafIn H E K c y = hm96LeafIn H E K c' y'
  · have ho' : K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y))) =
        K.leafPrefix ++ (xorBytes (E.dig (H (E.col c'))) (K.mask y') ++ E.dig (H (K.saltPrefix ++ y'))) := ho
    obtain ⟨hb, hs⟩ := List.append_inj (List.append_cancel_left ho') hbl
    rw [ite_eq_left ho]
    by_cases hy : y = y'
    · rw [ite_eq_left hy]
      rw [← hy] at hb
      have hx := xorBytes_inj (by rw [hL, K.mask_len]) (by rw [hL, K.mask_len]) hb
      exact ⟨fun e => hc (E.col_inj e), E.dig_inj hx⟩
    · rw [ite_eq_right hy]
      exact ⟨fun e => hy (List.append_cancel_left e), E.dig_inj hs⟩
  · rw [ite_eq_right ho]
    exact ⟨ho, h⟩

/-- hm96's salted leaf scheme (`hm96Leaf`), for digests of the mask's length. -/
def Leaf.hm96 (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) : Leaf Col (List UInt8) D H where
  leaf := hm96Leaf (H := H) (E := E) K
  pair := hm96LeafPair H E K
  pair_spec := hm96LeafPair_spec H E K hL

variable {S : Type}

/-- The cap entry an opening of `pos` reaches, starting from the leaf of `column` under `salt`. -/
def reachL (Lf : Leaf Col S D H) (pos : ℕ) (column : Col) (salt : S) (sibs : List D) : D :=
  walk H E pos (Lf.leaf column salt) sibs

/-- An opening verifies under a leaf scheme: from its column's leaf under its salt, it reaches the cap entry
`pos >> (d − c)`. -/
def VerifiesL (Lf : Leaf Col S D H) (cap : ℕ → D) (depth pos : ℕ) (column : Col) (salt : S) (sibs : List D) : Prop :=
  sibs.length = depth ∧ reachL H E Lf pos column salt sibs = cap (pos / 2 ^ depth)

end FlockSoundness.Merkle
