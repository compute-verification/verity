module

public import Proofs.Flock.Soundness.MerklePair
public import Mathlib.Tactic.Basic
import Mathlib.Tactic.Push
import Mathlib.Tactic.SplitIfs

/-!
# Trial: a leaf scheme that gives its pair

A scratch, not to merge. `LeafP` is `Merkle.Leaf` restated: in place of `bind`, which concludes that `H` has a
collision, it holds `pair`, the two byte strings, and `pair_spec`, that they collide. `LeafP.plain` and `LeafP.hm96`
are the two schemes of `Definitions.Flock.Merkle` in that shape, and the three base lemmas are corollaries of the
pairs' specs.
-/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}

/-- **A leaf scheme with its pair**: a column's leaf under a salt, and for two columns and salts the two byte strings
that collide when the columns differ and the leaves agree. -/
structure LeafP (Col S D : Type) (H : List UInt8 → D) where
  leaf : Col → S → D
  pair : Col → S → Col → S → List UInt8 × List UInt8
  pair_spec : ∀ c c' s s', c ≠ c' → leaf c s = leaf c' s' → Hash.Collision H (pair c s c' s').1 (pair c s c' s').2

/-- The scheme with its pair forgotten: today's `Leaf`. -/
def LeafP.toLeaf {S : Type} {H : List UInt8 → D} (L : LeafP Col S D H) : Leaf Col S D H where
  leaf := L.leaf
  bind c c' s s' hc h := ⟨_, _, L.pair_spec c c' s s' hc h⟩

variable (H : List UInt8 → D) (E : Enc Col D)

/-- The unsalted leaf `H (encode column)`, with the two columns' encodings as its pair. -/
def LeafP.plain : LeafP Col Unit D H where
  leaf c _ := H (E.col c)
  pair c _ c' _ := (E.col c, E.col c')
  pair_spec _ _ _ _ hc h := ⟨fun e => hc (E.col_inj e), h⟩

theorem LeafP.plain_toLeaf : (LeafP.plain H E).toLeaf = Leaf.plain H E := rfl

/-- The outer input of hm96's leaf: `leaf_prefix ‖ b ‖ c`. -/
def hm96Input (K : Hm96Key) (c : Col) (y : List UInt8) : List UInt8 :=
  K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y)))

/-- **hm96's pair**: the outer inputs when they differ, else the salt digest's inputs when the salts differ, else the
two columns' encodings. -/
def hm96Pair (K : Hm96Key) (c : Col) (y : List UInt8) (c' : Col) (y' : List UInt8) : List UInt8 × List UInt8 :=
  if hm96Input H E K c y = hm96Input H E K c' y' then
    if y = y' then (E.col c, E.col c') else (K.saltPrefix ++ y, K.saltPrefix ++ y')
  else (hm96Input H E K c y, hm96Input H E K c' y')

/-- **hm96's pair is a collision** whenever two different columns have one leaf, under any salts: `hm96Leaf_bind` with
its pair named. -/
theorem hm96Pair_spec (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) (c c' : Col) (y y' : List UInt8)
    (hc : c ≠ c') (h : hm96Leaf (H := H) (E := E) K c y = hm96Leaf (H := H) (E := E) K c' y') :
    Hash.Collision H (hm96Pair H E K c y c' y').1 (hm96Pair H E K c y c' y').2 := by
  have hbl : (xorBytes (E.dig (H (E.col c))) (K.mask y)).length =
      (xorBytes (E.dig (H (E.col c'))) (K.mask y')).length := by
    simp only [xorBytes, List.length_zipWith, hL, K.mask_len]
  unfold hm96Pair
  split_ifs with ho hy
  · have ho' : K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y))) =
        K.leafPrefix ++ (xorBytes (E.dig (H (E.col c'))) (K.mask y') ++ E.dig (H (K.saltPrefix ++ y'))) := ho
    obtain ⟨hb, -⟩ := List.append_inj (List.append_cancel_left ho') hbl
    rw [← hy] at hb
    have hx := xorBytes_inj (by rw [hL, K.mask_len]) (by rw [hL, K.mask_len]) hb
    exact ⟨fun e => hc (E.col_inj e), E.dig_inj hx⟩
  · have ho' : K.leafPrefix ++ (xorBytes (E.dig (H (E.col c))) (K.mask y) ++ E.dig (H (K.saltPrefix ++ y))) =
        K.leafPrefix ++ (xorBytes (E.dig (H (E.col c'))) (K.mask y') ++ E.dig (H (K.saltPrefix ++ y'))) := ho
    obtain ⟨-, hs⟩ := List.append_inj (List.append_cancel_left ho') hbl
    exact ⟨fun e => hy (List.append_cancel_left e), E.dig_inj hs⟩
  · exact ⟨ho, h⟩

/-- **The `hm96-sha512/v1` leaf with its pair.** -/
def LeafP.hm96 (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) : LeafP Col (List UInt8) D H where
  leaf := hm96Leaf (H := H) (E := E) K
  pair := hm96Pair H E K
  pair_spec c c' y y' hc h := hm96Pair_spec H E K hL c c' y y' hc h

theorem LeafP.hm96_toLeaf (K : Hm96Key) (hL : ∀ d, (E.dig d).length = K.L) :
    (LeafP.hm96 H E K hL).toLeaf = Leaf.hm96 H E K hL := rfl

/-! ## The three base lemmas, as corollaries -/

theorem walk_bindingP (sibs sibs' : List D) (idx : ℕ) (a a' : D) (hlen : sibs.length = sibs'.length) (hne : a ≠ a')
    (heq : walk H E idx a sibs = walk H E idx a' sibs') : Collision H :=
  ⟨_, _, walkPair_spec H E sibs sibs' idx a a' hlen hne heq⟩

theorem opening_bindingP (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (sibs sibs' : List D) (hc : c ≠ c')
    (h : Verifies H E cap depth pos c sibs) (h' : Verifies H E cap depth pos c' sibs') : Collision H :=
  ⟨_, _, openPair_spec H E cap depth pos c c' sibs sibs' hc h h'⟩

variable {S : Type}

theorem opening_bindingLP (Lf : LeafP Col S D H) (cap : ℕ → D) (depth pos : ℕ) (c c' : Col) (s s' : S)
    (sibs sibs' : List D) (hc : c ≠ c') (h : VerifiesL H E Lf.toLeaf cap depth pos c s sibs)
    (h' : VerifiesL H E Lf.toLeaf cap depth pos c' s' sibs') : Collision H :=
  ⟨_, _, openPairL_spec H E Lf.toLeaf Lf.pair Lf.pair_spec cap depth pos c c' s s' sibs sibs' hc h h'⟩

end FlockSoundness.Merkle
