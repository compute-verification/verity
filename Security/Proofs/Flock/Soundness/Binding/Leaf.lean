import Proofs.Flock.Soundness.Refine.Salted

/-!
# Leaf schemes with explicit collision extractors

A commit string's leaf binds its value: two values with one leaf give a SHA-512 collision. `Merkle.Leaf` states this as a
proposition; the link theorem's finder (`LinkFinder`) has to output the collision, so here each leaf scheme comes with the
two colliding inputs, computed from the two openings (`ExplicitLeaf.pair`). The hash is the model's SHA-512 on byte lists,
`Refine.H512`.

* `ExplicitLeaf.hm96`: `hm96-sha512/row/v1`'s leaf `SHA-512(leaf_prefix ‖ b ‖ c)` under the executable's key
  (`Refine.K512`, `Hm96.Default512`), with `x = SHA-512(row)`, `b = x ⊕ M·y` and `c = SHA-512(salt_prefix ‖ y)` for the
  row's bytes and the salt `y`. The extractor (`hm96Pair`) returns the outer inputs when they differ, else the salt
  digest's inputs when the salts differ, else the two rows' bytes, whose digests then agree because the masks cancel. It
  evaluates SHA-512 four times.
* `ExplicitLeaf.plain`: the leaf `SHA-512(enc v)` of a value in the clear, such as a returned output word.

`hm96_treeLeaf` ties `ExplicitLeaf.hm96`'s leaf to the executable's: it is `Hm96.Default512.treeLeaf` of the commit string
`Hm96.Default512.commitString (Sha512.hash row) salt`, the leaf the verifier computes from a public `b ‖ c`.
-/

namespace FlockSoundness.Binding

open Flock Refine Merkle

/-- **A SHA-512 collision**: two different byte strings with one digest. -/
def Collides (x : List UInt8 × List UInt8) : Prop := x.1 ≠ x.2 ∧ H512 x.1 = H512 x.2

/-- **A leaf scheme with an explicit extractor**: a value's leaf under a salt, and for two openings (value and salt), two
byte strings that collide when the values differ and the leaves agree. -/
structure ExplicitLeaf (Val : Type) where
  leaf : Val → List UInt8 → Dig
  pair : Val → List UInt8 → Val → List UInt8 → List UInt8 × List UInt8
  pair_spec : ∀ v y v' y', v ≠ v' → leaf v y = leaf v' y' → Collides (pair v y v' y')

variable {Val : Type}

/-! ## hm96-sha512 -/

/-- The digest's 64 bytes. -/
abbrev digBytes (d : Dig) : List UInt8 := enc512.dig d

/-- **The commit string `b ‖ c`** of a row's bytes `r` under the salt `y`: `b = SHA-512(r) ⊕ M·y`,
`c = SHA-512(salt_prefix ‖ y)`. -/
def commitString (r y : List UInt8) : List UInt8 :=
  xorBytes (digBytes (H512 r)) (K512.mask y) ++ digBytes (H512 (K512.saltPrefix ++ y))

/-- The outer input of hm96's leaf: `leaf_prefix ‖ b ‖ c`. -/
def hm96Input (r y : List UInt8) : List UInt8 := K512.leafPrefix ++ commitString r y

/-- **The extractor for hm96's leaf**, on two rows' bytes and salts: the outer inputs when they differ, else the salt
digest's inputs when the salts differ, else the rows' bytes. -/
def hm96Pair (r y r' y' : List UInt8) : List UInt8 × List UInt8 :=
  if hm96Input r y = hm96Input r' y' then
    if y = y' then (r, r') else (K512.saltPrefix ++ y, K512.saltPrefix ++ y')
  else (hm96Input r y, hm96Input r' y')

theorem commitString_split {r y r' y' : List UInt8} (h : commitString r y = commitString r' y') :
    xorBytes (digBytes (H512 r)) (K512.mask y) = xorBytes (digBytes (H512 r')) (K512.mask y') ∧
      digBytes (H512 (K512.saltPrefix ++ y)) = digBytes (H512 (K512.saltPrefix ++ y')) :=
  List.append_inj h (by simp only [xorBytes, List.length_zipWith, digBytes, hL512, K512.mask_len])

/-- **hm96's extractor outputs a SHA-512 collision** whenever two different rows' bytes have one leaf, under any salts:
equal leaves collide in the outer hash when `b ‖ c` differ, in the salt digest when the salts differ, and in the row hash
when the salts agree, since then the masks agree, so the row digests do. -/
theorem hm96Pair_spec (r y r' y' : List UInt8) (hr : r ≠ r') (h : H512 (hm96Input r y) = H512 (hm96Input r' y')) :
    Collides (hm96Pair r y r' y') := by
  unfold Collides hm96Pair
  split_ifs with ho hy
  · have hcs : K512.leafPrefix ++ commitString r y = K512.leafPrefix ++ commitString r' y' := ho
    have hb := (commitString_split (List.append_cancel_left hcs)).1
    rw [hy] at hb
    have hx := xorBytes_inj (by rw [hL512, K512.mask_len]) (by rw [hL512, K512.mask_len]) hb
    exact ⟨hr, enc512.dig_inj hx⟩
  · have hcs : K512.leafPrefix ++ commitString r y = K512.leafPrefix ++ commitString r' y' := ho
    exact ⟨fun e => hy (List.append_cancel_left e),
      enc512.dig_inj (commitString_split (List.append_cancel_left hcs)).2⟩
  · exact ⟨ho, h⟩

/-- **hm96-sha512/row/v1's leaf**, for rows encoded injectively as the bytes `sha512/row/v1` hashes (`enc`): the leaf of
row `v` under salt `y` is `SHA-512(leaf_prefix ‖ b ‖ c)` of the commit string of `enc v`. -/
def ExplicitLeaf.hm96 (enc : Val → List UInt8) (hinj : Function.Injective enc) : ExplicitLeaf Val where
  leaf v y := H512 (hm96Input (enc v) y)
  pair v y v' y' := hm96Pair (enc v) y (enc v') y'
  pair_spec v y v' y' hv h := hm96Pair_spec (enc v) y (enc v') y' (fun e => hv (hinj e)) h

/-- **A plain leaf** `SHA-512(enc v)`, which ignores the salt: a value in the clear. -/
def ExplicitLeaf.plain (enc : Val → List UInt8) (hinj : Function.Injective enc) : ExplicitLeaf Val where
  leaf v _ := H512 (enc v)
  pair v _ v' _ := (enc v, enc v')
  pair_spec _ _ _ _ hv h := ⟨fun e => hv (hinj e), h⟩

/-- hm96's leaf is the model's salted leaf `Merkle.hm96Leaf` at the executable's key, for the row encoding `enc`. -/
theorem hm96_eq_hm96Leaf (enc : Val → List UInt8) (hinj : Function.Injective enc) (v : Val) (y : List UInt8) :
    (ExplicitLeaf.hm96 enc hinj).leaf v y =
      hm96Leaf (H := H512) (E := ⟨enc, enc512.dig, hinj, enc512.dig_inj, enc512.dig_len⟩) K512 v y := by
  simp only [ExplicitLeaf.hm96, hm96Input, commitString, digBytes, hm96Leaf]

/-- **hm96's leaf is the executable's**: for a row's bytes and a salt, the leaf is `Hm96.Default512.treeLeaf` of the
commit string `Hm96.Default512.commitString (Sha512.hash row) salt`, the leaf `HmRow.checkPublic` computes from a
public `b ‖ c`. -/
theorem hm96_treeLeaf (enc : Val → List UInt8) (hinj : Function.Injective enc) (v : Val) (salt : ByteArray) :
    ((ExplicitLeaf.hm96 enc hinj).leaf v salt.data.toList).1 =
      Hm96.Default512.treeLeaf (Hm96.Default512.commitString (Sha512.hash ⟨(enc v).toArray⟩) salt) := by
  show (H512 (K512.leafPrefix ++ (xorBytes (enc512.dig (H512 (enc v))) (K512.mask salt.data.toList) ++
      enc512.dig (H512 (K512.saltPrefix ++ salt.data.toList))))).1 = _
  rw [H512_val, dig_eq, dig_eq, H512_val, H512_val, K512_mask, bytes_of_list,
    ← xor_range _ _ 64 (sha512_size _) (mask_size salt)]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ (_ ++
      (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList)).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++
      Hm96.Default512.commitString (Sha512.hash ⟨(enc v).toArray⟩) salt)
  rw [← data_append Hm96.Default512.saltPrefix salt, bytes_of_list, ← data_append, ← data_append, bytes_of_list]
  rfl

end FlockSoundness.Binding
