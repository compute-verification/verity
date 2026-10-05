import FlockSoundness.Refine.Salted
import FlockSoundness.Protocol.Binding.Leaf

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

variable {Val : Type}

/-! ## hm96-sha512 -/

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
