module

public import Proofs.Flock.Bridge.Spec.Tree
public import Proofs.Flock.Level3.RowV2
public import Proofs.Hash.Bytes
public import Definitions.Hash.RowV2
public import Definitions.Hash.Hm96
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The row's message and hm96's prefixes are the spec's

* `packBits_eq`, `rowPrefixV2_eq`, `enc_eq`: Flock's `sha512/row/v2` message (`FlockLevel3.RowV2.enc`, the Bridge's
  `rowMsg` at a row/v2 port) is `Hash.RowV2.msg`, the message `Hash.RowLeaf`'s `v2` format digests.
* `pad_eq`: `Flock.Hm96.pad` is `Hash.padTo`.
* `saltPrefix_eq`, `leafPrefix_eq`: `Hm96.Default512`'s salt and leaf prefixes are `Hash.Hm96`'s, the leaf's at the key
  `Hm96.Default512.key` and a hash that reads bytes as SHA-512 does (`HashOf`).
-/

@[expose] public section

namespace Flock.Bridge.Spec

open Properties.NetworkCertifier.Bytes (bs bs_append bs_length)

theorem bitsVal_eq : ∀ b : List Bool, HmRow.bitsVal b = Hash.RowSeg.bitsVal b
  | [] => rfl
  | x :: b => by simp only [HmRow.bitsVal, Hash.RowSeg.bitsVal, bitsVal_eq b]

theorem packBits_eq (b : List Bool) : HmRow.packBits b = Hash.RowSeg.packBits b := by
  simp only [HmRow.packBits, Hash.RowSeg.packBits, bitsVal_eq]

theorem digits_be (l : ℕ) : (FlockLevel3.RowV2.digits l).toList = Hash.be 8 l := by
  simp only [FlockLevel3.RowV2.digits, Hash.be, Array.toList_map, Array.toList_range, Nat.reduceSub]

/-- `P(ℓ)`, the code's, is the spec's. -/
theorem rowPrefixV2_eq (l : ℕ) : (HmRow.rowPrefixV2 l).data.toList = Hash.RowV2.rowPrefix l := by
  have ht : "verity/sha512-row/v2".toUTF8.data.toList.length = 20 := by
    rw [Array.length_toList, ByteArray.size_data, FlockLevel3.RowV2.tag_size]
  rw [FlockLevel3.RowV2.prefix_data, digits_be, Hash.RowV2.rowPrefix, Hash.padTo, Hash.cstr, Hash.utf8]
  simp only [List.length_append, ht, List.length_singleton, Hash.BindingProofs.be_length]

/-- **The code's `sha512/row/v2` message is `Hash.RowV2.msg`.** -/
theorem enc_eq (b : List Bool) : FlockLevel3.RowV2.enc b = Hash.RowV2.msg b := by
  rw [FlockLevel3.RowV2.enc, Hash.RowV2.msg, rowPrefixV2_eq, packBits_eq]

theorem pad_eq (b : ByteArray) (n : ℕ) : bs (Hm96.pad b n) = Hash.padTo n (bs b) := by
  rw [Hm96.pad, bs_append, Hash.padTo, bs_length]
  congr 1
  simp only [bs, Array.toList_replicate]
  congr 1
  omega

theorem saltTag_eq : bs Hm96.sha512.saltTag = Hash.cstr "verity/hm96-sha512/salt/v1" := by decide +kernel

theorem leafTag_eq : bs Hm96.sha512.leafTag = Hash.cstr "verity/hm96-sha512/leaf/v1" := by decide +kernel

/-- The salt digest's first block, the code's, is the spec's. -/
theorem saltPrefix_eq : bs Hm96.Default512.saltPrefix = Hash.Hm96.saltPrefix := by
  rw [Hm96.Default512.saltPrefix, pad_eq, saltTag_eq]
  rfl

/-- The leaf's first block, the code's, is the spec's at the code's key. -/
theorem leafPrefix_eq {H : Hash.Msg → Hash.Digest} (hH : HashOf H) :
    bs Hm96.Default512.leafPrefix = Hash.Hm96.leafPrefix H (bs Hm96.Default512.key) := by
  rw [Hm96.Default512.leafPrefix, Hm96.leafPrefix, pad_eq, bs_append, leafTag_eq, Hash.Hm96.leafPrefix, hH,
    dig_bytes (Sha512.hash_size _)]
  rfl

end Flock.Bridge.Spec
