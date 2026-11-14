module

public import Proofs.Flock.Bridge.Spec.Hm96
public import Proofs.Flock.Soundness.Binding.RowSeg
public import Definitions.Hash.RowLeaf
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The row-seg message is the spec's, and a registered schema is a `Hash.RowLeaf` format

* `rowPrefixSeg_eq`, `segmentPrefix_eq`: Flock's `Q(ℓ)` and `S(j, s, m)` (`HmRow.rowPrefixSeg`,
  `HmRow.segmentPrefix`) are `Hash.RowSeg.outerPrefix` and `Hash.RowSeg.segmentPrefix`.
* `segBytes_packBits`: segment `j`'s bytes of a row's bytes are the bytes of the row's segment `j`.
* `segEnc_eq`: **Flock's `sha512/row-seg/v1` outer message** (`FlockSoundness.Binding.RowSeg.enc`, the Bridge's
  `rowMsg` at a segmented port) **is `Hash.RowSeg.outerMsg`**, at a hash that reads bytes as SHA-512 does (`HashOf`).
* `rowPrefixV1_eq`, `encV1_eq`: **Flock's `sha512/row/v1` message** (the Bridge's `rowMsg` at a row/v1 port) **is
  `Hash.RowV1.msg`**.
* `formatOf`: the `Hash.RowLeaf` format of a registered schema, `seg` at `hm96-sha512/row-seg/v1`, `v2` at
  `hm96-sha512/row/v2` and `v1` otherwise (`formatOf_seg`, `formatOf_v2`, `formatOf_v1`); `formatOf_schema`: at each
  of the three schemas, the format's schema is the registered one; `formatOf_fits`: a row of whole words under `2^64`
  bits fits the format.
-/

@[expose] public section

namespace Flock.Bridge.Spec

open Properties.NetworkCertifier.Bytes (bs bs_append bs_length)

/-- `HmRow.be64` is `Hash.be 8`. -/
theorem bs_rowBe64 (x : ℕ) : bs (HmRow.be64 x) = Hash.be 8 x := by
  show ((Array.range 8).map fun i => (x / 256 ^ (7 - i) % 256).toUInt8).toList = _
  simp only [Hash.be, Array.toList_map, Array.toList_range, Nat.reduceSub]

theorem segTag_eq : bs ("verity/sha512-row-seg/v1".toUTF8.push 0) = Hash.cstr "verity/sha512-row-seg/v1" := by
  decide +kernel

theorem segmentTag_eq :
    bs ("verity/sha512-row-seg/v1/segment".toUTF8.push 0) = Hash.cstr "verity/sha512-row-seg/v1/segment" := by
  decide +kernel

theorem be32_1024 : bs (be32 1024) = Hash.be 4 1024 := by
  rw [bs_be32]
  all_goals rfl

theorem count_eq (l : ℕ) : FlockSoundness.Binding.RowSeg.count l = Hash.RowSeg.segCount l := rfl

/-- `Q(ℓ)`, the code's, is the spec's. -/
theorem rowPrefixSeg_eq (l : ℕ) : bs (HmRow.rowPrefixSeg l) = Hash.RowSeg.outerPrefix l := by
  rw [HmRow.rowPrefixSeg, pad_eq, bs_append, bs_append, bs_append, segTag_eq, bs_rowBe64, bs_rowBe64, be32_1024,
    Hash.RowSeg.outerPrefix]
  rfl

/-- `S(j, s, m)`, the code's, is the spec's. -/
theorem segmentPrefix_eq (j s m : ℕ) : bs (HmRow.segmentPrefix j s m) = Hash.RowSeg.segmentPrefix j s m := by
  rw [HmRow.segmentPrefix, pad_eq, bs_append, bs_append, bs_append, segmentTag_eq, bs_rowBe64, bs_rowBe64, bs_rowBe64,
    Hash.RowSeg.segmentPrefix]

theorem segment_length (b : List Bool) (j : ℕ) :
    (Hash.RowSeg.segment b j).length = FlockSoundness.Binding.RowSeg.segBits b.length j := by
  simp only [Hash.RowSeg.segment, Hash.RowSeg.SEG_BITS, List.length_take, List.length_drop,
    FlockSoundness.Binding.RowSeg.segBits, HmRow.ROW_SEG_BITS]
  omega

/-- **Segment `j`'s bytes of a row's bytes are its segment `j`'s bytes**: a segment is 1024 whole bytes. -/
theorem segBytes_packBits (b : List Bool) (j : ℕ) :
    FlockSoundness.Binding.RowSeg.segBytes b.length (HmRow.packBits b) j = HmRow.packBits (Hash.RowSeg.segment b j) := by
  apply List.ext_getElem
  · simp only [FlockSoundness.Binding.RowSeg.segBytes, HmRow.packBits, List.length_take, List.length_drop,
      List.length_map, List.length_range, segment_length, FlockSoundness.Binding.RowSeg.segBits, HmRow.ROW_SEG_BITS]
    omega
  · intro i h1 h2
    have hi : i < 1024 := by
      simp only [HmRow.packBits, List.length_map, List.length_range, segment_length,
        FlockSoundness.Binding.RowSeg.segBits, HmRow.ROW_SEG_BITS] at h2
      omega
    simp only [FlockSoundness.Binding.RowSeg.segBytes, HmRow.packBits, List.getElem_take, List.getElem_drop,
      List.getElem_map, List.getElem_range, Hash.RowSeg.segment, Hash.RowSeg.SEG_BITS]
    congr 2
    rw [List.drop_take, List.take_take, List.drop_drop, show min 8 (8192 - 8 * i) = 8 by omega]
    congr 2
    omega

/-- Segment `j`'s message, the code's, is the spec's. -/
theorem segMsg_eq (b : List Bool) (j : ℕ) :
    FlockSoundness.Binding.RowSeg.segMsg b.length (HmRow.packBits b) j = Hash.RowSeg.segmentMsg b j := by
  rw [FlockSoundness.Binding.RowSeg.segMsg, Hash.RowSeg.segmentMsg, segBytes_packBits, packBits_eq, segment_length,
    ← count_eq]
  exact congrArg (· ++ _) (segmentPrefix_eq _ _ _)

/-- A digest's bytes, as the code's outer message carries them, are `sha512Msg`'s. Stated at any message, so that no
proof unfolds SHA-512 on one whose prefix is concrete. -/
theorem digBytes_H512 (m : List UInt8) :
    FlockSoundness.Binding.digBytes (FlockSoundness.Refine.H512 m) = Claims.Sampling.sha512Msg m := by
  rw [FlockSoundness.Binding.digBytes, FlockSoundness.Refine.dig_eq, FlockSoundness.Refine.H512_val,
    Claims.Sampling.sha512Msg, Claims.Sampling.bytes, bs]

/-- **The code's `sha512/row-seg/v1` outer message is `Hash.RowSeg.outerMsg`**, at a hash that reads bytes as
SHA-512 does. -/
theorem segEnc_eq {H : Hash.Msg → Hash.Digest} (hH : HashOf H) (b : List Bool) :
    FlockSoundness.Binding.RowSeg.enc b.length (HmRow.packBits b) = Hash.RowSeg.outerMsg H b := by
  rw [FlockSoundness.Binding.RowSeg.enc, FlockSoundness.Binding.RowSeg.digests, Hash.RowSeg.outerMsg,
    Hash.RowSeg.segmentMsgs, List.flatMap_map, count_eq, ← rowPrefixSeg_eq, bs]
  refine congrArg ((HmRow.rowPrefixSeg b.length).data.toList ++ ·)
    (congrArg (fun f => List.flatMap f (List.range (Hash.RowSeg.segCount b.length))) (funext fun j => ?_))
  rw [digBytes_H512, sha512Msg_eq hH, segMsg_eq]

theorem v1Tag_eq : bs ("verity/sha512-row/v1".toUTF8.push 0) = Hash.cstr "verity/sha512-row/v1" := by
  decide +kernel

theorem v1Role_eq :
    bs (ByteArray.mk #[1, 16]) = Hash.be 1 Hash.RowV1.ROLE ++ Hash.be 1 Hash.RowV1.WORD_BITS := by
  decide +kernel

/-- A word count's four bytes, the code's, are `Hash.be 4` of it. -/
theorem bs_be32_nat (n : ℕ) : bs (be32 n.toUInt32) = Hash.be 4 n := by
  rw [bs_be32, Nat.toUInt32_eq, UInt32.toNat_ofNat', show (2 : ℕ) ^ 32 = 256 ^ 4 by norm_num, be_mod]

/-- `sha512/row/v1`'s first block, the code's, is the spec's. -/
theorem rowPrefixV1_eq (n : ℕ) : bs (HmRow.rowPrefix n) = Hash.RowV1.rowPrefix n := by
  rw [HmRow.rowPrefix, pad_eq, bs_append, bs_append, v1Tag_eq, v1Role_eq, bs_be32_nat, Hash.RowV1.rowPrefix]
  simp only [List.append_assoc]

/-- **The code's `sha512/row/v1` message** (the Bridge's `rowMsg` at a row/v1 port) **is `Hash.RowV1.msg`.** -/
theorem encV1_eq (b : List Bool) :
    (HmRow.rowPrefix (b.length / 16)).data.toList ++ HmRow.packBits b = Hash.RowV1.msg b := by
  show bs (HmRow.rowPrefix (b.length / 16)) ++ HmRow.packBits b =
    Hash.RowV1.rowPrefix (b.length / 16) ++ Hash.RowSeg.packBits b
  rw [rowPrefixV1_eq, packBits_eq]

/-- **The `Hash.RowLeaf` format of a registered schema**: `seg` at `hm96-sha512/row-seg/v1`, `v2` at
`hm96-sha512/row/v2`, else `v1`. -/
def formatOf (schema : String) : Hash.RowLeaf.Format :=
  if schema = HmRow.ROW_SCHEMA_SEG then .seg else if schema = HmRow.ROW_SCHEMA_V2 then .v2 else .v1

theorem formatOf_seg : formatOf HmRow.ROW_SCHEMA_SEG = .seg := ite_eq_left rfl

theorem formatOf_v2 : formatOf HmRow.ROW_SCHEMA_V2 = .v2 := (ite_eq_right (by decide)).trans (ite_eq_left rfl)

theorem formatOf_v1 : formatOf HmRow.ROW_SCHEMA = .v1 := (ite_eq_right (by decide)).trans (ite_eq_right (by decide))

/-- **At a schema Flock registers a port under, its format's schema is it.** -/
theorem formatOf_schema {schema : String}
    (h : schema = HmRow.ROW_SCHEMA_SEG ∨ schema = HmRow.ROW_SCHEMA_V2 ∨ schema = HmRow.ROW_SCHEMA) :
    (formatOf schema).schema = schema := by
  rcases h with rfl | rfl | rfl
  · rw [formatOf_seg]; rfl
  · rw [formatOf_v2]; rfl
  · rw [formatOf_v1]; rfl

/-- **A row of whole words under `2^64` bits fits its schema's format**, whichever it is. -/
theorem formatOf_fits (schema : String) {ℓ : ℕ} (h : ℓ < 2 ^ 64) (hw : ℓ % 16 = 0) : (formatOf schema).Fits ℓ := by
  unfold formatOf
  split
  · exact h
  · split
    · exact h
    · exact (show ℓ < 2 ^ 64 ∧ ℓ % Hash.RowV1.WORD_BITS = 0 from ⟨h, hw⟩)

end Flock.Bridge.Spec
