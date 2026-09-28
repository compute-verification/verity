/-
`sha512/row/v2`'s bytes for a bit row (`verity.commitments.rowleaf.sha512_row_v2_digest`'s message before SHA-512's
padding): the block `P(ℓ)` (`Flock.HmRow.rowPrefixV2`), then the bits least significant first, zero-filled to a byte
(`Flock.HmRow.packBits`). The encoding is injective on rows shorter than `2^64` bits (`enc_inj`): `P(ℓ)` is 128 bytes and
carries `ℓ` as a u64, and `ℓ` fixes how many bits the bytes after it hold. The soundness package's v2 `HmRows` takes its
`enc_inj` from here.
-/
import Flock.HmRow
import Mathlib.Data.List.Basic

open Flock.HmRow

namespace FlockLevel3.RowV2

/-- `sha512/row/v2`'s bytes: P(ℓ), then the bits least significant first, zero-filled to a byte. -/
def enc (bs : List Bool) : List UInt8 :=
  (Flock.HmRow.rowPrefixV2 bs.length).data.toList ++ Flock.HmRow.packBits bs

/-- `P(ℓ)`'s bytes 21 to 28: `ℓ` as a u64 big-endian. -/
def digits (l : ℕ) : Array UInt8 := (Array.range 8).map fun i => (l / 256 ^ (7 - i) % 256).toUInt8

theorem tag_size : "verity/sha512-row/v2".toUTF8.size = 20 := by decide

theorem digits_size (l : ℕ) : (digits l).size = 8 := by simp [digits]

theorem head_size (l : ℕ) : ("verity/sha512-row/v2".toUTF8.push 0 ++ ByteArray.mk (digits l)).size = 29 := by
  rw [ByteArray.size_append, ByteArray.size_push, tag_size]
  show 20 + 1 + (digits l).size = 29
  rw [digits_size]

theorem prefix_data (l : ℕ) : (rowPrefixV2 l).data.toList =
    "verity/sha512-row/v2".toUTF8.data.toList ++ [0] ++ (digits l).toList ++ List.replicate 99 0 := by
  unfold rowPrefixV2 Flock.Hm96.pad
  rw [← digits, head_size]
  rw [ByteArray.data_append, ByteArray.data_append, ByteArray.data_push]
  simp only [Array.toList_append, Array.toList_push, Array.toList_replicate]
  rfl

theorem prefix_length (l : ℕ) : (rowPrefixV2 l).data.toList.length = 128 := by
  rw [prefix_data]
  simp only [List.length_append, Array.length_toList, ByteArray.size_data, tag_size, digits_size, List.length_cons,
    List.length_nil, List.length_replicate]

theorem prefix_size (l : ℕ) : (Flock.HmRow.rowPrefixV2 l).size = 128 := by
  rw [← ByteArray.size_data, ← Array.length_toList, prefix_length]

theorem packBits_length (bs : List Bool) : (Flock.HmRow.packBits bs).length = (bs.length + 7) / 8 := by
  simp [packBits]

theorem byte_eq {a b : ℕ} (h : (a % 256).toUInt8 = (b % 256).toUInt8) : a % 256 = b % 256 := by
  have := congrArg UInt8.toNat h
  simpa only [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.reducePow, Nat.mod_mod] using this

theorem digits_inj {l l' : ℕ} (hl : l < 2 ^ 64) (hl' : l' < 2 ^ 64) (h : (digits l).toList = (digits l').toList) :
    l = l' := by
  simp only [digits, Array.toList_map, Array.toList_range, List.range_succ, List.range_zero, List.nil_append,
    List.map_cons, List.map_nil, List.cons_append, List.cons.injEq, and_true] at h
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7⟩ := h
  replace h0 := byte_eq h0; replace h1 := byte_eq h1; replace h2 := byte_eq h2; replace h3 := byte_eq h3
  replace h4 := byte_eq h4; replace h5 := byte_eq h5; replace h6 := byte_eq h6; replace h7 := byte_eq h7
  simp only [Nat.reduceSub, Nat.reducePow, Nat.pow_zero, Nat.div_one] at h0 h1 h2 h3 h4 h5 h6 h7 hl hl'
  omega

/-- `P(ℓ)` fixes `ℓ` below `2^64`. -/
theorem rowPrefixV2_inj {l l' : ℕ} (hl : l < 2 ^ 64) (hl' : l' < 2 ^ 64)
    (h : (rowPrefixV2 l).data.toList = (rowPrefixV2 l').data.toList) : l = l' := by
  rw [prefix_data, prefix_data] at h
  have hn : ("verity/sha512-row/v2".toUTF8.data.toList ++ [0] ++ (digits l).toList).length =
      ("verity/sha512-row/v2".toUTF8.data.toList ++ [0] ++ (digits l').toList).length := by
    simp only [List.length_append, Array.length_toList, digits_size]
  exact digits_inj hl hl' (List.append_cancel_left (List.append_inj h hn).1)

theorem bitsVal_lt : ∀ bs : List Bool, bitsVal bs < 2 ^ bs.length
  | [] => by simp [bitsVal]
  | b :: bs => by
    have := bitsVal_lt bs
    simp only [bitsVal, List.length_cons, Nat.pow_succ]
    split <;> omega

theorem bitsVal_inj : ∀ {as bs : List Bool}, as.length = bs.length → bitsVal as = bitsVal bs → as = bs
  | [], [], _, _ => rfl
  | a :: as, b :: bs, hl, h => by
    simp only [bitsVal] at h
    have hab : a = b := by cases a <;> cases b <;> simp at h ⊢ <;> omega
    subst hab
    have : bitsVal as = bitsVal bs := by cases a <;> simp at h <;> omega
    rw [bitsVal_inj (by simpa using hl) this]

theorem getElem_chunk (bs : List Bool) (i : ℕ) (hi : i < bs.length) :
    ((bs.drop (8 * (i / 8))).take 8)[i % 8]'(by simp; omega) = bs[i] := by
  simp only [List.getElem_take, List.getElem_drop]
  congr 1; omega

/-- Rows of one length with the same bytes are equal: the fill bits are zero, so each byte holds its eight bits. -/
theorem packBits_inj {bs bs' : List Bool} (hl : bs.length = bs'.length) (h : packBits bs = packBits bs') : bs = bs' := by
  apply List.ext_getElem hl
  intro i hi hi'
  have hj : i / 8 < (bs.length + 7) / 8 := by omega
  have hc : (bs.drop (8 * (i / 8))).take 8 = (bs'.drop (8 * (i / 8))).take 8 := by
    have : (packBits bs)[i / 8]? = (packBits bs')[i / 8]? := by rw [h]
    unfold packBits at this
    rw [List.getElem?_map, List.getElem?_map, List.getElem?_range hj, ← hl, List.getElem?_range hj] at this
    simp only [Option.map_some, Option.some.injEq] at this
    have hv := congrArg UInt8.toNat this
    simp only [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.reducePow] at hv
    have hlen : ((bs.drop (8 * (i / 8))).take 8).length = ((bs'.drop (8 * (i / 8))).take 8).length := by simp [hl]
    have h1 := bitsVal_lt ((bs.drop (8 * (i / 8))).take 8)
    have h2 := bitsVal_lt ((bs'.drop (8 * (i / 8))).take 8)
    have p1 := Nat.pow_le_pow_right (by decide : 0 < 2) (List.length_take_le 8 (bs.drop (8 * (i / 8))))
    have p2 := Nat.pow_le_pow_right (by decide : 0 < 2) (List.length_take_le 8 (bs'.drop (8 * (i / 8))))
    exact bitsVal_inj hlen (by simp only [Nat.reducePow] at p1 p2; omega)
  rw [← getElem_chunk bs i hi, ← getElem_chunk bs' i hi']
  simp only [hc]

theorem enc_inj : Function.Injective (fun r : {bs : List Bool // bs.length < 2 ^ 64} => enc r.1) := by
  rintro ⟨a, ha⟩ ⟨b, hb⟩ h
  simp only [enc] at h
  obtain ⟨hp, hk⟩ := List.append_inj h (by rw [prefix_length, prefix_length])
  exact Subtype.ext (packBits_inj (rowPrefixV2_inj ha hb hp) hk)

end FlockLevel3.RowV2
