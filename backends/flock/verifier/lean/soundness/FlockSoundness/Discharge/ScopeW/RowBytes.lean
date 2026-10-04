import FlockSoundness.Discharge.Layout.RowEnc
import FlockSoundness.Binding.Rows

set_option Elab.async false

/-!
# The bytes of `sha512/row-seg/v1`'s first blocks

What `ScopeW.RowEnc`'s proofs read about the messages a segmented row hashes:

* `rowPrefixSeg_length`, `segmentPrefix_length`: the outer first block `Q(ℓ)` and segment `j`'s `S(j, s, m)` are one
  SHA-512 block each, 128 bytes;
* `rowPrefixSeg_inj`: `Q(ℓ)` fixes `ℓ` below `2^64`, which its bytes 25 to 32 hold as a u64 big-endian;
* `flatMap_range_inj`: two concatenations of `s` blocks of one length are equal only block by block.
-/

namespace FlockSoundness.Discharge.ScopeW.RowBytes

open Flock FlockLevel3

theorem pad_length {b : ByteArray} {n : ℕ} (h : b.size ≤ n) : (Hm96.pad b n).data.toList.length = n := by
  rw [Hm96.pad, Refine.data_append, List.length_append, Refine.length_data]
  simp only [Array.length_toList, Array.size_replicate]
  omega

theorem be64_size (x : ℕ) : (HmRow.be64 x).size = 8 := by simp [HmRow.be64, ByteArray.size]

/-- **`Q(ℓ)` is one SHA-512 block**, 128 bytes, at every length. -/
theorem rowPrefixSeg_length (l : ℕ) : (HmRow.rowPrefixSeg l).data.toList.length = 128 := by
  have h0 : ("verity/sha512-row-seg/v1".toUTF8.push 0).size = 25 := by decide
  refine pad_length ?_
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, h0, be64_size, be64_size,
    Layout.be32_size]
  omega

/-- **`S(j, s, m)` is one SHA-512 block**, 128 bytes, at every segment. -/
theorem segmentPrefix_length (j s m : ℕ) : (HmRow.segmentPrefix j s m).data.toList.length = 128 := by
  have h0 : ("verity/sha512-row-seg/v1/segment".toUTF8.push 0).size = 33 := by decide
  refine pad_length ?_
  rw [ByteArray.size_append, ByteArray.size_append, ByteArray.size_append, h0, be64_size, be64_size, be64_size]
  omega

/-- **`Q(ℓ)` fixes `ℓ` below `2^64`**: its bytes after the 25-byte label are `ℓ`'s eight (`RowV2.digits`). -/
theorem rowPrefixSeg_inj {l l' : ℕ} (hl : l < 2 ^ 64) (hl' : l' < 2 ^ 64)
    (h : (HmRow.rowPrefixSeg l).data.toList = (HmRow.rowPrefixSeg l').data.toList) : l = l' := by
  unfold HmRow.rowPrefixSeg Hm96.pad at h
  simp only [Refine.data_append, List.append_assoc] at h
  have h1 := List.append_cancel_left h
  have h2 : (HmRow.be64 l).data.toList = (HmRow.be64 l').data.toList :=
    (List.append_inj h1 (by rw [Refine.length_data, Refine.length_data, be64_size, be64_size])).1
  exact RowV2.digits_inj hl hl' h2

variable {α : Type}

theorem length_flatMap_range {f : ℕ → List α} {k : ℕ} (hf : ∀ j, (f j).length = k) :
    ∀ s : ℕ, ((List.range s).flatMap f).length = k * s
  | 0 => by simp
  | s + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, length_flatMap_range hf s]
    simp [hf, Nat.mul_succ]

/-- **Blocks of one length concatenate injectively**: two concatenations of `s` blocks of `k` elements are equal only
when their blocks are. -/
theorem flatMap_range_inj {f g : ℕ → List α} {k : ℕ} (hf : ∀ j, (f j).length = k) (hg : ∀ j, (g j).length = k) :
    ∀ {s : ℕ}, (List.range s).flatMap f = (List.range s).flatMap g → ∀ j < s, f j = g j
  | 0, _, _, hj => absurd hj (Nat.not_lt_zero _)
  | s + 1, h, j, hj => by
    rw [List.range_succ, List.flatMap_append, List.flatMap_append] at h
    obtain ⟨h1, h2⟩ := List.append_inj h (by rw [length_flatMap_range hf, length_flatMap_range hg])
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · exact flatMap_range_inj hf hg h1 j hj
    · simpa using h2

end FlockSoundness.Discharge.ScopeW.RowBytes
