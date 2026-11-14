module

public import Flock.HmRow

/-!
# A 64-byte string's hex is a SHA-512 digest's

`HmRow.isSha512Hex` holds of `hexOf b` for every 64-byte `b`, so a check of a spliced digest's form (`hexOf (Sha512.hash
m)`) passes without evaluating the hash.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Hex

open Flock

theorem length_hexOf (b : ByteArray) : (hexOf b).length = 2 * b.size := by
  rw [← String.length_toList, Flock.Hex.hexOf_toList, Flock.Hex.length_flatMap2]
  simp

private theorem hd_digit : ∀ d < 16, ((Flock.Hex.hd d).isDigit ||
    (decide ('a' ≤ Flock.Hex.hd d) && decide (Flock.Hex.hd d ≤ 'f'))) = true := by
  decide +kernel

theorem isSha512Hex_hexOf {b : ByteArray} (h : b.size = 64) : HmRow.isSha512Hex (hexOf b) = true := by
  have hl := length_hexOf b
  unfold HmRow.isSha512Hex
  rw [Bool.and_eq_true]
  refine ⟨by simp [hl, h], ?_⟩
  rw [String.all_bool_eq, List.all_eq_true]
  intro c hc
  rw [Flock.Hex.hexOf_toList] at hc
  simp only [List.mem_flatMap, List.mem_cons, List.not_mem_nil, or_false] at hc
  obtain ⟨x, _, rfl | rfl⟩ := hc
  · exact hd_digit _ (by have := UInt8.toNat_lt x; omega)
  · exact hd_digit _ (by omega)

theorem size_hash (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

end Verity.Sampled.Instance.HonestSession.Hex
