import Proofs.Flock.Soundness.Binding.Layout

/-!
# Bytes from their bits

`Binding.bytesAt` reads a byte string off a witness, byte `j`'s bit `t` (bit 0 the least significant) at the address
`a j t` (`Binding.byteOf`). `bytesAt_eq` says what it reads: a byte string whose bits the witness holds there.
-/

namespace FlockSoundness.Discharge.Layout

open Model Binding

/-- A number below `2^k` is the sum of its first `k` bits' powers. -/
theorem sum_testBit : ∀ (k N : ℕ), N < 2 ^ k → ∑ t : Fin k, (if N.testBit t.val then 2 ^ t.val else 0) = N
  | 0, N, h => by simp at h; simp [h]
  | k + 1, N, h => by
      rw [Fin.sum_univ_succ]
      have hrest : ∑ t : Fin k, (if N.testBit (t.succ : Fin (k + 1)).val then 2 ^ (t.succ : Fin (k + 1)).val else 0) =
          2 * ∑ t : Fin k, (if (N / 2).testBit t.val then 2 ^ t.val else 0) := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun t _ => ?_
        simp only [Fin.val_succ, Nat.testBit_succ, pow_succ]
        split_ifs <;> ring
      have hN : N / 2 < 2 ^ k := by rw [pow_succ] at h; omega
      rw [hrest, sum_testBit k (N / 2) hN]
      simp only [Fin.val_zero, Nat.testBit_zero, pow_zero, decide_eq_true_eq]
      split_ifs with h0 <;> omega

/-- **A byte from its bits** is the byte. -/
theorem byteOf_testBit (x : UInt8) : byteOf (fun t => x.toNat.testBit t.val) = x := by
  unfold byteOf
  rw [sum_testBit 8 x.toNat x.toNat_lt]
  exact UInt8.ofNat_toNat

/-- **The bytes a witness holds** at addresses where it holds the bits of `B`, byte `j`'s bit `t` at `a j t`. -/
theorem bytesAt_eq {m N : ℕ} (z : Witness m) (a : Fin N → Fin 8 → Fin (2 ^ m)) (B : List UInt8) (hB : B.length = N)
    (h : ∀ j t, bit (z (a j t)) = (B.get ⟨j.val, hB ▸ j.isLt⟩).toNat.testBit t.val) : bytesAt z a = B := by
  subst hB
  refine List.ext_get (by simp [bytesAt]) fun j h1 h2 => ?_
  simp only [bytesAt, List.get_eq_getElem, List.getElem_ofFn]
  have : bitsAt z (a ⟨j, h2⟩) = fun t => (B.get ⟨j, h2⟩).toNat.testBit t.val := funext fun t => h ⟨j, h2⟩ t
  rw [this, byteOf_testBit]
  rfl

end FlockSoundness.Discharge.Layout
