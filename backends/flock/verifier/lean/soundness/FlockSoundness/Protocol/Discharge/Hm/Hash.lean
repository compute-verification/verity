/-
The verifier's SHA-512 (`Flock.Sha512.hash`) as the chain of its compressions: the padded message (`padData`), the
chaining value after each block (`chain`), and the digest's bytes, big-endian per word (`digestOf`), whose bit `t` is bit
`wordBit t` of the chaining value in word order.
-/
import Mathlib.Tactic
import FlockSoundness.Protocol.Discharge.Hm.Spec
import FlockLevel3.LookupRows

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## Message words -/

/-- `k ≤ 8` bytes `g 0, …, g (k-1)` read big-endian: bit `p` is bit `p % 8` of byte `k - 1 - p / 8`. -/
theorem getLsbD_foldBE (g : ℕ → UInt8) : ∀ k ≤ 8, ∀ p : ℕ,
    ((List.range' 0 k).foldl (fun (x : UInt64) j => x <<< 8 ||| (g j).toUInt64) 0).toBitVec.getLsbD p =
      (decide (p < 8 * k) && (g (k - 1 - p / 8)).toNat.testBit (p % 8))
  | 0, _, p => by simp
  | k + 1, hk, p => by
      rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.one_mul,
        getLsbD_step]
      by_cases hp : p < 8
      · simp only [hp, ↓reduceIte, show p < 8 * (k + 1) by omega, decide_true, Bool.true_and,
          show k + 1 - 1 - p / 8 = k by omega, Nat.mod_eq_of_lt hp]
      · simp only [hp, ↓reduceIte]
        rw [getLsbD_foldBE g k (by omega) (p - 8), show (p - 8) % 8 = p % 8 by omega,
          show k - 1 - (p - 8) / 8 = k + 1 - 1 - p / 8 by omega]
        by_cases h2 : p < 8 * (k + 1)
        · simp [h2, show p < 64 by omega, show p - 8 < 8 * k by omega]
        · simp [h2, show ¬ (p - 8 < 8 * k) by omega]

/-- Word `i` of the block at `off` of a byte list, big-endian. -/
def wordL (l : List UInt8) (off i : ℕ) : UInt64 :=
  (List.range' 0 8).foldl (fun x j => x <<< 8 ||| (l.getD (off + 8 * i + j) 0).toUInt64) 0

theorem get!_eq_getD (b : ByteArray) (k : ℕ) : b.get! k = b.data.toList.getD k 0 := by
  obtain ⟨bs⟩ := b
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, ByteArray.get!, getElem!_def]
  cases bs[k]? <;> rfl

theorem getLsbD_wordL (l : List UInt8) (off i : ℕ) {p : ℕ} (hp : p < 64) :
    (wordL l off i).toBitVec.getLsbD p = (l.getD (off + 8 * i + (7 - p / 8)) 0).toNat.testBit (p % 8) := by
  rw [wordL, getLsbD_foldBE _ 8 le_rfl p]
  simp [hp]

/-! ## The compression reads only its block -/

end FlockSoundness.Discharge.Hm
