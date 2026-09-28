/-
The hm96 slot outputs the commit string `b ‖ c` (`Binding.commitString`) of a row's bytes and a salt, bit for bit and as
bytes, when its inputs hold the row's last chaining value and block and the salt.
-/
import Proofs.Flock.Soundness.Binding.Leaf
import Proofs.Flock.Soundness.Discharge.Hm.Mask

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

theorem bitL_append_lo {A B : List UInt8} (hA : A.length = 64) {t : ℕ} (ht : t < 512) :
    bitL (A ++ B) t = bitL A t := by
  rw [bitL, bitL, List.getD_append _ _ _ _ (by omega)]

theorem bitL_append_hi {A B : List UInt8} (hA : A.length = 64) (t : ℕ) : bitL (A ++ B) (512 + t) = bitL B t := by
  rw [bitL, bitL, List.getD_append_right _ _ _ _ (by omega), hA, show (512 + t) / 8 - 64 = t / 8 by omega,
    show (512 + t) % 8 = t % 8 by omega]

theorem bitL_xorBytes {a b : List UInt8} (ha : a.length = 64) (hb : b.length = 64) {t : ℕ} (ht : t < 512) :
    bitL (Merkle.xorBytes a b) t = (bitL a t ^^ bitL b t) := by
  rw [bitL, bitL, bitL, Merkle.xorBytes, List.getD_eq_getElem?_getD, List.getElem?_zipWith,
    List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (show t / 8 < a.length by omega),
    List.getElem?_eq_getElem (show t / 8 < b.length by omega)]
  simp [UInt8.toNat_xor, Nat.testBit_xor]

/-- Byte strings of one length with the same bits are equal. -/
theorem eq_of_bitL {A B : List UInt8} (h1 : A.length = B.length) (h : ∀ t < 8 * A.length, bitL A t = bitL B t) :
    A = B := by
  refine List.ext_getElem h1 fun i hi hi' => UInt8.toNat_inj.mp (Nat.eq_of_testBit_eq fun b => ?_)
  by_cases hb : b < 8
  · have e := h (8 * i + b) (by omega)
    rw [bitL, bitL, show (8 * i + b) / 8 = i by omega, show (8 * i + b) % 8 = b by omega, List.getD_eq_getElem?_getD,
      List.getElem?_eq_getElem hi, Option.getD_some, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi',
      Option.getD_some] at e
    exact e
  · have hlt : ∀ x : UInt8, x.toNat < 2 ^ b := fun x =>
      lt_of_lt_of_le (UInt8.toNat_lt x) (Nat.pow_le_pow_right (by norm_num) (by omega))
    rw [Nat.testBit_lt_two_pow (hlt _), Nat.testBit_lt_two_pow (hlt _)]

theorem length_digBytes (d : Refine.Dig) : (Binding.digBytes d).length = 64 := Refine.hL512 d

theorem length_commitString (r s : List UInt8) : (Binding.commitString r s).length = 128 := by
  rw [Binding.commitString, List.length_append, Merkle.xorBytes, List.length_zipWith, length_digBytes,
    length_digBytes, Refine.K512.mask_len]
  rfl

/-- **The hm96 slot outputs the commit string's bits.** In a satisfying assignment `z` of the hm96 circuit's rows (under
the executable's key), if its `cv` inputs hold the chaining value of a row's bytes `r` before their last block, its `pad`
inputs their last block and its salt inputs a 192-byte salt `s`, then output bit `t < 1024` is bit `t` of
`commitString r s = SHA-512(r) ⊕ M·s ‖ SHA-512(salt_prefix ‖ s)` (bit `t % 8` of byte `t / 8`). -/
theorem hm96_commit (z : Var → Bool) (h : stHolds z ((hm96Row Hm96.Default512.key).run St.empty).2) (r : List UInt8)
    {s : List UInt8} (hs : s.length = 192) (hcv : ∀ q < 512, z q = wbit (lastCv ⟨r.toArray⟩) q)
    (hpad : ∀ q < 1024, z (512 + q) = wbit (lastBlock ⟨r.toArray⟩) q) (hy : ∀ q < 1536, z (1536 + q) = bitL s q) :
    ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2.length = 1024 ∧
      ∀ t < 1024, bit z ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t = bitL (Binding.commitString r s) t := by
  obtain ⟨hlen, hb⟩ := hm96_slot _ z h ⟨r.toArray⟩ hs hcv hpad hy
  have hx : (Merkle.xorBytes (Binding.digBytes (Refine.H512 r)) (Refine.K512.mask s)).length = 64 := by
    rw [Merkle.xorBytes, List.length_zipWith, length_digBytes, Refine.K512.mask_len]
    rfl
  refine ⟨hlen, fun t ht => ?_⟩
  by_cases h1 : t < 512
  · rw [(hb t h1).1, Binding.commitString, bitL_append_lo hx h1,
      bitL_xorBytes (length_digBytes _) (Refine.K512.mask_len s) h1, Refine.K512_mask, bitL_mask s h1,
      Binding.digBytes, Refine.dig_eq, Refine.H512_val]
  · obtain ⟨t', rfl⟩ : ∃ t', t = 512 + t' := ⟨t - 512, by omega⟩
    have hm : saltMsg s = ⟨(Refine.K512.saltPrefix ++ s).toArray⟩ := rfl
    rw [(hb t' (by omega)).2, Binding.commitString, bitL_append_hi hx, Binding.digBytes, Refine.dig_eq,
      Refine.H512_val, hm]

/-- **The bytes**: 128 bytes whose bits are the hm96 slot's outputs are the commit string. -/
theorem hm96_commit_bytes (z : Var → Bool) (h : stHolds z ((hm96Row Hm96.Default512.key).run St.empty).2)
    (r : List UInt8) {s : List UInt8} (hs : s.length = 192) (hcv : ∀ q < 512, z q = wbit (lastCv ⟨r.toArray⟩) q)
    (hpad : ∀ q < 1024, z (512 + q) = wbit (lastBlock ⟨r.toArray⟩) q) (hy : ∀ q < 1536, z (1536 + q) = bitL s q)
    {bc : List UInt8} (hbc : bc.length = 128)
    (hread : ∀ t < 1024, bitL bc t = bit z ((hm96Row Hm96.Default512.key).run St.empty).1.2.2.2 t) :
    bc = Binding.commitString r s :=
  eq_of_bitL (by rw [hbc, length_commitString]) fun t ht => by
    rw [hread t (by omega), (hm96_commit z h r hs hcv hpad hy).2 t (by omega)]

end FlockSoundness.Discharge.Hm
