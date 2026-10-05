/-
The hm96 slot outputs the commit string `b ‖ c` (`Binding.commitString`) of a row's bytes and a salt, bit for bit and as
bytes, when its inputs hold the row's last chaining value and block and the salt.
-/
import FlockSoundness.Binding.Leaf
import FlockSoundness.Discharge.Hm.Mask
import FlockSoundness.Protocol.Discharge.Hm.Commit
import FlockSoundness.Discharge.Hm.Slot

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

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
