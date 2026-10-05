/-
What the hm96 gadget's rows force (`Flock.HmNets.hm96Row`): its `b ‖ c` is the padding compression's digest XOR `M(key)·y`,
then SHA-512's last two compressions of `salt_prefix ‖ y`, bit by bit in byte-string order.
-/
import FlockSoundness.Discharge.Hm.Compress
import FlockSoundness.Protocol.Discharge.Hm.Hm96

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

theorem wordBit_lt {t : ℕ} (ht : t < 512) : wordBit t < 512 := by
  unfold wordBit; omega

/-- **The hm96 gadget computes `b ‖ c`** (`sound_hmTail`), with its inputs: `cv` (512 bits), `pad` (1024) and `y`
(1536). -/
theorem sound_hm96Row (key : ByteArray) : Sound (hm96Row key) fun z io =>
    io.1.length = 512 ∧ io.2.1.length = 1024 ∧ io.2.2.1.length = 1536 ∧ io.2.2.2.length = 1024 ∧
      ∀ t < 512,
        bit z io.2.2.2 t = (wbit (compressV ((words io.1 8).map (bv z)) ((words io.2.1 16).map (bv z))) (wordBit t) ^^
          maskV key (bit z io.2.2.1) t) ∧
        bit z io.2.2.2 (512 + t) = wbit (compressV (compressV saltMid (((saltWords io.2.2.1).map (bv z)).take 16))
          (((saltWords io.2.2.1).map (bv z)).drop 16 ++ saltTail)) (wordBit t) := by
  rw [hm96Row_eq]
  refine Sound.bind' (sound_input 512) fun cv => Sound.bind' (sound_input 1024) fun pad =>
    Sound.bind' (sound_input 1536) fun y => Sound.mono (sound_hmTail key cv pad y) fun z io h hy hpad hcv => ?_
  obtain ⟨h1, h2, h3, h4⟩ := h
  obtain ⟨h5, h6⟩ := h4 hcv hpad
  subst h1 h2 h3
  exact ⟨hcv, hpad, hy, h5, h6⟩

end FlockSoundness.Discharge.Hm
