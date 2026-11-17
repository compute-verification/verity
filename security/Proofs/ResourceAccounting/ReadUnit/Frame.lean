/- The frame-v3 read is sound (`Gates.FrameReadSpec`). -/
module

public import Proofs.ResourceAccounting.ReadUnit.Frame.Parts

@[expose] public section

namespace ResourceAccounting.ReadUnit.FrameRead

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Flock.ReadNet Flock.ReadNet.Frame
open ResourceAccounting.ReadUnit.Gates

theorem length_utf8_seg : (Hash.utf8 Hash.RowLeaf.Format.seg.schema).length = 22 := by
  show ("hm96-sha512/row-seg/v1".toUTF8.data.toList).length = 22
  rw [length_utf8]
  simp [Char.utf8Size]

/-- Past depth 32 a rank takes five bytes and the leaf's message a third block, which the net's one layout per length
doesn't follow. -/
theorem frameReadSpec {D : ℕ} (hD : D ≤ 32) :
    FrameReadSpec D Hash.RowLeaf.Format.seg.schema (Flock.ReadNet.frame D Hash.RowLeaf.Format.seg.schema) := by
  intro root domain leaf pos path
  simp only [Flock.ReadNet.frame, frameWith]
  refine Sound.bind' (sound_constBits (none, none) 0 _) fun pad =>
    Sound.bind' (sound_leafHash _ _ domain leaf _) fun cur =>
      Sound.bind' (sound_climb D domain pos path 0 pad.1 cur) fun cur' =>
        Sound.mono (sound_eqDigest cur' root) fun z r heq hcl hlf hpad hr hd hl hpos hpath hps hev => ?_
  obtain ⟨hc1, hpe, hpl⟩ := hpad ⟨fun _ h => (by cases h), fun _ h => (by cases h)⟩
  have hn1 : 1 ≤ indexBytes D := le_max_left _ _
  have hn4 : indexBytes D ≤ 4 := by
    unfold indexBytes
    omega
  have hx : (pos ++ pad.2).length = 8 * indexBytes D := by
    rw [List.length_append, hpl, hpos]
    unfold indexBytes
    omega
  obtain ⟨hcs, hce⟩ := hlf hn1 hx (by rw [length_utf8_seg]; omega) hd hl
  obtain ⟨hcs', hce'⟩ := hcl hc1 (by omega) (by omega) hd hcs hps
  have hroot := heq hcs' hr hev
  have hnum : ReadUnit.num (evs z (pos ++ pad.2)) = ReadUnit.num (evs z pos) := num_evs_pad z pos pad.2 _ hpe
  rw [← hroot, hce', hce, Nat.pow_zero, Nat.div_one, hnum]
  rfl

end ResourceAccounting.ReadUnit.FrameRead
