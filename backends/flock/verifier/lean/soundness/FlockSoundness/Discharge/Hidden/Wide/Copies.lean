import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Protocol.Discharge.Hidden.Wide.Copies
import FlockSoundness.Discharge.Hidden.Delta
import FlockSoundness.Discharge.Hidden.Rows

/-!
# Δ's copies at the rows, for `sha512/row/v1` and `sha512/row/v2` rows

Without a typed template Δ is its slot constants' pairs, the rows' constant bits as `HmRow.delta` writes them
(`rowPairsAll`), the units' leaf and cut bits and the wires' bits (`delta_flat_all`), for every row, `sha512/row/v1` or
`sha512/row/v2`, of any number of compressions. Each of the rows' writes is a copy (`rowCopiesAll`): a row's first `cv`
(its first compression's or, for a row of none, its `hm96` slot's), the constant bits of its `pad`, and those of its
compressions' messages from block `bits / 1024` on. The rows' wires (`rowWires_all`) give a row of no compression no
`hm96` wire. Every copy into VU `g`'s row `p` lands in that row's slots (`Dst`), and distinct rows' slots are distinct
(`dst_inj`), so Δ's copies at a column of row `p` are row `p`'s (`filter_rows`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-! ## The rows' wires, for any rows -/

/-! ## Positions in the rows' slots -/

section Pos

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

/-! ## Δ's copies at a row's own columns -/

/-- **No copy of `HmRow.delta`'s rows or of the rows' wires is at a message bit of a row whose message has no constant
bit before its last block** (an output row: `bits = 16 · words`, `2 · words = 128 · chunks`). -/
theorem msg_filterAll {pin g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hkb : c.ports[p]!.chunks ≤ c.ports[p]!.bits / 1024) (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  have hk : b / 128 < c.ports[p]!.chunks := by omega
  have hmo := msgOff_lt (b := b) ht
  have hx := deltaMsgCol_pos R hu (g := g) (t := t) hp hk
  rw [filter_rows R hu hg hp (Or.inl ⟨b / 128, 512 + msgOff b t, hk, by omega, hx⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases wireFAll_mem R hu hp he with ⟨k', t', -, hk', ht', rfl⟩ | ⟨-, t', ht', rfl⟩
    · intro h
      have := (shaPos_inj R hu hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
      omega
    · exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
  have hr : (rowCopiesOf c pin g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hu hp he with ⟨h0, t', ht', he1⟩ | ⟨h, -⟩
      · rw [he1]
        intro h
        have := (shaPos_inj R hu hg hp h0 (by omega) hg hp hk (by omega) h).2.2.2
        omega
      · omega
    · obtain ⟨t', ht', he1⟩ := padCopies_mem R hu he
      rw [he1]
      exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
    · obtain ⟨k', t', hk1, hk', -⟩ := msgCopies_mem R hu hp he
      omega
  rw [hw, hr, List.append_nil]

end Pos

end FlockSoundness.Discharge.Hidden
