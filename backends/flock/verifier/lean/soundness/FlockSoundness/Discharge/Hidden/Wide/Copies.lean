import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Protocol.Discharge.Hidden.Wide.Copies

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

/-- **Δ's one copy into compression `k ≥ 1`'s `cv` bit `t₀`**: the chain from compression `k − 1`'s output. -/
theorem chain_filterAll {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk1 : 1 ≤ k)
    (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) =
      [(Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀,
        Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t₀)] := by
  have h0 : 0 < c.ports[p]!.chunks := by omega
  have hs := (rowSlot_lt hp hk).2
  have hx := chainDst_pos R hu (g := g) hp hk t₀
  rw [filter_rows R hu hg hp (Or.inl ⟨k, t₀, hk, by omega, hx⟩)]
  have hr : (rowCopiesOf c pin g p).filter
      (fun e => e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hu hp he with ⟨-, t, ht, he1⟩ | ⟨h, -⟩
      · rw [he1]
        intro h
        have := (shaPos_inj R hu hg hp h0 (by omega) hg hp hk (by omega) h).2.2.1
        omega
      · omega
    · obtain ⟨t, ht, he1⟩ := padCopies_mem R hu he
      rw [he1]
      exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
    · obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_mem R hu hp he
      rw [he1]
      intro h
      have := (shaPos_inj R hu hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
      omega
  have hkm : k ∈ List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1) := by
    rw [List.mem_range'_1, rowSlots_size hp]; omega
  have hc := filter_flatMap_single _ List.nodup_range' (fun k => (List.range (Flat.wireWidth c
      (chainWire c.ports c.shaNet p k))).map fun t => (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t,
        Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t))
    (fun e => decide (e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀)) k hkm
    fun k' hk' hne => filter_nil_of _ _ fun e he => by
      rw [List.mem_range'_1, rowSlots_size hp] at hk'
      have hs' := (rowSlot_lt hp (show k' < c.ports[p]!.chunks by omega)).2
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t, ht, rfl⟩ := he
      rw [chainWidth_eq R.rowNets hs'] at ht
      rw [chainDst_eq R.rowNets hs', chainDst_eq R.rowNets hs]
      intro h
      exact hne (R.sha_dec hg hp (show k' < c.ports[p]!.chunks by omega) ht hg hp hk ht₀ h).2.2.1
  have hh := filter_nil_of ((List.range (Flat.wireWidth c (hmWire c.ports c.shaNet c.hmNet p))).map fun t =>
      (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t, Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t))
    (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) fun e he => by
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t, ht, rfl⟩ := he
      rw [hmWidth_eq R.rowNets] at ht
      show hmWireDst c g p + t ≠ _
      rw [hmWireDst_eq R.rowNets, hx]
      exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
  rw [hr, List.nil_append]
  unfold wireFAll
  rw [List.flatMap_append, List.filter_append, List.flatMap_map, hc, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  rw [List.flatMap_cons, List.flatMap_nil, List.append_nil, hh, List.append_nil, chainWidth_eq R.rowNets hs,
    filter_map_single _ _ _ _ ht₀]

/-- **Δ's one copy into the `hm96` slot's `cv` bit `t₀`** of a row of no compression: the row prefix's midstate bit. -/
theorem hmcv0_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (h0 : c.ports[p]!.chunks = 0)
    (ht₀ : t₀ < 512) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = c.slotCol c.hmNet g p + t₀) =
      [(c.slotCol c.hmNet g p + t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
  rw [filter_rows R hu hg hp (Or.inr ⟨t₀, by omega, rfl⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAll_mem R hu hp he with ⟨k, t, -, hk, -⟩ | ⟨h, -⟩ <;> omega
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨t, ht, he1⟩ := padCopies_mem R hu he
      rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨k, t, -, hk, -⟩ := msgCopies_mem R hu hp he
      omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) =
      [(c.slotCol c.hmNet g p + t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
    rw [← cvColOf_zero R hu hp h0 t₀]
    unfold cvCopies
    exact filter_map_single _ _ _ _ ht₀
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg]
  rfl

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
