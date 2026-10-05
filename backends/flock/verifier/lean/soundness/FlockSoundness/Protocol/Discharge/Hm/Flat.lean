/-
A row's `b ‖ c` in a satisfying witness of the untyped statement `Stmt.setupH` accepted (`row_bc_setupH`). Without a typed
template Δ is its slot constants' pairs, the rows' constant bits, the units' leaf and cut bits and the wires' bits, and
nothing after (`delta_flat_eq`). Past a row net's input rows Δ writes only slot constants (`Layout.not_inBit`), so the
`sha512x3` and `hm96` slots hold their nets' rows (`shaSlot_flat`, `hmSlot_flat`). At their `cv` and `pad` input bits the
leaf and cut bits are elsewhere (the unit is neither row net), and so are the wires after the rows' own (`RowWires`,
`HmRow.parse`'s wire check): Δ's one copy there is `rowCopies`' or the rows' wires' (`inBit_copyL`). The constant bits
read the pin and the unit range's forced-zero row (`Aliased.zeroPos_block`).
-/
import FlockSoundness.Protocol.Discharge.Hm.Copy
import FlockSoundness.Protocol.Discharge.Aliased.Keys

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll

/-- **Δ without a typed template, pair by pair**: the slot constants' pairs, the rows' constant bits, the units' leaf
and cut bits and the wires' bits (`flatPairs`). -/
theorem delta_flat_eq (c : Circuit) (pin : ℕ) (ht : c.typed = none)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hw : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) :
    (HmRow.delta c pin).1.toList = flatPairs c pin := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d.toList ++ rowPairs c pin) ?_ ?_
  · rw [← rowPairsAll_v1 c pin hv1 hw hs]
    refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rw [segs_none_of_v1 hv1 hs hp]
    refine ⟨_, rfl, ?_⟩
    rw [rowPairsOf, ← List.append_assoc, ← List.append_assoc]
    refine Eq.trans (forIn_range'_append (fun k => (List.range 1024).flatMap fun t => bitPairs c pin
        (msgColOf c g p k + t) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)) _ _
        (fun k _ _ d => ⟨_, rfl, ?_⟩) _) ?_
    · refine forIn_range_append _ _ (fun t _ d => ?_) d
      split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = none :=
          Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine Eq.trans (forIn_range_append (fun t => bitPairs c pin (padCol c g p + t)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)) 1024
        (fun t _ d => ?_) _) ?_
    · split
      · rename_i v hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          some v := hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hv
        have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
          none := Option.eq_none_iff_forall_ne_some.mpr hv
        refine ⟨_, rfl, ?_⟩
        rw [hv']
        simp only [bitPairs, List.append_nil]
    refine (List.append_left_inj _).mpr ?_
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d0 hd0
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairs c pin)
    (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  simp only [ht]
  show d3.toList = _
  rw [hd3, hd2, hd1, hd0, hd]
  rfl

/-- **`HmRow.parse`'s wire check, on the rows' nets**: the circuit's wires begin with the rows' chains and `hm96` inputs
(`HmRow.rowWires`), and no later wire ends in a `sha512x3` or `hm96` slot. -/
def RowWires (c : Circuit) : Prop :=
  ∃ rest, c.wires.toList = (HmRow.rowWires c.ports c.shaNet c.hmNet).toList ++ rest ∧
    ∀ w ∈ rest, w.dst.1 ≠ c.shaNet ∧ w.dst.1 ≠ c.hmNet

section Flat

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- A column at a slot of a net other than `n` is not at `n`'s slots. -/
theorem atSlot_ne {n n' s s' off off' x : ℕ} (hn : n < st.c.nets.size) (hn' : n' < st.c.nets.size) (hne : n' ≠ n)
    (hat : AtSlot st.c n s off x) (hat' : AtSlot st.c n' s' off' x) : False := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (F.check.one_range n hn)
  obtain ⟨hm', hk', -⟩ := rangeOf_spec (F.check.one_range n' hn')
  exact hne (hat'.inj L.disj hm' hk' hm hk hat).1

end Flat

end FlockSoundness.Discharge.Hm
