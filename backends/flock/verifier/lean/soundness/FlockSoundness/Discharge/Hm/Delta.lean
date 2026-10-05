/-
Δ in a typed template statement, pair by pair (`delta_rows`): every slot's constant from the pin (`constPairs`), then
each row's constant bits (`rowPairs`: its first `cv` the row prefix's midstate, its `hm96` slot's `pad` the padding
block, bit 1 from the pin and bit 0 from the unit range's forced-zero row), the units' leaf and cut bits, every wire's
destination bits from its source's, then the typed tail. Δ writes each row's constant bits as `rowPairsOf`, which for
`sha512/row/v1` rows (`16 · words` bits in at least one chunk) are `rowPairs`'s (`rowPairsAll_v1`).
-/
import FlockSoundness.Discharge.Hm.Accept
import FlockSoundness.ExecFlatCopies
import FlockSoundness.Protocol.Discharge.Hm.Delta

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

theorem delta_rows (c : Circuit) (pin : ℕ) {tb : TypedBlock} (ht : c.typed = some tb)
    (hv1 : ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hw : ∀ p ∈ c.ports.toList, p.v2 = false → 2 * p.words = 128 * p.chunks)
    (hs : ∀ p ∈ c.ports.toList, p.v2 = false → p.segs = none) :
    (HmRow.delta c pin).1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      (HmRow.delta c pin).2.toList = flatPairs c pin ++ Typed.tail c tb false := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d => d.toList = constPairs c pin) ?_ ?_
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
  -- the rows' constant bits
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d.toList ++ rowPairs c pin)
    ?_ ?_
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
  -- the units' leaf bits
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  -- the units' leaf cuts
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  -- the wires
  refine FlockSoundness.run_bind_pair (Q := fun r => r.1.toList = flatPairs c pin ++ Typed.tail c tb true ∧
      r.2.toList = flatPairs c pin ++ Typed.tail c tb false) (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c)
    ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  have hflat : d3.toList = flatPairs c pin := by
    rw [hd3, hd2, hd1, hd0, hd]
    rfl
  -- the typed tail
  simp only [ht]
  rw [← hflat]
  have H := fun f hf => forIn_range_append2 (f := f) (Typed.tailVU c tb true) (Typed.tailVU c tb false) c.g hf (d3, d3)
  refine ⟨(H _ ?hf).1, (H _ ?hf).2⟩
  intro g _ s
  refine ⟨_, rfl, ?_⟩
  have HW := fun f hf => forIn_range_append2 (f := f)
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)])
    (fun w => [(Typed.col c tb g tb.inCols[w]!, Typed.col c tb g tb.inCols[w]!),
      (Typed.col c tb g tb.inCols[w]!, Typed.copySrc c g w)]) tb.inCols.size hf
  have HB := fun f hf => forIn_array_append2 (f := f)
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x))
    (fun rs : ℕ × Array ℕ => (Typed.col c tb g rs.1, Typed.col c tb g rs.1) ::
      rs.2.toList.map fun x => (Typed.col c tb g rs.1, Typed.col c tb g x)) tb.bind hf
  have HC := fun f hf => forIn_array_append2 (f := f)
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.1.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x))
    (fun e : ℕ × Array ℕ × Array ℕ => e.2.2.toList.map fun x => (Typed.col c tb g e.1, Typed.col c tb g x)) tb.cross hf
  constructor
  all_goals first
    | refine (HC _ ?_ _).1.trans ?_
    | refine (HC _ ?_ _).2.trans ?_
  all_goals first
    | (intro e _ t
       exact ⟨_, rfl, by (simp only [Array.toList_append, Array.toList_map]; rfl),
         by (simp only [Array.toList_append, Array.toList_map]; rfl)⟩)
    | skip
  all_goals first
    | (dsimp only; rw [(HB _ ?_ _).1, (HW _ ?_ _).1])
    | (dsimp only; rw [(HB _ ?_ _).2, (HW _ ?_ _).2])
  all_goals first
    | (simp [Typed.tailVU]; done)
    | (intro rs _ t
       exact ⟨_, rfl,
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl),
         by (simp only [Array.toList_append, Array.toList_push, Array.toList_map, List.append_assoc]; rfl)⟩)

end FlockSoundness.Discharge.Hm
