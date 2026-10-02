/-
Δ in a typed template statement, pair by pair (`delta_rows`): every slot's constant from the pin (`constPairs`), then
each row's constant bits (`rowPairs`: its first `cv` the row prefix's midstate, its `hm96` slot's `pad` the padding
block, bit 1 from the pin and bit 0 from the unit range's forced-zero row), the units' leaf and cut bits, every wire's
destination bits from its source's, then the typed tail.
-/
import FlockSoundness.Discharge.Hm.Accept
import FlockSoundness.ExecFlatCopies

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

/-- The unit range's first slot's forced-zero row. -/
def zeroCol (c : Circuit) : ℕ := c.unitRange.slot 0 + c.unit.useful

/-- VU `g`, port `p`: its first compression's `cv` input's first column. -/
def cvCol (c : Circuit) (g p : ℕ) : ℕ :=
  c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[0]!.1 +
    (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[0]!.2]!.1

/-- VU `g`, port `p`: its `hm96` slot's `pad` input's first column. -/
def padCol (c : Circuit) (g p : ℕ) : ℕ := c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1

/-- A constant bit's source: the pin for bit 1, the forced-zero row for bit 0. -/
def constSrc (c : Circuit) (pin : ℕ) (ws : Array UInt64) (t : ℕ) : ℕ := if wordsBit ws t then pin else zeroCol c

/-- Δ's pairs at the rows' constant bits. -/
def rowPairs (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p =>
    (List.range 512).flatMap (fun t => [(cvCol c g p + t, cvCol c g p + t),
        (cvCol c g p + t, constSrc c pin (HmRow.midstate c.ports[p]!.words) t)]) ++
      (List.range 1024).flatMap fun t => [(padCol c g p + t, padCol c g p + t),
        (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)]

/-- Δ's pairs, before the typed tail. -/
def flatPairs (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  constPairs c pin ++ rowPairs c pin ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c

theorem delta_rows (c : Circuit) (pin : ℕ) {tb : TypedBlock} (ht : c.typed = some tb) :
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
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p _ d => ⟨_, rfl, ?_⟩) d
    refine Eq.trans (forIn_range_append (fun t => [(padCol c g p + t, padCol c g p + t),
        (padCol c g p + t, constSrc c pin (HmRow.padWords c.ports[p]!.words) t)]) 1024
        (fun t _ d => ⟨_, rfl, ?_⟩) _) ?_
    · simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
      rfl
    rw [← List.append_assoc]
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
