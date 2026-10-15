import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Copies
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegShape

/-!
# Δ at a segmented row, and Δ for rows of every kind

`HmRow.delta` writes a `sha512/row-seg/v1` row's constants compression by compression, its `hm96` slot as compression
`w.size` (`segPairsOf`): at a chain start (`HmRow.chainStart`) the `cv` is the midstate of the segment's first block, or of
the outer message's for `j = s`, and every message bit that is not a row bit (`HmRow.msgBit`) is a constant from the pin or
the forced-zero row, or a copy of a segment's last output. `delta_flat_any` is `delta_flat_all` for rows of every kind,
each row's pairs `rowPairsAny`: `rowPairsOf`'s for an unsegmented row, `segPairsOf`'s for a segmented one.
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- Template instance `g`, segmented row `p`: compression `k`'s `cv` and `msg` inputs' first columns, the `hm96` slot's
`cv` and `pad` for `k = w.size`. -/
def segCols (c : Circuit) (g p k : ℕ) : ℕ × ℕ :=
  if k < ((HmRow.rowSlots c.ports).2[p]!).size then
    (c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
      (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2]!.1,
     c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
      (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + 1]!.1)
  else (c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[0]!.1,
    c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1)

def segCvCol (c : Circuit) (g p k : ℕ) : ℕ := (segCols c g p k).1

def segMsgCol (c : Circuit) (g p k : ℕ) : ℕ := (segCols c g p k).2

/-- A segment's digest bit's source: column `col` of compression `kk`'s output. -/
def digestSrc (c : Circuit) (g p kk col : ℕ) : ℕ :=
  c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[kk]!.1 +
    (portCols (c.nets.getD c.shaNet default).2.outGroups)[((HmRow.rowSlots c.ports).2[p]!)[kk]!.2]!.1 + col

/-- Δ's pairs at column `x` of a segmented row's message bit. -/
def segBitPairs (c : Circuit) (pin g p x : ℕ) : HmRow.MsgBit → List (ℕ × ℕ)
  | .const v => [(x, x), (x, if v then pin else zeroCol c)]
  | .digest kk col => [(x, x), (x, digestSrc c g p kk col)]
  | .row _ => []

/-- Δ's pairs at compression `k` of template instance `g`'s segmented row `p`: its chain start's midstate, then its
message's constant and digest bits. -/
def segPairsK (c : Circuit) (pin g p : ℕ) (segs : Array (ℕ × ℕ)) (k : ℕ) : List (ℕ × ℕ) :=
  (match HmRow.chainStart c.ports[p]! k with
    | some j => (List.range 512).flatMap fun t => [(segCvCol c g p k + t, segCvCol c g p k + t),
        (segCvCol c g p k + t, constSrc c pin
          (if j < segs.size then HmRow.segMidstate c.ports[p]! j else HmRow.midstateOf c.ports[p]!) t)]
    | none => []) ++
  (List.range 1024).flatMap fun t => segBitPairs c pin g p (segMsgCol c g p k + t) (HmRow.msgBit c.ports[p]! k t)

/-- Δ's pairs at template instance `g`'s segmented row `p`, its compressions then its `hm96` slot. -/
def segPairsOf (c : Circuit) (pin g p : ℕ) (segs : Array (ℕ × ℕ)) : List (ℕ × ℕ) :=
  (List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1)).flatMap (segPairsK c pin g p segs)

/-- Δ's pairs at template instance `g`'s row `p`, of either kind. -/
def rowPairsAny (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  match c.ports[p]!.segs with
  | none => rowPairsOf c pin g p
  | some segs => segPairsOf c pin g p segs

def rowPairsAnyAll (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => rowPairsAny c pin g p

/-- Δ's pairs, without a typed template, for rows of every kind. -/
def flatPairsAny (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  constPairs c pin ++ rowPairsAnyAll c pin ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c

theorem rowPairsAnyAll_unseg {c : Circuit} (pin : ℕ) (hs : ∀ p ∈ c.ports.toList, p.segs = none) :
    rowPairsAnyAll c pin = rowPairsAll c pin := by
  unfold rowPairsAnyAll rowPairsAll
  refine flatMap_congr' fun g _ => flatMap_congr' fun p hp => ?_
  have hS : (HmRow.rowSlots c.ports).2.size = c.ports.size := rowSlots_size' c.ports
  rw [rowPairsAny, show c.ports[p]!.segs = none from hs _ (getElem_mem_toList' (hS ▸ List.mem_range.1 hp))]

/-- **Δ without a typed template, pair by pair, for rows of every kind.** -/
theorem delta_flat_any (c : Circuit) (pin : ℕ) (ht : c.typed = none) :
    (HmRow.delta c pin).1.toList = flatPairsAny c pin := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAny c pin)
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
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAny c pin)
    (P := fun d' => d'.toList = d.toList ++ rowPairsAnyAll c pin) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rcases hsp : c.ports[p]!.segs with _ | segs
    · refine ⟨_, rfl, ?_⟩
      rw [rowPairsAny, hsp]
      dsimp only
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
    · refine ⟨_, rfl, ?_⟩
      rw [rowPairsAny, hsp]
      dsimp only
      refine forIn_range_append (segPairsK c pin g p segs) _ (fun k _ d => ?_) d
      have hbits : ∀ d : Array (ℕ × ℕ), (forIn (m := Id) [:1024] d fun t (d : Array (ℕ × ℕ)) =>
            match HmRow.msgBit c.ports[p]! k t with
            | HmRow.MsgBit.const v =>
              pure (ForInStep.yield ((d.push (segMsgCol c g p k + t, segMsgCol c g p k + t)).push
                (segMsgCol c g p k + t, if v = true then pin else zeroCol c)))
            | HmRow.MsgBit.digest kk col =>
              pure (ForInStep.yield ((d.push (segMsgCol c g p k + t, segMsgCol c g p k + t)).push
                (segMsgCol c g p k + t, digestSrc c g p kk col)))
            | HmRow.MsgBit.row _ => pure (ForInStep.yield d)).toList =
          d.toList ++ (List.range 1024).flatMap fun t =>
            segBitPairs c pin g p (segMsgCol c g p k + t) (HmRow.msgBit c.ports[p]! k t) := by
        intro d
        refine forIn_range_append _ _ (fun t _ d => ?_) d
        cases HmRow.msgBit c.ports[p]! k t <;> exact ⟨_, rfl, by simp [segBitPairs]⟩
      rcases hcs : HmRow.chainStart c.ports[p]! k with _ | j
      · refine ⟨_, rfl, ?_⟩
        rw [segPairsK, hcs, List.nil_append]
        exact hbits d
      · refine ⟨_, rfl, ?_⟩
        rw [segPairsK, hcs, ← List.append_assoc]
        refine (hbits _).trans ((List.append_left_inj _).mpr ?_)
        refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
        simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
  intro d0 hd0
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAny c pin)
    (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAny c pin)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAny c pin)
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

/-! ## The rows' wires, for rows of every kind -/

/-- Row `p`'s wires: compression `k ≥ 1`'s `cv` from compression `k − 1`'s output where no chain starts at `k`, then
the `hm96` slot's `cv` from the last compression's output when the row has one and no chain starts at the slot. -/
def rowWiresOf (ports : Array Port) (sha hm p : ℕ) : List Wire :=
  (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).flatMap (fun k =>
      if (HmRow.chainStart ports[p]! k).isNone then [chainWire ports sha p k] else []) ++
    (if 0 < ((HmRow.rowSlots ports).2[p]!).size &&
        (HmRow.chainStart ports[p]! ((HmRow.rowSlots ports).2[p]!).size).isNone then [hmWire ports sha hm p] else [])

/-- **The rows' wires**, port by port, for rows of every kind (`rowWiresOf`). -/
theorem rowWires_any (ports : Array Port) (sha hm : ℕ) :
    (HmRow.rowWires ports sha hm).toList =
      (List.range (HmRow.rowSlots ports).2.size).flatMap (rowWiresOf ports sha hm) := by
  unfold HmRow.rowWires
  simp only
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (rowWiresOf ports sha hm) _ (fun p _ d => ?_) #[]) (by simp)
  have hloop : ∀ d : Array Wire, ∃ d', (forIn (m := Id) [1:((HmRow.rowSlots ports).2[p]!).size] d fun k (r : Array Wire) =>
        if (HmRow.chainStart ports[p]! k).isNone = true then
          pure (ForInStep.yield (r.push ⟨(sha, ((HmRow.rowSlots ports).2[p]!)[k]!.1, 2 * ((HmRow.rowSlots ports).2[p]!)[k]!.2),
            (sha, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.1, ((HmRow.rowSlots ports).2[p]!)[k - 1]!.2)⟩))
        else pure (ForInStep.yield r)) = d' ∧
      d'.toList = d.toList ++ (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).flatMap (fun k =>
        if (HmRow.chainStart ports[p]! k).isNone then [chainWire ports sha p k] else []) := fun d =>
    ⟨_, rfl, forIn_range1_append _ _ (fun k _ d => by
      by_cases h : (HmRow.chainStart ports[p]! k).isNone = true
      · simp only [h, ↓reduceIte]
        exact ⟨_, rfl, by rw [Array.toList_push]; rfl⟩
      · simp only [h, ↓reduceIte, Bool.false_eq_true]
        exact ⟨_, rfl, by simp⟩) d⟩
  obtain ⟨d', hd', hd'l⟩ := hloop d
  unfold rowWiresOf
  by_cases h : (0 < ((HmRow.rowSlots ports).2[p]!).size &&
      (HmRow.chainStart ports[p]! ((HmRow.rowSlots ports).2[p]!).size).isNone) = true
  · simp only [h, ↓reduceIte]
    refine ⟨_, rfl, ?_⟩
    rw [Array.toList_push, ← List.append_assoc, hd', hd'l]
    rfl
  · simp only [h, Bool.false_eq_true, ↓reduceIte, List.append_nil]
    exact ⟨_, rfl, by rw [hd', hd'l]⟩

/-- The rows' wires' copies at template instance `g`'s row `p`, for rows of every kind. -/
def wireFAny (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  (rowWiresOf c.ports c.shaNet c.hmNet p).flatMap
    fun w => (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopiesOf_rowsAny (c : Circuit) :
    wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireFAny c g p := by
  unfold wireCopiesOf wireFAny
  rw [rowWires_any]
  simp only [List.flatMap_assoc]

end FlockSoundness.Discharge.Hidden
