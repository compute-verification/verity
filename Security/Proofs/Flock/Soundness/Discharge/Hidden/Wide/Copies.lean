import Proofs.Flock.Soundness.Discharge.Hidden.Flat

/-!
# Δ's copies at the rows, for `sha512/row/v1` and `sha512/row/v2` rows

Without a typed template Δ is its slot constants' pairs, the rows' constant bits as `HmRow.delta` writes them
(`rowPairsAll`), the units' leaf and cut bits and the wires' bits (`delta_flat_all`), for every row, `sha512/row/v1` or
`sha512/row/v2`, of any number of compressions. Each of the rows' writes is a copy (`rowCopiesAll`): a row's first `cv`
(its first compression's or, for a row of none, its `hm96` slot's), the constant bits of its `pad`, and those of its
compressions' messages from block `bits / 1024` on. The rows' wires (`rowWires_all`) give a row of no compression no
`hm96` wire. Every copy into template instance `g`'s row `p` lands in that row's slots (`Dst`), and distinct rows' slots
are distinct (`dst_inj`), so Δ's copies at a column of row `p` are row `p`'s (`filter_rows`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- Δ's copy at column `x` for a constant bit `HmRow.constBit` gives, none at a row bit. -/
def bitCopy (c : Circuit) (pin x : ℕ) : Option Bool → List (ℕ × ℕ)
  | some v => [(x, if v then pin else zeroCol c)]
  | none => []

/-- Δ's copies into template instance `g`'s row `p`'s first `cv`: the row prefix's midstate. -/
def cvCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range 512).map fun t => (cvColOf c g p + t, constSrc c pin (HmRow.midstateOf c.ports[p]!) t)

/-- Δ's copies into template instance `g`'s row `p`'s `hm96` slot's `pad`: the constant bits of its last block. -/
def padCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range 1024).flatMap fun t => bitCopy c pin (padCol c g p + t)
    (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)

/-- Δ's copies into template instance `g`'s row `p`'s compressions' messages, from the first that holds a bit at or past
the row's length: their constant bits. -/
def msgCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range' (c.ports[p]!.bits / 1024) (((HmRow.rowSlots c.ports).2[p]!).size - c.ports[p]!.bits / 1024)).flatMap
    fun k => (List.range 1024).flatMap fun t => bitCopy c pin (msgColOf c g p k + t)
      (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)

/-- Δ's copies into template instance `g`'s row `p`'s constant bits (`rowPairsOf`'s). -/
def rowCopiesOf (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  cvCopies c pin g p ++ padCopies c pin g p ++ msgCopies c pin g p

/-- Δ's copies into the rows' constant bits. -/
def rowCopiesAll (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => rowCopiesOf c pin g p

theorem bitPairs_eq (c : Circuit) (pin x : ℕ) (v : Option Bool) :
    bitPairs c pin x v = (bitCopy c pin x v).flatMap pair2 := by
  cases v <;> rfl

theorem rowPairsOf_eq (c : Circuit) (pin g p : ℕ) : rowPairsOf c pin g p = (rowCopiesOf c pin g p).flatMap pair2 := by
  simp only [rowPairsOf, rowCopiesOf, cvCopies, padCopies, msgCopies, List.flatMap_append, List.flatMap_map,
    List.flatMap_assoc, bitPairs_eq]
  rfl

theorem rowPairsAll_eq (c : Circuit) (pin : ℕ) : rowPairsAll c pin = (rowCopiesAll c pin).flatMap pair2 := by
  simp only [rowPairsAll, rowCopiesAll, List.flatMap_assoc, rowPairsOf_eq]

/-- Δ's pairs, without a typed template. -/
def flatPairsAll (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  constPairs c pin ++ rowPairsAll c pin ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c

/-- **Δ without a typed template, pair by pair**, for any rows: the slot constants' pairs, the rows' constant bits as
`HmRow.delta` writes them, the units' leaf and cut bits and the wires' bits. -/
theorem delta_flat_all (c : Circuit) (pin : ℕ) (ht : c.typed = none) (hs : ∀ p ∈ c.ports.toList, p.segs = none) :
    (HmRow.delta c pin).1.toList = flatPairsAll c pin := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAll c pin)
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
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAll c pin)
    (P := fun d' => d'.toList = d.toList ++ rowPairsAll c pin) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rw [show c.ports[p]!.segs = none from hs _ (getElem_mem_toList' (rowSlots_size' c.ports ▸ hp))]
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
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAll c pin)
    (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAll c pin)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsAll c pin)
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

/-! ## The rows' wires, for any rows -/

/-- **The rows' wires**, port by port: compression `k`'s `cv` from compression `k − 1`'s output for `k ≥ 1`, then, for
a row of a compression or more, the `hm96` slot's `cv` from the last compression's output. -/
theorem rowWires_all (ports : Array Port) (sha hm : ℕ) (hs : ∀ p ∈ ports.toList, p.segs = none) :
    (HmRow.rowWires ports sha hm).toList = (List.range (HmRow.rowSlots ports).2.size).flatMap fun p =>
      (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).map (chainWire ports sha p) ++
        (if 0 < ((HmRow.rowSlots ports).2[p]!).size then [hmWire ports sha hm p] else []) := by
  unfold HmRow.rowWires
  simp only
  have hS : (HmRow.rowSlots ports).2.size = ports.size := by
    rw [← Array.length_toList, rowSlots_eq, List.length_map, List.length_range]
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (fun p =>
      List.map (chainWire ports sha p) (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)) ++
        (if 0 < ((HmRow.rowSlots ports).2[p]!).size then [hmWire ports sha hm p] else [])) _
      (fun p hp d => ?_) #[]) ?_
  · have hcs := chainStart_unseg (hs _ (getElem_mem_toList' (hS ▸ hp)))
    have hk1 : ∀ k, 1 ≤ k → (HmRow.chainStart ports[p]! k).isNone = true := fun k hk => by
      rw [hcs]; simp [show k ≠ 0 by omega]
    by_cases h0 : 0 < ((HmRow.rowSlots ports).2[p]!).size
    · simp only [hk1 _ h0, h0, decide_true, Bool.and_self, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [Array.toList_push, ← List.append_assoc]
      refine (List.append_left_inj _).mpr ?_
      refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k hk d => ?_) d) ?_
      · simp only [hk1 k hk, ↓reduceIte]
        exact ⟨_, rfl, by rw [Array.toList_push]; rfl⟩
      · rw [List.map_eq_flatMap]
    · simp only [h0, decide_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte, List.append_nil]
      refine ⟨_, rfl, ?_⟩
      refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k hk d => ?_) d) ?_
      · simp only [hk1 k hk, ↓reduceIte]
        exact ⟨_, rfl, by rw [Array.toList_push]; rfl⟩
      · rw [List.map_eq_flatMap]
  · simp

/-- The rows' wires' copies at template instance `g`'s row `p`. -/
def wireFAll (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  ((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map (chainWire c.ports c.shaNet p) ++
    (if 0 < ((HmRow.rowSlots c.ports).2[p]!).size then [hmWire c.ports c.shaNet c.hmNet p] else [])).flatMap
      fun w => (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopiesOf_rowsAll (c : Circuit) (hs : ∀ p ∈ c.ports.toList, p.segs = none) :
    wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireFAll c g p := by
  unfold wireCopiesOf wireFAll
  rw [rowWires_all _ _ _ hs]
  simp only [List.flatMap_assoc]

theorem bitCopy_fst {c : Circuit} {pin x : ℕ} {v : Option Bool} {e : ℕ × ℕ} (he : e ∈ bitCopy c pin x v) : e.1 = x := by
  cases v with
  | none => cases he
  | some v =>
      simp only [bitCopy, List.mem_cons, List.not_mem_nil, or_false] at he
      rw [he]

/-! ## Positions in the rows' slots -/

/-- Bit `off` of compression `k` of template instance `g`'s row `p` (its `cv` below 512, its `msg` from 512): in its
`sha512x3` slot, at `1536 · sub + off`. -/
def shaPos (c : Circuit) (g p k off : ℕ) : ℕ := shaBase c g p k + 1536 * shaSub c p k + off

/-- **The columns of template instance `g`'s row `p`**: its compressions' inputs and its `hm96` slot's. -/
def Dst (c : Circuit) (g p x : ℕ) : Prop :=
  (∃ k off, k < c.ports[p]!.chunks ∧ off < 1536 ∧ x = shaPos c g p k off) ∨
    ∃ off < 1536, x = c.slotCol c.hmNet g p + off

theorem hmIn0 {c : Circuit} (h : HmNets.check c = .ok ()) :
    (portCols (c.nets.getD c.hmNet default).2.inGroups)[0]!.1 = 0 := by
  rw [(hmNet_groups h).1]
  decide

section Pos

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

theorem shaPos_at {g p k off : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) :
    AtSlot c c.shaNet (g * c.perVu.getD c.shaNet 0 + ((HmRow.rowSlots c.ports).2[p]!)[k]!.1)
      (1536 * shaSub c p k + off) (shaPos c g p k off) := by
  have hs := rowSlot_lt hp hk
  refine ⟨R.slot_lt_count (Or.inl rfl) hg (by rw [R.perVu_sha]; exact hs.1),
    R.net_off_lt (Or.inl rfl) (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; unfold shaSub; omega), ?_⟩
  rw [shaPos, shaBase, Circuit.slotCol, Nat.add_assoc]

theorem shaPos_inj {g p k off g' p' k' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) (hg' : g' < c.g) (hp' : p' < c.ports.size) (hk' : k' < c.ports[p']!.chunks) (ho' : off' < 1536)
    (h : shaPos c g p k off = shaPos c g' p' k' off') : g = g' ∧ p = p' ∧ k = k' ∧ off = off' := by
  have hn := R.range R.shaNet
  obtain ⟨-, hs, ho2⟩ := (shaPos_at R hu hg hp hk ho).inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact shaPos_at R hu hg' hp' hk' ho')
  have h1 := rowSlot_lt hp hk
  have h2 := rowSlot_lt hp' hk'
  obtain ⟨rfl, hs⟩ := slot_inj (by rw [R.perVu_sha]; exact h1.1) (by rw [R.perVu_sha]; exact h2.1) hs
  unfold shaSub at ho2
  rw [rowSlots_entry hp hk, rowSlots_entry hp' hk'] at hs ho2
  simp only at hs ho2
  have : atOf c.ports p + k = atOf c.ports p' + k' := by omega
  obtain ⟨rfl, rfl⟩ := at_inj hp hp' hk hk' this
  exact ⟨rfl, rfl, rfl, by omega⟩

theorem shaPos_ne_hm {g p k off g' p' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) (hg' : g' < c.g) (hp' : p' < c.ports.size) (ho' : off' < 4096) :
    shaPos c g p k off ≠ c.slotCol c.hmNet g' p' + off' := by
  intro h
  have hn := R.range R.shaNet
  have hm := R.range R.hmNet
  exact sha_ne_hm R.rowNets ((shaPos_at R hu hg hp hk ho).inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [h]; exact R.hmIn_at hg' hp' ho')).1

theorem shaIn_pos {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (t : ℕ) :
    shaBase c g p k + shaIn c (2 * shaSub c p k) + t = shaPos c g p k t := by
  rw [(shaIn_eq R.rowNets (shaSub c p k) (rowSlot_lt hp hk).2).1, shaPos]

theorem msgColOf_pos {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (t : ℕ) :
    msgColOf c g p k + t = shaPos c g p k (512 + t) := by
  show shaBase c g p k + shaIn c (2 * shaSub c p k + 1) + t = _
  rw [(shaIn_eq R.rowNets (shaSub c p k) (rowSlot_lt hp hk).2).2, shaPos]
  omega

theorem deltaMsgCol_pos {g p b t : ℕ} (hp : p < c.ports.size) (hk : b / 128 < c.ports[p]!.chunks) :
    deltaMsgCol c g p b t = shaPos c g p (b / 128) (512 + msgOff b t) := by
  rw [deltaMsgCol_eq _ _ _ _ _ (segs_getD hu hp) hp hk, (shaIn_eq R.rowNets (shaSub c p (b / 128)) (rowSlot_lt hp hk).2).2, shaPos]
  omega

theorem cvColOf_pos {g p : ℕ} (hp : p < c.ports.size) (h0 : 0 < c.ports[p]!.chunks) (t : ℕ) :
    cvColOf c g p + t = shaPos c g p 0 t := by
  rw [cvColOf, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  exact shaIn_pos R hu hp h0 t

theorem cvColOf_zero {g p : ℕ} (hp : p < c.ports.size) (h0 : c.ports[p]!.chunks = 0) (t : ℕ) :
    cvColOf c g p + t = c.slotCol c.hmNet g p + t := by
  have h0' : ¬ 0 < c.ports[p]!.chunks := by omega
  rw [cvColOf, rowSlots_size hp]
  simp only [h0', ↓reduceIte, hmIn0 R.rowNets, Nat.add_zero]

theorem chainDst_pos {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (t : ℕ) :
    Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t = shaPos c g p k t := by
  rw [chainDst_eq R.rowNets (rowSlot_lt hp hk).2]
  exact shaIn_pos R hu hp hk t

/-- Distinct rows' columns are distinct. -/
theorem dst_inj {g p g' p' x : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hg' : g' < c.g)
    (hp' : p' < c.ports.size) (h : Dst c g p x) (h' : Dst c g' p' x) : g = g' ∧ p = p' := by
  rcases h with ⟨k, off, hk, ho, rfl⟩ | ⟨off, ho, rfl⟩ <;>
    rcases h' with ⟨k', off', hk', ho', he⟩ | ⟨off', ho', he⟩
  · obtain ⟨h1, h2, -⟩ := shaPos_inj R hu hg hp hk ho hg' hp' hk' ho' he
    exact ⟨h1, h2⟩
  · exact absurd he (shaPos_ne_hm R hu hg hp hk ho hg' hp' (by omega))
  · exact absurd he.symm (shaPos_ne_hm R hu hg' hp' hk' ho' hg hp (by omega))
  · obtain ⟨h1, h2, -⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) he
    exact ⟨h1, h2⟩

omit R hu in
theorem cvCopies_mem {pin g p : ℕ} {e : ℕ × ℕ} (he : e ∈ cvCopies c pin g p) :
    ∃ t < 512, e = (cvColOf c g p + t, constSrc c pin (HmRow.midstateOf c.ports[p]!) t) := by
  simp only [cvCopies, List.mem_map, List.mem_range] at he
  obtain ⟨t, ht, rfl⟩ := he
  exact ⟨t, ht, rfl⟩

theorem padCopies_mem {pin g p : ℕ} {e : ℕ × ℕ} (he : e ∈ padCopies c pin g p) :
    ∃ t < 1024, e.1 = c.slotCol c.hmNet g p + (512 + t) := by
  simp only [padCopies, List.mem_flatMap, List.mem_range] at he
  obtain ⟨t, ht, he⟩ := he
  exact ⟨t, ht, by rw [bitCopy_fst he, padCol_eq R.rowNets, Nat.add_assoc]⟩

theorem msgCopies_mem {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ msgCopies c pin g p) :
    ∃ k t, c.ports[p]!.bits / 1024 ≤ k ∧ k < c.ports[p]!.chunks ∧ t < 1024 ∧ e.1 = shaPos c g p k (512 + t) := by
  simp only [msgCopies, List.mem_flatMap, List.mem_range, List.mem_range'_1, rowSlots_size hp] at he
  obtain ⟨k, hk, t, ht, he⟩ := he
  exact ⟨k, t, hk.1, by omega, ht, by rw [bitCopy_fst he, msgColOf_pos R hu hp (by omega)]⟩

theorem cvCopies_dst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ cvCopies c pin g p) :
    Dst c g p e.1 := by
  obtain ⟨t, ht, rfl⟩ := cvCopies_mem he
  by_cases h0 : 0 < c.ports[p]!.chunks
  · exact Or.inl ⟨0, t, h0, by omega, cvColOf_pos R hu hp h0 t⟩
  · exact Or.inr ⟨t, by omega, cvColOf_zero R hu hp (by omega) t⟩

theorem rowCopiesOf_dst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ rowCopiesOf c pin g p) :
    Dst c g p e.1 := by
  simp only [rowCopiesOf, List.mem_append] at he
  rcases he with (he | he) | he
  · exact cvCopies_dst R hu hp he
  · obtain ⟨t, ht, he⟩ := padCopies_mem R hu he
    exact Or.inr ⟨512 + t, by omega, he⟩
  · obtain ⟨k, t, -, hk, ht, he⟩ := msgCopies_mem R hu hp he
    exact Or.inl ⟨k, 512 + t, hk, by omega, he⟩

theorem wireFAll_mem {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFAll c g p) :
    (∃ k t, 1 ≤ k ∧ k < c.ports[p]!.chunks ∧ t < 512 ∧
        e = (shaPos c g p k t, Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t)) ∨
      (0 < c.ports[p]!.chunks ∧ ∃ t < 512,
        e = (c.slotCol c.hmNet g p + t, Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t)) := by
  unfold wireFAll at he
  rw [rowSlots_size hp] at he
  by_cases h0 : 0 < c.ports[p]!.chunks
  · simp only [h0, ↓reduceIte, List.flatMap_append, List.flatMap_map, List.mem_append, List.mem_flatMap,
      List.mem_range'_1, List.mem_map, List.mem_range, List.flatMap_cons, List.flatMap_nil, List.append_nil] at he
    rcases he with ⟨k, hk, t, ht, rfl⟩ | ⟨t, ht, rfl⟩
    · have hs := (rowSlot_lt hp (show k < c.ports[p]!.chunks by omega)).2
      rw [chainWidth_eq R.rowNets hs] at ht
      exact Or.inl ⟨k, t, hk.1, by omega, ht, by rw [chainDst_pos R hu hp (by omega)]⟩
    · rw [hmWidth_eq R.rowNets] at ht
      refine Or.inr ⟨h0, t, ht, ?_⟩
      show (hmWireDst c g p + t, _) = _
      rw [hmWireDst_eq R.rowNets]
  · simp only [h0, ↓reduceIte, List.append_nil, List.flatMap_map, List.mem_flatMap, List.mem_range'_1,
      List.mem_map, List.mem_range] at he
    obtain ⟨k, hk, -⟩ := he
    omega

theorem wireFAll_dst {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFAll c g p) : Dst c g p e.1 := by
  rcases wireFAll_mem R hu hp he with ⟨k, t, -, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
  · exact Or.inl ⟨k, t, hk, by omega, rfl⟩
  · exact Or.inr ⟨t, by omega, rfl⟩

/-- **Δ's copies at a column of template instance `g`'s row `p`** are that row's: its constant bits' and its wires'. -/
theorem filter_rows {pin g p x : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hx : Dst c g p x) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = x) =
      (rowCopiesOf c pin g p).filter (fun e => e.1 = x) ++ (wireFAll c g p).filter (fun e => e.1 = x) := by
  have hsz := rowSlots_size' c.ports
  rw [List.filter_append, wireCopiesOf_rowsAll _ hu]
  unfold rowCopiesAll
  rw [filter_flatMap2_single _ _ _ x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hu hg' (hsz ▸ hp') hg hp (hex ▸ rowCopiesOf_dst R hu (hsz ▸ hp') he) hx),
    filter_flatMap2_single _ _ (fun g p => wireFAll c g p) x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hu hg' (hsz ▸ hp') hg hp (hex ▸ wireFAll_dst R hu (hsz ▸ hp') he) hx)]

theorem cvCopies_fst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ cvCopies c pin g p) :
    (0 < c.ports[p]!.chunks ∧ ∃ t < 512, e.1 = shaPos c g p 0 t) ∨
      (c.ports[p]!.chunks = 0 ∧ ∃ t < 512, e.1 = c.slotCol c.hmNet g p + t) := by
  obtain ⟨t, ht, rfl⟩ := cvCopies_mem he
  by_cases h0 : 0 < c.ports[p]!.chunks
  · exact Or.inl ⟨h0, t, ht, cvColOf_pos R hu hp h0 t⟩
  · exact Or.inr ⟨by omega, t, ht, cvColOf_zero R hu hp (by omega) t⟩

/-! ## Δ's copies at a row's own columns -/

omit R hu in
theorem filter_bitCopy (c : Circuit) (pin x : ℕ) (v : Option Bool) :
    (bitCopy c pin x v).filter (fun e => e.1 = x) = bitCopy c pin x v := by
  cases v <;> simp [bitCopy]

/-- **Δ's one copy into template instance `g`'s row `p`'s first compression's `cv` bit `t₀`**: the row prefix's midstate
bit. -/
theorem cv0_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (h0 : 0 < c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = shaPos c g p 0 t₀) =
      [(shaPos c g p 0 t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
  rw [filter_rows R hu hg hp (Or.inl ⟨0, t₀, h0, by omega, rfl⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAll_mem R hu hp he with ⟨k, t, hk1, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
    · intro h
      have := (shaPos_inj R hu hg hp hk (by omega) hg hp h0 (by omega) h).2.2.1
      omega
    · exact fun h => shaPos_ne_hm R hu hg hp h0 (by omega) hg hp (by omega) h.symm
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨t, ht, he1⟩ := padCopies_mem R hu he
    rw [he1]
    exact fun h => shaPos_ne_hm R hu hg hp h0 (by omega) hg hp (by omega) h.symm
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨k, t, -, hk, ht, he1⟩ := msgCopies_mem R hu hp he
    rw [he1]
    intro h
    have := (shaPos_inj R hu hg hp hk (by omega) hg hp h0 (by omega) h).2.2.2
    omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) =
      [(shaPos c g p 0 t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
    rw [← cvColOf_pos R hu hp h0 t₀]
    unfold cvCopies
    exact filter_map_single _ _ _ _ ht₀
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg]
  rfl

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

/-- **Δ's one copy into the `hm96` slot's `cv` bit `t₀`** of a row of a compression or more: the wire from the last
compression's output. -/
theorem hmcv_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (h0 : 0 < c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) =
      [(Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀,
        Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t₀)] := by
  have hx : Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀ = c.slotCol c.hmNet g p + t₀ := by
    show hmWireDst c g p + t₀ = _
    rw [hmWireDst_eq R.rowNets]
  rw [filter_rows R hu hg hp (Or.inr ⟨t₀, by omega, hx⟩)]
  have hr : (rowCopiesOf c pin g p).filter
      (fun e => e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hu hp he with ⟨-, t, ht, he1⟩ | ⟨h, -⟩
      · rw [he1]
        exact shaPos_ne_hm R hu hg hp h0 (by omega) hg hp (by omega)
      · omega
    · obtain ⟨t, ht, he1⟩ := padCopies_mem R hu he
      rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
    · obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_mem R hu hp he
      rw [he1]
      exact shaPos_ne_hm R hu hg hp hk' (by omega) hg hp (by omega)
  have hc := filter_nil_of (((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map
      (chainWire c.ports c.shaNet p)).flatMap fun w =>
        (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t))
    (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) fun e he => by
      simp only [List.flatMap_map, List.mem_flatMap, List.mem_range'_1, List.mem_map, List.mem_range] at he
      obtain ⟨k', hk', t, ht, rfl⟩ := he
      rw [rowSlots_size hp] at hk'
      have hk2 : k' < c.ports[p]!.chunks := by omega
      have hs' := (rowSlot_lt hp hk2).2
      rw [chainWidth_eq R.rowNets hs'] at ht
      rw [chainDst_pos R hu hp hk2, hx]
      exact shaPos_ne_hm R hu hg hp hk2 (by omega) hg hp (by omega)
  rw [hr, List.nil_append]
  unfold wireFAll
  rw [List.flatMap_append, List.filter_append, hc, List.nil_append, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  rw [List.flatMap_cons, List.flatMap_nil, List.append_nil, hmWidth_eq R.rowNets, filter_map_single _ _ _ _ ht₀]

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

/-- **Δ's copy into the `hm96` slot's `pad` bit `t₀`**: the constant bit of the row's last block there. -/
theorem pad_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht₀ : t₀ < 1024) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = padCol c g p + t₀) =
      bitCopy c pin (padCol c g p + t₀)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t₀) := by
  have hx : padCol c g p + t₀ = c.slotCol c.hmNet g p + (512 + t₀) := by rw [padCol_eq R.rowNets, Nat.add_assoc]
  rw [filter_rows R hu hg hp (Or.inr ⟨512 + t₀, by omega, hx⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases wireFAll_mem R hu hp he with ⟨k, t, -, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
    · exact shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega)
    · intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases cvCopies_fst R hu hp he with ⟨h0, t, ht, he1⟩ | ⟨-, t, ht, he1⟩
    · rw [he1]
      exact shaPos_ne_hm R hu hg hp h0 (by omega) hg hp (by omega)
    · rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    obtain ⟨k, t, -, hk, ht, he1⟩ := msgCopies_mem R hu hp he
    rw [he1]
    exact shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega)
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = padCol c g p + t₀) = bitCopy c pin (padCol c g p + t₀)
      (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t₀) := by
    unfold padCopies
    rw [filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t _ hne =>
      filter_nil_of _ _ fun e he => by rw [bitCopy_fst he]; omega]
    exact filter_bitCopy _ _ _ _
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg, List.nil_append, List.append_nil]

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
