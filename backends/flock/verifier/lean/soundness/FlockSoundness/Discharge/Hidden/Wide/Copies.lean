import FlockSoundness.Discharge.Hidden.Flat

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

/-- Δ's copy at column `x` for a constant bit `HmRow.constBit` gives, none at a row bit. -/
def bitCopy (c : Circuit) (pin x : ℕ) : Option Bool → List (ℕ × ℕ)
  | some v => [(x, if v then pin else zeroCol c)]
  | none => []

/-- Δ's copies into VU `g`'s row `p`'s first `cv`: the row prefix's midstate. -/
def cvCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range 512).map fun t => (cvColOf c g p + t, constSrc c pin (HmRow.midstateOf c.ports[p]!) t)

/-- Δ's copies into VU `g`'s row `p`'s `hm96` slot's `pad`: the constant bits of its last block. -/
def padCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range 1024).flatMap fun t => bitCopy c pin (padCol c g p + t)
    (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)

/-- Δ's copies into VU `g`'s row `p`'s compressions' messages, from the first that holds a bit at or past the row's
length: their constant bits. -/
def msgCopies (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range' (c.ports[p]!.bits / 1024) (((HmRow.rowSlots c.ports).2[p]!).size - c.ports[p]!.bits / 1024)).flatMap
    fun k => (List.range 1024).flatMap fun t => bitCopy c pin (msgColOf c g p k + t)
      (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)

/-- Δ's copies into VU `g`'s row `p`'s constant bits (`rowPairsOf`'s). -/
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
theorem delta_flat_all (c : Circuit) (pin : ℕ) (ht : c.typed = none) :
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
    refine forIn_range_append _ _ (fun p _ d => ⟨_, rfl, ?_⟩) d
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
theorem rowWires_all (ports : Array Port) (sha hm : ℕ) :
    (HmRow.rowWires ports sha hm).toList = (List.range (HmRow.rowSlots ports).2.size).flatMap fun p =>
      (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).map (chainWire ports sha p) ++
        (if 0 < ((HmRow.rowSlots ports).2[p]!).size then [hmWire ports sha hm p] else []) := by
  unfold HmRow.rowWires
  simp only
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (fun p =>
      List.map (chainWire ports sha p) (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)) ++
        (if 0 < ((HmRow.rowSlots ports).2[p]!).size then [hmWire ports sha hm p] else [])) _
      (fun p _ d => ?_) #[]) ?_
  · by_cases h0 : 0 < ((HmRow.rowSlots ports).2[p]!).size
    · simp only [h0, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [Array.toList_push, ← List.append_assoc]
      refine (List.append_left_inj _).mpr ?_
      refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k d => ⟨_, rfl, ?_⟩) d) ?_
      · rw [Array.toList_push]
        rfl
      · rw [List.map_eq_flatMap]
    · simp only [h0, ↓reduceIte, List.append_nil]
      refine ⟨_, rfl, ?_⟩
      refine Eq.trans (forIn_range1_append (fun k => [chainWire ports sha p k]) _ (fun k d => ⟨_, rfl, ?_⟩) d) ?_
      · rw [Array.toList_push]
        rfl
      · rw [List.map_eq_flatMap]
  · simp

/-- The rows' wires' copies at VU `g`'s row `p`. -/
def wireFAll (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  ((List.range' 1 (((HmRow.rowSlots c.ports).2[p]!).size - 1)).map (chainWire c.ports c.shaNet p) ++
    (if 0 < ((HmRow.rowSlots c.ports).2[p]!).size then [hmWire c.ports c.shaNet c.hmNet p] else [])).flatMap
      fun w => (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopiesOf_rowsAll (c : Circuit) :
    wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireFAll c g p := by
  unfold wireCopiesOf wireFAll
  rw [rowWires_all]
  simp only [List.flatMap_assoc]

theorem bitCopy_fst {c : Circuit} {pin x : ℕ} {v : Option Bool} {e : ℕ × ℕ} (he : e ∈ bitCopy c pin x v) : e.1 = x := by
  cases v with
  | none => cases he
  | some v =>
      simp only [bitCopy, List.mem_cons, List.not_mem_nil, or_false] at he
      rw [he]

/-! ## Positions in the rows' slots -/

/-- Bit `off` of compression `k` of VU `g`'s row `p` (its `cv` below 512, its `msg` from 512): in its `sha512x3` slot,
at `1536 · sub + off`. -/
def shaPos (c : Circuit) (g p k off : ℕ) : ℕ := shaBase c g p k + 1536 * shaSub c p k + off

/-- **The columns of VU `g`'s row `p`**: its compressions' inputs and its `hm96` slot's. -/
def Dst (c : Circuit) (g p x : ℕ) : Prop :=
  (∃ k off, k < c.ports[p]!.chunks ∧ off < 1536 ∧ x = shaPos c g p k off) ∨
    ∃ off < 1536, x = c.slotCol c.hmNet g p + off

theorem hmIn0 {c : Circuit} (h : HmNets.check c = .ok ()) :
    (portCols (c.nets.getD c.hmNet default).2.inGroups)[0]!.1 = 0 := by
  rw [(hmNet_groups h).1]
  decide

section Pos

variable {c : Circuit} (R : RowRanges c)
include R

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
  obtain ⟨-, hs, ho2⟩ := (shaPos_at R hg hp hk ho).inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact shaPos_at R hg' hp' hk' ho')
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
  exact sha_ne_hm R.rowNets ((shaPos_at R hg hp hk ho).inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
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
  rw [deltaMsgCol_eq, (shaIn_eq R.rowNets (shaSub c p (b / 128)) (rowSlot_lt hp hk).2).2, shaPos]
  omega

theorem cvColOf_pos {g p : ℕ} (hp : p < c.ports.size) (h0 : 0 < c.ports[p]!.chunks) (t : ℕ) :
    cvColOf c g p + t = shaPos c g p 0 t := by
  rw [cvColOf, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  exact shaIn_pos R hp h0 t

theorem cvColOf_zero {g p : ℕ} (hp : p < c.ports.size) (h0 : c.ports[p]!.chunks = 0) (t : ℕ) :
    cvColOf c g p + t = c.slotCol c.hmNet g p + t := by
  have h0' : ¬ 0 < c.ports[p]!.chunks := by omega
  rw [cvColOf, rowSlots_size hp]
  simp only [h0', ↓reduceIte, hmIn0 R.rowNets, Nat.add_zero]

theorem chainDst_pos {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (t : ℕ) :
    Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t = shaPos c g p k t := by
  rw [chainDst_eq R.rowNets (rowSlot_lt hp hk).2]
  exact shaIn_pos R hp hk t

/-- Distinct rows' columns are distinct. -/
theorem dst_inj {g p g' p' x : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hg' : g' < c.g)
    (hp' : p' < c.ports.size) (h : Dst c g p x) (h' : Dst c g' p' x) : g = g' ∧ p = p' := by
  rcases h with ⟨k, off, hk, ho, rfl⟩ | ⟨off, ho, rfl⟩ <;>
    rcases h' with ⟨k', off', hk', ho', he⟩ | ⟨off', ho', he⟩
  · obtain ⟨h1, h2, -⟩ := shaPos_inj R hg hp hk ho hg' hp' hk' ho' he
    exact ⟨h1, h2⟩
  · exact absurd he (shaPos_ne_hm R hg hp hk ho hg' hp' (by omega))
  · exact absurd he.symm (shaPos_ne_hm R hg' hp' hk' ho' hg hp (by omega))
  · obtain ⟨h1, h2, -⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) he
    exact ⟨h1, h2⟩

omit R in
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
  exact ⟨k, t, hk.1, by omega, ht, by rw [bitCopy_fst he, msgColOf_pos R hp (by omega)]⟩

theorem cvCopies_dst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ cvCopies c pin g p) :
    Dst c g p e.1 := by
  obtain ⟨t, ht, rfl⟩ := cvCopies_mem he
  by_cases h0 : 0 < c.ports[p]!.chunks
  · exact Or.inl ⟨0, t, h0, by omega, cvColOf_pos R hp h0 t⟩
  · exact Or.inr ⟨t, by omega, cvColOf_zero R hp (by omega) t⟩

theorem rowCopiesOf_dst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ rowCopiesOf c pin g p) :
    Dst c g p e.1 := by
  simp only [rowCopiesOf, List.mem_append] at he
  rcases he with (he | he) | he
  · exact cvCopies_dst R hp he
  · obtain ⟨t, ht, he⟩ := padCopies_mem R he
    exact Or.inr ⟨512 + t, by omega, he⟩
  · obtain ⟨k, t, -, hk, ht, he⟩ := msgCopies_mem R hp he
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
      exact Or.inl ⟨k, t, hk.1, by omega, ht, by rw [chainDst_pos R hp (by omega)]⟩
    · rw [hmWidth_eq R.rowNets] at ht
      refine Or.inr ⟨h0, t, ht, ?_⟩
      show (hmWireDst c g p + t, _) = _
      rw [hmWireDst_eq R.rowNets]
  · simp only [h0, ↓reduceIte, List.append_nil, List.flatMap_map, List.mem_flatMap, List.mem_range'_1,
      List.mem_map, List.mem_range] at he
    obtain ⟨k, hk, -⟩ := he
    omega

theorem wireFAll_dst {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFAll c g p) : Dst c g p e.1 := by
  rcases wireFAll_mem R hp he with ⟨k, t, -, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
  · exact Or.inl ⟨k, t, hk, by omega, rfl⟩
  · exact Or.inr ⟨t, by omega, rfl⟩

/-- **Δ's copies at a column of VU `g`'s row `p`** are that row's: its constant bits' and its wires'. -/
theorem filter_rows {pin g p x : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hx : Dst c g p x) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = x) =
      (rowCopiesOf c pin g p).filter (fun e => e.1 = x) ++ (wireFAll c g p).filter (fun e => e.1 = x) := by
  have hsz := rowSlots_size' c.ports
  rw [List.filter_append, wireCopiesOf_rowsAll]
  unfold rowCopiesAll
  rw [filter_flatMap2_single _ _ _ x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hg' (hsz ▸ hp') hg hp (hex ▸ rowCopiesOf_dst R (hsz ▸ hp') he) hx),
    filter_flatMap2_single _ _ (fun g p => wireFAll c g p) x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hg' (hsz ▸ hp') hg hp (hex ▸ wireFAll_dst R (hsz ▸ hp') he) hx)]

end Pos

end FlockSoundness.Discharge.Hidden
