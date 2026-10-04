import FlockSoundness.Discharge.ScopeW.Delta

set_option Elab.async false

/-!
# Where Δ's copies into the rows land, for rows of any shape

Every copy into VU `g`'s row `p` (`rowCopiesW`'s and the rows' wires', `rowWires_W`) lands at an input bit of one of
the row's compressions or of its `hm96` slot (`inPos`), and distinct rows' input bits are distinct (`inPos_inj`). So
Δ's copies at an input bit of row `p` are row `p`'s (`filter_rowsW`): at a `cv` bit, a chain start's midstate
(`filter_cv_start`) or the previous compression's output (`filter_cv_chain`); at a message bit, the copy `HmRow.msgBit`
gives there (`filter_msgW`). A row's message bit (`Hidden.deltaMsgCol`) is a row bit of one of its compressions'
messages (`msgBit_deltaMsgCol`), so distinct ones are distinct columns (`deltaMsgCol_injW`).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm

/-! ## Positions -/

/-- Input bit `off` of VU `g`'s row `p`'s compression `k` (its `cv` below 512, its `msg` from 512); past its
compressions, of its `hm96` slot (its `cv`, then its `pad`). -/
def inPos (c : Circuit) (g p k off : ℕ) : ℕ :=
  if k < ((HmRow.rowSlots c.ports).2[p]!).size then Hidden.shaPos c g p k off else c.slotCol c.hmNet g p + off

theorem inPos_lt {c : Circuit} {g p k off : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) :
    inPos c g p k off = Hidden.shaPos c g p k off := by
  simp only [inPos, rowSlots_size hp, hk, ↓reduceIte]

theorem inPos_hm {c : Circuit} {g p off : ℕ} (hp : p < c.ports.size) :
    inPos c g p c.ports[p]!.chunks off = c.slotCol c.hmNet g p + off := by
  simp only [inPos, rowSlots_size hp, Nat.lt_irrefl, ↓reduceIte]

theorem cvColOf_eq (c : Circuit) (g p : ℕ) : cvColOf c g p = cvAt c g p 0 := by
  unfold cvColOf cvAt cvCol shaBase shaIn shaSub hmIn
  simp only [Hidden.getElem!_eq_getD']

theorem msgColOf_eq {c : Circuit} {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) :
    msgColOf c g p k = msgAt c g p k := by
  rw [msgAt_lt hp hk]
  rfl

section Pos

variable {c : Circuit} (R : RowRanges c)
include R

theorem shaPos_atW {g p k off : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) :
    AtSlot c c.shaNet (g * c.perVu.getD c.shaNet 0 + ((HmRow.rowSlots c.ports).2[p]!)[k]!.1)
      (1536 * shaSub c p k + off) (Hidden.shaPos c g p k off) := by
  have hs := rowSlot_lt hp hk
  refine ⟨R.slot_lt_count (Or.inl rfl) hg (by rw [R.perVu_sha]; exact hs.1),
    R.net_off_lt (Or.inl rfl) (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; unfold shaSub; omega), ?_⟩
  rw [Hidden.shaPos, shaBase, Circuit.slotCol, Nat.add_assoc]

theorem shaPos_injW {g p k off g' p' k' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) (hg' : g' < c.g) (hp' : p' < c.ports.size) (hk' : k' < c.ports[p']!.chunks) (ho' : off' < 1536)
    (h : Hidden.shaPos c g p k off = Hidden.shaPos c g' p' k' off') : g = g' ∧ p = p' ∧ k = k' ∧ off = off' := by
  have hn := R.range R.shaNet
  obtain ⟨-, hs, ho2⟩ := (shaPos_atW R hg hp hk ho).inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact shaPos_atW R hg' hp' hk' ho')
  have h1 := rowSlot_lt hp hk
  have h2 := rowSlot_lt hp' hk'
  obtain ⟨rfl, hs⟩ := slot_inj (by rw [R.perVu_sha]; exact h1.1) (by rw [R.perVu_sha]; exact h2.1) hs
  unfold shaSub at ho2
  rw [rowSlots_entry hp hk, rowSlots_entry hp' hk'] at hs ho2
  simp only at hs ho2
  have : atOf c.ports p + k = atOf c.ports p' + k' := by omega
  obtain ⟨rfl, rfl⟩ := at_inj hp hp' hk hk' this
  exact ⟨rfl, rfl, rfl, by omega⟩

theorem shaPos_ne_hmW {g p k off g' p' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks)
    (ho : off < 1536) (hg' : g' < c.g) (hp' : p' < c.ports.size) (ho' : off' < 4096) :
    Hidden.shaPos c g p k off ≠ c.slotCol c.hmNet g' p' + off' := by
  intro h
  have hn := R.range R.shaNet
  have hm := R.range R.hmNet
  exact sha_ne_hm R.rowNets ((shaPos_atW R hg hp hk ho).inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [h]; exact R.hmIn_at hg' hp' ho')).1

theorem cvAt_pos {g p k : ℕ} (hp : p < c.ports.size) (t : ℕ) : cvAt c g p k + t = inPos c g p k t := by
  by_cases hk : k < c.ports[p]!.chunks
  · rw [cvAt_lt hp hk, inPos_lt hp hk, (shaIn_eq R.rowNets (shaSub c p k) (rowSlot_lt hp hk).2).1, Hidden.shaPos]
  · simp only [cvAt, inPos, rowSlots_size hp, hk, ↓reduceIte, (hmIn_eq R.rowNets).1, Nat.add_zero]

theorem msgAt_pos {g p k : ℕ} (hp : p < c.ports.size) (t : ℕ) : msgAt c g p k + t = inPos c g p k (512 + t) := by
  by_cases hk : k < c.ports[p]!.chunks
  · rw [msgAt_lt hp hk, inPos_lt hp hk, (shaIn_eq R.rowNets (shaSub c p k) (rowSlot_lt hp hk).2).2, Hidden.shaPos]
    omega
  · simp only [msgAt, inPos, rowSlots_size hp, hk, ↓reduceIte, (hmIn_eq R.rowNets).2.1]
    omega

theorem padCol_pos {g p : ℕ} (hp : p < c.ports.size) (t : ℕ) :
    padCol c g p + t = inPos c g p c.ports[p]!.chunks (512 + t) := by
  rw [padCol_eq R.rowNets, inPos_hm hp, Nat.add_assoc]

/-- **Distinct rows' input bits are distinct**: VU, row, compression (the `hm96` slot past them) and bit. -/
theorem inPos_inj {g p k off g' p' k' off' : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hk : k ≤ c.ports[p]!.chunks) (ho : off < 1536) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hk' : k' ≤ c.ports[p']!.chunks) (ho' : off' < 1536) (h : inPos c g p k off = inPos c g' p' k' off') :
    g = g' ∧ p = p' ∧ k = k' ∧ off = off' := by
  rcases Nat.lt_or_eq_of_le hk with hk | rfl <;> rcases Nat.lt_or_eq_of_le hk' with hk' | rfl
  · rw [inPos_lt hp hk, inPos_lt hp' hk'] at h
    exact shaPos_injW R hg hp hk ho hg' hp' hk' ho' h
  · rw [inPos_lt hp hk, inPos_hm hp'] at h
    exact absurd h (shaPos_ne_hmW R hg hp hk ho hg' hp' (by omega))
  · rw [inPos_hm hp, inPos_lt hp' hk'] at h
    exact absurd h.symm (shaPos_ne_hmW R hg' hp' hk' ho' hg hp (by omega))
  · rw [inPos_hm hp, inPos_hm hp'] at h
    obtain ⟨rfl, rfl, rfl⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) h
    exact ⟨rfl, rfl, rfl, rfl⟩

/-- **A row's input bit is a self row of a row net's slot, below the net's input rows**: the hypotheses
`Hm.inBit_rows` reads, at the `sha512x3` slot of compression `k` or at the `hm96` slot. -/
theorem inPos_cases {g p k off : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ho : off < 1536) {P : Prop}
    (H : ∀ {n s off' : ℕ}, (n = c.shaNet ∨ n = c.hmNet) → AtSlot c n s off' (inPos c g p k off) →
      ∀ {l : Flock.HmNets.Laid} {name : String},
      Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (c.nets.getD n default).2 →
      off' < (c.nets.getD n default).2.inWords * Flock.WORD →
      (l.ra.getD off' #[] = #[off'] ∧ l.rb.getD off' #[] = #[off']) → P) : P := by
  rcases Nat.lt_or_eq_of_le hk with hk | rfl
  · have hs := (rowSlot_lt hp hk).2
    have hat := shaPos_atW R hg hp hk ho
    rw [← inPos_lt (g := g) (off := off) hp hk] at hat
    exact H (Or.inl rfl) hat (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; unfold shaSub; omega) (shaLaid_self (by unfold shaSub; omega))
  · have hat := R.hmIn_at hg hp (off := off) (by omega)
    rw [← inPos_hm (g := g) (off := off) hp] at hat
    exact H (Or.inr rfl) hat (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega)
      (hmLaid_self (by omega))

end Pos

/-! ## The rows' wires, for rows of any shape -/

/-- Row `p`'s wires: compression `k ≥ 1`'s `cv` from compression `k − 1`'s output where `k` starts no chain, then the
`hm96` slot's `cv` from the last compression's output where the slot starts none. -/
def rowWiresOf (ports : Array Port) (sha hm p : ℕ) : List Wire :=
  ((List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).flatMap fun k =>
      if (HmRow.chainStart ports[p]! k).isNone then [chainWire ports sha p k] else []) ++
    if 0 < ((HmRow.rowSlots ports).2[p]!).size &&
        (HmRow.chainStart ports[p]! ((HmRow.rowSlots ports).2[p]!).size).isNone then [hmWire ports sha hm p] else []

/-- **The rows' wires**, row by row, for rows of any shape. -/
theorem rowWires_W (ports : Array Port) (sha hm : ℕ) :
    (HmRow.rowWires ports sha hm).toList =
      (List.range (HmRow.rowSlots ports).2.size).flatMap (rowWiresOf ports sha hm) := by
  unfold HmRow.rowWires
  simp only
  show (forIn (m := Id) [:(HmRow.rowSlots ports).2.size] (#[] : Array Wire) _).toList = _
  refine Eq.trans (forIn_range_append (rowWiresOf ports sha hm) _ (fun p _ d => ?_) #[]) ?_
  · have hloop : (forIn (m := Id) [1:(HmRow.rowSlots ports).2[p]!.size] d fun k __s =>
        if (HmRow.chainStart ports[p]! k).isNone = true then
          pure (ForInStep.yield (__s.push (chainWire ports sha p k)))
        else pure (ForInStep.yield __s)).toList =
        d.toList ++ (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).flatMap fun k =>
          if (HmRow.chainStart ports[p]! k).isNone then [chainWire ports sha p k] else [] := by
      refine forIn_range1_append _ _ (fun k _ d => ?_) d
      by_cases hc : (HmRow.chainStart ports[p]! k).isNone = true
      · simp only [hc, ↓reduceIte]
        exact ⟨_, rfl, by rw [Array.toList_push]⟩
      · simp only [hc, Bool.false_eq_true, ↓reduceIte]
        exact ⟨_, rfl, by simp⟩
    by_cases h0 : (decide (0 < (HmRow.rowSlots ports).2[p]!.size) &&
        (HmRow.chainStart ports[p]! (HmRow.rowSlots ports).2[p]!.size).isNone) = true
    · simp only [h0, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [Array.toList_push]
      erw [hloop]
      rw [rowWiresOf, h0, ite_eq_left_of_eq_true _ _ (eq_true rfl), List.append_assoc]
      rfl
    · simp only [h0, Bool.false_eq_true, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      erw [hloop]
      rw [Bool.not_eq_true] at h0
      rw [rowWiresOf, h0]
      simp
  · simp

/-- The rows' wires' copies at VU `g`'s row `p`. -/
def wireFW (c : Circuit) (g p : ℕ) : List (ℕ × ℕ) :=
  (rowWiresOf c.ports c.shaNet c.hmNet p).flatMap fun w =>
    (List.range (Flat.wireWidth c w)).map fun t => (Flat.wireDst c g w + t, Flat.wireSrc c g w + t)

theorem wireCopiesOf_rowsW (c : Circuit) :
    wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireFW c g p := by
  unfold wireCopiesOf wireFW
  rw [rowWires_W]
  simp only [List.flatMap_assoc]

/-- **A wire copy into VU `g`'s row `p`**: into compression `k`'s `cv` (the `hm96` slot's for `k = chunks`), for a
`k ≥ 1` that starts no chain, from compression `k − 1`'s output. -/
theorem wireFW_mem {c : Circuit} (R : RowRanges c) {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size)
    (he : e ∈ wireFW c g p) :
    ∃ k t, 1 ≤ k ∧ k ≤ c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! k = none ∧ t < 512 ∧
      e = (inPos c g p k t, outAt c g p (k - 1) + t) := by
  unfold wireFW rowWiresOf at he
  rw [List.flatMap_append, List.mem_append, rowSlots_size hp] at he
  rcases he with he | he
  · simp only [List.mem_flatMap, List.mem_range'_1] at he
    obtain ⟨w, ⟨k, hk, hw⟩, he⟩ := he
    by_cases hc : (HmRow.chainStart c.ports[p]! k).isNone = true
    · simp only [hc, ↓reduceIte, List.mem_singleton] at hw
      subst hw
      have hk2 : k < c.ports[p]!.chunks := by omega
      have hs := (rowSlot_lt hp hk2).2
      simp only [List.mem_map, List.mem_range, chainWidth_eq R.rowNets hs] at he
      obtain ⟨t, ht, rfl⟩ := he
      refine ⟨k, t, hk.1, hk2.le, Option.isNone_iff_eq_none.1 hc, ht, ?_⟩
      rw [chainDst_eq R.rowNets hs, ← cvAt_lt hp hk2, cvAt_pos R hp]
      rfl
    · simp only [hc, Bool.false_eq_true, ↓reduceIte, List.not_mem_nil] at hw
  · by_cases hc : (decide (0 < c.ports[p]!.chunks) &&
        (HmRow.chainStart c.ports[p]! c.ports[p]!.chunks).isNone) = true
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hc)] at he
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hc
      simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_map, List.mem_range,
        hmWidth_eq R.rowNets] at he
      obtain ⟨t, ht, rfl⟩ := he
      refine ⟨c.ports[p]!.chunks, t, hc.1, le_rfl, Option.isNone_iff_eq_none.1 hc.2, ht, ?_⟩
      show (hmWireDst c g p + t, _) = _
      rw [hmWireDst_eq R.rowNets, inPos_hm hp]
      unfold Flat.wireSrc hmWire outAt shaBase shaOut shaSub
      simp only [rowSlots_size hp]
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hc)] at he
      simp at he

/-! ## The rows' copies, row by row -/

/-- Δ's copies into VU `g`'s row `p`. -/
def rowOf (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  if c.ports[p]!.segs.isSome then rowCopiesSeg c pin g p else Hidden.rowCopiesOf c pin g p

theorem rowCopiesW_eq (c : Circuit) (pin : ℕ) :
    rowCopiesW c pin = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => rowOf c pin g p := rfl

theorem msgCopy_fst {c : Circuit} {pin x : ℕ} {out : ℕ → ℕ} {m : HmRow.MsgBit} {e : ℕ × ℕ}
    (he : e ∈ msgCopy c pin x out m) : e.1 = x := by
  cases m <;> simp only [msgCopy, List.mem_cons, List.not_mem_nil, or_false] at he <;> rw [he]

theorem filter_msgCopy (c : Circuit) (pin x : ℕ) (out : ℕ → ℕ) (m : HmRow.MsgBit) :
    (msgCopy c pin x out m).filter (fun e => e.1 = x) = msgCopy c pin x out m := by
  cases m <;> simp [msgCopy]

/-- An unsegmented row's message copies are `Hidden.bitCopy`'s. -/
theorem msgCopy_unseg (c : Circuit) (pin x : ℕ) (out : ℕ → ℕ) {p : Port} (hs : p.segs = none) (k t : ℕ) :
    msgCopy c pin x out (HmRow.msgBit p k t) = Hidden.bitCopy c pin x (HmRow.constBit p (HmRow.tailOf p) k t) := by
  rcases h : HmRow.constBit p (HmRow.tailOf p) k t with _ | v <;>
    simp only [HmRow.msgBit, hs, h, msgCopy, Hidden.bitCopy]

/-- Δ's copies into compression `k`'s `cv` of VU `g`'s segmented row `p`, at a chain start. -/
def segCv (c : Circuit) (pin g p k : ℕ) : List (ℕ × ℕ) :=
  (HmRow.chainStart c.ports[p]! k).toList.flatMap fun j =>
    (List.range 512).map fun t => (cvAt c g p k + t, constSrc c pin (startMid c.ports[p]! j) t)

/-- Δ's copies into compression `k`'s message of VU `g`'s segmented row `p`. -/
def segMsg (c : Circuit) (pin g p k : ℕ) : List (ℕ × ℕ) :=
  (List.range 1024).flatMap fun t => msgCopy c pin (msgAt c g p k + t) (outAt c g p) (HmRow.msgBit c.ports[p]! k t)

theorem rowCopiesSeg_eq (c : Circuit) (pin g p : ℕ) :
    rowCopiesSeg c pin g p = (List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1)).flatMap fun k =>
      segCv c pin g p k ++ segMsg c pin g p k := rfl

theorem inPos_add (c : Circuit) (g p k t : ℕ) : inPos c g p k t = inPos c g p k 0 + t := by
  unfold inPos Hidden.shaPos
  split <;> omega

theorem filter_inPos_map {β : Type} (c : Circuit) (g p k : ℕ) (S : ℕ → β) {n t₀ : ℕ} (ht₀ : t₀ < n) :
    ((List.range n).map fun t => (inPos c g p k t, S t)).filter (fun e => e.1 = inPos c g p k t₀) =
      [(inPos c g p k t₀, S t₀)] := by
  obtain ⟨D, hD⟩ : ∃ D, ∀ t, inPos c g p k t = D + t := ⟨_, inPos_add c g p k⟩
  simp only [hD]
  exact filter_map_single _ _ _ _ ht₀

/-- A row's block bits `128 (k + 1)` and on are past its length: none of block `k`'s is constant. -/
theorem constBit_none {p : Port} {tl : ByteArray} {k t : ℕ} (hk : 1024 * (k + 1) ≤ p.bits) (ht : t < 1024) :
    HmRow.constBit p tl k t = none := by
  have hb := blockBit_lt ht
  simp only [HmRow.constBit, show 8 * (128 * k + (HmRow.blockBit t).1) + (HmRow.blockBit t).2 < p.bits by omega,
    ↓reduceIte]

section RowPos

variable {c : Circuit} (R : RowRanges c)
include R

theorem padCol_msgAt {g p : ℕ} (hp : p < c.ports.size) : padCol c g p = msgAt c g p c.ports[p]!.chunks := by
  rw [msgAt_hm hp, padCol_eq R.rowNets, (hmIn_eq R.rowNets).2.1]

theorem segCv_pos {pin g p k : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ segCv c pin g p k) :
    ∃ t < 512, e.1 = inPos c g p k t := by
  simp only [segCv, List.mem_flatMap, List.mem_map, List.mem_range] at he
  obtain ⟨j, -, t, ht, rfl⟩ := he
  exact ⟨t, ht, cvAt_pos R hp t⟩

theorem segMsg_pos {pin g p k : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ segMsg c pin g p k) :
    ∃ t < 1024, e.1 = inPos c g p k (512 + t) := by
  simp only [segMsg, List.mem_flatMap, List.mem_range] at he
  obtain ⟨t, ht, he⟩ := he
  exact ⟨t, ht, by rw [msgCopy_fst he, msgAt_pos R hp]⟩

theorem cvCopies_pos {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ Hidden.cvCopies c pin g p) :
    ∃ t < 512, e.1 = inPos c g p 0 t := by
  obtain ⟨t, ht, rfl⟩ := Hidden.cvCopies_mem he
  exact ⟨t, ht, by rw [cvColOf_eq, cvAt_pos R hp]⟩

theorem padCopies_pos {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ Hidden.padCopies c pin g p) :
    ∃ t < 1024, e.1 = inPos c g p c.ports[p]!.chunks (512 + t) := by
  simp only [Hidden.padCopies, List.mem_flatMap, List.mem_range] at he
  obtain ⟨t, ht, he⟩ := he
  exact ⟨t, ht, by rw [Hidden.bitCopy_fst he, padCol_pos R hp]⟩

theorem msgCopies_pos {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ Hidden.msgCopies c pin g p) :
    ∃ k t, c.ports[p]!.bits / 1024 ≤ k ∧ k < c.ports[p]!.chunks ∧ t < 1024 ∧ e.1 = inPos c g p k (512 + t) := by
  simp only [Hidden.msgCopies, List.mem_flatMap, List.mem_range, List.mem_range'_1, rowSlots_size hp] at he
  obtain ⟨k, hk, t, ht, he⟩ := he
  exact ⟨k, t, hk.1, by omega, ht, by rw [Hidden.bitCopy_fst he, msgColOf_eq hp (by omega), msgAt_pos R hp]⟩

/-- Every copy into VU `g`'s row `p` lands at one of its input bits. -/
theorem rowOf_pos {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ rowOf c pin g p) :
    ∃ k off, k ≤ c.ports[p]!.chunks ∧ off < 1536 ∧ e.1 = inPos c g p k off := by
  unfold rowOf at he
  split at he
  · rw [rowCopiesSeg_eq] at he
    simp only [List.mem_flatMap, List.mem_range, List.mem_append, rowSlots_size hp] at he
    obtain ⟨k, hk, he | he⟩ := he
    · obtain ⟨t, ht, he⟩ := segCv_pos R hp he
      exact ⟨k, t, by omega, by omega, he⟩
    · obtain ⟨t, ht, he⟩ := segMsg_pos R hp he
      exact ⟨k, 512 + t, by omega, by omega, he⟩
  · simp only [Hidden.rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · obtain ⟨t, ht, he⟩ := cvCopies_pos R hp he
      exact ⟨0, t, Nat.zero_le _, by omega, he⟩
    · obtain ⟨t, ht, he⟩ := padCopies_pos R hp he
      exact ⟨_, 512 + t, le_rfl, by omega, he⟩
    · obtain ⟨k, t, -, hk, ht, he⟩ := msgCopies_pos R hp he
      exact ⟨k, 512 + t, hk.le, by omega, he⟩

theorem wireFW_pos {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFW c g p) :
    ∃ k off, k ≤ c.ports[p]!.chunks ∧ off < 1536 ∧ e.1 = inPos c g p k off := by
  obtain ⟨k, t, -, hk, -, ht, rfl⟩ := wireFW_mem R hp he
  exact ⟨k, t, hk, by omega, rfl⟩

/-- **Δ's copies at an input bit of VU `g`'s row `p`** are that row's: its copies and its wires'. -/
theorem filter_rowsW {pin g p k off : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ho : off < 1536) :
    (rowCopiesW c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = inPos c g p k off) =
      (rowOf c pin g p).filter (fun e => e.1 = inPos c g p k off) ++
        (wireFW c g p).filter (fun e => e.1 = inPos c g p k off) := by
  have hsz := rowSlots_size' c.ports
  rw [List.filter_append, wireCopiesOf_rowsW, rowCopiesW_eq,
    filter_flatMap2_single _ _ _ _ g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex => by
      obtain ⟨k', off', hk', ho', he1⟩ := rowOf_pos R (hsz ▸ hp') he
      obtain ⟨h1, h2, -⟩ := inPos_inj R hg' (hsz ▸ hp') hk' ho' hg hp hk ho (he1.symm.trans hex)
      exact hne ⟨h1, h2⟩,
    filter_flatMap2_single _ _ (fun g p => wireFW c g p) _ g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex => by
      obtain ⟨k', off', hk', ho', he1⟩ := wireFW_pos R (hsz ▸ hp') he
      obtain ⟨h1, h2, -⟩ := inPos_inj R hg' (hsz ▸ hp') hk' ho' hg hp hk ho (he1.symm.trans hex)
      exact hne ⟨h1, h2⟩]

theorem filter_inPos_ne {β : Type} {g p k k' t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hk : k ≤ c.ports[p]!.chunks) (ht₀ : t₀ < 1536) (hk' : k' ≤ c.ports[p]!.chunks) {l : List ℕ}
    (hl : ∀ t ∈ l, t < 1536) (S : ℕ → β) (hne : k' ≠ k ∨ ∀ t ∈ l, t ≠ t₀) :
    (l.map fun t => (inPos c g p k' t, S t)).filter (fun e => e.1 = inPos c g p k t₀) = [] :=
  filter_nil_of _ _ fun e he h => by
    obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
    have := inPos_inj R hg hp hk' (hl t ht) hg hp hk ht₀ h
    rcases hne with hne | hne
    · exact hne this.2.2.1
    · exact hne t ht this.2.2.2

theorem chainF_eq {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) :
    (List.range (Flat.wireWidth c (chainWire c.ports c.shaNet p k))).map (fun t =>
        (Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t, Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t)) =
      (List.range 512).map fun t => (inPos c g p k t, outAt c g p (k - 1) + t) := by
  have hs := (rowSlot_lt hp hk).2
  rw [chainWidth_eq R.rowNets hs]
  refine List.map_congr_left fun t _ => ?_
  rw [chainDst_eq R.rowNets hs, ← cvAt_lt hp hk, cvAt_pos R hp]
  rfl

theorem hmF_eq {g p : ℕ} (hp : p < c.ports.size) :
    (List.range (Flat.wireWidth c (hmWire c.ports c.shaNet c.hmNet p))).map (fun t =>
        (Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t,
          Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t)) =
      (List.range 512).map fun t => (inPos c g p c.ports[p]!.chunks t, outAt c g p (c.ports[p]!.chunks - 1) + t) := by
  rw [hmWidth_eq R.rowNets]
  refine List.map_congr_left fun t _ => ?_
  show (hmWireDst c g p + t, _) = _
  rw [hmWireDst_eq R.rowNets, inPos_hm hp]
  unfold Flat.wireSrc hmWire outAt shaBase shaOut shaSub
  simp only [rowSlots_size hp]

theorem wireFW_eq {g p : ℕ} (hp : p < c.ports.size) :
    wireFW c g p = ((List.range' 1 (c.ports[p]!.chunks - 1)).flatMap fun k =>
        if (HmRow.chainStart c.ports[p]! k).isNone then
          (List.range 512).map fun t => (inPos c g p k t, outAt c g p (k - 1) + t) else []) ++
      if 0 < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none then
        (List.range 512).map fun t => (inPos c g p c.ports[p]!.chunks t, outAt c g p (c.ports[p]!.chunks - 1) + t)
      else [] := by
  unfold wireFW rowWiresOf
  rw [List.flatMap_append, List.flatMap_assoc, rowSlots_size hp]
  congr 1
  · refine List.flatMap_congr fun k hk => ?_
    rw [List.mem_range'_1] at hk
    split
    · rw [List.flatMap_cons, List.flatMap_nil, List.append_nil, chainF_eq R hp (by omega)]
    · rfl
  · by_cases h : 0 < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none
    · have hb : (decide (0 < c.ports[p]!.chunks) &&
          (HmRow.chainStart c.ports[p]! c.ports[p]!.chunks).isNone) = true := by simp [h.1, h.2]
      rw [ite_eq_left_of_eq_true _ _ (eq_true hb), ite_eq_left_of_eq_true _ _ (eq_true h), List.flatMap_cons,
        List.flatMap_nil, List.append_nil, hmF_eq R hp]
    · have hb : ¬ (decide (0 < c.ports[p]!.chunks) &&
          (HmRow.chainStart c.ports[p]! c.ports[p]!.chunks).isNone) = true := by
        simpa [Option.isNone_iff_eq_none] using h
      rw [ite_eq_right_of_eq_false _ _ (eq_false hb), ite_eq_right_of_eq_false _ _ (eq_false h)]
      rfl

/-- **The rows' wires' copy at compression `k`'s `cv` bit `t₀`**: from compression `k − 1`'s output where `k ≥ 1`
starts no chain, none otherwise. -/
theorem wireFW_cv {g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (wireFW c g p).filter (fun e => e.1 = inPos c g p k t₀) =
      if 0 < k ∧ HmRow.chainStart c.ports[p]! k = none then [(inPos c g p k t₀, outAt c g p (k - 1) + t₀)]
      else [] := by
  have hr : ∀ t ∈ List.range 512, t < 1536 := fun t ht => by rw [List.mem_range] at ht; omega
  have hoth : ∀ k' ∈ List.range' 1 (c.ports[p]!.chunks - 1), k' ≠ k →
      (if (HmRow.chainStart c.ports[p]! k').isNone then
        (List.range 512).map fun t => (inPos c g p k' t, outAt c g p (k' - 1) + t) else []).filter
        (fun e => e.1 = inPos c g p k t₀) = [] := fun k' hk' hne => by
    rw [List.mem_range'_1] at hk'
    split
    · exact filter_inPos_ne R hg hp hk (by omega) (by omega) hr _ (Or.inl hne)
    · rfl
  rw [wireFW_eq R hp, List.filter_append]
  rcases Nat.lt_or_eq_of_le hk with hk' | rfl
  · have hh : (if 0 < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none then
        (List.range 512).map fun t => (inPos c g p c.ports[p]!.chunks t, outAt c g p (c.ports[p]!.chunks - 1) + t)
        else []).filter (fun e => e.1 = inPos c g p k t₀) = [] := by
      split
      · exact filter_inPos_ne R hg hp hk (by omega) le_rfl hr _ (Or.inl (by omega))
      · rfl
    rw [hh, List.append_nil]
    rcases Nat.eq_zero_or_pos k with rfl | hk0
    · rw [filter_flatMap_none _ _ _ fun k' hk'' => hoth k' hk'' (by rw [List.mem_range'_1] at hk''; omega)]
      rfl
    · have hkm : k ∈ List.range' 1 (c.ports[p]!.chunks - 1) := by rw [List.mem_range'_1]; omega
      rw [filter_flatMap_single _ List.nodup_range' _ _ k hkm hoth]
      by_cases hc : HmRow.chainStart c.ports[p]! k = none
      · rw [ite_eq_left_of_eq_true _ _ (eq_true (Option.isNone_iff_eq_none.2 hc)), filter_inPos_map _ _ _ _ _ ht₀,
          ite_eq_left_of_eq_true _ _ (eq_true ⟨hk0, hc⟩)]
      · have hc' : ¬ (HmRow.chainStart c.ports[p]! k).isNone = true := fun h => hc (Option.isNone_iff_eq_none.1 h)
        have hc'' : ¬ (0 < k ∧ HmRow.chainStart c.ports[p]! k = none) := fun h => hc h.2
        rw [ite_eq_right_of_eq_false _ _ (eq_false hc'), ite_eq_right_of_eq_false _ _ (eq_false hc'')]
        rfl
  · rw [filter_flatMap_none _ _ _ fun k' hk'' => hoth k' hk'' (by rw [List.mem_range'_1] at hk''; omega),
      List.nil_append]
    by_cases hC : 0 < c.ports[p]!.chunks ∧ HmRow.chainStart c.ports[p]! c.ports[p]!.chunks = none
    · rw [ite_eq_left_of_eq_true _ _ (eq_true hC), ite_eq_left_of_eq_true _ _ (eq_true hC),
        filter_inPos_map _ _ _ _ _ ht₀]
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hC), ite_eq_right_of_eq_false _ _ (eq_false hC)]
      rfl

theorem wireFW_msg {g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 1024) : (wireFW c g p).filter (fun e => e.1 = msgAt c g p k + t₀) = [] :=
  filter_nil_of _ _ fun e he h => by
    obtain ⟨k', t, -, hk', -, ht, rfl⟩ := wireFW_mem R hp he
    rw [msgAt_pos R hp] at h
    have := (inPos_inj R hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
    omega

theorem rowCopiesSeg_filter {pin g p k off : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hk : k ≤ c.ports[p]!.chunks) (ho : off < 1536) :
    (rowCopiesSeg c pin g p).filter (fun e => e.1 = inPos c g p k off) =
      (segCv c pin g p k).filter (fun e => e.1 = inPos c g p k off) ++
        (segMsg c pin g p k).filter (fun e => e.1 = inPos c g p k off) := by
  have hkm : k ∈ List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1) := by
    rw [List.mem_range, rowSlots_size hp]; omega
  have hne : ∀ k' ∈ List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1), k' ≠ k →
      (segCv c pin g p k' ++ segMsg c pin g p k').filter (fun e => e.1 = inPos c g p k off) = [] := by
    intro k' hk' hne
    rw [List.mem_range, rowSlots_size hp] at hk'
    refine filter_nil_of _ _ fun e he h => ?_
    rcases List.mem_append.1 he with he | he
    · obtain ⟨t, ht, he1⟩ := segCv_pos R hp he
      exact hne (inPos_inj R hg hp (by omega) (by omega) hg hp hk ho (he1.symm.trans h)).2.2.1
    · obtain ⟨t, ht, he1⟩ := segMsg_pos R hp he
      exact hne (inPos_inj R hg hp (by omega) (by omega) hg hp hk ho (he1.symm.trans h)).2.2.1
  rw [rowCopiesSeg_eq, filter_flatMap_single _ List.nodup_range _ _ k hkm hne, List.filter_append]

/-- **Row `p`'s own copy at compression `k`'s `cv` bit `t₀`**: its midstate's, where `k` starts a chain. -/
theorem rowOf_cv {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowOf c pin g p).filter (fun e => e.1 = inPos c g p k t₀) =
      (HmRow.chainStart c.ports[p]! k).toList.map fun j =>
        (inPos c g p k t₀, constSrc c pin (startMid c.ports[p]! j) t₀) := by
  have hr : ∀ t ∈ List.range 512, t < 1536 := fun t ht => by rw [List.mem_range] at ht; omega
  unfold rowOf
  split
  · rw [rowCopiesSeg_filter R hg hp hk (by omega)]
    have hm : (segMsg c pin g p k).filter (fun e => e.1 = inPos c g p k t₀) = [] :=
      filter_nil_of _ _ fun e he h => by
        obtain ⟨t, ht, he1⟩ := segMsg_pos R hp he
        have := (inPos_inj R hg hp hk (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.2
        omega
    rw [hm, List.append_nil]
    unfold segCv
    cases HmRow.chainStart c.ports[p]! k with
    | none => rfl
    | some j =>
      simp only [Option.toList_some, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.map_cons,
        List.map_nil, cvAt_pos R hp]
      exact filter_inPos_map _ _ _ _ _ ht₀
  · rename_i hs
    have hs' : c.ports[p]!.segs = none := by simpa using hs
    rw [ScopeW.chainStart_unseg hs' k]
    unfold Hidden.rowCopiesOf
    have hpad : (Hidden.padCopies c pin g p).filter (fun e => e.1 = inPos c g p k t₀) = [] :=
      filter_nil_of _ _ fun e he h => by
        obtain ⟨t, ht, he1⟩ := padCopies_pos R hp he
        have := (inPos_inj R hg hp le_rfl (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.2
        omega
    have hmsg : (Hidden.msgCopies c pin g p).filter (fun e => e.1 = inPos c g p k t₀) = [] :=
      filter_nil_of _ _ fun e he h => by
        obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_pos R hp he
        have := (inPos_inj R hg hp hk'.le (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.2
        omega
    rw [List.filter_append, List.filter_append, hpad, hmsg, List.append_nil, List.append_nil]
    unfold Hidden.cvCopies
    simp only [cvColOf_eq, cvAt_pos R hp]
    by_cases hk0 : k = 0
    · subst hk0
      rw [filter_inPos_map _ _ _ _ _ ht₀]
      simp [startMid, segsOf, hs']
    · simp only [hk0, ↓reduceIte, Option.toList_none, List.map_nil]
      exact filter_inPos_ne R hg hp hk (by omega) (Nat.zero_le _) hr _ (Or.inl (Ne.symm hk0))

/-- **Row `p`'s own copy at compression `k`'s message bit `t₀`**: the one `HmRow.msgBit` gives there. -/
theorem rowOf_msg {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 1024) :
    (rowOf c pin g p).filter (fun e => e.1 = msgAt c g p k + t₀) =
      msgCopy c pin (msgAt c g p k + t₀) (outAt c g p) (HmRow.msgBit c.ports[p]! k t₀) := by
  unfold rowOf
  split
  · rw [msgAt_pos R hp, rowCopiesSeg_filter R hg hp hk (by omega)]
    have hcv : (segCv c pin g p k).filter (fun e => e.1 = inPos c g p k (512 + t₀)) = [] :=
      filter_nil_of _ _ fun e he h => by
        obtain ⟨t, ht, he1⟩ := segCv_pos R hp he
        have := (inPos_inj R hg hp hk (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.2
        omega
    rw [hcv, List.nil_append, ← msgAt_pos R hp]
    unfold segMsg
    rw [filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t _ hne =>
      filter_nil_of _ _ fun e he => by rw [msgCopy_fst he]; omega]
    exact filter_msgCopy _ _ _ _ _
  · rename_i hs
    have hs' : c.ports[p]!.segs = none := by simpa using hs
    rw [msgCopy_unseg _ _ _ _ hs']
    unfold Hidden.rowCopiesOf
    have hcv : (Hidden.cvCopies c pin g p).filter (fun e => e.1 = msgAt c g p k + t₀) = [] :=
      filter_nil_of _ _ fun e he h => by
        obtain ⟨t, ht, he1⟩ := cvCopies_pos R hp he
        rw [msgAt_pos R hp] at h
        have := (inPos_inj R hg hp (Nat.zero_le _) (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.2
        omega
    rw [List.filter_append, List.filter_append, hcv, List.nil_append]
    rcases Nat.lt_or_eq_of_le hk with hk' | rfl
    · have hpad : (Hidden.padCopies c pin g p).filter (fun e => e.1 = msgAt c g p k + t₀) = [] :=
        filter_nil_of _ _ fun e he h => by
          obtain ⟨t, ht, he1⟩ := padCopies_pos R hp he
          rw [msgAt_pos R hp] at h
          have := (inPos_inj R hg hp le_rfl (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.1
          omega
      rw [hpad, List.nil_append]
      by_cases hb : c.ports[p]!.bits / 1024 ≤ k
      · have hkm : k ∈ List.range' (c.ports[p]!.bits / 1024)
            (((HmRow.rowSlots c.ports).2[p]!).size - c.ports[p]!.bits / 1024) := by
          rw [List.mem_range'_1, rowSlots_size hp]; omega
        have hne : ∀ k' ∈ List.range' (c.ports[p]!.bits / 1024)
            (((HmRow.rowSlots c.ports).2[p]!).size - c.ports[p]!.bits / 1024), k' ≠ k →
            ((List.range 1024).flatMap fun t => Hidden.bitCopy c pin (msgColOf c g p k' + t)
              (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k' t)).filter
              (fun e => e.1 = msgAt c g p k + t₀) = [] := by
          intro k' hk'' hne
          rw [List.mem_range'_1, rowSlots_size hp] at hk''
          refine filter_nil_of _ _ fun e he h => ?_
          simp only [List.mem_flatMap, List.mem_range] at he
          obtain ⟨t, ht, he⟩ := he
          rw [Hidden.bitCopy_fst he, msgColOf_eq hp (by omega), msgAt_pos R hp, msgAt_pos R hp] at h
          exact hne (inPos_inj R hg hp (by omega) (by omega) hg hp hk (by omega) h).2.2.1
        unfold Hidden.msgCopies
        rw [filter_flatMap_single _ List.nodup_range' _ _ k hkm hne, msgColOf_eq hp hk',
          filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t _ hne =>
            filter_nil_of _ _ fun e he => by rw [Hidden.bitCopy_fst he]; omega]
        exact Hidden.filter_bitCopy _ _ _ _
      · have hmsg : (Hidden.msgCopies c pin g p).filter (fun e => e.1 = msgAt c g p k + t₀) = [] :=
          filter_nil_of _ _ fun e he h => by
            obtain ⟨k', t, hk1, hk', ht, he1⟩ := msgCopies_pos R hp he
            rw [msgAt_pos R hp] at h
            have := (inPos_inj R hg hp hk'.le (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.1
            omega
        rw [hmsg, constBit_none (by omega) ht₀]
        rfl
    · have hmsg : (Hidden.msgCopies c pin g p).filter
          (fun e => e.1 = msgAt c g p c.ports[p]!.chunks + t₀) = [] :=
        filter_nil_of _ _ fun e he h => by
          obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_pos R hp he
          rw [msgAt_pos R hp] at h
          have := (inPos_inj R hg hp hk'.le (by omega) hg hp hk (by omega) (he1.symm.trans h)).2.2.1
          omega
      rw [hmsg, List.append_nil, ← padCol_msgAt R hp]
      unfold Hidden.padCopies
      rw [filter_flatMap_single _ List.nodup_range _ _ t₀ (List.mem_range.2 ht₀) fun t _ hne =>
        filter_nil_of _ _ fun e he => by rw [Hidden.bitCopy_fst he]; omega, Hidden.filter_bitCopy, rowSlots_size hp]

/-- **Δ's copies at compression `k`'s `cv` bit `t₀`** (the `hm96` slot's for `k = chunks`): a chain start's midstate
bit, and the previous compression's output bit where `k ≥ 1` starts no chain. -/
theorem filter_cvW {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowCopiesW c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = inPos c g p k t₀) =
      (HmRow.chainStart c.ports[p]! k).toList.map (fun j =>
          (inPos c g p k t₀, constSrc c pin (startMid c.ports[p]! j) t₀)) ++
        if 0 < k ∧ HmRow.chainStart c.ports[p]! k = none then [(inPos c g p k t₀, outAt c g p (k - 1) + t₀)]
        else [] := by
  rw [filter_rowsW R hg hp hk (by omega), rowOf_cv R hg hp hk ht₀, wireFW_cv R hg hp hk ht₀]

theorem filter_cv_start {pin g p k t₀ j : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) (hj : HmRow.chainStart c.ports[p]! k = some j) :
    (rowCopiesW c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = inPos c g p k t₀) = [(inPos c g p k t₀, constSrc c pin (startMid c.ports[p]! j) t₀)] := by
  have hn : ¬ (0 < k ∧ (some j : Option ℕ) = none) := fun h => Option.some_ne_none j h.2
  rw [filter_cvW R hg hp hk ht₀, hj, ite_eq_right_of_eq_false _ _ (eq_false hn)]
  rfl

theorem filter_cv_chain {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (h0 : 0 < k)
    (hk : k ≤ c.ports[p]!.chunks) (ht₀ : t₀ < 512) (hj : HmRow.chainStart c.ports[p]! k = none) :
    (rowCopiesW c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = inPos c g p k t₀) = [(inPos c g p k t₀, outAt c g p (k - 1) + t₀)] := by
  rw [filter_cvW R hg hp hk ht₀, hj, ite_eq_left_of_eq_true _ _ (eq_true ⟨h0, rfl⟩)]
  rfl

/-- **Δ's copies at compression `k`'s message bit `t₀`** (the `hm96` slot's `pad` for `k = chunks`): the copy
`HmRow.msgBit` gives there. -/
theorem filter_msgW {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : k ≤ c.ports[p]!.chunks)
    (ht₀ : t₀ < 1024) :
    (rowCopiesW c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = msgAt c g p k + t₀) =
      msgCopy c pin (msgAt c g p k + t₀) (outAt c g p) (HmRow.msgBit c.ports[p]! k t₀) := by
  have h := filter_rowsW R (pin := pin) hg hp hk (show 512 + t₀ < 1536 by omega)
  rw [← msgAt_pos R hp] at h
  rw [h, rowOf_msg R hg hp hk ht₀, wireFW_msg R hg hp hk ht₀, List.append_nil]

end RowPos

/-! ## A row's message bits -/

theorem byteAt_snd (p : Port) (b : ℕ) : (HmRow.byteAt p b).2 = b % 128 := by
  unfold HmRow.byteAt
  split <;> rfl

/-- Row `p`'s message byte `b`, bit `t`, as `HmOut.delta` writes it: column `64 (o / 8) + 56 − 8 (o % 8) + t`
(`o = b % 128`) of the message of compression `(HmRow.byteAt b).1`. -/
theorem deltaMsgCol_eqW (c : Circuit) (g p b t : ℕ) :
    Hidden.deltaMsgCol c g p b t = msgAt c g p (HmRow.byteAt c.ports[p]! b).1 +
      (64 * (b % 128 / 8) + 56 - 8 * (b % 128 % 8) + t) := by
  rw [← byteAt_snd c.ports[p]! b]
  unfold Hidden.deltaMsgCol msgAt shaBase shaIn shaSub hmIn
  simp only [Hidden.getElem!_eq_getD']
  split <;> omega

theorem segs_fst {p : Port} (h : RowSegWhole p) {j : ℕ} (hj : j < (segsOf p).size) :
    ((segsOf p)[j]!).1 = min (p.bits - 8192 * j) 8192 := by
  rw [getElem!_pos (segsOf p) j hj]
  simp [segsOf, h.2.1, HmRow.segShape, HmRow.ROW_SEG_BITS]

/-- **A row's message byte `b`, bit `t`, is its row bit `8 b + t`**, at a row of any widened shape: a bit of a whole
row's compressions, of a short row's `hm96` slot or of a segment's data blocks. -/
theorem msgBit_byteAt {tags : Tags} {c : Circuit} (P : ParseFacts tags c) {p b t : ℕ} (hp : p < c.ports.size)
    (hs : rowShapeW c.ports[p]! = true) (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) :
    (HmRow.byteAt c.ports[p]! b).1 ≤ c.ports[p]!.chunks ∧
      HmRow.msgBit c.ports[p]! (HmRow.byteAt c.ports[p]! b).1 (64 * (b % 128 / 8) + 56 - 8 * (b % 128 % 8) + t) =
        .row (8 * b + t) := by
  have hbb := blockBit_of_col (o := b % 128) (Nat.mod_lt _ (by omega)) ht
  have hcol : 64 * (b % 128 / 8) + 56 - 8 * (b % 128 % 8) + t < 1024 := by omega
  rcases rowShapeW_spec hs with h | h | h | h
  · have hw := rowWhole_of_v1 P (getElem_mem_toList' hp) h
    have h2 := hw.2.2.2
    rw [byteAt_unseg hw.1]
    refine ⟨by omega, ?_⟩
    rw [msgBit_whole_row hw (by omega) hcol, hbb]
    congr 1
    omega
  · have hw := rowWhole_of_v2 h
    have h2 := hw.2.2.2
    rw [byteAt_unseg hw.1]
    refine ⟨by omega, ?_⟩
    rw [msgBit_whole_row hw (by omega) hcol, hbb]
    congr 1
    omega
  · have hc := h.2.2.2.2.1
    have h17 := h.2.2.2.2.2
    rw [byteAt_unseg h.2.1]
    have h0 : b / 128 = c.ports[p]!.chunks := by omega
    refine ⟨h0.le, ?_⟩
    rw [h0, msgBit_short_row h hcol (by rw [hbb]; omega), hbb]
    congr 1
    omega
  · have hsz := segsOf_size h
    have hbits := h.2.2.1
    have h64 := h.2.2.2.2.1
    rw [byteAt_seg (segs_eq h) b]
    have hj : b / 1024 < (segsOf c.ports[p]!).size := by rw [hsz]; omega
    have hm := segs_fst h hj
    obtain ⟨-, hm1, hn⟩ := segs_whole h hj
    have hi : b % 1024 / 128 < ((segsOf c.ports[p]!)[b / 1024]!).1 / 1024 := by
      rw [hm] at hm1 ⊢
      omega
    refine ⟨?_, ?_⟩
    · have h1 := segFirst_succ c.ports[p]! hj
      have h2 := segFirst_mono c.ports[p]! (show b / 1024 + 1 ≤ (segsOf c.ports[p]!).size by omega) le_rfl
      have h3 := seg_chunks h
      omega
    · rw [msgBit_seg_row h hj hi hcol, hbb]
      congr 1
      unfold HmRow.ROW_SEG_BITS
      omega

theorem deltaMsgCol_at {tags : Tags} {c : Circuit} (P : ParseFacts tags c) {g p b t : ℕ} (hp : p < c.ports.size)
    (hs : rowShapeW c.ports[p]! = true) (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) :
    ∃ k tb, k ≤ c.ports[p]!.chunks ∧ tb < 1024 ∧ Hidden.deltaMsgCol c g p b t = msgAt c g p k + tb ∧
      HmRow.msgBit c.ports[p]! k tb = .row (8 * b + t) := by
  obtain ⟨hk, hm⟩ := msgBit_byteAt P hp hs hb ht
  exact ⟨_, _, hk, by omega, deltaMsgCol_eqW c g p b t, hm⟩

/-- **Distinct rows' message bits are distinct columns**, at rows of any widened shape. -/
theorem deltaMsgCol_injW {tags : Tags} {c : Circuit} (R : RowRanges c) (P : ParseFacts tags c)
    {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hs : rowShapeW c.ports[p]! = true)
    (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hs' : rowShapeW c.ports[p']! = true) (hb' : b' < 2 * c.ports[p']!.words) (ht' : t' < 8)
    (h : Hidden.deltaMsgCol c g p b t = Hidden.deltaMsgCol c g' p' b' t') : g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  obtain ⟨k, tb, hk, htb, hd, hm⟩ := deltaMsgCol_at P (g := g) hp hs hb ht
  obtain ⟨k', tb', hk', htb', hd', hm'⟩ := deltaMsgCol_at P (g := g') hp' hs' hb' ht'
  rw [hd, hd', msgAt_pos R hp, msgAt_pos R hp'] at h
  obtain ⟨rfl, rfl, rfl, h4⟩ := inPos_inj R hg hp hk (by omega) hg' hp' hk' (by omega) h
  have h5 : tb = tb' := by omega
  subst h5
  rw [hm] at hm'
  have := HmRow.MsgBit.row.inj hm'
  exact ⟨rfl, rfl, by omega, by omega⟩

end FlockSoundness.Discharge.ScopeW
