import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegDelta

/-!
# Δ's copies at the rows, for rows of every kind

Each of `HmRow.delta`'s writes at a row is a copy (`rowCopiesAny`, whose pairs are `rowPairsAnyAll`): an unsegmented
row's `rowCopiesOf`, a segmented row's `segCopiesOf` (each chain start's midstate, and its compressions' constant and
digest bits). Every copy into template instance `g`'s row `p` and every wire `rowWiresOf` gives it lands in that row's
slots (`Dst`), so Δ's copies at a column of row `p` are row `p`'s (`filter_rowsAny`), and at an unsegmented row they are
`filter_rows`' of `Wide.Copies`, whatever the other rows. `Any` restates `Wide.Copies`' copies at an unsegmented row's own
columns (`cv0_filterAll` … `msg_filterAll`) for circuits whose other rows may be segmented.
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- Δ's copy at column `x` of a segmented row's message bit. -/
def segBitCopy (c : Circuit) (pin g p x : ℕ) : HmRow.MsgBit → List (ℕ × ℕ)
  | .const v => [(x, if v then pin else zeroCol c)]
  | .digest kk col => [(x, digestSrc c g p kk col)]
  | .row _ => []

/-- Δ's copies into compression `k`'s `cv` of template instance `g`'s segmented row `p`: at a chain start, its
midstate. -/
def segCvCopies (c : Circuit) (pin g p : ℕ) (segs : Array (ℕ × ℕ)) (k : ℕ) : List (ℕ × ℕ) :=
  match HmRow.chainStart c.ports[p]! k with
  | some j => (List.range 512).map fun t => (segCvCol c g p k + t, constSrc c pin
      (if j < segs.size then HmRow.segMidstate c.ports[p]! j else HmRow.midstateOf c.ports[p]!) t)
  | none => []

/-- Δ's copies into compression `k`'s `msg` of template instance `g`'s segmented row `p`. -/
def segMsgCopies (c : Circuit) (pin g p k : ℕ) : List (ℕ × ℕ) :=
  (List.range 1024).flatMap fun t => segBitCopy c pin g p (segMsgCol c g p k + t) (HmRow.msgBit c.ports[p]! k t)

/-- Δ's copies into template instance `g`'s segmented row `p` (`segPairsOf`'s). -/
def segCopiesOf (c : Circuit) (pin g p : ℕ) (segs : Array (ℕ × ℕ)) : List (ℕ × ℕ) :=
  (List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1)).flatMap fun k =>
    segCvCopies c pin g p segs k ++ segMsgCopies c pin g p k

/-- Δ's copies into template instance `g`'s row `p`, of either kind. -/
def rowCopiesAnyOf (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  match c.ports[p]!.segs with
  | none => rowCopiesOf c pin g p
  | some segs => segCopiesOf c pin g p segs

/-- Δ's copies into the rows, of every kind. -/
def rowCopiesAny (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => rowCopiesAnyOf c pin g p

theorem segPairsOf_eq (c : Circuit) (pin g p : ℕ) (segs : Array (ℕ × ℕ)) :
    segPairsOf c pin g p segs = (segCopiesOf c pin g p segs).flatMap pair2 := by
  unfold segPairsOf segCopiesOf
  rw [List.flatMap_assoc]
  refine flatMap_congr' fun k _ => ?_
  unfold segPairsK segCvCopies segMsgCopies
  rw [List.flatMap_append, List.flatMap_assoc]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · rcases HmRow.chainStart c.ports[p]! k with _ | j
    · rfl
    · simp only [List.flatMap_map]
      rfl
  · refine flatMap_congr' fun t _ => ?_
    cases HmRow.msgBit c.ports[p]! k t <;> rfl

theorem rowPairsAny_eq (c : Circuit) (pin g p : ℕ) : rowPairsAny c pin g p = (rowCopiesAnyOf c pin g p).flatMap pair2 := by
  unfold rowPairsAny rowCopiesAnyOf
  rcases c.ports[p]!.segs with _ | segs
  · exact rowPairsOf_eq c pin g p
  · exact segPairsOf_eq c pin g p segs

theorem rowPairsAnyAll_eq (c : Circuit) (pin : ℕ) : rowPairsAnyAll c pin = (rowCopiesAny c pin).flatMap pair2 := by
  simp only [rowPairsAnyAll, rowCopiesAny, List.flatMap_assoc, rowPairsAny_eq]

theorem rowCopiesAnyOf_unseg {c : Circuit} {pin g p : ℕ} (hup : c.ports[p]!.segs = none) :
    rowCopiesAnyOf c pin g p = rowCopiesOf c pin g p := by
  simp only [rowCopiesAnyOf, hup]

theorem rowWiresOf_unseg {ports : Array Port} {sha hm p : ℕ} (hup : ports[p]!.segs = none) :
    rowWiresOf ports sha hm p = (List.range' 1 (((HmRow.rowSlots ports).2[p]!).size - 1)).map (chainWire ports sha p) ++
      (if 0 < ((HmRow.rowSlots ports).2[p]!).size then [hmWire ports sha hm p] else []) := by
  have hcs := chainStart_unseg hup
  unfold rowWiresOf
  congr 1
  · rw [List.map_eq_flatMap]
    refine flatMap_congr' fun k hk => ?_
    rw [List.mem_range'_1] at hk
    simp [hcs, show k ≠ 0 by omega]
  · by_cases h0 : 0 < ((HmRow.rowSlots ports).2[p]!).size
    · simp [hcs, h0, show ((HmRow.rowSlots ports).2[p]!).size ≠ 0 by omega]
    · simp [h0]

theorem wireFAny_unseg {c : Circuit} {g p : ℕ} (hup : c.ports[p]!.segs = none) : wireFAny c g p = wireFAll c g p := by
  unfold wireFAny wireFAll
  rw [rowWiresOf_unseg hup]

namespace Any


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

theorem deltaMsgCol_pos {g p b t : ℕ} (hp : p < c.ports.size) (hs : (c.ports.getD p default).segs = none)
    (hk : b / 128 < c.ports[p]!.chunks) :
    deltaMsgCol c g p b t = shaPos c g p (b / 128) (512 + msgOff b t) := by
  rw [deltaMsgCol_eq _ _ _ _ _ hs hp hk, (shaIn_eq R.rowNets (shaSub c p (b / 128)) (rowSlot_lt hp hk).2).2, shaPos]
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

/-- A segmented row's compression `k`'s `cv` and `msg` columns in its `sha512x3` slot. -/
theorem segCols_sha {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) (t : ℕ) :
    segCvCol c g p k + t = shaPos c g p k t ∧ segMsgCol c g p k + t = shaPos c g p k (512 + t) := by
  have hs := (rowSlot_lt hp hk).2
  have e := shaIn_eq R.rowNets (shaSub c p k) hs
  have hk' : k < ((HmRow.rowSlots c.ports).2[p]!).size := by rw [rowSlots_size hp]; exact hk
  unfold segCvCol segMsgCol segCols
  simp only [hk', ↓reduceIte]
  refine ⟨?_, ?_⟩
  · show shaBase c g p k + shaIn c (2 * shaSub c p k) + t = _
    rw [e.1, shaPos]
  · show shaBase c g p k + shaIn c (2 * shaSub c p k + 1) + t = _
    rw [e.2, shaPos]
    omega

/-- A segmented row's `hm96` slot's `cv` and `pad` columns (compression `w.size`). -/
theorem segCols_hm {g p k : ℕ} (hp : p < c.ports.size) (hk : ¬ k < c.ports[p]!.chunks) (t : ℕ) :
    segCvCol c g p k + t = c.slotCol c.hmNet g p + t ∧ segMsgCol c g p k + t = c.slotCol c.hmNet g p + (512 + t) := by
  have hk' : ¬ k < ((HmRow.rowSlots c.ports).2[p]!).size := by rw [rowSlots_size hp]; exact hk
  unfold segCvCol segMsgCol segCols
  simp only [hk', ↓reduceIte]
  refine ⟨by rw [hmIn0 R.rowNets, Nat.add_zero], ?_⟩
  show padCol c g p + t = _
  rw [padCol_eq R.rowNets]
  omega

omit R in
theorem segBitCopy_fst {pin g p x : ℕ} {b : HmRow.MsgBit} {e : ℕ × ℕ} (he : e ∈ segBitCopy c pin g p x b) :
    e.1 = x := by
  cases b <;> simp only [segBitCopy, List.mem_cons, List.not_mem_nil, or_false] at he <;> rw [he]

theorem segCopies_dst {pin g p : ℕ} {segs : Array (ℕ × ℕ)} {e : ℕ × ℕ} (hp : p < c.ports.size)
    (he : e ∈ segCopiesOf c pin g p segs) : Dst c g p e.1 := by
  simp only [segCopiesOf, List.mem_flatMap, List.mem_range, List.mem_append] at he
  obtain ⟨k, -, he⟩ := he
  by_cases hks : k < c.ports[p]!.chunks
  · have E := segCols_sha R (g := g) hp hks
    rcases he with he | he
    · unfold segCvCopies at he
      split at he
      · simp only [List.mem_map, List.mem_range] at he
        obtain ⟨t, ht, rfl⟩ := he
        exact Or.inl ⟨k, t, hks, by omega, (E t).1⟩
      · cases he
    · simp only [segMsgCopies, List.mem_flatMap, List.mem_range] at he
      obtain ⟨t, ht, he⟩ := he
      rw [segBitCopy_fst he]
      exact Or.inl ⟨k, 512 + t, hks, by omega, (E t).2⟩
  · have E := segCols_hm R (g := g) (k := k) hp hks
    rcases he with he | he
    · unfold segCvCopies at he
      split at he
      · simp only [List.mem_map, List.mem_range] at he
        obtain ⟨t, ht, rfl⟩ := he
        exact Or.inr ⟨t, by omega, (E t).1⟩
      · cases he
    · simp only [segMsgCopies, List.mem_flatMap, List.mem_range] at he
      obtain ⟨t, ht, he⟩ := he
      rw [segBitCopy_fst he]
      exact Or.inr ⟨512 + t, by omega, (E t).2⟩

theorem rowCopiesAnyOf_dst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ rowCopiesAnyOf c pin g p) :
    Dst c g p e.1 := by
  unfold rowCopiesAnyOf at he
  split at he
  · exact rowCopiesOf_dst R hp he
  · exact segCopies_dst R hp he

theorem wireFAny_dst {g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ wireFAny c g p) : Dst c g p e.1 := by
  unfold wireFAny at he
  obtain ⟨w, hw, he⟩ := List.mem_flatMap.1 he
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 he
  rw [List.mem_range] at ht
  unfold rowWiresOf at hw
  rcases List.mem_append.1 hw with hw | hw
  · obtain ⟨k, hk, hw⟩ := List.mem_flatMap.1 hw
    rw [List.mem_range'_1, rowSlots_size hp] at hk
    split at hw
    · rw [List.mem_singleton] at hw
      subst hw
      have hs := (rowSlot_lt hp (show k < c.ports[p]!.chunks by omega)).2
      rw [chainWidth_eq R.rowNets hs] at ht
      refine Or.inl ⟨k, t, by omega, by omega, ?_⟩
      show Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t = _
      rw [chainDst_eq R.rowNets hs, (shaIn_eq R.rowNets (shaSub c p k) hs).1, shaPos]
    · cases hw
  · split at hw
    · rw [List.mem_singleton] at hw
      subst hw
      rw [hmWidth_eq R.rowNets] at ht
      refine Or.inr ⟨t, by omega, ?_⟩
      show hmWireDst c g p + t = _
      rw [hmWireDst_eq R.rowNets]
    · cases hw

/-- **Δ's copies at a column of template instance `g`'s row `p`**, for rows of every kind, are that row's: its constant
bits' and its wires'. -/
theorem filter_rowsAny {pin g p x : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hx : Dst c g p x) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = x) =
      (rowCopiesAnyOf c pin g p).filter (fun e => e.1 = x) ++ (wireFAny c g p).filter (fun e => e.1 = x) := by
  have hsz := rowSlots_size' c.ports
  rw [List.filter_append, wireCopiesOf_rowsAny]
  unfold rowCopiesAny
  rw [filter_flatMap2_single _ _ _ x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hg' (hsz ▸ hp') hg hp (hex ▸ rowCopiesAnyOf_dst R (hsz ▸ hp') he) hx),
    filter_flatMap2_single _ _ (fun g p => wireFAny c g p) x g p hg (hsz ▸ hp) fun g' hg' p' hp' hne e he hex =>
      hne (dst_inj R hg' (hsz ▸ hp') hg hp (hex ▸ wireFAny_dst R (hsz ▸ hp') he) hx)]

/-- **Δ's copies at a column of template instance `g`'s unsegmented row `p`**, whatever the other rows, are that row's
(`filter_rows` of `Wide.Copies`, which takes every row unsegmented). -/
theorem filter_rows {pin g p x : ℕ} (hup : c.ports[p]!.segs = none) (hg : g < c.g) (hp : p < c.ports.size)
    (hx : Dst c g p x) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = x) =
      (rowCopiesOf c pin g p).filter (fun e => e.1 = x) ++ (wireFAll c g p).filter (fun e => e.1 = x) := by
  rw [filter_rowsAny R hg hp hx, rowCopiesAnyOf_unseg hup, wireFAny_unseg hup]

theorem cvCopies_fst {pin g p : ℕ} {e : ℕ × ℕ} (hp : p < c.ports.size) (he : e ∈ cvCopies c pin g p) :
    (0 < c.ports[p]!.chunks ∧ ∃ t < 512, e.1 = shaPos c g p 0 t) ∨
      (c.ports[p]!.chunks = 0 ∧ ∃ t < 512, e.1 = c.slotCol c.hmNet g p + t) := by
  obtain ⟨t, ht, rfl⟩ := cvCopies_mem he
  by_cases h0 : 0 < c.ports[p]!.chunks
  · exact Or.inl ⟨h0, t, ht, cvColOf_pos R hp h0 t⟩
  · exact Or.inr ⟨by omega, t, ht, cvColOf_zero R hp (by omega) t⟩

/-! ## Δ's copies at a row's own columns -/

omit R in
theorem filter_bitCopy (c : Circuit) (pin x : ℕ) (v : Option Bool) :
    (bitCopy c pin x v).filter (fun e => e.1 = x) = bitCopy c pin x v := by
  cases v <;> simp [bitCopy]

/-- **Δ's one copy into template instance `g`'s row `p`'s first compression's `cv` bit `t₀`**: the row prefix's midstate
bit. -/
theorem cv0_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none) (h0 : 0 < c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = shaPos c g p 0 t₀) =
      [(shaPos c g p 0 t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
  rw [filter_rows R hup hg hp (Or.inl ⟨0, t₀, h0, by omega, rfl⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAll_mem R hp he with ⟨k, t, hk1, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
    · intro h
      have := (shaPos_inj R hg hp hk (by omega) hg hp h0 (by omega) h).2.2.1
      omega
    · exact fun h => shaPos_ne_hm R hg hp h0 (by omega) hg hp (by omega) h.symm
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨t, ht, he1⟩ := padCopies_mem R he
    rw [he1]
    exact fun h => shaPos_ne_hm R hg hp h0 (by omega) hg hp (by omega) h.symm
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) = [] := filter_nil_of _ _ fun e he => by
    obtain ⟨k, t, -, hk, ht, he1⟩ := msgCopies_mem R hp he
    rw [he1]
    intro h
    have := (shaPos_inj R hg hp hk (by omega) hg hp h0 (by omega) h).2.2.2
    omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = shaPos c g p 0 t₀) =
      [(shaPos c g p 0 t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
    rw [← cvColOf_pos R hp h0 t₀]
    unfold cvCopies
    exact filter_map_single _ _ _ _ ht₀
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg]
  rfl

/-- **Δ's one copy into compression `k ≥ 1`'s `cv` bit `t₀`**: the chain from compression `k − 1`'s output. -/
theorem chain_filterAll {pin g p k t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none) (hk1 : 1 ≤ k)
    (hk : k < c.ports[p]!.chunks) (ht₀ : t₀ < 512) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) =
      [(Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀,
        Flat.wireSrc c g (chainWire c.ports c.shaNet p k) + t₀)] := by
  have h0 : 0 < c.ports[p]!.chunks := by omega
  have hs := (rowSlot_lt hp hk).2
  have hx := chainDst_pos R (g := g) hp hk t₀
  rw [filter_rows R hup hg hp (Or.inl ⟨k, t₀, hk, by omega, hx⟩)]
  have hr : (rowCopiesOf c pin g p).filter
      (fun e => e.1 = Flat.wireDst c g (chainWire c.ports c.shaNet p k) + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hp he with ⟨-, t, ht, he1⟩ | ⟨h, -⟩
      · rw [he1]
        intro h
        have := (shaPos_inj R hg hp h0 (by omega) hg hp hk (by omega) h).2.2.1
        omega
      · omega
    · obtain ⟨t, ht, he1⟩ := padCopies_mem R he
      rw [he1]
      exact fun h => shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega) h.symm
    · obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_mem R hp he
      rw [he1]
      intro h
      have := (shaPos_inj R hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
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
      exact fun h => shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega) h.symm
  rw [hr, List.nil_append]
  unfold wireFAll
  rw [List.flatMap_append, List.filter_append, List.flatMap_map, hc, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  rw [List.flatMap_cons, List.flatMap_nil, List.append_nil, hh, List.append_nil, chainWidth_eq R.rowNets hs,
    filter_map_single _ _ _ _ ht₀]

/-- **Δ's one copy into the `hm96` slot's `cv` bit `t₀`** of a row of a compression or more: the wire from the last
compression's output. -/
theorem hmcv_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none) (h0 : 0 < c.ports[p]!.chunks)
    (ht₀ : t₀ < 512) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) =
      [(Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀,
        Flat.wireSrc c g (hmWire c.ports c.shaNet c.hmNet p) + t₀)] := by
  have hx : Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀ = c.slotCol c.hmNet g p + t₀ := by
    show hmWireDst c g p + t₀ = _
    rw [hmWireDst_eq R.rowNets]
  rw [filter_rows R hup hg hp (Or.inr ⟨t₀, by omega, hx⟩)]
  have hr : (rowCopiesOf c pin g p).filter
      (fun e => e.1 = Flat.wireDst c g (hmWire c.ports c.shaNet c.hmNet p) + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hp he with ⟨-, t, ht, he1⟩ | ⟨h, -⟩
      · rw [he1]
        exact shaPos_ne_hm R hg hp h0 (by omega) hg hp (by omega)
      · omega
    · obtain ⟨t, ht, he1⟩ := padCopies_mem R he
      rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
    · obtain ⟨k', t, -, hk', ht, he1⟩ := msgCopies_mem R hp he
      rw [he1]
      exact shaPos_ne_hm R hg hp hk' (by omega) hg hp (by omega)
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
      rw [chainDst_pos R hp hk2, hx]
      exact shaPos_ne_hm R hg hp hk2 (by omega) hg hp (by omega)
  rw [hr, List.nil_append]
  unfold wireFAll
  rw [List.flatMap_append, List.filter_append, hc, List.nil_append, rowSlots_size hp]
  simp only [h0, ↓reduceIte]
  rw [List.flatMap_cons, List.flatMap_nil, List.append_nil, hmWidth_eq R.rowNets, filter_map_single _ _ _ _ ht₀]

/-- **Δ's one copy into the `hm96` slot's `cv` bit `t₀`** of a row of no compression: the row prefix's midstate bit. -/
theorem hmcv0_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none) (h0 : c.ports[p]!.chunks = 0)
    (ht₀ : t₀ < 512) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = c.slotCol c.hmNet g p + t₀) =
      [(c.slotCol c.hmNet g p + t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
  rw [filter_rows R hup hg hp (Or.inr ⟨t₀, by omega, rfl⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rcases wireFAll_mem R hp he with ⟨k, t, -, hk, -⟩ | ⟨h, -⟩ <;> omega
  have hpad : (padCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨t, ht, he1⟩ := padCopies_mem R he
      rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) = [] :=
    filter_nil_of _ _ fun e he => by
      obtain ⟨k, t, -, hk, -⟩ := msgCopies_mem R hp he
      omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = c.slotCol c.hmNet g p + t₀) =
      [(c.slotCol c.hmNet g p + t₀, constSrc c pin (HmRow.midstateOf c.ports[p]!) t₀)] := by
    rw [← cvColOf_zero R hp h0 t₀]
    unfold cvCopies
    exact filter_map_single _ _ _ _ ht₀
  rw [hw, List.append_nil]
  unfold rowCopiesOf
  rw [List.filter_append, List.filter_append, hcv, hpad, hmsg]
  rfl

/-- **Δ's copy into the `hm96` slot's `pad` bit `t₀`**: the constant bit of the row's last block there. -/
theorem pad_filterAll {pin g p t₀ : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none) (ht₀ : t₀ < 1024) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
        (fun e => e.1 = padCol c g p + t₀) =
      bitCopy c pin (padCol c g p + t₀)
        (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t₀) := by
  have hx : padCol c g p + t₀ = c.slotCol c.hmNet g p + (512 + t₀) := by rw [padCol_eq R.rowNets, Nat.add_assoc]
  rw [filter_rows R hup hg hp (Or.inr ⟨512 + t₀, by omega, hx⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases wireFAll_mem R hp he with ⟨k, t, -, hk, ht, rfl⟩ | ⟨-, t, ht, rfl⟩
    · exact shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega)
    · intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hcv : (cvCopies c pin g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases cvCopies_fst R hp he with ⟨h0, t, ht, he1⟩ | ⟨-, t, ht, he1⟩
    · rw [he1]
      exact shaPos_ne_hm R hg hp h0 (by omega) hg hp (by omega)
    · rw [he1]
      intro h
      have := (R.hm_dec hg hp (by omega) hg hp (by omega) h).2.2
      omega
  have hmsg : (msgCopies c pin g p).filter (fun e => e.1 = padCol c g p + t₀) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    obtain ⟨k, t, -, hk, ht, he1⟩ := msgCopies_mem R hp he
    rw [he1]
    exact shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega)
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
theorem msg_filterAll {pin g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hup : c.ports[p]!.segs = none)
    (hkb : c.ports[p]!.chunks ≤ c.ports[p]!.bits / 1024) (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  have hk : b / 128 < c.ports[p]!.chunks := by omega
  have hmo := msgOff_lt (b := b) ht
  have hx := deltaMsgCol_pos R (g := g) (t := t) hp (by rw [← getElem!_eq_getD']; exact hup) hk
  rw [filter_rows R hup hg hp (Or.inl ⟨b / 128, 512 + msgOff b t, hk, by omega, hx⟩)]
  have hw : (wireFAll c g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    rcases wireFAll_mem R hp he with ⟨k', t', -, hk', ht', rfl⟩ | ⟨-, t', ht', rfl⟩
    · intro h
      have := (shaPos_inj R hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
      omega
    · exact fun h => shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega) h.symm
  have hr : (rowCopiesOf c pin g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] := filter_nil_of _ _ fun e he => by
    rw [hx]
    simp only [rowCopiesOf, List.mem_append] at he
    rcases he with (he | he) | he
    · rcases cvCopies_fst R hp he with ⟨h0, t', ht', he1⟩ | ⟨h, -⟩
      · rw [he1]
        intro h
        have := (shaPos_inj R hg hp h0 (by omega) hg hp hk (by omega) h).2.2.2
        omega
      · omega
    · obtain ⟨t', ht', he1⟩ := padCopies_mem R he
      rw [he1]
      exact fun h => shaPos_ne_hm R hg hp hk (by omega) hg hp (by omega) h.symm
    · obtain ⟨k', t', hk1, hk', -⟩ := msgCopies_mem R hp he
      omega
  rw [hw, hr, List.append_nil]

end Any

end FlockSoundness.Discharge.Hidden
