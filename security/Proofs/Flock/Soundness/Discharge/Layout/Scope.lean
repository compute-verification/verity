module

public import Proofs.Flock.Soundness.Discharge.Layout.KeysRow
public import Flock.ProvedScope

@[expose] public section

set_option Elab.async false

/-! `rs`'s circuit conditions as checks on the parsed circuit (`scopeOk`, `scopeOkH`, `scopeGenH`) that run. -/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3

/-- **Every input column a computed row of `net` reads is a bit of one of its input ports.** -/
def portReadsB (net : Net) : Bool :=
  (List.range (net.constPos - min (net.inWords * WORD) net.constPos)).all fun i =>
    ((net.a.row (min (net.inWords * WORD) net.constPos + i)).toList ++
        (net.b.row (min (net.inWords * WORD) net.constPos + i)).toList).all fun x =>
      decide (x < min (net.inWords * WORD) net.constPos → x ∈ portBits net.inGroups)

/-- **No output port is pinned** (META `pinned_outputs`, `HmRow.pinnedCount`). -/
def unpinned (c : Circuit) : Bool := HmRow.pinnedCount c.metaJ == 0

/-- **Every row is an unsegmented `sha512/row/v1` row**: `16 · words` bits in at least one chunk. -/
def v1Rows (c : Circuit) : Bool :=
  c.ports.all fun p => !p.v2 && p.segs.isNone && p.bits == 16 * p.words && 0 < p.chunks

/-- **`v1Rows`, and no pinned output** (`unpinned`). -/
def v1Ports (c : Circuit) : Bool := v1Rows c && unpinned c

/-- **The circuits `rs` covers**: the unit reads only its leaf group, returns the outputs, and reads only port bits among
its inputs, and every row is a `sha512/row/v1` row. -/
def scopeOk (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && v1Ports c

theorem portReads_of_portReadsB {c : Circuit} (hU : Built c.unit) (hu : c.unit.unitLog ≤ 32)
    (h : portReadsB c.unit = true) : PortReads c hU := by
  intro j x hx hxn
  unfold portReadsB at h
  rw [List.all_eq_true] at h
  have hi := h j.val (List.mem_range.2 j.isLt)
  rw [List.all_eq_true] at hi
  have hmem : x.val ∈ (c.unit.a.row ((Rows.ofNet hU).nIn + j.val)).toList ++
      (c.unit.b.row ((Rows.ofNet hU).nIn + j.val)).toList := by
    rcases List.mem_append.1 hx with hx | hx
    · rw [← Rows.ofNet_a hU hu j]
      exact List.mem_append_left _ (List.mem_map_of_mem hx)
    · rw [← Rows.ofNet_b hU hu j]
      exact List.mem_append_right _ (List.mem_map_of_mem hx)
  exact of_decide_eq_true (hi x.val hmem) hxn

theorem scopeOk_spec {c : Circuit} (hU : Built c.unit) (h : scopeOk c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU := by
  unfold scopeOk at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, portReads_of_portReadsB hU hu hpr⟩

theorem scopeOk_ports {c : Circuit} (h : scopeOk c = true) :
    ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
  unfold scopeOk v1Ports v1Rows at h
  simp only [Bool.and_eq_true, Array.all_eq_true', Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h
  intro p hp
  obtain ⟨⟨⟨hv, -⟩, hb⟩, hc⟩ := h.2.1 p (Array.mem_toList_iff.1 hp)
  exact ⟨hv, hb, hc⟩

/-- **A `sha512/row/v2` row the hidden-output headlines cover**: unsegmented, `16 · words` bits, its bytes and SHA-512's
17 bytes of padding filling its compressions' blocks and ending in its `hm96` slot's block. -/
def v2Row (p : Port) : Bool :=
  p.v2 && p.segs.isNone && p.bits == 16 * p.words && decide (128 * p.chunks ≤ 2 * p.words + 16) &&
    decide (2 * p.words + 17 ≤ 128 * (p.chunks + 1))

/-- **A `sha512/row-seg/v1` row the hidden-output headlines cover**: segmented as `HmRow.segShape` cuts its bits,
`16 · words` bits, its compressions its segments' and then its outer message's, and each segment's bytes ending, with
SHA-512's 17 bytes of padding, in its last compression's block. -/
def segRow (p : Port) : Bool :=
  let s := (p.bits + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS
  p.segs == some (HmRow.segShape p.bits).1 && p.bits == 16 * p.words &&
    p.chunks == (((HmRow.segShape p.bits).1.toList.take s).map Prod.snd).sum + ((64 * s + 17 + 127) / 128 - 1) &&
    (List.range s).all fun j =>
      let m := min (p.bits - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS
      decide (128 * (((m + 7) / 8 + 17 + 127) / 128 - 1) ≤ (m + 7) / 8)

/-- **Every row is a `v2Row` or a `segRow`**, of any mix, and each output row is a `v2Row` or a `segRow` holding one
output port. -/
def bitRowsU (c : Circuit) : Bool :=
  c.ports.all (fun p => v2Row p || segRow p) && c.outRowPorts.all (· == 1) &&
    (List.range c.outRowPorts.size).all fun j =>
      v2Row (c.ports.getD (HmOut.nIn c + j) default) || segRow (c.ports.getD (HmOut.nIn c + j) default)

/-- **`bitRowsU`, and no pinned output** (`unpinned`). -/
def bitRows (c : Circuit) : Bool := bitRowsU c && unpinned c

/-- **The circuits the hidden-output headlines cover**: `scopeOk`'s conditions on the unit, and every row a
`sha512/row/v1` row (`v1Ports`) or every row a `v2Row` or a `segRow`, each output row of one output port
(`bitRows`). -/
def scopeOkH (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit &&
    (v1Ports c || bitRows c)

theorem scopeOkH_of_scopeOk {c : Circuit} (h : scopeOk c = true) : scopeOkH c = true := by
  unfold scopeOk at h
  unfold scopeOkH
  simp only [Bool.and_eq_true] at h ⊢
  exact ⟨h.1, Bool.or_eq_true_iff.2 (Or.inl h.2)⟩

theorem scopeOkH_spec {c : Circuit} (hU : Built c.unit) (h : scopeOkH c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU := by
  unfold scopeOkH at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, portReads_of_portReadsB hU hu hpr⟩

theorem v2Row_spec {p : Port} (h : v2Row p = true) :
    p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 128 * p.chunks ≤ 2 * p.words + 16 ∧
      2 * p.words + 17 ≤ 128 * (p.chunks + 1) := by
  unfold v2Row at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Option.isNone_iff_eq_none] at h
  obtain ⟨⟨⟨⟨hv, hs⟩, hb⟩, h1⟩, h2⟩ := h
  exact ⟨hv, hs, hb, h1, h2⟩

theorem segRow_spec {p : Port} (h : segRow p = true) :
    p.segs = some (HmRow.segShape p.bits).1 ∧ p.bits = 16 * p.words ∧
      p.chunks = (((HmRow.segShape p.bits).1.toList.take ((p.bits + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS)).map
        Prod.snd).sum + ((64 * ((p.bits + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS) + 17 + 127) / 128 - 1) ∧
      ∀ j < (p.bits + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS,
        128 * (((min (p.bits - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS + 7) / 8 + 17 + 127) / 128 - 1) ≤
          (min (p.bits - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS + 7) / 8 := by
  unfold segRow at h
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range, decide_eq_true_eq] at h
  obtain ⟨⟨⟨hs, hb⟩, hn⟩, hf⟩ := h
  exact ⟨hs, hb, hn, hf⟩

theorem rowsH_spec {c : Circuit} (h : (v1Ports c || bitRows c) = true) :
    (∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) ∨
      ((∀ p ∈ c.ports.toList, v2Row p = true ∨ segRow p = true) ∧ (∀ r ∈ c.outRowPorts.toList, r = 1) ∧
        ∀ j < c.outRowPorts.size, v2Row (c.ports.getD (HmOut.nIn c + j) default) = true ∨
          segRow (c.ports.getD (HmOut.nIn c + j) default) = true) := by
  simp only [Bool.or_eq_true] at h
  rcases h with h1 | h2
  · left
    unfold v1Ports v1Rows at h1
    simp only [Array.all_eq_true', Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h1
    intro p hp
    obtain ⟨⟨⟨hv, -⟩, hb⟩, hc⟩ := h1.1 p (Array.mem_toList_iff.1 hp)
    exact ⟨hv, hb, hc⟩
  · right
    unfold bitRows bitRowsU at h2
    simp only [Bool.and_eq_true, Array.all_eq_true', Bool.or_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at h2
    obtain ⟨⟨⟨hp, hr⟩, ho⟩, -⟩ := h2
    exact ⟨fun p hm => hp p (Array.mem_toList_iff.1 hm), fun r hm => hr r (Array.mem_toList_iff.1 hm), ho⟩

theorem rowsH_unpinned {c : Circuit} (h : (v1Ports c || bitRows c) = true) : HmRow.pinnedCount c.metaJ = 0 := by
  have : unpinned c = true := by
    unfold v1Ports bitRows at h
    simp only [Bool.or_eq_true, Bool.and_eq_true] at h
    rcases h with h | h <;> exact h.2
  simpa [unpinned] using this

theorem scopeOkH_rowsB {c : Circuit} (h : scopeOkH c = true) : (v1Ports c || bitRows c) = true := by
  unfold scopeOkH at h
  simp only [Bool.and_eq_true] at h
  exact h.2

theorem scopeOkH_rows {c : Circuit} (h : scopeOkH c = true) :
    (∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) ∨
      ((∀ p ∈ c.ports.toList, v2Row p = true ∨ segRow p = true) ∧ (∀ r ∈ c.outRowPorts.toList, r = 1) ∧
        ∀ j < c.outRowPorts.size, v2Row (c.ports.getD (HmOut.nIn c + j) default) = true ∨
          segRow (c.ports.getD (HmOut.nIn c + j) default) = true) :=
  rowsH_spec (scopeOkH_rowsB h)

/-- **The general form under hidden outputs**: the wires compute in some order (`ProvedScope.wiresAcyclic`), the unit
has at most `2^32` rows and reads only port bits among its inputs, and the rows are the hidden-output headlines'
(`v1Ports` or `bitRows`). No condition on the unit's input groups, on the stages or on which net returns the outputs. -/
def scopeGenH (c : Circuit) : Bool :=
  ProvedScope.wiresAcyclic c && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && (v1Ports c || bitRows c)

theorem scopeGenH_spec {c : Circuit} (hU : Built c.unit) (h : scopeGenH c = true) :
    PortReads c hU ∧ ProvedScope.wiresAcyclic c = true := by
  unfold scopeGenH at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨hw, hu⟩, hpr⟩, -⟩ := h
  exact ⟨portReads_of_portReadsB hU hu hpr, hw⟩

theorem scopeGenH_rowsB {c : Circuit} (h : scopeGenH c = true) : (v1Ports c || bitRows c) = true := by
  unfold scopeGenH at h
  simp only [Bool.and_eq_true] at h
  exact h.2

theorem scopeGenH_of_scopeOkH {c : Circuit} (h : scopeOkH c = true) (hw : ProvedScope.wiresAcyclic c = true) :
    scopeGenH c = true := by
  unfold scopeOkH at h
  unfold scopeGenH
  simp only [Bool.and_eq_true] at h ⊢
  exact ⟨⟨⟨hw, h.1.1.2⟩, h.1.2⟩, h.2⟩

/-- `scopeGenH` without the wires' order, which the check requires wherever the wires feed the unit or the outputs
(`ProvedScope.wiresFeed`). -/
def scopeOkI (c : Circuit) : Bool :=
  decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && (v1Ports c || bitRows c)

theorem scopeOkI_of_scopeOkH {c : Circuit} (h : scopeOkH c = true) : scopeOkI c = true := by
  unfold scopeOkH at h
  unfold scopeOkI
  simp only [Bool.and_eq_true] at h ⊢
  exact ⟨⟨h.1.1.2, h.1.2⟩, h.2⟩

theorem outNet_of_wiresFeed {c : Circuit} (hf : ProvedScope.wiresFeed c = false) : c.outNet = c.unitNet := by
  unfold ProvedScope.wiresFeed at hf
  simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hf
  exact hf.2

theorem scopeOkH_of_scopeOkI {c : Circuit} (h : scopeOkI c = true) (hf : ProvedScope.wiresFeed c = false) :
    scopeOkH c = true := by
  unfold ProvedScope.wiresFeed at hf
  unfold scopeOkI at h
  unfold scopeOkH
  simp only [Bool.or_eq_false_iff, bne_eq_false_iff_eq] at hf
  simp only [Bool.and_eq_true, beq_iff_eq] at h ⊢
  exact ⟨⟨⟨⟨hf.1, hf.2⟩, h.1.1⟩, h.1.2⟩, h.2⟩

theorem scopeGenH_of_scopeOkI {c : Circuit} (h : scopeOkI c = true) (hw : ProvedScope.wiresAcyclic c = true) :
    scopeGenH c = true := by
  unfold scopeOkI at h
  unfold scopeGenH
  simp only [Bool.and_eq_true] at h ⊢
  exact ⟨⟨⟨hw, h.1.1⟩, h.1.2⟩, h.2⟩

end FlockSoundness.Discharge.Layout
