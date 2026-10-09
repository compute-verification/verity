import Proofs.Flock.Soundness.Discharge.Layout.KeysRow

set_option Elab.async false

/-!
# `rs`'s circuit conditions as one check on the circuit

`keyedRowsOuts` takes three conditions on the session's circuit: its unit reads only its leaf group (`h1`), the unit
returns the outputs (`ho`), and every input a unit row reads is a port bit (`PortReads`). `PortReads` is stated on
`Rows.ofNet`, which is not computable. `scopeOk` checks all three on the parsed circuit, with code that runs, and
`scopeOk_spec` gives them. `flock-verify verify` refuses every circuit failing it (`Flock.ProvedScope.check`, which
`FailClosed.scopeOk_eq` proves this one), so the headlines over what it accepts (`Discharge/FailClosed`) take none of them.

`scopeOk` also checks the unit's `unitLog ≤ 32`, under which the net's sparse rows are `Rows.ofNet`'s
(`Rows.ofNet_a`). Every accepted statement has it (`Layout.unitLog_le`), so it narrows nothing. And it checks that every
row is an unsegmented `sha512/row/v1` row of `16 · words` bits in at least one chunk (`scopeOk_ports`): `rowEnc` is that row's bytes,
and a bit row (`sha512/row/v2`) commits other bytes.

`scopeOkH` is the hidden-output headlines' scope: `scopeOk`'s conditions on the unit, and every row a `sha512/row/v1`
row or every row an unsegmented `sha512/row/v2` row of `16 · words` bits whose bytes end in its `hm96` slot's block, one
output port each (`scopeOkH_rows`). Those headlines commit each row under its own prefix (`Hidden.rowPre`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3

/-- **Every input column a computed row of `net` reads is a bit of one of its input ports.** -/
def portReadsB (net : Net) : Bool :=
  (List.range (net.constPos - min (net.inWords * WORD) net.constPos)).all fun i =>
    ((net.a.row (min (net.inWords * WORD) net.constPos + i)).toList ++
        (net.b.row (min (net.inWords * WORD) net.constPos + i)).toList).all fun x =>
      decide (x < min (net.inWords * WORD) net.constPos → x ∈ portBits net.inGroups)

/-- **Every row is an unsegmented `sha512/row/v1` row**: `16 · words` bits in at least one chunk. -/
def v1Ports (c : Circuit) : Bool :=
  c.ports.all fun p => !p.v2 && p.segs.isNone && p.bits == 16 * p.words && 0 < p.chunks

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

/-- **`scopeOk` gives `rs`'s circuit conditions**: `h1`, `ho` and `PortReads`. -/
theorem scopeOk_spec {c : Circuit} (hU : Built c.unit) (h : scopeOk c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU := by
  unfold scopeOk at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, portReads_of_portReadsB hU hu hpr⟩

/-- **`scopeOk` gives `sha512/row/v1` rows**: no bit row, `16 · words` bits, at least one chunk. -/
theorem scopeOk_ports {c : Circuit} (h : scopeOk c = true) :
    ∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks := by
  unfold scopeOk v1Ports at h
  simp only [Bool.and_eq_true, Array.all_eq_true', Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h
  intro p hp
  obtain ⟨⟨⟨hv, -⟩, hb⟩, hc⟩ := h.2 p (Array.mem_toList_iff.1 hp)
  exact ⟨hv, hb, hc⟩

/-! ## The hidden-output headlines' scope -/

/-- **A `sha512/row/v2` row the hidden-output headlines cover**: unsegmented, `16 · words` bits, its bytes filling its
compressions' blocks and ending, with SHA-512's 17 bytes of padding, in its `hm96` slot's block. -/
def v2Row (p : Port) : Bool :=
  p.v2 && p.segs.isNone && p.bits == 16 * p.words && decide (128 * p.chunks ≤ 2 * p.words) &&
    decide (2 * p.words + 17 ≤ 128 * (p.chunks + 1))

/-- **Every row is a `v2Row`**, and each output row holds one output port. -/
def v2Rows (c : Circuit) : Bool := c.ports.all v2Row && c.outRowPorts.all (· == 1)

/-- **The circuits the hidden-output headlines cover**: `scopeOk`'s conditions on the unit, and every row a
`sha512/row/v1` row (`v1Ports`) or every row a `v2Row`, one output port each (`v2Rows`). -/
def scopeOkH (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit &&
    (v1Ports c || v2Rows c)

theorem scopeOkH_of_scopeOk {c : Circuit} (h : scopeOk c = true) : scopeOkH c = true := by
  unfold scopeOk at h
  unfold scopeOkH
  simp only [Bool.and_eq_true] at h ⊢
  exact ⟨h.1, Bool.or_eq_true_iff.2 (Or.inl h.2)⟩

/-- **`scopeOkH` gives `rs`'s circuit conditions**: `h1`, `ho` and `PortReads`. -/
theorem scopeOkH_spec {c : Circuit} (hU : Built c.unit) (h : scopeOkH c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU := by
  unfold scopeOkH at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, portReads_of_portReadsB hU hu hpr⟩

/-- **`scopeOkH`'s rows**: all `sha512/row/v1` rows, or all unsegmented `sha512/row/v2` rows of `16 · words` bits
ending in their `hm96` slot's block, each output row of one output port. -/
theorem scopeOkH_rows {c : Circuit} (h : scopeOkH c = true) :
    (∀ p ∈ c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks) ∨
      ((∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 128 * p.chunks ≤ 2 * p.words ∧
        2 * p.words + 17 ≤ 128 * (p.chunks + 1)) ∧ ∀ r ∈ c.outRowPorts.toList, r = 1) := by
  unfold scopeOkH at h
  simp only [Bool.and_eq_true, Bool.or_eq_true] at h
  rcases h.2 with h1 | h2
  · left
    unfold v1Ports at h1
    simp only [Array.all_eq_true', Bool.and_eq_true, Bool.not_eq_true', beq_iff_eq, decide_eq_true_eq] at h1
    intro p hp
    obtain ⟨⟨⟨hv, -⟩, hb⟩, hc⟩ := h1 p (Array.mem_toList_iff.1 hp)
    exact ⟨hv, hb, hc⟩
  · right
    unfold v2Rows v2Row at h2
    simp only [Bool.and_eq_true, Array.all_eq_true', beq_iff_eq, decide_eq_true_eq, Option.isNone_iff_eq_none] at h2
    refine ⟨fun p hp => ?_, fun r hr => h2.2 r (Array.mem_toList_iff.1 hr)⟩
    obtain ⟨⟨⟨⟨hv, hs⟩, hb⟩, h1⟩, h2'⟩ := h2.1 p (Array.mem_toList_iff.1 hp)
    exact ⟨hv, hs, hb, h1, h2'⟩

end FlockSoundness.Discharge.Layout
