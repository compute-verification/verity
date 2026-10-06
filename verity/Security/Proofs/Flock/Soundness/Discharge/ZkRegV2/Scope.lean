import Proofs.Flock.Soundness.Discharge.Layout.Scope

set_option Elab.async false

/-!
# `rs`'s circuit conditions for registered sessions: `sha512/row/v2` rows

`Layout.scopeOk` admits only `sha512/row/v1` rows (`v1Ports`), and the verifier's registered check admits a registered
port only as a `sha512/row/v2` bit row (`Registered.Port.fits`), so no circuit with a registered port meets both.
`scopeOkV2` is `scopeOk`'s unit conditions with every row a `sha512/row/v2` bit row, not segmented, of `16 · words` bits
in whole SHA-512 blocks (`2 · words = 128 · chunks`, `v2Ports`): the rows `Hidden.row_bc_tgt` hashes under their own
prefix. One circuit's rows are all of one kind: the rows' byte encoding is one function of a value's width.
-/

namespace FlockSoundness.Discharge.ZkRegV2

open Flock FlockLevel3 FlockSoundness.Discharge.Layout

/-- **Every row is a whole-block `sha512/row/v2` bit row**: not segmented, `16 · words` bits, `2 · words = 128 · chunks`. -/
def v2Ports (c : Circuit) : Bool :=
  c.ports.all fun p => p.v2 && p.segs.isNone && p.bits == 16 * p.words && 2 * p.words == 128 * p.chunks

/-- **The circuits `rs` covers with `sha512/row/v2` rows**: `scopeOk`'s unit conditions, and `v2Ports`. -/
def scopeOkV2 (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && portReadsB c.unit && v2Ports c

/-- **`scopeOkV2` gives `rs`'s circuit conditions**: `h1`, `ho` and `PortReads`. -/
theorem scopeOkV2_spec {c : Circuit} (hU : Built c.unit) (h : scopeOkV2 c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU := by
  unfold scopeOkV2 at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, portReads_of_portReadsB hU hu hpr⟩

/-- **`scopeOkV2` gives whole-block `sha512/row/v2` rows.** -/
theorem scopeOkV2_ports {c : Circuit} (h : scopeOkV2 c = true) :
    ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks := by
  unfold scopeOkV2 v2Ports at h
  simp only [Bool.and_eq_true, Array.all_eq_true', beq_iff_eq, decide_eq_true_eq, Option.isNone_iff_eq_none] at h
  intro p hp
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h.2 p (Array.mem_toList_iff.1 hp)
  exact ⟨h1, h2, h3, h4⟩

/-! ## Hidden outputs -/

/-- **The hidden-output circuits `rs` covers with `sha512/row/v2` rows**: `scopeOkV2`, and one output port per output
row (`outRowPorts`), as C-Flock's composer writes a hidden-output statement's output rows (`out_row`, no
`out_row_ports`). At `sha512/row/v1` rows `HmOut.checkRows` gives the second (`Hidden.out_rows_v1`); a `sha512/row/v2`
output row may hold a run of output ports, which `Hidden.OutRows` does not cover. -/
def scopeOkHV2 (c : Circuit) : Bool :=
  scopeOkV2 c && c.outRowPorts == Array.replicate c.outPorts.size 1

theorem scopeOkHV2_v2 {c : Circuit} (h : scopeOkHV2 c = true) : scopeOkV2 c = true := by
  unfold scopeOkHV2 at h
  simp only [Bool.and_eq_true] at h
  exact h.1

/-- **`scopeOkHV2` gives one output port per output row.** -/
theorem scopeOkHV2_rows {c : Circuit} (h : scopeOkHV2 c = true) :
    c.outRowPorts = Array.replicate c.outPorts.size 1 := by
  unfold scopeOkHV2 at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.2

theorem scopeOkHV2_spec {c : Circuit} (hU : Built c.unit) (h : scopeOkHV2 c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ PortReads c hU :=
  scopeOkV2_spec hU (scopeOkHV2_v2 h)

theorem scopeOkHV2_ports {c : Circuit} (h : scopeOkHV2 c = true) :
    ∀ p ∈ c.ports.toList, p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 2 * p.words = 128 * p.chunks :=
  scopeOkV2_ports (scopeOkHV2_v2 h)

end FlockSoundness.Discharge.ZkRegV2
