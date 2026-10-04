import FlockSoundness.Discharge.Layout.Scope

set_option Elab.async false

/-!
# The scope widened to bit rows and segmented rows

`Layout.scopeOk` checks that every row is a `sha512/row/v1` row (`Layout.v1Ports`). A manifest with `row_schemas`
(every PoUW circuit) makes every port a bit row (`HmRow.rowPorts` sets `v2`), so no PoUW circuit is in that scope.
`scopeOkW` keeps `scopeOk`'s circuit conditions and replaces `v1Ports` by `rowShapeW`, which admits four row shapes:

* `RowV1`: a `sha512/row/v1` row, as `scopeOk` admits;
* `RowV2Whole`: a `sha512/row/v2` row of `16 · words` bits in whole SHA-512 blocks (`2 · words = 128 · chunks`), its
  padding a block of its own in the `hm96` slot, as a `sha512/row/v1` row's;
* `RowV2Short`: a `sha512/row/v2` row of at most 111 bytes with no compression before its `hm96` slot (`chunks = 0`):
  its bytes and SHA-512's padding are the `hm96` slot's one message block;
* `RowSegWhole`: a `sha512/row-seg/v1` row of a multiple of 64 words, so every segment's bytes are whole blocks and
  every segment's padding is a block of its own; the outer message (`64 s` digest bytes) ends in the `hm96` slot, with
  or without digest bytes in its block (`s` odd or even).

Input rows are one port each; an output row is one port or a run of output ports (`HmOut.outGroup`), of any of the
four shapes. The shapes left out are the bit rows whose last data block is partial (a `sha512/row/v2` row of more than
111 bytes not a multiple of 128, a segmented row whose last segment is not whole blocks).

`scopeOkW_of_scopeOk`: the widened scope contains `scopeOk`'s. The PoUW examples are `HmRow.rowPorts`'s ports for the
rows of `verity_pouw.circuit.leaves` (`bitPort`): every one is in scope.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3

/-- **A row the widened scope covers**: `16 · words` bits, and a `sha512/row/v1` row with at least one chunk, a
`sha512/row/v2` row of whole blocks or of one short block, or a `sha512/row-seg/v1` row of whole 64-word blocks, its
segments and compressions `HmRow.segShape`'s. -/
def rowShapeW (p : Port) : Bool :=
  p.bits == 16 * p.words && (if !p.v2 then 0 < p.chunks else 0 < p.words && match p.segs with
    | none => (0 < p.chunks && 2 * p.words == 128 * p.chunks) || (p.chunks == 0 && 2 * p.words + 17 ≤ 128)
    | some segs => p.words % 64 == 0 && segs == (HmRow.segShape p.bits).1 &&
        p.chunks + 1 == segs.foldl (· + ·.2) 0 + (HmRow.segShape p.bits).2)

/-- **The circuits the widened scope covers**: `Layout.scopeOk`'s circuit conditions, and every row of a shape
`rowShapeW` admits. -/
def scopeOkW (c : Circuit) : Bool :=
  c.unit.inGroups.size == 1 && c.outNet == c.unitNet && decide (c.unit.unitLog ≤ 32) && Layout.portReadsB c.unit &&
    c.ports.all rowShapeW

/-- A `sha512/row/v1` row of `16 · words` bits in at least one chunk. -/
def RowV1 (p : Port) : Prop := p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks

/-- A `sha512/row/v2` row of `16 · words` bits in whole blocks, `chunks` of them. -/
def RowV2Whole (p : Port) : Prop :=
  p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 0 < p.words ∧ 0 < p.chunks ∧ 2 * p.words = 128 * p.chunks

/-- A `sha512/row/v2` row of `16 · words` bits whose bytes and padding are one block, the `hm96` slot's. -/
def RowV2Short (p : Port) : Prop :=
  p.v2 = true ∧ p.segs = none ∧ p.bits = 16 * p.words ∧ 0 < p.words ∧ p.chunks = 0 ∧ 2 * p.words + 17 ≤ 128

/-- A `sha512/row-seg/v1` row of `16 · words` bits, `words` a multiple of 64, its segments and compressions
`HmRow.segShape`'s. -/
def RowSegWhole (p : Port) : Prop :=
  p.v2 = true ∧ p.segs = some (HmRow.segShape p.bits).1 ∧ p.bits = 16 * p.words ∧ 0 < p.words ∧ p.words % 64 = 0 ∧
    p.chunks + 1 = (HmRow.segShape p.bits).1.foldl (· + ·.2) 0 + (HmRow.segShape p.bits).2

/-- **`rowShapeW`'s four shapes.** -/
theorem rowShapeW_spec {p : Port} (h : rowShapeW p = true) :
    RowV1 p ∨ RowV2Whole p ∨ RowV2Short p ∨ RowSegWhole p := by
  unfold rowShapeW at h
  unfold RowV1 RowV2Whole RowV2Short RowSegWhole
  rcases hv : p.v2 with _ | _ <;> rcases hs : p.segs with _ | segs <;>
    simp only [hv, hs, Bool.not_false, Bool.not_true, ↓reduceIte, Bool.false_eq_true, Bool.and_eq_true,
      Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  · exact Or.inl ⟨rfl, h.1, h.2⟩
  · exact Or.inl ⟨rfl, h.1, h.2⟩
  · obtain ⟨hb, hw, (⟨hc, he⟩ | ⟨hc, he⟩)⟩ := h
    · exact Or.inr (Or.inl ⟨rfl, rfl, hb, hw, hc, he⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨rfl, rfl, hb, hw, hc, he⟩))
  · obtain ⟨hb, hw, ⟨⟨h64, rfl⟩, hc⟩⟩ := h
    exact Or.inr (Or.inr (Or.inr ⟨rfl, rfl, hb, hw, h64, hc⟩))

/-- **`scopeOkW` gives `rs`'s circuit conditions**, as `Layout.scopeOk_spec`: `h1`, `ho` and `PortReads`. -/
theorem scopeOkW_spec {c : Circuit} (hU : Built c.unit) (h : scopeOkW c = true) :
    c.unit.inGroups.size = 1 ∧ c.outNet = c.unitNet ∧ Layout.PortReads c hU := by
  unfold scopeOkW at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := h
  exact ⟨h1, ho, Layout.portReads_of_portReadsB hU hu hpr⟩

/-- **`scopeOkW` gives every row one of the four shapes** (`rowShapeW`). -/
theorem scopeOkW_ports {c : Circuit} (h : scopeOkW c = true) : ∀ p ∈ c.ports.toList, rowShapeW p = true := by
  unfold scopeOkW at h
  simp only [Bool.and_eq_true, Array.all_eq_true'] at h
  exact fun p hp => h.2 p (Array.mem_toList_iff.1 hp)

/-- **The widened scope contains `scopeOk`'s**: a `sha512/row/v1` row is `rowShapeW`'s first shape. -/
theorem scopeOkW_of_scopeOk {c : Circuit} (h : Layout.scopeOk c = true) : scopeOkW c = true := by
  have hp := Layout.scopeOk_ports h
  unfold Layout.scopeOk at h
  unfold scopeOkW
  simp only [Bool.and_eq_true] at h ⊢
  refine ⟨h.1, ?_⟩
  simp only [Array.all_eq_true']
  intro p hm
  obtain ⟨hv, hb, hc⟩ := hp p (Array.mem_toList_iff.2 hm)
  unfold rowShapeW
  simp only [hv, hb, Bool.not_false, ↓reduceIte, beq_self_eq_true, Bool.true_and, decide_eq_true_eq]
  exact hc

/-! ## PoUW's rows -/

/-- **A bit row of `l` bits as `HmRow.rowPorts` reads it** from META `row_schemas`: `(l + 15) / 16` words, and the
compressions before the last block of a `sha512/row-seg/v1` row's messages (`seg`) or of a `sha512/row/v2` row's. -/
def bitPort (l : ℕ) (seg : Bool) : Port :=
  if seg then
    { name := "", words := (l + 15) / 16, nb := 0, bits := l, v2 := true, segs := some (HmRow.segShape l).1,
      chunks := (HmRow.segShape l).1.foldl (· + ·.2) 0 + (HmRow.segShape l).2 - 1 }
  else
    { name := "", words := (l + 15) / 16, nb := 0, bits := l, v2 := true,
      chunks := ((l + 7) / 8 + 17 + 127) / 128 - 1 }

/-- An nvfp4-p X row at `K = 4096`: 1216 words, `sha512/row-seg/v1`, three segments (`s` odd). -/
example : rowShapeW (bitPort (16 * 1216) true) = true := by decide +kernel

/-- An nvfp4-p X row at `K = 8192`: 2368 words, five segments, the last of 5120 bits. -/
example : rowShapeW (bitPort (16 * 2368) true) = true := by decide +kernel

/-- A bf16 X row at `K = 4096`: 4096 words, eight segments (`s` even). -/
example : rowShapeW (bitPort (16 * 4096) true) = true := by decide +kernel

/-- An nvfp4-p W row at `K = 4096` and `K = 8192`: `sha512/row/v2`, 19 and 37 whole blocks. -/
example : rowShapeW (bitPort (16 * 1216) false) = true := by decide
example : rowShapeW (bitPort (16 * 2368) false) = true := by decide

/-- The ok row (one word) and the tile digest's row (16 words): `sha512/row/v2`, one short block. -/
example : rowShapeW (bitPort 16 false) = true := by decide
example : rowShapeW (bitPort (16 * 16) false) = true := by decide

/-- Out of scope: a `sha512/row/v2` row of 65 words, its last data block partial. -/
example : rowShapeW (bitPort (16 * 65) false) = false := by decide

end FlockSoundness.Discharge.ScopeW
