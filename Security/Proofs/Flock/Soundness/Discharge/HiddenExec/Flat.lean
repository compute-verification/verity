import Proofs.Flock.Soundness.Discharge.Hidden.Commit
import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS
import Proofs.Flock.Soundness.Discharge.Layout.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Unpin
import Proofs.Flock.Soundness.ParseStmt

/-!
# A hidden-output statement's flat twin

`Stmt.setupHidden`'s Δ is `HmRow.delta`'s, then `Hidden.outPairs` (`Hidden.HiddenDelta`): each output row's message
bits, copied from the output bits the circuit computes. Its flat twin `Hidden.flatTwin st` is the statement with
`HmRow.delta`'s Δ alone, the one `Stmt.setupH` would have built on the same circuit. It has `layout_of`'s layout
(`flat_layout`) and the unit facts the copy positions read (`flat_unitFacts`), as setupH's statement does.

Every first column of `outPairs` is a message bit of a row's `sha512x3` slot (`deltaMsgCol`), which no unit slot's row
is (`outPairs_ne_unit`, for unsegmented output rows, `Hidden.Any.OutRowsS`). So at every unit slot's row the hidden
statement's level-0 matrices are its flat twin's (`unitRowsFlat`): the rows that place a template instance and copy its
inputs read the same in both. The hidden-output headlines' rows (`Layout.v1Ports` or `Layout.bitRows`, which
`Layout.scopeOkH` and `Layout.scopeGenH` require) are each unsegmented, their padding perhaps starting in their last
compression's block, or `sha512/row-seg/v1`, the output rows unsegmented (`rowsFit_of_rows`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **The hidden statement reads its flat twin's rows at every unit slot's row**, in `A` and in `B`. -/
def UnitRowsFlat (st : Stmt) : Prop :=
  ∀ s₀ < st.c.unitRange.count, ∀ b < 2 ^ st.c.unit.unitLog, ∀ r j : Fin (2 ^ st.c.kLog),
    r.val = st.c.unitRange.slot s₀ + b → placedA st r j = placedA (Hidden.flatTwin st) r j ∧
      placedB st r j = placedB (Hidden.flatTwin st) r j

/-! ## What `HmOut.checkRows` checks -/

/-- `HmOut.checkRows` passes only when the rows are the input rows, then one per run of output ports. -/
theorem checkRows_le {c : Circuit} (h : HmOut.checkRows c = .ok ()) : c.outRowPorts.size ≤ c.ports.size := by
  unfold HmOut.checkRows at h
  obtain ⟨k, -, h⟩ := ebind_ok h
  by_contra hlt
  have hne : (k + c.outRowPorts.size != c.ports.size) = true := by
    simp only [bne_iff_ne, ne_eq]
    omega
  simp only [hne, Bool.true_or, ↓reduceIte] at h
  cases h

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-- **The flat twin of an untyped hidden statement has the layout the placement reads** (`Hidden.flat_layoutC`). -/
theorem flat_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (Hidden.flatTwin st) :=
  Hidden.flat_layoutC ht (Hidden.setupHidden_specC h)

/-- **The flat twin's unit facts** (`Hidden.flat_unitFactsC`). -/
theorem flat_unitFacts (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) :=
  Hidden.flat_unitFactsC ht (Hidden.setupHidden_specC h)

/-! ## Δ's output pairs are at the rows' `sha512x3` slots -/

/-- The first column of each of Δ's output pairs is `deltaMsgCol` at a template instance, output row, word and bit. -/
theorem outPairs_fst {c : Circuit} {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs c) :
    ∃ g < c.g, ∃ j < c.outRowPorts.size, ∃ k < (c.ports.getD (HmOut.nIn c + j) default).words, ∃ t < 16,
      q.1 = Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) := by
  simp only [Hidden.outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
  exact ⟨g, hg, j, hj, k, hk, t, ht, by rcases hq with rfl | rfl <;> rfl⟩

/-- **A row's message bit at its `sha512x3` slot is no unit slot's row**: compression `byte / 128` of row `p` is one of
the template instance's `sha512x3` slots, its message input 1024 bits wide, and the bit is inside it. -/
theorem deltaMsgCol_ne_unit (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) {g p byte t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hs : (st.c.ports.getD p default).segs = none) (hbyte : byte / 128 < st.c.ports[p]!.chunks) (ht : t < 8) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : Hidden.deltaMsgCol st.c g p byte t ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨h1, h2⟩ := Hm.rowSlot_lt hp hbyte
  have hsz := C.sha_in.1
  have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[p]!)[byte / 128]!.2 + 1 < 6 := by omega
  have hw := C.sha_in.2 _ hidx
  rw [sha_width _ _ (by norm_num : 1 < 2)] at hw
  have hne := L.ne_unit_at P C P.shaNet hsha.symm hg (by rw [C.perVu_sha]; exact h1)
    (getElem_mem_toList' (by rw [hsz]; exact hidx)) (tt := 64 * (byte % 128 / 8) + 56 - 8 * (byte % 128 % 8) + t)
    (by rw [hw]; omega) hs₀ hb
  have hk : (HmRow.byteAt (st.c.ports.getD p default) byte).1 < ((HmRow.rowSlots st.c.ports).2[p]!).size := by
    rw [Typed.byteAt_none hs, Hm.rowSlots_size hp]
    exact hbyte
  simp only [Hidden.deltaMsgCol, hk, ↓reduceIte]
  rw [Typed.byteAt_none hs]
  dsimp only
  intro he
  refine hne (Eq.trans ?_ he)
  omega

/-- **No output pair of Δ is at a unit slot's row**, at unsegmented output rows: every output row's data bit is an input bit of
its compression's `sha512x3` slot or of its `hm96` slot (`Hidden.Any.msg_at`), another net's slot. -/
theorem outPairs_ne_unitP (L : Layout st) (F : Aliased.UnitFacts tags st) (hrows : Hidden.Any.OutRowsP st.c)
    {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  obtain ⟨hp, -, ht8⟩ := hrows.bound hj hk ht
  have R := Hm.RowRanges.ofL L F.parse F.check
  obtain ⟨n, s, off, hn, hat, -⟩ := Hidden.Any.msg_at R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hun : st.c.unitNet ≠ n := by
    rcases hn with rfl | rfl
    · exact F.sha
    · exact F.hm
  intro he
  have hb' : b < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact hb
  exact Hm.atSlot_ne L F (R.rowNet_lt hn) F.parse.unitNet hun hat
    (show AtSlot st.c st.c.unitNet s₀ b _ from ⟨hs₀, hb', he⟩)

/-- `outPairs_ne_unitP` at one output row per output port. -/
theorem outPairs_ne_unit (L : Layout st) (F : Aliased.UnitFacts tags st) (hrows : Hidden.Any.OutRowsS st.c)
    {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b :=
  outPairs_ne_unitP L F hrows.toP hq hs₀ hb

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, at unsegmented output rows, its Δ being
`HmRow.delta`'s, then `outPairs` (`Hidden.hidden_delta_unpinned`: one output row per output port pins none), of a session
on its committed circuit, flat or templated (`Hidden.SpecC`). -/
theorem unitRowsFlatC (ht : tags.typed = false) (sc : Hidden.SpecC tags cf pf coins tables draw st)
    (hrows : Hidden.Any.OutRowsS st.c) : UnitRowsFlat st := by
  have s := sc.self
  have L := sc.flatLayout ht
  have F := sc.unitFacts ht
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F hrows hq hs₀ hb
  have hpin := sc.committed.unpinned_of_rep ht hrows.1
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta_unpinned hpin]
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.db, Hidden.hidden_delta_unpinned hpin]
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

/-- **`unitRowsFlatC` at `Stmt.setupHidden`'s accept.** -/
theorem unitRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hrows : Hidden.Any.OutRowsS st.c) : UnitRowsFlat st :=
  unitRowsFlatC ht (Hidden.setupHidden_specCC h) hrows

/-- **`unpin` of an untyped hidden statement reads its flat twin's rows at every unit slot's row**, pinned outputs or
not, of a session on its committed circuit (`Hidden.SpecC`): its Δ is `HmRow.delta`'s, then `outPairs`, by definition
(`Hidden.unpin`). -/
theorem unitRowsFlatC_unpin (ht : tags.typed = false) (sc : Hidden.SpecC tags cf pf coins tables draw st)
    (hrows : Hidden.Any.OutRowsP st.c) : UnitRowsFlat (Hidden.unpin st) := by
  have L := sc.flatLayout ht
  have F := sc.unitFacts ht
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unitP L F hrows hq hs₀ hb
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val +
        deltaEntry ((HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray) r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val +
        deltaEntry ((HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray) r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [Hidden.deltaEntry_append_out hout]

/-- **`unitRowsFlatC_unpin` at `Stmt.setupHidden`'s accept.** -/
theorem unitRowsFlat_unpin (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hrows : Hidden.Any.OutRowsP st.c) : UnitRowsFlat (Hidden.unpin st) :=
  unitRowsFlatC_unpin ht (Hidden.setupHidden_specCC h) hrows

/-! ## The rows `scopeOkH` covers -/

/-- **The hidden-output headlines' rows are covered**: on a circuit `HmRow.parse` accepts after any nets ahead of the
file's (`pre`: a templated file's read unit) and `HmOut.checkRows` passes, all `sha512/row/v1` rows (whole blocks, `ParseFacts.portWords`; one output row per output port,
`Hidden.out_rows_v1`), or each row a `v2Row` or a `segRow`, each output row a `v2Row` of one output port. -/
theorem rowsFit_of_rowsPre {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} {c : Circuit}
    (hp : HmRow.parse tags cf tables (pre := pre) = .ok c) (hc : HmOut.checkRows c = .ok ())
    (h : (Layout.v1Ports c || Layout.bitRows c) = true) : Hidden.Any.RowsFitS c := by
  have P := (parse_factsPre_nets hp).1
  rcases Layout.rowsH_spec h with h1 | ⟨h2, hr, ho⟩
  · have hv : ∀ p ∈ c.ports.toList, p.v2 = false := fun p hm => (h1 p hm).1
    obtain ⟨hu, hrep, hj⟩ := Hidden.out_rows_v1Pre hp P hc hv
    refine Hidden.Any.RowsFit.rowsFitS ⟨⟨fun p hm => ?_, hrep, fun j hj' => (hj j hj').1⟩, fun p hm => ?_⟩
    · have hw := P.portWords p hm (hv p hm)
      exact ⟨hu p hm, (h1 p hm).2.1, by omega, by omega⟩
    · have hr2 : Hidden.rowV2 c = false := by
        unfold Hidden.rowV2
        rw [Array.any_eq_false']
        intro p hm
        rw [hv p (Array.mem_toList_iff.2 hm)]
        decide
      rw [hv p hm, hr2]
  · have hsum : c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size := by
      rw [(Hidden.parse_rowPorts hp).2, Layout.rowsH_unpinned h, Nat.sub_zero]
    have hrep := Hidden.outRows_single hsum fun j hj => by
      simp only [HmOut.outGroup, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj, Option.getD_some]
      exact hr _ (Array.getElem_mem_toList hj)
    have hsz : c.outRowPorts.size = c.outPorts.size := by rw [hrep, Array.size_replicate]
    refine ⟨⟨hrep, fun j hj => ⟨(Hidden.checkRows_out hc (by rw [hsz]; exact hj)).1, ?_⟩⟩, fun p hm => ?_⟩
    · obtain ⟨-, hs, hb, h1', h2'⟩ := Layout.v2Row_spec (ho j (by rw [hsz]; exact hj))
      rw [Hidden.getElem!_eq_getD']
      exact ⟨hs, hb, h1', h2'⟩
    · rcases h2 p hm with hv | hs
      · obtain ⟨-, hs, hb, h1', h2'⟩ := Layout.v2Row_spec hv
        exact Or.inl ⟨hs, hb, h1', h2'⟩
      · obtain ⟨hs, hb, hn, hf⟩ := Layout.segRow_spec hs
        exact Or.inr ⟨hs, hb, hn, hf⟩

theorem rowsFit_of_rows {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (h : (Layout.v1Ports c || Layout.bitRows c) = true) : Hidden.Any.RowsFitS c :=
  rowsFit_of_rowsPre hp hc h

/-- **`scopeOkH`'s rows are covered** (`rowsFit_of_rows`). -/
theorem rowsFit_of_scope {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ()) (h : Layout.scopeOkH c = true) :
    Hidden.Any.RowsFitS c :=
  rowsFit_of_rows hp hc (Layout.scopeOkH_rowsB h)

theorem v1PortsL_unrolled (c : Circuit) : Layout.v1Ports (HmOut.unrolled c) = Layout.v1Ports c := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem bitRowsL_unrolled (c : Circuit) : Layout.bitRows (HmOut.unrolled c) = Layout.bitRows c := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

/-- Unrolling a circuit keeps its rows: `RowsFitS` reads its ports, never its nets or template. -/
theorem rowsFit_unrolled (c : Circuit) : Hidden.Any.RowsFitS (HmOut.unrolled c) ↔ Hidden.Any.RowsFitS c := by
  unfold HmOut.unrolled; split
  · split <;> exact Iff.rfl
  · exact Iff.rfl

/-- **A committed circuit's rows are covered** (`Hidden.Committed`): a flat file's `HmOut.parse`'s (`rowsFit_of_rows`),
or a templated file's, whose rows are its `HmRow.parse`'s after the read unit (`rowsFit_of_rowsPre`). -/
theorem rowsFit_of_committed {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hty : tags.typed = false) (h : Hidden.Committed tags cf tables c)
    (hr : (Layout.v1Ports c || Layout.bitRows c) = true) : Hidden.Any.RowsFitS c := by
  rcases h with h | h
  · obtain ⟨hp, hc⟩ := Hidden.parse_untyped hty h
    exact rowsFit_of_rows hp hc hr
  · obtain ⟨tu, c₀, hp, hck, -, rfl⟩ := TmplValue.parseTmpl_cases h
    exact rowsFit_of_rowsPre (c := c₀) hp hck hr

/-- **The statement circuit's rows are covered** where `Layout.scopeOkH` passes: `HmOut.parseStmt`'s circuit, a flat
file's `HmOut.parse`'s or a templated file's committed circuit unrolled, whose rows are the committed circuit's
(`rowsFit_unrolled`). -/
theorem rowsFit_of_stmt {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hty : tags.typed = false) (hc : HmOut.parseStmt tags cf tables = .ok c) (h : Layout.scopeOkH c = true) :
    Hidden.Any.RowsFitS c := by
  have hr := Layout.scopeOkH_rowsB h
  rcases parseStmt_cases hc with ⟨-, hc⟩ | ⟨-, ct, hct, rfl⟩
  · exact rowsFit_of_committed hty (Or.inl hc) hr
  · rw [v1PortsL_unrolled, bitRowsL_unrolled] at hr
    exact (rowsFit_unrolled ct).2 (rowsFit_of_committed hty (Or.inr hct) hr)

/-- **`rowsFit_of_rows` with pinned outputs**: at `bitRowsU`, which `bitRows` is without `unpinned`, each output row is
a `v2Row` of one output port, and the output ports past the rows (META `pinned_outputs`') have none (`RowsFitP`). -/
theorem rowsFitP_of_rows {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (h : (Layout.v1Ports c || Layout.bitRowsU c) = true) : Hidden.Any.RowsFitP c := by
  simp only [Bool.or_eq_true] at h
  rcases h with h1 | h2
  · exact (rowsFit_of_rows hp hc (by rw [h1]; rfl)).toP
  · unfold Layout.bitRowsU at h2
    simp only [Bool.and_eq_true, Array.all_eq_true', Bool.or_eq_true, beq_iff_eq, List.all_eq_true,
      List.mem_range] at h2
    obtain ⟨⟨hpr, hr⟩, ho⟩ := h2
    have hrep : c.outRowPorts = Array.replicate c.outRowPorts.size 1 := by
      refine Array.ext (by simp) fun i hi _ => ?_
      have := hr _ (Array.getElem_mem hi)
      simp [this]
    refine ⟨⟨hrep, fun j hj => ⟨(Hidden.checkRows_out hc hj).1, ?_⟩⟩, fun p hm => ?_⟩
    · obtain ⟨-, hs, hb, h1', h2'⟩ := Layout.v2Row_spec (ho j hj)
      rw [Hidden.getElem!_eq_getD']
      exact ⟨hs, hb, h1', h2'⟩
    · rcases hpr p (Array.mem_toList_iff.1 hm) with hv | hs
      · obtain ⟨-, hs, hb, h1', h2'⟩ := Layout.v2Row_spec hv
        exact Or.inl ⟨hs, hb, h1', h2'⟩
      · obtain ⟨hs, hb, hn, hf⟩ := Layout.segRow_spec hs
        exact Or.inr ⟨hs, hb, hn, hf⟩

end FlockSoundness.Discharge.HiddenExec
