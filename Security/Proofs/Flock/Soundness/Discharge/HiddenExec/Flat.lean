import Proofs.Flock.Soundness.Discharge.Hidden.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.FitS
import Proofs.Flock.Soundness.Discharge.Layout.Scope

/-!
# A hidden-output statement's flat twin

`Stmt.setupHidden`'s Δ is `HmRow.delta`'s, then `Hidden.outPairs` (`Hidden.HiddenDelta`): each output row's message
bits, copied from the output bits the circuit computes. Its flat twin `Hidden.flatTwin st` is the statement with
`HmRow.delta`'s Δ alone, the one `Stmt.setupH` would have built on the same circuit. It has `layout_of`'s layout
(`flat_layout`) and the unit facts the copy positions read (`flat_unitFacts`), as setupH's statement does.

Every first column of `outPairs` is a message bit of a row's `sha512x3` slot (`deltaMsgCol`), which no unit slot's row
is (`outPairs_ne_unit`, for output rows that fit, `Hidden.Any.OutRowsS`). So at every unit slot's row the hidden
statement's level-0 matrices are its flat twin's (`unitRowsFlat`): the rows that place a template instance and copy its
inputs read the same in both. `Layout.scopeOkH` gives rows each fit or `sha512/row-seg/v1`, the output rows fit
(`rowsFit_of_scope`).
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

/-- **The flat twin of an untyped hidden statement has the layout the placement reads** (`layout_of`). -/
theorem flat_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (Hidden.flatTwin st) :=
  have s := Hidden.setupHidden_spec' h
  layout_of (st := Hidden.flatTwin st) (parse_facts (Hidden.parse_untyped ht s.parse).1) s.pin rfl rfl s.m s.kLog

/-- **The flat twin's unit facts** (`Hidden.unitFacts_of_parse`). -/
theorem flat_unitFacts (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) :=
  Hidden.unitFacts_of_parse (st := Hidden.flatTwin st) (Hidden.parse_untyped ht (Hidden.setupHidden_spec' h).parse).1 rfl

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

/-- **No output pair of Δ is at a unit slot's row**, at rows that fit: every output row's data bit is an input bit of
its compression's `sha512x3` slot or of its `hm96` slot (`Hidden.Any.msg_at`), another net's slot. -/
theorem outPairs_ne_unit (L : Layout st) (F : Aliased.UnitFacts tags st) (hrows : Hidden.Any.OutRowsS st.c)
    {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  rw [hrows.size] at hj
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

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, at rows that fit, its Δ being
`HmRow.delta`'s, then `outPairs` ([wave2-hidden]'s `Hidden.hidden_delta'`). -/
theorem unitRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hrows : Hidden.Any.OutRowsS st.c) : UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F hrows hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

/-! ## The rows `scopeOkH` covers -/

/-- **`scopeOkH`'s rows are covered**: on a circuit `HmOut.parse` accepts (`HmRow.parse`, then `HmOut.checkRows`), all
`sha512/row/v1` rows (whole blocks, `ParseFacts.portWords`; one output row per output port, `Hidden.out_rows_v1`), or
each row a `v2Row` or a `segRow`, each output row a `v2Row` of one output port. -/
theorem rowsFit_of_scope {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ()) (h : Layout.scopeOkH c = true) :
    Hidden.Any.RowsFitS c := by
  have P := parse_facts hp
  rcases Layout.scopeOkH_rows h with h1 | ⟨h2, hr, ho⟩
  · have hv : ∀ p ∈ c.ports.toList, p.v2 = false := fun p hm => (h1 p hm).1
    obtain ⟨hu, hrep, hj⟩ := Hidden.out_rows_v1 hp hc hv
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
  · have hrep := Hidden.outRows_single (Hidden.parse_rowPorts hp).2 fun j hj => by
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

end FlockSoundness.Discharge.HiddenExec
