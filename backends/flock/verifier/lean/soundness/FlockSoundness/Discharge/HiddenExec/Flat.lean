import FlockSoundness.Discharge.Hidden.Flat

/-!
# A hidden-output statement's flat twin

`Stmt.setupHidden`'s Δ is `HmRow.delta`'s, then `Hidden.outPairs` (`Hidden.HiddenDelta`): each output row's message
bits, copied from the output bits the circuit computes. Its flat twin `Hidden.flatTwin st` is the statement with
`HmRow.delta`'s Δ alone, the one `Stmt.setupH` would have built on the same circuit. It has `layout_of`'s layout
(`flat_layout`) and the unit facts the copy positions read (`flat_unitFacts`), as setupH's statement does.

Every first column of `outPairs` is a message bit of a row's `sha512x3` slot (`deltaMsgCol`), which no unit slot's row
is (`outPairs_ne_unit`, for `sha512/row/v1` rows). So at every unit slot's row the hidden statement's level-0 matrices
are its flat twin's (`unitRowsFlat`): the rows that place a VU and copy its inputs read the same in both.
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

/-- The first column of each of Δ's output pairs is `deltaMsgCol` at a VU, output row, word and bit. -/
theorem outPairs_fst {c : Circuit} {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs c) :
    ∃ g < c.g, ∃ j < c.outRowPorts.size, ∃ k < (c.ports.getD (HmOut.nIn c + j) default).words, ∃ t < 16,
      q.1 = Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) := by
  simp only [Hidden.outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
  exact ⟨g, hg, j, hj, k, hk, t, ht, by rcases hq with rfl | rfl <;> rfl⟩

/-- **A row's message bit at its `sha512x3` slot is no unit slot's row**: compression `byte / 128` of row `p` is one of
the VU's `sha512x3` slots, its message input 1024 bits wide, and the bit is inside it. -/
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

/-- **No output pair of Δ is at a unit slot's row**, for `sha512/row/v1` rows: every output row's bytes are within its
compressions (`2 · words = 128 · chunks`). -/
theorem outPairs_ne_unit (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false)
    (hle : st.c.outRowPorts.size ≤ st.c.ports.size) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  rw [hq1]
  have hp : HmOut.nIn st.c + j < st.c.ports.size := by unfold HmOut.nIn; omega
  have hw := P.portWords _ (getElem_mem_toList' hp) (hv1 _ (getElem_mem_toList' hp))
  have e : st.c.ports.getD (HmOut.nIn st.c + j) default = st.c.ports[HmOut.nIn st.c + j]! := by
    rw [getElem!_pos st.c.ports _ hp, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp, Option.getD_some]
  have hs : (st.c.ports.getD (HmOut.nIn st.c + j) default).segs = none := by
    rw [e]; exact P.v1Segs _ (getElem_mem_toList' hp) (hv1 _ (getElem_mem_toList' hp))
  rw [e] at hk
  exact deltaMsgCol_ne_unit L P C hsha hg hp hs (by omega) (Nat.mod_lt _ (by norm_num)) hs₀ hb

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, for `sha512/row/v1` rows, when
its Δ being `HmRow.delta`'s, then `outPairs` ([wave2-hidden]'s `Hidden.hidden_delta'`). -/
theorem unitRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false) : UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F.parse F.check F.sha hv1 (checkRows_le hcr) hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

end FlockSoundness.Discharge.HiddenExec
