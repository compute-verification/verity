import FlockSoundness.Discharge.HiddenExec.Flat

/-!
# Δ's output pairs are no unit slot's row, at every row shape

`HiddenExec.outPairs_ne_unit` reads `sha512/row/v1` rows to place each output row's byte inside one of its compressions.
No row shape is needed: `Hidden.deltaMsgCol` puts a row's byte in a `sha512x3` slot's message input when the byte's
compression (`HmRow.byteAt`, segment by segment for a segmented row) is one of the row's, and past them in its `hm96`
slot's `pad` input (a short `sha512/row/v2` row's one block). Either is an input bit of a slot of another net than the
unit's (`Layout.ne_unit_at`), so no unit slot's row (`deltaMsgCol_ne_unitW`, `outPairs_ne_unitW`). So an untyped hidden
statement reads its flat twin's rows at every unit slot's row whatever its rows' shapes (`unitRowsFlat_any`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {st : Stmt}

/-- **A row's message bit is no unit slot's row**, at any row shape: in its byte's compression's `sha512x3` slot
(`HmRow.byteAt`, a segmented row's included), or past the row's compressions in its `hm96` slot's `pad`. -/
theorem deltaMsgCol_ne_unitW (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet) {g p byte t : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (ht : t < 8) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (hb : b < 2 ^ st.c.unit.unitLog) : Hidden.deltaMsgCol st.c g p byte t ≠ st.c.unitRange.slot s₀ + b := by
  have ho : (HmRow.byteAt (st.c.ports.getD p default) byte).2 < 128 := by
    rw [Typed.byteAt_snd]
    exact Nat.mod_lt _ (by norm_num)
  by_cases hk : (HmRow.byteAt (st.c.ports.getD p default) byte).1 < ((HmRow.rowSlots st.c.ports).2[p]!).size
  · simp only [Hidden.deltaMsgCol, hk, ↓reduceIte]
    have hmem : (HmRow.rowSlots st.c.ports).2[p]! ∈ (HmRow.rowSlots st.c.ports).2.toList :=
      getElem_mem_toList' (by rw [rowSlots_spec]; exact hp)
    obtain ⟨h1, h2⟩ := rowSlots_all _ _ hmem _ (getElem_mem_toList' hk)
    have hidx : 2 * (((HmRow.rowSlots st.c.ports).2[p]!)[(HmRow.byteAt (st.c.ports.getD p default) byte).1]!).2 + 1 <
        6 := by omega
    have hw := C.sha_in.2 _ hidx
    rw [sha_width _ _ (by norm_num : 1 < 2)] at hw
    have hne := L.ne_unit_at P C P.shaNet hsha.symm hg (by rw [C.perVu_sha]; exact h1)
      (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx))
      (tt := 64 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 / 8) + 56 -
        8 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 % 8) + t) (by rw [hw]; omega) hs₀ hb
    intro he
    refine hne (Eq.trans ?_ he)
    omega
  · simp only [Hidden.deltaMsgCol, hk, ↓reduceIte]
    obtain ⟨hil, hw⟩ := hm_width C.hm_in (by norm_num : 1 < 2)
    have hne := L.ne_unit_at P C P.hmNet hhm.symm hg
      (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)) (getElem_mem_toList' hil)
      (tt := 64 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 / 8) + 56 -
        8 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 % 8) + t) (by rw [hw]; omega) hs₀ hb
    intro he
    refine hne (Eq.trans ?_ he)
    omega

/-- **No output pair of Δ is at a unit slot's row**, at any row shape (`deltaMsgCol_ne_unitW` at each output row's
message bit). -/
theorem outPairs_ne_unitW (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet)
    (hle : st.c.outRowPorts.size ≤ st.c.ports.size) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) : q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, -, t, -, hq1⟩ := HiddenExec.outPairs_fst hq
  rw [hq1]
  exact deltaMsgCol_ne_unitW L P C hsha hhm hg (by unfold HmOut.nIn; omega) (Nat.mod_lt _ (by norm_num)) hs₀ hb

variable {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {draw partition : Option Json}
  {program : Option ByteArray}

/-- **An untyped hidden statement reads its flat twin's rows at every unit slot's row**, at any row shape:
`HiddenExec.unitRowsFlat` with `outPairs_ne_unitW`. -/
theorem unitRowsFlat_any (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : HiddenExec.UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨-, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := HiddenExec.flat_layout ht h
  have F := HiddenExec.flat_unitFacts ht h
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unitW L F.parse F.check F.sha F.hm (HiddenExec.checkRows_le hcr) hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

end FlockSoundness.Discharge.ScopeW
