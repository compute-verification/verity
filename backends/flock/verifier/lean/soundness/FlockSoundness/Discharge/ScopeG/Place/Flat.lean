import FlockSoundness.Discharge.ScopeW.OutPairs
import FlockSoundness.Discharge.ScopeG.Vu

set_option Elab.async false

/-!
# P1 piece: a hidden statement reads its flat twin's rows at every row of a VU's slots

`vuRowsFlat`: an untyped statement `Stmt.setupHidden` accepted reads its flat twin's (`Hidden.flatTwin`,
Δ = `HmRow.delta`'s) `placedA` and `placedB` at every row of every slot of every net a VU computes (`vuNets`: all but
the row circuits `sha512x3` and `hm96`), the constant's row included.

**Twin of** `ScopeW.unitRowsFlat_any` (`Discharge/ScopeW/OutPairs.lean`), which proves this at the unit's slots only
(`HiddenExec.UnitRowsFlat`, `Discharge/HiddenExec/Flat.lean`).

**Route.** As `unitRowsFlat_any`: the hidden Δ is the flat Δ followed by `Hidden.outPairs` (`Hidden.hidden_delta'`), so
by `Hidden.deltaEntry_append_out` it is enough that no output pair's row is a row of net `k`'s slot. An output pair's row
is a row's message bit (`HiddenExec.outPairs_fst`, `Hidden.deltaMsgCol`), an input bit of a `sha512x3` or `hm96` slot
(`deltaMsgCol_ne_vu`, as `ScopeW.deltaMsgCol_ne_unitW` reads it). That replaces `Layout.ne_unit_at` by its twin at net
`k` (`ne_net_at`): a column inside a slot of the row circuits' ranges is in no slot of net `k ∉ {shaNet, hmNet}`'s range
(`CheckFacts.disj`, `CheckFacts.one_range`, `covers_slot`, the ranges' `Apart`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **A statement reads its flat twin's rows at every row of a VU's slots**: every net of `vuNets`, every VU, every
slot of the VU, every column up to the slot's constant. -/
def VuRowsFlat (st : Stmt) : Prop :=
  ∀ k ∈ vuNets st.c, ∀ g < st.c.g, ∀ i < st.c.perVu.getD k 0, ∀ x ≤ (st.c.nets.getD k default).2.constPos,
    ∀ r j : Fin (2 ^ st.c.kLog), r.val = st.c.slotCol k g i + x →
      placedA st r j = placedA (Hidden.flatTwin st) r j ∧ placedB st r j = placedB (Hidden.flatTwin st) r j

variable {tags : Tags} {st : Stmt}

/-- `vuNets`' members: the nets but the row circuits. -/
theorem mem_vuNets {c : Circuit} {k : ℕ} : k ∈ vuNets c ↔ k < c.nets.size ∧ k ≠ c.shaNet ∧ k ≠ c.hmNet := by
  simp [vuNets, List.mem_filter, List.mem_range]

/-- A net slot's input port bit is no column of another net's slot up to its constant. -/
theorem ne_net_at (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c) {n k : ℕ} (hn : n < st.c.nets.size)
    (hk : k < st.c.nets.size) (hne : n ≠ k) {g i : ℕ} {q : ℕ × ℕ} {tt : ℕ} (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) (hq : q ∈ (portCols (st.c.nets.getD n default).2.inGroups).toList) (htt : tt < q.2)
    {g' i' x : ℕ} (hg' : g' < st.c.g) (hi' : i' < st.c.perVu.getD k 0)
    (hx : x ≤ (st.c.nets.getD k default).2.constPos) :
    st.c.slotCol n g i + q.1 + tt ≠ st.c.slotCol k g' i' + x := by
  intro he
  obtain ⟨hm, hkd, -⟩ := rangeOf_spec (C.one_range n hn)
  obtain ⟨hm', hkd', -⟩ := rangeOf_spec (C.one_range k hk)
  have hs : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := by
    rw [C.count _ hm _ hkd]; exact Rows.mul_add_lt hg hi
  have hs' : g' * st.c.perVu.getD k 0 + i' < (st.c.rangeOf k).count := by
    rw [C.count _ hm' _ hkd']; exact Rows.mul_add_lt hg' hi'
  have hq' := (nets_ok P n).1.2.2 q hq
  have hfit := (L.fit n).2
  have ha : q.1 + tt < 2 ^ (st.c.rangeOf n).slotLog := by rw [L.slotLog _ hm n hkd]; omega
  have ha' : x < 2 ^ (st.c.rangeOf k).slotLog := by
    rw [L.slotLog _ hm' k hkd']; exact lt_of_le_of_lt hx (L.fit k).1
  have h1 : InSpan (st.c.rangeOf n) (st.c.slotCol n g i + q.1 + tt) := by
    rw [Nat.add_assoc]; unfold Circuit.slotCol; exact inSpan_of_covers (covers_slot hs ha)
  have h2 : InSpan (st.c.rangeOf k) (st.c.slotCol n g i + q.1 + tt) := by
    rw [he]; unfold Circuit.slotCol; exact inSpan_of_covers (covers_slot hs' ha')
  have hR := eq_of_inSpan L.disj hm hm' h1 h2
  rw [hR] at hkd
  exact hne (Kind.net.inj (hkd.symm.trans hkd'))

/-- **A row's message bit is no column of a VU net's slot**, at any row shape: it is an input bit of a `sha512x3` or
`hm96` slot (`ScopeW.deltaMsgCol_ne_unitW`'s route, at net `k` by `ne_net_at`). -/
theorem deltaMsgCol_ne_vu (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c) {k : ℕ}
    (hk : k ∈ vuNets st.c) {g p byte t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (ht : t < 8)
    {g' i x : ℕ} (hg' : g' < st.c.g) (hi : i < st.c.perVu.getD k 0)
    (hx : x ≤ (st.c.nets.getD k default).2.constPos) :
    Hidden.deltaMsgCol st.c g p byte t ≠ st.c.slotCol k g' i + x := by
  obtain ⟨hkn, hks, hkh⟩ := mem_vuNets.1 hk
  have ho : (HmRow.byteAt (st.c.ports.getD p default) byte).2 < 128 := by
    rw [Typed.byteAt_snd]
    exact Nat.mod_lt _ (by norm_num)
  by_cases hk' : (HmRow.byteAt (st.c.ports.getD p default) byte).1 < ((HmRow.rowSlots st.c.ports).2[p]!).size
  · simp only [Hidden.deltaMsgCol, hk', ↓reduceIte]
    have hmem : (HmRow.rowSlots st.c.ports).2[p]! ∈ (HmRow.rowSlots st.c.ports).2.toList :=
      getElem_mem_toList' (by rw [rowSlots_spec]; exact hp)
    obtain ⟨h1, h2⟩ := rowSlots_all _ _ hmem _ (getElem_mem_toList' hk')
    have hidx : 2 * (((HmRow.rowSlots st.c.ports).2[p]!)[(HmRow.byteAt (st.c.ports.getD p default) byte).1]!).2 + 1 <
        6 := by omega
    have hw := C.sha_in.2 _ hidx
    rw [sha_width _ _ (by norm_num : 1 < 2)] at hw
    have hne := ne_net_at L P C P.shaNet hkn (Ne.symm hks) hg (by rw [C.perVu_sha]; exact h1)
      (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx))
      (tt := 64 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 / 8) + 56 -
        8 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 % 8) + t) (by rw [hw]; omega) hg' hi hx
    intro he
    refine hne (Eq.trans ?_ he)
    omega
  · simp only [Hidden.deltaMsgCol, hk', ↓reduceIte]
    obtain ⟨hil, hw⟩ := hm_width C.hm_in (by norm_num : 1 < 2)
    have hne := ne_net_at L P C P.hmNet hkn (Ne.symm hkh) hg
      (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)) (getElem_mem_toList' hil)
      (tt := 64 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 / 8) + 56 -
        8 * ((HmRow.byteAt (st.c.ports.getD p default) byte).2 % 8) + t) (by rw [hw]; omega) hg' hi hx
    intro he
    refine hne (Eq.trans ?_ he)
    omega

/-- **No output pair of Δ is at a column of a VU net's slot**, at any row shape. -/
theorem outPairs_ne_vu (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hle : st.c.outRowPorts.size ≤ st.c.ports.size) {q : ℕ × ℕ} (hq : q ∈ Hidden.outPairs st.c) {k : ℕ}
    (hk : k ∈ vuNets st.c) {g' i x : ℕ} (hg' : g' < st.c.g) (hi : i < st.c.perVu.getD k 0)
    (hx : x ≤ (st.c.nets.getD k default).2.constPos) : q.1 ≠ st.c.slotCol k g' i + x := by
  obtain ⟨g, hg, j, hj, b, -, t, -, hq1⟩ := HiddenExec.outPairs_fst hq
  rw [hq1]
  exact deltaMsgCol_ne_vu L P C hk hg (by unfold HmOut.nIn; omega) (Nat.mod_lt _ (by norm_num)) hg' hi hx

variable {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {draw partition : Option Json}
  {program : Option ByteArray}

/-- **An untyped hidden statement reads its flat twin's rows at every row of a VU's slots** (`ScopeW.unitRowsFlat_any`
at every net of `vuNets`). -/
theorem vuRowsFlat (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : VuRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨-, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := HiddenExec.flat_layout ht h
  have F := HiddenExec.flat_unitFacts ht h
  intro k hk g hg i hi x hx r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_vu L F.parse F.check (HiddenExec.checkRows_le hcr) hq hk hg hi hx
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by
    rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

end FlockSoundness.Discharge.ScopeG
