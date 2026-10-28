import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFit
import Proofs.Flock.Soundness.Discharge.Hidden.PortsP
import Proofs.Flock.Soundness.Discharge.Hm.Flat

/-!
# Unsegmented output rows at public inputs

`Wide.SegFit`'s `Any.msg_copyF` and `Any.sat_flatF` at a statement whose Δ is `HmOut.delta`'s without the pairs into a
public leaf's bits (`HmIn.delta`). Every row it drops is a public leaf's bit, inside a unit slot's input rows
(`unitIn_of_drop`), so off those rows (`UnitIn`) its Δ is `HmOut.delta`'s (`da_offUnit`), and:

* `Any.msg_copyFP`: an output row's message bit holds its copy in every block of a witness that satisfies the statement
  off the unit input rows (`SatEx`);
* `Any.sat_flatFP`: a satisfying witness of the statement satisfies its flat twin off the unit input rows. At a public
  leaf the twin copies the zero, which the statement leaves out.
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- **A row `HmIn.delta` drops is a unit input row**: a public leaf's bit (`publicCol_at`). -/
theorem unitIn_of_drop {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {pi : HmIn.Ports}
    (hp : HmIn.parse st.c = .ok pi) {r : ℕ} (hk : keepCol st.c pi r = false) : UnitIn st.c r := by
  simp only [keepCol, Bool.or_eq_false_iff] at hk
  have hq : HmIn.publicCol st.c pi r = true := by simpa using hk.2
  have hgr := (built_spec L.unit_built).2.2.2.2.1
  rw [L.unit_at.groups, L.unit_at.in_eq] at hgr
  obtain ⟨k, hk', jj, -, t, -, hr, hlt, -⟩ := publicCol_at hp hk.1 F.check hgr hq
  have hz : st.c.unit.useful < 2 ^ st.c.unitRange.slotLog := by rw [L.unit_slotLog]; exact F.check.unit_zero
  have hz' : st.c.unit.useful < 2 ^ (st.c.rangeOf st.c.unitNet).slotLog := hz
  have h1 := F.useful
  have h2 := L.unit_at.in_le
  rw [L.unit_at.in_eq] at h2
  have h3 := L.unit_at.const_le
  exact ⟨k, _, ⟨hk', by omega, hr⟩, hlt⟩

/-- **Off the unit input rows, `HmIn.delta`'s level-0 entries are `HmOut.delta`'s**: `HmRow.delta`'s, then `outPairs`, where
no output is pinned (`hidden_delta_unpinned`). -/
theorem da_offUnit {tags : Tags} {st : Stmt} (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
    {pi : HmIn.Ports} (hp : HmIn.parse st.c = .ok pi) (hda : st.da = (HmIn.delta st.c pi st.pin).1)
    (hpin : HmRow.pinnedCount st.c.metaJ = 0) (r : ℕ) (hr : ¬ UnitIn st.c r) (i : ℕ) :
    deltaEntry st.da r i = deltaEntry ((HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) r i := by
  have hk : keepCol st.c pi r = true := by
    by_contra hn
    exact hr (unitIn_of_drop L0 F0 hp (Bool.eq_false_iff.2 hn))
  rw [hda, (deltaEntry_kept hk i).1, hidden_delta_unpinned hpin]

namespace Any

section HiddenP

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : ∀ r, ¬ UnitIn st.c r → ∀ i,
    deltaEntry st.da r i = deltaEntry ((HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) r i)
  (hrows : OutRowsS st.c)
include L L0 F0 hda hrows

/-- **An output row's message bit holds its copy** in every block of a witness that satisfies the hidden statement off
the unit input rows, with Δ `HmOut.delta`'s there (`msg_copyF`): the bit is a `sha512x3` or `hm96` slot's row. -/
theorem msg_copyFP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) (UnitIn st.c) z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
  obtain ⟨n, s, off, hn, hat, -, l, name, hl, hself⟩ := msg_at R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hx := deltaMsgCol_ltF R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hU : ¬ UnitIn st.c (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) :=
    not_unitIn_of_rowNet L0 F0 hn hat
  have hfc := flat_count_msgF L0 F0 hg hp (hrows.2 j hj).2 hb ht8
  have hδ : ∀ i : ℕ, deltaEntry st.da (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) := by
    intro i
    have e : deltaEntry st.da (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) := by
      rw [hda _ hU i, deltaEntry_eq_count, Array.toList_append, List.count_append, List.toList_toArray,
        delta_flat_any _ _ F0.parse.typed, hfc, Nat.zero_add, outPairs_countF R hrows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact block_of_rows_ex hz o hx hU fun i => inBit_rows (st := st) R hn hat hl hself hx hδ i

/-- **A satisfying witness of the hidden statement satisfies its flat twin off the unit input rows**, with Δ
`HmOut.delta`'s there (`sat_flatF`): past `outPairs`' message bits the rows agree, and there the twin's row is a self
row. -/
theorem sat_flatFP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : SatEx (L0.model regs) (UnitIn st.c) z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i hi0 => ?_, hz.2.1, hz.2.2⟩
  unfold RowHolds
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    rw [hrows.size] at hj
    obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
    have hx := deltaMsgCol_ltF R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfF L0 F0 rfl hg hp (hrows.2 j hj).2 hb ht8 hx i'
      have e : (if i'.val = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then (1 : ZMod 2) else 0) =
          if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0 := by
        simp [Fin.ext_iff]
      exact ⟨hs.1.trans e, hs.2.trans e⟩
    rw [mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').1),
      mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').2)]
    have hsq : ∀ a : ZMod 2, a * a = a := by decide
    exact hsq _
  · push Not at hi
    have hA : ∀ i', (L0.model regs).A₀ i i' = (L.model regs).A₀ i i' := fun i' => by
      show rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).1 i.val i'.val =
        rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry st.da i.val i'.val
      rw [hda _ hi0, deltaEntry_append_out hi]
    have hB : ∀ i', (L0.model regs).B₀ i i' = (L.model regs).B₀ i i' := fun i' => by
      show rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).2 i.val i'.val =
        rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry st.db i.val i'.val
      rw [L.delta_b, hda _ hi0, deltaEntry_append_out hi, delta_snd _ _ F0.parse.typed]
    have eA : (L0.model regs).A₀.mulVec ((L0.model regs).block z o) i =
        (L.model regs).A₀.mulVec ((L.model regs).block z o) i := by
      simp only [Matrix.mulVec, dotProduct, hA]
      rfl
    have eB : (L0.model regs).B₀.mulVec ((L0.model regs).block z o) i =
        (L.model regs).B₀.mulVec ((L.model regs).block z o) i := by
      simp only [Matrix.mulVec, dotProduct, hB]
      rfl
    rw [eA, eB]
    exact hz.1 o i

end HiddenP

end Any

end FlockSoundness.Discharge.Hidden
