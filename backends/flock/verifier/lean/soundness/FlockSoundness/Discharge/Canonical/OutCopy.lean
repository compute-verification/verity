import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Hidden.Wide.RowBc
import FlockSoundness.Discharge.HiddenExec.SiteH
import FlockSoundness.Protocol.Discharge.Canonical.OutCopy

set_option Elab.async false

/-!
# Output rows of runs of output ports, exact length: Δ's copy into a data bit

`Hidden.msg_copyAll` and `HiddenExec.out_copyH` at canonical output rows: one per run of output ports
(`HmOut.outGroup`), each a `sha512/row/v2` row of exactly its run's `16 · words` bits. Such a row's data may end inside
its last block, the `hm96` slot's (`chunks = ⌈(2 · words + 17) / 128⌉ - 1`), so a data byte `b` is in compression
`b / 128` or, at `b / 128 = chunks`, in the `hm96` slot's `pad` (`deltaMsgCol_hm`). Either way:

* no copy of `HmRow.delta` is at a data bit (`msg_filterC`): `HmRow.constBit` is `none` below the row's length
  (`constBit_msgOff`), so neither the compressions' constant bits nor the `pad`'s write there;
* `outPairs` copies into it once, from `outSrc` (`outPairsC_count`): distinct data bits are distinct columns
  (`deltaMsgCol_injC`);
* so a satisfying witness holds `outSrc`'s bit there (`msg_copyC`), and a returned output's column holds its run's row
  bit (`out_copyHC`).

A tail that returns the outputs leaves the units no returned output (`check_tail`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Canonical

open FlockSoundness.Discharge.Hidden FlockSoundness.Discharge.Hm Compose Flock.DeriveAll Types.Dag

/-! ## A data bit's column -/

section Cols

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

theorem deltaMsgCol_ltC {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht : t < 8) :
    deltaMsgCol c g p b t < 2 ^ c.kLog := by
  rcases Nat.lt_or_ge (b / 128) c.ports[p]!.chunks with hk | hk
  · have hlt : b < 128 * c.ports[p]!.chunks := by omega
    exact deltaMsgCol_lt R hu hg hp hlt ht
  · exact (deltaMsgCol_atHm R hu hg hp hk ht).1.lt R.bound (R.range R.hmNet).1

end Cols

/-! ## Δ's pairs at an output row's data bit -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **The flat Δ's row at a data bit is a self row**, in `A` and in `B` (`flat_selfAll` at any row length). -/
theorem flat_selfC (hda : st.da = (HmRow.delta st.c st.pin).1) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hb : b / 128 ≤ st.c.ports[p]!.chunks)
    (hd : 8 * b + t < st.c.ports[p]!.bits) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  have hfc := flat_count_msgC L F F.parse.rowWires hu hg hp hb hd ht
  have hδ : ∀ j : ℕ, deltaEntry st.da (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) ∧
      deltaEntry st.db (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) := by
    intro j
    have e : deltaEntry st.da (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) := by
      rw [deltaEntry_eq_count, hda, delta_flat_all _ _ F.parse.typed hu, hfc]
      split_ifs <;> rfl
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  rcases Nat.lt_or_ge (b / 128) st.c.ports[p]!.chunks with hk | hk
  · have hlt : b < 128 * st.c.ports[p]!.chunks := by omega
    obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hlt ht
    exact inBit_rows R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ j
  · obtain ⟨hat, hoff⟩ := deltaMsgCol_atHm R hu hg hp hk ht
    exact inBit_rows R (Or.inr rfl) hat (hmNet_ofRows R.rowNets) (hmLaid_self hoff) hx hδ j

end Twin

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRowsC c)
include R hrows

end Out

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st))
  (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsC st.c)
include L L0 F0 hda hrows

/-- **A satisfying witness of the hidden statement satisfies its flat twin**, at canonical output rows
(`sat_flatAll`): the rows agree past `outPairs`' data bits, and there the twin's row is a self row (`flat_selfC`). -/
theorem sat_flatC {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i => ?_, hz.2.1, hz.2.2⟩
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    obtain ⟨hp, hb, hd, ht8⟩ := outRowC_bound hrows hj hk ht
    have hx := deltaMsgCol_ltC R hrows.1 (b := 2 * k + t / 8) hg hp ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfC L0 F0 rfl hrows.1 hg hp hb hd ht8 hx i'
      have e : (if i'.val = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then (1 : ZMod 2) else 0) =
          if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0 := by
        simp [Fin.ext_iff]
      exact ⟨hs.1.trans e, hs.2.trans e⟩
    rw [mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').1),
      mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').2)]
    have hsq : ∀ a : ZMod 2, a * a = a := by decide
    exact hsq _
  · push Not at hi
    have hdb : st.db = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := L.delta_b.trans hda
    have hA : ∀ i', (L0.model regs).A₀ i i' = (L.model regs).A₀ i i' := fun i' => by
      show rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).1 i.val i'.val =
        rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry st.da i.val i'.val
      rw [hda, deltaEntry_append_out hi]
    have hB : ∀ i', (L0.model regs).B₀ i i' = (L.model regs).B₀ i i' := fun i' => by
      show rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).2 i.val i'.val =
        rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry st.db i.val i'.val
      rw [hdb, deltaEntry_append_out hi, delta_snd _ _ F0.parse.typed]
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

end Hidden

/-! ## At an accepted statement -/

end FlockSoundness.Discharge.Canonical
