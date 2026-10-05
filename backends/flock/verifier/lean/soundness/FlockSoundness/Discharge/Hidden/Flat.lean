import FlockSoundness.Discharge.Hidden.Layout
import FlockSoundness.Protocol.Discharge.Hidden.Flat

/-!
# A hidden-output statement's flat twin, and its output rows' message bits

`flatTwin st` is the statement with `HmRow.delta`'s Δ alone, the one `Stmt.setupH` would build on the same circuit; it
has the unit facts `Hm.row_bc_flat` reads (`unitFacts_of_parse`). An output row's message bit (`deltaMsgCol`) is an input
bit of a `sha512x3` slot past the compression's `cv` (`deltaMsgCol_at`), where no copy of `HmRow.delta` is (`msg_filter`),
so its flat Δ count is zero (`flat_count_msg`), and `outPairs` copies into it once (`outPairs_count`): the hidden
statement's row there is a copy of `outSrc` (`msg_copy`), its flat twin's a self row (`flat_self`). So a satisfying
witness of the hidden statement satisfies its flat twin (`sat_flat`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-! ## An output row's message bits -/

section Cols

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

end Cols

/-! ## `HmRow.delta`'s Δ at a message bit: nothing -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **The flat Δ's row at a row's message bit is a self row**, in `A` and in `B`. -/
theorem flat_self (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  have hu : ∀ p ∈ st.c.ports.toList, p.segs = none := fun p hp => F.parse.v1Segs p hp (hv1 p hp).1
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hb ht
  have hfc := flat_count_msg L F F.parse.rowWires hu hpos hg hp hb ht
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
      rw [deltaEntry_eq_count, hda, delta_flat_eq _ _ F.parse.typed hv1 F.parse.portWords F.parse.v1Segs, hfc]
      split_ifs <;> rfl
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact inBit_rows R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ j

end Twin

/-! ## `outPairs` at a message bit: one copy -/

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRows c)
include R hrows

end Out

/-! ## The hidden statement and its flat twin -/

/-- Δ's count at a row with no output pair is `HmRow.delta`'s. -/
theorem deltaEntry_append_out {d : Array (ℕ × ℕ)} {c : Circuit} {r : ℕ} (hr : ∀ q ∈ outPairs c, q.1 ≠ r) (j : ℕ) :
    deltaEntry (d ++ (outPairs c).toArray) r j = deltaEntry d r j := by
  rw [deltaEntry_eq_count, deltaEntry_eq_count, Array.toList_append, List.count_append, List.toList_toArray,
    List.count_eq_zero.2 fun hm => hr _ hm rfl, Nat.add_zero]

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRows st.c)
include L L0 F0 hv1 hda hrows

/-- **A satisfying witness of the hidden statement satisfies its flat twin**: the rows agree past `outPairs`' message
bits (`deltaEntry_append_out`), and there the twin's row is a self row (`flat_self`), which every bit satisfies. -/
theorem sat_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i => ?_, hz.2.1, hz.2.2⟩
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    rw [hrows.size] at hj
    obtain ⟨hp, hb, ht8⟩ := outRow_bound hrows hj hk ht
    have hx := deltaMsgCol_lt R hrows.1 hg hp hb ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_self L0 F0 hv1 rfl hg hp hb ht8 hx i'
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

end FlockSoundness.Discharge.Hidden
