import Proofs.Flock.Soundness.Discharge.Composed.Base
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.LinkSound
import Proofs.Flock.Soundness.Refine.Setup

/-!
# Every statement `verify --zk` sets up avoids the mask slot

`Zk.shapeOk` refuses a statement whose Δ or regions meet the mask slot (`Zk.maskFree`, D12), so `avoidsMask` holds at
every statement it passes (`avoidsMask_of_shapeOk`), and so at every `--zk` setup (`DrawSetupZK.avoidsMask`,
`DrawSetupZHJ.avoidsMask`): the zero-knowledge views take it from the setup, not as a hypothesis.

`maskFree` checks each region against the mask range's words in one block; `regionWordsOff` checks it against them in every
block, which differ only above bit `k_log − 7` of the word index, where the check reads no bit. `regionCellsOff` follows
from the words: a region cell in the mask range has its word there, and the cell's fixed in-block bits are the region's.
-/

namespace FlockSoundness.Discharge.FailClosed

open Flock Refine Composed Exec

/-- Below bit `K`, adding a multiple of `2^K` changes no bit. -/
theorem testBit_mul_two_pow_add {b y K i : ℕ} (hi : i < K) : (b * 2 ^ K + y).testBit i = y.testBit i := by
  have hm : ∀ x : ℕ, x.testBit i = (x % 2 ^ K).testBit i := fun x => by
    rw [Nat.testBit_mod_two_pow]; simp [hi]
  rw [hm, hm y, Nat.add_comm, Nat.add_mul_mod_self_right]

/-- `Zk.shapeOk`'s mask range, and its `maskFree`. -/
theorem maskFree_of_shapeOk {st : Stmt} (h : Zk.shapeOk st = .ok ()) :
    ∃ r, st.c.ranges.find? (·.kind == Kind.mask) = some r ∧ Zk.maskFree st r = true := by
  unfold Zk.shapeOk at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases hr : st.c.ranges.find? (·.kind == Kind.mask) with
  | none => simp only [hr] at h; cases h
  | some r =>
    simp only [hr] at h
    refine ⟨r, rfl, ?_⟩
    have h := (Refine.ite_throw_ok'.1 (Refine.ite_throw_ok.1 h).2).1
    simpa using h

/-- **`maskFree` at the mask range is `avoidsMask`'s three checks.** -/
theorem avoidsMask_of_maskFree {st : Stmt} {r : Range} (hr : st.c.ranges.find? (·.kind == Kind.mask) = some r)
    (hw : RegionsWF st) (h : Zk.maskFree st r = true) : avoidsMask st = true := by
  have hfind : st.c.ranges.toList.find? (fun r => r.kind == Kind.mask) = some r := by
    rw [Array.find?_toList]; exact hr
  have hcol : ∀ i, maskColB st i =
      decide (r.start / 128 ≤ i / 128 ∧ i / 128 < r.start / 128 + 2 ^ (r.slotLog - 7)) := by
    intro i; unfold maskColB; rw [hfind]
  have hwords : maskWordsOf st = (List.range (2 ^ st.nbl)).flatMap fun b =>
      (List.range (2 ^ (r.slotLog - 7))).map fun j => b * 2 ^ (st.c.kLog - 7) + r.start / 128 + j := by
    unfold maskWordsOf; rw [hfind]
  unfold Zk.maskFree at h
  simp only [Bool.and_eq_true] at h
  obtain ⟨⟨hda, hdb⟩, hreg⟩ := h
  rw [Array.all_eq_true'] at hda hdb hreg
  -- each region, at each of the mask range's words in one block, fixes a word bit the word lacks
  have hW : ∀ reg ∈ st.regions.toList, ∀ j < 2 ^ (r.slotLog - 7), ∃ i < st.c.kLog - 7,
      reg.free.contains (i + 7) = false ∧
        (((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1) =
          !(r.start / 128 + j).testBit i) := by
    intro reg hreg' j hj
    have := hreg reg (Array.mem_toList_iff.mp hreg')
    rw [List.all_eq_true] at this
    obtain ⟨i, hi, hx⟩ := List.any_eq_true.mp (this j (List.mem_range.mpr hj))
    refine ⟨i, List.mem_range.mp hi, ?_⟩
    revert hx
    cases reg.free.contains (i + 7) <;> cases (r.start / 128 + j).testBit i <;>
      cases ((reg.mask >>> (i + 7)) % 2 == 1 && (reg.fixed >>> (i + 7)) % 2 == 1) <;> simp
  unfold avoidsMask deltaAvoidsMask regionsAvoidMask regionCellsOff regionWordsOff regionsOf
  simp only [Bool.and_eq_true]
  refine ⟨?_, ?_, ?_⟩
  · rw [List.all_eq_true]
    intro p hp
    rw [hcol, hcol]
    rcases List.mem_append.mp hp with hp | hp
    · simpa [Bool.decide_and] using hda p (Array.mem_toList_iff.mp hp)
    · simpa [Bool.decide_and] using hdb p (Array.mem_toList_iff.mp hp)
  · rw [List.all_eq_true]
    intro R hR
    obtain ⟨reg, hreg', rfl⟩ := List.mem_map.mp hR
    rw [List.all_eq_true]
    intro i _
    rw [hcol]
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not, not_and, not_lt]
    intro hlo
    by_contra hhi
    rw [Nat.not_le] at hhi
    have hj : (regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i / 128 - r.start / 128 < 2 ^ (r.slotLog - 7) := by
      omega
    obtain ⟨i', hi', hfree, hfix⟩ := hW reg hreg' _ hj
    rw [Nat.add_sub_cancel' hlo] at hfix
    have hk : i' + 7 < st.c.kLog := by omega
    have hnd : (regionOf st.c.kLog (st.m - st.c.kLog) reg).free.Nodup :=
      filterMap_fin_nodup _ _ (hw.regions reg hreg').nodup
    have hnone : (regionOf st.c.kLog (st.m - st.c.kLog) reg).free.idxOf? ⟨i' + 7, hk⟩ = none := by
      rw [List.idxOf?_eq_none_iff]
      intro hm
      simp only [regionOf, List.mem_filterMap] at hm
      obtain ⟨b, hb, hbeq⟩ := hm
      split at hbeq
      · cases hbeq
        have : reg.free.contains (i' + 7) = true := Array.contains_iff_mem.mpr (Array.mem_toList_iff.mp hb)
        rw [hfree] at this; cases this
      · cases hbeq
    have hbit := testBit_inBlock (regionOf st.c.kLog (st.m - st.c.kLog) reg) hnd i ⟨i' + 7, hk⟩
    rw [hnone] at hbit
    have hdiv : ((regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i / 128).testBit i' =
        ((regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i).testBit (i' + 7) :=
      Nat.testBit_div_two_pow (n := 7) _ _
    rw [hdiv] at hfix
    have hb' : ((regionOf st.c.kLog (st.m - st.c.kLog) reg).inBlock i).testBit (i' + 7) =
        ((reg.mask >>> (i' + 7)) % 2 == 1 && (reg.fixed >>> (i' + 7)) % 2 == 1) := hbit
    rw [hb'] at hfix
    revert hfix
    cases ((reg.mask >>> (i' + 7)) % 2 == 1 && (reg.fixed >>> (i' + 7)) % 2 == 1) <;> simp
  · rw [hwords, List.all_eq_true]
    intro reg hreg'
    rw [List.all_eq_true]
    intro x hx
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx
    obtain ⟨b, -, j, hj, rfl⟩ := hx
    obtain ⟨i, hi, hfree, hfix⟩ := hW reg hreg' j hj
    rw [List.any_eq_true]
    refine ⟨i, List.mem_range.mpr hi, ?_⟩
    rw [Nat.add_assoc, testBit_mul_two_pow_add hi, hfree, hfix]
    cases (r.start / 128 + j).testBit i <;> rfl

/-- **Every statement `Zk.shapeOk` passes avoids the mask slot.** -/
theorem avoidsMask_of_shapeOk {st : Stmt} (hw : RegionsWF st) (h : Zk.shapeOk st = .ok ()) : avoidsMask st = true := by
  obtain ⟨r, hr, hm⟩ := maskFree_of_shapeOk h
  exact avoidsMask_of_maskFree hr hw hm

/-- A `--zk` setup of one table's statement avoids the mask slot. -/
theorem DrawSetupZK.avoidsMask {I : Inputs} {n mPts : ℕ} {S : Finset (Fin n)} {d : Lean.Json}
    (y : ZkLink.DrawSetupZK I mPts S d) : Composed.avoidsMask y.x.st = true :=
  avoidsMask_of_shapeOk (setupH_wf y.x.h).2 y.hzk

/-- A `--zk` setup of a hidden-output table's statement avoids the mask slot. -/
theorem DrawSetupZHJ.avoidsMask {I' : Inputs} {n mPts : ℕ} {S : Finset (Fin n)} {d : Lean.Json}
    (y : ZkHidden.DrawSetupZHJ I' mPts S d) : Composed.avoidsMask y.x.st = true :=
  avoidsMask_of_shapeOk (Hidden.setupHidden_wf y.x.h).2 y.hzk

end FlockSoundness.Discharge.FailClosed
