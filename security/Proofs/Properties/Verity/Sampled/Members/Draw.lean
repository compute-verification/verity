module

public import Proofs.Flock.EndToEnd.Joint.V5.Layout
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Verity.Sampled.Deployed

/-!
# The period's one sample, member by member

v2 samples once, over the whole window's population of proof blocks, from the coins it reads (`Drew`). At a window's
audit of several members (`Flock.Guarantees.JointAuditV5`) laid out in order (`JointAuditV5.Laid`), that one sample is
the audit's, and each member's sample is its part of it:

* **The window's sample is v2's units** (`joint_draw_eq`, `joint_drawUnits_of_drew`, `joint_draw_of_drew`): at coins
  whose bytes start with the coins v2 drew its units from, at a window whose law and population are the ones v2 drew
  under, the window's sample is the proof blocks v2's units name. A sample reads only a prefix of its bytes
  (`drawUnits_append`).
* **Member `k`'s sample is its part of v2's units** (`member_draw_eq`, `member_draw_of_drew`): the units `u` with
  `base k ≤ u < base k + n_k`, each less the member's base, in the units' order (`MemberLayout.memberUnits`).
* **Member `k`'s sample file is the subset print of that part** (`printedPart_member`, `printedPart_of_drew`): where
  v2's units are ascending below the population, member `k`'s part of them is ascending below its count
  (`unitsAscending_member`), so the sample file its session is served at its part of the sample (`printedPart`) is
  `UnitDraw.toJson` of that part under `subset` of its size, the form a record's sample is held to (`recordDrawn`).
-/

@[expose] public section

namespace MemberLayout

/-- **A member's part of an ascending list is ascending.** -/
theorem memberUnits_pairwise (cnt : List ℕ) (k n : ℕ) {us : List ℕ} (h : us.Pairwise (· < ·)) :
    (memberUnits cnt k n us).Pairwise (· < ·) := by
  rw [memberUnits, List.pairwise_map]
  refine (h.filter _).imp_of_mem fun {a b} ha hb hab => ?_
  have ha' := (List.mem_filter.1 ha).2
  have hb' := (List.mem_filter.1 hb).2
  simp only [decide_eq_true_eq] at ha' hb'
  omega

/-- A member's part of a list is below its count. -/
theorem memberUnits_lt (cnt : List ℕ) (k n : ℕ) (us : List ℕ) {v : ℕ} (h : v ∈ memberUnits cnt k n us) :
    v < n :=
  ((mem_memberUnits cnt k n us v).1 h).1

end MemberLayout

namespace Verity.Sampled

open Verity.Game Flock.Guarantees
open Properties.Challenger (bitsOf Draws)

/-! ## Ascending units -/

/-- **An ascending list below `n`** passes `unitsAscending`. -/
theorem unitsAscending_of {n : ℕ} {l : List ℕ} (hlt : ∀ x ∈ l, x < n) (hs : l.Pairwise (· < ·)) :
    unitsAscending n l.toArray = true := by
  rw [unitsAscending, List.all_eq_true]
  intro i hi
  have hi' : i < l.length := by simpa using hi
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, beq_iff_eq]
  have e : l.toArray[i]! = l[i] := by simp [hi']
  refine ⟨by rw [e]; exact hlt _ (List.getElem_mem hi'), ?_⟩
  by_cases h0 : i = 0
  · exact Or.inl h0
  · refine Or.inr ?_
    have e' : l.toArray[i - 1]! = l[i - 1] := by
      have : i - 1 < l.length := by omega
      simp [this]
    rw [e, e']
    exact List.pairwise_iff_getElem.1 hs (i - 1) i (by omega) hi' (by omega)

/-- **A member's part of ascending units is ascending below its count.** -/
theorem unitsAscending_member {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) (cnt : List ℕ)
    (k nk : ℕ) : unitsAscending nk (MemberLayout.memberUnits cnt k nk units.toList).toArray = true :=
  unitsAscending_of (fun _ hv => MemberLayout.memberUnits_lt cnt k nk _ hv)
    (MemberLayout.memberUnits_pairwise cnt k nk (unitsAscending_spec h).2)

/-- **An ascending sample names as many proof blocks as it has units.** -/
theorem card_filter_ascending {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) :
    (Finset.univ.filter fun u : Fin n => u.val ∈ units).card = units.size := by
  obtain ⟨hlt, hs⟩ := unitsAscending_spec h
  have himg : (Finset.univ.filter fun u : Fin n => u.val ∈ units).image Fin.val = units.toList.toFinset := by
    ext x
    simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and, List.mem_toFinset,
      Array.mem_toList_iff]
    constructor
    · rintro ⟨u, hu, rfl⟩
      exact hu
    · intro hx
      exact ⟨⟨x, hlt x (Array.mem_toList_iff.2 hx)⟩, hx, rfl⟩
  rw [← Finset.card_image_of_injective _ Fin.val_injective, himg, List.toFinset_card_of_nodup (hs.imp ne_of_lt),
    Array.length_toList]

/-! ## The window's sample -/

/-- **The window's sample at coins whose sample ends** with `units` is the proof blocks they name. -/
theorem joint_draw_eq (w : JointAuditV5) (ω : w.L.Ω) {units : Array ℕ} {read : ℕ}
    (h : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read)) :
    w.L.draw ω = Finset.univ.filter fun u => u.val ∈ units :=
  Finset.ext fun u => (w.draw_ok ω h u).trans (by simp)

/-- **v2's sample ends the same on the window's bytes**: at coins whose bytes start with the coins v2 drew its units
from, at a window whose law and population are the ones v2 drew under. -/
theorem joint_drawUnits_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws}
    {L : Flock.Draw.Law} {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units)
    (w : JointAuditV5) (ω : w.L.Ω) (hL : w.law = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    ∃ read, _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) =
      .ok (units, read) := by
  obtain ⟨_, _, _, _, _, _, _, _, _, _, read, hdu⟩ := h
  obtain ⟨t, ht⟩ := append_of_bitsOf_prefix hξ
  refine ⟨read, ?_⟩
  rw [ht, hL, hn]
  exact drawUnits_append L n ξ t units read hdu

/-- **v2's units are the window's sample** at coins whose bytes start with the coins v2 drew them from, at a window
whose law and population are the ones v2 drew under: the period's one sample is the joint audit's. -/
theorem joint_draw_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) (w : JointAuditV5) (ω : w.L.Ω)
    (hL : w.law = L) (hn : w.nJ = n) (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    w.L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨read, hr⟩ := joint_drawUnits_of_drew h w ω hL hn hξ
  exact joint_draw_eq w ω hr

/-! ## Each member's sample -/

/-- **Member `k`'s sample at coins whose sample ends** with `units` is the proof blocks its part of them names: the
units `u` with `base k ≤ u < base k + n_k`, each less the member's base. -/
theorem member_draw_eq {w : JointAuditV5} (hw : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) {units : Array ℕ}
    {read : ℕ}
    (h : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read)) :
    (w.design k).draw ω = Finset.univ.filter fun v : Fin (w.ms.get k).n =>
      v.val ∈ (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray := by
  ext v
  rw [hw.design_draw_ok k ω h v]
  simp

/-- **Member `k`'s sample is its part of v2's units**, at coins whose bytes start with the coins v2 drew them from, at a
window laid out in order whose law and population are the ones v2 drew under. -/
theorem member_draw_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) {w : JointAuditV5} (hw : w.Laid)
    (j : Fin w.ms.length) (ω : w.L.Ω) (hL : w.law = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    (w.design j).draw ω = Finset.univ.filter fun v : Fin (w.ms.get j).n =>
      v.val ∈ (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray := by
  obtain ⟨read, hr⟩ := joint_drawUnits_of_drew h w ω hL hn hξ
  exact member_draw_eq hw j ω hr

/-- **Member `k`'s sample file is the subset print of its part of the units**: where the window's sample ends with
`units`, ascending below the population, the sample file member `k`'s session is served at its part of the sample
(`printedPart`) is `UnitDraw.toJson` of its part of `units`, in its own indices, under `subset` of that part's size. -/
theorem printedPart_member {w : JointAuditV5} (hw : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) {units : Array ℕ}
    {read : ℕ}
    (h : _root_.Spot.Draw.drawUnits w.law w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read))
    (hu : unitsAscending w.nJ units = true) :
    printedPart ((w.design k).draw ω) =
      Flock.Draw.UnitDraw.toJson
        ⟨.subset (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray.size, (w.ms.get k).n,
          (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray, #[]⟩ := by
  have ha := unitsAscending_member hu w.counts k.1 (w.ms.get k).n
  rw [member_draw_eq hw k ω h, printedPart, printedDraw, card_filter_ascending ha, drawOf_filter ha]

/-- **Member `k`'s sample file is the subset print of its part of v2's units**, at coins whose bytes start with the
coins v2 drew them from, at a window laid out in order whose law and population are the ones v2 drew under, where
v2's units are ascending below the population. -/
theorem printedPart_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.Law}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) {w : JointAuditV5} (hw : w.Laid)
    (j : Fin w.ms.length) (ω : w.L.Ω) (hL : w.law = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) (hu : unitsAscending n units = true) :
    printedPart ((w.design j).draw ω) =
      Flock.Draw.UnitDraw.toJson
        ⟨.subset (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray.size, (w.ms.get j).n,
          (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray, #[]⟩ := by
  obtain ⟨read, hr⟩ := joint_drawUnits_of_drew h w ω hL hn hξ
  exact printedPart_member hw j ω hr (by rw [hn]; exact hu)

end Verity.Sampled
