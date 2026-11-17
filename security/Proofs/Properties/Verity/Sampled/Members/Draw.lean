module

public import Proofs.Flock.EndToEnd.Joint.V5.Layout
public import Proofs.Properties.Verity.Sampled.RecordDraw
public import Proofs.Properties.Verity.Sampled.Deployed

/-! The period's one sample, member by member. -/

@[expose] public section

namespace MemberLayout

theorem memberUnits_pairwise (cnt : List ℕ) (k n : ℕ) {us : List ℕ} (h : us.Pairwise (· < ·)) :
    (memberUnits cnt k n us).Pairwise (· < ·) := by
  rw [memberUnits, List.pairwise_map]
  refine (h.filter _).imp_of_mem fun {a b} ha hb hab => ?_
  have ha' := (List.mem_filter.1 ha).2
  have hb' := (List.mem_filter.1 hb).2
  simp only [decide_eq_true_eq] at ha' hb'
  omega

theorem memberUnits_lt (cnt : List ℕ) (k n : ℕ) (us : List ℕ) {v : ℕ} (h : v ∈ memberUnits cnt k n us) :
    v < n :=
  ((mem_memberUnits cnt k n us v).1 h).1

end MemberLayout

namespace Verity.Sampled

open Verity.Game Flock.Guarantees
open Properties.Challenger (bitsOf Draws)

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

theorem unitsAscending_member {n : ℕ} {units : Array ℕ} (h : unitsAscending n units = true) (cnt : List ℕ)
    (k nk : ℕ) : unitsAscending nk (MemberLayout.memberUnits cnt k nk units.toList).toArray = true :=
  unitsAscending_of (fun _ hv => MemberLayout.memberUnits_lt cnt k nk _ hv)
    (MemberLayout.memberUnits_pairwise cnt k nk (unitsAscending_spec h).2)

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

theorem joint_sample_eq (w : JointAuditV5) (ω : w.L.Ω) {units : Array ℕ} {read : ℕ}
    (h : _root_.Spot.Draw.sampleUnits w.samplingDesign w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read)) :
    w.L.draw ω = Finset.univ.filter fun u => u.val ∈ units :=
  Finset.ext fun u => (w.draw_ok ω h u).trans (by simp)

theorem joint_sampleUnits_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws}
    {L : Flock.Draw.SamplingDesign} {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units)
    (w : JointAuditV5) (ω : w.L.Ω) (hL : w.samplingDesign = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    ∃ read, _root_.Spot.Draw.sampleUnits w.samplingDesign w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) =
      .ok (units, read) := by
  obtain ⟨_, _, _, _, _, _, _, _, _, _, read, hdu⟩ := h
  obtain ⟨t, ht⟩ := append_of_bitsOf_prefix hξ
  refine ⟨read, ?_⟩
  rw [ht, hL, hn]
  exact sampleUnits_append L n ξ t units read hdu

theorem joint_sample_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.SamplingDesign}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) (w : JointAuditV5) (ω : w.L.Ω)
    (hL : w.samplingDesign = L) (hn : w.nJ = n) (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    w.L.draw ω = Finset.univ.filter fun u => u.val ∈ units := by
  obtain ⟨read, hr⟩ := joint_sampleUnits_of_drew h w ω hL hn hξ
  exact joint_sample_eq w ω hr

theorem member_sample_eq {w : JointAuditV5} (hw : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) {units : Array ℕ}
    {read : ℕ}
    (h : _root_.Spot.Draw.sampleUnits w.samplingDesign w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read)) :
    (w.design k).draw ω = Finset.univ.filter fun v : Fin (w.ms.get k).n =>
      v.val ∈ (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray := by
  ext v
  rw [hw.design_sample_ok k ω h v]
  simp

theorem member_sample_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.SamplingDesign}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) {w : JointAuditV5} (hw : w.Laid)
    (j : Fin w.ms.length) (ω : w.L.Ω) (hL : w.samplingDesign = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) :
    (w.design j).draw ω = Finset.univ.filter fun v : Fin (w.ms.get j).n =>
      v.val ∈ (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray := by
  obtain ⟨read, hr⟩ := joint_sampleUnits_of_drew h w ω hL hn hξ
  exact member_sample_eq hw j ω hr

theorem printedPart_member {w : JointAuditV5} (hw : w.Laid) (k : Fin w.ms.length) (ω : w.L.Ω) {units : Array ℕ}
    {read : ℕ}
    (h : _root_.Spot.Draw.sampleUnits w.samplingDesign w.nJ (_root_.Properties.Spot.bytesOf (w.src ω 0)) = .ok (units, read))
    (hu : unitsAscending w.nJ units = true) :
    printedPart ((w.design k).draw ω) =
      Flock.Draw.UnitSample.toJson
        ⟨.subset (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray.size, (w.ms.get k).n,
          (MemberLayout.memberUnits w.counts k.1 (w.ms.get k).n units.toList).toArray, #[]⟩ := by
  have ha := unitsAscending_member hu w.counts k.1 (w.ms.get k).n
  rw [member_sample_eq hw k ω h, printedPart, printedSample, card_filter_ascending ha, sampleOf_filter ha]

theorem printedPart_of_drew {k : Verity.Registration.Composite} {s : Verity.Sent} {dr : Draws} {L : Flock.Draw.SamplingDesign}
    {n : ℕ} {ξ : ByteArray} {units : Array ℕ} (h : Drew k s dr L n ξ units) {w : JointAuditV5} (hw : w.Laid)
    (j : Fin w.ms.length) (ω : w.L.Ω) (hL : w.samplingDesign = L) (hn : w.nJ = n)
    (hξ : bitsOf ξ <+: bitsOf (_root_.Properties.Spot.bytesOf (w.src ω 0))) (hu : unitsAscending n units = true) :
    printedPart ((w.design j).draw ω) =
      Flock.Draw.UnitSample.toJson
        ⟨.subset (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray.size, (w.ms.get j).n,
          (MemberLayout.memberUnits w.counts j.1 (w.ms.get j).n units.toList).toArray, #[]⟩ := by
  obtain ⟨read, hr⟩ := joint_sampleUnits_of_drew h w ω hL hn hξ
  exact printedPart_member hw j ω hr (by rw [hn]; exact hu)

end Verity.Sampled
