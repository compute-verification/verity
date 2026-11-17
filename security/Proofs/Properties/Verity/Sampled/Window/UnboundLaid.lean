module

public import Proofs.Properties.Verity.Sampled.Window.UnboundMember

/-! The claim at a window: across its members. -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

namespace MemberV5

open Classical

variable {Reg : Type} {nJ : ℕ} (m : MemberV5 Reg nJ)

theorem fileRead_units (hc : m.Checked) : m.fileRead.units = m.n :=
  Flock.Bridge.unitsOfFile_eq hc.parsed

theorem lt_of_mem_faulty {Xv : Fin m.Cc.N → Bool} {v : ℕ} (h : v ∈ m.faulty Xv) : v < m.n := by
  have hv : v < m.blocks.blocks.length := Finset.mem_range.1 (Finset.mem_filter.1 h).1
  have hl : m.blocks.blocks.length = m.n := Flock.Bridge.blocks_length_partOf m.P (ns := [m.n]) (by simp)
  rwa [hl] at hv

end MemberV5

namespace JointAuditV5

open Classical

variable (w : JointAuditV5) [Nonempty w.EndΩ]

theorem seats_length (τ : Strategy w.game) : (w.seats τ).length = w.ms.length := by
  simp [seats]

theorem seats_drop (τ : Strategy w.game) {j : ℕ} {s : Seat w.Reg w.nJ} {ss : List (Seat w.Reg w.nJ)}
    (e : (w.seats τ).drop j = s :: ss) :
    ∃ h : j < w.ms.length, s = ⟨w.ms.get ⟨j, h⟩, w.memberX τ ⟨j, h⟩⟩ ∧ ss = (w.seats τ).drop (j + 1) := by
  have hj : j < (w.seats τ).length := by
    by_contra hh
    rw [List.drop_eq_nil_of_le (Nat.le_of_not_lt hh)] at e
    cases e
  have hj' : j < w.ms.length := by
    rw [← w.seats_length τ]
    exact hj
  rw [List.drop_eq_getElem_cons hj] at e
  obtain ⟨h1, h2⟩ := List.cons.inj e
  refine ⟨hj', ?_, h2.symm⟩
  rw [← h1]
  simp only [seats, List.getElem_ofFn]

theorem faulty_of_readsWith (τ : Strategy w.game) :
    ∀ (rs : List Verity.FileRead) (r : Verity.FileRead)
      (op : (g : Fin (Verity.FileRead.circuitWith r rs).N) → BitVec ((Verity.FileRead.circuitWith r rs).w g) → Prop)
      (tr : (Verity.FileRead.circuitWith r rs).Trace) (j : ℕ), ReadsWith r rs op tr ((w.seats τ).drop j) →
      ∀ u ∈ ResourceAccounting.unbound (Verity.FileRead.partsWith r rs) op tr,
        ∃ (k : Fin w.ms.length) (v : ℕ), (w.ms.get k).Checked ∧ v ∈ (w.ms.get k).faulty (w.memberX τ k) ∧
          MemberLayout.base w.counts j + u.val = w.base k + v
  | [], r, op, tr, j, h, u, hu => by
    obtain ⟨s, e, hread⟩ := h
    obtain ⟨hj, rfl, -⟩ := w.seats_drop τ e
    have hsub := (w.ms.get ⟨j, hj⟩).unboundFaulty r op tr (w.memberX τ ⟨j, hj⟩) hread
    exact ⟨⟨j, hj⟩, u.val, hread.checked, hsub (Finset.mem_image_of_mem Fin.val hu), rfl⟩
  | r' :: rs, r, op, tr, j, h, u, hu => by
    obtain ⟨s, ss', e, hread, hrest⟩ := h
    obtain ⟨hj, rfl, rfl⟩ := w.seats_drop τ e
    have hsub := (w.ms.get ⟨j, hj⟩).unboundFaulty r _ _ (w.memberX τ ⟨j, hj⟩) hread
    have hu' := (Finset.ext_iff.1 (Verity.unbound_beside r.parts (Verity.FileRead.partsWith r' rs) op tr) u).1 hu
    rcases Finset.mem_union.1 hu' with hl | hr
    · obtain ⟨a, ha, hau⟩ := Finset.mem_map.1 hl
      have hval : u.val = a.val := (congrArg Fin.val hau).symm
      exact ⟨⟨j, hj⟩, a.val, hread.checked, hsub (Finset.mem_image_of_mem Fin.val ha),
        congrArg (MemberLayout.base w.counts j + ·) hval⟩
    · obtain ⟨b, hb, hbu⟩ := Finset.mem_map.1 hr
      have hval : u.val = r.units + b.val := (congrArg Fin.val hbu).symm
      obtain ⟨k, v, hc, hv, hbase⟩ := faulty_of_readsWith τ rs r' _ _ (j + 1) hrest b hb
      have hn : r.units = (w.ms.get ⟨j, hj⟩).n :=
        (congrArg Verity.FileRead.units hread.read).trans ((w.ms.get ⟨j, hj⟩).fileRead_units hread.checked)
      have hs : MemberLayout.base w.counts (j + 1) = MemberLayout.base w.counts j + (w.ms.get ⟨j, hj⟩).n :=
        (MemberLayout.base_succ w.counts (by rw [w.counts_length]; exact hj)).trans
          (congrArg (MemberLayout.base w.counts j + ·) (w.counts_get ⟨j, hj⟩))
      exact ⟨k, v, hc, hv, by omega⟩

theorem unboundIncorrect : w.UnboundIncorrect := by
  intro hl R op tr τ hr x hx
  obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hx
  obtain ⟨k, v, hc, hv, hbase⟩ := w.faulty_of_readsWith τ R.rest R.first op tr 0
    (show ReadsWith R.first R.rest op tr ((w.seats τ).drop 0) from hr) u hu
  have hbase' : u.val = w.base k + v := by
    rw [MemberLayout.base_zero, Nat.zero_add] at hbase
    exact hbase
  have hvn : v < (w.ms.get k).n := (w.ms.get k).lt_of_mem_faulty hv
  have hle : w.base k + (w.ms.get k).n ≤ w.nJ := by
    have h1 := MemberLayout.base_add_le_sum w.counts (k := k.val) (by rw [w.counts_length]; exact k.2)
    rw [w.counts_get k, hl.total] at h1
    exact h1
  have hlt : u.val < w.nJ := by
    rw [hbase']
    exact lt_of_lt_of_le (Nat.add_lt_add_left hvn _) hle
  have hw : (⟨v, hvn⟩ : Fin (w.ms.get k).n) ∈ w.memberIncorrect τ k := by
    rw [w.memberIncorrect_eq τ k]
    exact (w.ms.get k).faultyIncorrect hc _ _ ⟨v, hvn⟩ hv
  exact Finset.mem_image.2 ⟨⟨u.val, hlt⟩, (hl.mem_incorrectSet τ _).2 ⟨k, ⟨v, hvn⟩, hbase', hw⟩, rfl⟩

end JointAuditV5

end Flock.Guarantees

namespace Verity.Sampled

theorem ExecutionPhase.unboundIncorrect (W : ExecutionPhase) : W.UnboundIncorrect := fun t _ => (W.joint t).unboundIncorrect

end Verity.Sampled
