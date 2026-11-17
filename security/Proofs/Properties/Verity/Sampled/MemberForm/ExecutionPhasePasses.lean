module

public import Proofs.Properties.Verity.Sampled.MemberForm.Laid
public import Proofs.Properties.Verity.Sampled.MemberForm.Reads
public import Proofs.Properties.Verity.Sampled.Accepts

/-! The window's instance at an accept of the per-member form. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Verity.Game
open Lean (Json)
open Properties.Challenger (Draws)

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem executionPhasePasses_members (I : ExecutionPhaseInstance d α) (k : Verity.Registration.Composite) (hF : FormAccepts)
    (hA : FormApart) (hR : RegisteredHolds I.base) : I.ExecutionPhasePasses k := by
  rintro θ F W - hon ⟨c, o, hreach, hv, ht, hfs⟩
  have hθ : o.params = θ := (Binding.play_reaches hreach).1
  subst hθ
  obtain ⟨p, ms, vms, hp, hms, -⟩ := hon.members
  obtain ⟨a, b, e, c', ev, ξ, dr, sts, r, -, hacc, hM⟩ := accept_members hF hp hms hv
  obtain ⟨roots, -, hheld⟩ := _root_.Spot.OneStageZk.membersRun_files hM.run
  have hpass : ∀ m ∈ W.ms, FilesPass m :=
    hon.passes ht hp hms (fun vm hvm => readsFiles_of_sent hR hfs hp hms hvm) hheld
  exact ⟨hon.laid hpass hp hms (laidHeld_one (membersStructure_laid (membersRun_ok hM.run hM.all).1))
    (hA _ _ _ _ _ _ _ _ _ _ hacc hms), hpass⟩

theorem executionPhaseAt_form (I : ExecutionPhaseInstance d α) (k : Verity.Registration.Composite) (hW : I.ExecutionPhaseRegistered)
    (hF : FormAccepts) (hA : FormApart) (hR : RegisteredHolds I.base) : I.ExecutionPhaseAt k :=
  I.executionPhaseAt_members k hW (executionPhasePasses_members I k hF hA hR)

theorem executionPhaseReads_members (I : ExecutionPhaseInstance d α) (k : Verity.Registration.Composite) (hF : FormAccepts)
    (hR : RegisteredHolds I.base) : I.ExecutionPhaseReads k sentEvidenceAt (statementFiles statementTable) := by
  intro θ F W hW c σ z hv hl ht hfs hsd
  have hθ := (play_ends I.base θ F c σ z).1
  have hset : MembersSetUp I.base I.pinned (CoinTape.run (I.base.game θ F c) σ z).params F W := by
    rw [hθ]
    exact (I.executionPhases_some hW).2
  have h := executionPhaseRead_members hF hR hset hv hl ht hfs hsd
  rw [hθ] at h
  exact h

end Verity.Sampled.MemberForm
