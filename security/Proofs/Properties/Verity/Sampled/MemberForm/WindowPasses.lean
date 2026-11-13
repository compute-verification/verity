module

public import Proofs.Properties.Verity.Sampled.MemberForm.Laid
public import Proofs.Properties.Verity.Sampled.MemberForm.Reads
public import Proofs.Properties.Verity.Sampled.Accepts

/-!
# The window's instance at an accept of the per-member form

The window's theorem takes three facts of the deployment's windows at accepted plays (`WindowInstance.WindowPasses`,
`WindowAt`, `WindowReads`). This file proves each of an accept of the per-member form.

* **`WindowPasses`** (`windowPasses_members`): the deployment's window, the parameters' members on the registered
  files, is laid out in order and each member's files pass the files check. The form's run reads every member's
  files, sampled or not (`Spot.OneStageZk.membersRun_files`), on blobs that hold each member's registered file
  (`readsFiles_of_sent`), so every member's files pass (`WindowOn.passes`); and the form's structure holds every
  member to one range (`laidHeld_one`), the parameters' members share no unit (`FormApart`), so the window is laid
  out in order (`WindowOn.laid`).
* **`WindowAt`** (`executionPhaseAt_form`): so the verifier sets the deployment's window up at every accepted play, where
  the window is the parameters' members on the registered files (`WindowInstance.WindowRegistered`).
* **`WindowReads`** (`windowReads_members`): the read at a play (`windowRead_members`) at each play of the period's
  game, whose parameters are the game's.

It takes three statements besides the play: `FormAccepts` and `FormApart`, which hold of the check's accepts and
its parse (`formAccepts`, `formApart`), and `RegisteredHolds`, the registration's.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Verity.Game
open Lean (Json)
open Properties.Challenger (Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **At an accepted play of the per-member form, the deployment's window is laid out in order and its members'
files pass** (`WindowInstance.WindowPasses`), where the form's accepts hold `MembersAccept` and their members share
no unit, and the registered files hold each member's file. -/
theorem windowPasses_members (I : WindowInstance d α) (k : Verity.Registration.Composite) (hF : FormAccepts)
    (hA : FormApart) (hR : RegisteredHolds I.base) : I.WindowPasses k := by
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

/-- **The verifier sets the deployment's window up at every accepted play of the per-member form**
(`WindowInstance.WindowAt`): the window is the parameters' members on the registered files
(`WindowInstance.WindowRegistered`), laid out with each member's files passing at each accepted play
(`windowPasses_members`, `WindowInstance.executionPhaseAt_members`). The check's accepts hold `FormAccepts` and `FormApart`
(`formAccepts`, `formApart`). -/
theorem executionPhaseAt_form (I : WindowInstance d α) (k : Verity.Registration.Composite) (hW : I.WindowRegistered)
    (hF : FormAccepts) (hA : FormApart) (hR : RegisteredHolds I.base) : I.WindowAt k :=
  I.executionPhaseAt_members k hW (windowPasses_members I k hF hA hR)

/-- **What the verifier's acceptance gives of each play of the per-member form, at the deployment's windows**
(`WindowInstance.WindowReads`), at the evidence sent for each session and the proof files decoded from it at the
statement's table: `windowRead_members` at each play of the period's game, where the form's accepts hold
`MembersAccept` (`formAccepts`) and the registered files hold each member's file. -/
theorem windowReads_members (I : WindowInstance d α) (k : Verity.Registration.Composite) (hF : FormAccepts)
    (hR : RegisteredHolds I.base) : I.WindowReads k sentEvidenceAt (statementFiles statementTable) := by
  intro θ F W hW c σ z hv hl ht hfs hsd
  have hθ := (play_ends I.base θ F c σ z).1
  have hset : MembersSetUp I.base (CoinTape.run (I.base.game θ F c) σ z).params F W := by
    rw [hθ]
    exact (I.windows_some hW).2
  have h := windowRead_members hF hR hset hv hl ht hfs hsd
  rw [hθ] at h
  exact h

end Verity.Sampled.MemberForm
