module

public import Proofs.Properties.Verity.Sampled.MemberForm.Laid
public import Proofs.Properties.Verity.Sampled.MemberForm.Registered
public import Proofs.Properties.Verity.Sampled.MemberForm.Run

/-!
# At an accepted play of the per-member form, the window is laid out and its members' files pass

`WindowInstance.WindowPasses` is what the window's theorem takes of an accepted play besides the registration's
own facts: the deployment's window, the parameters' members on the registered files, is laid out in order and each
member's files pass the files check. This file proves it of an accept of the per-member form.

* **An accepted play at parameters that name members is an accept of the form** (`accept_members`): v2's case in
  it accepted at the coins it read, on those parameters, and the form's accepts hold `MembersAccept`
  (`FormAccepts`).
* **`WindowPasses`** (`windowPasses_members`): the form's run reads every member's files, sampled or not
  (`Spot.OneStageZk.membersRun_files`), on blobs that hold each member's registered file (`readsFiles_of_sent`), so
  every member's files pass (`WindowOn.passes`); and the form's structure holds every member to one range
  (`laidHeld_one`), the parameters' members share no unit (`FormApart`), so the window is laid out in order
  (`WindowOn.laid`).

It takes three statements besides the play: `FormAccepts` and `FormApart`, which hold of the check as it is and
which a check that accepts the form proves of its own accepts and its own parse, and `RegisteredHolds`, the
registration's.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

open Lean (Json)
open Properties.Challenger (Draws)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An accepted play at parameters that name members is an accept of the form**: v2's case in the play accepted
at the coins it read, on the play's parameters, and the form's accepts hold `MembersAccept`. -/
theorem accept_members {V : Vocab d α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    (hF : FormAccepts) {p : _root_.Spot.OneStageZk.Params} {ms : Array _root_.Spot.OneStageZk.Member}
    (hp : ParamsAt o.params p) (hms : p.members = some ms) (hv : V.Verdict k .sampled o) :
    ∃ (a b e : ByteArray) (c : _root_.Spot.OneStage.Commitment) (ev : _root_.Spot.OneStageZk.Evidence)
      (ξ : ByteArray) (dr : Draws) (sts : Array _root_.Spot.OneStageZk.StatementFiles)
      (r : _root_.Spot.OneStageZk.Run),
      V2Case k o.sent a b e ∧ V2Accepts (_root_.Core.Blobs.ofArray o.sent.blobs) a b e p c ev ξ dr ∧
        MembersAccept p c ξ ev (_root_.Core.Blobs.ofArray o.sent.blobs) ms sts r := by
  obtain ⟨a, b, e, p', c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp (verdict_case hv)
  have hpp : p = p' := paramsAt_case hv.2.1 hcase hacc hp
  subst hpp
  obtain ⟨sts, r, hM⟩ := hF _ _ _ _ _ _ _ _ _ _ hacc hms
  exact ⟨a, b, e, c, ev, ξ, dr, sts, r, hcase, hacc, hM⟩

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

end Verity.Sampled.MemberForm
