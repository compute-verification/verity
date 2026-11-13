module

public import Proofs.Properties.Verity.Sampled.MemberForm.ClaimReads
public import Proofs.Properties.Verity.Sampled.MemberForm.WindowPasses

/-!
# SPOT's leg at the per-member form's accepts

SPOT's leg at the deployment's windows (`WindowInstance.spotSound`) and remote's soundness over windows of several
members (`remote_soundness_window`) take three facts of the verifier: what the claim reads of the window
(`WindowInstance.ClaimReads`), the verifier's window at an accepted play (`WindowInstance.WindowAt`) and what the
verifier's acceptance gives of each play (`WindowInstance.WindowReads`). This module gives them the three as proved
of the per-member form (`claimReads`, `windowAt_form`, `windowReads_members`), at the evidence sent for each session
and the proof files decoded from it at the statement's table, so that what is left is stated in one place:

* the window is the parameters' members on the registered files (`WindowInstance.WindowRegistered`);
* the form's accepts hold `MembersAccept` (`FormAccepts`) and their members share no unit (`FormApart`);
* the registered files hold each member's file (`RegisteredHolds`);
* each member's commitment is ready for its openings (`Commit.Ready`).

The check as it is accepts no parameters that name members (`formAccepts_refused`), so no accepted play has a
window of members and both statements are empty until the per-member form accepts such parameters.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **SPOT's leg at the per-member form's accepts** (`WindowInstance.spotSound`, at `claimReads`, `windowAt_form`
and `windowReads_members`): where the window is the parameters' members on the registered files, the form's accepts
hold `MembersAccept` with members that share no unit, the registered files hold each member's file and each
member's commitment is ready for its openings. The check as it is accepts no parameters that name members, so the
statement is empty until the per-member form accepts such parameters. -/
theorem spotSound_form (I : WindowInstance d α) (K₀ : WindowConstructions I.base)
    (k : Verity.Registration.Composite) (reg : Option Claims.Msg → ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hW : I.WindowRegistered) (hF : FormAccepts) (hA : FormApart) (hR : RegisteredHolds I.base)
    (hready : ∀ θ F W, I.windows θ F = some W → ∀ m ∈ W.ms, Commit.Ready m.D m.hU m.n {} m.pub) :
    (I.vocab (I.constructionsAt K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SpotSound k :=
  I.spotSound K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _ hready) (windowAt_form I k hW hF hA hR) (windowReads_members I k hF hR)

/-- **Remote's soundness over windows of several members, at the per-member form's accepts**
(`remote_soundness_window`, at `claimReads`, `windowAt_form` and `windowReads_members`): under `spotSound_form`'s
premises and the base's PoUW and inference-only targets. The check as it is accepts no parameters that name
members, so the statement is empty until the per-member form accepts such parameters. -/
theorem soundnessAt_form (I : WindowInstance d α) (K₀ : WindowConstructions I.base)
    (reg : Option Claims.Msg → ByteArray)
    (record₀ : Window → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Claims.Msg)
    (recordDigest : Window → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hW : I.WindowRegistered) (hF : FormAccepts) (hA : FormApart) (hR : RegisteredHolds I.base)
    (hready : ∀ θ F W, I.windows θ F = some W → ∀ m ∈ W.ms, Commit.Ready m.D m.hU m.n {} m.pub)
    (hP : I.base.PouwSound .remote) (hI : I.base.InferenceOnlySound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SoundnessAt .remote :=
  remote_soundness_window I K₀ reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _ hready) (windowAt_form I .remote hW hF hA hR) (windowReads_members I .remote hF hR) hP hI

end Verity.Sampled.MemberForm
