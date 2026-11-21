module

public import Proofs.Properties.Verity.Sampled.MemberForm.ClaimReads
public import Proofs.Properties.Verity.Sampled.MemberForm.ExecutionPhasePasses

/-!
# SPOT's soundness at the per-member form's accepts

SPOT's soundness at the deployment's windows (`ExecutionPhaseInstance.spotSound`) and remote's soundness over windows of
several members (`remote_soundness_executionPhase`) take three facts of the verifier: what the claim reads of the window
(`ExecutionPhaseInstance.ClaimReads`), the verifier's window at an accepted play (`ExecutionPhaseInstance.ExecutionPhaseAt`) and what the
verifier's acceptance gives of each play (`ExecutionPhaseInstance.ExecutionPhaseReads`). This module gives them the three as proved
of the per-member form (`claimReads`, `executionPhaseAt_form`, `executionPhaseReads_members`), at the evidence sent for each session
and the proof files decoded from it at the statement's table, and at what the check's accepts hold of the form: they
hold `MembersAccept` (`formAccepts`) and their members share no unit (`formApart`). What is left is stated in one
place, two premises no refusal of the check enforces:

* the window is the parameters' members on the registered files (`ExecutionPhaseInstance.ExecutionPhaseRegistered`);
* the registered files hold each member's file (`RegisteredHolds`);

and the two-stage variant: the verifier accepts a two-stage run whose every test passes; its soundness is the open
premise `TwoStageBound`; an accepted run of it is recorded as unproved.
-/

@[expose] public section

namespace Verity.Sampled.MemberForm

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- **SPOT's soundness at the per-member form's accepts** (`ExecutionPhaseInstance.spotSound`, at `claimReads`,
`executionPhaseAt_form` and `executionPhaseReads_members`, at the check's accepts: `formAccepts`, `formApart`): where the window is
the parameters' members on the registered files and the registered files hold each member's file. A window with a
member needs the flat parse `Flock.Desc.parse` at tags the form fixes to the templated statement (`MemberOn.tags`),
so the theorem's window branch has no served instance. -/
theorem spotSound_form (I : ExecutionPhaseInstance d α) (K₀ : ExecutionPhaseConstructions I.base)
    (k : Verity.Registration.Composite) (reg : Option Hash.Msg → ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Hash.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Hash.Msg)
    (hW : I.ExecutionPhaseRegistered) (hR : RegisteredHolds I.base)
    (hTwo : TwoStageBound (I.vocab (I.constructionsAt K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)) k) :
    (I.vocab (I.constructionsAt K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SpotSound k :=
  I.spotSound K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _) (executionPhaseAt_form I k hW formAccepts formApart hR)
    (executionPhaseReads_members I k formAccepts hR) hTwo

/-- **Remote's soundness over windows of several members, at the per-member form's accepts**
(`remote_soundness_executionPhase`, at `claimReads`, `executionPhaseAt_form` and `executionPhaseReads_members`): under `spotSound_form`'s
premises and the base's PoUW target. A window with a member needs the flat parse `Flock.Desc.parse` at tags the form
fixes to the templated statement (`MemberOn.tags`), so the theorem's window branch has no served instance. -/
theorem soundnessAt_form (I : ExecutionPhaseInstance d α) (K₀ : ExecutionPhaseConstructions I.base)
    (reg : Option Hash.Msg → ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Hash.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Hash.Msg)
    (hW : I.ExecutionPhaseRegistered) (hR : RegisteredHolds I.base)
    (hTwo : TwoStageBound (I.vocab (I.constructionsAt K₀ .remote reg sentEvidenceAt (statementFiles statementTable)
      record₀ recordDigest)) .remote)
    (hP : I.base.PouwSound .remote) :
    (I.vocab (I.constructionsAt K₀ .remote reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SoundnessAt .remote :=
  remote_soundness_executionPhase I K₀ reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _) (executionPhaseAt_form I .remote hW formAccepts formApart hR)
    (executionPhaseReads_members I .remote formAccepts hR) hTwo hP

end Verity.Sampled.MemberForm
