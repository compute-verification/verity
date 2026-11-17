module

public import Proofs.Properties.Verity.Sampled.MemberForm.ClaimReads
public import Proofs.Properties.Verity.Sampled.MemberForm.ExecutionPhasePasses

/-! SPOT's soundness at the per-member form's accepts, on two premises no refusal of the check enforces:
the window is the parameters' members on the registered files, and the registered files hold each member's file. -/

@[expose] public section

namespace Verity.Sampled.MemberForm

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem spotSound_form (I : ExecutionPhaseInstance d α) (K₀ : ExecutionPhaseConstructions I.base)
    (k : Verity.Registration.Composite) (reg : Option Claims.Msg → ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Claims.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hW : I.ExecutionPhaseRegistered) (hR : RegisteredHolds I.base) :
    (I.vocab (I.constructionsAt K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SpotSound k :=
  I.spotSound K₀ k reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _) (executionPhaseAt_form I k hW formAccepts formApart hR)
    (executionPhaseReads_members I k formAccepts hR)

theorem soundnessAt_form (I : ExecutionPhaseInstance d α) (K₀ : ExecutionPhaseConstructions I.base)
    (reg : Option Claims.Msg → ByteArray)
    (record₀ : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Verity.Exchange → Option Claims.Msg)
    (recordDigest : ExecutionPhase → ByteArray → Array ByteArray → ByteArray → Claims.Msg)
    (hW : I.ExecutionPhaseRegistered) (hR : RegisteredHolds I.base)
    (hP : I.base.PouwSound .remote)
    (hWi : (I.vocab (I.constructionsAt K₀ .remote reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).WiresSound .remote)
    (hG : GateARow) :
    (I.vocab (I.constructionsAt K₀ .remote reg sentEvidenceAt (statementFiles statementTable) record₀
      recordDigest)).SoundnessAt .remote :=
  remote_soundness_executionPhase I K₀ reg sentEvidenceAt (statementFiles statementTable) record₀ recordDigest
    (claimReads I _) (executionPhaseAt_form I .remote hW formAccepts formApart hR)
    (executionPhaseReads_members I .remote formAccepts hR) hP hWi hG

end Verity.Sampled.MemberForm
