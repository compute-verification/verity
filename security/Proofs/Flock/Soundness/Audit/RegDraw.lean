module

public import Proofs.Flock.Soundness.Audit.OneStage

/-! Audits whose sample depends on the registration, with the outcome probabilities of the fixed-law audit at the registration's sampling design. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game

variable {n : ℕ}

def auditReg {Reg : Type} (L : Reg → SamplingDesign n) (session : Finset (Fin n) → Reg → Game Bool) :
    Game (Bool × Finset (Fin n)) :=
  .send Reg fun R => .coin (L R).Ω fun ω => (session ((L R).draw ω) R).map fun ok => (ok, (L R).draw ω)

def atReg {Reg : Type} {L : Reg → SamplingDesign n} {session : Finset (Fin n) → Reg → Game Bool}
    (σ : Strategy (auditReg L session)) : Strategy (audit (L σ.1) Reg session) :=
  ⟨σ.1, σ.2⟩

theorem prob_auditReg {Reg : Type} (L : Reg → SamplingDesign n) (session : Finset (Fin n) → Reg → Game Bool)
    (p : Bool × Finset (Fin n) → Prop) (σ : Strategy (auditReg L session)) :
    prob p (auditReg L session) σ = prob p (audit (L σ.1) Reg session) (atReg σ) := rfl

end FlockSoundness.Audit
