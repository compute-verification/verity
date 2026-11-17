module

public import Definitions.Poub.Guarantees.Bandwidth

/-! The bandwidth audit's honest device, which shows `AuditRO`'s event is not vacuous; it is no guarantee or
witness, and no record pins it. -/

@[expose] public section

namespace Bandwidth.Guarantees

def HonestPasses : Prop :=
  ∀ N b h : ℕ, ∃ A : Device N b h 0 0 0 N,
    ∀ (E : Store N b) (O : Oracle N b h) (s : ℕ) (I : Fin s → Fin N), Accepts A E O I ∧ A.ingress E O = N * b

end Bandwidth.Guarantees
