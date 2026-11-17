module

public import Definitions.Poub.Guarantees.RateAudit
public import Definitions.Poub.Guarantees.RateAuditPartial

/-! The rate audit's honest devices, which show its events are not vacuous; neither is a guarantee or a witness, and
no record pins them. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting ResourceAccounting.Rate
open Pous (Bits)

def RateHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.Prover N b h ρ 0 ρ N,
    ∀ (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.compute cTag cEnc E R O = N * cTag

def RatePartialHonest : Prop :=
  ∀ N b h ρ : ℕ, ∃ A : Rate.PartialDevice N b h ρ 0 ρ N,
    ∀ (cTag : ℕ) (cost : ℕ → ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) (s : ℕ) (I : Fin s → Fin N),
      A.Accepts E R O I ∧ A.footprint E R O = N * b ∧ A.ingress E R O = N * b ∧ A.completed E R O = 0 ∧
        A.compute cTag cost E R O = N * cTag

end ResourceAccounting.Guarantees
