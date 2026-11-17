module

public import Properties.Pous.EndToEnd

/-! The honest server's non-vacuity lemmas for PoUS end to end in the ring model; neither is a property or a
witness, and no record pins them. -/

@[expose] public section

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties

namespace Pous.EndToEnd

open scoped ENNReal

def HonestPassesRing : Prop :=
  S < FULL ∧ ∃ (A₁ : Preprocessor sch FULL) (A₂ : SeqResponderOn sch FULL 0 0 0),
    ∀ W, pr (seqAcceptsOn sch 105 W A₁ A₂) = 1

def HonestPassesAccepts : Prop :=
  ∃ A : Server FULL 0 0 0, ∀ session : ByteArray, session.size = 32 → ∀ W : Bits sch.n,
    pr (Passes session W A) = 1

end Pous.EndToEnd
