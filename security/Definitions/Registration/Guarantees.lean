module

public import Definitions.Registration.View

/-! `RegistrationSound`, what registration establishes. -/

@[expose] public section

namespace Verity.Guarantees

open Verity

/-- **What registration establishes.** A valid log's view holds the registered facts the theorems read, and each entry
names the head of the log before it. -/
def RegistrationSound : Prop :=
  ∀ (D : Registration.Digests) (A : Registration.Auditor) (c : Registration.Composite) (log : Registration),
    Registration.Valid D A c log → (Registration.view D c log).Holds D A c ∧ Registration.Chained D c log

end Verity.Guarantees
