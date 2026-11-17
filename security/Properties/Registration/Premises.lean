module

public import RegistrationLog
public import Definitions.Core.Compiler

/-! What the registration log's code property takes (`Caps`). -/

@[expose] public section

namespace Properties.Registration

/-- **The compilers `B` names carry the log's caps**: each builder the log `l` registers names, by its digest, a
compiler whose advice cap is the builder's public cap. The check reads no compiler, so this scopes the code's
property. -/
def Caps (B : String → Verity.Compiler) (l : Core.Registration.Log) : Prop :=
  ∀ p items, Core.Registration.Entry.information p items ∈ l → ∀ b, Core.Registration.Item.builder b ∈ items →
    (B b.digest).ℓa = b.cap

end Properties.Registration
