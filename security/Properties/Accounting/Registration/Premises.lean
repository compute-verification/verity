module

public import CoreProtocol.Registration
public import Definitions.Core.Compiler

/-!
# What the registration log's code property takes

`Caps`: the compilers a builder's digest names carry the caps the log registers. The code's builder holds a digest, the
format of its advice and a public cap, and no compiler, while the spec's builder holds the compiler itself; the property
of the code (`Properties.Registration.Checks`) reads each builder's compiler by its digest, and takes that the compiler
so named has the builder's cap.
-/

@[expose] public section

namespace Properties.Registration

/-- **The compilers `B` names carry the log's caps**: each builder an information entry of the log `l` registers names,
by its digest, the compiler `B b.digest`, whose advice cap (`Verity.Compiler.ℓa`) is the builder's public cap. The check
reads a builder's digest and cap and no compiler, so this is a scope of the code's property: the compiler a builder's
digest names is one with the cap the log registers for it. -/
def Caps (B : String → Verity.Compiler) (l : Core.Registration.Log) : Prop :=
  ∀ p items, Core.Registration.Entry.information p items ∈ l → ∀ b, Core.Registration.Item.builder b ∈ items →
    (B b.digest).ℓa = b.cap

end Properties.Registration
