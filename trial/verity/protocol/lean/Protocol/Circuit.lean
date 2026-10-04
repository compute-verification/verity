/-! What a circuit is and what it computes: accounting's definition, which `verity.circuits` implements. A placeholder
for the layout trial; the real statement is `packages/verity/lean/Verity/Protocol/Circuit.lean`. -/

namespace Protocol

structure Circuit where
  inputs : Nat
  gates : List (String × Nat × Nat)
  outputs : List Nat

end Protocol
