import Pous.Protocol.P2.Cost

/-!
# P2's open steps in the ring model (named assumptions)

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`: the primality of P2's
modulus (`PrimeP16448`).
-/

namespace Pous.P2.Assumptions

/-- **P2's registered modulus is prime (open in Lean)**: `p = 2^16448 − 21065`. Its primality certificate is ECPP, and
Lean checks none here. `p ≡ 3 mod 4` is proved (`PousP2Ring.p16448_mod_four`). -/
def PrimeP16448 : Prop := Nat.Prime (2 ^ 16448 - 21065)

end Pous.P2.Assumptions
