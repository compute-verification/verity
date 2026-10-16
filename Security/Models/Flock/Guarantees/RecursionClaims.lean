import Definitions.Claims.Recursion

/-!
# The recursion's claims are stated (trusted)

`RecursionClaimsStated` reads every claim of `Definitions.Claims.Recursion`, so the Lean check records each one's text
(and every definition it reads) under `reads`: a changed claim is a changed record, which reaches Daniel when it lands.
It says nothing about the compiled verifier. A claim leaves it when a guarantee states that claim at the deployed
compiled verifier, and the guarantee's own record holds the text from then on.
-/

namespace Flock.Guarantees

/-- **The recursion's claims are stated**: the tuple of the compiled verifier's claims (`KS`, `Bits`, `Efficient`,
`Complete`, `ScopeMatches`, `ZK`, `Nontrivial`), the private variant's (`KSPriv`, `ZKPriv`), the claims at Flock as the
inner protocol (`KSFlock`, `BitsFlock`) and the inner protocol's own `KS` that the proof reads, equal to itself. -/
def RecursionClaimsStated : Prop :=
  let claims := (@Claims.Recursion.KS, @Claims.Recursion.Bits, @Claims.Recursion.Efficient,
    @Claims.Recursion.Complete, @Claims.Recursion.ScopeMatches, @Claims.Recursion.ZK, @Claims.Recursion.Nontrivial,
    @Claims.Recursion.KSPriv, @Claims.Recursion.ZKPriv, @Claims.Recursion.KSFlock, @Claims.Recursion.BitsFlock,
    @Claims.Recursion.Inner.KS)
  claims = claims

end Flock.Guarantees
