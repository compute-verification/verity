import Definitions.Claims

/-!
# The claims are stated (trusted)

`ClaimsStated` reads every claim of `Definitions.Claims`, the shared forms of `Definitions.Core.Interaction` they
instantiate, and their named assumptions and trusted base, so the Lean check
records each one's text (and every definition it reads) under `reads`: a changed claim is a changed record, which
is reported when it lands. It says nothing about the verifier. A claim leaves it when a guarantee states that claim
at the deployed verifier, and the guarantee's own record holds the text from then on. `Nontrivial` stays: it is
established by a test against recorded sessions (`check`'s `lean-agreement` job), not by a Lean proof.
-/

namespace Flock.Guarantees

/-- **The claims are stated**: the tuple of Flock's claims (`KS`, `Complete`, `ScopeMatches`, `Bits`, `Efficient`,
`Nontrivial`), the list-decoding spec they read (`Flock.ListWithin`), the shared claims they instantiate (`Claims.KS`,
`KSLin`, `Complete`, `ScopeMatches`, `ZKmalR`, `SimCost`, `Bits`, `ExtractRunsLe`, `ExtractRunsExpLe`, `ERepCost`,
`Nontrivial`), the named assumptions (`StrictBirthday`, `ExpectedCR`, `Hm96Hiding`, `JsonSpec`) and `trustedBase`,
equal to itself. `ZK` left it for `Flock.SecurityProofs.MaskedZK`, which states it at the deployed masked record, the
only kind of record `ZK` reads. `KS` stays: `Flock.SecurityProofs.MaskedKS` states it at the masked record only, and no
guarantee states it at the plain record (`FlockDeploy.deployPlain`, whose trim is the recursion's element-level
record). -/
def ClaimsStated : Prop :=
  let claims := (@Claims.Flock.KS, @Claims.Flock.Complete, @Claims.Flock.ScopeMatches,
    @Claims.Flock.Bits, @Claims.Flock.Efficient, @Claims.Flock.Nontrivial, @Claims.Flock.ListWithin,
    @Claims.KS, @Claims.KSLin, @Claims.Complete, @Claims.ScopeMatches, @Claims.ZKmalR, @Claims.SimCost, @Claims.Bits,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.ERepCost, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.Hm96Hiding, @Claims.JsonSpec, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
