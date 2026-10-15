import Definitions.Claims

/-!
# The claims are stated (trusted)

`ClaimsStated` reads every claim of `Definitions.Claims`, and its named assumptions and trusted base, so the Lean check
records each one's text (and every definition it reads) under `reads`: a changed claim is a changed record, which
reaches Daniel when it lands. It says nothing about the verifier. A claim leaves it when a guarantee states that claim
at the deployed verifier, and the guarantee's own record holds the text from then on.
-/

namespace Flock.Guarantees

/-- **The claims are stated**: the tuple of Flock's claims (`KS`, `Complete`, `ScopeMatches`, `ZK`, `Bits`, `Efficient`,
`Nontrivial`), the shared claims they instantiate (`Claims.KS`, `KSLin`, `Complete`, `ScopeMatches`, `ZKmal`,
`SimCost`, `Bits`, `ExtractRunsLe`, `ExtractRunsExpLe`, `ERepCost`, `DecoderComplexity`, `Nontrivial`), the named
assumptions (`StrictBirthday`, `ExpectedCR`, `CommitHides`) and `trustedBase`, equal to itself. -/
def ClaimsStated : Prop :=
  let claims := (@Claims.Flock.KS, @Claims.Flock.Complete, @Claims.Flock.ScopeMatches, @Claims.Flock.ZK,
    @Claims.Flock.Bits, @Claims.Flock.Efficient, @Claims.Flock.Nontrivial,
    @Claims.KS, @Claims.KSLin, @Claims.Complete, @Claims.ScopeMatches, @Claims.ZKmal, @Claims.SimCost, @Claims.Bits,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.ERepCost, @Claims.DecoderComplexity, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.CommitHides, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
