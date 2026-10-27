import Definitions.Claims.Sampling

/-!
# The sampling claims are stated (trusted)

`SamplingClaimsStated` reads every claim of `Definitions.Claims.Sampling`, the shared claims they instantiate, and the
named assumptions and trusted base they use, so the Lean check records each one's text (and every definition it reads)
under `reads`: a changed claim is a changed record, which is reported when it lands. It says nothing about the
verifier. A claim leaves it when a guarantee states that claim at the deployed sampled verifier, and the guarantee's own
record holds the text from then on.
-/

namespace Flock.Guarantees

/-- **The sampling claims are stated**: the tuple of the sampling claims (`KS`, `KSCount`, `Binding`, `Universal`,
`FaultyBoundary`, `Complete`, `ScopeMatches`, `ZK`, `Bits`, `Efficient`, `Nontrivial`), the claims they add (`KSLinOn`,
`BitsLin`, `ExtractOwnLe`) and the shared ones they instantiate (`ZKmalRx`, `Flock.ListWithin`, `Complete`,
`ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `Nontrivial`), the named assumptions (`StrictBirthday`,
`ExpectedCR`) and `trustedBase`, equal to itself. -/
def SamplingClaimsStated : Prop :=
  let claims := (@Claims.Sampling.KS, @Claims.Sampling.KSCount, @Claims.Sampling.Binding, @Claims.Sampling.Universal,
    Claims.Sampling.FaultyBoundary, @Claims.Sampling.Complete, @Claims.Sampling.ScopeMatches, @Claims.Sampling.ZK,
    @Claims.Sampling.Bits, @Claims.Sampling.Efficient, @Claims.Sampling.Nontrivial,
    @Claims.Sampling.KSLinOn, @Claims.Sampling.BitsLin, @Claims.Sampling.ExtractOwnLe,
    @Claims.ZKmalRx, @Claims.Flock.ListWithin, @Claims.Complete, @Claims.ScopeMatches,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
