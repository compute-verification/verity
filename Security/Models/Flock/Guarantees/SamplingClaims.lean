import Definitions.Claims.Sampling

/-!
# The sampling claims are stated (trusted)

`SamplingClaimsStated` reads every claim of `Definitions.Claims.Sampling`, the shared claims they instantiate, and the
named assumptions and trusted base they use, so the Lean check records each one's text (and every definition it reads)
under `reads`: a changed claim is a changed record, which reaches Daniel when it lands. It says nothing about the
verifier. A claim leaves it when a guarantee states that claim at the deployed sampled verifier, and the guarantee's own
record holds the text from then on.
-/

namespace Flock.Guarantees

/-- **The sampling claims are stated**: the tuple of the sampling claims (`KS`, `Complete`, `ScopeMatches`, `ZK`,
`Bits`, `Efficient`, `Nontrivial`), the claims they add (`BitsLin`, `ExtractOwnLe`, `ZKaux`) and the shared ones they
instantiate (`Claims.KSLin`, `Complete`, `ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `DecoderComplexity`,
`Nontrivial`, and Flock's `ListComplete`), the named assumptions (`StrictBirthday`, `ExpectedCR`, `CommitHides`) and
`trustedBase`, equal to itself. -/
def SamplingClaimsStated : Prop :=
  let claims := (@Claims.Sampling.KS, @Claims.Sampling.Complete, @Claims.Sampling.ScopeMatches, @Claims.Sampling.ZK,
    @Claims.Sampling.Bits, @Claims.Sampling.Efficient, @Claims.Sampling.Nontrivial,
    @Claims.Sampling.BitsLin, @Claims.Sampling.ExtractOwnLe, @Claims.Sampling.ZKaux,
    @Claims.KSLin, @Claims.Complete, @Claims.ScopeMatches, @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe,
    @Claims.DecoderComplexity, @Claims.Nontrivial, @Claims.Flock.ListComplete,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.CommitHides, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
