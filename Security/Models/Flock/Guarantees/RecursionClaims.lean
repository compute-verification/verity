import Definitions.Claims.Recursion

/-!
# The recursion's claims are stated (trusted)

`RecursionClaimsStated` reads every claim of `Definitions.Claims.Recursion`, the shared claims they instantiate, and the
named assumptions and trusted base they use, so the Lean check records each one's text (and every definition it reads)
under `reads`: a changed claim is a changed record, which reaches Daniel when it lands. It says nothing about the
compiled verifier. A claim leaves it when a guarantee states that claim at the deployed compiled verifier, and the
guarantee's own record holds the text from then on.
-/

namespace Flock.Guarantees

/-- **The recursion's claims are stated**: the tuple of the compiled verifier's claims (`KS`, `Bits`, `Efficient`,
`Complete`, `ScopeMatches`, `ZK`, `Nontrivial`), the private variant's (`KSPriv`, `ZKPriv`), the claims at Flock as the
inner protocol (`KSFlock`, `BitsFlock`), the inner protocol's KS and its KS against provers that may abort
(`Inner.KS`, `Inner.KSAbort`), which the proof reads, the readable scope (`Scope`, `InScope`) and the decoders the
claims quantify over (`decOK`); option (d)'s KS at the representation V* binds (`KSBound`, `KSPrivBound`), the inner
claim it rests on (`Bridge`, with its premises `Bound.PlainKSAbort`, `Bound.Binds` and `Bound.WeightCovers`) and the
last message's binding the recursion's proof reads (`Bound.LastBinds`); the shared claims they instantiate (`Claims.KS`, `ZKmalR`, `Flock.ListWithin`,
`Complete`, `ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `Nontrivial`), the named assumptions
(`StrictBirthday`, `ExpectedCR`, `Hm96Hiding`) and `trustedBase`, equal to itself. -/
def RecursionClaimsStated : Prop :=
  let claims := (@Claims.Recursion.KS, @Claims.Recursion.Bits, @Claims.Recursion.Efficient,
    @Claims.Recursion.Complete, @Claims.Recursion.ScopeMatches, @Claims.Recursion.ZK, @Claims.Recursion.Nontrivial,
    @Claims.Recursion.KSPriv, @Claims.Recursion.ZKPriv, @Claims.Recursion.KSFlock, @Claims.Recursion.BitsFlock,
    @Claims.Recursion.Inner.KS, @Claims.Recursion.Inner.KSAbort,
    @Claims.Recursion.Scope, Claims.Recursion.InScope, @Claims.Recursion.decOK,
    @Claims.Recursion.KSBound, @Claims.Recursion.KSPrivBound, @Claims.Recursion.Bridge,
    @Claims.Recursion.Bound.PlainKSAbort, @Claims.Recursion.Bound.LastBinds,
    @Claims.KS, @Claims.ZKmalR, @Claims.Flock.ListWithin, @Claims.Complete, @Claims.ScopeMatches,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.Hm96Hiding, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
