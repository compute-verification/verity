import Definitions.Claims.Recursion

/-!
# The recursion's claims are stated (trusted)

`RecursionClaimsStated` reads every claim of `Definitions.Claims.Recursion`, the shared claims they instantiate, and the
named assumptions and trusted base they use, so the Lean check records each one's text (and every definition it reads)
under `reads`: a changed claim is a changed record, which is reported when it lands. It says nothing about the
compiled verifier. A claim leaves it when a guarantee states that claim at the deployed compiled verifier, and the
guarantee's own record holds the text from then on.

The recursion's KS (`KS`, `Bits`, `KSFlock`, `KSBound`, and the private variant's) is at its level
`Claims.Recursion.lam`, about 65 bits per SHA-512 call at the deployed shape (one inner table at `m = 35`, the deployed
bound record's weight at `IVStar` at `coverW`), below the 100-bit floor and the `2^-128` target under every
weight. `BitsLin2`'s level λ₂ (`lamLin2`) evaluates to at least 100 at the deployed records
(`RecDeploy.lamLin2_deployed_ge_dep`, about 165.8) as a number, and `KSLin2` at the deployment's dense statements
(`KSLin2On` at `RecDeploy.DenseAt`) is proved with no hypothesis at the deployed reader's record, the coin server's
(`Flock.Guarantees.RecursiveKS`); `RecDeploy.ksLin2_deployed_open`, at the earlier `finOpen` layout, still takes
`PartChecksBV`, `PartOpensB` and `DecompB`. Holo statements have a model of their own, and `BitsLin2` at the dense
statements follows from it (`bitsLin2On_of_ksLin2On`).
-/

namespace Flock.Guarantees

/-- **The recursion's claims are stated**: the tuple of the compiled verifier's claims (`KS`, `Bits`, `Efficient`,
`Complete`, `ScopeMatches`, `ZK`, `Nontrivial`), the private variant's (`KSPriv`, `ZKPriv`), zero knowledge with the
reference prover's public stop and the pieces it reads (`ZKStop`, `StopDeploy`, `ZkDeploy.prover`,
`StopDeploy.halted`, `SimStop`, `εzkStop`), the claims at Flock as the inner protocol (`KSFlock`, `BitsFlock`), the
inner protocol's KS and its KS against provers that may abort (`Inner.KS`, `Inner.KSAbort`), which the proof reads,
the readable scope (`Scope`, `InScope`) and the decoders the claims quantify over (`decOK`); the KS at the
representation V* binds (`KSBound`, `KSPrivBound`), the inner claim it rests on (`Bridge`, with its premises
`Bound.PlainKSAbort`, `Bound.Binds` and `Bound.WeightCovers`), the last message's binding as first stated
(`Bound.LastBinds`) and modulo what `E*` reads, with the last step reading no last coin (`Bound.LastBindsEC`,
`Bound.LastCoinFree`), which the recursion's proof reads; the shared claims they instantiate (`Claims.KS`,
`ZKmalRx`, `Flock.ListWithin`, `Complete`, `ScopeMatches`, `ExtractRunsLe`, `ExtractRunsExpLe`, `Nontrivial`), the
named assumptions
(`StrictBirthday`, `ExpectedCR`, `Hm96Hiding`) and `trustedBase`, equal to itself. -/
def RecursionClaimsStated : Prop :=
  let claims := (@Claims.Recursion.KS, @Claims.Recursion.Bits, @Claims.Recursion.Efficient,
    @Claims.Recursion.Complete, @Claims.Recursion.ScopeMatches, @Claims.Recursion.ZK, @Claims.Recursion.Nontrivial,
    @Claims.Recursion.KSPriv, @Claims.Recursion.ZKPriv, @Claims.Recursion.ZKStop, @Claims.Recursion.StopDeploy,
    @Claims.Recursion.ZkDeploy.prover, @Claims.Recursion.StopDeploy.halted, @Claims.Recursion.SimStop,
    @Claims.Recursion.εzkStop,
    @Claims.Recursion.KSFlock, @Claims.Recursion.BitsFlock,
    @Claims.Recursion.Inner.KS, @Claims.Recursion.Inner.KSAbort,
    @Claims.Recursion.Scope, Claims.Recursion.InScope, @Claims.Recursion.decOK,
    @Claims.Recursion.KSBound, @Claims.Recursion.KSPrivBound, @Claims.Recursion.Bridge,
    @Claims.Recursion.Bound.PlainKSAbort, @Claims.Recursion.Bound.LastBinds, @Claims.Recursion.Bound.LastBindsEC,
    @Claims.Recursion.Bound.LastCoinFree,
    @Claims.KS, @Claims.ZKmalRx, @Claims.Flock.ListWithin, @Claims.Complete, @Claims.ScopeMatches,
    @Claims.ExtractRunsLe, @Claims.ExtractRunsExpLe, @Claims.Nontrivial,
    @Claims.StrictBirthday, @Claims.ExpectedCR, @Claims.Hm96Hiding, Claims.trustedBase)
  claims = claims

end Flock.Guarantees
