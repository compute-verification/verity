import Proofs.Flock.Claims.Recursion.DeployZK

/-!
# Probe: what the recursion's claims say at `deployRec` is what its parameters admit

`deployRec IV ivScope b` takes the inner verifier `IV`, its scope check `ivScope` and what the deployment brings `b` as
parameters. Its scope check reads `ivScope` (`RecScope.scopeCheckOf`): at an `ivScope` that refuses everything, the
record's scope check refuses every statement (`scopeCheck_deployRec_dead`), so a claim at `deployRec` whose antecedent
is a statement the scope check takes has nothing to speak of there. `zkPriv_deployRec` is the other such case: `ZKPriv`
holds at every `deployRec`, as no statement is in `RPriv`.
-/

namespace RecDeploy

/-- **Probe: at an inner scope check that refuses everything, `deployRec`'s scope check refuses every statement.** -/
theorem scopeCheck_deployRec_dead (IV : Claims.Verifier FlockDeploy.Coin) (b : Brings) (x : Claims.Msg) :
    (deployRec IV (fun _ => false) b).scopeCheck x = false := by
  show Flock.RecScope.scopeCheckOf (fun _ => false) decodeR x = false
  unfold Flock.RecScope.scopeCheckOf
  split <;> simp

end RecDeploy
