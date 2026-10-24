import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.ZK.CoinLayer
import Proofs.Flock.Claims.ZK.Split

/-!
# C-Flock's masked knowledge soundness and zero knowledge at the deployed record (trusted)

`MaskedKS` and `MaskedZK` are `def G : Prop`s in the form of `Flock.Guarantees.EndToEnd`: two claims of
`Definitions.Claims`, `Claims.Flock.KS` and `Claims.Flock.ZK`, at the deployed masked record of the deployed relation
(`Flock.Readable.relation`), with SHA-512 as the oracle (`FlockDeploy.sha512`). Their proofs are
`Flock.SecurityProofs.MaskedKS : Flock.Guarantees.MaskedKS` and
`Flock.SecurityProofs.MaskedZK : Flock.Guarantees.MaskedZK`. `MaskedZK` holds `Claims.Flock.ZK`'s text in the lock in
place of `Flock.Guarantees.ClaimsStated`.

C-Flock's deployed records (`FlockDeploy.deployZk`, `Claims.ZKMal.zkDeployed` at the deployed plan) are still in its
proofs package, so this module imports them from `Proofs.Flock.Claims`, and `Models.lean`, which the `security` package
builds without `security_proofs`, does not import it: `Proofs.Flock.Masked` does.
-/

namespace Flock.Guarantees

/-- **Masked knowledge soundness (G1).** `Claims.Flock.KS` at SHA-512 at the untrimmed masked record of the deployed
relation (`FlockDeploy.deployZk Flock.Readable.relation`): at the record's extractor and every decoder that meets the
list-decoding spec, for every statement and every prover, the probability that the verifier accepts and the extractor
recovers no witness under the relation is at most `κ(x) + √(γ(x)·Pr[B outputs a collision])`, at the record's `κ` and
`γ`, with `B` the claim's collision finder built from the prover. It takes no hypothesis. -/
def MaskedKS : Prop :=
  Claims.Flock.KS FlockDeploy.sha512 (FlockDeploy.deployZk _root_.Flock.Readable.relation)

open Claims.ZKMal.Masked FlockSoundness.Discharge in
open Classical in
/-- **Masked zero knowledge (G2).** `Claims.Flock.ZK` at SHA-512 at the deployed masked record
(`Claims.ZKMal.zkDeployed`: the deployed coin layer `coin-tree/hm96-sha512/v2`, the deployed relation, and the deployed
plan's tables: the prover of levels ≥ 1 `PE`, the round plan of `czExe`, `encExe` stopped where a refusal is seen
(`encStop`) and `finExe`, and the executable's tape plan `UE`), at every rep rounds' coin counts `nr` and every
repetition count `0 < t`.
Against every malicious verifier, the record's simulator's view is within `ε_zk(x) + γ·Pr[fork]` of the real one, where
for a statement `x` in scope (`Claims.Flock.Scope` of the masked record)
$$\varepsilon_{zk}(x) = 8\,n_L(x)\,\delta_1 + 8\,t\,n_L(x)\,\delta_1 + 2 \cdot 3^{-t},$$
with `n_L(x)` the deployed plan's leaf count at `x` (`nL PE x`) and `δ₁` the hm96 salted-leaf hiding gap at the pinned
key (`FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash`); outside scope `ε_zk` is `1`. The fork term's factor
is `γ = 4·t·3^t`. It takes no other hypothesis. -/
def MaskedZK : Prop :=
  ∀ (nr : Claims.Msg → ℕ → ℕ) (t : ℕ), 0 < t →
    Claims.Flock.ZK FlockDeploy.sha512 (Claims.ZKMal.zkDeployed _root_.Flock.Readable.relation nr
        (tableData PE (RoundPlan.stop (czExe PE) (encStop (encExe PE)) (finExe PE)) UE) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash +
          8 * t * nL PE x * FlockSoundness.CROnly.hidingGap ZkView.My ZkView.saltHash + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t

end Flock.Guarantees
