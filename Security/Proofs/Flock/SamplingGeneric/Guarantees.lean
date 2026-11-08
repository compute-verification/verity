module

public import Proofs.Flock.SamplingGeneric.TwoAt

/-!
# The deployed two-stage sampled verifier's guarantee

`Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo` is `SamplingDeploy.ksTwo_atOrder`: the two-stage
sampled verifier at its round order is knowledge sound at SHA-512 on the challenger's bytes uniform at each of its two
calls and seven facts of its code. Its witnesses are `Flock.SecurityProofs.SampledKSTwo_honest` and
`Claims.SamplingGeneric.compilerKSTwo_nontrivial`.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**, at its round order (`codeTwoAt`): on
the challenger's bytes uniform at each of its two calls (`CallsUniform`, `UniformRandomBytes`), and the code's facts
(the registration is the statement's two-stage design, the verifier refuses outside the two-stage scope, an accepted
run's replay sample has at most `cap` proof blocks, the interior roots don't read the second stage's coins, each
checked unit's session runs Flock on its rounds within its numbers and opens the unit's proof block on its replay
block), `CompiledTwo.KS` holds at Flock's masked record over the readable relation and SHA-512, its outer wait's count
the statement's footprint bits and its inner wait's `cap`: `KSLinOn` at the two-stage relation, the extractor's errors
and the binding finder `Bext`. -/
def SampledKSTwo : Prop :=
  ∀ (c : SamplingDeploy.Code) (reg : Msg → Option SamplingDeploy.RegTwo)
    (two : Sampling.AuditStatement → Option Sampling.Design.Two) (L₁ L₂ : Msg → ℕ)
    (j : (x : Msg) → SamplingDeploy.Order (c.V.k x))
    (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256)
    (b₂ : (x : Msg) → FlockDeploy.Coin → Fin (L₂ x) → Fin 256) (iroots : Msg → Seen → ℕ → List Msg) (cap : Msg → ℕ),
    SamplingDeploy.CallsUniform L₁ b₁ → SamplingDeploy.CallsUniform L₂ b₂ →
    SamplingDeploy.RegMatchesTwo (c.atOrder j) reg two →
    (c.atOrder j).RefusesTwo FlockDeploy.sha512 two →
    (c.atOrder j).ReplayCap FlockDeploy.sha512 two
      (SamplingDeploy.replayTwo (c.atOrder j) reg L₁ (SamplingDeploy.bytesFirst c L₁ j b₁)) cap →
    (c.atOrder j).InteriorFixed FlockDeploy.sha512 iroots →
    Sampling.SessionRuns FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) →
    SamplingDeploy.SessionsIn (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) →
    SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) two iroots →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) iroots two
      (SamplingDeploy.replayTwo (c.atOrder j) reg L₁ (SamplingDeploy.bytesFirst c L₁ j b₁)) cap).KS FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ j b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_atOrder c reg two L₁ L₂ j b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens
