module

public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.SpotFlock.Frame
public import Proofs.Flock.SpotFlock.FrameIsBuilt

/-!
# The deployed two-stage sampled verifier's guarantee

`Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo` is `SamplingDeploy.ksTwo_atOrder`: the two-stage
sampled verifier at its round order is knowledge sound at SHA-512 on the challenger's bytes uniform at each of its two
calls and seven facts of its code. Its witnesses are `Flock.SecurityProofs.SampledKSTwo_honest` and
`Claims.SamplingGeneric.compilerKSTwo_nontrivial`.

`Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo` is the same at SPOT[Flock]'s code (`SpotFlock.code`),
whose code facts, coins and sessions' runs are proved (`SpotFlock.ksTwo_spot`), with the sessions' openings at SPOT's
own layout (`SpotFlock.layoutFor`, `SpotFlock.sessionOpensTwo_of_frameIs`) from two facts: the registration's frame is
SPOT's own (`SpotFlock.Built`), and each gate's value has one address that holds at every checked unit that needs it
(`SpotFlock.Addresses`). Together they give that the frame opens as it says (`SpotFlock.frameIs_built`), and the layout
embeds the statement at every registration (`SpotFlock.embeds_layoutFor`). Every registration whose frame is set to
SPOT's has `Built` (`SpotFlock.built_spot`). Its honest instance needs `Addresses` at a two-stage registration, which
nothing proves yet; `Addresses` fails at a registration that commits a value one proof block writes and another reads in
two trees.

`Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo` is `CompiledTwo.Exfil` at the same code and layout
facts, by `CompiledTwo.exfil_of_ks` on `SpotKSTwo`: what an accepted run serves is in the influence set at `δ`, at
`SpotKSTwo`'s errors, and that set has at most `Σ_{T admissible} 2^{b(T)}` members. Its nontrivial witness is
`Claims.SamplingGeneric.compilerExfilTwo_nontrivial`; its honest instance waits on the same `Addresses`.
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

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at SHA-512 at its own frame, where each value has one
address**: at a registration `h` whose frame is SPOT's own (`SpotFlock.Built`), at which each gate's value has one
address that holds at every checked unit that needs it (`SpotFlock.Addresses`), `CompiledTwo.KS` holds at Flock's
masked record over the readable relation and SHA-512, at SPOT[Flock]'s two-stage code (`SpotFlock.code`), the interior
registrations' roots (`SpotFlock.iroots`), the two-stage design (`SpotFlock.two`) and the replay sample of record
(`SpotFlock.rs`), its inner wait's count the replay cap (`SpotFlock.cap`). -/
def SpotKSTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.Built h → SpotFlock.Addresses h →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h)).KS FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds what an accepted run serves, at SHA-512, at its own frame, where each
value has one address**: at `SpotKSTwo`'s registration, code, interior roots, design, replay sample and replay cap
(`SpotFlock.cap`), `CompiledTwo.Exfil` holds at every `δ ≥ 0`: `Pr[acc ∧ the served outputs are outside the influence
set at δ]` is at most `KSLinOn`'s bound at the two-stage errors and the binding finder `Bext`, and the influence set at
any input values has at most `Σ_{T admissible} 2^{b(T)}` members. -/
def SpotExfilTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.Built h → SpotFlock.Addresses h →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h) (SpotFlock.cap h)).Exfil FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ j b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_atOrder c reg two L₁ L₂ j b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns.to hnum hopens

theorem Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo :=
  fun h hB hS => SpotFlock.ksTwo_spot h
    (SpotFlock.sessionOpensTwo_of_frameIs h _ (SpotFlock.embeds_layoutFor h) (SpotFlock.frameIs_built h hB hS))

theorem Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo :=
  fun h hB hS => Claims.SamplingGeneric.CompiledTwo.exfil_of_ks (Flock.SecurityProofs.SpotKSTwo h hB hS)
