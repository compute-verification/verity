module

public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SpotFlock.WiredKS

/-! The deployed two-stage sampled verifier's guarantees (`SampledKSTwo`, `SpotKSTwo`, `SpotExfilTwo`); `SpotKSTwo` rests on `SpotFlock.AddressesConnected`, which nothing proves, and speaks only of the outputs a held link names. -/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**, at its stage order (`codeTwoStage`),
the second stage reading the coins of `R x` rounds: on the challenger's bytes uniform at each of its two calls
(`CallsUniform` on the first stage's coin, `StageUniform` on the second stage's coins, `UniformRandomBytes`), and the
code's facts (the registration is the statement's two-stage design, the verifier refuses outside the two-stage scope,
an accepted run's replay sample has at most `cap` proof blocks, the interior roots don't read the second stage's
coins, each checked unit's session runs Flock on its rounds within its numbers and opens the unit's proof block on its
replay block), `CompiledTwo.KS` holds at Flock's masked record over the readable relation and SHA-512, its outer wait's
count the statement's footprint bits and its inner wait's `cap`: `KSLinOn` at the two-stage relation, the extractor's
errors and the binding finder `Bext`. -/
def SampledKSTwo : Prop :=
  ∀ (c : SamplingDeploy.Code) (reg : Msg → Option SamplingDeploy.RegTwo)
    (two : Sampling.AuditStatement → Option Sampling.Design.Two) (L₁ L₂ R : Msg → ℕ)
    (s : (x : Msg) → SamplingDeploy.StageOrder (c.V.k x) (R x))
    (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256)
    (b₂ : (x : Msg) → (Fin (R x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256) (iroots : Msg → Seen → ℕ → List Msg)
    (cap : Msg → ℕ),
    SamplingDeploy.CallsUniform L₁ b₁ → SamplingDeploy.StageUniform R L₂ b₂ →
    SamplingDeploy.RegMatchesTwo (c.atOrder (SamplingDeploy.orderOf s)) reg two →
    (c.atOrder (SamplingDeploy.orderOf s)).RefusesTwo FlockDeploy.sha512 two →
    (c.atOrder (SamplingDeploy.orderOf s)).ReplayCap FlockDeploy.sha512 two
      (SamplingDeploy.replayTwo (c.atOrder (SamplingDeploy.orderOf s)) reg L₁
        (SamplingDeploy.bytesFirst c L₁ (SamplingDeploy.orderOf s) b₁)) cap →
    (c.atOrder (SamplingDeploy.orderOf s)).InteriorFixed FlockDeploy.sha512 iroots →
    Sampling.SessionRuns FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoStage c reg L₁ L₂ b₁ R s b₂)) →
    SamplingDeploy.SessionsIn (SamplingDeploy.codeTwoStage c reg L₁ L₂ b₁ R s b₂) →
    SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoStage c reg L₁ L₂ b₁ R s b₂)) two
      iroots →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      (SamplingDeploy.codeTwoStage c reg L₁ L₂ b₁ R s b₂) iroots two
      (SamplingDeploy.replayTwo (c.atOrder (SamplingDeploy.orderOf s)) reg L₁
        (SamplingDeploy.bytesFirst c L₁ (SamplingDeploy.orderOf s) b₁)) cap).KS FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at SHA-512 at the connected program**: at a held code `h`
whose frame is SPOT's own at the connected program (`SpotFlock.BuiltWired`) and at which each gate's value has one
address there that holds at every checked unit that needs it (`SpotFlock.AddressesConnected`), `CompiledTwo.KS` holds
at Flock's masked record over the readable relation and SHA-512, at SPOT[Flock]'s two-stage code at the connected
program (`SpotFlock.codeConnected`: the members' programs wired by `SpotFlock.wireMap`, returning
`SpotFlock.returned`), the interior registrations' roots (`SpotFlock.iroots`), the two-stage design there
(`SpotFlock.twoConnected`) and the replay sample of record (`SpotFlock.rsConnected`), its inner wait's count the
replay cap (`SpotFlock.cap`). The design evaluates only at a held code whose registered wiring passes its check
(`SpotFlock.wiringOk`), so a held code that fails it has no accepted run. -/
def SpotKSTwo : Prop :=
  ∀ h : SpotFlock.Held,
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) → SpotFlock.AddressesConnected h →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KS FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds what an accepted run serves, at SHA-512, at the connected program**: at
`SpotKSTwo`'s held code, premises, code, interior roots, design, replay sample and replay cap (`SpotFlock.cap`),
`CompiledTwo.Exfil` holds at every `δ ≥ 0`: `Pr[acc ∧ the served outputs are outside the influence set at δ]` is at
most `KSLinOn`'s bound at the two-stage errors and the binding finder `Bext`, and the influence set at any input
values has at most `Σ_{T admissible} 2^{b(T)}` members. The served outputs are the extractor's witness's openings,
under the statement's roots, at the connected program's returned gates, the written bits at an output port a link
names (`SpotFlock.returned`). An output that leaves and that no held link names is outside the bound, and at a held
code with no link nothing is returned and the bound is on nothing. -/
def SpotExfilTwo : Prop :=
  ∀ h : SpotFlock.Held,
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) → SpotFlock.AddressesConnected h →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).Exfil FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ R s b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_stage c reg two L₁ L₂ b₁ R s b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns.to hnum hopens

theorem Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo :=
  fun h hB hA => SpotFlock.ksTwo_connected h hB hA

theorem Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo :=
  fun h hB hA => SpotFlock.exfilTwo_connected h hB hA
