module

public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SpotFlock.WiredKS

/-!
# The deployed two-stage sampled verifier's guarantee

`Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo` is `SamplingDeploy.ksTwo_stage`: the two-stage
sampled verifier at its stage order, its second stage reading the coins of `R x` rounds, is knowledge sound at SHA-512
on the challenger's bytes uniform at each of its two calls and seven facts of its code. A second stage of one round
(`SamplingDeploy.Order.stage`) is `SamplingDeploy.ksTwo_atOrder`'s code (`SamplingDeploy.codeTwoAt_eq_stage`). Its
witnesses are `Flock.SecurityProofs.SampledKSTwo_honest` and `Claims.SamplingGeneric.compilerKSTwo_nontrivial`.

`Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo` is the same at SPOT[Flock]'s code over the connected
program (`SpotFlock.codeConnected`): the members' programs with every wired read sent to its writer's output
(`SpotFlock.wireMap`), returning the gates the boundary record holds (`SpotFlock.returned`). Its code facts, coins and
sessions' runs are proved (`SpotFlock.ksTwo_connected`), with the sessions' openings from three facts of the
registration. Its wiring is one the proof covers (`SpotFlock.wiredOk`, decided on what the verifier holds): the
verifier is `SpotFlock.V h`, whose statement check reads no wire, so the statement says nothing at a registration that
fails it. Its frame is SPOT's own at the connected program (`SpotFlock.BuiltWired`), as every registration whose
frame is set to SPOT's has it (`SpotFlock.builtWired_spot`). And each gate's value has an address that holds at every
checked unit that needs it, a gate the wiring moves at the place and slot of the gate it is sent to
(`SpotFlock.AddressesWired`). Its honest instance needs `AddressesWired` at a two-stage registration, which nothing
proves yet: no registration is shown to have it.

`Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo` is `CompiledTwo.Exfil` at the same code and facts,
by `CompiledTwo.exfil_of_ks` on `SpotKSTwo`: what an accepted run serves is in the influence set at `δ`, at
`SpotKSTwo`'s errors, and that set has at most `Σ_{T admissible} 2^{b(T)}` members, `b(T)` the bits the proof blocks
of `T` commit at the connected program. Its nontrivial witness is
`Claims.SamplingGeneric.compilerExfilTwo_nontrivial`; its honest instance waits on the same `AddressesWired`.
-/

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

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at SHA-512 at the connected program**: at a registration
`h` whose wiring is one the proof covers (`SpotFlock.wiredOk`), whose frame is SPOT's own there
(`SpotFlock.BuiltWired`) and at which each gate's value has an address that holds at every checked unit that needs it
(`SpotFlock.AddressesWired`), both at the code's wiring map, `CompiledTwo.KS` holds at Flock's masked record over the
readable relation and SHA-512, at SPOT[Flock]'s two-stage code over the connected program
(`SpotFlock.codeConnected`), the interior registrations' roots (`SpotFlock.iroots`), the two-stage design
(`SpotFlock.twoConnected`) and the replay sample of record (`SpotFlock.rsConnected`), its inner wait's count the
replay cap (`SpotFlock.cap`). The statement's program is the members' with every wired read sent to its writer's
output. The verifier is `SpotFlock.V h`, which reads no wire: at a registration that fails `SpotFlock.wiredOk` this
says nothing. -/
def SpotKSTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KS
      FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds what an accepted run serves, at SHA-512, at the connected program**: at
`SpotKSTwo`'s registration, code, interior roots, design, replay sample and replay cap (`SpotFlock.cap`),
`CompiledTwo.Exfil` holds at every `δ ≥ 0`: `Pr[acc ∧ the served outputs are outside the influence set at δ]` is at
most `KSLinOn`'s bound at the two-stage errors and the binding finder `Bext`, and the influence set at any input
values has at most `Σ_{T admissible} 2^{b(T)}` members, `b(T)` the bits the proof blocks of `T` commit at the
connected program. -/
def SpotExfilTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).Exfil
      FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ R s b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_stage c reg two L₁ L₂ b₁ R s b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns.to hnum hopens

theorem Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo :=
  fun h hok hB hA => SpotFlock.ksTwo_connected h hok hB hA

theorem Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo :=
  fun h hok hB hA => SpotFlock.exfilTwo_connected h hok hB hA
