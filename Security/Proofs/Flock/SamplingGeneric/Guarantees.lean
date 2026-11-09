module

public import Proofs.Flock.SamplingGeneric.TwoAt
public import Proofs.Flock.SamplingGeneric.TwoCount
public import Proofs.Flock.SpotFlock.KS
public import Proofs.Flock.SpotFlock.Frame

/-!
# The deployed two-stage sampled verifier's guarantee

`Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo` is `SamplingDeploy.ksTwo_atOrder`: the two-stage
sampled verifier at its round order is knowledge sound at SHA-512 on the challenger's bytes uniform at each of its two
calls and six facts of its code. Its witnesses are `Flock.SecurityProofs.SampledKSTwo_honest` and
`Claims.SamplingGeneric.compilerKSTwo_nontrivial`.

`Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo` is the same at SPOT[Flock]'s code (`SpotFlock.code`),
whose code facts, coins and sessions' runs are proved (`SpotFlock.ksTwo_spot`), with the sessions' openings from two
facts of a layout (`SpotFlock.sessionOpensTwo_of_frameIs`): it embeds the statement (`SpotFlock.Embeds`) and the
frame opens as it says (`SpotFlock.FrameIs`). Its honest instance needs a built two-stage layout and frame, which
nothing builds yet.

`Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo` is `CompiledTwo.Exfil` at the same code and layout
facts, by `CompiledTwo.exfil_of_ks` on `SpotKSTwo`: what an accepted run serves is in the influence set at `δ`, at
`SpotKSTwo`'s errors, and that set has at most `Σ_{T admissible} 2^{b(T)}` members. Its nontrivial witness is
`Claims.SamplingGeneric.compilerExfilTwo_nontrivial`; its honest instance waits on the same layout and frame.
-/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**, at its round order (`codeTwoAt`): on
the challenger's bytes uniform at each of its two calls (`CallsUniform`, `UniformRandomBytes`), and the code's facts
(the registration is the statement's two-stage design, the verifier refuses outside the two-stage scope, the interior
roots don't read the second stage's coins, each checked unit's session runs Flock on its rounds within its numbers and
opens the unit's proof block on its replay block), `CompiledTwo.KS` holds at Flock's masked record over the readable
relation and SHA-512: `KSLinOn` at the two-stage relation, the extractor's errors and the binding finder `Bext`. -/
def SampledKSTwo : Prop :=
  ∀ (c : SamplingDeploy.Code) (reg : Msg → Option SamplingDeploy.RegTwo)
    (two : Sampling.AuditStatement → Option Sampling.Design.Two) (L₁ L₂ : Msg → ℕ)
    (j : (x : Msg) → SamplingDeploy.Order (c.V.k x))
    (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256)
    (b₂ : (x : Msg) → FlockDeploy.Coin → Fin (L₂ x) → Fin 256) (iroots : Msg → Seen → ℕ → List Msg),
    SamplingDeploy.CallsUniform L₁ b₁ → SamplingDeploy.CallsUniform L₂ b₂ →
    SamplingDeploy.RegMatchesTwo (c.atOrder j) reg two →
    (c.atOrder j).RefusesTwo FlockDeploy.sha512 two →
    (c.atOrder j).InteriorFixed FlockDeploy.sha512 iroots →
    Sampling.SessionRuns FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) →
    SamplingDeploy.SessionsIn (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) →
    SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) two iroots →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) iroots two
      (SamplingDeploy.replayTwo (c.atOrder j) reg L₁ (SamplingDeploy.bytesFirst c L₁ j b₁))).KS FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at SHA-512 at a layout its statement embeds and its frame
opens**: at a registration `h` and a session layout `L` at whose wires the session statement holds each checked unit's
gates (`SpotFlock.Embeds`) and at which the registration's frame opens the session's rows (`SpotFlock.FrameIs`),
`CompiledTwo.KS` holds at Flock's masked record over the readable relation and SHA-512, at SPOT[Flock]'s two-stage code
(`SpotFlock.code`), the interior registrations' roots (`SpotFlock.iroots`), the two-stage design (`SpotFlock.two`) and
the replay sample of record (`SpotFlock.rs`). -/
def SpotKSTwo : Prop :=
  ∀ (h : SpotFlock.Held) (L : SpotFlock.Layout), SpotFlock.Embeds h L → SpotFlock.FrameIs h L →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h)).KS FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds what an accepted run serves, at SHA-512, at a layout its statement embeds
and its frame opens**: at `SpotKSTwo`'s registration, layout, code, interior roots, design and replay sample,
`CompiledTwo.Exfil` holds at every `δ ≥ 0`: `Pr[acc ∧ the served outputs are outside the influence set at δ]` is at
most `KSLinOn`'s bound at the two-stage errors and the binding finder `Bext`, and the influence set at any input values
has at most `Σ_{T admissible} 2^{b(T)}` members. -/
def SpotExfilTwo : Prop :=
  ∀ (h : SpotFlock.Held) (L : SpotFlock.Layout), SpotFlock.Embeds h L → SpotFlock.FrameIs h L →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.code h)
      SpotFlock.iroots (SpotFlock.two h) (SpotFlock.rs h)).Exfil FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ j b₁ b₂ iroots hA₁ hA₂ hreg hr hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_atOrder c reg two L₁ L₂ j b₁ b₂ iroots hA₁ hA₂ hreg hr hi hruns hnum hopens

theorem Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo :=
  fun h L he hF => SpotFlock.ksTwo_spot h (SpotFlock.sessionOpensTwo_of_frameIs h L he hF)

theorem Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo :=
  fun h L he hF => Claims.SamplingGeneric.CompiledTwo.exfil_of_ks (Flock.SecurityProofs.SpotKSTwo h L he hF)
