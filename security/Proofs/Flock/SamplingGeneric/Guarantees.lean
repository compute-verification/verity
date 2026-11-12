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

`Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo` is the same at SPOT[Flock]'s code at the connected
program (`SpotFlock.codeConnected`). The verifier, the sessions, their rounds, the samples and the frame are
`SpotFlock.code`'s. The statement's program is the members' programs with every read sent through the code's own
wiring map (`SpotFlock.wireMap`), returning the written bits at an output port a link names (`SpotFlock.returned`).
The map sends a source bit to the gate on the same bit of its writer unit's output row (`SpotFlock.wireTarget`) only
where a wire names the bit's port and no link does, the port's rows place the unit, the wire's writer is a template's
name, no link names the writer's output port, and the writer has a written bit there at a gate the program does not
return; every other position stays where it is, so a port a link and a wire both name stays an input. A gate the map
moves stays in the program, an input gate that no gate reads and no proof block or replay block holds, and a session's
opening of a gate is revealed at the gate the map sends it to, under the roots the connected program opens that gate
under. The code facts, coins and sessions' runs are proved (`SpotFlock.ksTwo_connected`), with the sessions' openings
at the connected program (`SpotFlock.stmtOpens_connected`) from five premises on the held code, three of them
conditions that evaluation decides and the verifier does not run, one a fact nothing proves:

* `SpotFlock.BuiltWired`: the registration's frame is SPOT's own at the connected program. Every held code whose frame
  is set to it has it (`SpotFlock.builtWired_spot`).
* `SpotFlock.AddressesConnected`: each gate's value has one address at the connected program, fixed before the run,
  which holds at every accepted run's checked unit whose proof block needs the gate; it is asked only of a held code
  with `SpotFlock.SourcesNamed`. Nothing proves it at a registration: not that the checked unit's instance is one of
  its session's (`SpotFlock.CheckedRun.Counted`), not that the address's place reads the row its session holds the
  gate in (`SpotFlock.CheckedRun.RowAt`), and not that `SpotFlock.V` holds each needed gate's read to one entry, a
  linked gate's to the boundary record.
* `SpotFlock.SourcesNamed`: every source bit is at a port a link or a wire names. Nothing the verifier runs decides
  it, so the guarantee says nothing of an input the developer supplies or a row round 2 opens off the interior record.
* `SpotFlock.WiresFirst`: each wire's writer's part is before its reader's, so each wire's writer names a template.
  The deployed round 2 decides it (`SecondStage.wiresOk`); the verifier the guarantee is stated about (`SpotFlock.V`)
  does not run that check. With no wires held it holds, where `wiresOk` fails.
* `SpotFlock.WiresMove`: every wired source bit has a gate to go to. Nothing the verifier runs decides it. It enters
  no step of the proof: it is in the statement so that the connected program has no wired read left as a free input.

At a held code that fails one of them the guarantee says nothing, though `SpotFlock.V` still accepts its runs. The
guarantee speaks of what a replay block returns: its served outputs are the written bits at an output port a link
names, so at a design with no link it says nothing. What it covers rests on the links the held code holds: an output
that leaves and that no held link names is outside it. The held links are the auditor's copy, which nothing in the
statement holds to the replay block program's own links (`SecondStage.replayLinks`), and `SpotFlock.V` does not run
the design's check of them (`SecondStage.linksOk`). That check asks that each link is for a template one of the parts
is of, whose units its rows place, and not that an output that leaves has a link. Over the members' programs side by
side every written bit is returned, whatever the links; the connected program of a design with no wire still returns
only the linked ones. So at every design, with a wire or without, the guarantee speaks only of the linked outputs,
where the side-by-side statement speaks of every written bit. Its honest instance needs a registration that meets the
five premises, which nothing proves yet.

Over the members' programs side by side, `SpotFlock.ksTwo_spot` gives the same knowledge soundness at `SpotFlock.code`
from `SpotFlock.Built` and `SpotFlock.Addresses` (`SpotFlock.frameIs_built`, `SpotFlock.sessionOpensTwo_of_frameIs`).
No guarantee reads those.

`Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo` is `CompiledTwo.Exfil` at the same code and
premises, by `CompiledTwo.exfil_of_ks` on `SpotKSTwo` (`SpotFlock.exfilTwo_connected`): the served outputs are in the
influence set at `δ`, at `SpotKSTwo`'s errors, and that set has at most `Σ_{T admissible} 2^{b(T)}` members. The served
outputs are the extractor's witness's openings, under the statement's roots, at the gates the connected program
returns. Its nontrivial witness is
`Claims.SamplingGeneric.compilerExfilTwo_nontrivial`; its honest instance waits on the same registration.
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

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at SHA-512 at the connected program**: at a held code `h`
whose wires' writers come first (`SpotFlock.WiresFirst`), whose every wired source bit has a gate to go to
(`SpotFlock.WiresMove`), whose every source bit is at a port a link or a wire names (`SpotFlock.SourcesNamed`), whose
frame is SPOT's own at the connected program (`SpotFlock.BuiltWired`) and at which each gate's value has one address
there that holds at every checked unit that needs it (`SpotFlock.AddressesConnected`), `CompiledTwo.KS` holds at
Flock's masked record over the readable relation and SHA-512, at SPOT[Flock]'s two-stage code at the connected program
(`SpotFlock.codeConnected`: the members' programs wired by `SpotFlock.wireMap`, returning `SpotFlock.returned`), the
interior registrations' roots (`SpotFlock.iroots`), the two-stage design there (`SpotFlock.twoConnected`) and the
replay sample of record (`SpotFlock.rsConnected`), its inner wait's count the replay cap (`SpotFlock.cap`).
`SpotFlock.WiresMove` enters no step of the proof: it is in the statement so that the connected program has no wired
read left as a free input. -/
def SpotKSTwo : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.WiresFirst h → SpotFlock.WiresMove h → SpotFlock.SourcesNamed h →
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
  ∀ h : SpotFlock.Held, SpotFlock.WiresFirst h → SpotFlock.WiresMove h → SpotFlock.SourcesNamed h →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) → SpotFlock.AddressesConnected h →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).Exfil FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ R s b₁ b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_stage c reg two L₁ L₂ b₁ R s b₂ iroots cap hA₁ hA₂ hreg hr hcap hi hruns.to hnum hopens

theorem Flock.SecurityProofs.SpotKSTwo : Flock.Guarantees.SpotKSTwo :=
  fun h hF hM hS hB hA => SpotFlock.ksTwo_connected h hF hM hS hB hA

theorem Flock.SecurityProofs.SpotExfilTwo : Flock.Guarantees.SpotExfilTwo :=
  fun h hF hM hS hB hA => SpotFlock.exfilTwo_connected h hF hM hS hB hA
