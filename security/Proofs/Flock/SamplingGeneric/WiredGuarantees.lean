module

public import Proofs.Flock.SamplingGeneric.Guarantees
public import Proofs.Flock.SpotFlock.WiredCode
public import Definitions.Claims.SamplingTwoServedX

/-!
# The two-stage guarantees at the connected program: statements

`Flock.Guarantees.SpotKSTwo` and `SpotExfilTwo` at SPOT[Flock]'s code over the connected program
(`SpotFlock.codeConnected`): the members' programs with every wired read sent to its writer's output, where the
side-by-side statements read programs none of whose gates reads another's. Statements only.

Each is at a registration whose wiring is one the proof covers (`SpotFlock.wiredOk h`, a hypothesis: the verifier is
`SpotFlock.V h` itself and refuses no run for its wiring), whose frame is SPOT's own at the connected program
(`SpotFlock.BuiltWired`) and where each gate's value has an address that holds at every checked unit that needs it
(`SpotFlock.AddressesWired`).

* `SpotKSTwoWired`, `SpotExfilTwoWired`: the two guarantees in the extractor's form (`CompiledTwo.KS`,
  `CompiledTwo.Exfil`), at the connected code, design and replay sample.
* `SpotKSRegTwoXWired`, `SpotExfilRegTwoXWired`: the same at the committed trace, read at the statement and fixed
  before the sample (`CompiledTwo.KSRegTwoX`, `CompiledTwo.ExfilRegTwoX`).
-/

@[expose] public section

namespace Flock.Guarantees

open Claims

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at the connected program**: at every registration whose
wiring is one the proof covers (`SpotFlock.wiredOk`), whose frame is SPOT's own there (`SpotFlock.BuiltWired`) and
where each gate's value has an address that holds at every checked unit that needs it (`SpotFlock.AddressesWired`),
both at the code's wiring map, `CompiledTwo.KS` holds at the connected code (`SpotFlock.codeConnected`), its design
(`SpotFlock.twoConnected`) and its replay sample, at Flock's masked record and SHA-512. The statement's program is
the members' with every wired read sent to its writer's output. The verifier is `SpotFlock.V h`, which reads no
wire: at a registration that fails `SpotFlock.wiredOk` this says nothing. -/
def SpotKSTwoWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KS
      FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds exfiltration at the connected program**: `CompiledTwo.Exfil` at
`SpotKSTwoWired`'s code and premises: what an accepted run serves is in the influence set at `δ`, and that set has
at most `Σ_{T admissible} 2^{b(T)}` members, `b(T)` the bits the proof blocks of `T` commit at the connected
program. -/
def SpotExfilTwoWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).Exfil
      FlockDeploy.sha512

/-- **The committed form at the connected program**: `CompiledTwo.KSRegTwoX` at `SpotKSTwoWired`'s code and
premises, the trace the one read at the statement and fixed before the sample. -/
def SpotKSRegTwoXWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KSRegTwoX
      FlockDeploy.sha512

/-- **The exfiltration bound at the committed trace, at the connected program**: `CompiledTwo.ExfilRegTwoX` at
`SpotKSTwoWired`'s code and premises. -/
def SpotExfilRegTwoXWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.wiredOk h = true →
    SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).ExfilRegTwoX
      FlockDeploy.sha512

end Flock.Guarantees
