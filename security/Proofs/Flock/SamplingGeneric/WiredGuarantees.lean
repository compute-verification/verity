module

public import Proofs.Flock.SamplingGeneric.Guarantees
public import Proofs.Flock.SpotFlock.WiredCode
public import Definitions.Claims.SamplingTwoRegX

/-!
# The two-stage guarantees at the connected program: statements

`Flock.Guarantees.SpotKSTwo` and `SpotExfilTwo` at SPOT[Flock]'s code over the connected program
(`SpotFlock.codeConnected`): the members' programs with every wired read sent to its writer's output, where the
side-by-side statements read programs none of whose gates reads another's. Statements only.

* `SpotKSTwoWired`, `SpotExfilTwoWired`: the two guarantees in the extractor's form (`CompiledTwo.KS`,
  `CompiledTwo.Exfil`), at the connected code, design and replay sample.
* `SpotKSRegTwoXWired`, `SpotExfilRegTwoXWired`: the same at the committed trace, read at the statement and fixed
  before the sample (`CompiledTwo.KSRegTwoX`, `CompiledTwo.ExfilRegTwoX`).
* `Claims.Sampling.influenceTwoAt`, `CompiledTwo.ExfilRegTwoX`: the influence set and the exfiltration bound of a
  trace, where `influenceTwo` and `CompiledTwo.Exfil` read a witness's openings.
-/

@[expose] public section

namespace Claims.Sampling

/-- The values a trace holds at the program's input gates, `none` at every other gate (`inputsOf` at a trace). -/
def traceInputs (s : AuditStatement) (val : ℕ → Option Bool) (g : ℕ) : Option Bool :=
  if s.prog.gates[g]? = some .input then val g else none

/-- **What a trace serves**: its values at the program's returned gates, a constant's its own (`Prog.read`), a gate
it holds no value at `false` (`served` at a trace). -/
def traceServed (s : AuditStatement) (val : ℕ → Option Bool) : Fin s.prog.outs.length → Bool :=
  fun i => (s.prog.read val s.prog.outs[i]).getD false

open Classical in
/-- **The influence set of traces** at `δ` and input values `a`: what every trace with inputs `a` serves whose
footprint is in `M̄_δ` at some interiors (`influenceTwo` at a trace). -/
noncomputable def influenceTwoAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) (δ : ℝ)
    (a : ℕ → Option Bool) : Finset (Fin s.prog.outs.length → Bool) :=
  Finset.univ.filter fun y => ∃ val : ℕ → Option Bool, traceInputs s val = a ∧
    (∃ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part val ι ∈ d.M Pt δ) ∧ traceServed s val = y

end Claims.Sampling

namespace Claims.SamplingGeneric

open Claims
open Claims.Sampling (AuditStatement Partition InScopeTwo footprintAt admissibleTwo bitsOf traceInputs traceServed
  influenceTwoAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The exfiltration bound at the committed trace**, read at the statement and fixed before the sample
(`Compiled.regValX`): the committed form holds (`KSRegTwoX`); on a statement in scope, at every prover and `δ ≥ 0`, a
run accepts while what that trace serves is outside the influence set at its own inputs with probability at most
`δ + ε_reg,2,X`; and the influence set at any input values has at most `Σ_{T admissible} 2^{b(T)}` members. -/
def ExfilRegTwoX (H : Msg → Msg) (L : CompiledTwo Coin A) : Prop :=
  L.KSRegTwoX H ∧
    (∀ x, L.Scope x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      ∀ s Pt d, L.parse x = some s → L.partition s = some Pt → L.two s = some d → ∀ δ : ℝ, 0 ≤ δ →
        Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
            traceServed s (L.regValX H P x) ∉ influenceTwoAt s Pt d δ (traceInputs s (L.regValX H P x))) ≤
          δ + L.εregTwoX H x P) ∧
    ∀ x s Pt d (a : ℕ → Option Bool), L.parse x = some s → L.partition s = some Pt → L.two s = some d →
      InScopeTwo Pt d s → ∀ δ : ℝ,
        (influenceTwoAt s Pt d δ a).card ≤ ∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T

end CompiledTwo

end Claims.SamplingGeneric

namespace Flock.Guarantees

open Claims

/-- **SPOT[Flock]'s two-stage verifier is knowledge sound at the connected program**: at every registration whose
frame is SPOT's own there (`SpotFlock.BuiltWired`) and where each gate's value has an address that holds at every
checked unit that needs it (`SpotFlock.AddressesWired`), both at the code's wiring map, `CompiledTwo.KS` holds at the
connected code (`SpotFlock.codeConnected`), its design (`SpotFlock.twoConnected`) and its replay sample, at Flock's
masked record and SHA-512. The statement's program is the members' with every wired read sent to its writer's
output; the verifier refuses every run at a registration whose wiring the proof does not cover
(`SpotFlock.wiredOk`). -/
def SpotKSTwoWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KS
      FlockDeploy.sha512

/-- **SPOT[Flock]'s two-stage verifier bounds exfiltration at the connected program**: `CompiledTwo.Exfil` at
`SpotKSTwoWired`'s code and premises: what an accepted run serves is in the influence set at `δ`, and that set has
at most `Σ_{T admissible} 2^{b(T)}` members, `b(T)` the bits the proof blocks of `T` commit at the connected
program. -/
def SpotExfilTwoWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).Exfil
      FlockDeploy.sha512

/-- **The committed form at the connected program**: `CompiledTwo.KSRegTwoX` at `SpotKSTwoWired`'s code and
premises, the trace the one read at the statement and fixed before the sample. -/
def SpotKSRegTwoXWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).KSRegTwoX
      FlockDeploy.sha512

/-- **The exfiltration bound at the committed trace, at the connected program**: `CompiledTwo.ExfilRegTwoX` at
`SpotKSTwoWired`'s code and premises. -/
def SpotExfilRegTwoXWired : Prop :=
  ∀ h : SpotFlock.Held, SpotFlock.BuiltWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    SpotFlock.AddressesWired h (SpotFlock.wireMap h) (SpotFlock.returned h) →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (SpotFlock.codeConnected h)
      SpotFlock.iroots (SpotFlock.twoConnected h) (SpotFlock.rsConnected h) (SpotFlock.cap h)).ExfilRegTwoX
      FlockDeploy.sha512

end Flock.Guarantees
