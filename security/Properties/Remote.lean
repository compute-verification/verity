module

public import CoreRemote
public import Definitions.Core
public import Properties.Composition
public import Properties.InferenceOnly.Desc
public import Properties.InferenceOnly.Theta
public import Properties.Spot
public import VStarRecursive.Caller

/-! Remote's acceptance and the inference-only policy: no statement takes an assumption, and none states that the
registered circuit is the registered compiler's output on committed advice. -/

@[expose] public section

namespace Properties.Remote

open Core Core.Composition Verity
open Lean (Json)
open Flock.Program (Prog)
open InferenceOnlyCheck.Properties (Parses Expands LaysOut Names Divides Meets ProvenanceHeld)
open InferenceOnlyCheck.Protocol (instanceOf trustedOf)

/-- The policy, met by the declaration `d` and its division `U` at `Γ` and `α`: every inference block of `U` has at most
`Γ` gates and reads at most `α` fresh input bits per gate (`Meets`), so `d` complies (`InferenceOnly.CompliesRate`); and `d`
with its constants registered (`RegisteredCircuit.constInputs`, as the model takes a constant) has a division that is
a `(Γ, ⌊αΓ⌋)`-partition (`Partition.IsCK`) when every gate of `d` is one bit wide, and a bit-level expansion with a
division that is a `(ΓW, ⌊αΓW⌋)`-partition when every gate is at most `W` bits wide. Neither clause assumes `d`
Boolean or of positive fan-in: registering its constants makes it so. Where `d` is Boolean, or of positive fan-in in
bits with every computed gate at least a bit wide, it has no constant, and the clauses give partitions of `d` and its
expansion themselves (`SecurityProofs.policyMet_old_clauses`). -/
def PolicyMet (d : RegisteredCircuit) {n : ℕ} (U : Partition d.D n) (Γ : ℕ) (α : ℝ) : Prop :=
  (∀ u, Meets d U Γ α u) ∧ InferenceOnly.CompliesRate Γ α d ∧
    ((∀ g, d.D.w g = 1) → ∃ (B : ℕ) (P : Partition d.constInputs.D B), P.IsCK d.constInputs Γ ⌊α * Γ⌋₊) ∧
    ∀ W : ℕ, (∀ g, d.D.w g ≤ W) →
      ∃ (B : ℕ) (P : Partition d.constInputs.expand.D B), P.IsCK d.constInputs.expand (Γ * W) ⌊α * (Γ * W : ℕ)⌋₊

/-- The parse `i` of the policy's instance carries the policy's own parameters `p`: its program, and `Γ` and
`α = num / den` as the parameters write them. -/
def VerifierOwn (p : InferenceOnlyCheck.Protocol.Params) (i : InferenceOnlyCheck.Instance) : Prop :=
  i.program = p.program ∧ InferenceOnlyCheck.natOf? p.gamma = some i.gamma ∧
    ∃ a b, p.alpha = .arr #[a, b] ∧ InferenceOnlyCheck.natOf? a = some i.num ∧ InferenceOnlyCheck.natOf? b = some i.den

/-- What the policy's case establishes, decoded as its parameters `p`, the run's record `c` and the evidence `e`: the
registered circuit decodes to `P`; the record registers `P` by its SHA-512; θ holds what the log's first record
registers; the instance the policy checked has an expansion gate by gate with a parse and a registered circuit `P`
lays out with the θ it names, whose inference blocks `d.𝓘` its owners are, which meet the policy; and on every such
expansion and parse, the parse carries the policy's parameters and every such registered circuit's inference blocks
meet the policy at its `Γ` and `α`. -/
def PolicyHeld (p : InferenceOnlyCheck.Protocol.Params) (c : InferenceOnlyCheck.Protocol.Commitment) (e : InferenceOnlyCheck.Protocol.Evidence) :
    Prop :=
  ∃ P : Prog, Flock.Program.decode p.program.toUTF8 = .ok P ∧
    InferenceOnlyCheck.Theta.strAt c.record "program" = some (Flock.hexOf P.sha512) ∧
    ProvenanceHeld (instanceOf p e) (trustedOf p c) ∧
    (∃ (j' : Json) (i : InferenceOnlyCheck.Instance) (d : RegisteredCircuit),
      Expands (instanceOf p e) j' ∧ Parses j' i ∧ LaysOut P d.D ∧ Names i.theta d.θ ∧ Divides i.owner d.𝓘 ∧
        VerifierOwn p i ∧ PolicyMet d d.𝓘 i.gamma ((i.num : ℝ) / i.den)) ∧
    ∀ (j' : Json) (i : InferenceOnlyCheck.Instance) (d : RegisteredCircuit),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner d.𝓘 →
        VerifierOwn p i ∧ PolicyMet d d.𝓘 i.gamma ((i.num : ℝ) / i.den)

/-- The policy's case `k` decodes under the policy's own decoders, and what it decodes to is `PolicyHeld`. -/
def CaseHeld (k : Case) : Prop :=
  ∃ p c e, (canonicalJson k.params.toUTF8 >>= InferenceOnlyCheck.Protocol.paramsOf) = .ok p ∧
    InferenceOnlyCheck.Protocol.commitmentOf k.commitment = .ok c ∧ InferenceOnlyCheck.Protocol.evidenceOf k.evidence = .ok e ∧
    PolicyHeld p c e

def CallerCheckMeans : Prop :=
  ∀ (rec : Json) (v : Spot.Docket.Verifier),
    VStar.Recursive.Caller.check rec v = #[] ↔ Properties.Spot.CallerHeld rec v

def CallerCheckSound : Prop :=
  ∀ (rec : Json) (v : Spot.Docket.Verifier),
    VStar.Recursive.Caller.check rec v = #[] → Properties.Spot.CallerHeld rec v

/-- SPOT's case `k` decodes under SPOT's own decoders (`VStar.Recursive.Sampled`, whose params name
the form) in the one-stage form, and the record it decodes to names, by its SHA-512, the program of one-stage v2's own
verifier (the v1 part of its params, `Spot.OneStageZk.Params.base`), which it verifies the sessions against
(`Properties.Spot.R2`). The record keeps the caller's facts against that verifier (`Properties.Spot.CallerHeld`): it names its day's declared program, its window names the stream, the
prover, the seed salt, the registered compiler and the pool's reset the verifier holds, it opens the pool the previous
window of record left, and no output value reads a port of a record the verifier's verdict log names refused. SPOT's own statement (`Properties.Spot.AcceptMeans`) holds none of them: they are what SPOT's caller hands the
verifier, so they are stated here, of the case the caller's check (`VStar.Recursive.Caller.check`) accepted. It has no
clause for the two-stage form, which the verifier refuses while `SecondStage.FORM` is false; switching it on needs the
two-stage clause first. -/
def StageHeld (k : Case) : Prop :=
  ∃ (p : Spot.OneStageZk.Params) (c : Spot.OneStage.Commitment),
    (canonicalJson k.params.toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = .ok (.inl p) ∧
    Spot.OneStage.commitmentOf k.commitment = .ok c ∧ Properties.Spot.R2 c.record p.base.verifier ∧
    Properties.Spot.CallerHeld c.record p.base.verifier

/-- What remote's acceptance establishes about its cases `ev`: they are the three subprotocols' cases in order, the
policy's `kI`, PoUW's period's `kW` and SPOT's `kS`; the policy's and SPOT's commitments decode,
each under its own subprotocol's decoder (SPOT's is the one-stage registration record's), to one digest `r`,
the run's record's;
every Commit record the
period's windows name, when they decode, has digest `r` (a period whose windows don't decode is the period's own
refusal, `PouwCheck.Period.check`, which this doesn't read); the policy's case is `CaseHeld`: the record names the
registered circuit by its SHA-512, θ holds what the log's first record registers, the instance checked has a parse, a
declaration the registered circuit lays out and a division, every parse carries the policy's `program`, `Γ` and `α`,
and every declaration the registered circuit lays out, with the declared division, meets the policy; and SPOT's case is `StageHeld`: it is in the one-stage form, its record names, by its SHA-512, the program of one-stage
v2's verifier, and keeps the caller's facts against that verifier. -/
def InferenceOnlyHeld (ev : List Case) : Prop :=
  ∃ (kI kW kS : Case) (r : ByteArray), ev = [kI, kW, kS] ∧
    Properties.Composition.digestOf InferenceOnlyCheck.Protocol.protocol kI.commitment = some r ∧
    (∀ ds, Core.Remote.executionPhaseRecords kW = some ds → ∀ d ∈ ds, d = r) ∧
    Properties.Composition.digestOf Spot.OneStage.protocol kS.commitment = some r ∧
    CaseHeld kI ∧ StageHeld kS

/-- **Remote's acceptance gives the inference-only policy on the registered circuit.** On every input remote's check
accepts, on any blobs, its cases are `InferenceOnlyHeld`. -/
def AcceptInferenceOnly : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof → InferenceOnlyHeld ev

/-- What remote's acceptance establishes in the form `Slowdown.Guarantees.SlowdownThreaded` reads, about its cases `ev`: they
are the three subprotocols' cases in order, the policy's case decodes to the parameters `p`, the run's record `cr` and
the evidence `e`, the record names the registered circuit `P` by its SHA-512, the instance the policy checked, expanded
gate by gate, has a parse carrying the policy's parameters and a declaration `P` lays out with its θ, which complies at
the policy's `Γ` and `α`, and on every such expansion and parse, the parse carries the policy's parameters and every
such declaration complies. -/
def CompliesHeld (ev : List Case) : Prop :=
  ∃ (kI kW kS : Case) (p : InferenceOnlyCheck.Protocol.Params) (cr : InferenceOnlyCheck.Protocol.Commitment)
    (e : InferenceOnlyCheck.Protocol.Evidence) (P : Prog), ev = [kI, kW, kS] ∧
    (canonicalJson kI.params.toUTF8 >>= InferenceOnlyCheck.Protocol.paramsOf) = .ok p ∧
    InferenceOnlyCheck.Protocol.commitmentOf kI.commitment = .ok cr ∧ InferenceOnlyCheck.Protocol.evidenceOf kI.evidence = .ok e ∧
    Flock.Program.decode p.program.toUTF8 = .ok P ∧
    InferenceOnlyCheck.Theta.strAt cr.record "program" = some (Flock.hexOf P.sha512) ∧
    (∃ (j' : Json) (i : InferenceOnlyCheck.Instance) (d : RegisteredCircuit),
      Expands (instanceOf p e) j' ∧ Parses j' i ∧ LaysOut P d.D ∧ Names i.theta d.θ ∧ VerifierOwn p i ∧
        InferenceOnly.CompliesRate i.gamma ((i.num : ℝ) / i.den) d) ∧
    ∀ (j' : Json) (i : InferenceOnlyCheck.Instance) (d : RegisteredCircuit),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ →
        VerifierOwn p i ∧ InferenceOnly.CompliesRate i.gamma ((i.num : ℝ) / i.den) d

/-- **Remote's acceptance gives `InferenceOnly.CompliesRate`**, the hypothesis `Slowdown.Guarantees.SlowdownThreaded` takes: on every input
remote's check accepts, on any blobs, its cases are `CompliesHeld`. -/
def AcceptComplies : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof → CompliesHeld ev

end Properties.Remote
