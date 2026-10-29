import CoreRemote
import Definitions.Core
import Properties.Composition
import Properties.NciDesc
import Properties.Nci.Baseline
import Properties.SampledProofs

/-!
# Remote's acceptance and the inference-only policy

`Core.Remote.protocol` (`verity/remote/v0`) is `Core.Composition.sameB` over three subprotocols, in the order of their
ids: the inference-only policy (`NciCheck.Protocol.protocol`), PoUW's window check
(`PouwCheck.protocol`) and sampled proofs' one-stage audit (`SampledProofs.OneStageZk.protocol`). Each one's commitment is
the run's registration record, and the composite rejects unless all three decode to one digest
(`Core.Composition.strays`). The policy's check (`NciCheck.Protocol.check`) refuses a record whose `program` is not the
registered circuit's SHA-512 (`names`), then runs
the check of record, `NciCheck.Baseline.evaluate`: the policy check at the size of the description
(`NciCheck.Desc.evaluate`), then θ's provenance against the registration log through the run's record.

* `AcceptInferenceOnly`: on every input remote's check accepts, the three subprotocols' records have one digest, the
  record the policy's case commits to names the registered circuit by its SHA-512, θ holds what the log's first record
  registers (`NciCheck.Properties.ProvenanceHeld`), and the instance the policy checked, expanded gate by gate
  (`NciCheck.Properties.Expands`), has a parse, a declaration the registered circuit lays out with θ and a division its
  owners are; on every such parse, `program`, `Γ` and `α = num / den` are the policy's parameters (`VerifierOwn`), and
  every such declaration and division meet the policy (`PolicyMet`): every inference block of `U` has at most `Γ` gates
  and reads at most `α`
  fresh input bits per gate, so the declaration complies (`Nci.Complies Γ α`), and it has a padded
  `(Γ, ⌊αΓ⌋)`-partition when Boolean, its bit-level expansion a padded `(ΓW, ⌊αΓW⌋)`-partition when its gates are 1 to
  `W` bits wide; and the record the one-stage audit's case commits to names, by its SHA-512, the program the audit
  verifies the sessions against (`StageHeld`). With one digest for the two records, the registered circuit the policy
  checked and the program the one-stage audit verifies have one SHA-512.
* `AcceptComplies`: its corollary in the form `Nci.Guarantees.DampenerRemote` reads, `Nci.Complies Γ α d` from
  acceptance, on a declaration acceptance exhibits and on every other.

`AcceptInferenceOnlyBy`, `AcceptCompliesBy` and `AcceptsAll` put another check in remote's place, so that the proofs
can show that no check accepting every input has either property.

`Γ` and `α = num / den` are the verifier's own: the parameters the composite names for the policy. No statement takes an
assumption: the digests are those the code computes and compares, so that PoUW and the one-stage audit read the policy's
record is up to a SHA-512 collision between two records the run carries, which the statements leave to their readers.

The registered circuit is the program the policy's parameters carry and the record names by its SHA-512. That it is the
registered compiler's output on committed advice, explaining the committed trace (each proof block's inputs and outputs
committed values), is not stated: v0 of the policy and remote read neither the log's builder item nor the trace.

Proved as `Properties.Remote.SecurityProofs.X : Properties.Remote.X` in `Proofs.Properties.Remote`.
-/

namespace Properties.Remote

open Core Core.Composition Verity
open Lean (Json)
open Flock.Program (Prog)
open NciCheck.Properties (Parses Expands LaysOut Names Divides Meets ProvenanceHeld)
open NciCheck.Protocol (instanceOf trustedOf)

/-- The policy, met by the declaration `d` and its division `U` at `Γ` and `α`: every inference block of `U` has at most
`Γ` gates and reads at most `α` fresh input bits per gate (`Meets`), so `d` complies (`Nci.Complies`); when `d` is
Boolean the division pads to a `(Γ, ⌊αΓ⌋)`-partition of a declaration containing `d` (`Pad`, `Partition.IsCK`); and when
every gate of `d` reads a gate of positive width where it has one (`BitFaninPos`) and every computed gate is 1 to `W`
bits wide, `d`'s bit-level expansion has a division that pads to a `(ΓW, ⌊αΓW⌋)`-partition. -/
def PolicyMet (d : Declaration) {n : ℕ} (U : Partition d.D n) (Γ : ℕ) (α : ℝ) : Prop :=
  (∀ u, Meets d U Γ α u) ∧ Nci.Complies Γ α d ∧
    (d.D.Boolean → ∃ (B : ℕ) (P : Partition d.D B) (d' : Declaration) (P' : Partition d'.D B),
      Nonempty (Pad d P d' P') ∧ P'.IsCK d' Γ ⌊α * Γ⌋₊) ∧
    ∀ W : ℕ, d.D.BitFaninPos → (∀ g, d.D.isInput g = false → 1 ≤ d.D.w g) → (∀ g, d.D.w g ≤ W) →
      ∃ (B : ℕ) (P : Partition d.expand.D B) (d' : Declaration) (P' : Partition d'.D B),
        Nonempty (Pad d.expand P d' P') ∧ P'.IsCK d' (Γ * W) ⌊α * (Γ * W : ℕ)⌋₊

/-- The parse `i` of the policy's instance carries the policy's own parameters `p`: its program, and `Γ` and
`α = num / den` as the parameters write them. -/
def VerifierOwn (p : NciCheck.Protocol.Params) (i : NciCheck.Instance) : Prop :=
  i.program = p.program ∧ NciCheck.natOf? p.gamma = some i.gamma ∧
    ∃ a b, p.alpha = .arr #[a, b] ∧ NciCheck.natOf? a = some i.num ∧ NciCheck.natOf? b = some i.den

/-- What the policy's case establishes, decoded as its parameters `p`, the run's record `c` and the evidence `e`: the
registered circuit decodes to `P`; the record registers `P` by its SHA-512; θ holds what the log's first record
registers; the instance the policy checked has an expansion gate by gate with a parse, a declaration `P` lays out with
the θ it names and a division its owners are, which meet the policy; and on every such expansion and parse, the parse
carries the policy's parameters and every such declaration and division meet the policy at its `Γ` and `α`. -/
def PolicyHeld (p : NciCheck.Protocol.Params) (c : NciCheck.Protocol.Commitment) (e : NciCheck.Protocol.Evidence) :
    Prop :=
  ∃ P : Prog, Flock.Program.decode p.program.toUTF8 = .ok P ∧
    NciCheck.Baseline.strAt c.record "program" = some (Flock.hexOf P.sha512) ∧
    ProvenanceHeld (instanceOf p e) (trustedOf p c) ∧
    (∃ (j' : Json) (i : NciCheck.Instance) (d : Declaration) (n : ℕ) (U : Partition d.D n),
      Expands (instanceOf p e) j' ∧ Parses j' i ∧ LaysOut P d.D ∧ Names i.theta d.θ ∧ Divides i.owner U ∧
        VerifierOwn p i ∧ PolicyMet d U i.gamma ((i.num : ℝ) / i.den)) ∧
    ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration) (n : ℕ) (U : Partition d.D n),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner U →
        VerifierOwn p i ∧ PolicyMet d U i.gamma ((i.num : ℝ) / i.den)

/-- The policy's case `k` decodes under the policy's own decoders, and what it decodes to is `PolicyHeld`. -/
def CaseHeld (k : Case) : Prop :=
  ∃ p c e, (canonicalJson k.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok p ∧
    NciCheck.Protocol.commitmentOf k.commitment = .ok c ∧ NciCheck.Protocol.evidenceOf k.evidence = .ok e ∧
    PolicyHeld p c e

/-- The one-stage audit's case `k` decodes under the audit's own decoders, and the record it decodes to names the
program of the audit's own verifier, the program its sessions are verified against, by its SHA-512
(`Properties.SampledProofs.R2`). -/
def StageHeld (k : Case) : Prop :=
  ∃ (p : SampledProofs.OneStageZk.Params) (c : SampledProofs.OneStage.Commitment),
    (canonicalJson k.params.toUTF8 >>= SampledProofs.OneStageZk.paramsOfJson) = .ok p ∧
    SampledProofs.OneStage.commitmentOf k.commitment = .ok c ∧ Properties.SampledProofs.R2 c.record p.base.verifier

/-- What remote's acceptance establishes about its cases `ev`: they are the three subprotocols' cases in order, the
policy's `kI`, PoUW's `kW` and the one-stage audit's `kS`; the three commitments decode, each under its own
subprotocol's decoder, to one digest `r`, the run's record's; the policy's case is `CaseHeld`: the record names the
registered circuit by its SHA-512, θ holds what the log's first record registers, the instance checked has a parse, a
declaration the registered circuit lays out and a division, every parse carries the policy's `program`, `Γ` and `α`,
and every declaration the registered circuit lays out, with the declared division, meets the policy; and the one-stage
audit's case is `StageHeld`: its record names, by its SHA-512, the program the audit verifies the sessions against. -/
def InferenceOnlyHeld (ev : List Case) : Prop :=
  ∃ (kI kW kS : Case) (r : ByteArray), ev = [kI, kW, kS] ∧
    Properties.Composition.digestOf NciCheck.Protocol.protocol kI.commitment = some r ∧
    Properties.Composition.digestOf PouwCheck.protocol kW.commitment = some r ∧
    Properties.Composition.digestOf SampledProofs.OneStageZk.protocol.toProtocol kS.commitment = some r ∧
    CaseHeld kI ∧ StageHeld kS

/-- **Remote's acceptance gives the inference-only policy on the registered circuit.** On every input remote's check
accepts, on any blobs, its cases are `InferenceOnlyHeld`. -/
def AcceptInferenceOnly : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof → InferenceOnlyHeld ev

/-! ## Another check in remote's place -/

/-- `AcceptInferenceOnly` with the check `v` in place of remote's on the blobs: `v`'s acceptance establishes what the
property says remote's does. -/
def AcceptInferenceOnlyBy (v : List Named → ByteArray → List Case → Outcome Keyed Keyed) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    v ps c ev = .accept pub prof → InferenceOnlyHeld ev

/-- `v` accepts every input, whatever it publishes. -/
def AcceptsAll (v : List Named → ByteArray → List Case → Outcome Keyed Keyed) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case), ∃ pub prof, v ps c ev = .accept pub prof

/-- What remote's acceptance establishes in the form `Nci.Guarantees.DampenerRemote` reads, about its cases `ev`: they
are the three subprotocols' cases in order, the policy's case decodes to the parameters `p`, the run's record `cr` and
the evidence `e`, the record names the registered circuit `P` by its SHA-512, the instance the policy checked, expanded
gate by gate, has a parse carrying the policy's parameters and a declaration `P` lays out with its θ, which complies at
the policy's `Γ` and `α`, and on every such expansion and parse, the parse carries the policy's parameters and every
such declaration complies. -/
def CompliesHeld (ev : List Case) : Prop :=
  ∃ (kI kW kS : Case) (p : NciCheck.Protocol.Params) (cr : NciCheck.Protocol.Commitment)
    (e : NciCheck.Protocol.Evidence) (P : Prog), ev = [kI, kW, kS] ∧
    (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok p ∧
    NciCheck.Protocol.commitmentOf kI.commitment = .ok cr ∧ NciCheck.Protocol.evidenceOf kI.evidence = .ok e ∧
    Flock.Program.decode p.program.toUTF8 = .ok P ∧
    NciCheck.Baseline.strAt cr.record "program" = some (Flock.hexOf P.sha512) ∧
    (∃ (j' : Json) (i : NciCheck.Instance) (d : Declaration),
      Expands (instanceOf p e) j' ∧ Parses j' i ∧ LaysOut P d.D ∧ Names i.theta d.θ ∧ VerifierOwn p i ∧
        Nci.Complies i.gamma ((i.num : ℝ) / i.den) d) ∧
    ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ →
        VerifierOwn p i ∧ Nci.Complies i.gamma ((i.num : ℝ) / i.den) d

/-- **Remote's acceptance gives `Nci.Complies`**, the hypothesis `Nci.Guarantees.DampenerRemote` takes: on every input
remote's check accepts, on any blobs, its cases are `CompliesHeld`. -/
def AcceptComplies : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof → CompliesHeld ev

/-- `AcceptComplies` with the check `v` in place of remote's on the blobs. -/
def AcceptCompliesBy (v : List Named → ByteArray → List Case → Outcome Keyed Keyed) : Prop :=
  ∀ (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    v ps c ev = .accept pub prof → CompliesHeld ev

end Properties.Remote
