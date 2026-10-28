import CoreRemote
import Definitions.Core
import Properties.Composition
import Properties.NciDesc
import Properties.Nci.Baseline

/-!
# Remote's acceptance and the inference-only policy

`Core.Remote.protocol` (`verity/remote/v0`) is `Core.Composition.sameB` over three components, in the order of their
ids: the inference-only policy (`NciCheck.Protocol.protocol`), PoUW's window check (`PouwCheck.protocol`) and sampled
proofs' one-stage audit (`SampledProofs.OneStageZk.protocol`). Each one's commitment is the run's registration record,
and the composite rejects unless all three decode to one digest (`Core.Composition.strays`). The policy's check
(`NciCheck.Protocol.check`) refuses a record whose `program` is not the declared circuit's SHA-512 (`names`), then runs
the check of record, `NciCheck.Baseline.evaluate`: the policy check at the size of the description
(`NciCheck.Desc.evaluate`), then θ's provenance against the registration log through the run's record.

* `AcceptInferenceOnly`: on every input remote's check accepts, the three components' records have one digest, the
  record the policy's case commits to names the declared circuit by its SHA-512, θ holds what the log's first record
  registers (`NciCheck.Properties.ProvenanceHeld`), and on the instance the policy checked, expanded gate by gate
  (`NciCheck.Properties.Expands`), every declaration the circuit lays out with θ, and the declared division `U` the
  owners are, meet the policy (`PolicyMet`): every inference block of `U` has at most `Γ` gates and reads at most `α`
  fresh input bits per gate, so the declaration complies (`Nci.Complies Γ α`), and it has a padded
  `(Γ, ⌊αΓ⌋)`-partition when Boolean, its bit-level expansion a padded `(ΓW, ⌊αΓW⌋)`-partition when its gates are 1 to
  `W` bits wide.
* `AcceptComplies`: its corollary in the form `Nci.Guarantees.DampenerRemote` reads, `Nci.Complies Γ α d` from
  acceptance.

`Γ` and `α = num / den` are the verifier's own: the parameters the composite names for the policy. No statement takes an
assumption: the digests are those the code computes and compares, so that PoUW and the one-stage audit read the policy's
record is up to a SHA-512 collision between two records the run carries, which the statements leave to their readers.

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

/-- What the policy's case establishes, decoded as its parameters `p`, the run's record `c` and the evidence `e`: the
declared circuit decodes to `P`; the record registers `P` by its SHA-512; θ holds what the log's first record registers;
and the instance the policy checked has an expansion gate by gate, on each of which every declaration `P` lays out with
the θ it names, and the division its owners are, meet the policy at the verifier's `Γ` and `α`. -/
def PolicyHeld (p : NciCheck.Protocol.Params) (c : NciCheck.Protocol.Commitment) (e : NciCheck.Protocol.Evidence) :
    Prop :=
  ∃ P : Prog, Flock.Program.decode p.program.toUTF8 = .ok P ∧
    NciCheck.Baseline.strAt c.record "program" = some (Flock.hexOf P.sha512) ∧
    ProvenanceHeld (instanceOf p e) (trustedOf p c) ∧ (∃ j', Expands (instanceOf p e) j') ∧
    ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration) (n : ℕ) (U : Partition d.D n),
      Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ → Divides i.owner U →
        i.program = p.program ∧ PolicyMet d U i.gamma ((i.num : ℝ) / i.den)

/-- The policy's case `k` decodes under the policy's own decoders, and what it decodes to is `PolicyHeld`. -/
def CaseHeld (k : Case) : Prop :=
  ∃ p c e, (canonicalJson k.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok p ∧
    NciCheck.Protocol.commitmentOf k.commitment = .ok c ∧ NciCheck.Protocol.evidenceOf k.evidence = .ok e ∧
    PolicyHeld p c e

/-- **Remote's acceptance gives the inference-only policy on the registered circuit.** On every input remote's check
accepts, on any blobs: the evidence is the three components' cases in order, the policy's `kI`, PoUW's `kW` and the
one-stage audit's `kS`; the three commitments decode, each under its own component's decoder, to one digest `r`, the
run's record's; and the policy's case is `CaseHeld`: the record names the declared circuit by its SHA-512, θ holds what
the log's first record registers, and every declaration the circuit lays out, with the declared division, meets the
policy. -/
def AcceptInferenceOnly : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof →
      ∃ (kI kW kS : Case) (r : ByteArray), ev = [kI, kW, kS] ∧
        digestOf NciCheck.Protocol.protocol kI.commitment = some r ∧
        digestOf PouwCheck.protocol kW.commitment = some r ∧
        digestOf SampledProofs.OneStageZk.protocol.toProtocol kS.commitment = some r ∧
        CaseHeld kI

/-- **Remote's acceptance gives `Nci.Complies`**, the hypothesis `Nci.Guarantees.DampenerRemote` takes. On every input
remote's check accepts, the evidence is the three components' cases in order, the policy's case decodes to the
parameters `p`, the run's record `cr` and the evidence `e`, the record names the declared circuit `P` by its SHA-512, and
every declaration `P` lays out, with the θ the policy checked expanded gate by gate, complies at the verifier's `Γ` and
`α`. -/
def AcceptComplies : Prop :=
  ∀ (blobs : Array ByteArray) (ps : List Named) (c : ByteArray) (ev : List Case) (pub prof : Keyed),
    (Core.Remote.protocol.on blobs).check ps c ev = .accept pub prof →
      ∃ (kI kW kS : Case) (p : NciCheck.Protocol.Params) (cr : NciCheck.Protocol.Commitment)
        (e : NciCheck.Protocol.Evidence) (P : Prog), ev = [kI, kW, kS] ∧
        (canonicalJson kI.params.toUTF8 >>= NciCheck.Protocol.paramsOf) = .ok p ∧
        NciCheck.Protocol.commitmentOf kI.commitment = .ok cr ∧ NciCheck.Protocol.evidenceOf kI.evidence = .ok e ∧
        Flock.Program.decode p.program.toUTF8 = .ok P ∧
        NciCheck.Baseline.strAt cr.record "program" = some (Flock.hexOf P.sha512) ∧
        ∀ (j' : Json) (i : NciCheck.Instance) (d : Declaration),
          Expands (instanceOf p e) j' → Parses j' i → LaysOut P d.D → Names i.theta d.θ →
            Nci.Complies i.gamma ((i.num : ℝ) / i.den) d

end Properties.Remote
