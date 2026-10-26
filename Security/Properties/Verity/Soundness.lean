import CoreOnsite
import Properties.Composition
import Definitions.Accounting.JointPeriod

/-!
# Soundness at the verifier of record: one statement over what onsite's compiled Lean accepts

The verifier of record is one compiled Lean function, `Core.Onsite.protocol`'s `answer`, the composition
(`Core.Composition.product`) of the five protocols that decide a period (`Accounting.Audit`):

| audit | component |
| --- | --- |
| sampled proofs, integrity | `SampledProofs.OneStage.protocol` |
| PoUW, work | `PouwCheck.protocol` |
| PoUS, space | `PousCheck.RunProtocol.protocol` |
| bandwidth | `RemoteAudit.Protocol.protocol` |
| the network certifier, communication | `NetworkCheck.Period.protocol` |

`Accounting.Guarantees.JointPeriod` and `JointHorizon` compose the five audits' per-period guarantees over a period run
whose verdicts are a free field (`PeriodRun.accepts`). Here the verdicts are not free: audit `a` accepts at the coins
when its compiled component accepts its own case in what the prover sent (`compiled`), and the event is the verifier
of record's acceptance (`Accepts`). `Soundness` states that, over a horizon of periods, onsite accepts some period while
one of that period's claims fails with probability at most every period's errors summed, given each audit's
per-period guarantee at its compiled component (the legs). It is proved by `Properties.Composition` (onsite's
acceptance is every component's) and `JointHorizon`.

The legs are hypotheses until each is proved at its component (the one-theorem plan's ONE-8 and ONE-18, in
`note:proofs/20261026T0332Z-report-onethm`), and so are the run's other fields: the partition and the circuit
(ONE-7), the network certifier's claim (ONE-12), and the coordinate of the coins each component reads (ONE-4). Each
piece replaces one of them with a definition from the files, until the statement takes no hypothesis but the named
assumptions.
-/

namespace Verity

open Accounting Core

/-- The compiled component of onsite that decides audit `a`. -/
def component : Audit → Protocol
  | .sampled => SampledProofs.OneStage.protocol
  | .pouw => PouwCheck.protocol
  | .pous => PousCheck.RunProtocol.protocol
  | .poub => RemoteAudit.Protocol.protocol
  | .network => NetworkCheck.Period.protocol

/-- What the prover sends onsite's verifier of record: the parameters, the commitment and the evidence, as bytes. -/
structure Sent where
  params : ByteArray
  commitment : ByteArray
  evidence : ByteArray

/-- **The verifier of record accepts what the prover sent.** -/
def Accepts (s : Sent) : Prop :=
  (Onsite.protocol.answer s.params s.commitment s.evidence).verdict = .accept

/-- Audit `a`'s component accepts its own case in what the prover sent: onsite's three byte strings decode, and one of
their cases is `component a`'s, which decodes under the parameters onsite names for it, and its check accepts it. -/
def ComponentAccepts (a : Audit) (s : Sent) : Prop :=
  ∃ (ps : List Composition.Named) (c : ByteArray) (ev : List Composition.Case),
    (canonicalJson s.params >>= Composition.decodeParams) = .ok ps ∧ Composition.decodeCommitment s.commitment = .ok c ∧
      Composition.decodeEvidence s.evidence = .ok ev ∧
      ∃ x ∈ Properties.Composition.triples Onsite.components ps ev, x.1 = component a ∧
        Properties.Composition.DecodedAccepted x.1 x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence

section

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}

/-- The period `r` with the compiled verdicts: audit `a` accepts at the coins `ξ` when its component accepts its own
case in `s ξ`, what the prover sent at those coins. -/
def compiled (r : PeriodRun d α C Ω) (s : ((i : I) → Ω i) → Sent) : PeriodRun d α C Ω :=
  { r with accepts := fun a ξ => ComponentAccepts a (s ξ) }

end

/-- **Soundness over a horizon, at the verifier of record.** On any device, for any `m` periods of a node's run over
one circuit on one coin space, in which the prover sends onsite `s j ξ` in period `j` at the coins `ξ` (a function of
every coin, so it may adapt to all of them), and whose audits have their per-period guarantees at their compiled
components (period `j`'s at certificates `c j`, price `p` and errors `δ j`): the probability that onsite's compiled
verifier accepts some period while one of that period's claims fails (more than an `ε` share of the circuit's units
wrong, less than `W` of work, fewer than `S` bits held on average, fewer than `Q` bits read, or more than `K` bits
through the network certifiers) is at most `∑_j ∑_a δ j a`. -/
def Soundness : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (m : ℕ) (r : Fin m → PeriodRun d α C Ω)
    (s : Fin m → ((i : I) → Ω i) → Sent) (c : Fin m → Certificates) (p : Price) (δ : Fin m → Audit → ℝ),
    (∀ j a, (compiled (r j) (s j)).Sound (c j) p a (δ j a)) →
    Pouw.pr (fun ξ => ∃ j, Accepts (s j ξ) ∧ ∃ a, ¬ (r j).Holds (c j) p a ξ) ≤ ∑ j, ∑ a, δ j a

end Verity
