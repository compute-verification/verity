/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are the inference-only policy (`NciCheck.Protocol.protocol`, `verity/inference-only/v0`: the declared circuit
divides into inference blocks within their gate and fresh-input budgets, which reads only the declared circuit and the
registered information and so needs no hardware at the site), work by proof of useful work over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), onsite's work
component, and integrity by sampled proofs' one-stage audit on the proofs themselves
(`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify `--zk` run on each session's
files, which the caller hands in as blobs). `Core.Composition.sameB` composes them on the caller's blobs: every
component accepts or remote doesn't, and any one's rejection rejects remote whatever the others answer. The policy's
commitment and the one-stage audit's are the run's record, and every window of the period must name that record as its
Commit record (`windowRecords`): two records reject remote whatever the components answer.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Remote composes it for the run of record of 28 Oct only, as an exception to
everything-zero-knowledge, by Daniel's ruling: "Let's run it in the clear tonight as a stated exception." v1's
private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it after
15:00Z on 28 Oct.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import NciCheck.Protocol
import PouwCheck.Period
import SampledProofs.OneStageZk

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List BlobProtocol :=
  [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStageZk.protocol]

/-- The digests of the window records a period's case names, in its close's order: each window's Commit record as the
period's credit reads it (`PouwCheck.Credit.commitOf?`, under the auditor's ledger); `none` when the evidence doesn't
decode or a window names no Commit record, which the period refuses. -/
def windowRecords : Records := fun _ k => do
  let e ← (PouwCheck.Period.evidenceOf k.evidence).toOption
  let .arr ws := e.windows | none
  ws.toList.mapM fun w => (PouwCheck.Credit.commitOf? (PouwCheck.get w "commit")).map (·.1)

/-- The policy's record and the one-stage audit's first, then the period's windows'. -/
def records : List (String × Records) :=
  [(NciCheck.Protocol.protocol.id, committed), (SampledProofs.OneStageZk.protocol.id, committed),
    (PouwCheck.Period.protocol.id, windowRecords)]

def protocol : BlobProtocol := sameB "verity/remote/v0" records components

/-- A case of the component `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/inference-only/v0", "verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v2"]
#guard (records.map (·.1)).all (components.map (·.id)).contains

-- every component refuses its case: remote rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/period/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v2 reject: params: ") == some true
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is missing"
-- PoUW's window check is not remote's work component, and v1's audit is not its integrity component
#guard (answer [refused NciCheck.Protocol.protocol, refusedAs "verity/pouw/v1",
    refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/pouw/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- the policy and the one-stage audit commit to the run's record, and a period's windows name records by their Commit
-- records' digests: a record that is another is a stray
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"
private def windowOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"
private def periodOn (rs : List String) : Case :=
  { refused PouwCheck.Period.protocol with
    evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map windowOn) ++ "]}").toUTF8 }
private def committing (P : BlobProtocol) (r : String) : Case := { refused P with commitment := r.toUTF8 }
private def strayOf (policy v2 : String) (rs : List String) : Option String :=
  strays records (components.map (·.toProtocol))
    [committing NciCheck.Protocol.protocol policy, periodOn rs, committing SampledProofs.OneStageZk.protocol v2]
#guard (windowRecords PouwCheck.Period.protocol (periodOn [record 0, record 0])).map (·.length) == some 2
#guard (windowRecords PouwCheck.Period.protocol (periodOn [record 0])) ==
  (committed SampledProofs.OneStageZk.protocol.toProtocol (committing SampledProofs.OneStageZk.protocol (record 0)))
#guard (committed NciCheck.Protocol.protocol (committing NciCheck.Protocol.protocol (record 0))) ==
  (committed SampledProofs.OneStageZk.protocol.toProtocol (committing SampledProofs.OneStageZk.protocol (record 0)))
#guard strayOf (record 0) (record 0) [record 0, record 0] == none
-- a window on another record
#guard (strayOf (record 0) (record 0) [record 0, record 1]).map (·.startsWith
  "record: verity/inference-only/v0 committed ") == some true
#guard (strayOf (record 0) (record 0) [record 0, record 1]).map (contains · ", verity/pouw/period/v1 committed ") ==
  some true
-- the policy's record is not the one-stage audit's
#guard (strayOf (record 1) (record 0) [record 0]).map
  (contains · ", verity/sampled-proofs/one-stage/v2 committed ") == some true
-- a window naming no Commit record leaves the period to refuse it
#guard strayOf (record 0) (record 0) [record 0, "{}"] == none
end Core.Remote
