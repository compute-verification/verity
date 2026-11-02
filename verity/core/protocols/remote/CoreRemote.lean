/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
subprotocols are the inference-only policy (`NciCheck.Protocol.protocol`, `verity/inference-only/v0`: the declared
circuit divides into inference blocks within their gate and fresh-input budgets, which reads only the declared circuit
and the registered information and so needs no hardware at the site), work by proof of useful work over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), onsite's work
subprotocol, and integrity by sampled proofs' one-stage audit (`SampledProofs.OneStage.protocol`,
`verity/sampled-proofs/one-stage/v1`; v2, the audit on the proofs themselves, is not remote's until its check reads the
challenger's signed coins). `Core.Composition.tiedSameB` composes them on the caller's blobs: every subprotocol accepts
or remote doesn't, and any one's rejection rejects remote whatever the others answer. The policy's commitment and the
one-stage audit's are the run's record, and both must commit to the same one; every window of the period must name
that record as its Commit record (`windows`). Two records reject remote whatever the subprotocols answer.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Remote composes it for the run of record of 28 Oct only, as an exception to
everything-zero-knowledge, by Daniel's ruling: "Let's run it in the clear tonight as a stated exception." v1's
private form (the division committed under hiding leaves, compliance proved, a padded profile) replaces it after
15:00Z on 28 Oct.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import CoreComposition.Tie
import NciCheck.Protocol
import PouwCheck.Period
import SampledProofs.OneStage
import SampledProofs.OneStageZk

namespace Core.Remote

open Composition

/-- Remote's subprotocols, in the order of their ids. -/
def components : List BlobProtocol :=
  [NciCheck.Protocol.protocol, PouwCheck.Period.protocol, SampledProofs.OneStage.protocol]

/-- The subprotocols that must commit to one record: the policy and the one-stage audit. -/
def tied : List String := [NciCheck.Protocol.protocolId, SampledProofs.OneStage.protocol.id]

/-- The record the policy's case commits to: its commitment's digest, as the policy decodes it. -/
def policyRecord (k : Case) : Option (List ByteArray) :=
  (NciCheck.Protocol.protocol.commitment k.commitment).toOption.map fun c => [NciCheck.Protocol.protocol.digest c]

/-- The digests of the window records a period's case names, in its close's order: each window's Commit record as the
period's credit reads it (`PouwCheck.Credit.commitOf?`, under the auditor's ledger); `none` when the evidence doesn't
decode or a window names no Commit record, which the period refuses. -/
def windowRecords (k : Case) : Option (List ByteArray) := do
  let e ← (PouwCheck.Period.evidenceOf k.evidence).toOption
  let .arr ws := e.windows | none
  ws.toList.mapM fun w => (PouwCheck.Credit.commitOf? (PouwCheck.get w "commit")).map (·.1)

/-- Every window of the period on the policy's record. -/
def windows : Tie :=
  ⟨"record", [(NciCheck.Protocol.protocolId, policyRecord), (PouwCheck.Period.protocolId, windowRecords)]⟩

def protocol : BlobProtocol := tiedSameB "verity/remote/v0" tied [windows] components

/-- A case of the subprotocol `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/inference-only/v0", "verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v1"]
#guard tied.all fun s => (components.map (·.id)).contains s
#guard windows.names.all fun n => (components.map (·.id)).contains n.1

-- every subprotocol refuses its case: remote rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/period/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v1 reject: params: ") == some true
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is missing"
-- PoUW's window check is not remote's work subprotocol, and v2's audit is not its integrity subprotocol
#guard (answer [refused NciCheck.Protocol.protocol, refusedAs "verity/pouw/v1",
    refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/pouw/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.Period.protocol,
    refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"

/-- A registration record every record reader decodes, a window's Commit record among them. -/
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def committing (P : BlobProtocol) (r : String) : Case := { refused P with commitment := r.toUTF8 }

private def recordsOf (policy period audit : String) : Answer :=
  answer [committing NciCheck.Protocol.protocol policy, committing PouwCheck.Period.protocol period,
    committing SampledProofs.OneStage.protocol audit]

-- the policy and the one-stage audit on one record: the tie holds, and the verdict is the subprotocols' (each refuses
-- its params)
#guard (recordsOf (record 0) "[]" (record 0)).reason.map
  (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
-- the policy's record is not the one-stage audit's: remote rejects on the tie, whatever the subprotocols answer
#guard (recordsOf (record 1) "[]" (record 0)).verdict == .reject
#guard (recordsOf (record 1) "[]" (record 0)).published.isNone
#guard (recordsOf (record 1) "[]" (record 0)).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (recordsOf (record 1) "[]" (record 0)).reason.map
  (contains · ", verity/sampled-proofs/one-stage/v1 committed ") == some true

-- a period's windows name records by their Commit records' digests, which are the policy's and the one-stage audit's
-- digests of one record; a window on another record is a stray
private def windowOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"
private def periodOn (rs : List String) : Case :=
  { refused PouwCheck.Period.protocol with
    evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map windowOn) ++ "]}").toUTF8 }
private def strayOf (policy : String) (rs : List String) : Option String :=
  windows.strays [committing NciCheck.Protocol.protocol policy, periodOn rs]
#guard (windowRecords (periodOn [record 0, record 0])).map (·.length) == some 2
#guard windowRecords (periodOn [record 0]) == policyRecord (committing NciCheck.Protocol.protocol (record 0))
#guard policyRecord (committing NciCheck.Protocol.protocol (record 0)) ==
  (SampledProofs.OneStage.protocol.commitment (record 0).toUTF8).toOption.map
    fun c => [SampledProofs.OneStage.protocol.digest c]
#guard strayOf (record 0) [record 0, record 0] == none
-- a window on another record
#guard (strayOf (record 0) [record 0, record 1]).map (·.startsWith
  "record: verity/inference-only/v0 committed ") == some true
#guard (strayOf (record 0) [record 0, record 1]).map (contains · ", verity/pouw/period/v1 committed ") == some true
-- every window on a record that isn't the policy's
#guard (strayOf (record 1) [record 0]).isSome
-- a window naming no Commit record leaves the period to refuse it
#guard strayOf (record 0) [record 0, "{}"] == none

end Core.Remote
