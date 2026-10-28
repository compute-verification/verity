/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are integrity by sampled proofs' one-stage audit on the proofs themselves
(`SampledProofs.OneStageZk.protocol`, `verity/sampled-proofs/one-stage/v2`: flock-verify `--zk` run on each session's
files, which the caller hands in as blobs) and work by proof of useful work over the period
(`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the period's windows credited all or nothing), onsite's work
component. `Core.Composition.sameB` composes them on the caller's blobs: both accept or remote doesn't, and either one's
rejection rejects remote whatever the other answers. The one-stage audit's commitment is the window record, and every
window of the period must name that record as its Commit record (`windowRecords`): a window on another record rejects
remote whatever the components answer.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import PouwCheck.Period
import SampledProofs.OneStageZk

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List BlobProtocol := [PouwCheck.Period.protocol, SampledProofs.OneStageZk.protocol]

/-- The digests of the window records a period's case names, in its close's order: each window's Commit record as the
period's credit reads it (`PouwCheck.Credit.commitOf?`, under the auditor's ledger); `none` when the evidence doesn't
decode or a window names no Commit record, which the period refuses. -/
def windowRecords : Records := fun _ k => do
  let e ← (PouwCheck.Period.evidenceOf k.evidence).toOption
  let .arr ws := e.windows | none
  ws.toList.mapM fun w => (PouwCheck.Credit.commitOf? (PouwCheck.get w "commit")).map (·.1)

/-- The one-stage audit's record first, then the period's windows'. -/
def records : List (String × Records) :=
  [(SampledProofs.OneStageZk.protocol.id, committed), (PouwCheck.Period.protocol.id, windowRecords)]

def protocol : BlobProtocol := sameB "verity/remote/v0" records components

/-- A case of the component `id` that its own `answer` refuses: empty parameters, commitment and evidence. -/
private def refusedAs (id : String) : Case :=
  { id := id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def refused (P : BlobProtocol) : Case := refusedAs P.id

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/pouw/period/v1", "verity/sampled-proofs/one-stage/v2"]
#guard (records.map (·.1)).all (components.map (·.id)).contains

-- both refuse their cases: remote rejects on both, and makes nothing public
private def bothRefused : Answer :=
  answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStageZk.protocol]
#guard bothRefused.verdict == .reject
#guard bothRefused.published.isNone && bothRefused.profile.isNone
#guard bothRefused.reason.map (·.startsWith "verity/pouw/period/v1 reject: params: ") == some true
#guard bothRefused.reason.map (contains · "; verity/sampled-proofs/one-stage/v2 reject: params: ") == some true
#guard (answer [refused PouwCheck.Period.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is missing"
-- PoUW's window check is not remote's work component, and v1's audit is not its integrity component
#guard (answer [refusedAs "verity/pouw/v1", refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/pouw/v1 is not a component of this composition"
#guard (answer [refused PouwCheck.Period.protocol, refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"

-- a period's windows name records by their Commit records' digests, and a window on another record is a stray
private def record (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"
private def windowOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"
private def periodOn (rs : List String) : Case :=
  { refused PouwCheck.Period.protocol with
    evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map windowOn) ++ "]}").toUTF8 }
private def v2On (r : String) : Case := { refused SampledProofs.OneStageZk.protocol with commitment := r.toUTF8 }
private def strayOf (r : String) (rs : List String) : Option String :=
  strays records (components.map (·.toProtocol)) [periodOn rs, v2On r]
#guard (windowRecords PouwCheck.Period.protocol (periodOn [record 0, record 0])).map (·.length) == some 2
#guard (windowRecords PouwCheck.Period.protocol (periodOn [record 0])) ==
  (committed SampledProofs.OneStageZk.protocol (v2On (record 0)))
#guard strayOf (record 0) [record 0, record 0] == none
#guard (strayOf (record 0) [record 0, record 1]).map (·.startsWith
  "record: verity/sampled-proofs/one-stage/v2 committed ") == some true
#guard (strayOf (record 0) [record 0, record 1]).map (contains · ", verity/pouw/period/v1 committed ") == some true
-- a window naming no Commit record leaves the period to refuse it
#guard strayOf (record 0) [record 0, "{}"] == none
end Core.Remote
