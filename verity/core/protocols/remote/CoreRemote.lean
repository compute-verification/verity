/-
Remote Verity (`verity/remote/v0`): what the verifier establishes with no hardware of its own at the site. Its
components are the inference-only policy (`NciCheck.Protocol.protocol`, `verity/inference-only/v0`: the declared circuit
divides into inference blocks within their gate and fresh-input budgets, which reads only the declared circuit and the
registered information and so needs no hardware at the site), work by proof of useful work (`PouwCheck.protocol`), and
integrity by sampled proofs on the proofs themselves (`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`, in
the form its params name: one-stage v2's audit, flock-verify `--zk` run on each session's files, or the recursive audit,
V*'s compiled check run on the recursive session's parts, each on the files the caller hands in as blobs).
`Core.Composition.sameB` composes them on the caller's blobs: every component accepts or remote doesn't, and any
one's rejection rejects remote whatever the others answer. Each one's commitment is the run's record, and all three must
commit to the same one: two records' digests reject remote whatever the components answer.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Remote composes it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it.
-/
import CoreComposition.Blobs
import CoreComposition.Cases
import NciCheck.Protocol
import PouwCheck.Protocol
import SampledProofs.OneStageZk
import VStarRecursive.Sampled

namespace Core.Remote

open Composition

/-- Remote's components, in the order of their ids. -/
def components : List BlobProtocol :=
  [NciCheck.Protocol.protocol, PouwCheck.protocol, VStar.Recursive.Sampled.protocol]

def protocol : BlobProtocol := sameB "verity/remote/v0" (components.map (·.id)) components

/-- A case its component's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) : Answer := Cases.answerCases (protocol.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) ==
  ["verity/inference-only/v0", "verity/pouw/v1", "verity/sampled-proofs/v1"]

-- every component refuses its case: remote rejects on each, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/pouw/v1 reject: params: ") == some true
#guard allRefused.reason.map (contains · "; verity/sampled-proofs/v1 reject: params: ") == some true
#guard (answer [refused PouwCheck.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.protocol]).reason ==
  some "params: verity/sampled-proofs/v1 is missing"
-- neither v1's nor v2's one-stage audit alone is remote's integrity component
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.protocol,
    refused SampledProofs.OneStage.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer [refused NciCheck.Protocol.protocol, refused PouwCheck.protocol,
    refused SampledProofs.OneStageZk.protocol]).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"

/-- A registration record every component's commitment decodes, naming the program `p`. -/
private def record (p : String) : ByteArray :=
  ("{\"format\":\"verity/registration/v1\",\"law\":{},\"partition\":\"p\",\"population\":1,\"program\":\"" ++ p ++
    "\",\"query\":\"q\",\"roots\":[],\"scheme\":\"s\",\"window\":{\"context\":\"c\",\"kind\":\"k\",\"public\":{}}}").toUTF8

private def committing (P : BlobProtocol) (r : ByteArray) : Case := { refused P with commitment := r }

private def recordsOf (rs : List ByteArray) : Answer := answer (components.zipWith committing rs)

-- one record in all three: the tie holds, and the verdict is the components' (each refuses its params)
#guard (recordsOf [record "x", record "x", record "x"]).reason.map
  (·.startsWith "verity/inference-only/v0 reject: params: ") == some true
-- the policy's record is not the others': remote rejects on the tie, whatever the components answer
#guard (recordsOf [record "y", record "x", record "x"]).verdict == .reject
#guard (recordsOf [record "y", record "x", record "x"]).published.isNone
#guard (recordsOf [record "y", record "x", record "x"]).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (recordsOf [record "y", record "x", record "x"]).reason.map (contains · ", verity/pouw/v1 committed ") == some true
#guard !((recordsOf [record "y", record "x", record "x"]).reason.map (contains · "reject: params")).getD true
-- so is sampled proofs of another record than the policy's and PoUW's
#guard (recordsOf [record "x", record "x", record "y"]).reason.map
  (contains · ", verity/sampled-proofs/v1 committed ") == some true

end Core.Remote
