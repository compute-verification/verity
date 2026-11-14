/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its subprotocols are integrity by SPOT
(`VStar.Recursive.Sampled.protocol`, `verity/sampled-proofs/v1`: one-stage v2's audit on the proofs themselves, on the
blobs the caller hands in; one-stage v1, whose verdicts are the evidence's, and the recursive audit, which SPOT's
soundness doesn't cover, are refused), the
inference-only policy (`NciCheck.Protocol.protocol`,
`verity/inference-only/v0`: the declared circuit divides into inference blocks within their gate and fresh-input
budgets), work by proof of useful work over the period (`PouwCheck.Period.protocol`, `verity/pouw/period/v1`: the
period's windows credited all or nothing), space by PoUS's check of a run (`PousCheck.RunProtocol.protocol`), bandwidth
by the remote bandwidth audit's P2 session on PoUS's store (`RemoteAudit.P2Protocol.protocol`,
`verity/remote-bandwidth/p2/v0`), communication by the network certifier over the period
(`NetworkCheck.Period.protocol`, `verity/certifier/v0`: the period's link-windows certified and proved, and the bound
`K` on what left the site), and the wipe by the memory challenger's streamed-label wipe of the period
(`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`). `Core.Composition.tiedSameB` composes them on the caller's
blobs: every subprotocol accepts or onsite doesn't, and any subprotocol's rejection rejects onsite whatever the others
answer. The policy's commitment and sampled proofs' are the run's registration record (in either form, the one-stage
one), and both must commit to the same one; every window of the period must name that record as its Commit record,
remote's tie (`Core.Remote.windows`). Two records reject onsite whatever the subprotocols answer. PoUS and the remote
bandwidth audit must also be on one store (`store`): two stores reject onsite unless it rejects already.

`counted` (`verity/onsite/counted/v0`) is onsite with the wipe's count held to the network certifier's: the wipe's
subprotocol is `countedWipe` (`MemoryChallenger.WipeProtocol.mkCounted`, `verity/wipe/counted/v0`), which certifies at
the count its evidence gives, and beside it stands the count over the span between the wipe's two marks (`hostSpan`,
`NetworkCheck.Span.protocol` on the registered hardware, `verity/certifier/span/v0`). Its ties hold the two to one
another: the wipe is of the host the count's link is on, and that link is the host's one link besides the memory
challengers' (`wipeNode`, `wipeLink`); the start mark the wipe's signed message names is the count's start
certificate's, the count's two marks open to the wipe's start and end messages, and the count the wipe's evidence gives
is the count's, in whole MiB (`wipeCount`). It is a development composite: `protocol` is the composite a verifier runs
(`Core.Composites`) until a served wipe holds marks, and `counted` then takes its place.

A stated exception: the policy's v0 publishes the declared division and each inference block's profile (its gates and
fresh-input bits) in the clear. Onsite composes it for one run of record only, as an exception to
everything-zero-knowledge, and v1's private form (the division committed under hiding leaves, compliance proved, a
padded profile) replaces it.
-/
module

public import CoreComposition.Blobs
public import CoreComposition.Cases
public import CoreComposition.Tie
public import CoreRemote
public import MemoryChallenger.WipeCount
public import MemoryChallenger.WipeProtocol
public import NciCheck.Protocol
public import NetworkCheck.Period
public import NetworkCheck.Span
public import PousCheck.RunProtocol
public import PouwCheck.Period
public import RemoteAudit.P2Protocol
public import Spot.OneStage
public import Spot.OneStageZk
public import VStarRecursive.Sampled
meta import CoreComposition.Blobs
meta import CoreComposition.Cases
meta import CoreComposition.Tie
meta import CoreRemote
meta import MemoryChallenger.WipeCount
meta import MemoryChallenger.WipeProtocol
meta import NciCheck.Protocol
meta import NetworkCheck.Period
meta import NetworkCheck.Span
meta import PousCheck.RunProtocol
meta import PouwCheck.Period
meta import RemoteAudit.P2Protocol
meta import Spot.OneStage
meta import Spot.OneStageZk
meta import VStarRecursive.Sampled

@[expose] public section

namespace Core.Onsite

open Composition

/-- Onsite's subprotocols, in the order of their ids. -/
def components : List BlobProtocol :=
  [NetworkCheck.Period.protocol, NciCheck.Protocol.protocol, PousCheck.RunProtocol.protocol, PouwCheck.Period.protocol,
    RemoteAudit.P2Protocol.protocol, VStar.Recursive.Sampled.protocol, MemoryChallenger.WipeProtocol.protocol]

/-- The subprotocols that must commit to one record: the policy and sampled proofs. -/
def tied : List String := [NciCheck.Protocol.protocolId, VStar.Recursive.Sampled.protocol.id]

/-- PoUS and the remote bandwidth audit on one store: the root of `C*` PoUS's setup published (its commitment's `root`)
and the store the P2 session audits, the root its registration names (its commitment's `reg.root`, the case's
`store_root`), each read by its own decoder. -/
def store : Tie :=
  ⟨"store",
    [(PousCheck.RunProtocol.protocol.id,
        fun k => (PousCheck.RunProtocol.decodeCommitment k.commitment).toOption.map ([·.root])),
      (RemoteAudit.P2Protocol.protocol.id,
        fun k => (RemoteAudit.P2Protocol.decodeCommitment k.commitment).toOption.map ([·.held.reg.root]))]⟩

/-- Onsite's ties: every window of PoUW's period on the policy's record (remote's, `Core.Remote.windows`), and PoUS and
the remote bandwidth audit on one store. -/
def ties : List Tie := [Core.Remote.windows, store]

def protocol : BlobProtocol := tiedSameB "verity/onsite/v0" tied ties components

/-- A case its subprotocol's own `answer` refuses: empty parameters, commitment and evidence. -/
private def refused (P : BlobProtocol) : Case :=
  { id := P.id, params := "{}", commitment := ByteArray.empty, evidence := ByteArray.empty }

private def answer (xs : List Case) (blobs : Array ByteArray := #[]) : Answer :=
  Cases.answerCases (protocol.on blobs) xs ⟨(List.replicate 64 0).toArray⟩

private def contains (s pat : String) : Bool := (s.splitOn pat).length > 1

#guard orderedB components
#guard components.map (·.id) == ["verity/certifier/v0", "verity/inference-only/v0", "verity/pous/p2/v4/run",
  "verity/pouw/period/v1", "verity/remote-bandwidth/p2/v0", "verity/sampled-proofs/v1", "verity/wipe/v0"]
#guard tied.all fun s => (components.map (·.id)).contains s
#guard ties.all fun t => t.names.all fun n => (components.map (·.id)).contains n.1

-- every subprotocol refuses its case: onsite rejects on each, with or without blobs, and makes nothing public
private def allRefused : Answer := answer (components.map refused)
#guard allRefused.verdict == .reject
#guard allRefused.published.isNone && allRefused.profile.isNone
#guard allRefused.reason.map (·.startsWith "verity/certifier/v0 reject: params: ") == some true
#guard components.all fun P => (allRefused.reason.map (contains · s!"{P.id} reject: params: ")).getD false
#guard (answer (components.map refused) #["not a circuit file".toUTF8]).reason == allRefused.reason
-- a part of onsite's subprotocols is not onsite's, onsite has no policy-less, wipe-less or PoUS-less form, and neither
-- v1's nor v2's one-stage audit alone is onsite's integrity subprotocol
#guard (answer [refused PouwCheck.Period.protocol, refused VStar.Recursive.Sampled.protocol]).reason ==
  some "params: verity/certifier/v0 is missing"
#guard (answer ((components.filter (·.id != NciCheck.Protocol.protocolId)).map refused)).reason ==
  some "params: verity/inference-only/v0 is missing"
#guard (answer ((components.take 6).map refused)).reason == some "params: verity/wipe/v0 is missing"
#guard (answer ((components.filter (·.id != PousCheck.RunProtocol.protocol.id)).map refused)).reason ==
  some "params: verity/pous/p2/v4/run is missing"
#guard (answer ((components.take 5).map refused ++
    [refused Spot.OneStage.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v1 is not a component of this composition"
#guard (answer ((components.take 5).map refused ++
    [refused Spot.OneStageZk.protocol, refused MemoryChallenger.WipeProtocol.protocol])).reason ==
  some "params: verity/sampled-proofs/one-stage/v2 is not a component of this composition"
-- one-stage v1, whose verdicts are the evidence's, is refused by name: a one-stage statement naming no proofs is v1's,
-- and onsite rejects on it with sampled proofs' refusal (`VStar.Recursive.Sampled.V1_REFUSED`)
private def v1 : Answer := answer (components.map fun P =>
  if P.id == VStar.Recursive.Sampled.protocol.id then { refused P with params := "{\"form\":\"one-stage\",\"law\":{}}" }
  else refused P)
#guard v1.verdict == .reject && v1.published.isNone
#guard (v1.reason.map (contains · (s!"verity/sampled-proofs/v1 reject: params: the parameters name no proofs: " ++
  VStar.Recursive.Sampled.V1_REFUSED))).getD false

-- the policy and sampled proofs committing to one record leave each subprotocol to answer as itself; two records
-- reject onsite, naming both, whatever the subprotocols answer
private def record (p : String) : ByteArray :=
  ("{\"format\":\"verity/registration/v1\",\"law\":{},\"partition\":\"p\",\"population\":1,\"program\":\"" ++ p ++
    "\",\"query\":\"q\",\"roots\":[],\"scheme\":\"s\",\"window\":{\"context\":\"c\",\"kind\":\"k\",\"public\":{}}}").toUTF8

private def committed (policy audit : ByteArray) : Answer :=
  answer (components.map fun P =>
    if P.id == NciCheck.Protocol.protocolId then { refused P with commitment := policy }
    else if P.id == VStar.Recursive.Sampled.protocol.id then { refused P with commitment := audit }
    else refused P)

#guard (committed (record "x") (record "x")).reason == allRefused.reason
#guard (committed (record "y") (record "x")).verdict == .reject
#guard (committed (record "y") (record "x")).published.isNone
#guard (committed (record "y") (record "x")).reason.map
  (·.startsWith "record: verity/inference-only/v0 committed ") == some true
#guard (committed (record "y") (record "x")).reason.map
  (contains · ", verity/sampled-proofs/v1 committed ") == some true
#guard !((committed (record "y") (record "x")).reason.map (contains · "reject: params")).getD true

-- every window of PoUW's period on the policy's record: one record leaves the verdict to the subprotocols, and a window
-- on another record rejects onsite, naming both, whatever the subprotocols answer but a rejection, which stands
/-- A registration record every record reader decodes, a window's Commit record among them. -/
private def registration (n : Nat) : String := s!"\{\"format\":\"verity/registration/v2\",\"n\":{n},\"registered\":[]}"

private def executionPhaseOn (r : String) : String := s!"\{\"commit\":\"{Flock.hexOf r.toUTF8}\"}"

/-- Onsite's cases, each refused, with the policy's committing to `policy` and the period's windows on the records
`rs`. -/
private def executionPhasesOn (policy : String) (rs : List String) : List Case :=
  components.map fun P =>
    if P.id == NciCheck.Protocol.protocolId then { refused P with commitment := policy.toUTF8 }
    else if P.id == PouwCheck.Period.protocolId then
      { refused P with evidence := ("{\"pins\":[],\"windows\":[" ++ ",".intercalate (rs.map executionPhaseOn) ++ "]}").toUTF8 }
    else refused P

/-- Why onsite's ties reject the outcome `o` its subprotocols reached on `executionPhasesOn policy rs`; `none` when they hold
it. -/
private def stray (policy : String) (rs : List String) (o : Outcome Keyed Keyed) : Option String :=
  match holdTies ties (executionPhasesOn policy rs) o with
  | .reject r => some r
  | _ => none

private def answered : List (Outcome Keyed Keyed) := [.accept [] [], .undecided "u"]

#guard answered.all fun o => stray (registration 0) [registration 0, registration 0] o == none
-- a window on another record
#guard answered.all fun o => stray (registration 0) [registration 0, registration 1] o ==
  some "record: verity/inference-only/v0 and verity/pouw/period/v1 named different values"
-- every window on a record that isn't the policy's
#guard answered.all fun o => (stray (registration 1) [registration 0] o).isSome
-- a window naming no Commit record leaves the period to refuse it
#guard answered.all fun o => stray (registration 0) [registration 0, "{}"] o == none
-- a subprotocol's rejection stands, and onsite's answer on those cases is its subprotocols' refusals
#guard stray (registration 0) [registration 1] (.reject "r") == some "r"
#guard (answer (executionPhasesOn (registration 0) [registration 1])).reason.map
  (·.startsWith "verity/certifier/v0 reject: params: ") == some true

/-! ## The wipe's count, held to the network certifier's -/

/-- The wipe whose count the composition holds (`MemoryChallenger.WipeProtocol.mkCounted`, `verity/wipe/counted/v0`),
under the accepted schemes. -/
def countedWipe : Protocol :=
  MemoryChallenger.WipeProtocol.mkCounted MemoryChallenger.WipeProtocol.COUNTED_ID MemoryChallenger.SCHEMES

/-- The one link what leaves `host` leaves on, as the registered hardware `S` has it: `host`'s links besides the memory
challengers' are one link, and it joins `host` and a network certifier. Refused, by name: an id that is no host's; a
host with no such link; a host with more than one, a link to another host and a bare uplink among them, since a count
on one link is not of what left on another; and a host whose one link joins no network certifier. -/
def wipeLink (S : Core.Site) (host : String) : Except String String := do
  unless S.kind? host = some .host do throw s!"{host} is no host of the registered hardware"
  let own := S.links.filter fun k => (k.ends.1 = host ∨ k.ends.2 = host) ∧ ¬ S.Joins k .challenger .host
  match own with
  | [] => throw s!"host {host}: it has no link besides the memory challengers'"
  | [k] =>
    unless S.Joins k .certifier .host do
      throw s!"host {host}: its one link, {k.id}, joins no network certifier, so no certificate counts what leaves on it"
    pure k.id
  | ks =>
    throw s!"host {host}: it has {ks.length} links besides the memory challengers' \
      ({", ".intercalate (ks.map (·.id))}); a count on one of them is not of what left on the others"

/-- The count's parameters as the composition takes them: the count's own (`NetworkCheck.Span.paramsOf`), which must
hold the registered hardware, the link they name being its host's one link (`wipeLink`). A refusal is a refusal of the
parameters. Nothing here holds the site to an accepted log: the parameters are the verifier's. -/
def spanParams (j : Lean.Json) : Flock.V NetworkCheck.Span.Params := do
  let p ← NetworkCheck.Span.paramsOf j
  let some r := p.registered
    | throw "the parameters hold no registered hardware (site and head): a wipe's count is on a registered host's link"
  let S ← NetworkCheck.Registered.siteOf (← Flock.J.get j "site" "the object")
  let _ ← wipeLink S r.host
  pure p

/-- The count over the span between a wipe's two marks, on its host's one link: `NetworkCheck.Span.protocol`
(`verity/certifier/span/v0`) on `spanParams`, the same check on parameters that hold the registered hardware. -/
def hostSpan : Protocol := { NetworkCheck.Span.protocol with params := spanParams }

/-- What the wipe's case names of its span (`MemoryChallenger.WipeProtocol.named`); `none` unless its commitment and
its evidence decode and name it. -/
def wipeNamed (k : Case) : Option MemoryChallenger.WipeProtocol.Named :=
  match MemoryChallenger.WipeProtocol.decodeCommitment k.commitment,
    MemoryChallenger.WipeProtocol.decodeEvidence k.evidence with
  | .ok c, .ok e => (MemoryChallenger.WipeProtocol.named c e).toOption.join
  | _, _ => none

/-- What the count's case names of its span (`NetworkCheck.Span.named`); `none` unless its parameters and its commitment
decode. -/
def spanNamed (k : Case) : Option NetworkCheck.Span.Named := NetworkCheck.Span.namedOf k.params k.commitment

/-- A count of egress in whole MiB as the bytes a tie compares: its decimal digits. -/
def mibBytes (n : Int) : ByteArray := (toString n).toUTF8

/-- The wipe is of the host the count's link is on: the server the wipe's parameters name is the host end of the
count's link, as the registered hardware in the count's parameters has it (its id's UTF-8). -/
def wipeNode : Tie :=
  ⟨"node",
    [(countedWipe.id, fun k =>
        (Core.canonicalJson k.params.toUTF8 >>= MemoryChallenger.WipeProtocol.decodeParams).toOption.map ([·.server])),
      (hostSpan.id, fun k =>
        ((Core.canonicalJson k.params.toUTF8 >>= spanParams).toOption.bind (·.registered)).map ([·.host.toUTF8]))]⟩

/-- The wipe and the count over its span name one span and one count, a tie a value, since a tie holds every value its
subprotocols name to the first: the start mark the wipe's signed message names is the SHA-512 of the count's start
certificate; the count's start mark opens to the wipe's start message and its end mark to the wipe's end message; and
the count the wipe's evidence gives is the bits that can have left over the span and its tail, in whole MiB
(`MemoryChallenger.StreamedWipe.egressUnits`: rounded up, saturating). -/
def wipeCount : List Tie :=
  [⟨"start mark",
      [(countedWipe.id, fun k => (wipeNamed k).map ([·.startMark])),
        (hostSpan.id, fun k => (spanNamed k).map ([·.startMark]))]⟩,
    ⟨"start message",
      [(countedWipe.id, fun k => (wipeNamed k).map ([·.startMessage])),
        (hostSpan.id, fun k => (spanNamed k).map ([·.startMessage]))]⟩,
    ⟨"end message",
      [(countedWipe.id, fun k => (wipeNamed k).map ([·.endMessage])),
        (hostSpan.id, fun k => (spanNamed k).map ([·.endMessage]))]⟩,
    ⟨"egress",
      [(countedWipe.id, fun k => (wipeNamed k).map fun n => [mibBytes n.egressMib]),
        (hostSpan.id, fun k => (spanNamed k).bind fun n =>
          n.egressBits.map fun b => [mibBytes (MemoryChallenger.StreamedWipe.egressUnits b)])]⟩]

def COUNTED_ID : String := "verity/onsite/counted/v0"

/-- `counted`'s subprotocols, in the order of their ids: onsite's, with the count over the wipe's span and the wipe
whose count the composition holds. -/
def countedComponents : List BlobProtocol :=
  (hostSpan : BlobProtocol) :: components.dropLast ++ [(countedWipe : BlobProtocol)]

/-- `counted`'s ties: onsite's, the wipe on the count's host, and the wipe and the count on one span and one count. -/
def countedTies : List Tie := ties ++ wipeNode :: wipeCount

/-- Onsite with the wipe's count held to the network certifier's count over the span between its two marks. -/
def counted : BlobProtocol := tiedSameB COUNTED_ID tied countedTies countedComponents

#guard orderedB countedComponents
#guard countedComponents.map (·.id) == ["verity/certifier/span/v0", "verity/certifier/v0", "verity/inference-only/v0",
  "verity/pous/p2/v4/run", "verity/pouw/period/v1", "verity/remote-bandwidth/p2/v0", "verity/sampled-proofs/v1",
  "verity/wipe/counted/v0"]
#guard tied.all fun s => (countedComponents.map (·.id)).contains s
#guard countedTies.all fun t => t.names.all fun n => (countedComponents.map (·.id)).contains n.1
#guard countedTies.map (·.what) == ["record", "store", "node", "start mark", "start message", "end message", "egress"]

private def countedAnswer (xs : List Case) : Answer :=
  Cases.answerCases (counted.on #[]) xs ⟨(List.replicate 64 0).toArray⟩

-- every subprotocol refuses its case: the composition rejects on each, and it has no form without the count or the wipe
#guard (countedAnswer (countedComponents.map refused)).verdict == .reject
#guard (countedAnswer (countedComponents.map refused)).reason.map
  (·.startsWith "verity/certifier/span/v0 reject: params: ") == some true
#guard (countedAnswer ((countedComponents.take 7).map refused)).reason ==
  some "params: verity/wipe/counted/v0 is missing"
#guard (countedAnswer ((countedComponents.drop 1).map refused)).reason ==
  some "params: verity/certifier/span/v0 is missing"
#guard (countedAnswer (components.map refused)).reason.map (·.startsWith "params: ") == some true

/-- A key in the registered form, the count's `#guard`s' zero key. -/
private def key0 : String := "".pushn '0' 48

/-- Two hosts, a network certifier registering `key0` and a memory challenger, joined by `links`. -/
private def siteWith (links : List Core.Site.Link) : Core.Site :=
  ⟨[],
    [{ id := "node0", kind := .host }, { id := "node1", kind := .host },
      { id := "cert0", kind := .certifier, owner := .auditor, key := key0 },
      { id := "chal0", kind := .challenger, owner := .auditor }],
    links⟩

private def uplink : Core.Site.Link := { id := "eth0", ends := ("node0", "cert0"), β := 100, latency := 1 }
private def challenged : Core.Site.Link := { id := "mc0", ends := ("chal0", "node0"), β := 10, latency := 1 }
private def between : Core.Site.Link := { id := "hh0", ends := ("node0", "node1"), β := 400, latency := 1 }
private def bare : Core.Site.Link := { id := "up1", ends := ("node0", Core.Site.outside), β := 10, latency := 1 }

-- a host's one link, its ends in either order, the memory challenger's link aside
#guard (wipeLink (siteWith [uplink]) "node0").toOption == some "eth0"
#guard (wipeLink (siteWith [challenged, { uplink with ends := ("cert0", "node0") }]) "node0").toOption == some "eth0"
-- no host, a host with no link, one whose one link is bare, and one with a second link: to another host, to the outside
#guard !(wipeLink (siteWith [uplink]) "cert0").isOk && !(wipeLink (siteWith [uplink]) "node9").isOk
#guard !(wipeLink (siteWith [uplink, challenged]) "node1").isOk && !(wipeLink (siteWith [challenged]) "node0").isOk
#guard !(wipeLink (siteWith [bare]) "node0").isOk
#guard !(wipeLink (siteWith [uplink, between]) "node0").isOk && !(wipeLink (siteWith [uplink, bare]) "node0").isOk
#guard (match wipeLink (siteWith [uplink, between, challenged]) "node0" with | .error e => e | .ok _ => "") ==
  "host node0: it has 2 links besides the memory challengers' (eth0, hh0); a count on one of them is not of what left \
    on the others"

/-- The count's `#guard`s' parameters (`NetworkCheck.Span.guardParams`) with more keys. -/
private def spanJson (more : List (String × Lean.Json)) : Lean.Json :=
  more.foldl (fun j (k, v) => j.setObjVal! k v)
    ((Core.canonicalJson NetworkCheck.Span.guardParams.toUTF8).toOption.getD .null)

private def onSite (S : Core.Site) : Lean.Json := spanJson [("head", .str "h"), ("site", S.toJson)]

-- the count's parameters hold the registered hardware, and their link is its host's one link
#guard (spanParams (onSite (siteWith [uplink, challenged]))).isOk
#guard (NetworkCheck.Span.paramsOf (spanJson [])).isOk && !(spanParams (spanJson [])).isOk
#guard (NetworkCheck.Span.paramsOf (onSite (siteWith [uplink, between]))).isOk &&
  !(spanParams (onSite (siteWith [uplink, between]))).isOk
#guard hostSpan.id == NetworkCheck.Span.protocolId

private def canonText (j : Lean.Json) : String := (Flock.canon j).toOption.getD ""

/-- A mark in the count's commitment bytes that opens to `m`: the count's `#guard`s' certificate and salt
(`NetworkCheck.Span.guardMark`), then `m`. -/
private def markOf (phase fill : UInt8) (m : ByteArray) : ByteArray :=
  (NetworkCheck.Span.guardMark phase fill).extract 0 175 ++ m

private def wipeCommitment : MemoryChallenger.StreamedWipe.Commitment :=
  ⟨"node0".toUTF8, ⟨#[0]⟩, ⟨16, 3, 1, 1000, #[("gpu", 16 * MemoryChallenger.StreamedWipe.LABEL_BYTES)]⟩⟩

/-- A closed session of no batches under no key, naming the count's start certificate (`guardMark 5 1`'s) as its start
mark. No check accepts it; the ties read it. -/
private def wipeSigned : MemoryChallenger.StreamedWipe.Signed :=
  ⟨wipeCommitment, MemoryChallenger.StreamedWipe.COMPLETE,
    Flock.Sha512.hash ((NetworkCheck.Span.guardMark 5 1).extract 4 111), 0, 0, #[], "", .empty, .empty⟩

private def startMsg : ByteArray := MemoryChallenger.StreamedWipe.startMessage wipeCommitment
private def endMsg : ByteArray := (MemoryChallenger.StreamedWipe.endMessage wipeSigned).toOption.getD .empty

/-- The wipe's case: of `server`, its evidence the signed session and a count. -/
private def wipeCase (server : String) (signed : Option MemoryChallenger.StreamedWipe.Signed) (egressMib : Int) : Case :=
  { id := countedWipe.id,
    params := canonText (MemoryChallenger.WipeProtocol.Params.toJson ⟨server.toUTF8, ⟨#[0]⟩, #[]⟩),
    commitment := (canonText wipeCommitment.toJson).toUTF8,
    evidence := (canonText (MemoryChallenger.WipeProtocol.Evidence.toJson ⟨signed, some egressMib⟩)).toUTF8 }

/-- The count's case on the host `node0`'s one link: its marks at execution phases 5 and 7, opening to `first` and
`last`. Over them and the schedule's tail of one, four execution phases of 1,572,864 bits: one MiB, rounded up. -/
private def spanCase (first last : ByteArray) : Case :=
  { id := hostSpan.id, params := canonText (onSite (siteWith [uplink, challenged])),
    commitment := markOf 5 1 first ++ markOf 7 4 last, evidence := .empty }

/-- Why `counted`'s ties reject an accepted outcome on `cases`; `none` when they hold it. -/
private def countStray (cases : List Case) : Option String :=
  match holdTies countedTies cases (.accept [] []) with
  | .reject r => some r
  | _ => none

/-- A stray of the wipe's tie `what`: its name and the two subprotocols, and neither value. -/
private def wipeStray (what : String) : Option String :=
  some s!"{what}: {countedWipe.id} and {hostSpan.id} named different values"

#guard endMsg.size == 64 && (spanNamed (spanCase startMsg endMsg)).isSome
#guard (wipeNamed (wipeCase "node0" (some wipeSigned) 1)).isSome
-- the wipe and the count over its span on one node, one span and one count
#guard countStray [spanCase startMsg endMsg, wipeCase "node0" (some wipeSigned) 1] == none
-- a wipe of another server than the count's host
#guard countStray [spanCase startMsg endMsg, wipeCase "node1" (some wipeSigned) 1] == wipeStray "node"
#guard wipeStray "node" == some "node: verity/wipe/counted/v0 and verity/certifier/span/v0 named different values"
-- a session that names another start certificate
#guard countStray [spanCase startMsg endMsg,
    wipeCase "node0" (some { wipeSigned with startMark := ⟨Array.replicate 64 9⟩ }) 1] == wipeStray "start mark"
-- marks that open to another wipe's start message, or to another end message than the signed session's
#guard countStray [spanCase ⟨Array.replicate 64 3⟩ endMsg, wipeCase "node0" (some wipeSigned) 1] ==
  wipeStray "start message"
#guard countStray [spanCase startMsg ⟨Array.replicate 64 6⟩, wipeCase "node0" (some wipeSigned) 1] ==
  wipeStray "end message"
-- a count that is not the count's: one MiB less, one more
#guard [0, 2].all fun n => countStray [spanCase startMsg endMsg, wipeCase "node0" (some wipeSigned) n] ==
  wipeStray "egress"
-- an unsigned wipe names nothing, and the outcome stands: undecided there, as the wipe is
#guard countStray [spanCase startMsg endMsg, wipeCase "node0" none 1] == none
-- a subprotocol's rejection stands
#guard (match holdTies countedTies [spanCase startMsg endMsg, wipeCase "node1" (some wipeSigned) 0]
    (.reject "r") with | .reject r => r | _ => "") == "r"

end Core.Onsite
