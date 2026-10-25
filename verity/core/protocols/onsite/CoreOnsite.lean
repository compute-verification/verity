/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space, bandwidth and communication. Its components are space by proof of useful space (PoUS's composed
check of a run, `PousCheck.RunProtocol.protocol`), bandwidth by the remote bandwidth audit's session
(`RemoteAudit.Protocol.protocol`), integrity by sampled proofs' one-stage audit, work by proof of useful work, and
communication by the network certifier; the last three are stubs (`Core.Composition.Stub`) until their ports land.
`Core.Composition.product` composes them: every component accepts or onsite doesn't, and any component's rejection
rejects onsite whatever the others answer.
-/
import CoreComposition.Cases
import CoreComposition.Stubs
import PousCheck.RunProtocol
import RemoteAudit.Protocol

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List Protocol :=
  [PousCheck.RunProtocol.protocol, RemoteAudit.Protocol.protocol, Stub.certifier, Stub.pouw, Stub.sampledProofs]

def protocol : Protocol := product "verity/onsite/v0" components

open Cases

private def space : Protocol := PousCheck.RunProtocol.protocol
private def bandwidth : Protocol := RemoteAudit.Protocol.protocol

private def hex (n : Nat) (b : String) : String := String.join (List.replicate n b)

/-- PoUS's case at a reduced point (8 blocks of 64 bits), with evidence that doesn't decode: it rejects. -/
private def spaceCase : Part := Part.of space
  ("{\"point\":{\"k\":3,\"limit_ns\":1000,\"params\":{\"blocks_per_segment\":4,\"c0\":189,\"segments\":2,\"w\":64}}," ++
   "\"session_id\":\"" ++ hex 32 "07" ++ "\",\"trusted\":[]}")
  ("{\"blocks\":8,\"commitment\":\"" ++ hex 64 "01" ++ "\",\"root\":\"" ++ hex 64 "02" ++ "\",\"salt\":\"" ++
   hex 24 "03" ++ "\"}")
  "{}"

/-- A remote bandwidth session over 6 registered blocks whose log holds no event yet: it is undecided. -/
private def bandwidthCase : Part := Part.of bandwidth
  ("{\"not_before_ns\":0,\"params\":{\"deadline_ns\":100,\"deferred\":false,\"lifetime\":2,\"min_passes\":1," ++
   "\"pace\":null,\"samples\":10,\"window_ns\":10,\"windows\":2}}")
  ("{\"bind\":false,\"blocks\":6,\"codec\":{\"field\":\"p512\",\"masked\":false,\"rounds\":2},\"codec_seed\":\"" ++
   hex 32 "f0" ++ "\",\"layout\":\"store-order/v0\",\"node\":\"6e6f6465\",\"payload_root\":\"" ++ hex 64 "fd" ++
   "\",\"receipt\":\"sketch-sha512\",\"rows\":16}")
  ("{\"append_coins\":[],\"coins\":[" ++ ",".intercalate (List.replicate 4 ("\"" ++ hex 32 "aa" ++ "\"")) ++
   "],\"events\":[]}")

private def stubs (f : String → Stubbed) : List Part := [Stub.certifier, Stub.pouw, Stub.sampledProofs].map (f ·.id |>.part)

private def answer (xs : List Part) (ps : List Part := xs) : Answer := answerOfParts protocol xs ps

private def own (x : Part) (P : Protocol) : Answer := P.answer x.params.toUTF8 x.commitment x.evidence

#guard ordered components
#guard components.map (·.id) == ["verity/pous/p2/v4/run", "verity/remote-bandwidth/v0", "verity/stub/certifier/v0",
  "verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]
-- each real component's case on its own: space rejects with its commitment's digest, bandwidth is undecided
#guard (own spaceCase space).verdict == .reject
#guard (own spaceCase space).digest == some (Flock.Sha512.hash spaceCase.commitment)
#guard (own spaceCase space).reason.map (·.startsWith "evidence: ") == some true
#guard (own bandwidthCase bandwidth).verdict == .undecided
#guard (own bandwidthCase bandwidth).digest.map (·.size) == some 64
-- space rejected and bandwidth undecided: onsite rejects whatever the stubs answer, and makes nothing public
#guard (answer (spaceCase :: bandwidthCase :: stubs Cases.accepts)).verdict == .reject
#guard (answer (spaceCase :: bandwidthCase :: stubs Cases.accepts)).published.isNone
#guard (answer (spaceCase :: bandwidthCase :: stubs Cases.accepts)).reason.map
  (·.startsWith "verity/pous/p2/v4/run reject: evidence: ") == some true
#guard (answer (spaceCase :: bandwidthCase :: stubs Cases.undecided)).verdict == .reject
-- the components out of order, or remote's alone, are not onsite's
#guard (answer (bandwidthCase :: spaceCase :: stubs Cases.accepts)).reason ==
  some "params: name each component once, in the order of their ids (verity/pous/p2/v4/run, verity/remote-bandwidth/v0, verity/stub/certifier/v0, verity/stub/pouw/v0, verity/stub/sampled-proofs/v0)"
#guard (answer ([Stub.pouw, Stub.sampledProofs].map (Cases.accepts ·.id |>.part))).reason ==
  some "params: verity/pous/p2/v4/run is missing"

end Core.Onsite
