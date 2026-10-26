import CoreProtocol.Cli
import CoreProtocol.Ratio
import PousCheck.Certificate
import PousCheck.Run

/-!
# A PoUS run as a core protocol

`PousCheck.Run.checkAt` behind the core interface (`Core.Protocol`), PoUS's component of onsite Verity
(`verity/core/protocols/onsite`), each byte string canonical JSON:

* params `{"point": {"k", "limit_ns", "params": {"blocks_per_segment", "c0", "segments", "w"}}, "session_id": hex,
  "trusted": [hex, …]}`: the point the run is checked at, the id the auditor designated for the period's session, and
  the verifier's list of memory-challenger keys, which never comes from the run's record;
* the commitment: the setup's public outputs (`SetupProtocol.Public.toJson`), which the prover published before any
  challenge; its digest SHA-512 of those bytes;
* the evidence: `{"digests": hex, "session": Run.Session.toJson}`, the memory challenger's table of `C*` and the session
  it signed;
* the check: `checkAt`, its first refusal the reason, word for word.

On acceptance it publishes the storage it certifies and the error that holds at, `{"certified", "eps", "s_bits"}`
(`certify`): when P2's one certificate (`Certificate.p2`) covers the point (its `(w, c0, blocks per segment)`, its
segment count, `k` at least its own and `Δ + RTT` within its rounds), `s_bits` is `⌊ρ·blocks·w⌋` and `eps` is
`δ + 2^eps_log2`: a node holding fewer bits when the challenges arrived passes with probability at most `eps`, under
the certificate's claims. Off it, `s_bits` is 0 and `eps` 0, the vacuous statement, and the profile names each gap.

`protocol` (`verity/pous/p2/v4/run`) accepts S1's LMS (`SCHEMES`) only; `testProtocol`, under an id of its own, the test
signers' schemes and only them.
-/

namespace PousCheck.RunProtocol

open Lean (Json JsonNumber)
open Flock MemoryChallenger

def ID : String := "verity/pous/p2/v4/run"

structure Params where
  point : Run.Point
  sessionId : ByteArray
  trusted : Array ByteArray

private def natJ (n : Nat) : Json := .num (JsonNumber.fromNat n)
private def hexJ (b : ByteArray) : Json := .str (hexOf b)

def pointJson (pt : Run.Point) : Json :=
  let q := pt.params
  Json.mkObj [("k", natJ pt.k), ("limit_ns", natJ pt.limitNs), ("params", Json.mkObj [
    ("blocks_per_segment", natJ q.blocksPerSegment), ("c0", natJ q.c0), ("segments", natJ q.segments), ("w", natJ q.w)])]

def Params.toJson (p : Params) : Json :=
  Json.mkObj [("point", pointJson p.point), ("session_id", hexJ p.sessionId), ("trusted", .arr (p.trusted.map hexJ))]

def decodeParams : Json → V Params :=
  exactly (fun j => do
    pure ⟨← Run.Point.ofJson (← J.get j "point" "params"), ← hexField j "session_id" "params", ← hexArr j "trusted" "params"⟩)
    Params.toJson

def decodeCommitment (b : ByteArray) : V SetupProtocol.Public := do Run.decodeSetup (← Core.canonicalJson b)

structure Evidence where
  digests : ByteArray
  session : Run.Session

def Evidence.toJson (e : Evidence) : Json := Json.mkObj [("digests", hexJ e.digests), ("session", e.session.toJson)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do pure ⟨← hexField j "digests" "evidence", ← Run.Session.ofJson (← J.get j "session" "evidence")⟩)
    Evidence.toJson (← Core.canonicalJson b)

/-- SHA-512 of the setup's public outputs as canonical JSON, the bytes they decoded from. -/
def digest (pub : SetupProtocol.Public) : ByteArray :=
  match canon pub.toJson with
  | .ok s => Sha512.hash s.toUTF8
  | .error _ => ByteArray.empty

/-- What an accepted run certifies at `pt`: P2's certificate when it covers the point, and why it doesn't otherwise. -/
def gaps (pt : Run.Point) : Option Certificate.Certificate × List String :=
  let q := pt.params
  match Certificate.p2 q.w q.c0 q.blocksPerSegment with
  | none => (none, [s!"P2 has no certificate at (w, c0, blocks per segment) = ({q.w}, {q.c0}, {q.blocksPerSegment})"])
  | some c =>
    let segments := match c.segments with
      | some n => if q.segments != n then [s!"W has {q.segments} segments; {c.theoremName} is about {n}"] else []
      | none => []
    (some c, c.refusals pt.k pt.limitNs ++ segments)

/-- `δ + 2^eps_log2`, a certificate's error, exactly. -/
def Certificate.eps (c : Certificate.Certificate) : Core.Ratio :=
  let e := 2 ^ (-c.epsLog2).toNat
  (Core.Ratio.mk' (c.delta.1 * e + c.delta.2) (c.delta.2 * e)).getD ⟨1, 1⟩

/-- The public outputs: whether the certificate covers the point, the error, and the storage it certifies, in bits. -/
structure Public where
  certified : Bool
  eps : Core.Ratio
  sBits : Nat

def certify (pt : Run.Point) : Public :=
  match gaps pt with
  | (some c, []) => ⟨true, Certificate.eps c, pt.params.blocks * pt.params.w * c.rho.1 / c.rho.2⟩
  | _ => ⟨false, ⟨0, 1⟩, 0⟩

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("eps", .str p.eps.toString), ("s_bits", natJ p.sBits)]

/-- The profile: the store and the audit the run was checked at, the certificate and every gap. -/
def profileJson (pt : Run.Point) : Json :=
  let (c, gs) := gaps pt
  Json.mkObj [("block_bits", natJ pt.params.w), ("blocks", natJ pt.params.blocks),
    ("certificate", match c with | some c => c.toJson | none => .null), ("experimental", .str Certificate.EXPERIMENTAL),
    ("gaps", .arr (gs.map Json.str).toArray), ("k", natJ pt.k), ("limit_ns", natJ pt.limitNs)]

def check (schemes : Schemes) (p : Params) (pub : SetupProtocol.Public) (e : Evidence) :
    Core.Outcome Public Run.Point :=
  match Run.checkAt schemes p.point p.trusted.toList pub e.digests e.session p.sessionId with
  | .ok () => .accept (certify p.point) p.point
  | .error reason => .reject reason

def mkProtocol (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := Params
  Commitment := SetupProtocol.Public
  Evidence := Evidence
  Public := Public
  Profile := Run.Point
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := digest
  check := check schemes
  publicJson := Public.toJson
  profileJson := profileJson

/-- `verity/pous/p2/v4/run` under S1's LMS. -/
def protocol : Core.Protocol := mkProtocol ID SCHEMES

/-- The same check under the test signers' schemes alone, its id `test/verity/pous/p2/v4/run`. -/
def testProtocol : Core.Protocol := mkProtocol (TEST_PREFIX ++ ID) TEST_SCHEMES

#guard (certify Run.DEPLOYED).sBits == 2 ^ 19 * 16448 * 18 / 19
#guard (certify Run.DEPLOYED).eps.toString == s!"{2 ^ 126 + 25}/{25 * 2 ^ 128}"
#guard (certify ⟨⟨64, 189, 4, 2⟩, 3, 1000⟩).sBits == 0
#guard (gaps { Run.DEPLOYED with k := 104 }).2 == ["k = 104 is below the certificate's 105"]
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end PousCheck.RunProtocol
