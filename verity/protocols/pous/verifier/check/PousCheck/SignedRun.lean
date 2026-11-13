module

public import PousCheck.RunProtocol
public import MemoryChallenger.SetupRecord
meta import PousCheck.RunProtocol
meta import MemoryChallenger.SetupRecord

/-!
# A PoUS run on the memory challenger's signed setup, as a core protocol

`RunProtocol`'s run with the store's place taken by what the memory challenger's device signed at setup:
`period.answer_signed`'s check in Lean. The run's check (`Run.checkAt`) reads the memory challenger's table of `C*` and
ties it to the published root; this one reads the device's setup record (`MemoryChallenger.SetupRecord`, the published
outputs it was given), signed by the session's own key under the session's id, and one Merkle path per round, so its
case is the session, the record and the paths, a few megabytes at a served store where the run's is the whole table and
`C*`. Its guarantee is not proved, and it certifies nothing: its profile says why (`GAPS`). Setup's acceptance is not
checked here: it is the open premise `SETUP_ACCEPTED`, which the profile's certificate names among its hypotheses.

Each byte string is canonical JSON:

* params and the commitment are `RunProtocol`'s: `{"point", "session_id", "trusted"}` and the setup's public outputs
  (`SetupProtocol.Public.toJson`), its digest SHA-512 of those bytes;
* the evidence: `{"paths": [{"path": [hex, …], "row": hex}, …], "session": Run.ChallengePhaseRecord.toJson, "setup":
  SignedSetupRecord.toJson}`, for round `j` the row digest of block `rounds[j].index` and its path, siblings from the
  leaf up (`Merkle.path`, `verifier.MerkleTree.path`);
* the check: `checkAt`, its first refusal the reason, word for word the Python's (`run.check_signed_at`):
  1. `Run.checkAt`'s lines that don't read the table: the setup published the point's blocks and a 24-byte salt; the
     session audits that setup at the point, bound to the published root, under the designated id;
  2. one opening for each of the session's rounds, and each opening's row, as the leaf `Run.rowLeaf` of its round's
     index under the published salt, has the published root on its path;
  3. `verdict`, unchanged, against the rows the openings opened (`expectedOf`: block `i`'s row is the first opening
     of a round at `i`; a block no round asks has none): an answer that is not its block is wrong;
  4. the setup record is signed by the session's key, under the session's scheme and id (`setupRecordOk`), and holds
     the published blocks, commitment, salt and root.

On acceptance it publishes `RunProtocol`'s `{"certified": false, "eps": "0", "s_bits": 0}`, and its profile is
`RunProtocol`'s with `GAPS` after the point's gaps and `SETUP_ACCEPTED` after its certificate's hypotheses, when it has a
certificate. `protocol` (`verity/pous/p2/v4/signed-run`) accepts S1's LMS
(`SCHEMES`) only; `testProtocol`, under an id of its own, the test signers' schemes and only them.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace PousCheck.SignedRun

open Lean (Json)
open Flock MemoryChallenger

def ID : String := "verity/pous/p2/v4/signed-run"

/-- Why the signed run certifies nothing, after the point's own gaps. -/
def GAPS : List String :=
  ["the signed run's guarantee is not proved: this check takes the memory challenger's signed setup where the run's " ++
     "check reads the store",
   "setup's check is not yet a proof, and it ran on the developer's node"]

/-- Setup's acceptance, an open premise of the signed run, named as soundness names it (`SetupAccepted`): the store at
the published root is the encoding of the registered weights under the published commitment and salt. -/
def SETUP_ACCEPTED : String :=
  "the store at the published root is the encoding of the registered weights under the published commitment and salt " ++
    "(SetupAccepted)"

/-- One round's opening: the row digest of the block its round asks for, and its Merkle path. -/
structure Opening where
  row : ByteArray
  path : Array ByteArray

structure Evidence where
  paths : Array Opening
  session : Run.ChallengePhaseRecord
  setup : SignedSetupRecord

private def hexJ (b : ByteArray) : Json := .str (hexOf b)

def Opening.toJson (o : Opening) : Json := Json.mkObj [("path", .arr (o.path.map hexJ)), ("row", hexJ o.row)]

def Opening.ofJson (j : Json) : V Opening := do
  pure ⟨← hexField j "row" "opening", ← hexArr j "path" "opening"⟩

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("paths", .arr (e.paths.map Opening.toJson)), ("session", e.session.toJson), ("setup", e.setup.toJson)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do
    pure ⟨← objs j "paths" "evidence" Opening.ofJson,
      ← Run.ChallengePhaseRecord.ofJson (← J.get j "session" "evidence"),
      ← SignedSetupRecord.ofJson (← J.get j "setup" "evidence")⟩) Evidence.toJson (← Core.canonicalJson b)

/-- The first round whose opening does not open its index's row under `pub`'s root and salt, if any. -/
def badOpening (q : P2.Params) (pub : SetupProtocol.Public) (rounds : Array MemoryChallenger.Round)
    (paths : Array Opening) : Option Nat :=
  (Array.range rounds.size).find? fun j =>
    match rounds[j]?, paths[j]? with
    | some r, some o =>
      !Merkle.verifyPath pub.root q.blocks r.index
        (Run.rowLeaf pub.salt (r.index / q.blocksPerSegment) (r.index % q.blocksPerSegment) o.row) o.path
    | _, _ => true

/-- The rows the openings opened: block `i`'s is the row of the first round at index `i`, none for a block no round
asks. -/
def expectedOf (rounds : Array MemoryChallenger.Round) (paths : Array Opening) (i : Nat) : Option ByteArray :=
  ((rounds.zip paths).find? (·.1.index == i)).map (·.2.row)

/-- The signed run's check at `pt` under `schemes` (module docstring). `trusted` is the verifier's list of
memory-challenger keys and `sid` the designated id; neither comes from the case's evidence. -/
def checkAt (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (pub : SetupProtocol.Public) (e : Evidence)
    (sid : ByteArray) : Except String Unit := do
  let q := pt.params
  let r := e.session
  let sc := r.commitment
  let rounds := r.signed.rounds
  let s := e.setup.record
  Run.need (pub.blocks == q.blocks) s!"the setup published {pub.blocks} blocks, not the point's {q.blocks}"
  Run.need (pub.salt.size == Run.SALT) s!"the setup's salt is {pub.salt.size} bytes, not {Run.SALT}"
  Run.need (sc.blocks == q.blocks) s!"the session audits {sc.blocks} blocks, not the setup's {q.blocks}"
  Run.need (sc.k == pt.k) s!"the session runs {sc.k} rounds, not the point's {pt.k}"
  Run.need (decide (sc.limitNs ≤ pt.limitNs))
    s!"the session's limit of {sc.limitNs} ns is past the point's Δ + RTT = {pt.limitNs} ns"
  Run.need (sc.blockBytes == q.blockBytes)
    s!"the session's answers are {sc.blockBytes} bytes, not the point's {q.blockBytes}"
  Run.need (sc.binding == pub.root) "the session is not bound to the root the setup published"
  Run.need (sc.session == sid) "the session does not run under the id the auditor designated"
  Run.need (e.paths.size == rounds.size)
    s!"the case holds {e.paths.size} openings, not one for each of the session's {rounds.size} rounds"
  if let some j := badOpening q pub rounds e.paths then
    throw s!"round {j}'s opening is not a row under the root the setup published"
  let v := verdict schemes trusted (expectedOf rounds e.paths) r.signed sc r.openings
  Run.need v.accepted ("; ".intercalate v.reasons.toList)
  Run.need (setupRecordOk schemes r.signed.scheme r.signed.publicKey sid e.setup)
    "the setup record is not signed by the session's key under the session's id"
  Run.need (s.blocks == pub.blocks) s!"the setup record holds {s.blocks} blocks, not the published {pub.blocks}"
  Run.need (s.commitment == pub.commitment) "the setup record's commitment to W is not the published one"
  Run.need (s.salt == pub.salt) "the setup record's salt is not the published one"
  Run.need (s.root == pub.root) "the setup record's root is not the published one"

/-- The profile: `RunProtocol`'s, its gaps the point's and then `GAPS`, and its certificate's hypotheses, when it has a
certificate, followed by `SETUP_ACCEPTED`. -/
def profileJson (pt : Run.Point) : Json :=
  let (c, gs) := RunProtocol.gaps pt
  let c := c.map fun c => { c with hypotheses := c.hypotheses ++ [SETUP_ACCEPTED] }
  ((RunProtocol.profileJson pt).setObjVal! "gaps" (.arr ((gs ++ GAPS).map Json.str).toArray)).setObjVal! "certificate"
    (match c with | some c => c.toJson | none => .null)

/-- Nothing certified: `eps` 0 and `s_bits` 0, the vacuous statement. -/
def PUBLIC : RunProtocol.Public := ⟨false, ⟨0, 1⟩, 0⟩

def check (schemes : Schemes) (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e : Evidence) :
    Core.Outcome RunProtocol.Public Run.Point :=
  match checkAt schemes p.point p.trusted.toList pub e p.sessionId with
  | .ok () => .accept PUBLIC p.point
  | .error reason => .reject reason

def mkProtocol (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := RunProtocol.Params
  Commitment := SetupProtocol.Public
  Evidence := Evidence
  Public := RunProtocol.Public
  Profile := Run.Point
  params := RunProtocol.decodeParams
  commitment := RunProtocol.decodeCommitment
  evidence := decodeEvidence
  digest := RunProtocol.digest
  check := check schemes
  publicJson := RunProtocol.Public.toJson
  profileJson := profileJson

/-- `verity/pous/p2/v4/signed-run` under S1's LMS. -/
def protocol : Core.Protocol := mkProtocol ID SCHEMES

/-- The same check under the test signers' schemes alone, its id `test/verity/pous/p2/v4/signed-run`. -/
def testProtocol : Core.Protocol := mkProtocol (TEST_PREFIX ++ ID) TEST_SCHEMES

#guard protocol.id == "verity/pous/p2/v4/signed-run"
#guard testProtocol.id == "test/verity/pous/p2/v4/signed-run"
#guard RunProtocol.protocol.id == "verity/pous/p2/v4/run"
#guard RunProtocol.testProtocol.id == "test/verity/pous/p2/v4/run"
#guard RunProtocol.Public.toJson PUBLIC == Json.mkObj [("certified", .bool false), ("eps", .str "0"), ("s_bits", 0)]
#guard ((profileJson Run.DEPLOYED).getObjValD "gaps") == .arr (GAPS.map Json.str).toArray
#guard ((profileJson { Run.DEPLOYED with k := 104 }).getObjValD "gaps") ==
  .arr (("k = 104 is below the certificate's 105" :: GAPS).map Json.str).toArray
#guard (((profileJson Run.DEPLOYED).getObjValD "certificate").getObjValD "hypotheses") ==
  .arr #[.str SETUP_ACCEPTED]
#guard ((profileJson Run.DEPLOYED).getObjValD "certificate").getObjValD "theorem" ==
  ((RunProtocol.profileJson Run.DEPLOYED).getObjValD "certificate").getObjValD "theorem"
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end PousCheck.SignedRun
