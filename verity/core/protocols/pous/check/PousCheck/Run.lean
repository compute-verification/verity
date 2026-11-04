import MemoryChallenger.Judge
import PousCheck.SetupProtocol
import PousCheck.Merkle

/-!
# A PoUS run, composed

`verity/core/protocols/pous/run.py`'s `check`: what setup published and what the memory challenger signed, joined. A run
is the setup's public outputs (`SetupProtocol.Public`, `root` the root of `C*`'s leaves under `salt`), the memory
challenger's table of `C*` (every block's `sha512/row/v2` digest in block order, `blockDigest`: the harness's
`DIGESTS.bin`) and its record of one signed session (`ChallengerRecord`). `check` judges it with two inputs of the
verifier's own: `trusted`, the verifier's list of memory-challenger keys, which never comes from the run's record
(whoever writes the record could otherwise list its own key), and the id the auditor designated for the audit's session.
It accepts the run at P2's deployed point (`DEPLOYED`) under S1's LMS (`SCHEMES`) when, in order:

1. the setup published the point's block count and a 24-byte salt;
2. the table holds one 64-byte digest a block, and its leaves under that salt (`rowLeaf`, `verifier.row_leaf`) have the
   root the setup published, so the table is the one of the `C*` setup checked;
3. the session audits that setup at the point: every block, the point's `k`, a limit within the point's `Δ + RTT`,
   answers of the point's block size, and its `binding` the published root;
4. the session runs under the designated id (`session`, in the signed header), which the memory challenger runs at most
   one session under, so an audit has at most one session `check` accepts;
5. `verdict` accepts the session against the table, signed under a key in `trusted`.

The first refusal is the reason, word for word the Python's. `checkAt` takes another point and other schemes, for the
difftest.

`pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID` (`run`) prints `{"accepted": true}` or
`{"accepted": false, "reason": …}` and exits 0 iff accepted; `TRUSTED.json` is the verifier's `{"trusted": [hex, …]}`,
`SETUP.json` the setup's public outputs (`Public.toJson`) and `SESSION.json` `{"commitment", "openings", "signed"}`,
each exactly the canonical JSON of what it decodes to, and `SESSION_ID` the designated id in lowercase hex.
-/

namespace PousCheck.Run

open Lean (Json)
open Flock MemoryChallenger

/-- What a run is checked at: the codec's parameters, the rounds `k` and the largest limit `Δ + RTT`, in ns. -/
structure Point where
  params : P2.Params
  k : Nat
  limitNs : Nat

/-- P2's deployed point: `p2-p16448-c21065`, `2^19` blocks of 2,056 bytes, `k = 105`, `Δ + RTT ≤ 854,000 ns`. -/
def DEPLOYED : Point := ⟨P2.Params.deployed, K, DELTA_NS + RTT_CAP_NS⟩

/-- The setup salt's bytes. -/
def SALT : Nat := 24

/-- One session as the memory challenger records it: what it signed, the commitment it released with it and its coin
openings. Which keys are trusted is the verifier's, never the record's (`check`'s `trusted`). -/
structure ChallengerRecord where
  signed : SignedSession
  commitment : SessionCommitment
  openings : Array CoinOpening

/-- `Merkle.packageLeaf` from a block's row digest `R` (`verifier.row_leaf`): `0x00 ‖ tag_s ‖ u16le(v) ‖ R`. -/
def rowLeaf (salt : ByteArray) (s v : Nat) (row : ByteArray) : ByteArray :=
  Sha512.hash (ByteArray.mk #[0] ++ salt ++ Merkle.le s 8 ++ Merkle.le v 2 ++ row)

/-- Row `i` of the table: block `i`'s digest. -/
def row (digests : ByteArray) (i : Nat) : ByteArray :=
  digests.extract (Merkle.DIGEST * i) (Merkle.DIGEST * (i + 1))

/-- The table's leaves under `salt`, in block order. -/
def rowLeaves (q : P2.Params) (salt digests : ByteArray) : Array ByteArray :=
  (Array.range q.blocks).map fun i => rowLeaf salt (i / q.blocksPerSegment) (i % q.blocksPerSegment) (row digests i)

/-- The digests the session is judged against: row `i` for a block of the store. -/
def expected (q : P2.Params) (digests : ByteArray) (i : Nat) : Option ByteArray :=
  if i < q.blocks then some (row digests i) else none

def need (ok : Bool) (reason : String) : Except String Unit := if ok then pure () else throw reason

/-- `check` at `pt` under `schemes`. -/
def checkAt (schemes : Schemes) (pt : Point) (trusted : List ByteArray) (pub : SetupProtocol.Public)
    (digests : ByteArray) (r : ChallengerRecord) (sid : ByteArray) : Except String Unit := do
  let q := pt.params
  let sc := r.commitment
  need (pub.blocks == q.blocks) s!"the setup published {pub.blocks} blocks, not the point's {q.blocks}"
  need (pub.salt.size == SALT) s!"the setup's salt is {pub.salt.size} bytes, not {SALT}"
  need (digests.size == q.blocks * Merkle.DIGEST)
    s!"the table holds {digests.size} bytes, not a {Merkle.DIGEST}-byte digest for each of {q.blocks} blocks"
  need (Merkle.root (rowLeaves q pub.salt digests) == some pub.root)
    "the table's leaves do not have the root the setup published"
  need (sc.blocks == q.blocks) s!"the session audits {sc.blocks} blocks, not the setup's {q.blocks}"
  need (sc.k == pt.k) s!"the session runs {sc.k} rounds, not the point's {pt.k}"
  need (decide (sc.limitNs ≤ pt.limitNs))
    s!"the session's limit of {sc.limitNs} ns is past the point's Δ + RTT = {pt.limitNs} ns"
  need (sc.blockBytes == q.blockBytes) s!"the session's answers are {sc.blockBytes} bytes, not the point's {q.blockBytes}"
  need (sc.binding == pub.root) "the session is not bound to the root the setup published"
  need (sc.session == sid) "the session does not run under the id the auditor designated"
  let v := verdict schemes trusted (expected q digests) r.signed sc r.openings
  need v.accepted ("; ".intercalate v.reasons.toList)

/-- The composed check of a run (module docstring). `trusted` is the verifier's list of memory-challenger keys and never
comes from the run's record. -/
def check (trusted : List ByteArray) (pub : SetupProtocol.Public) (digests : ByteArray) (r : ChallengerRecord)
    (sid : ByteArray) : Except String Unit :=
  checkAt SCHEMES DEPLOYED trusted pub digests r sid

/-! ## Its formats -/

def ChallengerRecord.toJson (r : ChallengerRecord) : Json :=
  Json.mkObj [("commitment", r.commitment.toJson), ("openings", .arr (r.openings.map CoinOpening.toJson)),
    ("signed", r.signed.toJson)]

def ChallengerRecord.ofJson (j : Json) : V ChallengerRecord := do
  let ctx := "session"
  pure { signed := ← SignedSession.ofJson (← J.get j "signed" ctx),
         commitment := ← SessionCommitment.ofJson (← J.get j "commitment" ctx),
         openings := ← objs j "openings" ctx CoinOpening.ofJson }

def trustedJson (trusted : Array ByteArray) : Json :=
  Json.mkObj [("trusted", .arr (trusted.map fun b => .str (hexOf b)))]

/-- The verifier's `{"trusted": [hex, …]}`. -/
def decodeTrusted : Json → V (Array ByteArray) := exactly (fun j => hexArr j "trusted" "trusted") trustedJson

def decodeSetup : Json → V SetupProtocol.Public := exactly SetupProtocol.Public.ofJson SetupProtocol.Public.toJson

def decodeSession : Json → V ChallengerRecord := exactly ChallengerRecord.ofJson ChallengerRecord.toJson

/-- The outcome as the record's verdict: `{"accepted": true}`, or `{"accepted": false, "reason": …}`. -/
def outcomeJson : Except String Unit → Json
  | .ok () => Json.mkObj [("accepted", .bool true)]
  | .error reason => Json.mkObj [("accepted", .bool false), ("reason", .str reason)]

/-- A point from `{"params": {"blocks_per_segment", "c0", "segments", "w"}, "k", "limit_ns"}`, its parameters valid. -/
def Point.ofJson (j : Json) : V Point := do
  let ctx := "point"
  let p ← J.get j "params" ctx
  let q ← P2.Params.validate ⟨← J.nat p "w" ctx, ← J.nat p "c0" ctx, ← J.nat p "blocks_per_segment" ctx,
    ← J.nat p "segments" ctx⟩
  pure ⟨q, ← J.nat j "k" ctx, ← J.nat j "limit_ns" ctx⟩

/-- `checkAt` on the difftest's line: `{"point", "schemes": [id, …], "trusted": [hex, …], "setup", "digests": hex,
"session", "session_id": hex}`, the schemes by id among those `MemoryChallenger` verifies. -/
def runFn (j : Json) : V Json := do
  let ctx := "run"
  let pt ← Point.ofJson (← J.get j "point" ctx)
  let schemes ← Judge.schemesOf (← (← J.arr j "schemes" ctx).mapM (J.asStr · s!"{ctx}.schemes"))
  let digests ← match unhex (← J.str j "digests" ctx) with
    | .ok b => pure b
    | .error e => throw s!"{ctx}.digests: {e}"
  let sid ← match unhex (← J.str j "session_id" ctx) with
    | .ok b => pure b
    | .error e => throw s!"{ctx}.session_id: {e}"
  let trusted ← hexArr j "trusted" ctx
  pure (outcomeJson (checkAt schemes pt trusted.toList (← decodeSetup (← J.get j "setup" ctx)) digests
    (← decodeSession (← J.get j "session" ctx)) sid))

private def readJson {α : Type} (path : String) (f : Json → V α) : IO (V α) := do
  return (Json.parse (← IO.FS.readFile path)).bind f

/-- `pous-run TRUSTED.json SETUP.json DIGESTS.bin SESSION.json SESSION_ID`: `check`'s outcome as one canonical JSON
line; exit 0 iff accepted. A file that is not JSON, or not exactly what it decodes to, or an id that is not lowercase
hex, is refused with the decoder's reason. -/
def run (trustedPath setupPath digestsPath sessionPath sessionId : String) : IO UInt32 := do
  let digests ← IO.FS.readBinFile digestsPath
  let out := match ← readJson trustedPath decodeTrusted, ← readJson setupPath decodeSetup,
      ← readJson sessionPath decodeSession, unhex sessionId with
    | .ok trusted, .ok pub, .ok r, .ok sid => check trusted.toList pub digests r sid
    | .error e, _, _, _ => .error s!"trusted: {e}"
    | _, .error e, _, _ => .error s!"setup: {e}"
    | _, _, .error e, _ => .error s!"session: {e}"
    | _, _, _, .error e => .error s!"session id: {e}"
  match canon (outcomeJson out) with
  | .ok text => IO.println text
  | .error e => IO.eprintln e; return 2
  return if out matches .ok () then 0 else 1

end PousCheck.Run
