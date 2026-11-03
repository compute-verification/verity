/-
# The firewall's release as a core protocol (`verity/firewall-release/v0`)

One outer session's release decision, through `Core.Protocol`:

* `params`: the verifier's public input, `verity/firewall-public/v0` (`Flock.Firewall.Contract.PublicInput.ofJson`), which
  the contract holds the session's release log to;
* `commitment`: the firewall's record before the session (`record.jsonl`, `Firewall.Record`), bound by the SHA-512 of its
  bytes;
* `evidence`: canonical JSON `{"now_ms", "part", "session", "transcript"}`: the outer session asked for (recursive session
  and statement), the firewall's clock when its proofs are ready, and the session's release log
  (`verity/firewall-transcript/v0`, `ReleaseLog.ofJson`);
* `check`: `decide` with the send check `Contract.run`: take the slot, admit the session, hold the release log to the
  contract, then the release time. It accepts with the session, its statement and stream and when its proofs leave
  (public), and the release log's phase and item count (the profile: the contract let every item through); it rejects a
  refusal (`FIREWALL-REFUSED: …`), a stop (its why), and a record the session's own ops left unreadable (`Record.op`).
  It is never undecided.

The record events the session appends (its `slot`, a dead holder's stop, a refusal or a stop) are not in the outcome: an
`Outcome` carries a verdict and a reason, and nothing of the firewall's state after it. `Firewall.Loop` records them on
the firewall's state directory.
-/
import CoreProtocol.Cli
import Flock.Firewall.Contract
import Flock.Hash
import Firewall.Release

namespace Core.Firewall

open Lean (Json)
open Flock (V)
open Flock.Firewall.Contract (PublicInput ReleaseLog)

def ID : String := "verity/firewall-release/v0"

structure Evidence where
  session : String
  part : String
  nowMs : Nat
  log : ReleaseLog
  deriving Inhabited

/-- What an accepted session makes public: which outer session leaves, and when. -/
structure Released where
  session : String
  part : String
  stream : String
  releaseMs : Option Nat
  deriving BEq, Repr

/-- What the check established: the contract let every item of the release log through. -/
structure Checked where
  phase : String
  items : Nat
  deriving BEq, Repr

def Released.toJson (r : Released) : Json :=
  Json.mkObj [("part", Json.str r.part), ("release_ms", match r.releaseMs with | some t => Json.num t | none => Json.null),
    ("session", Json.str r.session), ("stream", Json.str r.stream)]

def Checked.toJson (c : Checked) : Json :=
  Json.mkObj [("contract", Json.str Flock.Firewall.Contract.FORMAT), ("items", Json.num c.items),
    ("phase", Json.str c.phase)]

/-- The send check: the contract on the session's release log, under the verifier's public input. -/
def sendCheck (pub : PublicInput) (t : ReleaseLog) : Run :=
  match (Flock.Firewall.Contract.run pub t).cut with
  | none => .ok
  | some why => .err why

def check (pub : PublicInput) (r : Record) (e : Evidence) : Core.Outcome Released Checked :=
  match decide r e.session e.part e.nowMs .ok (sendCheck pub e.log) with
  | .error why => .reject why
  | .ok (.refused why, _) => .reject s!"FIREWALL-REFUSED: {why}"
  | .ok (.stop why, _) => .reject why
  | .ok (.go a, _) => .accept { session := e.session, part := e.part, stream := a.stream, releaseMs := a.releaseMs }
      { phase := e.log.phase, items := e.log.items.size }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← canonicalJson b
  let ctx := "the evidence"
  pure { session := ← Flock.J.str j "session" ctx, part := ← Flock.J.str j "part" ctx, nowMs := ← Flock.J.nat j "now_ms" ctx,
         log := ← ReleaseLog.ofJson (← Flock.J.get j "transcript" ctx) }

def recordOf (b : ByteArray) : V Record := do
  let some text := String.fromUTF8? b | throw "the record is not UTF-8"
  Record.read text

def protocol : Protocol where
  id := ID
  Params := PublicInput
  Commitment := Record
  Evidence := Evidence
  Public := Released
  Profile := Checked
  params := PublicInput.ofJson
  commitment := recordOf
  evidence := evidenceOf
  digest r := Flock.Sha512.hash r.text.toUTF8
  check := check
  publicJson := Released.toJson
  profileJson := Checked.toJson

end Core.Firewall
