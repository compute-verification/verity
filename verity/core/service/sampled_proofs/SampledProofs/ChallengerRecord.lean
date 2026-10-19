/-
The challenger's record: the verifier's trusted coin commitment (`challenger.py`'s record, which
`tests/test_challenger_difftest.py` holds to this, the verifier of record).

Before any coin opens, the challenger signs its coin commitment with its LMS key over SHA-512 and writes the record on
its own host:

    record  = {commitment, host, signature}                       host: the auditor machine the challenger runs on
    message = SHA-512([RECORD_ROLE] ‖ I ‖ [[host]] ‖ [taggedSha512(COMMITMENT_TAG, commitment)])

each `[x]` a u64 big-endian length then `x`, and `I` the key's identifier (its public key's first 16 bytes): the LMS
role digest's form (`verity/ml/crypto/lms.py`). The verifier reads the record from the challenger's file, never from the
prover's bundle, and checks it under its own challenger's key (`ChallengerKey`: the LMS parameters and public key).
`recordReason` is `record_reason`: no record, a malformed one, or one whose signature doesn't verify under the key, each
a reason in Python's words. `trustedReason` is `trusted_reason`: `recordReason`, then that a coin commitment the prover
presents is the record's. A client checks a coin against the commitment `trustedReason` accepts (`liveReason`), so a C2
without the challenger's record is refused.

The difftest's pieces: `challenger_key` (`ChallengerKey.from_json`), `challenger_record` (`ChallengerRecord.from_json`
and `record_message`) and `challenger_trusted` (`trusted_reason`), each `{"refused": why}` where Python refuses.
-/
import Lms.Verify
import SampledProofs.Challenger

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def RECORD_ROLE : String := "verity/challenger-record/v1"
def COMMITMENT_TAG : String := "verity/challenger/commitment/v1"
def PUBLIC_BYTES : Nat := 80
def RECORD_FORM : String :=
  "a challenger record is {commitment, host, signature}: a live coin commitment, a host of ASCII letters, digits, '-', '.' and '_', and lowercase hex"
def KEY_FORM : String :=
  "a challenger key is {H, W, public}: LMS over SHA-512 (N = 64) with W in 1, 2, 4, 8 and 1 <= H <= 30, and its 80-byte public key in lowercase hex"
def NO_RECORD : String :=
  "no challenger record: the verifier holds no coin commitment its challenger recorded and signed"

/-- A host: nonempty, of ASCII letters, digits, '-', '.' and '_'. -/
def isHost (s : String) : Bool :=
  !s.isEmpty && s.all fun c => c.isAlphanum && c.toNat < 128 || c == '-' || c == '.' || c == '_'

/-- `lms._field`: a u64 big-endian length, then the bytes. -/
def field (b : ByteArray) : ByteArray := Flock.be64 b.size.toUInt64 ++ b

/-- `record_message(I, host, commitment)`: what the challenger's record signs. -/
def recordMessage (I : ByteArray) (host : String) (c : LiveCommitment) : V ByteArray := do
  unless I.size == 16 do throw s!"an LMS identifier is 16 bytes, not {I.size}"
  let d ← taggedSha512 COMMITMENT_TAG c.toJson
  return Flock.Sha512.hash (field RECORD_ROLE.toUTF8 ++ I ++ field (field host.toUTF8) ++ field d)

private def hexOf? (j : Json) : Option ByteArray :=
  match j with
  | .str s => (Flock.unhex s).toOption
  | _ => none

/-- The verifier's key for its own challenger: LMS over SHA-512 at `params`, and the public key `I ‖ T[1]`. -/
structure ChallengerKey where
  params : Lms.Params
  pub : ByteArray

/-- `ChallengerKey.from_json`. -/
def ChallengerKey.ofJson (j : Json) : V ChallengerKey := do
  unless keys? j == some #["H", "W", "public"] do throw KEY_FORM
  match nat? (getD j "H"), nat? (getD j "W"), hexOf? (getD j "public") with
  | some h, some w, some pub =>
    let .ok p := Lms.Params.make 64 w h false | throw KEY_FORM
    unless pub.size == PUBLIC_BYTES do throw KEY_FORM
    pure ⟨p, pub⟩
  | _, _, _ => throw KEY_FORM

def ChallengerKey.toJson (k : ChallengerKey) : Json :=
  Json.mkObj [("H", Lean.toJson k.params.H), ("W", Lean.toJson k.params.W), ("public", Json.str (Flock.hexOf k.pub))]

/-- The challenger's record of an epoch's coin commitment, signed on host `host` before any coin opened. -/
structure ChallengerRecord where
  commitment : LiveCommitment
  host : String
  signature : ByteArray

/-- `ChallengerRecord.from_json`. -/
def ChallengerRecord.ofJson (j : Json) : V ChallengerRecord := do
  unless keys? j == some #["commitment", "host", "signature"] do throw RECORD_FORM
  let host := match getD j "host" with | .str s => s | _ => ""
  let some signature := hexOf? (getD j "signature") | throw RECORD_FORM
  unless isHost host do throw RECORD_FORM
  pure ⟨← LiveCommitment.ofJson (getD j "commitment"), host, signature⟩

def ChallengerRecord.toJson (r : ChallengerRecord) : Json :=
  Json.mkObj [("commitment", r.commitment.toJson), ("host", Json.str r.host),
    ("signature", Json.str (Flock.hexOf r.signature))]

/-- `record_reason` (module docstring): why `record` (`null` when the verifier has none) is not a challenger record
signed under `key`; `none` when it is. -/
def recordReason (key : ChallengerKey) (record : Json) : V (Option String) := do
  if record == Json.null then return some NO_RECORD
  match ChallengerRecord.ofJson record with
  | .error why => return some s!"the challenger record is malformed: {why}"
  | .ok r =>
    let m ← recordMessage (key.pub.extract 0 16) r.host r.commitment
    let ok : Bool := match Lms.Signature.decode key.params r.signature with
      | .ok sig => (Lms.verify key.params key.pub m sig).toOption == some true
      | .error _ => false
    unless ok do return some "the challenger record's signature does not verify under the verifier's challenger key"
    return none

/-- `trusted_reason` (module docstring): why `c`, a coin commitment the prover presents, is not the one in the
challenger's record; `none` when it is. -/
def trustedReason (key : ChallengerKey) (record : Json) (c : LiveCommitment) : V (Option String) := do
  if let some why ← recordReason key record then return some why
  let t := (← ChallengerRecord.ofJson record).commitment
  unless c.scheme == t.scheme && c.epoch == t.epoch && c.families == t.families && c.root == t.root do
    return some s!"coin commitment {(Flock.hexOf c.root).take 16} is not the one the verifier's challenger recorded ({(Flock.hexOf t.root).take 16})"
  return none

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def request (j : Json) (ks : Array String) : V Unit :=
  unless keys? j == some ks do throw s!"a request's keys are {", ".intercalate ks.toList}"

/-- A key's JSON to its own `toJson`. -/
def keyPiece (j : Json) : V Json := pure <| refusing do pure (← ChallengerKey.ofJson j).toJson

/-- `{"I", "record"}` to the record's own `toJson` and `record_message` under the identifier `I` (lowercase hex). -/
def recordPiece (j : Json) : V Json := do
  request j #["I", "record"]
  let some I := hexOf? (getD j "I") | throw "I is not lowercase hex"
  pure <| refusing do
    let r ← ChallengerRecord.ofJson (getD j "record")
    let m ← recordMessage I r.host r.commitment
    return Json.mkObj [("message", Json.str (Flock.hexOf m)), ("record", r.toJson)]

/-- `{"commitment", "key", "record"}` to `trusted_reason`'s. -/
def trustedPiece (j : Json) : V Json := do
  request j #["commitment", "key", "record"]
  pure <| refusing do
    let key ← ChallengerKey.ofJson (getD j "key")
    let c ← LiveCommitment.ofJson (getD j "commitment")
    return Json.mkObj [("reason", (← trustedReason key (getD j "record") c).elim Json.null Json.str)]

end SampledProofs.Challenger
