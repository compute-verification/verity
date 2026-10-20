/-
The challenger's session: its signed close over the release log, and the auditor's record of which session is a
window's (`challenger.py`'s "The session's close" and "The auditor's record", which `tests/test_challenger_difftest.py`
holds to this, the verifier of record).

The challenger logs each root it receives and each release after it, a line each in canonical JSON (`releases.jsonl`, its
head the commitment, its times integers), and when its session ends, however it ends, signs the log at its key's next
leaf (`challenger.json`):

    close   = {chain, ended, lines, session, signature}       ended: null when the client closed the session, else why not
    chain   = c_n,  c_0 = SHA-512([LOG_TAG]),  c_(i+1) = SHA-512(c_i ‖ [line_i])
    message = SHA-512([CLOSE_ROLE] ‖ I ‖ [record message] ‖ [session] ‖ [u64be(n)] ‖ [chain] ‖ [end])

`end` 0x00 when the client closed the session and 0x01 ‖ ended otherwise. `sessionOf` is `session_of`: the record under
the key (`recordReason`), the close signed under it, the log the one the close signs, line for line, a session its
client closed, and the roots it received (`releasedRootsOf`). A line is refused unless it is its own canonical JSON
(`Flock.canon`) in at most `MAX_LOG_LINE` characters, so Lean's parser and Python's read every accepted line alike.

The auditor's record names one session per window (`auditedSession`, `audited_session`): entries
`{code, commit, key, period, pouw_salt, session, stream}`, the window `{commit, period, pouw_salt, stream}`. The verifier
refuses a window it names no session for, or several; a session it names for two windows; a key other than the window's
pinned one; and a session whose signed close names another id (`trustedSessionOf`, `trusted_session_of`). Each refusal
throws Python's `CoinError` text.

The difftest's pieces: `challenger_close` (`ChallengerClose.from_json` and `close_message`), `challenger_released`
(`log_chain` and `released_roots_of`) and `challenger_session` (`trusted_session_of`), each `{"refused": why}` where
Python refuses.
-/
import SampledProofs.ChallengerRecord

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def CLOSE_ROLE : String := "verity/challenger-close/v1"
def LOG_TAG : String := "verity/challenger/release-log/v1"
def MAX_SESSION_ID : Nat := 128
def MAX_LOG_LINES : Nat := 2 * MAX_COINS + 1
def MAX_LOG_LINE : Nat := 4096
def CLOSE_FORM : String :=
  "a challenger close record is {chain, ended, lines, session, signature}: a 64-byte chain and a signature in " ++
  "lowercase hex, how the session ended (null when its client closed it), the release log's line count, 1 to " ++
  toString MAX_LOG_LINES ++ ", and a session id of at most " ++ toString MAX_SESSION_ID ++
  " ASCII letters, digits, '-', '.' and '_'"
def NO_CLOSE : String :=
  "no challenger close record: a session the challenger didn't close and sign shows nothing of the order it released its coins in"
def LOG_FORM : String := "the challenger's release log is a list of its lines"
def LOG_MALFORMED : String := "the challenger's release log is malformed"
def SESSION_FORM : String :=
  "a challenger session is {close, log, record}: its close record, its release log's lines and its record"
def WINDOW_FORM : String :=
  "a window's identity is {commit, period, pouw_salt, stream}: the Commit, the PoUW salt and the stream, 64 bytes each in lowercase hex, and the period, a nonnegative integer"
def AUDIT_FORM : String :=
  "an auditor's session record is a list of {code, commit, key, period, pouw_salt, session, stream}: a window's identity, the challenger's pinned source commit (40 lowercase hex), the SHA-512 of its public key (lowercase hex) and the session's id"
def NO_AUDIT : String :=
  "no auditor's session record: the verifier trusts a challenger session only as the auditor's record names it"

/-- A session id: a host's characters, at most `MAX_SESSION_ID` of them. -/
def isSessionId (s : String) : Bool := s.length ≤ MAX_SESSION_ID && isHost s

private def hexAt? (j : Json) : Option ByteArray :=
  match j with
  | .str s => (Flock.unhex s).toOption
  | _ => none

/-- `log_chain`: the hash chain over the release log's lines. -/
def logChain (lines : List String) : ByteArray :=
  lines.foldl (fun h l => Flock.Sha512.hash (h ++ field l.toUTF8)) (Flock.Sha512.hash (field LOG_TAG.toUTF8))

/-- `close_message(I, host, commitment, session, lines, chain, ended)`: what the challenger's close signs. -/
def closeMessage (I : ByteArray) (host : String) (c : LiveCommitment) (session : String) (lines : Nat) (chain : ByteArray)
    (ended : Option String) : V ByteArray := do
  let r ← recordMessage I host c
  let fin := match ended with
    | none => ByteArray.mk #[0]
    | some e => ByteArray.mk #[1] ++ e.toUTF8
  return Flock.Sha512.hash (field CLOSE_ROLE.toUTF8 ++ I ++ field r ++ field session.toUTF8 ++
    field (Flock.be64 lines.toUInt64) ++ field chain ++ field fin)

/-- The challenger's signed close of session `session`. -/
structure ChallengerClose where
  session : String
  lines : Nat
  chain : ByteArray
  ended : Option String
  signature : ByteArray

/-- `ChallengerClose.from_json`. -/
def ChallengerClose.ofJson (j : Json) : V ChallengerClose := do
  unless keys? j == some #["chain", "ended", "lines", "session", "signature"] do throw CLOSE_FORM
  let ended : Option (Option String) := match getD j "ended" with
    | .null => some none
    | .str s => some (some s)
    | _ => none
  match hexAt? (getD j "chain"), hexAt? (getD j "signature"), nat? (getD j "lines"), ended, getD j "session" with
  | some chain, some signature, some lines, some ended, .str session =>
    unless chain.size == 64 && 1 ≤ lines && lines ≤ MAX_LOG_LINES && isSessionId session do throw CLOSE_FORM
    pure ⟨session, lines, chain, ended, signature⟩
  | _, _, _, _, _ => throw CLOSE_FORM

def ChallengerClose.toJson (c : ChallengerClose) : Json :=
  Json.mkObj [("chain", Json.str (Flock.hexOf c.chain)), ("ended", c.ended.elim Json.null Json.str),
    ("lines", Lean.toJson c.lines), ("session", Json.str c.session), ("signature", Json.str (Flock.hexOf c.signature))]

/-- A release log line's event: the line, refused unless it is its own canonical JSON in at most `MAX_LOG_LINE`
characters. -/
def event (line : String) : V Json := do
  unless line.length ≤ MAX_LOG_LINE do throw LOG_MALFORMED
  let .ok j := Json.parse line | throw LOG_MALFORMED
  let .ok text := Flock.canon j | throw LOG_MALFORMED
  unless text == line do throw LOG_MALFORMED
  pure j

private def pairs : List String → List (String × String)
  | a :: b :: rest => (a, b) :: pairs rest
  | _ => []

private def isTime (j : Json) : Bool := (nat? j).isSome

/-- The roots the challenger received, by family in order. -/
abbrev Roots := List (String × List ByteArray)

/-- `released_roots_of` (module docstring): the roots the release log `log` shows the challenger received, its head
naming `c`, each root followed by its coin's release. -/
def releasedRootsOf (c : LiveCommitment) (log : List String) : V Roots := do
  let head := "{\"event\":\"commitment\",\"root\":\"" ++ Flock.hexOf c.root ++ "\"}"
  unless log.head? == some head do
    throw s!"the challenger's release log is not coin commitment {(Flock.hexOf c.root).take 16}'s"
  if log.length % 2 == 0 then throw "the challenger's release log ends on a root without its release"
  let mut roots : Array (String × Array ByteArray) := (c.families.map fun (l, _) => (l, #[])).toArray
  for (a, b) in pairs log.tail do
    let r ← event a
    let s ← event b
    let ok := keys? r == some #["event", "index", "label", "received_at", "root"] &&
      keys? s == some #["after", "event", "index", "label", "released_at"] &&
      getD r "event" == Json.str "root" && getD s "event" == Json.str "release" &&
      isTime (getD r "received_at") && isTime (getD s "released_at")
    let .str label := getD r "label" | throw LOG_MALFORMED
    let some k := roots.findIdx? (·.1 == label) | throw LOG_MALFORMED
    let .str root := getD r "root" | throw LOG_MALFORMED
    let some bytes := (Flock.unhex root).toOption | throw LOG_MALFORMED
    let n : Int := (roots[k]?.map (·.2.size)).getD 0
    unless ok && int? (getD r "index") == some n && int? (getD s "index") == some n &&
        getD s "label" == Json.str label && getD s "after" == Json.str root && root.length == 128 do
      throw LOG_MALFORMED
    roots := roots.modify k fun (l, rs) => (l, rs.push bytes)
  return roots.toList.map fun (l, rs) => (l, rs.toList)

def rootsJson (roots : Roots) : Json :=
  Json.mkObj (roots.map fun (l, rs) => (l, Json.arr (rs.toArray.map fun r => Json.str (Flock.hexOf r))))

/-- `session_of` (module docstring): the coin commitment the challenger signed under `key`, the roots its release log
shows it received and its session id, from its record, its release log's lines and its signed close (`null` when the
verifier has none). -/
def sessionOf (key : ChallengerKey) (record log close : Json) : V (LiveCommitment × Roots × String) := do
  if let some why ← recordReason key record then throw why
  let r ← ChallengerRecord.ofJson record
  if close == Json.null then throw NO_CLOSE
  let c ← match ChallengerClose.ofJson close with
    | .ok c => pure c
    | .error e => throw s!"the challenger's close record is malformed: {e}"
  let m ← closeMessage (key.pub.extract 0 16) r.host r.commitment c.session c.lines c.chain c.ended
  let ok : Bool := match Lms.Signature.decode key.params c.signature with
    | .ok sig => (Lms.verify key.params key.pub m sig).toOption == some true
    | .error _ => false
  unless ok do throw "the challenger's close record's signature does not verify under the verifier's challenger key"
  let .arr ls := log | throw LOG_FORM
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM
  unless lines.length == c.lines do
    throw s!"the challenger's release log has {lines.length} lines, not the {c.lines} its close record signs"
  unless logChain lines == c.chain do throw "the challenger's release log is not the one its close record signs"
  if let some e := c.ended then throw s!"the challenger's session ended without its client closing it: {e}"
  return (r.commitment, ← releasedRootsOf r.commitment lines, c.session)

private def isDigest (j : Json) (n : Nat := 128) : Bool :=
  match j with
  | .str s => s.length == n && (Flock.unhex s).toOption.isSome
  | _ => false

def WINDOW_KEYS : List String := ["commit", "period", "pouw_salt", "stream"]

private def isWindowIn (j : Json) : Bool :=
  isDigest (getD j "commit") && isDigest (getD j "pouw_salt") && isDigest (getD j "stream") &&
    (nat? (getD j "period")).isSome

def isWindow (j : Json) : Bool := keys? j == some #["commit", "period", "pouw_salt", "stream"] && isWindowIn j

def isAuditEntry (j : Json) : Bool :=
  keys? j == some #["code", "commit", "key", "period", "pouw_salt", "session", "stream"] && isWindowIn j &&
    isDigest (getD j "code") 40 && isDigest (getD j "key") &&
    (match getD j "session" with | .str s => isSessionId s | _ => false)

/-- `audited_session` (module docstring): the id of the one session the auditor's record `auditor` (`null` when the
verifier has none) names for `window`, under the window's pinned key `key`. -/
def auditedSession (auditor window : Json) (key : ChallengerKey) : V String := do
  if auditor == Json.null then throw NO_AUDIT
  let .arr entries := auditor | throw s!"the auditor's session record is malformed: {AUDIT_FORM}"
  unless entries.all isAuditEntry do throw s!"the auditor's session record is malformed: {AUDIT_FORM}"
  unless isWindow window do throw s!"the window's identity is malformed: {WINDOW_FORM}"
  let named := entries.filter fun e => WINDOW_KEYS.all fun k => jeq (getD e k) (getD window k)
  let some first := named[0]? | throw "the auditor's record names no challenger session for this window"
  if named.size > 1 then
    throw s!"the auditor's record names {named.size} challenger sessions for this window, and a window has one"
  let .str session := getD first "session" | throw s!"the auditor's session record is malformed: {AUDIT_FORM}"
  if (entries.filter (getD · "session" == Json.str session)).size > 1 then
    throw s!"the auditor's record names challenger session {session} for more than one window"
  let pinned := Flock.hexOf (Flock.Sha512.hash key.pub)
  let .str k := getD first "key" | throw s!"the auditor's session record is malformed: {AUDIT_FORM}"
  unless k == pinned do
    throw s!"the auditor's record names challenger key {k.take 16}... for this window's session, not the window's pinned key {pinned.take 16}..."
  return session

/-- `trusted_session_of` (module docstring): the coin commitment and the roots of the one session the auditor's record
names for `window`, from that session's files `{close, log, record}`. -/
def trustedSessionOf (auditor window : Json) (key : ChallengerKey) (session : Json) : V (LiveCommitment × Roots) := do
  let named ← auditedSession auditor window key
  unless keys? session == some #["close", "log", "record"] do throw SESSION_FORM
  let (c, roots, sid) ← sessionOf key (getD session "record") (getD session "log") (getD session "close")
  unless sid == named do
    throw s!"the challenger's session is {sid}, not {named}, the one the auditor's record names for this window"
  return (c, roots)

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def request (j : Json) (ks : Array String) : V Unit :=
  unless keys? j == some ks do throw s!"a request's keys are {", ".intercalate ks.toList}"

/-- `{"I", "close", "commitment", "host"}` to the close's own `toJson` and `close_message` under the identifier `I`. -/
def closePiece (j : Json) : V Json := do
  request j #["I", "close", "commitment", "host"]
  let some I := hexAt? (getD j "I") | throw "I is not lowercase hex"
  let .str host := getD j "host" | throw "host is not a string"
  pure <| refusing do
    let c ← ChallengerClose.ofJson (getD j "close")
    let m ← closeMessage I host (← LiveCommitment.ofJson (getD j "commitment")) c.session c.lines c.chain c.ended
    return Json.mkObj [("close", c.toJson), ("message", Json.str (Flock.hexOf m))]

/-- `{"commitment", "log"}` to `log_chain` and `released_roots_of`'s. -/
def releasedPiece (j : Json) : V Json := do
  request j #["commitment", "log"]
  let .arr ls := getD j "log" | throw "log is not a list of strings"
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw "log is not a list of strings"
  pure <| refusing do
    let roots ← releasedRootsOf (← LiveCommitment.ofJson (getD j "commitment")) lines
    return Json.mkObj [("chain", Json.str (Flock.hexOf (logChain lines))), ("roots", rootsJson roots)]

/-- `{"auditor", "key", "session", "window"}` to `trusted_session_of`'s. -/
def sessionPiece (j : Json) : V Json := do
  request j #["auditor", "key", "session", "window"]
  pure <| refusing do
    let (c, roots) ← trustedSessionOf (getD j "auditor") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
      (getD j "session")
    return Json.mkObj [("commitment", c.toJson), ("roots", rootsJson roots)]

end SampledProofs.Challenger
