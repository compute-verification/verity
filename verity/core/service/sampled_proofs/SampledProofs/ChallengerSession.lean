/-
The challenger's session: its signed close over the release log, and the auditor's signed ledger of which session is a
window's (`challenger.py`'s "The session's close" and "The auditor's ledger", which `tests/test_challenger_difftest.py`
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

The auditor's ledger names the one session each window's coins come from (`auditedSession`, `audited_session`): infra's
`windows.jsonl`, a canonical JSON line per entry `{body, kind, prev, schema, seq, session, sig}`, an open naming the
window `{commit, period}`, the challenger's host and the SHA-512 of its key's public key, an end whether the session
closed and the SHA-512 of each file it left, each entry signed under the challenger's key:

    message = SHA-512([LEDGER_ROLE] ‖ I ‖ [identity_digest_sha512(LEDGER_TAG, entry without sig)])

The key is the one the window's deployment pins, from outside the ledger: an open naming another key's SHA-512 is
refused before any signature is read, so a ledger signed under its own key can't pin itself. Then each entry is
chained (`seq` from 0, `prev` the SHA-512 of the previous line) and signed, a window and a session opened once, a session
ended once after its open; the window's session must be opened and ended closed, so a ledger older than its end is
refused. The session's files must be that session's (`sessionNamed`): its signed close names the ledger's session id,
its record the open's host, and its release log is the one the end sealed. Each refusal throws Python's `CoinError`
text.

The difftest's pieces: `challenger_close` (`ChallengerClose.from_json` and `close_message`), `challenger_released`
(`log_chain` and `released_roots_of`), `challenger_ledger` (`audited_session`) and `challenger_session`
(`trusted_session_of`), each `{"refused": why}` where Python refuses.
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
def LOG_FILE : String := "releases.jsonl"
def LEDGER_SCHEMA : String := "verity/challenger-auditor/v1"
def LEDGER_ROLE : String := "verity/challenger-auditor-record/v1"
def LEDGER_TAG : String := "verity/challenger-auditor/entry/v1"
/-- An auditor's ledger line's most characters: its signature at W = 1, H = 30 is 70928 hex digits. -/
def MAX_LEDGER_LINE : Nat := 262144
def ZERO : String := String.ofList (List.replicate 128 '0')
def WINDOW_FORM : String :=
  "a window's identity is {commit, period}: the served Commit's root, 64 bytes in lowercase hex, and the " ++
  "period, 1 to 64 ASCII letters, digits, '.', '_', ':' and '-', the first a letter or digit"
def LEDGER_FORM : String :=
  "an auditor's ledger line is canonical JSON {body, kind, prev, schema, seq, session, sig}: kind open, its " ++
  "body naming the window, the challenger's host, the pinned source commit (40 lowercase hex) and the SHA-512 " ++
  "of the challenger key's public key (key_sha512), or kind end, its body whether its client closed the " ++
  "session (closed) and the SHA-512 of each file the session left (files)"
def NO_AUDIT : String :=
  "no auditor's ledger: the verifier trusts a challenger session only as the auditor's signed ledger names it"

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

private def isPeriod (s : String) : Bool :=
  1 ≤ s.length && s.length ≤ 64 && !"._:-".contains s.front &&
    s.all fun c => c.isAlphanum && c.toNat < 128 || c == '.' || c == '_' || c == ':' || c == '-'

/-- `_window`: a window's identity `{commit, period}`. -/
def isWindow (j : Json) : Bool :=
  keys? j == some #["commit", "period"] && isDigest (getD j "commit") &&
    (match getD j "period" with | .str s => isPeriod s | _ => false)

/-- `ledger_message(I, entry)`: what an auditor's ledger entry's `sig` signs, `entry` without its `sig`. -/
def ledgerMessage (I : ByteArray) (entry : Json) : V ByteArray := do
  unless I.size == 16 do throw s!"an LMS identifier is 16 bytes, not {I.size}"
  let d ← taggedSha512 LEDGER_TAG (Json.mkObj ((entries entry).toList.filter (·.1 != "sig")))
  return Flock.Sha512.hash (field LEDGER_ROLE.toUTF8 ++ I ++ field d)

/-- An auditor's ledger entry as its line holds it. -/
structure LedgerEntry where
  entry : Json
  kind : String
  session : String
  sig : ByteArray
  body : Json

private def isOpenBody (b : Json) : Bool :=
  isWindow (getD b "window") && (match getD b "host" with | .str h => isHost h | _ => false) &&
    isDigest (getD b "pinned") 40 && isDigest (getD b "key_sha512")

private def isEndBody (b : Json) : Bool :=
  (match getD b "closed" with | .bool _ => true | _ => false) &&
    (match getD b "files" with | .obj _ => (entries (getD b "files")).all (isDigest ·.2) | _ => false)

/-- `_ledger_entry`: a ledger line's entry when it is its own canonical JSON of the ledger's form, in at most
`MAX_LEDGER_LINE` characters. -/
def ledgerEntry? (line : Json) : Option LedgerEntry := do
  let .str text := line | none
  guard (text.length ≤ MAX_LEDGER_LINE)
  let e ← (Json.parse text).toOption
  guard ((Flock.canon e).toOption == some text)
  guard (keys? e == some #["body", "kind", "prev", "schema", "seq", "session", "sig"])
  let .str _ := getD e "schema" | none
  let _ ← nat? (getD e "seq")
  guard (isDigest (getD e "prev"))
  let .str session := getD e "session" | none
  guard (isSessionId session)
  let .str sig := getD e "sig" | none
  let sig ← (Flock.unhex sig).toOption
  let body := getD e "body"
  let .obj _ := body | none
  let .str kind := getD e "kind" | none
  guard (kind == "open" && isOpenBody body || kind == "end" && isEndBody body)
  pure ⟨e, kind, session, sig, body⟩

private def verifies (key : ChallengerKey) (m sig : ByteArray) : Bool :=
  match Lms.Signature.decode key.params sig with
  | .ok s => (Lms.verify key.params key.pub m s).toOption == some true
  | .error _ => false

/-- The window's session as the auditor's ledger names it: its id, the host its open names, and the SHA-512 of each
file its end sealed (`Audited`). -/
structure Audited where
  session : String
  host : String
  files : Json

/-- `audited_session` (module docstring): the one challenger session the auditor's ledger `ledger` (its lines, `null`
when the verifier has none) opens for `window` under the window's pinned key `key`, and ends closed. -/
def auditedSession (ledger window : Json) (key : ChallengerKey) : V Audited := do
  if ledger == Json.null then throw NO_AUDIT
  let .arr lines := ledger | throw s!"the auditor's ledger is malformed: {LEDGER_FORM}"
  unless isWindow window do throw s!"the window's identity is malformed: {WINDOW_FORM}"
  let pinned := Flock.hexOf (Flock.Sha512.hash key.pub)
  let mut es : Array LedgerEntry := #[]
  for (line, i) in lines.zipIdx do
    let some e := ledgerEntry? line | throw s!"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}"
    if e.kind == "open" then
      let .str k := getD e.body "key_sha512" | throw s!"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}"
      unless k == pinned do
        throw s!"the auditor's ledger entry {i} opens session {e.session} under challenger key {k.take 16}..., not the window's pinned key {pinned.take 16}..."
    es := es.push e
  let mut opens : Array LedgerEntry := #[]
  let mut ends : Array LedgerEntry := #[]
  let mut prev := ZERO
  for ((line, e), i) in (lines.zip es).zipIdx do
    unless getD e.entry "schema" == Json.str LEDGER_SCHEMA && nat? (getD e.entry "seq") == some i &&
        getD e.entry "prev" == Json.str prev do
      throw s!"the auditor's ledger entry {i} is not entry {i} of a {LEDGER_SCHEMA} ledger chained to its predecessor"
    unless verifies key (← ledgerMessage (key.pub.extract 0 16) e.entry) e.sig do
      throw s!"the auditor's ledger entry {i}'s signature does not verify under the verifier's challenger key"
    let s := e.session
    if e.kind == "open" then
      if opens.any (fun o => o.session == s || jeq (getD o.body "window") (getD e.body "window")) then
        throw s!"the auditor's ledger entry {i} opens a second session for its window, or session {s} again: a window has one session"
      opens := opens.push e
    else
      if !opens.any (·.session == s) || ends.any (·.session == s) then
        throw s!"the auditor's ledger entry {i} ends session {s}, which it has not opened or has ended"
      ends := ends.push e
    let .str text := line | throw s!"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}"
    prev := Flock.hexOf (Flock.Sha512.hash text.toUTF8)
  let some o := opens.find? fun o => jeq (getD o.body "window") window
    | throw "the auditor's ledger opens no challenger session for this window"
  let s := o.session
  let some e := ends.find? (·.session == s)
    | throw s!"the auditor's ledger has no end for session {s}: the session never ended, or the ledger is older than its end"
  unless getD e.body "closed" == Json.bool true do
    throw s!"the auditor's ledger ends session {s} unclosed: its client didn't close it"
  let .str host := getD o.body "host" | throw s!"the auditor's ledger is malformed: {LEDGER_FORM}"
  return ⟨s, host, getD e.body "files"⟩

/-- `log_digest`: the SHA-512 of the release log file whose lines are `lines`, each ended by a newline. -/
def logDigest (lines : List String) : String :=
  Flock.hexOf (Flock.Sha512.hash (String.join (lines.map (· ++ "\n"))).toUTF8)

/-- `session_named` (module docstring): the coin commitment and the roots of session `a`, from its files
`{close, log, record}`, when its close names the ledger's session, its record the open's host and its release log is the
one the ledger's end sealed. -/
def sessionNamed (a : Audited) (key : ChallengerKey) (session : Json) : V (LiveCommitment × Roots) := do
  unless keys? session == some #["close", "log", "record"] do throw SESSION_FORM
  let (c, roots, sid) ← sessionOf key (getD session "record") (getD session "log") (getD session "close")
  unless sid == a.session do
    throw s!"the challenger's session is {sid}, not {a.session}, the one the auditor's ledger opens for this window"
  let r ← ChallengerRecord.ofJson (getD session "record")
  unless r.host == a.host do
    throw s!"the challenger's record names host {r.host}, not {a.host}, the one the auditor's ledger opened its session on"
  let .arr ls := getD session "log" | throw LOG_FORM
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM
  unless getD a.files LOG_FILE == Json.str (logDigest lines) do
    throw "the challenger's release log is not the one the auditor's ledger sealed at its session's end"
  return (c, roots)

/-- `trusted_session_of` (module docstring): the coin commitment and the roots of the one session the auditor's ledger
opens for `window` under `key`, from that session's files `{close, log, record}`. -/
def trustedSessionOf (ledger window : Json) (key : ChallengerKey) (session : Json) : V (LiveCommitment × Roots) := do
  sessionNamed (← auditedSession ledger window key) key session

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

/-- `{"key", "ledger", "window"}` to `audited_session`'s: the session, its host and the files its end sealed. -/
def ledgerPiece (j : Json) : V Json := do
  request j #["key", "ledger", "window"]
  pure <| refusing do
    let a ← auditedSession (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
    return Json.mkObj [("files", a.files), ("host", Json.str a.host), ("session", Json.str a.session)]

/-- `{"key", "ledger", "session", "window"}` to `trusted_session_of`'s. -/
def sessionPiece (j : Json) : V Json := do
  request j #["key", "ledger", "session", "window"]
  pure <| refusing do
    let (c, roots) ← trustedSessionOf (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
      (getD j "session")
    return Json.mkObj [("commitment", c.toJson), ("roots", rootsJson roots)]

end SampledProofs.Challenger
