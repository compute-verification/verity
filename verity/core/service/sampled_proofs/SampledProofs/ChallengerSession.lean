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
`windows.jsonl`, a line per entry `{body, kind, prev, schema, seq, session, sig}` as the auditor writes it (`ledger_text`,
`PyJson.canonical`), schema v1 or v2 and no v1 after a v2. A reserve records a window's request, its start and its expiry
(`later`); an open names the window `{root}` (v1's `{commit, period}`), the challenger's host, the SHA-512 of its key's
public key and in v2 its prover (`for`, else `by`), and after the ledger's first reserve binds its session's reserve
(its seq), carries its request (`RESERVED`), is opened by its prover and inside its minutes; an end whether the
session closed, the SHA-512 of each file it left, and in v2 the leaf its record was signed at, its commitment's root and
how many hellos it refused; a lane's period is a `period-open` (its cap and circuit) and a `period-close` listing its
windows with their outcomes. Each entry is signed under the challenger's key:

    message = SHA-512([LEDGER_ROLE] ‖ I ‖ [identity_digest_sha512(LEDGER_TAGS[schema], entry without sig)])

The key is the one the window's deployment pins, from outside the ledger: an open or a period-open naming another key's
SHA-512 is refused before any signature is read, so a ledger signed under its own key can't pin itself. Then the ledger
is infra's `_verify` (`verifyLedger`): each entry chained (`seq` from 0, `prev` the SHA-512 of the previous line) and
signed at strictly increasing leaves (`u32(q)`, the signature's first four bytes), a root and a session opened once, a
session ended once after its open with its record's leaf between the two and no entry's, and each period's close
listing exactly its lane's windows, counted at their reserves and at most its cap, a reserve never opened with root
null. The window's session must be opened under v2 and ended closed,
naming its record's leaf, commitment and refused hellos and sealing `SEALED_FILES`, so a ledger older than its end is
refused. The session's files must be that session's (`sessionNamed`): its signed close names the ledger's session id,
its record the open's host and was signed at that leaf over that commitment, and its release log and refused hellos are
the ones the end sealed. `ledgerHead` takes infra's published heads: each checked, any two extending one another, the
longest. A window counts toward its period only when the descriptor of the circuit its sessions prove hashes (SHA-512)
to the circuit the period's open committed (`creditedSession`, `credited_session`). A session counts as main's code
only when the host's pins log, whose first lines its open seals (`pins: {lines, sha512}`), pins the open's `pinned`
commit (`pinsReason`) and the verifier's own copy of main's first-parent history (`main`, the auditor's
`git rev-list --first-parent origin/main`) holds it (`firstParentReason`): a branch commit merged into main is in
main's history but on no tree main's check ran (`pinnedSession`, `pinned_session`). Only the open's pin is held: a repin
of the host after the open neither helps nor harms the session, credited by the commit its open sealed. Every path that
credits a session reads it there: `creditedSession` (the pin first, then the circuit) and `trustedSessionOf`. Each
refusal throws Python's `CoinError` text.

The difftest's pieces: `challenger_close` (`ChallengerClose.from_json` and `close_message`), `challenger_released`
(`log_chain` and `released_roots_of`), `challenger_ledger` (`audited_session`), `challenger_head` (`ledger_head`),
`challenger_session` (`trusted_session_of`), `challenger_credited` (`credited_session`) and `challenger_pinned`
(`pinned_session`), each `{"refused": why}` where Python refuses.
-/
import SampledProofs.ChallengerRecord
import SampledProofs.PyJson

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
  "a challenger session is {close, hellos, log, record}: its close record, its refused hellos' lines, its " ++
  "release log's lines and its record"
def LOG_FILE : String := "releases.jsonl"
def RECORD_FILE : String := "record.json"
def CLOSE_FILE : String := "challenger.json"
def HELLOS_FILE : String := "hellos.jsonl"
/-- The files of a session the verifier reads, each of which the ledger's end must seal. -/
def SEALED_FILES : List String := [RECORD_FILE, CLOSE_FILE, LOG_FILE, HELLOS_FILE]
def LEDGER_SCHEMA_V1 : String := "verity/challenger-auditor/v1"
def LEDGER_SCHEMA : String := "verity/challenger-auditor/v2"
def LEDGER_ROLE : String := "verity/challenger-auditor-record/v1"
/-- `LEDGER_TAGS`: each schema's entry tag. -/
def ledgerTag? : String → Option String
  | "verity/challenger-auditor/v1" => some "verity/challenger-auditor/entry/v1"
  | "verity/challenger-auditor/v2" => some "verity/challenger-auditor/entry/v2"
  | _ => none
/-- An auditor's ledger line's most characters: its signature at W = 1, H = 30 is 70928 hex digits. -/
def MAX_LEDGER_LINE : Nat := 262144
/-- `CAP`: a period's cap is 1 to this many windows. -/
def MAX_CAP : Nat := 2 ^ 24
/-- `MINUTES`: a reserve's minutes are 1 to this many. -/
def MAX_MINUTES : Nat := 720
/-- `RESERVED`: the request a reserve records and its session's open carries. -/
def RESERVED : List String := ["by", "for", "peer", "scheme", "epoch", "families", "minutes"]
def ZERO : String := String.ofList (List.replicate 128 '0')
def WINDOW_FORM : String := "a window's identity is {root}: the served Commit's root, 64 bytes in lowercase hex"
def LEDGER_FORM : String :=
  "an auditor's ledger line is an entry {body, kind, prev, schema, seq, session, sig} as the auditor writes it " ++
  "(sorted keys, no whitespace, ASCII, no number but an integer): kind open, its body naming the window, the " ++
  "challenger's host, the pinned source commit (40 lowercase hex), the SHA-512 of the challenger key's public " ++
  "key (key_sha512) and in v2 its opener (by, and for when opened for a lane); kind reserve, its body the " ++
  "window's request; kind end, its body whether its client closed the session (closed), the SHA-512 of each " ++
  "file the session left (files) and in v2 the leaf its record was signed at, if any; kind period-open, its " ++
  "period, lane and key_sha512; or kind period-close, its period, lane, windows ({complete, root, session} " ++
  "each, root null for a reserve never opened) and whether all are complete"
def END_FORM : String :=
  "a window's end names the leaf its record was signed at (record_leaf), its commitment's root (commitment), how " ++
  "many hellos it refused (refused_hellos) and seals " ++ ", ".intercalate SEALED_FILES
def HEADS_FORM : String := "the auditor's published heads are a list of ledgers, each its lines"
def HELLOS_FORM : String := "the challenger's refused hellos are a list of their lines"
def PINS_FORM : String :=
  "the host's pins log is a list of its lines, each {commit, main, pinned_utc} as the auditor writes it: the " ++
  "commit the host was pinned to and the origin/main commit whose history held it (40 lowercase hex each), " ++
  "and when, YYYY-MM-DDTHH:MM:SSZ"
def NO_PINS : String :=
  "no pins log: the verifier trusts a challenger session only from a tree the host's pins log records as a " ++
  "commit of main's history"
def NO_AUDIT : String :=
  "no auditor's ledger: the verifier trusts a challenger session only as the auditor's signed ledger names it"
def MAIN_FORM : String :=
  "a list of main's first-parent commits, each 40 lowercase hex (`git rev-list --first-parent origin/main` in " ++
  "the auditor's own checkout)"
def NO_MAIN : String :=
  "the verifier holds no copy of main's history, so it can't tell whether the host's pinned tree is one main's " ++
  "check passed: give it the auditor's own `git rev-list --first-parent origin/main`"

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

/-- `_LEDGER_NAME`: a period or a lane, 1 to 64 ASCII letters, digits, '.', '_', ':' and '-', the first a letter or
digit. -/
private def isLedgerName (s : String) : Bool :=
  s.length ≤ 64 && (match s.toList with | c :: _ => !"._:-".contains c | [] => false) &&
    s.all fun c => c.isAlphanum && c.toNat < 128 || c == '.' || c == '_' || c == ':' || c == '-'

private def nameAt (j : Json) : Bool := match j with | .str s => isLedgerName s | _ => false

private def isStr (j : Json) : Bool := match j with | .str _ => true | _ => false

private def strOf (j : Json) : String := match j with | .str s => s | _ => ""

/-- `j == Json.bool true`, by a match the kernel reduces (`BEq Json` is a `partial def`, opaque to the kernel). -/
private def jsonTrue (j : Json) : Bool := match j with | .bool true => true | _ => false

example : jsonTrue (getD (Json.mkObj [("closed", .bool true)]) "closed") = true := by decide
example : jsonTrue (getD (Json.mkObj [("closed", .bool false)]) "closed") = false := by decide
example : jsonTrue (getD (Json.mkObj [("files", .null)]) "closed") = false := by decide
example : (getD (Json.mkObj [("record_leaf", .null)]) "record_leaf").isNull = true := by decide
example : (getD (Json.mkObj [("schema", .str "x")]) "record_leaf").isNull = true := by decide
example : (getD (Json.mkObj [("record_leaf", .str "7")]) "record_leaf").isNull = false := by decide

/-- `_window`: a window's identity `{root}`. -/
def isWindow (j : Json) : Bool := keys? j == some #["root"] && isDigest (getD j "root")

/-- `_v1_window`: a v1 open's window `{commit, period}`. -/
private def isV1Window (j : Json) : Bool :=
  keys? j == some #["commit", "period"] && isDigest (getD j "commit") && nameAt (getD j "period")

/-- `cap_of`'s domain: an integer, not a boolean, from 1 to `MAX_CAP`. -/
def isCap (j : Json) : Bool := match int? j with | some v => 1 ≤ v && v ≤ MAX_CAP | none => false

/-- `circuit_of`'s fixed points: a SHA-512's 128 hex digits in lowercase, never a Program's id. -/
def isCircuit (j : Json) : Bool :=
  match j with
  | .str s => s.length == 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')
  | _ => false

/-- `_RESERVED_SESSION`: a reserved session's id, `s[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}`. -/
def isReservedSession (s : String) : Bool :=
  let pat := "sddddddddTddddddZ-hhhh".toList
  s.length == pat.length && (s.toList.zip pat).all fun (c, p) =>
    match p with
    | 'd' => c.isDigit
    | 'h' => c.isDigit || ('a' ≤ c && c ≤ 'f')
    | p => c == p

private def leap (y : Nat) : Bool := y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)

private def daysIn (y m : Nat) : Nat :=
  if m == 2 then (if leap y then 29 else 28) else if m == 4 || m == 6 || m == 9 || m == 11 then 30 else 31

private def digitsAt (cs : List Char) (i n : Nat) : Option Nat :=
  let ds := (cs.drop i).take n
  if ds.length == n && ds.all Char.isDigit then some (ds.foldl (fun a c => a * 10 + (c.toNat - '0'.toNat)) 0) else none

private def pad (w n : Nat) : String := let s := toString n; "".pushn '0' (w - s.length) ++ s

/-- `later(t, minutes)`: `t` (`YYYY-MM-DDTHH:MM:SSZ`, a UTC time Python's `datetime` holds, years 1 to 9999) plus
`minutes` in that form, a reserve's expiry; `none` when `t` isn't such a time or the sum passes year 9999. -/
def later (t : String) (minutes : Nat) : Option String := do
  let cs := t.toList
  guard (cs.length == 20 && cs[4]? == some '-' && cs[7]? == some '-' && cs[10]? == some 'T' &&
    cs[13]? == some ':' && cs[16]? == some ':' && cs[19]? == some 'Z')
  let y ← digitsAt cs 0 4
  let mo ← digitsAt cs 5 2
  let d ← digitsAt cs 8 2
  let h ← digitsAt cs 11 2
  let mi ← digitsAt cs 14 2
  let s ← digitsAt cs 17 2
  guard (1 ≤ y && 1 ≤ mo && mo ≤ 12 && 1 ≤ d && d ≤ daysIn y mo && h ≤ 23 && mi ≤ 59 && s ≤ 59)
  let total := h * 60 + mi + minutes
  let (y, mo, d) := (List.range (total / 1440)).foldl (fun (y, mo, d) _ =>
    if d < daysIn y mo then (y, mo, d + 1) else if mo < 12 then (y, mo + 1, 1) else (y + 1, 1, 1)) (y, mo, d)
  guard (y ≤ 9999)
  let r := total % 1440
  return s!"{pad 4 y}-{pad 2 mo}-{pad 2 d}T{pad 2 (r / 60)}:{pad 2 (r % 60)}:{pad 2 s}Z"

/-- `ledger_message(I, entry)`: what an auditor's ledger entry's `sig` signs, `entry` without its `sig`, under its schema's
tag. -/
def ledgerMessage (I : ByteArray) (entry : Json) : V ByteArray := do
  unless I.size == 16 do throw s!"an LMS identifier is 16 bytes, not {I.size}"
  let .str schema := getD entry "schema" | throw "an entry's schema is a string"
  let some tag := ledgerTag? schema | throw s!"schema {pyRepr schema} is none of the ledger's"
  let d ← taggedSha512 tag (Json.mkObj ((entries entry).toList.filter (·.1 != "sig")))
  return Flock.Sha512.hash (field LEDGER_ROLE.toUTF8 ++ I ++ field d)

/-- `leaf_of`: the LMS leaf a signature was made at, its first four bytes. -/
def leafOf (sig : ByteArray) : Nat := (sig.extract 0 4).toList.foldl (fun a b => a * 256 + b.toNat) 0

/-- An auditor's ledger entry as its line holds it. -/
structure LedgerEntry where
  entry : Json
  text : String
  schema : String
  seq : Nat
  prev : String
  kind : String
  session : String
  sig : ByteArray
  body : Json
  deriving Inhabited

/-- Whether `text` holds `\ud8` to `\udf`, the escape of a UTF-16 surrogate, anywhere. -/
private def escapesSurrogate (text : String) : Bool := Id.run do
  let cs := text.toList.toArray
  for i in [0:cs.size] do
    if cs[i]! == '\\' && cs.getD (i + 1) ' ' == 'u' && cs.getD (i + 2) ' ' == 'd' &&
        "89abcdef".contains (cs.getD (i + 3) ' ') then
      return true
  return false

private def isOutcome (w : Json) : Bool :=
  keys? w == some #["complete", "root", "session"] && (getD w "root" == Json.null || isDigest (getD w "root")) &&
    (match getD w "session" with | .str s => isSessionId s | _ => false) &&
    (match getD w "complete" with | .bool _ => true | _ => false)

private def isBodyOf (kind : String) (v2 : Bool) (b : Json) : Bool :=
  match kind with
  | "open" =>
    (if v2 then isWindow (getD b "window") else isV1Window (getD b "window")) &&
      (match getD b "host" with | .str h => isHost h | _ => false) && isDigest (getD b "pinned") 40 &&
      isDigest (getD b "key_sha512") &&
      (!v2 || isStr (getD b "by") && (match field? b "for" with | none => true | some j => isStr j))
  | "reserve" => true
  | "end" =>
    (match getD b "closed" with | .bool _ => true | _ => false) &&
      (match getD b "files" with | .obj _ => (entries (getD b "files")).all (isDigest ·.2) | _ => false) &&
      (!v2 || getD b "record_leaf" == Json.null || (nat? (getD b "record_leaf")).isSome)
  | "period-open" => nameAt (getD b "period") && nameAt (getD b "lane") && isDigest (getD b "key_sha512")
  | "period-close" =>
    nameAt (getD b "period") && nameAt (getD b "lane") &&
      (match getD b "windows" with | .arr ws => ws.all isOutcome | _ => false) &&
      (match getD b "complete" with | .bool _ => true | _ => false)
  | _ => false

/-- `_ledger_entry`: a ledger line's entry when it is its own `ledger_text` (`PyJson.canonical`), escapes no surrogate,
has only integers (`Flock.canon`), in at most `MAX_LEDGER_LINE` characters, and is of the ledger's form: a session's
entry names its session, a period's names none. -/
def ledgerEntry? (line : Json) : Option LedgerEntry := do
  let .str text := line | none
  guard (text.length ≤ MAX_LEDGER_LINE)
  let e ← (Json.parse text).toOption
  guard ((PyJson.canonical e).toOption == some text && !escapesSurrogate text && (Flock.canon e).toOption.isSome)
  guard (keys? e == some #["body", "kind", "prev", "schema", "seq", "session", "sig"])
  let .str schema := getD e "schema" | none
  let _ ← ledgerTag? schema
  let seq ← nat? (getD e "seq")
  let .str prev := getD e "prev" | none
  guard (isDigest (.str prev))
  let .str sigHex := getD e "sig" | none
  guard (8 ≤ sigHex.length)
  let sig ← (Flock.unhex sigHex).toOption
  let body := getD e "body"
  let .obj _ := body | none
  let .str kind := getD e "kind" | none
  let session ← match kind, getD e "session" with
    | "reserve", .str s | "open", .str s | "end", .str s => if isSessionId s then some s else none
    | "period-open", .null | "period-close", .null => some ""
    | _, _ => none
  guard (isBodyOf kind (schema == LEDGER_SCHEMA) body)
  pure ⟨e, text, schema, seq, prev, kind, session, sig, body⟩

/-- `verify`'s at LMS over the key's parameters, false for a signature that doesn't decode. -/
private def verifies (key : ChallengerKey) (m sig : ByteArray) : Bool :=
  match Lms.Signature.decode key.params sig with
  | .ok s => (Lms.verify key.params key.pub m s).toOption == some true
  | .error _ => false

/-- A window `verifyLedger` read: its root, its open and the leaf the open was signed at, and its end once read. -/
structure Win where
  root : String
  opened : LedgerEntry
  leaf : Nat
  ended : Option LedgerEntry := none
  deriving Inhabited

/-- A monitoring period `verifyLedger` read: its name and lane, its windows' sessions, counted at their reserves (or
one-step opens), whether it closed, and the cap and circuit its open committed (`null` both for a period opened before
they were required). -/
structure MonitoringPeriod where
  name : String
  lane : String
  windows : Array String := #[]
  closed : Bool := false
  cap : Json
  circuit : Json
  deriving Inhabited

/-- A reserve `verifyLedger` read: its session, its entry, and the period it joined, if any. -/
structure Reserve where
  session : String
  entry : LedgerEntry
  period : Option String
  deriving Inhabited

/-- `lane_of`: a reserve's or a v2 open's prover, its `for` when nonempty, else its `by`. -/
def laneOf (body : Json) : String := if truthy (getD body "for") then strOf (getD body "for") else strOf (getD body "by")

/-- `_verify`'s `join`: lane `lane`'s open period, if any, takes window `session` at entry `i`, refused past its cap;
the periods after it, and the period's name. -/
private def join (periods : Array MonitoringPeriod) (current : Array (String × String)) (i : Nat) (lane session : String) :
    V (Array MonitoringPeriod × Option String) := do
  let some (_, name) := current.find? (·.1 == lane) | return (periods, none)
  let some k := periods.findIdx? (·.name == name) | return (periods, none)
  let p := periods[k]!
  if let some cap := int? p.cap then
    if cap ≤ p.windows.size then throw s!"entry {i}: period {name}'s window {p.windows.size} is past its cap {cap}"
  return (periods.modify k fun p => { p with windows := p.windows.push session }, some name)

/-- A reserve's request as `_verify` requires it: its prover's names, minutes 1 to `MAX_MINUTES`, and its expiry
`later(started_utc, minutes)`. -/
private def isReserve (b : Json) : Bool :=
  nameAt (getD b "by") && (match field? b "for" with | none => true | some j => nameAt j) &&
    match int? (getD b "minutes"), getD b "started_utc" with
    | some m, .str t =>
      1 ≤ m && m ≤ MAX_MINUTES && (match later t m.toNat with
        | some x => getD b "expires_utc" == Json.str x
        | none => false)
    | _, _ => false

/-- Whether an open's `started_utc` is a time in `later`'s form from its reserve's `start` to its `expiry`. -/
private def inReserve (t : Json) (start expiry : String) : Bool :=
  match t with
  | .str t => later t 0 == some t && !decide (t < start) && !decide (expiry < t)
  | _ => false

/-- `_verify` (module docstring), over entries `ledgerEntry?` read: the ledger's windows and periods, or the first entry
that fails, in `_verify`'s words. -/
def verifyLedger (es : Array LedgerEntry) (key : ChallengerKey) : V (Array Win × Array MonitoringPeriod) := do
  let I := key.pub.extract 0 16
  let mut wins : Array Win := #[]
  let mut leaves : Array Nat := #[]
  let mut records : Array (Nat × Nat) := #[]
  let mut periods : Array MonitoringPeriod := #[]
  let mut current : Array (String × String) := #[]
  let mut reserves : Array Reserve := #[]
  let mut prev := ZERO
  let mut last : Int := -1
  let mut v2 := false
  let mut twoStep := false
  for (e, i) in es.zipIdx do
    unless !(v2 && e.schema == LEDGER_SCHEMA_V1) && e.seq == i && e.prev == prev do
      throw s!"entry {i}: not entry {i} of a {LEDGER_SCHEMA} ledger chained to its predecessor"
    v2 := v2 || e.schema == LEDGER_SCHEMA
    let q := leafOf e.sig
    unless verifies key (← ledgerMessage I e.entry) e.sig do
      throw s!"entry {i}: its signature doesn't verify under the challenger key"
    unless last < q do throw s!"entry {i}: signed at leaf {q}, not after the last entry's {last}"
    let s := e.session
    match e.kind with
    | "reserve" =>
      unless e.schema == LEDGER_SCHEMA && isReservedSession s && !reserves.any (·.session == s) &&
          !wins.any (·.opened.session == s) do
        throw s!"entry {i}: a reserve is v2 and the first entry of its session {s}"
      unless isReserve e.body do
        throw s!"entry {i}: a reserve names its prover, its minutes, its start and its expiry (start + minutes)"
      twoStep := true
      let (ps, period) ← join periods current i (laneOf e.body) s
      periods := ps
      reserves := reserves.push { session := s, entry := e, period }
    | "open" =>
      let w := getD e.body "window"
      let root := strOf (getD w (if e.schema == LEDGER_SCHEMA then "root" else "commit"))
      if wins.any (·.root == root) || wins.any (·.opened.session == s) then
        throw s!"entry {i}: a second open for root {root.take 16}… or session {s}"
      let b := e.body
      if e.schema == LEDGER_SCHEMA && (field? b "reserve").isSome then
        let some r := reserves.find? (·.session == s)
          | throw s!"entry {i}: an open for session {s} binds no reserve of it"
        unless int? (getD b "reserve") == some (Int.ofNat r.entry.seq) do
          throw s!"entry {i}: an open for session {s} binds no reserve of it"
        let rb := r.entry.body
        unless RESERVED.all (fun k => jeq (getD b k) (getD rb k)) do
          throw s!"entry {i}: session {s}'s open doesn't carry its reserve's request"
        let who := getD b "opened_by"
        let lane := if let .str w := who then some w else none
        let shown := (lane.map pyRepr).getD "no lane"
        unless lane.any (fun w => w == strOf (getD rb "by") || field? rb "for" == some (Json.str w)) do
          throw s!"entry {i}: session {s} was opened by {shown}, not its reserve's prover"
        unless inReserve (getD b "started_utc") (strOf (getD rb "started_utc")) (strOf (getD rb "expires_utc")) do
          throw s!"entry {i}: session {s} was opened outside its reserve's minutes"
        if let some name := r.period then
          if periods.any (fun p => p.name == name && p.closed) then
            throw s!"entry {i}: session {s} was opened after its period {name} closed"
      else if e.schema == LEDGER_SCHEMA && twoStep then
        throw s!"entry {i}: a v2 open without a reserve after the ledger's first reserve"
      else if e.schema == LEDGER_SCHEMA then
        let (ps, _) ← join periods current i (laneOf b) s
        periods := ps
      wins := wins.push { root, opened := e, leaf := q }
    | "end" =>
      let some k := wins.findIdx? (·.opened.session == s)
        | throw s!"entry {i}: an end for session {s}, which has no open or has ended"
      let w := wins[k]!
      if w.ended.isSome then throw s!"entry {i}: an end for session {s}, which has no open or has ended"
      let r := getD e.body "record_leaf"
      if e.schema == LEDGER_SCHEMA && !r.isNull then
        let some r := nat? r | throw s!"entry {i}: record leaf {r.compress} isn't between its open's and its end's, once"
        unless w.leaf < r && r < q && !records.any (·.1 == r) do
          throw s!"entry {i}: record leaf {r} isn't between its open's and its end's, once"
        records := records.push (r, i)
      wins := wins.modify k fun w => { w with ended := some e }
    | "period-open" | "period-close" =>
      let name := strOf (getD e.body "period")
      let lane := strOf (getD e.body "lane")
      unless e.schema == LEDGER_SCHEMA do throw s!"entry {i}: a {e.kind} is v2 and names its period and its lane"
      if e.kind == "period-open" then
        if periods.any (·.name == name) || current.any (·.1 == lane) then
          throw s!"entry {i}: period {name} is named twice, or lane {lane} has a period open"
        let cap := getD e.body "cap"
        let circuit := getD e.body "circuit"
        if (cap != Json.null || circuit != Json.null) && !(isCap cap && isCircuit circuit) then
          throw s!"entry {i}: period {name}'s cap is a positive integer and its circuit a SHA-512 in lowercase"
        periods := periods.push { name, lane, cap, circuit }
        current := current.push (lane, name)
      else
        let some k := periods.findIdx? (·.name == name)
          | throw s!"entry {i}: a period-close for {name}, which isn't lane {lane}'s open period"
        let p := periods[k]!
        if p.closed || p.lane != lane then
          throw s!"entry {i}: a period-close for {name}, which isn't lane {lane}'s open period"
        let outcomes := p.windows.map fun ses =>
          let w := wins.find? (·.opened.session == ses)
          let complete := match w.bind (·.ended) with
            | some x => getD x.body "complete" == Json.bool true
            | none => false
          Json.mkObj [("complete", Json.bool complete), ("root", (w.map (Json.str ·.root)).getD Json.null),
            ("session", Json.str ses)]
        let complete := !outcomes.isEmpty && outcomes.all (getD · "complete" == Json.bool true)
        unless jeq (getD e.body "windows") (Json.arr outcomes) && getD e.body "complete" == Json.bool complete do
          throw s!"entry {i}: period {name}'s close doesn't list exactly its windows and their outcomes"
        periods := periods.modify k fun p => { p with closed := true }
        current := current.filter (·.1 != lane)
    | kind => throw s!"entry {i}: kind {pyRepr kind}"
    leaves := leaves.push q
    prev := Flock.hexOf (Flock.Sha512.hash e.text.toUTF8)
    last := q
  let reused := records.filter fun (r, _) => leaves.contains r
  if let some (r, i) := reused.foldl (fun m x => match m with
      | some y => if x.1 < y.1 then some x else some y
      | none => some x) none then
    throw s!"entry {i}: record leaf {r} is also an entry's"
  return (wins, periods)

/-- `_checked`: `ledger` (its lines) as `verifyLedger` reads it under the pinned key `key`, each line first refused unless
`ledgerEntry?` reads it and, opening a window or a period, it names no other key. -/
def checked (ledger : Json) (key : ChallengerKey) : V (Array Win × Array MonitoringPeriod) := do
  if ledger == Json.null then throw NO_AUDIT
  let .arr lines := ledger | throw s!"the auditor's ledger is malformed: {LEDGER_FORM}"
  let pinned := Flock.hexOf (Flock.Sha512.hash key.pub)
  let mut es : Array LedgerEntry := #[]
  for (line, i) in lines.zipIdx do
    let some e := ledgerEntry? line | throw s!"the auditor's ledger entry {i} is malformed: {LEDGER_FORM}"
    if e.kind == "open" || e.kind == "period-open" then
      let k := strOf (getD e.body "key_sha512")
      unless k == pinned do
        throw s!"the auditor's ledger entry {i} is signed under challenger key {k.take 16}..., not the window's pinned key {pinned.take 16}..."
    es := es.push e
  match verifyLedger es key with
  | .ok r => pure r
  | .error why => throw s!"the auditor's ledger is refused: {why}"

/-- The window's session as the auditor's ledger names it (`Audited`): its id, the host its open names, its prover, the
SHA-512 of each file its end sealed, the leaf its record was signed at, its commitment's root, how many hellos it
refused, and the period holding the window with the cap and circuit its open committed (`none` and `null` when no
period holds it). -/
structure Audited where
  session : String
  host : String
  prover : String
  files : Json
  recordLeaf : Nat
  commitment : String
  refusedHellos : Nat
  pinned : String
  period : Option String
  cap : Json
  circuit : Json

/-- `audited_session` (module docstring): the one challenger session the auditor's ledger `ledger` (its lines, `null`
when the verifier has none) opens under v2 for `window` (`{root}`) under the window's pinned key `key`, and ends closed,
its end naming its record's leaf, its commitment and its refused hellos and sealing `SEALED_FILES`, in a period, if any,
that committed its cap and circuit; and the body of the window's open (`_audited`). -/
def auditedOpen (ledger window : Json) (key : ChallengerKey) : V (Audited × Json) := do
  unless isWindow window do throw s!"the window's identity is malformed: {WINDOW_FORM}"
  let (wins, periods) ← checked ledger key
  let root := strOf (getD window "root")
  let some w := wins.find? (·.root == root) | throw "the auditor's ledger opens no challenger session for this window"
  let o := w.opened
  let s := o.session
  unless o.schema == LEDGER_SCHEMA do
    throw s!"the auditor's ledger opens this window's session {s} under {o.schema}, which names no prover and seals no refused hellos"
  let some e := w.ended
    | throw s!"the auditor's ledger has no end for session {s}: the session never ended, or the ledger is older than its end"
  let b := e.body
  unless jsonTrue (getD b "closed") do
    throw s!"the auditor's ledger ends session {s} unclosed: its client didn't close it"
  let files := getD b "files"
  let malformed := s!"the auditor's ledger's end of session {s} is malformed: {END_FORM}"
  let some recordLeaf := nat? (getD b "record_leaf") | throw malformed
  let some refusedHellos := nat? (getD b "refused_hellos") | throw malformed
  unless isDigest (getD b "commitment") && SEALED_FILES.all (fun f => (field? files f).isSome) do throw malformed
  let p := periods.find? (·.windows.contains s)
  if let some p := p then
    if p.cap == Json.null || p.circuit == Json.null then
      throw s!"the auditor's ledger's period {p.name}, which holds session {s}, commits no cap or circuit"
  return ({ session := s, host := strOf (getD o.body "host"), prover := laneOf o.body, files, recordLeaf,
            commitment := strOf (getD b "commitment"), refusedHellos, pinned := strOf (getD o.body "pinned"),
            period := p.map (·.name), cap := (p.map (·.cap)).getD Json.null,
            circuit := (p.map (·.circuit)).getD Json.null }, o.body)

/-- `audited_session` (module docstring): `auditedOpen`'s session. -/
def auditedSession (ledger window : Json) (key : ChallengerKey) : V Audited :=
  return (← auditedOpen ledger window key).1

/-- `extends`: whether `ledger` is `published` with entries appended. -/
def extendsHead (published ledger : Array Json) : Bool :=
  published.size ≤ ledger.size && ledger.extract 0 published.size == published

/-- `ledger_head` (module docstring): the auditor's ledger as infra published it, every head of `heads` (each its lines;
`null` when the verifier has none) passing `checked` under `key`, any two extending one another, and of them the first
longest. -/
def ledgerHead (heads : Json) (key : ChallengerKey) : V Json := do
  if heads == Json.null || heads == Json.arr #[] then throw NO_AUDIT
  let .arr hs := heads | throw HEADS_FORM
  for (h, i) in hs.zipIdx do
    match checked h key with
    | .ok _ => pure ()
    | .error why => throw (if hs.size == 1 then why else s!"the auditor's published head {i} is refused: {why}")
  let lines (h : Json) : Array Json := match h with | .arr ls => ls | _ => #[]
  for i in [0:hs.size] do
    for j in [i + 1:hs.size] do
      unless extendsHead (lines hs[i]!) (lines hs[j]!) || extendsHead (lines hs[j]!) (lines hs[i]!) do
        throw s!"the auditor's published heads {i} and {j} don't extend one another: its ledger was forked or rewritten"
  return hs.foldl (fun best h => if (lines best).size < (lines h).size then h else best) hs[0]!

/-- `log_digest`: the SHA-512 of the file whose lines are `lines`, each ended by a newline. -/
def logDigest (lines : List String) : String :=
  Flock.hexOf (Flock.Sha512.hash (String.join (lines.map (· ++ "\n"))).toUTF8)

/-- `session_named` (module docstring): the coin commitment and the roots of session `a`, from its files
`{close, hellos, log, record}`, when its close names the ledger's session, its record the open's host, the record was
signed at the leaf the ledger's end names over the commitment it names, and its release log and refused hellos are the
ones the end sealed, as many hellos as the end counts. -/
def sessionNamed (a : Audited) (key : ChallengerKey) (session : Json) : V (LiveCommitment × Roots) := do
  unless keys? session == some #["close", "hellos", "log", "record"] do throw SESSION_FORM
  let (c, roots, sid) ← sessionOf key (getD session "record") (getD session "log") (getD session "close")
  unless sid == a.session do
    throw s!"the challenger's session is {sid}, not {a.session}, the one the auditor's ledger opens for this window"
  let r ← ChallengerRecord.ofJson (getD session "record")
  unless r.host == a.host do
    throw s!"the challenger's record names host {r.host}, not {a.host}, the one the auditor's ledger opened its session on"
  let q := leafOf r.signature
  let root := Flock.hexOf c.root
  unless q == a.recordLeaf && root == a.commitment do
    throw s!"the challenger's record is signed at leaf {q} over commitment {root.take 16}..., not at leaf {a.recordLeaf} over {a.commitment.take 16}..., the ones the auditor's ledger's end names"
  let .arr ls := getD session "log" | throw LOG_FORM
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM
  unless getD a.files LOG_FILE == Json.str (logDigest lines) do
    throw "the challenger's release log is not the one the auditor's ledger sealed at its session's end"
  let .arr hs := getD session "hellos" | throw HELLOS_FORM
  let hellos ← hs.toList.mapM fun | .str s => pure s | _ => throw HELLOS_FORM
  unless getD a.files HELLOS_FILE == Json.str (logDigest hellos) && hellos.length == a.refusedHellos do
    throw s!"the challenger's refused hellos are not the {a.refusedHellos} the auditor's ledger sealed at its session's end"
  return (c, roots)


/-- `circuit_reason` (module docstring): why a window doesn't count toward `period` (`none`: no period holds it), whose
open committed `circuit`, given the bytes of the descriptor of the circuit its sessions prove; `none` when their SHA-512
is that circuit. One hash over the descriptor, never a walk over the circuit it unrolls to. -/
def circuitReason (period : Option String) (circuit : Json) (descriptor : ByteArray) : Option String :=
  match period with
  | none => some ("the auditor's ledger holds the window's session in no period, so it counts toward none: open the " ++
      "lane's period (period-open --cap N --circuit SHA512) before reserving the window")
  | some p =>
    if !isCircuit circuit then
      some s!"period {p} commits no circuit, so no window counts toward it: open a period whose --circuit is the declared circuit's SHA-512"
    else
      let got := Flock.hexOf (Flock.Sha512.hash descriptor)
      let c := strOf circuit
      if got != c then
        some s!"the circuit the window's sessions prove (descriptor SHA-512 {got.take 16}...) is not {c.take 16}..., the one period {p} committed: prove the period's declared circuit, or open a period whose --circuit is this descriptor's SHA-512"
      else none


/-- `_pin`: a pins log line's commit when the line is its own `ledger_text` (`PyJson.canonical`) of the pins log's form,
`{commit, main, pinned_utc}`, two commits in 40 lowercase hex and a time `later` holds fixed. -/
def pinOf? (line : Json) : Option String := do
  let .str text := line | none
  let e ← (Json.parse text).toOption
  guard ((PyJson.canonical e).toOption == some text && keys? e == some #["commit", "main", "pinned_utc"])
  guard (isDigest (getD e "commit") 40 && isDigest (getD e "main") 40)
  let .str t := getD e "pinned_utc" | none
  guard (later t 0 == some t)
  return strOf (getD e "commit")

/-- `pins_reason` (module docstring): why session `session`, whose open's body is `opened`, didn't run main's code by the
host's pins log `pins` (its lines, `null` when the verifier has none); `none` when one of the lines the open sealed
(`{lines, sha512}`, `log_digest` of the log's first lines) pins the open's `pinned`. -/
def pinsReason (session : String) (opened pins : Json) : Option String := Id.run do
  if pins == Json.null then return some NO_PINS
  let .arr ls := pins | return some s!"the host's pins log is malformed: {PINS_FORM}"
  let mut commits : Array String := #[]
  for (line, i) in ls.zipIdx do
    let some c := pinOf? line | return some s!"the host's pins log line {i} is malformed: {PINS_FORM}"
    commits := commits.push c
  let sj := getD opened "pins"
  let sealed := match nat? (getD sj "lines") with
    | some n => if keys? sj == some #["lines", "sha512"] && 1 ≤ n && isDigest (getD sj "sha512") then some n else none
    | none => none
  let some n := sealed
    | return some (s!"the auditor's ledger's open of session {session} seals no pins log (pins: " ++
        "{lines, sha512}): open the window on a host whose gate seals its pins log in each open")
  if ls.size < n then
    return some s!"the host's pins log has {ls.size} lines, fewer than the {n} the open of session {session} sealed: take infra's newest published pins log"
  let lines := (ls.extract 0 n).toList.map strOf
  unless getD sj "sha512" == Json.str (logDigest lines) do
    return some s!"the host's pins log is not the one the open of session {session} sealed: its first {n} lines were edited or rewritten"
  let pinned := strOf (getD opened "pinned")
  unless (commits.extract 0 n).contains pinned do
    return some s!"session {session} ran commit {pinned}, which the host's pins log doesn't record as a commit of main's history: repin the challenger host to a commit origin/main contains (vy-challenger-pin) and rerun the window there"
  return none

/-- `first_parent_reason` (module docstring): why session `session`, run from commit `pinned`, didn't run a tree main's
check passed by the verifier's own copy of main's first-parent history `main` (its commits, `null` when the verifier has
none); `none` when `pinned` is on it. Ancestry is not enough: a branch commit merged into main is in its history, but
main's check ran on the merge. -/
def firstParentReason (session pinned : String) (main : Json) : Option String :=
  let malformed := s!"the verifier's copy of main's history is malformed: {MAIN_FORM}"
  match main with
  | .null => some NO_MAIN
  | .arr cs =>
    if !cs.all (isDigest · 40) then some malformed
    else if cs.contains (Json.str pinned) then none
    else some (s!"session {session} ran commit {pinned}, which the host's pins log records but which is not on " ++
      "main's first-parent line in the verifier's copy (`git rev-list --first-parent origin/main`): a branch commit " ++
      "merged into main is in main's history but no tree main's check ran on; repin the challenger host to a commit " ++
      "on main's first-parent line (vy-challenger-pin) and rerun the window there")
  | _ => some malformed

/-- `pinned_session` (module docstring): `auditedSession`, when the host's pins log `pins` records the open's pin
(`pinsReason`) and the verifier's copy of main's first-parent history `main` holds it (`firstParentReason`). -/
def pinnedSession (ledger window : Json) (key : ChallengerKey) (pins main : Json) : V Audited := do
  let (a, opened) ← auditedOpen ledger window key
  if let some why := pinsReason a.session opened pins then throw why
  if let some why := firstParentReason a.session (strOf (getD opened "pinned")) main then throw why
  return a

/-- `trusted_session_of` (module docstring): the coin commitment and the roots of the one session the auditor's ledger
opens for `window` under `key`, run from a commit the host's pins log `pins` records on main's first-parent line `main`
(`pinnedSession`), from that session's files `{close, hellos, log, record}`. -/
def trustedSessionOf (ledger window : Json) (key : ChallengerKey) (pins main session : Json)
    : V (LiveCommitment × Roots) := do
  sessionNamed (← pinnedSession ledger window key pins main) key session

/-- `credited_session` (module docstring): `pinnedSession`, counted toward its period when `descriptor` hashes to the
circuit the period committed (`circuitReason`): the pin first, then the circuit. -/
def creditedSession (ledger window : Json) (key : ChallengerKey) (pins main : Json) (descriptor : ByteArray)
    : V Audited := do
  let a ← pinnedSession ledger window key pins main
  if let some why := circuitReason a.period a.circuit descriptor then throw why
  return a

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

private def auditedJson (a : Audited) : Json :=
  Json.mkObj [("cap", a.cap), ("circuit", a.circuit), ("commitment", Json.str a.commitment),
    ("files", a.files), ("host", Json.str a.host), ("period", a.period.elim Json.null Json.str),
    ("pinned", Json.str a.pinned), ("prover", Json.str a.prover), ("record_leaf", Lean.toJson a.recordLeaf),
    ("refused_hellos", Lean.toJson a.refusedHellos), ("session", Json.str a.session)]

/-- `{"key", "ledger", "window"}` to `audited_session`'s `Audited`, its fields by Python's names. -/
def ledgerPiece (j : Json) : V Json := do
  request j #["key", "ledger", "window"]
  pure <| refusing do
    return auditedJson (← auditedSession (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key")))

/-- `{"descriptor", "key", "ledger", "main", "pins", "window"}` to `credited_session`'s `Audited`, the descriptor in
lowercase hex. -/
def creditedPiece (j : Json) : V Json := do
  request j #["descriptor", "key", "ledger", "main", "pins", "window"]
  let some d := hexAt? (getD j "descriptor") | throw "descriptor is not lowercase hex"
  pure <| refusing do
    return auditedJson (← creditedSession (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
      (getD j "pins") (getD j "main") d)

/-- `{"key", "ledger", "main", "pins", "window"}` to `pinned_session`'s `Audited`. -/
def pinnedPiece (j : Json) : V Json := do
  request j #["key", "ledger", "main", "pins", "window"]
  pure <| refusing do
    return auditedJson (← pinnedSession (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
      (getD j "pins") (getD j "main"))

/-- `{"heads", "key"}` to `ledger_head`'s, `{"ledger": lines}`. -/
def headPiece (j : Json) : V Json := do
  request j #["heads", "key"]
  pure <| refusing do
    let k ← ChallengerKey.ofJson (getD j "key")
    return Json.mkObj [("ledger", ← ledgerHead (getD j "heads") k)]

/-- `{"key", "ledger", "main", "pins", "session", "window"}` to `trusted_session_of`'s. -/
def sessionPiece (j : Json) : V Json := do
  request j #["key", "ledger", "main", "pins", "session", "window"]
  pure <| refusing do
    let (c, roots) ← trustedSessionOf (getD j "ledger") (getD j "window") (← ChallengerKey.ofJson (getD j "key"))
      (getD j "pins") (getD j "main") (getD j "session")
    return Json.mkObj [("commitment", c.toJson), ("roots", rootsJson roots)]

end SampledProofs.Challenger
