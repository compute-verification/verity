/-
A sample's coins from one family of the auditor's challenger, as the verifier replays them from the record (`coins.py`'s
`challenger_stream`, `read_reason` and `coins_reason`, which `tests/test_challenger_difftest.py` holds to this, the
verifier).

The record of a sample's coins is its segment `{after, label, openings, start}` (`Segment.to_json`): family `label`'s coins
`start`, `start` + 1, ..., each opened after `after`, the prover's commitment. `challengerStream` is `challenger_stream`:
the concatenation of the openings' coins, each checked against the challenger's commitment as the family's coin
`start + i`, opened after the commitment the verifier holds (`liveReason`); with the family's roots as the challenger's
signed release log shows it received them, the openings are exactly the coins it released after that commitment,
consecutive, so the prover picked none of them. `readReason` is `read_reason`: the bytes the sample read are the start of
that stream, and they read every coin opened.

`coinsReason` is `coins_reason`: the commitment and the roots come from the code that already trusts them,
`trustedSessionOf` over the evidence `{key, ledger, main, pins, segment, session}` (the verifier's challenger key, the
auditor's ledger, the verifier's copy of main's first-parent history, the host's pins log, the segment and the
session's files) for the window the verifier holds (its open root R, `windowRoot`, which it reads from the challenger's
signed open of the registration's window, never from the record or the evidence), so the record names neither, the
prover picks no window, and the session counts only from a commit the pins log records on main's first-parent line
(`pinnedSession`): coins a session of another window released after the registration's digest are refused. The session
must be a withheld one (`withheldOn`: every ledger open of the window says `withheld: true`), whose challenger released
no root, salt or path before it closed, as `drawCoins` requires. `Run.run` refuses a sample from the challenger whose evidence is missing or gives a
reason (`select`, code `coins`), its `after` the registration's digest.

`withinAt` is `coins_within`: the sizes of what the evidence holds, each within its `Limits` (`LIMITS`, which v2 reads
before anything else of the evidence, `OneStageZk.evidenceOf`): the session's release log lines (`MAX_LOG_LINES`, as
its close counts them) and refused hellos (`MAX_HELLOS`), the ledger's lines (`MAX_LEDGER_LINES`, each signed at its
own leaf of a key at the deployed parameters), the pins log's lines (`MAX_PINS`), the segment's openings (`MAX_COINS`,
each a coin of one layout) and each opening's path (`MAX_PATH`, a layout's depth). A value absent or not a list counts
as none: what reads it refuses it.

The difftest's pieces (`Pieces.table`): `challenger_read` (`read_reason`) and `challenger_draw_coins`
(`coins_reason`), each `{"reason"}`, or `{"refused": why}` where Python raises; `challenger_within` (`coins_within`),
`{"limits", "reason"}`.
-/
module

public import Spot.ChallengerSession

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Challenger

open Lean (Json)
open Flock (V)

def SEGMENT_FORM : String :=
  "a draw's coins are {after, label, openings, start}: the prover's commitment (64 bytes in lowercase hex), the challenger family, its coins' openings and the first one's index"
def COINS_FORM : String :=
  "a draw's coin evidence is {key, ledger, main, pins, segment, session}: the verifier's challenger key, the auditor's ledger, the verifier's copy of main's first-parent history, the host's pins log, the draw's coins (Segment.to_json) and the challenger session's files"

/-- `challenger_stream` (module docstring): the sample's coins from `segment`, opened after `after`, checked against `c`
and, with `roots`, against the family's roots the challenger's log shows. -/
def challengerStream (c : LiveCommitment) (segment : Json) (after : ByteArray) (roots : Option (List ByteArray)) :
    V ByteArray := do
  unless keys? segment == some #["after", "label", "openings", "start"] do throw SEGMENT_FORM
  let (.str label, some start, .arr openings) := (getD segment "label", nat? (getD segment "start"),
      getD segment "openings") | throw SEGMENT_FORM
  unless getD segment "after" == Json.str (Flock.hexOf after) do
    throw "the draw's coins were not opened after the prover's commitment the verifier holds"
  let mut out := ByteArray.empty
  for (doc, i) in openings.toList.zipIdx do
    let o ← LiveOpening.ofJson doc
    if let some why ← liveReason c o label ((start + i : Nat) : Int) after then
      throw s!"the draw's {label} coin {start + i}: {why}"
    out := out ++ o.coin
  if let some rs := roots then
    let afterIt := (rs.zipIdx.filter (·.1 == after)).map (·.2)
    unless afterIt == List.range' start openings.size do
      throw s!"the draw's {label} coins are not the ones the challenger's log shows released after the prover's commitment"
  return out

/-- `read_reason` (module docstring): why `read` is not the start of the sample's coins using every coin opened; `none`
when it is. -/
def readReason (c : LiveCommitment) (segment : Json) (after read : ByteArray) (roots : Option (List ByteArray)) :
    Option String :=
  match challengerStream c segment after roots with
  | .error why => some why
  | .ok stream =>
    if stream.extract 0 read.size != read then some "the bytes the draw read are not its coins'"
    else if (read.size + COIN_BYTES - 1) / COIN_BYTES != stream.size / COIN_BYTES then
      some s!"the draw read {read.size} bytes of {stream.size / COIN_BYTES} coins: a coin opened and not read"
    else none

def WITHHELD : String :=
  "the session the auditor's ledger opens on the window does not withhold its coin commitment until it closes " ++
  "(its open's body has no `withheld: true`: `remote.open_session(..., withhold=True)`)"

/-- `withheld_on`: the ledger opens `window` (its open root), and every open of it marks its session withheld
(`withheld: true` in its body), a session whose challenger released only its coins' raw bytes while it ran, and its
commitment, salts and paths once it closed. The root and `withheld` are compared by a match, not by `BEq Json`, a
`partial def` proofs can't see through. -/
def withheldOn (ledger : Json) (window : String) : Bool :=
  let opens : List LedgerEntry := match ledger with
    | .arr ls => ls.toList.filterMap fun l => (ledgerEntry? l).filter fun e =>
        e.kind == "open" && (match getD (getD e.body "window") "root" with | .str r => r == window | _ => false)
    | _ => []
  !opens.isEmpty && opens.all fun e => match getD e.body "withheld" with | .bool true => true | _ => false

/-- `coins_reason` (module docstring): why `coins`, the evidence of a sample's coins, does not show `read` to be the coins
the session the auditor's ledger opens for window `window` (its open root) released after `after`; `none` when it
does. The session must be a withheld one (`withheldOn`), as `drawCoins`'s. -/
def coinsReason (coins : Json) (window : String) (after read : ByteArray) : Option String :=
  if keys? coins != some #["key", "ledger", "main", "pins", "segment", "session"] then some COINS_FORM else
  let trusted : V (LiveCommitment × Roots) := do
    trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)])
      (← ChallengerKey.ofJson (getD coins "key")) (getD coins "pins") (getD coins "main") (getD coins "session")
  match trusted with
  | .error why => some why
  | .ok (c, roots) =>
    if !withheldOn (getD coins "ledger") window then some WITHHELD else
    let segment := getD coins "segment"
    let rs := match getD segment "label" with
      | .str l => ((roots.find? (·.1 == l)).map (·.2)).getD []
      | _ => []
    readReason c segment after read (some rs)

/-- The most refused hellos a session's evidence holds. -/
def MAX_HELLOS : Nat := 2 ^ 20
/-- The most lines an auditor's ledger holds: each is signed at its own leaf of the challenger's key, of which a key at
the deployed parameters (`Lms.deployed`) has `2 ^ 20`. -/
def MAX_LEDGER_LINES : Nat := 2 ^ 20
/-- The most lines a host's pins log holds. -/
def MAX_PINS : Nat := 2 ^ 20
/-- The longest path a coin opening holds: the depth of a layout of `MAX_COINS` coins. -/
def MAX_PATH : Nat := 20

/-- Bounds on the sizes of a sample's coin evidence (module docstring, `withinAt`). -/
structure Limits where
  log : Nat
  hellos : Nat
  ledger : Nat
  pins : Nat
  openings : Nat
  path : Nat

def LIMITS : Limits := ⟨MAX_LOG_LINES, MAX_HELLOS, MAX_LEDGER_LINES, MAX_PINS, MAX_COINS, MAX_PATH⟩

def Limits.toJson (l : Limits) : Json :=
  Json.mkObj [("hellos", Lean.toJson l.hellos), ("ledger", Lean.toJson l.ledger), ("log", Lean.toJson l.log),
    ("openings", Lean.toJson l.openings), ("path", Lean.toJson l.path), ("pins", Lean.toJson l.pins)]

/-- The refusal of evidence past `l`. -/
def Limits.form (l : Limits) : String :=
  s!"a draw's coin evidence holds at most {l.log} release log lines, {l.hellos} refused hellos, {l.ledger} ledger " ++
    s!"lines, {l.pins} pins log lines and {l.openings} openings, each opening's path at most {l.path} digests"

/-- A JSON list's length; 0 for anything else. -/
def arrSize : Json → Nat
  | .arr a => a.size
  | _ => 0

/-- `coins_within` (module docstring): every size `coins` holds within `l`. -/
def withinAt (l : Limits) (coins : Json) : Bool :=
  let session := getD coins "session"
  let openings := getD (getD coins "segment") "openings"
  arrSize (getD session "log") ≤ l.log && arrSize (getD session "hellos") ≤ l.hellos &&
    arrSize (getD coins "ledger") ≤ l.ledger && arrSize (getD coins "pins") ≤ l.pins &&
    arrSize openings ≤ l.openings &&
    match openings with
    | .arr os => os.all fun o => arrSize (getD o "path") ≤ l.path
    | _ => true

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def reasonJson (r : Option String) : Json := Json.mkObj [("reason", r.elim Json.null Json.str)]

private def request (j : Json) (ks : Array String) : V Unit :=
  unless keys? j == some ks do throw s!"a request's keys are {", ".intercalate ks.toList}"

private def lowerHex (j : Json) (what : String) : V ByteArray := do
  let .str s := j | throw s!"{what} is not lowercase hex"
  Flock.unhex s

/-- `{"after", "commitment", "read", "roots", "segment"}` to `read_reason`'s, `roots` null or a list. -/
def readPiece (j : Json) : V Json := do
  request j #["after", "commitment", "read", "roots", "segment"]
  let after ← lowerHex (getD j "after") "after"
  let read ← lowerHex (getD j "read") "read"
  let roots ← match getD j "roots" with
    | .null => pure none
    | .arr rs => some <$> rs.toList.mapM (lowerHex · "a root")
    | _ => throw "roots is not a list"
  pure <| refusing do
    return reasonJson (readReason (← LiveCommitment.ofJson (getD j "commitment")) (getD j "segment") after read roots)

/-- `{"after", "coins", "read", "window"}` to `coins_reason`'s, `window` the window's open root. -/
def drawCoinsPiece (j : Json) : V Json := do
  request j #["after", "coins", "read", "window"]
  let after ← lowerHex (getD j "after") "after"
  let read ← lowerHex (getD j "read") "read"
  let .str window := getD j "window" | throw "window is not a string"
  return reasonJson (coinsReason (getD j "coins") window after read)

/-- `{"coins", "limits"}` to `coins_within`'s, `limits` `{hellos, ledger, log, openings, path, pins}` or null for
`LIMITS`: the limits read and the refusal past them. -/
def withinPiece (j : Json) : V Json := do
  request j #["coins", "limits"]
  let l ← match getD j "limits" with
    | .null => pure LIMITS
    | ls => do
      request ls #["hellos", "ledger", "log", "openings", "path", "pins"]
      let n (k : String) : V Nat := match nat? (getD ls k) with
        | some v => pure v
        | none => throw s!"limits: {k} is not a natural number"
      pure ⟨← n "log", ← n "hellos", ← n "ledger", ← n "pins", ← n "openings", ← n "path"⟩
  return Json.mkObj [("limits", l.toJson),
    ("reason", if withinAt l (getD j "coins") then Json.null else Json.str l.form)]

end Spot.Challenger
