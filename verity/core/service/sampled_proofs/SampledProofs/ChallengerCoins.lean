/-
A draw's coins from one family of the auditor's challenger, as the verifier replays them from the record (`coins.py`'s
`challenger_stream`, `read_reason` and `coins_reason`, which `tests/test_challenger_difftest.py` holds to this, the
verifier of record).

The record of a draw's coins is its segment `{after, label, openings, start}` (`Segment.to_json`): family `label`'s coins
`start`, `start` + 1, ..., each opened after `after`, the prover's commitment. `challengerStream` is `challenger_stream`:
the concatenation of the openings' coins, each checked against the challenger's commitment as the family's coin
`start + i`, opened after the commitment the verifier holds (`liveReason`); with the family's roots as the challenger's
signed release log shows it received them, the openings are exactly the coins it released after that commitment,
consecutive, so the prover picked none of them. `readReason` is `read_reason`: the bytes the draw read are the start of
that stream, and they read every coin opened.

`coinsReason` is `coins_reason`: the commitment and the roots come from the code that already trusts them,
`trustedSessionOf` over the evidence `{key, ledger, main, pins, segment, session}` (the verifier's challenger key, the
auditor's ledger, the verifier's copy of main's first-parent history, the host's pins log, the segment and the
session's files) for the window the verifier holds (its open root R, `windowRoot`, which it reads from the challenger's
signed open of the registration's window, never from the record or the evidence), so the record names neither, the
prover picks no window, and the session counts only from a commit the pins log records on main's first-parent line
(`pinnedSession`): coins a session of another window released after the registration's digest are refused. `Run.run` refuses a draw from the challenger whose evidence is missing or gives a
reason (`select`, code `coins`), its `after` the registration's digest.

The difftest's pieces (`Pieces.table`): `challenger_read` (`read_reason`) and `challenger_draw_coins`
(`coins_reason`), each `{"reason"}`, or `{"refused": why}` where Python raises.
-/
import SampledProofs.ChallengerSession

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def SEGMENT_FORM : String :=
  "a draw's coins are {after, label, openings, start}: the prover's commitment (64 bytes in lowercase hex), the challenger family, its coins' openings and the first one's index"
def COINS_FORM : String :=
  "a draw's coin evidence is {key, ledger, main, pins, segment, session}: the verifier's challenger key, the auditor's ledger, the verifier's copy of main's first-parent history, the host's pins log, the draw's coins (Segment.to_json) and the challenger session's files"

/-- `challenger_stream` (module docstring): the draw's coins from `segment`, opened after `after`, checked against `c`
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

/-- `read_reason` (module docstring): why `read` is not the start of the draw's coins using every coin opened; `none`
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

/-- `coins_reason` (module docstring): why `coins`, the evidence of a draw's coins, does not show `read` to be the coins
the session the auditor's ledger opens for window `window` (its open root) released after `after`; `none` when it
does. -/
def coinsReason (coins : Json) (window : String) (after read : ByteArray) : Option String :=
  if keys? coins != some #["key", "ledger", "main", "pins", "segment", "session"] then some COINS_FORM else
  let trusted : V (LiveCommitment × Roots) := do
    trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)])
      (← ChallengerKey.ofJson (getD coins "key")) (getD coins "pins") (getD coins "main") (getD coins "session")
  match trusted with
  | .error why => some why
  | .ok (c, roots) =>
    let segment := getD coins "segment"
    let rs := match getD segment "label" with
      | .str l => ((roots.find? (·.1 == l)).map (·.2)).getD []
      | _ => []
    readReason c segment after read (some rs)

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

end SampledProofs.Challenger
