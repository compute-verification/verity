/-
A draw's live coins as a component's check reads them: `coinsReason`'s path with the verifier's own key, window and
family, never the evidence's.

`drawCoins key window label after coins` reads the evidence `{ledger, main, pins, segment, session}`: the auditor's
ledger, the verifier's copy of main's first-parent history, the host's pins log, the draw's segment and the session's
files. The session is the one the ledger opens on `window` under `key` (`trustedSessionOf`), a withheld one
(`withheldOn`: its challenger released no root, salt or path before it closed). The segment must be of
family `label`, and its openings exactly the coins that session's signed log shows released after `after`
(`challengerStream`). The result is their concatenation, the bytes a draw reads. A component whose check reads its
coins through this (`Core.Live`) takes its draw from the challenger's record, and
`Properties.Challenger.DrawnByDevice` says what that record shows: the coins the device drew, or a break.
-/
import SampledProofs.ChallengerCoins

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def DRAW_FORM : String :=
  "a draw's coin evidence is {ledger, main, pins, segment, session}: the auditor's ledger, the verifier's copy of " ++
  "main's first-parent history, the host's pins log, the draw's coins (Segment.to_json) and the challenger session's files"

def WITHHELD : String :=
  "the session the auditor's ledger opens on the window does not withhold its coin commitment until it closes " ++
  "(its open's body has no `withheld: true`: `remote.open_session(..., withhold=True)`)"

/-- `withheld_on`: the ledger opens `window` (its open root), and every open of it marks its session withheld
(`withheld: true` in its body), a session whose challenger released only its coins' raw bytes while it ran, and its
commitment, salts and paths once it closed. -/
def withheldOn (ledger : Json) (window : String) : Bool :=
  let opens : List LedgerEntry := match ledger with
    | .arr ls => ls.toList.filterMap fun l => (ledgerEntry? l).filter fun e =>
        e.kind == "open" && getD (getD e.body "window") "root" == Json.str window
    | _ => []
  !opens.isEmpty && opens.all fun e => getD e.body "withheld" == Json.bool true

/-- The coins of family `label` that the session the auditor's ledger opens on `window` (its open root, lowercase hex)
under `key` released after `after`, read from `coins`, in order: the bytes a draw reads. The session must be a withheld
one (`withheldOn`). -/
def drawCoins (key : ChallengerKey) (window label : String) (after : ByteArray) (coins : Json) : V ByteArray := do
  unless keys? coins == some #["ledger", "main", "pins", "segment", "session"] do throw DRAW_FORM
  let (c, roots) ← trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
    (getD coins "pins") (getD coins "main") (getD coins "session")
  unless withheldOn (getD coins "ledger") window do throw WITHHELD
  let segment := getD coins "segment"
  unless getD segment "label" == Json.str label do
    throw s!"the draw's coins are not the challenger's {label} family's"
  challengerStream c segment after (some (((roots.find? (·.1 == label)).map (·.2)).getD []))

end SampledProofs.Challenger
