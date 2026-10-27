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
