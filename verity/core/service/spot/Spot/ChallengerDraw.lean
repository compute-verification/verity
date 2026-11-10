/-
A sample's live coins as a component's check reads them: `coinsReason`'s path with the verifier's own key, window and
family, never the evidence's.

`drawCoins key window label after coins` reads the evidence `{ledger, main, pins, segment, session}`: the auditor's
ledger, the verifier's copy of main's first-parent history, the host's pins log, the sample's segment and the session's
files. The session is the one the ledger opens on `window` under `key` (`trustedSessionOf`), a withheld one
(`withheldOn`: its challenger released no root, salt or path before it closed). The segment must be of
family `label`, and its openings exactly the coins that session's signed log shows released after `after`
(`challengerStream`). The result is their concatenation, the bytes a sample reads. A component whose check reads its
coins through this (`Core.Live`) takes its sample from the challenger's record, and
`Properties.Challenger.DrawnByDevice` says what that record shows: the coins the device drew, or a break.

`seedSaltOf key window coins` reads a window's seed salt the same way: the one coin of the challenger's
`SEED_SALT_FAMILY`, a family no sample reads, that the session released after the window's open root itself, which
R6 holds the window's registrations to name (`Registration.Verifier.seedSalt`).
-/
module

public import Spot.ChallengerCoins

@[expose] public section

namespace Spot.Challenger

open Lean (Json)
open Flock (V)

def DRAW_FORM : String :=
  "a draw's coin evidence is {ledger, main, pins, segment, session}: the auditor's ledger, the verifier's copy of " ++
  "main's first-parent history, the host's pins log, the draw's coins (Segment.to_json) and the challenger session's files"

/-- The coins of family `label` that the session the auditor's ledger opens on `window` (its open root, lowercase hex)
under `key` released after `after`, read from `coins`, in order: the bytes a sample reads. The session must be a withheld
one (`withheldOn`). -/
def drawCoins (key : ChallengerKey) (window label : String) (after : ByteArray) (coins : Json) : V ByteArray := do
  unless keys? coins == some #["ledger", "main", "pins", "segment", "session"] do throw DRAW_FORM
  let (c, roots) ← trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
    (getD coins "pins") (getD coins "main") (getD coins "session")
  unless withheldOn (getD coins "ledger") window do throw WITHHELD
  let segment := getD coins "segment"
  unless jsonStrIs (getD segment "label") label do
    throw s!"the draw's coins are not the challenger's {label} family's"
  challengerStream c segment after (some (((roots.find? (·.1 == label)).map (·.2)).getD []))

/-- The challenger's family of a window's seed salt, which no sample reads. -/
def SEED_SALT_FAMILY : String := "seed-salt"
def SEED_SALT_WINDOW : String := "a window's open root is lowercase hex of 64 bytes"
def SEED_SALT_ONE : String :=
  "a window's seed salt is one coin, its family's first, released after the window's open root: the challenger " ++
  "released another"

/-- A window's seed salt as the verifier reads it from the challenger's log (`coins.seed_salt_of`): the `SEED_SALT_FAMILY`
coins that the session the auditor's ledger opens on `window` (its open root) under `key` released after that open root
(`drawCoins`), which must be one coin, the family's first. So the salt is the one coin the challenger released for it,
and a server that saw others could pick none of them. -/
def seedSaltOf (key : ChallengerKey) (window : String) (coins : Json) : V ByteArray := do
  unless window.length == 128 && isHex window do throw SEED_SALT_WINDOW
  let salt ← drawCoins key window SEED_SALT_FAMILY (← Flock.unhex window) coins
  unless nat? (getD (getD coins "segment") "start") == some 0 && salt.size == COIN_BYTES do throw SEED_SALT_ONE
  return salt

/-- `{"coins", "key", "window"}` to `seed_salt_of`'s: `{"reason": null, "salt"}`, the salt in hex, or `{"reason", "salt":
null}` where it raises; `{"refused"}` for a key `ChallengerKey.ofJson` refuses, and refused for a request in another
form. -/
def seedSaltPiece (j : Json) : V Json := do
  unless keys? j == some #["coins", "key", "window"] do throw "a request's keys are coins, key, window"
  let .str window := getD j "window" | throw "window is not a string"
  return match ChallengerKey.ofJson (getD j "key") with
    | .error why => Json.mkObj [("refused", Json.str why)]
    | .ok key => match seedSaltOf key window (getD j "coins") with
      | .ok salt => Json.mkObj [("reason", Json.null), ("salt", Json.str (Flock.hexOf salt))]
      | .error why => Json.mkObj [("reason", Json.str why), ("salt", Json.null)]

end Spot.Challenger
