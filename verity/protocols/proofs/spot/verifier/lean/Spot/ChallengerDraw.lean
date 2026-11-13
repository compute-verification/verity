/-
A sample's live coins as a component's check reads them (`drawCoins`), with the verifier's own key, window and family,
never the evidence's, and as a stream's run holds the bytes it read to them (`coinsReason`): `coins.py`'s `draw_coins`,
`seed_salt_of` and `coins_reason`, which `tests/test_challenger_difftest.py` holds to this, the verifier.

`drawCoins key window label after coins` reads the evidence `{ledger, main, pins, segment, session}`: the auditor's
ledger, the verifier's copy of main's first-parent history, the host's pins log, the sample's segment and the session's
files. The session is the one the ledger opens on `window` under `key` (`trustedChallengePhaseOf`), a withheld one
(`withheldOn`: its challenger released no root, salt or path before it closed). The segment must be of
family `label`, and its openings exactly the coins that session's signed log shows released after `after`
(`challengerStream`). The result is their concatenation, the bytes a sample reads. A component whose check reads its
coins through this (`Core.Live`) takes its sample from the challenger's record, and
`Properties.Challenger.DrawnByDevice` says what that record shows: the coins the device drew, or a break.

`coinsReason key main coins window label after read` is `coins_reason`: why the bytes `read` are not the start of what
`drawCoins` reads from the same evidence, every coin of it read (`startReason`), under the verifier's challenger key and
its copy of main's first-parent history, which its caller holds and hands it (`key`, `main`). The evidence is
`drawCoins`'s, one object for both readers, so it names no key: evidence that carries one is refused in form, and the
session is one signed under the verifier's key, whoever wrote the evidence. The copy of main's history the evidence
carries must be the verifier's own, commit for commit (`sameMain`), so the session counts only from a commit the pins
log records on main's first-parent line as the verifier holds it (`pinnedChallengePhase`), and what `drawCoins` reads from that
evidence is read under the verifier's copy too. The session is the one of the window the verifier holds (its open root
R, `windowRoot`, which it reads from the challenger's signed open of the registration's window, never from the record
or the evidence), so the record names neither, the prover picks no window, and coins a session of another window
released after the registration's digest are refused. The family is the verifier's too (`label`): a segment of another
family is refused, so a prover that reserved several families picks none of them after reading. `Run.run` refuses a
sample from the challenger whose evidence is missing or gives a reason (`select`, code `coins`, or `family` where the
reason is `FAMILY_ROOT`), its `after` the registration's digest.

`seedSaltOf key window coins` reads a window's seed salt the same way: the one coin of the challenger's
`SEED_SALT_FAMILY`, a family no sample reads, that the session released after the window's open root itself, which
R6 holds the window's registrations to name (`Registration.Verifier.seedSalt`).

The difftest's pieces (`Pieces.table`): `challenger_draw_coins` (`coins_reason`), `{"reason"}`, or `{"refused": why}`
for a key `ChallengerKey.ofJson` refuses, and `challenger_seed_salt` (`seed_salt_of`).
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
  let (c, roots) ← trustedChallengePhaseOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
    (getD coins "pins") (getD coins "main") (getD coins "session")
  unless withheldOn (getD coins "ledger") window do throw WITHHELD
  let segment := getD coins "segment"
  unless jsonStrIs (getD segment "label") label do
    throw s!"the draw's coins are not the challenger's {label} family's"
  challengerStream c segment after (some (((roots.find? (·.1 == label)).map (·.2)).getD []))

def COINS_MAIN : String :=
  "the copy of main's first-parent history a draw's coin evidence carries is not the verifier's own, commit for " ++
  "commit, or the verifier holds none: the evidence's `main` is the verifier's `git rev-list --first-parent origin/main`"

/-- A commit's name where `c` is a string. -/
def commitName? : Json → Option String
  | .str s => some s
  | _ => none

/-- A copy of main's first-parent history as its commits' names; `none` for anything but a list of strings. -/
def mainLines? : Json → Option (List String)
  | .arr cs => cs.toList.mapM commitName?
  | _ => none

/-- Whether the copy `theirs` of main's first-parent history is the verifier's own `mine`: each a list of strings, the
same ones in the same order. Compared through `mainLines?`, not by `BEq Json`, a `partial def` proofs can't see
through. -/
def sameMain (mine theirs : Json) : Bool :=
  match mainLines? mine, mainLines? theirs with
  | some a, some b => a == b
  | _, _ => false

/-- `coins_reason` (module docstring): why `coins`, the evidence of a sample's coins, does not show `read` to be the
start of the coins `drawCoins` reads from it, every one of them read: the coins of the verifier's family `label` that
the session the auditor's ledger opens for window `window` (its open root) under the verifier's challenger key `key`
released after `after`, the copy of main's first-parent history the evidence carries being the verifier's own `main`;
`none` when it does. -/
def coinsReason (key : ChallengerKey) (main coins : Json) (window label : String) (after read : ByteArray) :
    Option String :=
  match drawCoins key window label after coins with
  | .error why => some why
  | .ok stream => if !sameMain main (getD coins "main") then some COINS_MAIN else startReason stream read

/-- `{"after", "coins", "key", "label", "main", "read", "window"}` to `coins_reason`'s, `key` the verifier's challenger
key (`ChallengerKey.from_json`'s form), `main` its copy of main's first-parent history, `window` the window's open root
and `label` the verifier's family: `{"reason"}`, or `{"refused"}` for a key `ChallengerKey.ofJson` refuses. -/
def drawCoinsPiece (j : Json) : V Json := do
  unless keys? j == some #["after", "coins", "key", "label", "main", "read", "window"] do
    throw "a request's keys are after, coins, key, label, main, read, window"
  let .str after := getD j "after" | throw "after is not lowercase hex"
  let .str read := getD j "read" | throw "read is not lowercase hex"
  let after ← Flock.unhex after
  let read ← Flock.unhex read
  let .str window := getD j "window" | throw "window is not a string"
  let .str label := getD j "label" | throw "label is not a string"
  return match ChallengerKey.ofJson (getD j "key") with
    | .error why => Json.mkObj [("refused", Json.str why)]
    | .ok key => Json.mkObj [("reason",
        (coinsReason key (getD j "main") (getD j "coins") window label after read).elim Json.null Json.str)]

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
