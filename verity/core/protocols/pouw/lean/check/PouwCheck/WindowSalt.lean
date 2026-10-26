import PouwCheck.Pass

/-!
# A live window's salt (`core_check.window_salt`)

A live window's (`coins` live) service answer carries its salt, `{coins, epoch, opening, salt, session}`, and its
verifier holds its salt session's inputs (`Params.live`, `{key, ledger, main, pins, session}`). The salt's epoch root R
is `epochOpenRoot` of the registered weights' root W and the answer's `epoch` k, computed here; the session is the one
the auditor's ledger opened on `{root: R}`, and it counts as a live Pearl-C pass's does (`saltEpoch`, the steps of
`liveSaltCheck`); then the answer's coins are the commitment it signed, its opening that commitment's `pouw/salt` coin 0
opened after R, its salt that coin's round-0 unit salt after R (`h1tile.live_salt`: `PouwWindow.roundKey`'s stream 0,
`SALT_BYTES`), and its session the gate's answer for the session without its secret (`carries`). A seeded window's
answer carries no salt. Every refusal is `core_check.py`'s.
-/

namespace PouwCheck

open Lean (Json)
open SampledProofs.Challenger (Audited LedgerEntry LiveCommitment LiveOpening)

def WINDOW_SALT_KEYS : List String := ["coins", "epoch", "opening", "salt", "session"]
def SEEDED_SALT : String := "the window's coins are seeded, and its service answer carries a live salt"
def NO_WINDOW_SALT : String :=
  "the window's coins are live, and its service answer carries no salt: its salt coin is a salt-only session's, " ++
    "opened on its salt epoch's root"
def WINDOW_SALT_FORM : String := "the answer's salt is not {coins, epoch, opening, salt, session}"
def WINDOW_NO_W : String :=
  "the window registers no weights' root W, a SHA-512 digest at its W port, which its salt epoch's root binds"
def WINDOW_EPOCH : String :=
  "the answer's salt names no epoch: `epoch` is an integer in [0, 2^64), its number among W's salt epochs"
def WINDOW_NOT_SIGNED : String := "the answer's salt's coin commitment is not the one its session's challenger signed"
def WINDOW_SALT : String := "the answer's salt is not its live coin's round-0 salt after its epoch's root"
def WINDOW_SESSION : String :=
  "the answer's salt's session is not the gate's answer, without its secret, for the session the auditor's ledger " ++
    "opened on its epoch's root: its session, seq, root and commitment"

/-- `core_check._salt_epoch`: the salt-only session opened on salt epoch root `r`, when it counts (LIVESALT2): a
Pearl-C salt session's commitment, reserved for its salt coin alone and opened binding that reserve, ended complete
having released that coin and nothing else, its log showing `r` alone as the root the coin followed. The ledger's
session, its signed commitment and its open entry. -/
def saltEpoch (key ledger pins main session : Json) (r : ByteArray) : R (Audited × LiveCommitment × LedgerEntry) := do
  let (a, c, received) ← saltSession key ledger pins main (Json.mkObj [("root", .str (Flock.hexOf r))]) session
  if let some why := PouwWindow.saltEpochReason c then throw ("salt", why)
  let named := sessionEntries ledger a.session
  let (some reserve, some opened, some ended) :=
      (named.find? (·.kind == "reserve"), named.find? (·.kind == "open"), named.find? (·.kind == "end"))
    | throw ("salt", NO_RESERVE)
  unless SampledProofs.jeq (SampledProofs.getD reserve.body "families")
      (SampledProofs.Challenger.familiesJson PouwWindow.SALT_FAMILIES) do throw ("salt", RESERVE_LAYOUT)
  unless SampledProofs.int? (SampledProofs.getD opened.body "reserve") == some (reserve.seq : Int) do
    throw ("salt", NO_RESERVE)
  unless jsonTrue (SampledProofs.getD ended.body "complete") &&
      SampledProofs.jeq (SampledProofs.getD ended.body "released")
        (Json.mkObj [(PouwWindow.SALT, Lean.toJson (1 : Nat))]) do throw ("salt", INCOMPLETE)
  unless received == [(PouwWindow.SALT, [r])] do throw ("salt", LIVE_NOT_RECEIVED)
  pure (a, c, opened)

/-- The steps of `core_check.window_salt` on a live window's answer's salt `s`, under the verifier's `live` inputs and the
registered weights' root `rootW`. -/
def windowSaltCheck (live : Json) (rootW : Option String) (s : Json) : R Unit := do
  unless keys s == some WINDOW_SALT_KEYS do throw ("salt", WINDOW_SALT_FORM)
  let some w := (rootW.filter isHexDigest).bind fun h => (PouwCommit.pyFromHex h).toOption
    | throw ("salt", WINDOW_NO_W)
  let some k := (PouwWindow.natOf? (at? s "epoch")).filter (· < 2 ^ 64) | throw ("salt", WINDOW_EPOCH)
  let r := epochOpenRoot w k
  let g := SampledProofs.getD live
  let (a, c, opened) ← saltEpoch (g "key") (g "ledger") (g "pins") (g "main") (g "session") r
  let claimed ← (LiveCommitment.ofJson (at? s "coins")).mapError fun e => ("coin", s!"the salt's coins are malformed: {e}")
  unless sameCommitment claimed c do coin WINDOW_NOT_SIGNED
  let o ← (LiveOpening.ofJson (at? s "opening")).mapError fun e => ("coin", s!"the salt's {e}")
  if let some why ← port (SampledProofs.Challenger.liveReason c o PouwWindow.SALT 0 r) then coin s!"the salt's {why}"
  let key ← port (PouwWindow.roundKey c o.coin 0 r)
  unless jsonStrIs (at? s "salt") (Flock.hexOf (FlockSoundness.Randomness.stream key [.int 0] SALT_BYTES)) do
    coin WINDOW_SALT
  unless carries (at? s "session") a.session opened.seq r a.commitment do coin WINDOW_SESSION

/-- `core_check.window_salt`: why the salt an accepted window's answer carries (`none` when it carries none) is not its
live salt, or `none`. -/
def windowSaltReason (p : Params) (rootW : Option String) (salt : Option Json) : Option String :=
  if p.check.coins != some "live" then
    if salt.isSome then some SEEDED_SALT else none
  else match salt with
    | none => some NO_WINDOW_SALT
    | some s => match windowSaltCheck (p.live.getD .null) rootW s with
      | .ok () => none
      | .error (_, why) => some why

end PouwCheck
