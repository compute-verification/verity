/-
Every coin a v2 window reads, from the auditor's challenger (`coins.py`'s `round_coins`, `exactly_reason` and
`use_roots`, which `tests/test_challenger_difftest.py` holds to this, the verifier of record).

A use is one read of a family's coins: its label, what they were released after, and how many coins. v2 reads the
window's draw (after the registration's digest), and each session's link points and rounds (after root_B's digest
and each round's `Round.sha`), as uses of the one withheld session the auditor's ledger opens on the window.
`roundCoins key window uses coins` reads each use's coins with `drawCoins` on its own segment, and refuses unless the
session's signed release log shows every family of its layout released after exactly the uses' roots, coin for coin,
in order (`exactlyReason`, `Use.roots`). So the log holds one release a use: a second release for one round (a
prover's second query), a release on a family no use reads (a slot the evidence doesn't hold, an abandoned try) and a
draw after another root are each refused, not charged.

The difftest's piece (`Pieces.table`): `challenger_rounds` (`round_coins`), `{"coins": [hex, ...]}`, or
`{"refused": why}` where Python raises.
-/
import SampledProofs.ChallengerDraw

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

/-- One read of a family's coins: its label, what they were released after (64 bytes), and how many coins. -/
structure Use where
  label : String
  after : ByteArray
  coins : Nat
  deriving Inhabited

/-- The roots family `label`'s coins must be released after, in release order: each use of it in turn, once a coin. -/
def Use.roots (us : List Use) (label : String) : List ByteArray :=
  (us.filter (·.label == label)).flatMap fun u => List.replicate u.coins u.after

/-- `exactly_reason` (module docstring): why some family of `c`'s layout shows its coins released after other roots
than exactly the uses', coin for coin, in order; `none` when none does. -/
def exactlyReason (c : LiveCommitment) (roots : Roots) (us : List Use) : Option String :=
  c.families.findSome? fun (l, _) =>
    let got := ((roots.find? (·.1 == l)).map (·.2)).getD []
    let want := Use.roots us l
    if got == want then none
    else some (s!"the challenger's log shows family {l}'s coins released after {got.length} roots, not exactly " ++
      s!"the {want.length} the evidence's uses read, once each in order: a release no use reads, or a use " ++
      "released twice")

def ROUNDS_FORM : String :=
  "a session's round coins are {ledger, main, pins, segments, session}: the auditor's ledger, the verifier's " ++
  "copy of main's first-parent history, the host's pins log, one segment a use (Segment.to_json) and the " ++
  "challenger session's files"

/-- `round_coins` (module docstring): each use's coins, `drawCoins` under `key`, `window`, the use's family and
`after` on its segment (`coins`' `segments`, one a use, in order), `coins` coins of `COIN_BYTES` each; and the release
log of the session the ledger opens on `window` exactly the uses (`exactlyReason`). -/
def roundCoins (key : ChallengerKey) (window : String) (us : List Use) (coins : Json) : V (List ByteArray) := do
  unless keys? coins == some #["ledger", "main", "pins", "segments", "session"] do throw ROUNDS_FORM
  let .arr segments := getD coins "segments" | throw ROUNDS_FORM
  unless segments.size == us.length do throw s!"{segments.size} segments for {us.length} uses: one segment a use"
  let out ← (us.zip segments.toList).mapM fun (u, s) => do
    let at_ := Json.mkObj [("ledger", getD coins "ledger"), ("main", getD coins "main"), ("pins", getD coins "pins"),
      ("segment", s), ("session", getD coins "session")]
    let ξ ← drawCoins key window u.label u.after at_
    unless ξ.size == u.coins * COIN_BYTES do
      throw (s!"the {u.label} coins after {(Flock.hexOf u.after).take 16}… are {ξ.size / COIN_BYTES} coins, not the " ++
        s!"use's {u.coins}")
    pure ξ
  let (c, roots) ← trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key
    (getD coins "pins") (getD coins "main") (getD coins "session")
  unless withheldOn (getD coins "ledger") window do throw WITHHELD
  if let some why := exactlyReason c roots us then throw why
  return out

/-- `{"coins", "key", "uses", "window"}` to `round_coins`'s: `uses` a list of `{after, coins, label}`. -/
def roundsPiece (j : Json) : V Json := do
  unless keys? j == some #["coins", "key", "uses", "window"] do throw "a request's keys are coins, key, uses, window"
  let .str window := getD j "window" | throw "window is not a string"
  let .arr us := getD j "uses" | throw "uses is not a list"
  let uses ← us.toList.mapM fun u => do
    unless keys? u == some #["after", "coins", "label"] do throw "a use is {after, coins, label}"
    let (.str label, .str after, some n) := (getD u "label", getD u "after", nat? (getD u "coins"))
      | throw "a use is {after, coins, label}"
    pure ({ label, after := ← Flock.unhex after, coins := n } : Use)
  return match (do roundCoins (← ChallengerKey.ofJson (getD j "key")) window uses (getD j "coins")) with
    | .ok xs => Json.mkObj [("coins", Json.arr (xs.toArray.map fun x => Json.str (Flock.hexOf x)))]
    | .error why => Json.mkObj [("refused", Json.str why)]

end SampledProofs.Challenger
