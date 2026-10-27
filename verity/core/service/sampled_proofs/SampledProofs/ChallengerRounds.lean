/-
Every coin a v2 window reads, from the auditor's challenger (`coins.py`'s `round_coins`, `exactly_reason`,
`use_roots`, `received_order`, `release_order_reason`, `runs_reason` and `runs_why`, which `tests/test_challenger_difftest.py` holds to this, the verifier
of record).

A use is one read of a family's coins: its label, what they were released after, and how many coins. v2 reads the
window's draw (after the registration's digest), and each session's link points and rounds (after root_B's digest
and each round's request at its stream's chain over `Round.sha`), as uses of the one withheld session the auditor's ledger opens on the window.
`roundCoins key window uses order groups coins` reads each use's coins with `drawCoins` on its own segment, and refuses unless the
session's signed release log shows every family of its layout released after exactly the uses' roots, coin for coin,
in order (`exactlyReason`, `Use.roots`). So the log holds one release a use: a second release for one round (a
prover's second query), a release on a family no use reads (a slot the evidence doesn't hold, an abandoned try) and a
draw after another root are each refused, not charged. The log's order across families is the play's: for each pair
`(a, b)` of `order`, every root of family `a` is received before any of family `b` (`releaseOrderReason`), which v2 asks of
the draw before every session's families and of each session's link points before its rounds. Within each group of
`groups` (v2's: one session's families), each use's coins are one run of the log's releases of the group's families
(`runsWhy`): no other release of the group falls between two coins of one use, so a session's rounds are released one
after another, each whole, while releases of other groups may fall anywhere between them.

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

/-- `received_order` (module docstring): the family of each root the release log `log` shows the challenger received,
in the log's order (a log `releasedRootsOf` has read: its head, then a root and its release in turn). -/
def receivedOrder (log : List String) : List String :=
  log.tail.zipIdx.filterMap labelAt
where
  /-- The family of the root line `line` at an even place `i` after the log's head. -/
  labelAt : String × Nat → Option String
    | (line, i) =>
      if i % 2 != 0 then none else
      match Flock.JsonParse.parse line with
      | .ok r => match getD r "label" with | .str l => some l | _ => none
      | .error _ => none

/-- `release_order_reason` (module docstring): why the log's order `labels` receives a root of some family `b` before the last
of family `a`'s, for a pair `(a, b)` of `order`; `none` when every pair's releases are in order. -/
def releaseOrderReason (labels : List String) (order : List (String × String)) : Option String :=
  order.findSome? fun (a, b) =>
    match (labels.zipIdx.filter (·.1 == a)).getLast?, labels.findIdx? (· == b) with
    | some (_, i), some j =>
      if j < i then some s!"the challenger's log releases family {b}'s coins before the last of family {a}'s, not after"
      else none
    | _, _ => none

def RUNS : String :=
  "the challenger's log releases another coin of a use's group inside one use's coins, or other coins than the " ++
    "uses': each use's coins must be one run among its group's releases"

/-- `runs_reason` (module docstring): why the labels `ls` don't release the uses `pending` one run each, walking `ls`:
its first label starts that family's first use of `pending`, whose `coins` releases are the next ones, all of that
family, and the walk goes on after them without that use; `none` when the labels and the uses run out together. Each
step takes one use, so `fuel` of one more than the uses is enough. -/
def runsReason : (fuel : Nat) → (pending : List Use) → (ls : List String) → Option String
  | _, [], [] => none
  | _, _ :: _, [] => some RUNS
  | 0, _, _ :: _ => some RUNS
  | f + 1, pending, l :: ls =>
    match pending.findIdx? (·.label == l) with
    | none => some RUNS
    | some i =>
      let n := (pending.getD i default).coins
      if n ≤ ls.length + 1 && ((l :: ls).take n).all (· == l) then runsReason f (pending.eraseIdx i) ((l :: ls).drop n)
      else some RUNS

/-- `runs_why` (module docstring): for each group `g` of `groups` (a list of families), why the log's labels `labels`
restricted to `g`'s families don't release `g`'s uses of `us` (those with a coin) one run each (`runsReason`); `none`
when every group's do. -/
def runsWhy (us : List Use) (groups : List (List String)) (labels : List String) : Option String :=
  groups.findSome? fun g =>
    let pending := us.filter fun u => g.contains u.label && u.coins != 0
    runsReason (pending.length + 1) pending (labels.filter g.contains)

def ROUNDS_FORM : String :=
  "a session's round coins are {ledger, main, pins, segments, session}: the auditor's ledger, the verifier's " ++
  "copy of main's first-parent history, the host's pins log, one segment a use (Segment.to_json) and the " ++
  "challenger session's files"

/-- `round_coins` (module docstring): each use's coins, `drawCoins` under `key`, `window`, the use's family and
`after` on its segment (`coins`' `segments`, one a use, in order), `coins` coins of `COIN_BYTES` each; the release
log of the session the ledger opens on `window` exactly the uses (`exactlyReason`); and, for each pair `(a, b)` of
`order`, every release of family `a` before every release of family `b` (`releaseOrderReason`); and, for each group
of `groups`, each of its uses' coins one run among the group's releases (`runsWhy`). -/
def roundCoins (key : ChallengerKey) (window : String) (us : List Use) (order : List (String × String))
    (groups : List (List String)) (coins : Json) : V (List ByteArray) := do
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
  let .arr ls := getD (getD coins "session") "log" | throw LOG_FORM
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM
  if let some why := releaseOrderReason (receivedOrder lines) order then throw why
  if let some why := runsWhy us groups (receivedOrder lines) then throw why
  return out

/-- `{"coins", "groups", "key", "order", "uses", "window"}` to `round_coins`'s: `uses` a list of `{after, coins,
label}`, `order` a list of `[a, b]` family pairs, `groups` a list of lists of family labels. -/
def roundsPiece (j : Json) : V Json := do
  unless keys? j == some #["coins", "groups", "key", "order", "uses", "window"] do
    throw "a request's keys are coins, groups, key, order, uses, window"
  let .str window := getD j "window" | throw "window is not a string"
  let .arr us := getD j "uses" | throw "uses is not a list"
  let uses ← us.toList.mapM fun u => do
    unless keys? u == some #["after", "coins", "label"] do throw "a use is {after, coins, label}"
    let (.str label, .str after, some n) := (getD u "label", getD u "after", nat? (getD u "coins"))
      | throw "a use is {after, coins, label}"
    pure ({ label, after := ← Flock.unhex after, coins := n } : Use)
  let .arr os := getD j "order" | throw "order is not a list"
  let order ← os.toList.mapM fun
    | .arr #[.str a, .str b] => pure (a, b)
    | _ => throw "an order entry is [a, b], two family labels"
  let .arr gs := getD j "groups" | throw "groups is not a list"
  let groups ← gs.toList.mapM fun
    | .arr ls => ls.toList.mapM (fun | .str l => pure l | _ => throw "a group is a list of family labels")
    | _ => throw "a group is a list of family labels"
  return match (do roundCoins (← ChallengerKey.ofJson (getD j "key")) window uses order groups (getD j "coins")) with
    | .ok xs => Json.mkObj [("coins", Json.arr (xs.toArray.map fun x => Json.str (Flock.hexOf x)))]
    | .error why => Json.mkObj [("refused", Json.str why)]

#guard receivedOrder ["head", "{\"label\":\"a\"}", "r", "{\"label\":\"b\"}", "r"] == ["a", "b"]
#guard releaseOrderReason ["d", "p", "t", "t"] [("d", "p"), ("p", "t")] == none
#guard (releaseOrderReason ["d", "t", "p", "t"] [("d", "p"), ("p", "t")]).isSome
#guard releaseOrderReason ["p", "t"] [("d", "p")] == none
#guard runsWhy [⟨"a", .empty, 2⟩, ⟨"b", .empty, 1⟩] [["a", "b"]] ["a", "a", "b"] == none
#guard runsWhy [⟨"a", .empty, 2⟩, ⟨"b", .empty, 1⟩] [["a", "b"]] ["b", "a", "a"] == none
#guard (runsWhy [⟨"a", .empty, 2⟩, ⟨"b", .empty, 1⟩] [["a", "b"]] ["a", "b", "a"]).isSome
#guard runsWhy [⟨"a", .empty, 2⟩, ⟨"b", .empty, 1⟩] [["a"], ["b"]] ["a", "b", "a"] == none
#guard (runsWhy [⟨"a", .empty, 2⟩] [["a"]] ["a"]).isSome
#guard (runsWhy [⟨"a", .empty, 1⟩] [["a"]] ["a", "a"]).isSome

end SampledProofs.Challenger
