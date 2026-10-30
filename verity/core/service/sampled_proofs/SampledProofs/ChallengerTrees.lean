/-
A `--zk` session's coins as the challenger's release log shows it drew and opened them (`challenger.py`'s "A --zk
session's coins", which `tests/test_challenger_difftest.py` holds to this, the verifier of record).

The challenger draws each session's keyed coin tree (`coin-tree/hm96-sha512/v2`) in the window's session and opens each
round itself, after what the round follows, logging a line for each before what it records leaves its host, so the
session's signed close covers them:

    tree    = {event: coin-tree, key, nonce, root, spec}
    round   = {after, coins, event: tree-open, n, round, stream, tree}     tree: its root; coins: the round's coins' SHA-512

`coinTreesOf` reads those lines (`coin_trees_of`): each tree's key, nonce and root in lowercase hex and a spec the
challenger draws (`isTreeSpec`), no root or key twice; each round of a tree drawn on an earlier line, of one of its
streams, that stream's next round, below `2^rounds_log`, at most `block` coins. `sessionTreeReason` holds a session's
record to them (`session_tree_reason`): its `coin_tree` a logged tree, its link points that tree's `session` round 0,
opened after its link's `root_F` and each table's `root_B` in table-name order (`linkAfter`), and each round `k` of each
of its streams that tree's round `k` of the stream, opened after its `msg_sha512`, its `n` coins hashing to the logged
digest (each coin's 16 bytes as the record's 32 hex digits write them). `sessionsTreeReason` holds an audit's sessions
to them (`sessions_tree_reason`): each session's record so, no tree two sessions', and every tree a session's, so no
statement's coins are drawn twice. `trustedTreesOf` reads the trees from the one session the auditor's ledger opens on
the verifier's window (`trustedSessionOf`), the session a draw's coins come from.

The difftest's pieces: `challenger_coin_trees` (`coin_trees_of`) and `challenger_session_trees`
(`sessions_tree_reason` over the log's trees), each `{"refused": why}` where Python refuses.
-/
import SampledProofs.ChallengerDraw

namespace SampledProofs.Challenger

open Lean (Json)
open Flock (V)

def TREE_SCHEME : String := "coin-tree/hm96-sha512/v2"
/-- A coin tree's spec as the challenger draws one: at most `2^MAX_ROUNDS_LOG` rounds a stream, `MAX_BLOCK` coins a round. -/
def MAX_ROUNDS_LOG : Nat := 16
def MAX_BLOCK : Nat := 2 ^ 16
def TREE_SESSION : String := "session"
def NO_TREE : String := "its coin tree is none the challenger's release log shows it drew"
def SHARED_TREE : String := "its coin tree is another session's"
def UNUSED_TREE : String := "a tree no session records is a redraw, and each session draws its coins once"
def TREE_RECORD_FORM : String :=
  "its record names no coin_tree, link {points, root_b, root_f} and streams [{stream, rounds: [{coins, msg_sha512, " ++
  "n}]}] in lowercase hex, as a --zk session's record does"

/-- A round of a coin tree the release log shows opened: stream `stream`'s round `round`, its first `n` coins, after
`after`, whose SHA-512 is `coins`, each lowercase hex. -/
structure TreeOpening where
  stream : String
  round : Nat
  n : Nat
  after : String
  coins : String

def TreeOpening.toJson (o : TreeOpening) : Json :=
  Json.mkObj [("after", Json.str o.after), ("coins", Json.str o.coins), ("n", Lean.toJson o.n),
    ("round", Lean.toJson o.round), ("stream", Json.str o.stream)]

/-- A coin tree the release log shows drawn, its key, nonce and root in lowercase hex and its spec, and the rounds it
shows opened of it, in order. -/
structure LoggedTree where
  key : String
  nonce : String
  root : String
  spec : Json
  opened : Array TreeOpening := #[]

/-- The `coin_tree` of a session record on this tree. -/
def LoggedTree.record (t : LoggedTree) : Json :=
  Json.mkObj [("key", Json.str t.key), ("nonce", Json.str t.nonce), ("root", Json.str t.root), ("spec", t.spec)]

def LoggedTree.toJson (t : LoggedTree) : Json :=
  Json.mkObj [("key", Json.str t.key), ("nonce", Json.str t.nonce), ("opened", Json.arr (t.opened.map (·.toJson))),
    ("root", Json.str t.root), ("spec", t.spec)]

def LoggedTree.opening (t : LoggedTree) (stream : String) (k : Nat) : Option TreeOpening :=
  t.opened.find? fun o => o.stream == stream && o.round == k

private def isHexN (j : Json) (n : Nat) : Bool :=
  match j with
  | .str s => s.length == n && (Flock.unhex s).toOption.isSome
  | _ => false

private def strOf (j : Json) : String := match j with | .str s => s | _ => ""

/-- `_stream`: a stream's name, 1 to `MAX_SESSION_ID` ASCII letters, digits, '-', '.', '_' and '/'. -/
private def isStream (j : Json) : Bool :=
  match j with
  | .str s => 0 < s.length && s.length ≤ MAX_SESSION_ID &&
      s.all fun c => c.isAlphanum && c.toNat < 128 || c == '-' || c == '.' || c == '_' || c == '/'
  | _ => false

private def distinct : List Json → Bool
  | [] => true
  | x :: rest => !rest.contains x && distinct rest

/-- `tree_spec_reason`'s domain: `{block, rounds_log, scheme, streams}`, `TREE_SCHEME`, 1 to `MAX_BLOCK` coins a round,
at most `2^MAX_ROUNDS_LOG` rounds a stream, and its streams distinct names (`isStream`). -/
def isTreeSpec (spec : Json) : Bool :=
  keys? spec == some #["block", "rounds_log", "scheme", "streams"] && getD spec "scheme" == Json.str TREE_SCHEME &&
    (match nat? (getD spec "block") with | some b => 1 ≤ b && b ≤ MAX_BLOCK | none => false) &&
    (match nat? (getD spec "rounds_log") with | some r => r ≤ MAX_ROUNDS_LOG | none => false) &&
    (match getD spec "streams" with
      | .arr ss => !ss.isEmpty && ss.all isStream && distinct ss.toList
      | _ => false)

/-- `coin_trees_of` (module docstring): the coin trees the release log `log` shows the challenger drew, in the order
drawn, each with the rounds it shows opened; every other line is a root's or a release's (`releasedRootsOf`). -/
def coinTreesOf (log : List String) : V (Array LoggedTree) := do
  let mut trees : Array LoggedTree := #[]
  for line in log do
    unless treeLine line do continue
    let d ← event line
    if getD d "event" == Json.str TREE_EVENT then
      unless keys? d == some #["event", "key", "nonce", "root", "spec"] && isHexN (getD d "key") 512 &&
          isHexN (getD d "nonce") 64 && isHexN (getD d "root") 128 && isTreeSpec (getD d "spec") do
        throw LOG_MALFORMED
      let key := strOf (getD d "key")
      let root := strOf (getD d "root")
      if trees.any (fun t => t.root == root || t.key == key) then throw LOG_MALFORMED
      trees := trees.push { key, nonce := strOf (getD d "nonce"), root, spec := getD d "spec" }
    else
      unless keys? d == some #["after", "coins", "event", "n", "round", "stream", "tree"] do throw LOG_MALFORMED
      let .str root := getD d "tree" | throw LOG_MALFORMED
      let some i := trees.findIdx? (·.root == root) | throw LOG_MALFORMED
      let t := trees[i]!
      let .str s := getD d "stream" | throw LOG_MALFORMED
      let (some k, some n) := (nat? (getD d "round"), nat? (getD d "n")) | throw LOG_MALFORMED
      let streams := match getD t.spec "streams" with | .arr ss => ss | _ => #[]
      unless streams.contains (Json.str s) && k == (t.opened.filter (·.stream == s)).size &&
          k >>> (nat? (getD t.spec "rounds_log")).getD 0 == 0 && n ≤ (nat? (getD t.spec "block")).getD 0 &&
          isHexN (getD d "after") 128 && isHexN (getD d "coins") 128 do
        throw LOG_MALFORMED
      trees := trees.modify i fun t =>
        { t with opened := t.opened.push ⟨s, k, n, strOf (getD d "after"), strOf (getD d "coins")⟩ }
  return trees

/-- `link_after`: what a session's link points are opened after, the SHA-512 of its `root_F` and each table's `root_B`
in table-name order. -/
def linkAfter (rootF : ByteArray) (rootB : Array (String × ByteArray)) : ByteArray :=
  Flock.Sha512.hash ((rootB.qsort (·.1 < ·.1)).foldl (fun acc (_, b) => acc ++ b) rootF)

/-- `_record_coins`: a record's coins, each 32 lowercase hex digits, as their bytes in order. -/
private def recordCoins? (cs : Json) : Option ByteArray :=
  match cs with
  | .arr xs => xs.foldlM (init := ByteArray.empty) fun acc c => match c with
    | .str s => if s.length == 32 then (Flock.unhex s).toOption.map (acc ++ ·) else none
    | _ => none
  | _ => none

/-- `session_tree_reason` (module docstring): why `record`, a `--zk` session's record, doesn't show its coins to be ones
the release log shows the challenger released (`trees`, `coinTreesOf`'s); `none` when it does. -/
def sessionTreeReason (trees : Array LoggedTree) (record : Json) : Option String := Id.run do
  let ct := getD record "coin_tree"
  let some t := (match getD ct "root" with | .str r => trees.find? (·.root == r) | _ => none) | return some NO_TREE
  unless jeq ct t.record do
    return some "its coin tree is not the one the challenger's release log shows it drew under that root"
  let link := getD record "link"
  let (.obj _, .obj _, .arr ps) := (link, getD link "root_b", getD link "points") | return some TREE_RECORD_FORM
  let flat := ps.foldl (fun acc p => match p with | .arr cs => acc ++ cs | _ => acc.push Json.null) #[]
  let rootB := (entries (getD link "root_b")).filterMap fun (k, v) => match v with
    | .str s => (Flock.unhex s).toOption.map (k, ·)
    | _ => none
  let (some rootF, some points, .arr streams) := ((match getD link "root_f" with
      | .str s => (Flock.unhex s).toOption
      | _ => none), recordCoins? (Json.arr flat), getD record "streams") | return some TREE_RECORD_FORM
  unless rootB.size == (entries (getD link "root_b")).size do return some TREE_RECORD_FORM
  let linkOk := match t.opening TREE_SESSION 0 with
    | some o => o.n == points.size / 16 && o.after == Flock.hexOf (linkAfter rootF rootB) && o.coins == sha512Hex points
    | none => false
  unless linkOk do
    return some "its link points are not the coins the challenger opened as its tree's session round 0 after its link's roots"
  for s in streams do
    let (.str name, .arr rounds) := (getD s "stream", getD s "rounds") | return some TREE_RECORD_FORM
    for (r, k) in rounds.zipIdx do
      let (some coins, some n) := (recordCoins? (getD r "coins"), nat? (getD r "n")) | return some TREE_RECORD_FORM
      unless isHexN (getD r "msg_sha512") 128 do return some TREE_RECORD_FORM
      let ok := match t.opening name k with
        | some o => o.n == n && Json.str o.after == getD r "msg_sha512" && o.coins == sha512Hex coins
        | none => false
      unless ok do
        return some s!"its {name} round {k} is not coins the challenger opened under its tree after that round's message"
  return none

/-- `sessions_tree_reason` (module docstring): why the `--zk` sessions `records`, each its name and its record, don't
show their coins to be the ones the release log shows drawn and opened (`trees`), one tree a session; `none` when they
do. In order, each session's coins its tree's (`sessionTreeReason`), a tree no earlier session's (`SHARED_TREE`); then
every tree a session's (`UNUSED_TREE`). -/
def sessionsTreeReason (trees : Array LoggedTree) (records : Array (String × Json)) : Option String := Id.run do
  let mut used : Array String := #[]
  for (name, record) in records do
    let root := strOf (getD (getD record "coin_tree") "root")
    let why := sessionTreeReason trees record <|> if used.contains root then some SHARED_TREE else none
    if let some w := why then
      return some s!"session {name}'s coins are not the challenger's: {w}"
    used := used.push root
  unless used.size == trees.size do
    return some s!"the challenger's release log shows {trees.size} coin trees and the sessions use {used.size}: {UNUSED_TREE}"
  return none

/-- The coin trees the one session the auditor's ledger opens on `window` under `key` shows the challenger drew and
opened (`coinTreesOf`), from a draw's coin evidence `{ledger, main, pins, segment, session}` (`drawCoins`'s, read
through `trustedSessionOf`). -/
def trustedTreesOf (key : ChallengerKey) (window : String) (coins : Json) : V (Array LoggedTree) := do
  unless keys? coins == some #["ledger", "main", "pins", "segment", "session"] do throw DRAW_FORM
  let _ ← trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key (getD coins "pins")
    (getD coins "main") (getD coins "session")
  let .arr ls := getD (getD coins "session") "log" | throw LOG_FORM
  coinTreesOf (← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM)

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def logOf (j : Json) : V (List String) := do
  let .arr ls := getD j "log" | throw "log is not a list of strings"
  ls.toList.mapM fun | .str s => pure s | _ => throw "log is not a list of strings"

/-- `{"log"}` to `coin_trees_of`'s, `{"trees": [tree, ...]}` in the order drawn, each with its `opened` rounds. -/
def coinTreesPiece (j : Json) : V Json := do
  unless keys? j == some #["log"] do throw "a request's keys are log"
  let lines ← logOf j
  pure <| refusing do
    return Json.mkObj [("trees", Json.arr ((← coinTreesOf lines).map (·.toJson)))]

/-- `{"log", "records": [{name, record}, ...]}` to `sessions_tree_reason`'s over `coin_trees_of`'s trees of `log`,
`{"reason"}`. -/
def sessionTreesPiece (j : Json) : V Json := do
  unless keys? j == some #["log", "records"] do throw "a request's keys are log, records"
  let lines ← logOf j
  let .arr rs := getD j "records" | throw "records is not a list of {name, record}"
  let records ← rs.mapM fun r => do
    unless keys? r == some #["name", "record"] do throw "records is not a list of {name, record}"
    let .str name := getD r "name" | throw "records is not a list of {name, record}"
    pure (name, getD r "record")
  pure <| refusing do
    let why := sessionsTreeReason (← coinTreesOf lines) records
    return Json.mkObj [("reason", why.elim Json.null Json.str)]

end SampledProofs.Challenger
