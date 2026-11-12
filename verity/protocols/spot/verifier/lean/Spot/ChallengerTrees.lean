/-
A `--zk` session's coins as the challenger's release log shows it sampled and opened them (`challenger.py`'s "A --zk
session's coins", which `tests/test_challenger_difftest.py` holds to this, the verifier).

The challenger samples each session's keyed coin tree (`coin-tree/hm96-sha512/v2`) in the window's session and opens
each round itself, after what the round follows, logging a line for each before what it records leaves its host, so the
session's signed close covers them:

    tree    = {event: coin-tree, key, nonce, root, spec}
    round   = {after, coins, event: tree-open, n, round, stream, tree}     tree: its root; coins: the round's coins' SHA-512

`coinTreesOf` reads those lines (`coin_trees_of`): each tree's key, nonce and root in lowercase hex and a spec the
challenger samples (`isTreeSpec`), no root or key twice; each round of a tree sampled on an earlier line, of one of its
streams, that stream's next round, below `2^rounds_log`, at most `block` coins. `sessionTreeReason` holds a session's
record on its tables, each with the byte length its statement fixes for its `root_B` where the caller set one up, to
them (`session_tree_reason`): its `coin_tree` a logged tree, its link's `root_b` keys exactly its session's tables
(`LINK_TABLES`), each entry that table's length (`LINK_ROOT_LEN`), its link points that tree's `session` round 0, opened
after its link's `root_F` and each table's `root_B` in table-name order (`linkAfter`), and each round `k` of each of its
streams that tree's round `k` of the stream, opened after its `msg_sha512`, its `n` coins hashing to the logged digest
(each coin's 16 bytes as the record's 32 hex digits write them); and each round the tree shows opened one the record
gives its stream (`recordRounds`, `EXTRA_OPENING`), so the tree opens its `session` stream once, each stream the record
names exactly its `rounds`, and no other stream. `sessionsTreeReason` holds an audit's sessions on one
statement's tables to them (`sessions_tree_reason`): each session's record so, at its own tables' lengths, no tree two
sessions', and every tree a session's, so no statement's coins are flipped twice. `trustedTreesOf` reads the trees from
the one session the auditor's ledger opens on the verifier's window (`trustedSessionOf`), the session the sample's coins
come from. `sampleLinkFirst` holds the log's order (`sample_link_first`): the sample, then the link, then the other
rounds; no coin of the sample released after a round is opened, and the first round opened a tree's `session` stream,
the link points.

A member's session, one of several in a window, ends with a closing round: one more round of its tree's `session`
stream (`CLOSING_ROUND`), of no coins, opened after the SHA-512 of the session's evidence, so the log holds a member's
evidence before the next member's coins. `memberTreeReason` holds a member's session's record to the trees so: the
last round the log shows opened of its tree is that one (`TreeOpening.closes`, `NO_CLOSING`), and the tree without it
is the session's as `sessionTreeReason` holds one, so the closing round is the one opening beyond the record's that
passes. `membersTreeReason` holds the members' sessions to them: each so, one tree a session, every tree a session's,
and the log opening the sessions' trees one after another in the sessions' order, no round of a tree before the
closing round of the tree before it or after its own (`membersOrder`, `MEMBERS_ORDER`).

The difftest's pieces: `challenger_coin_trees` (`coin_trees_of`), `challenger_session_trees` (`sessions_tree_reason`
over the log's trees and the request's records, each with its tables), each `{"refused": why}` where Python refuses,
and `challenger_sample_link_first` (`sample_link_first`).
-/
module

public import Spot.ChallengerDraw
meta import Spot.ChallengerDraw

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Challenger

open Lean (Json)
open Flock (V)

def TREE_SCHEME : String := "coin-tree/hm96-sha512/v2"
/-- A coin tree's spec as the challenger samples one: at most `2^MAX_ROUNDS_LOG` rounds a stream, `MAX_BLOCK` coins a
round. -/
def MAX_ROUNDS_LOG : Nat := 16
def MAX_BLOCK : Nat := 2 ^ 16
def TREE_SESSION : String := "session"
def NO_TREE : String := "its coin tree is none the challenger's release log shows it sampled"
def SHARED_TREE : String := "its coin tree is another session's"
def UNUSED_TREE : String := "a tree no session records is a second sample, and each session's coins are flipped once"
def TREE_RECORD_FORM : String :=
  "its record names no coin_tree, link {points, root_b, root_f} and streams [{stream, rounds: [{coins, msg_sha512, " ++
  "n}]}] in lowercase hex, as a --zk session's record does"
/-- The link points are opened after every `root_b` entry (`linkAfter`) and `Flock.Record.decode` reads only the
session's tables' (S5), so a key of no table, or a table without its key, would let the prover pick a table's `root_B`
after the link's coins. -/
def LINK_TABLES : String := "its link's root_b keys are not exactly its session's tables"
/-- The link points are opened after the SHA-512 of `root_F`, then each table's `root_B` in table-name order
(`linkAfter`), concatenated with no length prefix. A table's `root_B` is its proof's level-0 cap, `2^cap₀` digests of
its Merkle scheme at its statement's `m` (`OneStageZk.rootBLen`), so its statement fixes its length. Refusing any other
length before the hash is what makes the message parse one way: the `root_B`s are its last bytes, each of its table's
length in table order, and `root_F` is the rest. Without it one message could read as other roots. -/
def LINK_ROOT_LEN : String :=
  "its link's root_b for a table is not the length its statement fixes for that table's root_B (2^cap0 digests)"

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

/-- A coin tree the release log shows sampled, its key, nonce and root in lowercase hex and its spec, and the rounds it
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

/-- `_stream`: a stream's name, 1 to `MAX_CHALLENGE_PHASE_ID` ASCII letters, digits, '-', '.', '_' and '/'. -/
private def isStream (j : Json) : Bool :=
  match j with
  | .str s => 0 < s.length && s.length ≤ MAX_CHALLENGE_PHASE_ID &&
      s.all fun c => c.isAlphanum && c.toNat < 128 || c == '-' || c == '.' || c == '_' || c == '/'
  | _ => false

/-- `tree_spec_reason`'s domain: `{block, rounds_log, scheme, streams}`, `TREE_SCHEME`, 1 to `MAX_BLOCK` coins a round,
at most `2^MAX_ROUNDS_LOG` rounds a stream, and its streams distinct names (`isStream`). -/
def isTreeSpec (spec : Json) : Bool :=
  keys? spec == some #["block", "rounds_log", "scheme", "streams"] && jsonStrIs (getD spec "scheme") TREE_SCHEME &&
    (match nat? (getD spec "block") with | some b => 1 ≤ b && b ≤ MAX_BLOCK | none => false) &&
    (match nat? (getD spec "rounds_log") with | some r => r ≤ MAX_ROUNDS_LOG | none => false) &&
    (match getD spec "streams" with
      | .arr ss => !ss.isEmpty && ss.all isStream && distinct (ss.toList.map strOf)
      | _ => false)

/-- `coin_trees_of` (module docstring): the coin trees the release log `log` shows the challenger sampled, in the order
sampled, each with the rounds it shows opened; every other line is a root's or a release's (`releasedRootsOf`). -/
def coinTreesOf (log : List String) : V (Array LoggedTree) := do
  let mut trees : Array LoggedTree := #[]
  for line in log do
    unless treeLine line do continue
    let d ← event line
    if jsonStrIs (getD d "event") TREE_EVENT then
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
      let some t := trees[i]? | throw LOG_MALFORMED
      let .str s := getD d "stream" | throw LOG_MALFORMED
      let (some k, some n) := (nat? (getD d "round"), nat? (getD d "n")) | throw LOG_MALFORMED
      let streams := match getD t.spec "streams" with | .arr ss => ss | _ => #[]
      unless streams.any (jsonStrIs · s) && k == (t.opened.filter (·.stream == s)).size &&
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
def recordCoins? (cs : Json) : Option ByteArray :=
  match cs with
  | .arr xs => xs.foldlM (init := ByteArray.empty) fun acc c => match c with
    | .str s => if s.length == 32 then (Flock.unhex s).toOption.map (acc ++ ·) else none
    | _ => none
  | _ => none

/-- `record_rounds`, **the rounds a session's record gives stream `stream` of its coin tree**: one of `TREE_SESSION`, its
round 0 the link points; of another stream, the size of `rounds` of the record's first stream so named; none of a stream
the record doesn't name. -/
def recordRounds (record : Json) (stream : String) : Nat :=
  if stream == TREE_SESSION then 1
  else
    match getD record "streams" with
    | .arr ss =>
      match ss.find? (fun x => match getD x "stream" with | .str n => n == stream | _ => false) with
      | some x =>
        match getD x "rounds" with
        | .arr rs => rs.size
        | _ => 0
      | none => 0
    | _ => 0

private def streamT (name : String) (rounds : Nat) : Json :=
  Json.mkObj [("rounds", Json.arr (Array.replicate rounds Json.null)), ("stream", Json.str name)]
private def roundsT : Json := Json.mkObj [("streams", Json.arr #[streamT "t/rep0" 2, streamT "t/rep0" 5,
  streamT TREE_SESSION 3])]

#guard recordRounds roundsT TREE_SESSION == 1
#guard recordRounds roundsT "t/rep0" == 2
#guard recordRounds roundsT "t/rep1" == 0
#guard recordRounds Json.null "t/rep0" == 0
#guard recordRounds Json.null TREE_SESSION == 1

/-- The refusal of a session whose coin tree opens a round its record doesn't give its stream (`recordRounds`). -/
def EXTRA_OPENING : String :=
  "its coin tree opens a round its record doesn't record: the challenger opens a tree's " ++ TREE_SESSION ++
  " round 0 and each round of each stream the record names, as the session records it, and no other"

/-- `session_tree_reason` (module docstring): why `record`, a `--zk` session's record on the tables `tables`, each with
its `root_B`'s byte length where its statement is set up, doesn't show its coins to be ones the release log shows the
challenger released (`trees`, `coinTreesOf`'s); `none` when it does. -/
def sessionTreeReason (trees : Array LoggedTree) (tables : Array (String × Option Nat)) (record : Json) :
    Option String := Id.run do
  let ct := getD record "coin_tree"
  let some t := trees.find? (fun (t : LoggedTree) => jsonStrIs (getD ct "root") t.root) | return some NO_TREE
  unless jeq ct t.record do
    return some "its coin tree is not the one the challenger's release log shows it sampled under that root"
  let link := getD record "link"
  let (.obj _, .obj _, .arr ps) := (link, getD link "root_b", getD link "points") | return some TREE_RECORD_FORM
  unless keys? (getD link "root_b") == some (sortStrings (tables.map (·.1))) do return some LINK_TABLES
  let flat := ps.foldl (fun (acc : Array Json) (p : Json) => match p with
    | .arr cs => acc ++ cs
    | _ => acc.push Json.null) #[]
  let rootB := (entries (getD link "root_b")).filterMap fun (kv : String × Json) => match kv.2 with
    | .str s => (Flock.unhex s).toOption.map (kv.1, ·)
    | _ => none
  let (some rootF, some points, .arr streams) := ((match getD link "root_f" with
      | .str s => (Flock.unhex s).toOption
      | _ => none), recordCoins? (Json.arr flat), getD record "streams") | return some TREE_RECORD_FORM
  unless rootB.size == (entries (getD link "root_b")).size do return some TREE_RECORD_FORM
  unless tables.all (fun (tn : String × Option Nat) => tn.2.all fun n =>
      rootB.any fun (kb : String × ByteArray) => kb.1 == tn.1 && kb.2.size == n) do
    return some LINK_ROOT_LEN
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
        | some o => o.n == n && jsonStrIs (getD r "msg_sha512") o.after && o.coins == sha512Hex coins
        | none => false
      unless ok do
        return some s!"its {name} round {k} is not coins the challenger opened under its tree after that round's message"
  unless t.opened.all fun o => o.round < recordRounds record o.stream do return some EXTRA_OPENING
  return none

/-- `sessions_tree_reason` (module docstring): why the `--zk` sessions `records`, each its name, its tables with their
`root_B` lengths (`sessionTreeReason`'s) and its record, don't show their coins to be the ones the release log shows
sampled and opened (`trees`), one tree a session; `none` when they do. In order, each session's coins its tree's
(`sessionTreeReason`), a tree no earlier session's (`SHARED_TREE`); then every tree a session's (`UNUSED_TREE`). -/
def sessionsTreeReason (trees : Array LoggedTree) (records : Array (String × Array (String × Option Nat) × Json)) :
    Option String := Id.run do
  let mut used : Array String := #[]
  for (name, tables, record) in records do
    let root := strOf (getD (getD record "coin_tree") "root")
    let why := sessionTreeReason trees tables record <|> if used.contains root then some SHARED_TREE else none
    if let some w := why then
      return some s!"session {name}'s coins are not the challenger's: {w}"
    used := used.push root
  unless used.size == trees.size do
    return some s!"the challenger's release log shows {trees.size} coin trees and the sessions use {used.size}: {UNUSED_TREE}"
  return none

/-- The coin trees the one session the auditor's ledger opens on `window` under `key` shows the challenger sampled and
opened (`coinTreesOf`), from the sample's coin evidence `{ledger, main, pins, segment, session}` (`drawCoins`'s, read
through `trustedSessionOf`). -/
def trustedTreesOf (key : ChallengerKey) (window : String) (coins : Json) : V (Array LoggedTree) := do
  unless keys? coins == some #["ledger", "main", "pins", "segment", "session"] do throw DRAW_FORM
  let _ ← trustedSessionOf (getD coins "ledger") (Json.mkObj [("root", Json.str window)]) key (getD coins "pins")
    (getD coins "main") (getD coins "session")
  let .arr ls := getD (getD coins "session") "log" | throw LOG_FORM
  coinTreesOf (← ls.toList.mapM fun | .str s => pure s | _ => throw LOG_FORM)

/-- The refusal of a release log out of the order `sampleLinkFirst` holds it to. -/
def SAMPLE_LINK_FIRST : String :=
  "the challenger's release log is not the sample, then the link, then the other rounds: a coin of the sample is " ++
  "released after a coin tree's round is opened, or the first round opened is not a tree's " ++ TREE_SESSION ++
  " stream, the link points"

/-- **The family and the root a root line of the release log shows**, as `releasedRootsOf` reads them; `none` where the
line doesn't parse or names no family or no root. -/
def rootOf (l : String) : Option (String × ByteArray) := do
  let r ← (event l).toOption
  let .str label := getD r "label" | none
  let .str root := getD r "root" | none
  pure (label, ← (Flock.unhex root).toOption)

/-- Whether release log line `l` opens a coin tree's round: an event whose `event` is `OPEN_EVENT`. -/
def openLine (l : String) : Bool :=
  match event l with
  | .ok j => jsonStrIs (getD j "event") OPEN_EVENT
  | .error _ => false

/-- Whether release log line `l` opens a round of a coin tree's stream `TREE_SESSION`, the stream a session's link
points are opened from. -/
def linkLine (l : String) : Bool :=
  match event l with
  | .ok j => jsonStrIs (getD j "event") OPEN_EVENT && jsonStrIs (getD j "stream") TREE_SESSION
  | .error _ => false

/-- `sampleLinkFirst`'s scan, `opened` whether a line opening a coin tree's round came already and `pending` the root's
line awaiting its coin's release. -/
def sampleLinkFirstFrom (label : String) (after : ByteArray) : Bool → Option String → List String → Bool
  | _, _, [] => true
  | opened, pending, l :: ls =>
    if treeLine l then
      if opened = false ∧ openLine l = true ∧ linkLine l = false then false
      else sampleLinkFirstFrom label after (opened || openLine l) pending ls
    else match pending with
      | none => sampleLinkFirstFrom label after opened (some l) ls
      | some a =>
        if opened = true ∧ rootOf a = some (label, after) then false
        else sampleLinkFirstFrom label after opened none ls

/-- `sample_link_first`, **the sample, then the link, then the other rounds**, in the release log's lines `log` after its
head. Read as `releasedRootsOf` reads them, a coin tree's line aside (`treeLine`) and the rest in pairs, a root's line
and its coin's release: no coin of family `label` whose root's line names `after` (`rootOf`) is released after a line
opening a coin tree's round (`openLine`), and the first such line opens the stream `TREE_SESSION` (`linkLine`), whose
first round is the link points. A coin tree's own line may come anywhere. In v2's one-stage audit `label` is the
parameters' challenger family and `after` the registration's digest, so the coins so released are the sample's
(`challengerStream` holds the sample to every one of them), and the rounds opened are the sessions' links and their
streams' rounds (`sessionsTreeReason`). -/
def sampleLinkFirst (label : String) (after : ByteArray) (log : List String) : Bool :=
  sampleLinkFirstFrom label after false none log

private def rootT : String := "{\"event\":\"root\",\"index\":0,\"label\":\"draw\",\"received_at\":0,\"root\":\"ab\"}"
private def releaseT : String := "{\"after\":\"ab\",\"event\":\"release\",\"index\":0,\"label\":\"draw\",\"released_at\":0}"
private def openT (stream : String) : String :=
  "{\"after\":\"cd\",\"coins\":\"ef\",\"event\":\"tree-open\",\"n\":1,\"round\":0,\"stream\":\"" ++ stream ++
    "\",\"tree\":\"01\"}"
private def treeT : String := "{\"event\":\"coin-tree\",\"key\":\"02\",\"nonce\":\"03\",\"root\":\"01\",\"spec\":{}}"

#guard sampleLinkFirst "draw" ⟨#[0xab]⟩ [treeT, rootT, releaseT, openT TREE_SESSION, openT "circuit/rep0"]
#guard sampleLinkFirst "draw" ⟨#[0xab]⟩ [rootT, treeT, releaseT, openT TREE_SESSION]
#guard !sampleLinkFirst "draw" ⟨#[0xab]⟩ [treeT, openT TREE_SESSION, rootT, releaseT]
#guard !sampleLinkFirst "draw" ⟨#[0xab]⟩ [rootT, openT TREE_SESSION, releaseT]
#guard !sampleLinkFirst "draw" ⟨#[0xab]⟩ [rootT, releaseT, openT "circuit/rep0", openT TREE_SESSION]
#guard sampleLinkFirst "other" ⟨#[0xab]⟩ [openT TREE_SESSION, rootT, releaseT]
#guard sampleLinkFirst "draw" ⟨#[0xcd]⟩ [openT TREE_SESSION, rootT, releaseT]

/-! A member's session: its closing round, and the members' sessions one after another. -/

/-- The round of a coin tree's `session` stream that closes a member's session: the stream's second, after the link
points. -/
def CLOSING_ROUND : Nat := 1

/-- The refusal of a member's session whose coin tree's last opening is not its closing round
(`TreeOpening.closes`). -/
def NO_CLOSING : String :=
  "its coin tree's last opening is not its " ++ TREE_SESSION ++ " round " ++ toString CLOSING_ROUND ++
  " of no coins, opened after the SHA-512 of the session's evidence: a member's session ends with that round"

/-- The refusal of members' sessions the release log doesn't show one after another, in the sessions' order
(`membersOrder`). -/
def MEMBERS_ORDER : String :=
  "the challenger's release log does not open the sessions' coin trees one after another, in the sessions' order: " ++
  "a round of a tree is opened before the closing round of the tree opened before it, or after its own"

/-- Whether `o` closes a member's session whose evidence has the SHA-512 `evidence` (lowercase hex): round
`CLOSING_ROUND` of the stream `TREE_SESSION`, of no coins, opened after `evidence`. -/
def TreeOpening.closes (o : TreeOpening) (evidence : String) : Bool :=
  o.stream == TREE_SESSION && o.round == CLOSING_ROUND && o.n == 0 && o.after == evidence &&
    o.coins == sha512Hex ByteArray.empty

/-- `t` without the last round the release log shows opened of it. -/
def LoggedTree.unclosed (t : LoggedTree) : LoggedTree := { t with opened := t.opened.pop }

/-- **A member's session's coins and its closing round**: why `record`, the record of a member's `--zk` session on
the tables `tables` (`sessionTreeReason`'s), whose evidence has the SHA-512 `evidence` (lowercase hex), doesn't show
its coins to be ones the release log shows the challenger released (`trees`) and its session closed after that
evidence; `none` when it does. Its `coin_tree` is a logged tree (`NO_TREE`), the last round the log shows opened of
that tree closes the session after `evidence` (`TreeOpening.closes`, `NO_CLOSING`), and the tree without that round
is the session's as `sessionTreeReason` holds one: so the closing round is the one opening the record doesn't give
its stream that `EXTRA_OPENING` lets pass. -/
def memberTreeReason (trees : Array LoggedTree) (tables : Array (String × Option Nat)) (record : Json)
    (evidence : String) : Option String :=
  match trees.find? (fun (t : LoggedTree) => jsonStrIs (getD (getD record "coin_tree") "root") t.root) with
  | none => some NO_TREE
  | some t =>
    match t.opened.back? with
    | some o => if o.closes evidence then sessionTreeReason #[t.unclosed] tables record else some NO_CLOSING
    | none => some NO_CLOSING

/-- The tree a release log line opens a round of (its root) and whether the round is of the stream `TREE_SESSION`;
`none` of a line that opens no round. -/
def openOf (l : String) : Option (String × Bool) :=
  match event l with
  | .ok j =>
    if jsonStrIs (getD j "event") OPEN_EVENT then
      match getD j "tree" with
      | .str root => some (root, jsonStrIs (getD j "stream") TREE_SESSION)
      | _ => none
    else none
  | .error _ => none

/-- `membersOrder`'s scan: `roots` the trees opened so far, each when a round of it follows another tree's, `last`
the last of them, and `closed` whether it has had a second round of its stream `TREE_SESSION` opened. -/
def membersOrderFrom : Array String → Option String → Bool → List String → Option (Array String)
  | roots, _, _, [] => some roots
  | roots, last, closed, l :: ls =>
    match openOf l with
    | none => membersOrderFrom roots last closed ls
    | some (root, session) =>
      if last = some root then
        if closed then none else membersOrderFrom roots last session ls
      else if closed ∨ last = none then membersOrderFrom (roots.push root) (some root) false ls
      else none

/-- **The members' sessions one after another**, in the release log's lines `log`: the trees it opens rounds of, in
the order it first opens each, where every round it opens is of the tree opened last, before that tree's second
round of its stream `TREE_SESSION` (its closing round), or the first of the next tree, after it; `none` where a round
of a tree is opened before the closing round of the tree opened before it, or after its own. A tree opened again
after another is listed again, so a list without a tree twice shows each tree's rounds together. O(the log's lines). -/
def membersOrder (log : List String) : Option (Array String) := membersOrderFrom #[] none false log

/-- **The members' sessions' coins, each session closed before the next begins**: why the members' `--zk` sessions
`records`, each its name, its tables with their `root_B` lengths, its record and its evidence's SHA-512, don't show
their coins to be the ones the release log `log` shows sampled and opened (`trees`, `coinTreesOf`'s of it), one tree a
session, the sessions one after another in the order given; `none` when they do. The first session whose coins are
not its tree's or whose closing round is not after its evidence (`memberTreeReason`); then no tree two sessions'
(`SHARED_TREE`); then every tree a session's (`UNUSED_TREE`); then the log opens the sessions' trees one after another
in the sessions' order, each tree's closing round before the next tree's first round (`membersOrder`,
`MEMBERS_ORDER`). -/
def membersTreeReason (trees : Array LoggedTree) (log : List String)
    (records : Array (String × Array (String × Option Nat) × Json × String)) : Option String :=
  let used := records.map fun x => strOf (getD (getD x.2.2.1 "coin_tree") "root")
  match records.find? fun x => (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).isSome with
  | some x =>
    (memberTreeReason trees x.2.1 x.2.2.1 x.2.2.2).map (s!"session {x.1}'s coins are not the challenger's: {·}")
  | none =>
    if ¬ distinct used.toList then some s!"two sessions' coins are one tree's: {SHARED_TREE}"
    else if used.size ≠ trees.size then
      some s!"the challenger's release log shows {trees.size} coin trees and the sessions use {used.size}: {UNUSED_TREE}"
    else if membersOrder log ≠ some used then some MEMBERS_ORDER
    else none

private def openAtT (tree stream : String) : String :=
  "{\"after\":\"cd\",\"coins\":\"ef\",\"event\":\"tree-open\",\"n\":1,\"round\":0,\"stream\":\"" ++ stream ++
    "\",\"tree\":\"" ++ tree ++ "\"}"
private def closingT (after : String) : TreeOpening :=
  ⟨TREE_SESSION, CLOSING_ROUND, 0, after, sha512Hex ByteArray.empty⟩
private def loggedT (opened : Array TreeOpening) : LoggedTree :=
  { key := "02", nonce := "03", root := "01", spec := Json.null, opened }
private def recordT : Json := Json.mkObj [("coin_tree", Json.mkObj [("root", Json.str "01")])]

#guard (closingT "ab").closes "ab" && !(closingT "ab").closes "cd"
#guard !({ closingT "ab" with n := 1 } : TreeOpening).closes "ab"
#guard !({ closingT "ab" with round := 0 } : TreeOpening).closes "ab"
#guard !({ closingT "ab" with stream := "circuit/rep0" } : TreeOpening).closes "ab"
#guard memberTreeReason #[loggedT #[]] #[] recordT "ab" == some NO_CLOSING
#guard memberTreeReason #[loggedT #[closingT "cd"]] #[] recordT "ab" == some NO_CLOSING
#guard memberTreeReason #[loggedT #[closingT "ab", ⟨"circuit/rep0", 0, 1, "cd", "ef"⟩]] #[] recordT "ab" == some NO_CLOSING
#guard memberTreeReason #[] #[] recordT "ab" == some NO_TREE
#guard (memberTreeReason #[loggedT #[closingT "ab"]] #[] recordT "ab").all (· != NO_CLOSING)
#guard membersOrder [treeT, rootT, releaseT] == some #[]
#guard membersOrder [openAtT "01" TREE_SESSION, openAtT "01" "circuit/rep0", openAtT "01" TREE_SESSION,
  openAtT "02" TREE_SESSION, openAtT "02" TREE_SESSION] == some #["01", "02"]
#guard membersOrder [openAtT "01" TREE_SESSION, openAtT "02" TREE_SESSION] == none
#guard membersOrder [openAtT "01" TREE_SESSION, openAtT "01" "circuit/rep0", openAtT "02" TREE_SESSION] == none
#guard membersOrder [openAtT "01" TREE_SESSION, openAtT "01" TREE_SESSION, openAtT "01" "circuit/rep0"] == none
#guard membersOrder [openAtT "01" TREE_SESSION, openAtT "01" TREE_SESSION, openAtT "02" TREE_SESSION,
  openAtT "02" TREE_SESSION, openAtT "01" TREE_SESSION] == some #["01", "02", "01"]

/-! Two members' sessions as an honest challenger's release log shows them: each tree sampled, its link points opened
after its link's roots, one round of its table's first rep, then its closing round after its evidence, the second
member's first round after the first member's closing round. -/

private def lineT (j : Json) : String := (Flock.canon j).toOption.getD ""
private def hexT (c : Char) (n : Nat) : String := "".pushn c n
private def bytesT (s : String) : ByteArray := (Flock.unhex s).toOption.getD ByteArray.empty
private def specT : Json :=
  Json.mkObj [("block", Lean.toJson (512 : Nat)), ("rounds_log", Lean.toJson (9 : Nat)),
    ("scheme", Json.str TREE_SCHEME),
    ("streams", Json.arr #[Json.str TREE_SESSION, Json.str "t/rep0", Json.str "t/rep1"])]
private def coinTreeT (c : Char) : Json :=
  Json.mkObj [("key", Json.str (hexT c 512)), ("nonce", Json.str (hexT c 64)), ("root", Json.str (hexT c 128)),
    ("spec", specT)]
private def helloT (c : Char) : String := lineT ((coinTreeT c).setObjVal! "event" (Json.str TREE_EVENT))
private def roundT (c : Char) (stream : String) (k n : Nat) (after coins : String) : String :=
  lineT (Json.mkObj [("after", Json.str after), ("coins", Json.str coins), ("event", Json.str OPEN_EVENT),
    ("n", Lean.toJson n), ("round", Lean.toJson k), ("stream", Json.str stream), ("tree", Json.str (hexT c 128))])
private def pointsT : String := hexT '1' 32 ++ hexT '2' 32
private def memberRecordT (c : Char) : Json :=
  Json.mkObj [("coin_tree", coinTreeT c),
    ("link", Json.mkObj [("points", Json.arr #[Json.arr #[Json.str (hexT '1' 32)], Json.arr #[Json.str (hexT '2' 32)]]),
      ("root_b", Json.mkObj [("t", Json.str (hexT 'b' 16))]), ("root_f", Json.str (hexT 'f' 16))]),
    ("streams", Json.arr #[Json.mkObj [("rounds", Json.arr #[Json.mkObj [("coins", Json.arr #[Json.str (hexT '3' 32)]),
      ("msg_sha512", Json.str (hexT 'e' 128)), ("n", Lean.toJson (1 : Nat))]]), ("stream", Json.str "t/rep0")]])]
private def evidenceT (s : String) : String := sha512Hex s.toUTF8
private def memberLogT (c : Char) (evidence : String) : List String :=
  [roundT c TREE_SESSION 0 2 (Flock.hexOf (linkAfter (bytesT (hexT 'f' 16)) #[("t", bytesT (hexT 'b' 16))]))
      (sha512Hex (bytesT pointsT)),
    roundT c "t/rep0" 0 1 (hexT 'e' 128) (sha512Hex (bytesT (hexT '3' 32))),
    roundT c TREE_SESSION CLOSING_ROUND 0 evidence (sha512Hex ByteArray.empty)]
private def membersT (log : List String) (members : List (Char × String)) : Option String :=
  match coinTreesOf log with
  | .ok trees => membersTreeReason trees log
      (members.toArray.map fun (c, ev) => (String.singleton c, #[("t", some 8)], memberRecordT c, ev))
  | .error why => some why
private def membersReasonT (log : List String) (evA evB : String) : Option String :=
  membersT log [('a', evA), ('b', evB)]
private def oneReasonT (log : List String) : Option String :=
  match coinTreesOf log with
  | .ok trees => sessionsTreeReason trees #[("a", #[("t", some 8)], memberRecordT 'a')]
  | .error why => some why
private def closedT (c : Char) : List String := memberLogT c (evidenceT (String.singleton c))

#guard membersReasonT ([helloT 'a', helloT 'b'] ++ closedT 'a' ++ closedT 'b') (evidenceT "a") (evidenceT "b") == none
#guard membersReasonT ([helloT 'a'] ++ closedT 'a' ++ [helloT 'b'] ++ closedT 'b') (evidenceT "a") (evidenceT "b") ==
  none
#guard membersReasonT ([helloT 'a', helloT 'b'] ++ closedT 'a' ++ closedT 'b') (evidenceT "a") (evidenceT "x") ==
  some s!"session b's coins are not the challenger's: {NO_CLOSING}"
#guard membersReasonT ([helloT 'a', helloT 'b'] ++ (closedT 'a').dropLast ++ closedT 'b') (evidenceT "a")
  (evidenceT "b") == some s!"session a's coins are not the challenger's: {NO_CLOSING}"
#guard membersReasonT ([helloT 'a', helloT 'b'] ++ (closedT 'a').dropLast ++ (closedT 'b').take 1 ++
  (closedT 'a').drop 2 ++ (closedT 'b').drop 1) (evidenceT "a") (evidenceT "b") == some MEMBERS_ORDER
#guard membersReasonT ([helloT 'a', helloT 'b'] ++ closedT 'b' ++ closedT 'a') (evidenceT "a") (evidenceT "b") ==
  some MEMBERS_ORDER
#guard membersT ([helloT 'a'] ++ closedT 'a') [('a', evidenceT "a"), ('a', evidenceT "a")] ==
  some s!"two sessions' coins are one tree's: {SHARED_TREE}"
#guard membersT ([helloT 'a', helloT 'b'] ++ closedT 'a') [('a', evidenceT "a")] ==
  some s!"the challenger's release log shows 2 coin trees and the sessions use 1: {UNUSED_TREE}"
#guard oneReasonT ([helloT 'a'] ++ (closedT 'a').dropLast) == none
#guard oneReasonT ([helloT 'a'] ++ closedT 'a') == some s!"session a's coins are not the challenger's: {EXTRA_OPENING}"

/-! The difftest's pieces. -/

private def refusing (f : V Json) : Json :=
  match f with
  | .ok j => j
  | .error why => Json.mkObj [("refused", Json.str why)]

private def logOf (j : Json) : V (List String) := do
  let .arr ls := getD j "log" | throw "log is not a list of strings"
  ls.toList.mapM fun | .str s => pure s | _ => throw "log is not a list of strings"

/-- `{"log"}` to `coin_trees_of`'s, `{"trees": [tree, ...]}` in the order sampled, each with its `opened` rounds. -/
def coinTreesPiece (j : Json) : V Json := do
  unless keys? j == some #["log"] do throw "a request's keys are log"
  let lines ← logOf j
  pure <| refusing do
    return Json.mkObj [("trees", Json.arr ((← coinTreesOf lines).map (·.toJson)))]

/-- `{"log", "records": [{name, record, tables: {table: length or null}}, ...]}` to `sessions_tree_reason`'s over
`coin_trees_of`'s trees of `log`, `{"reason"}`. -/
def sessionTreesPiece (j : Json) : V Json := do
  unless keys? j == some #["log", "records"] do throw "a request's keys are log, records"
  let lines ← logOf j
  let .arr rs := getD j "records" | throw "records is not a list of {name, record, tables}"
  let records ← rs.mapM fun r => do
    unless keys? r == some #["name", "record", "tables"] do throw "records is not a list of {name, record, tables}"
    let (.str name, .obj _) := (getD r "name", getD r "tables") | throw "records is not a list of {name, record, tables}"
    let tables ← (entries (getD r "tables")).mapM fun (t, n) => match n, nat? n with
      | .null, _ => pure (t, none)
      | _, some n => pure (t, some n)
      | _, none => throw "a record's tables are not lengths or null"
    pure (name, tables, getD r "record")
  pure <| refusing do
    let why := sessionsTreeReason (← coinTreesOf lines) records
    return Json.mkObj [("reason", why.elim Json.null Json.str)]

/-- `{"after", "label", "log"}` to `sample_link_first`'s over the release log `log` after its head, at family `label` and
digest `after` (lowercase hex), `{"reason"}`: null when it holds, else `SAMPLE_LINK_FIRST`. -/
def sampleLinkFirstPiece (j : Json) : V Json := do
  unless keys? j == some #["after", "label", "log"] do throw "a request's keys are after, label, log"
  let lines ← logOf j
  let .str label := getD j "label" | throw "label is not a string"
  let .str a := getD j "after" | throw "after is not lowercase hex"
  let after ← Flock.unhex a
  let why := if sampleLinkFirst label after lines.tail then Json.null else Json.str SAMPLE_LINK_FIRST
  return Json.mkObj [("reason", why)]

end Spot.Challenger
