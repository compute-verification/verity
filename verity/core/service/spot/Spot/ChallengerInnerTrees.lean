/-
A recursive session's inner coins as the challenger's release log shows it sampled and opened them: `rec_live`'s live-coin
part (`verity_flock.rec_live.CoinServer`) on the challenger's coin tree, held to the log as `rec_live.tree_held_reason`
holds it (`tests/test_challenger_inner_trees.py` holds that to this, the verifier).

`rec_live`'s live-coin part opens each round of its session's tree once given what the round answers, which are other bytes
than the ones `flock-circuit serve` opens a round after (`sessionTreeReason`): the link points, the tree's `session`
round 0, after `linkAfterRoots` of its record's `link.roots` (`rec_live.link_after`: the SHA-512 of their canonical
JSON), and each round its record's `rounds` list (`{stream, k, coins, ...}`), its stream's round `k`, after `roundAfter`
of that entry (`rec_live.round_after`: the SHA-512 of its canonical JSON without its coins).

`innerTreeReason` holds one inner session to the trees the log shows (`coinTreesOf`'s): its `coin_tree` a logged one,
under its key, nonce, root and spec; its link points and every round its live-coin part's record (`public.json`,
`rec/inner-transcript/v1`) lists opened as the record says, after those bytes, the record's coins for it as many as the
log shows and hashing to its digest; no round read twice; and every opening the log shows of its tree one the record
reads. `innerTreesReason` holds an audit's inner sessions to them: each session's so, no tree two sessions', and every
logged tree a session's, so no session's coins are flipped twice.

The difftest's piece: `challenger_inner_trees` (`tree_held_reason`), `{"refused": why}` where `coin_trees_of` refuses
the log.
-/
module

public import Spot.ChallengerTrees

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Challenger

open Lean (Json)
open Flock (V)

def INNER_RECORD_FORM : String :=
  "its coin server's record is not one rec_live writes (link {points, roots}, rounds [{stream, k, coins, ...}])"

private def formed {α : Type} (x : V α) : V α := x.mapError fun _ => INNER_RECORD_FORM

private def hexBytes (j : Json) : V ByteArray :=
  match j with
  | .str s => formed (Flock.unhex s)
  | _ => throw INNER_RECORD_FORM

/-- `rec_live.link_after`: what the challenger's live-coin part opens the link points after, the SHA-512 of `Commit`'s salted tops as
its record writes them (`link.roots`: each table's nodes in hex), in canonical JSON. -/
def linkAfterRoots (roots : Json) : V ByteArray := do
  let .obj kvs := roots | throw INNER_RECORD_FORM
  let rs ← kvs.toArray.mapM fun (t, v) => do
    let .arr xs := v | throw INNER_RECORD_FORM
    pure (t, Json.arr (← xs.mapM fun x => do pure (Json.str (Flock.hexOf (← hexBytes x)))))
  pure (Flock.Sha512.hash (← formed (Flock.canon (Json.mkObj rs.toList))).toUTF8)

/-- `rec_live.round_after`: what the challenger's live-coin part opens a round's coins after, the SHA-512 of its record of the round
(`rounds`' entry) without its coins, in canonical JSON. -/
def roundAfter (entry : Json) : V ByteArray := do
  let .obj kvs := entry | throw INNER_RECORD_FORM
  let rest := Json.mkObj (kvs.toArray.filter (·.1 != "coins")).toList
  pure (Flock.Sha512.hash (← formed (Flock.canon rest)).toUTF8)

/-- A round's coins as the record writes them, each in hex: how many, and the SHA-512 of their bytes in order. -/
def coinsRead (cs : Json) : V (Nat × String) := do
  let .arr xs := cs | throw INNER_RECORD_FORM
  let b ← xs.foldlM (init := ByteArray.empty) fun acc c => do pure (acc ++ (← hexBytes c))
  pure (xs.size, sha512Hex b)

/-- A round an inner session's record reads of its tree: the stream, the round, how many coins, what it was opened
after and its coins' SHA-512, each in lowercase hex. -/
structure InnerRead where
  stream : String
  round : Nat
  n : Nat
  after : String
  coins : String

/-- What the challenger's live-coin part's record `pub` reads of its session's tree, in order: the link points as the tree's `session`
round 0, after `linkAfterRoots` of `link.roots`, then each entry of `rounds`, its stream's round `k`, after `roundAfter`
of the entry. -/
def innerReads (pub : Json) : V (Array InnerRead) := do
  let link := getD pub "link"
  let (n, coins) ← coinsRead (getD link "points")
  let after ← linkAfterRoots (getD link "roots")
  let .arr rounds := getD pub "rounds" | throw INNER_RECORD_FORM
  let rs ← rounds.mapM fun r => do
    let (.str s, some k) := (getD r "stream", nat? (getD r "k")) | throw INNER_RECORD_FORM
    let (n, coins) ← coinsRead (getD r "coins")
    pure { stream := s, round := k, n, after := Flock.hexOf (← roundAfter r), coins : InnerRead }
  pure (#[{ stream := TREE_SESSION, round := 0, n, after := Flock.hexOf after, coins }] ++ rs)

/-- `tree_held_reason`'s one session (module docstring): why `tree`, an inner session's `coin_tree`, and `pub`, its live-coin
part's record, don't show its coins to be ones the release log shows the challenger released (`trees`,
`coinTreesOf`'s); `none` when they do. -/
def innerTreeReason (trees : Array LoggedTree) (tree pub : Json) : Option String := Id.run do
  let some t := trees.find? (fun (t : LoggedTree) => getD tree "root" == Json.str t.root) | return some NO_TREE
  unless jeq tree t.record do
    return some "its coin tree is not the one the challenger's release log shows it sampled under that root"
  let .ok reads := innerReads pub | return some INNER_RECORD_FORM
  let mut seen : Array (String × Nat) := #[]
  for r in reads do
    if seen.contains (r.stream, r.round) then
      return some s!"its coin server's record reads its {r.stream} round {r.round} twice"
    seen := seen.push (r.stream, r.round)
    let ok := match t.opening r.stream r.round with
      | some o => o.n == r.n && o.after == r.after && o.coins == r.coins
      | none => false
    unless ok do
      return some (s!"its {r.stream} round {r.round} is not coins the challenger opened under its tree after what " ++
        "the coin server recorded")
  unless t.opened.size == reads.size do
    return some s!"the challenger's release log shows {t.opened.size} openings of its tree and its record {reads.size}"
  return none

/-- `tree_held_reason` (module docstring): why the inner sessions `sessions`, each its name, its `coin_tree` and its live-coin
part's record, don't show their coins to be the ones the release log shows sampled and opened (`trees`), one tree a
session; `none` when they do. In order, each session's coins its tree's (`innerTreeReason`), a tree no earlier
session's (`SHARED_TREE`); then every tree a session's (`UNUSED_TREE`). -/
def innerTreesReason (trees : Array LoggedTree) (sessions : Array (String × Json × Json)) : Option String := Id.run do
  let mut used : Array String := #[]
  for (name, tree, pub) in sessions do
    let root := match getD tree "root" with | .str s => s | _ => ""
    let why := innerTreeReason trees tree pub <|> if used.contains root then some SHARED_TREE else none
    if let some w := why then
      return some s!"session {name}'s coins are not the challenger's: {w}"
    used := used.push root
  unless used.size == trees.size do
    return some s!"the challenger's release log shows {trees.size} coin trees and the sessions use {used.size}: {UNUSED_TREE}"
  return none

/-! The difftest's piece. -/

/-- `{"log", "sessions": [{name, public, tree}, ...]}` to `tree_held_reason`'s over `coin_trees_of`'s trees of `log`,
`{"reason"}`, or `{"refused": why}` where `coin_trees_of` refuses the log. -/
def innerTreesPiece (j : Json) : V Json := do
  unless keys? j == some #["log", "sessions"] do throw "a request's keys are log, sessions"
  let .arr ls := getD j "log" | throw "log is not a list of strings"
  let lines ← ls.toList.mapM fun | .str s => pure s | _ => throw "log is not a list of strings"
  let .arr ss := getD j "sessions" | throw "sessions is not a list of {name, public, tree}"
  let sessions ← ss.mapM fun s => do
    unless keys? s == some #["name", "public", "tree"] do throw "sessions is not a list of {name, public, tree}"
    let .str name := getD s "name" | throw "sessions is not a list of {name, public, tree}"
    pure (name, getD s "tree", getD s "public")
  match coinTreesOf lines with
  | .ok trees => pure (Json.mkObj [("reason", (innerTreesReason trees sessions).elim Json.null Json.str)])
  | .error why => pure (Json.mkObj [("refused", Json.str why)])

end Spot.Challenger
