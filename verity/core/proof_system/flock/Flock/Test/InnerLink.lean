/-
The circuit table's tops and the link points as the coin server recorded them, at every shape: `RecVStar.innerLink`,
`topRow`, `checkTopRowsAt` and `checkLinkRounds`, `RecCoins.checkLevelTopReads`, and the record on the verifier's
schedule (`checkSchedule`). A transcript of two reps whose
`root_B` has two nodes and whose reps commit one later top each, at round 1 (rep 0's of two nodes, rep 1's of one), and
two link points of two coins: `rec-top-L0`'s row `x` is node `x` of `root_B`, `rec-top-L1`'s rep 0's nodes then rep 1's.
A row other than the coin server's `b ‖ c` at `root_B` or at a later top is refused, and so is an inner session record
whose link points are not the coin server's.
-/
import Flock.RecSetup

namespace Flock.Test.InnerLink

open Lean Flock Flock.RecVStar

def b (n : Nat) : ByteArray := ByteArray.mk (Array.replicate 128 n.toUInt8)
def hx (n : Nat) : Json := Json.str (hexOf (b n))
def coin (n : Nat) : Json := Json.str (hexOf (ByteArray.mk (Array.replicate 16 n.toUInt8)))
def nodes (xs : List Nat) : Json := Json.arr (xs.map hx).toArray

def round (s : String) (k : Nat) (tops : List (List Nat)) : Json :=
  Json.mkObj [("stream", Json.str s), ("k", toJson k), ("tops", Json.arr (tops.map nodes).toArray),
    ("level0", toJson (0 : Nat)), ("commits", Json.arr #[]), ("n", toJson (0 : Nat)), ("coins", Json.arr #[])]

def transcript (root : List Nat := [1, 2]) (top0 : List Nat := [3, 4]) (top1 : List Nat := [5])
    (points : List Nat := [10, 11, 12, 13]) (tables : List String := ["circuit"]) : Json :=
  Json.mkObj [("kind", Json.str TRANSCRIPT_FORMAT),
    ("hello", Json.mkObj [("tables", toJson tables), ("link", Json.mkObj [("m", toJson 2), ("points", toJson 2)])]),
    ("link", Json.mkObj [("roots", Json.mkObj [("circuit", nodes root)]), ("points", Json.arr (points.map coin).toArray),
      ("y", Json.null)]),
    ("rounds", Json.arr #[round "circuit/rep1" 1 [top1], round "circuit/rep0" 0 [], round "circuit/rep0" 1 [top0],
      round "circuit/rep1" 0 []]), ("end", Json.str "finish")]

def record (points : List (List Nat) := [[10, 11], [12, 13]]) : String :=
  (Json.mkObj [("link", Json.mkObj [("points", Json.arr (points.map fun p => Json.arr (p.map coin).toArray).toArray)])]).compress

def why {α : Type} (x : V α) : Option String :=
  match x with
  | .error e => some e
  | .ok _ => none

def linkOf (tr : Json) : InnerLink := (innerLink tr (record)).toOption.getD default
def link : InnerLink := linkOf (transcript)

#guard why (innerLink (transcript) (record)) == none
#guard link.table == "circuit" && link.root.toList.map (·.data.toList) == [(b 1).data.toList, (b 2).data.toList]
#guard link.tops.map (·.map fun (k, t) => (k, t.size)) == #[#[(1, 2)], #[(1, 1)]]
#guard link.points.map (·.size) == #[2, 2] && link.points[1]![0]! == ((f128OfHex (hexOf (ByteArray.mk (Array.replicate 16 12)))).toOption.getD default)

-- only a finished session's record is read (`checkFinished`): one the coin server marked stopped, or naming no end, is not
def stopWhy : String :=
  "rec inner transcript: the inner session ended stop, not finish: a stopped session is no inner proof"
#guard why (innerLink ((transcript).setObjVal! "end" (Json.str "stop")) (record)) == some stopWhy
#guard why (checkFinished ((transcript).setObjVal! "end" (Json.str "stop"))) == some stopWhy
#guard why (checkFinished (transcript)) == none
#guard why (checkFinished (Json.mkObj [("kind", Json.str TRANSCRIPT_FORMAT)])) ==
  some "rec inner transcript: the coin server's record names no end; only a finished session is read"
#guard why (innerCommits ((transcript).setObjVal! "end" (Json.str "stop")) (record)) == some stopWhy

-- the link points: the inner session's record (the prover's, which check 5 reads) holds the coin server's
def pointsWhy : String :=
  "rec inner transcript: the inner session's record's link points are not the ones the coin server drew (link.points): " ++
    "check 5 would take the extra claims at points the prover chose"
#guard why (innerLink (transcript) (record [[10, 11], [12, 14]])) == some pointsWhy
#guard why (innerLink (transcript (points := [10, 11, 12, 14])) (record)) == some pointsWhy
#guard why (innerLink (transcript) (record [[12, 13], [10, 11]])) == some pointsWhy
#guard why (innerLink (transcript) (record [[10, 11]])) == some pointsWhy
#guard why (innerLink (transcript) "{\"link\": {}}") ==
  some "rec inner transcript: the inner session's record holds no link points (link.points)"
#guard why (innerLink (transcript (points := [10, 11, 12])) (record)) ==
  some "rec inner transcript: 3 link coins for 2 points of 2"
#guard why (innerLink (transcript (tables := ["circuit", "other"])) (record)) ==
  some "rec inner transcript: Hello's tables #[circuit, other], not one table"
#guard why (innerLink (transcript (root := [])) (record)) == some "rec inner transcript: circuit's root_B has no nodes"
#guard why (innerLink (transcript (top0 := [])) (record)) ==
  some "rec inner transcript: circuit/rep0 round 1's top 0 has no nodes"

-- Every node kept is a `b ‖ c` of 128 bytes, which a top's piece length rests on; a shorter node is refused
#guard link.root.all (·.size == 128) && link.tops.all (·.all fun (_, t) => t.all (·.size == 128))
def shortTop : Json :=
  (transcript).setObjVal! "rounds" (Json.arr #[round "circuit/rep1" 1 [[5]], round "circuit/rep0" 0 [],
    (round "circuit/rep0" 1 []).setObjVal! "tops"
      (Json.arr #[Json.arr #[hx 3, Json.str (hexOf (ByteArray.mk (Array.replicate 64 4)))]]),
    round "circuit/rep1" 0 []])
#guard why (innerLink (transcript) (record)) == none && why (innerLink shortTop (record)) ==
  some "rec inner transcript: circuit/rep0 round 1's top 0: a b ‖ c of 64 bytes, not 128"

-- `rec-top-L<l>`'s rows: root_B at level 0, both reps'; above it rep 0's top, then rep 1's
def row (lk : InnerLink) (l x : Nat) : Option (List UInt8) := (topRow lk l x).toOption.map (·.2.data.toList)
def is (n : Nat) : Option (List UInt8) := some (b n).data.toList

#guard row link 0 0 == is 1 && row link 0 1 == is 2 && row link 0 2 == none
#guard row link 1 0 == is 3 && row link 1 1 == is 4 && row link 1 2 == is 5 && row link 1 3 == none
#guard row link 2 0 == none
#guard (topRow link 1 2).toOption.map (·.1) == some "node 0 of rep 1's level 1 top"

-- the statement's rows at its reads of rec-top-L<l> are the coin server's: honest rows pass, and the prover's rows against a
-- transcript whose root_B or later top differs are refused, with the row and the node it is not
def rows (xs : List Nat) (r : Nat) : ByteArray := b (xs.getD r 0)
#guard why (checkTopRowsAt "u" "out" "rec-top-L0" link 0 #[1, 0, 1] (rows [2, 1, 2])) == none
#guard why (checkTopRowsAt "u" "out" "rec-top-L1" link 1 #[0, 2, 1] (rows [3, 5, 4])) == none
#guard why (checkTopRowsAt "u" "out" "rec-top-L0" (linkOf (transcript (root := [1, 9]))) 0 #[1, 0] (rows [2, 1])) ==
  some "rec-tops: u: port out's table row 0 is not node 1 of circuit's root_B, which the coin server committed"
#guard why (checkTopRowsAt "u" "out" "rec-top-L1" (linkOf (transcript (top0 := [3, 6]))) 1 #[0, 1] (rows [3, 4])) ==
  some "rec-tops: u: port out's table row 1 is not node 1 of rep 0's level 1 top, which the coin server committed"
#guard why (checkTopRowsAt "u" "out" "rec-top-L1" (linkOf (transcript (top1 := [6]))) 1 #[0, 2] (rows [3, 5])) ==
  some "rec-tops: u: port out's table row 1 is not node 0 of rep 1's level 1 top, which the coin server committed"
#guard why (checkTopRowsAt "u" "out" "rec-top-L1" link 1 #[3] (rows [5])) ==
  some "rec-tops: u: port out's table row 0 reads row 3 of rec-top-L1, and the reps' level 1 tops have 3 nodes, no row 3"
#guard why (checkTopRowsAt "u" "out" "rec-top-L2" link 2 #[0] (rows [5])) ==
  some "rec-tops: u: port out's table row 0 reads row 0 of rec-top-L2, and the inner transcript has no level 2 top of both reps"

#guard topLevel "rec-top-L0" == some 0 && topLevel "rec-top-L12" == some 12
#guard topLevel "rec-top-stacked-L1" == none && topLevel "rec-acc" == none && topLevel "rec-top-L" == none

/-! The tops at the inner shape's schedule (m 25, k_log 22, claims 8): `root_B` of level 0's 128 nodes, and each rep's
later tops at `repRounds`' rounds and node counts (`holoRepRounds`' at a holo shape, one round later). -/

def i25 : Inner := { m := 25, kLog := 22, claims := 8, regions := 3 }
def schedTops (rs : Array SchedRound) : Array (Nat × Array ByteArray) :=
  ((List.range rs.size).flatMap fun k => (rs.getD k default).2.1.toList.map fun n => (k, Array.replicate n (b 0))).toArray
def linkAt (rs : Array SchedRound) (root : Nat := 128) : InnerLink :=
  { table := "circuit", root := Array.replicate root (b 0), tops := #[schedTops rs, schedTops rs], points := #[] }
def reps25 : Array SchedRound := (repRounds 25 22 8).getD #[]
def holo25 : Array SchedRound := (holoRepRounds 25 22 8).getD #[]

#guard (schedTops reps25).map (fun (k, t) => (k, t.size)) == #[(58, 64), (69, 64), (80, 32)]
#guard why (checkLinkRounds i25 (linkAt reps25)) == none
#guard why (checkLinkRounds { i25 with holo := true } (linkAt holo25)) == none
#guard why (checkLinkRounds i25 (linkAt reps25 64)) ==
  some "rec inner transcript: circuit's root_B has 64 nodes, and its schedule's level 0 top 128"
#guard why (checkLinkRounds { i25 with holo := true } (linkAt reps25)) ==
  some "rec inner transcript: rep 0's tops (round, nodes) are [(58, 64), (69, 64), (80, 32)], and its schedule's [(59, 64), (70, 64), (81, 32)]"
#guard why (checkLinkRounds i25 { linkAt reps25 with tops := #[schedTops reps25, (schedTops reps25).pop] }) ==
  some "rec inner transcript: rep 1's tops (round, nodes) are [(58, 64), (69, 64)], and its schedule's [(58, 64), (69, 64), (80, 32)]"
#guard why (checkLinkRounds i25 { linkAt reps25 with
    tops := #[schedTops reps25, (schedTops reps25).modify 1 fun (k, t) => (k, t.pop)] }) ==
  some "rec inner transcript: rep 1's tops (round, nodes) are [(58, 64), (69, 63), (80, 32)], and its schedule's [(58, 64), (69, 64), (80, 32)]"

/-! A circuit-table level's `out` port: at level 1 with `H = 2`, rep 0's openings at positions 5, 13 and 6 read
`rec-top-L1` at nodes 1, 3 and 1 of rep 0's top (4 nodes), rep 1's at 1, 2 and 15 nodes 0, 0 and 3 of rep 1's, rows 4, 4
and 7; at level 0 both reps read `root_B` from row 0. -/

def port (name value : String) (reads : List Nat) : Registered.Port :=
  { name, value, root := .empty, leaves := 8, words := some 64, wordBits := 16, domain := .empty,
    readsRel := (reads.map some).toArray }
def pos0 : Array Nat := #[5, 13, 6]
def pos1 : Array Nat := #[1, 2, 15]
def reads (own : Array Registered.Port) (l : Nat) : Option String :=
  why (RecCoins.checkLevelTopReads "rec-open/l8-h2" own l 3 2 4 pos0 pos1)
def readsWhy : String :=
  "rec-coins: rec-open/l8-h2: rec-top-L1: the reads file does not read rep r's opening j at node pos[j] >>> 2 of its " ++
    "rep's top, rep 1's after rep 0's 4 rows"

#guard reads #[port "acc" "rec-acc" [0, 1], port "out" "rec-top-L1" [1, 3, 1, 4, 4, 7]] 1 == none
#guard reads #[port "out" "rec-top-L0" [1, 3, 1, 0, 0, 3]] 0 == none
#guard reads #[port "out" "rec-top-L1" [1, 3, 1, 0, 0, 3]] 1 == some readsWhy
#guard reads #[port "out" "rec-top-L1" [1, 3, 1, 4, 4]] 1 == some readsWhy
#guard reads #[port "out" "rec-top-L1" [5, 13, 6, 5, 6, 19]] 1 == some readsWhy
#guard reads #[port "out" "rec-top-L0" [1, 3, 1, 4, 4, 7]] 0 ==
  some ("rec-coins: rec-open/l8-h2: rec-top-L0: the reads file does not read rep r's opening j at node pos[j] >>> 2 of " ++
    "its rep's top, rep 1's after rep 0's 0 rows")
#guard reads #[port "out" "rec-top-L0" [1, 3, 1, 4, 4, 7]] 1 ==
  some "rec-coins: rec-open/l8-h2: port out reads rec-top-L0, not level 1's tops rec-top-L1"
#guard reads #[port "out" "rec-top-stacked-L1" [1, 3, 1, 4, 4, 7]] 1 ==
  some "rec-coins: rec-open/l8-h2: port out reads rec-top-stacked-L1, not level 1's tops rec-top-L1"
#guard reads #[port "acc" "rec-acc" [0, 1]] 1 ==
  some "rec-coins: rec-open/l8-h2: the verifier's registered reads hold no port out: level 1's tops would be the prover's"

/-! The coin server's record on the verifier's schedule (`checkSchedule`): a record of `scheduleOf`'s streams and rounds
passes at its inner shape, plain and candidate 3's. A record that ends `finish` and is off it is refused: an extra
round, a missing round, a round of another coin count, `n` or `level0`, rounds out of order, a stream declared with
another round count (a holo record under a public shape), streams beside the schedule's, Hello of two tables, and a
shape with no schedule. -/

def onSched (i : Inner) : Json :=
  let want := (scheduleOf i "circuit").getD #[]
  let rnd := fun (s : String) (k : Nat) (w : SchedRound) => Json.mkObj [("stream", Json.str s), ("k", toJson k),
    ("tops", Json.arr (w.2.1.map fun n => Json.arr (Array.replicate n (hx 0)))), ("level0", toJson w.2.2.1),
    ("commits", Json.arr (if w.2.2.2 then #[hx 0] else #[])), ("n", toJson w.1),
    ("coins", Json.arr (Array.replicate w.1 (coin 0)))]
  Json.mkObj [("kind", Json.str TRANSCRIPT_FORMAT), ("hello", Json.mkObj [("tables", toJson ["circuit"])]),
    ("streams", Json.mkObj (want.toList.map fun (s, p, pos, rs) =>
      (s, Json.mkObj [("parent", Json.str p), ("at", toJson pos), ("rounds", toJson rs.size)]))),
    ("rounds", Json.arr (want.flatMap fun (s, _, _, rs) => rs.mapIdx fun k w => rnd s k w)), ("end", Json.str "finish")]

def roundsOf (j : Json) : Array Json := (j.getObjValAs? (Array Json) "rounds").toOption.getD #[]
def withRounds (f : Array Json → Array Json) (j : Json := onSched i25) : Json :=
  j.setObjVal! "rounds" (Json.arr (f (roundsOf j)))
def offWhy (i : Inner) (why : String) : Option String :=
  some (s!"rec inner transcript: the coin server's record is off the schedule of the verifier's inner shape " ++
    s!"(m {i.m}, k_log {i.kLog}, claims {i.claims}): {why}")
def c3i : Inner :=
  { i25 with holo := true, sparse := some { nu := 18, ell := 13, sLog := 6, tLog := 0 }, stacked := some ⟨23, 2⟩ }

#guard (roundsOf (onSched i25)).size == 188
#guard why (checkSchedule i25 (onSched i25)) == none
#guard why (checkSchedule c3i (onSched c3i)) == none
#guard why (checkSchedule i25 (withRounds fun rs => rs.push rs.back!)) ==
  offWhy i25 "189 rounds, and the schedule's 188"
#guard why (checkSchedule i25 (withRounds (·.pop))) == offWhy i25 "187 rounds, and the schedule's 188"
#guard why (checkSchedule i25 (withRounds (·.modify 3 fun r =>
    (r.setObjVal! "n" (toJson 2)).setObjVal! "coins" (Json.arr #[coin 0, coin 0])))) ==
  offWhy i25 ("round 3 is circuit/rep0 round 3 (n 2) (coins 2, tops [], level0 0, commits true), and the schedule's " ++
    "circuit/rep0 round 3 (n 1) (coins 1, tops [], level0 0, commits true)")
#guard why (checkSchedule i25 (withRounds (·.modify 3 fun r => r.setObjVal! "n" (toJson 2)))) ==
  offWhy i25 ("round 3 is circuit/rep0 round 3 (n 2) (coins 1, tops [], level0 0, commits true), and the schedule's " ++
    "circuit/rep0 round 3 (n 1) (coins 1, tops [], level0 0, commits true)")
#guard why (checkSchedule i25 (withRounds (·.modify 0 fun r => r.setObjVal! "level0" (toJson 0)))) ==
  offWhy i25 ("round 0 is circuit/rep0 round 0 (n 6) (coins 6, tops [], level0 0, commits false), " ++
    "and the schedule's circuit/rep0 round 0 (n 6) (coins 6, tops [], level0 1, commits false)")
#guard why (checkSchedule i25 (withRounds (·.reverse))) ==
  offWhy i25 ("round 0 is circuit/rep1 round 93 (n 1) (coins 1, tops [], level0 0, commits false), " ++
    "and the schedule's circuit/rep0 round 0 (n 6) (coins 6, tops [], level0 1, commits false)")
#guard why (checkSchedule i25 (onSched { i25 with holo := true })) ==
  offWhy i25 "circuit/rep0 is not declared at no parent with 94 rounds"
#guard why (checkSchedule i25 (onSched c3i)) ==
  offWhy i25 ("it declares the streams [circuit/rep0, circuit/rep1, sparse/rep0, sparse/rep1, stacked/rep0], " ++
    "and the schedule [circuit/rep0, circuit/rep1]")
#guard why (checkSchedule i25 ((onSched i25).setObjVal! "hello" (Json.mkObj [("tables", toJson ["circuit", "x"])]))) ==
  offWhy i25 "Hello names no one table"
#guard why (checkSchedule { i25 with m := 6 } (onSched i25)) == offWhy { i25 with m := 6 } "the shape has no schedule"

/-! Below k_log 6 `SchedRound`'s absorb flag reads a 2-element round as z_partial's 64-element one, so `checkSchedule`
alone passes a record on that misread schedule; `statements`, which every V* session's setup runs (`RecSetup.checks`,
`checkStmt`, `check`), refuses the shape first. -/

def i5 : Inner := { i25 with kLog := 5 }
#guard why (checkSchedule i5 (onSched i5)) == none
#guard why (statements i5) == some ("the inner session's shape (m 25, k_log 5, claims 8) is not V*'s: a fast100 " ++
  "schedule, k_log at least 6, claims at least 2")
#guard why (statements { c3i with kLog := 5 }) == some ("the inner session's shape (m 25, k_log 5, claims 8) is not " ++
  "V*'s: a fast100 schedule, k_log at least 6, claims at least 2")

end Flock.Test.InnerLink
