/-
`RecSetup.checkReads`: the inner transcript read in the coin server's own format (`RecRecord.readFull`) holds what
its JSON readers took of it. At the plain session's inner shape (m 27, k_log 22, 8 claims), a record on the schedule
that `recordFullW` writes passes. The same record as compact JSON is refused as not in its own format, and bytes that
differ from what the JSON readers read in a `c_k`, a salted top, a node of `root_B`, a link point, a round's coin count
or Hello's tables are refused at that field. So are bytes whose coins differ from the JSON's, among them a tail that
repeats the key `"rounds"` with other coins, which `Json.parse` reads and `readFull` doesn't.
-/
import Flock.RecSetup

namespace Flock.Test.RecReads

open Lean Flock.RecVStar Flock.RecRecord Flock.RecSetup

def inner : Inner := { m := 27, kLog := 22, claims := 8, regions := 1 }

def bs (s : String) : List UInt8 := s.toUTF8.toList

/-- A `b ‖ c` of 128 bytes `v`. -/
def node (v : Nat) : List UInt8 := List.replicate 128 (UInt8.ofNat v)

def coin : F128 := ⟨0, 0⟩

/-- The rounds of a session on `inner`'s schedule over the table `circuit`: tops of `node 1`, `c_k` `node 2`. -/
def rounds : List RecRecord.Round :=
  ((scheduleOf inner "circuit").getD #[]).toList.flatMap fun (s, _, _, rs) => rs.toList.mapIdx fun k w =>
    ({ stream := bs s, k, tops := w.2.1.toList.map (List.replicate · (node 1)), level0 := w.2.2.1,
       commits := if w.2.2.2 then [node 2] else [], n := w.1, coins := List.replicate w.1 coin } : RecRecord.Round)

def hello : Hello :=
  { digests := bs "d", flavor := bs "f", m := 2, points := 1, profile := bs "p", reps := 2, rounds := bs "r",
    tables := [bs "circuit"] }

def honest : Record :=
  { rounds, kind := bs TRANSCRIPT_FORMAT, tops := bs "t", hello,
    roots := [(bs "circuit", List.replicate 128 (node 3))], points := [coin, coin] }

/-- The record's fields past its link points, and its close. -/
def rest : List UInt8 :=
  fld 2 "y" ++ bs "null" ++ [10] ++ sp 1 ++ [125] ++ fld 1 "end" ++ strStr (bs "finish") ++ [10, 125]

def text (r : Record) : String := (String.fromUTF8? ⟨(recordFullW r rest).toArray⟩).getD ""

def tr : Json := (Json.parse (text honest)).toOption.getD Json.null

/-- The inner session's record, holding the coin server's link points. -/
def record : String :=
  let c := "\"" ++ String.ofList (List.replicate 32 '0') ++ "\""
  "{\"link\": {\"points\": [[" ++ c ++ ", " ++ c ++ "]]}}"

/-- Each rep's `c_k` as `innerCommits` holds them. -/
def cs : InnerCommits :=
  { reps := #[0, 1].map fun (x : Nat) => ((rounds.filter (·.stream == bs s!"circuit/rep{x}")).filterMap fun r =>
      if r.commits.isEmpty then none else some (r.k, ByteArray.mk (node 2).toArray)).toArray }

def why (t : String) : String := match checkReads inner t tr record cs with
  | .ok () => ""
  | .error e => e

def at_ : String := "rec inner transcript: the coin server's record read in its own format holds "

/-- `r`'s rounds with the `j`-th round of stream `s` holding something under `f`. -/
def change (r : Record) (s : String) (j : Nat) (f : RecRecord.Round → RecRecord.Round) : Record :=
  let i := ((r.rounds.zipIdx.filter (·.1.stream == bs s)).map (·.2)).getD j 0
  { r with rounds := r.rounds.mapIdx fun j' x => if j' = i then f x else x }

/-- The `j`-th round of stream `s` that holds a top. -/
def topAt (r : Record) (s : String) (j : Nat) : Nat :=
  let ks := (r.rounds.filter fun x => x.stream == bs s && !x.tops.isEmpty).map (·.k)
  ks.getD j 0

#guard (scheduleOf inner "circuit").isSome && rounds.length == 192
#guard readFull (bs (text honest)) == some honest
#guard why (text honest) == ""
#guard why tr.compress == notOwnFormat
#guard why (text (change honest "circuit/rep0" 2 fun x => { x with commits := x.commits.map fun _ => node 4 })) ==
  at_ ++ "rep 0's c_k other than its JSON's"
#guard ((honest.rounds.filter (·.stream == bs "circuit/rep0"))[2]?.map (·.commits.isEmpty)) == some false
#guard why (text (change honest "circuit/rep1" (topAt honest "circuit/rep1" 0) fun x =>
    { x with tops := x.tops.map (·.map fun _ => node 5) })) == at_ ++ "rep 1's salted tops other than its JSON's"
#guard why (text { honest with roots := [(bs "circuit", node 6 :: List.replicate 127 (node 3))] }) ==
  at_ ++ "a link.roots other than circuit's root_B alone, as its JSON reads it"
#guard why (text { honest with points := [coin, ⟨1, 0⟩] }) == at_ ++ "link points other than its JSON's"
#guard why (text (change honest "circuit/rep0" 0 fun x => { x with n := x.n + 1 })) ==
  at_ ++ "rounds other than its schedule's (stream, round, coin count, tops, level 0 and commits)"
#guard why (text { honest with hello := { hello with tables := [bs "other"] } }) ==
  at_ ++ "Hello's tables other than circuit alone, as its JSON reads them"

/-! Coins: every round's coins read in the record's own format must be the JSON's, hex-parsed. -/

/-- `honest` with coin `j` of round `i` the value `(hi, lo) = (i + 1, 2^40 · j + 7)`, so no two coins agree. -/
def coined : Record :=
  { honest with rounds := honest.rounds.mapIdx fun i x =>
      { x with coins := x.coins.mapIdx fun j _ => ⟨UInt64.ofNat (2 ^ 40 * j + 7), UInt64.ofNat (i + 1)⟩ } }

/-- `rest`, then the tail repeating the key `"rounds"` with the rounds of `r`: `Json.parse` keeps the last, `readFull`
reads the first. -/
def restRepeat (r : Record) : List UInt8 :=
  fld 2 "y" ++ bs "null" ++ [10] ++ sp 1 ++ [125] ++ fld 1 "rounds" ++ listW 1 roundW r.rounds ++ fld 1 "end" ++
    strStr (bs "finish") ++ [10, 125]

def textWith (r : Record) (tail : List UInt8) : String := (String.fromUTF8? ⟨(recordFullW r tail).toArray⟩).getD ""

def whyAt (t : String) (j : Json) : String := match checkReads inner t j record cs with
  | .ok () => ""
  | .error e => e

def parsed (t : String) : Json := (Json.parse t).toOption.getD Json.null

/-- `coined` with its first coin of rep 0's round 1 changed. -/
def recoined : Record :=
  change coined "circuit/rep0" 1 fun x =>
    { x with coins := x.coins.mapIdx fun j c => if j = 0 then ⟨c.lo + 1, c.hi⟩ else c }

def roundsOf (r : Record) : List (Option (List UInt8) × Option Nat × List F128) :=
  r.rounds.map fun x => (some x.stream, some x.k, x.coins)

def refusedAt : String := at_ ++ "rounds other than its JSON's (stream, round and coins)"

#guard ((coined.rounds.filter (·.stream == bs "circuit/rep0"))[1]?.map (·.coins.length)) != some 0
#guard recoined != coined
-- An honest record with distinct coins passes: `readFull`'s coins are `f128OfHex`'s of the same hex.
#guard readFull (bs (text coined)) == some coined
#guard whyAt (text coined) (parsed (text coined)) == ""
#guard (jsonRounds (parsed (text coined))).toOption == some (roundsOf coined)
-- Bytes whose coins differ from what the JSON readers read are refused.
#guard whyAt (text recoined) (parsed (text coined)) == refusedAt
-- A tail that repeats `"rounds"` with other coins: the JSON reads the tail's, the record's own format the first, and
-- every other field agrees, so only the coins refuse it.
#guard readFull (bs (textWith coined (restRepeat recoined))) == some coined
#guard (jsonRounds (parsed (textWith coined (restRepeat recoined)))).toOption == some (roundsOf recoined)
#guard whyAt (textWith coined (restRepeat recoined)) (parsed (textWith coined (restRepeat recoined))) == refusedAt
-- The same tail repeating the record's own rounds passes.
#guard whyAt (textWith coined (restRepeat coined)) (parsed (textWith coined (restRepeat coined))) == ""

end Flock.Test.RecReads
