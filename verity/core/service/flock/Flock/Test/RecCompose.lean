/-
`RecCompose.check`: honest parts compose, one log per V* statement with one `VERDICT`, each accepted, `--zk` and without
an identifier at the one staged rows file and inner shape, every part naming one set of shared inputs in the compiled
verifier's form and its own coin key. Each shared input changed in one part, a part checked against other rows or
another inner shape, a shared set missing an input or holding check 5's coins, a reused or missing coin key, an
identifier, a session without `--zk`, a refused session, a missing or repeated part, a log with no `VERDICT` or two, and a
`VERDICT` that is not JSON are refused. Against the challenger's interactive-run record, a part retried (two sessions
opened for it), a part with no session opened, an opened session that is not the composed one, a session opened for a
part V* lacks, a staging after the first Hello or a second one, other rows or another shared set staged, a record not
closed or going on past its close, and a record that is not one or holds an event out of its form are refused.
-/
import Flock.RecCompose

namespace Flock.Test.RecCompose

open Lean (Json)
open Flock.RecCompose

def hex (c : Char) : String := String.ofList (List.replicate 128 c)

def rows : String := hex 'a'

def inner : String := hex '1'

def sh : Shared :=
  ⟨some inner, some (hex '2'), some (hex '3'), some (hex '4'), none, some (hex '5'), some (hex '6'), some (hex '7'),
    some "batched", some true, some [("range8", hex '8')], none⟩

/-- A part's `VERDICT` line as `verify --zk --rec-part J` writes it. -/
def verdict (j : Nat) (s : Shared := sh) (digest : String := rows) (ok : Bool := true) (zk : Bool := true)
    (key : String := s!"k{j}") (ident : Bool := false) : String :=
  let fields : List (String × Json) :=
    [("session", .str s!"s{j}"), ("accepted", .bool ok), ("why", .str ""), ("rec_part", .num j),
      ("rec_rows_sha512", .str digest), ("rec_shared", s.toJson), ("zk", .bool zk), ("coin_key", .str key)] ++
    (if ident then [("identifier", .str "id")] else [])
  "VERDICT " ++ (Json.mkObj fields).compress

def log (j : Nat) (v : String := verdict j) : String := s!"setup: 3 ms\n{v}\n"

def honest : List (String × String) := [("L0", log 0), ("L1", log 1), ("alg-p0", log 2), ("cap", log 3)]

/-- The challenger's interactive-run record's events as `challenger.RunRecord` writes them. -/
def stageJ (r : String := rows) (s : Shared := sh) : Json :=
  Json.mkObj [("event", .str "stage"), ("rec_rows_sha512", .str r), ("rec_shared", s.toJson)]

def openJ (j : Nat) (key : String := s!"k{j}") : Json :=
  Json.mkObj [("event", .str "open"), ("session", .str s!"s{j}-{key}"), ("rec_part", .num j), ("coin_key", .str key)]

def closeJ : Json := Json.mkObj [("event", .str "close")]

def runText (evs : List Json) : String := (Json.mkObj [("kind", .str RUN_KIND), ("events", .arr evs.toArray)]).compress

/-- The honest run of `n` parts: the staging, one session opened per part, the close. -/
def opens (n : Nat) : List Json := (List.range n).map (openJ ·)

def run (n : Nat) (s : Shared := sh) : String := runText (stageJ (s := s) :: opens n ++ [closeJ])

def passes (n : Nat) (parts : List (String × String)) (at_ : String := inner) (r : String := run n) : Bool :=
  (check n rows at_ r parts).toBool

def why (n : Nat) (parts : List (String × String)) (r : String := run n) : String :=
  match check n rows inner r parts with
  | .error e => e
  | .ok () => "accepted"

#guard passes 4 honest
#guard passes 1 [("L0", log 0)]

/-- Each shared input changed in one part: another digest, or none where the first part held one. -/
def tampered : List Shared :=
  [{ sh with recInner := some (hex 'f') }, { sh with innerCircuit := some (hex 'f') },
    { sh with innerPublic := some (hex 'f') }, { sh with innerPartition := none },
    { sh with innerCoins := some (hex 'f') }, { sh with innerSession := some (hex 'f') },
    { sh with transcript := some (hex 'f') }, { sh with registration := none },
    { sh with innerStatement := some "unbatched" }, { sh with innerStatementCoins := some false },
    { sh with innerTables := some [("range8", hex 'f')] }, { sh with innerDraw := some (hex 'f') }]

#guard tampered.all fun s => s != sh
#guard tampered.all fun s => !passes 4 (honest.set 2 ("alg-p0", log 2 (verdict 2 s)))
#guard why 4 (honest.set 2 ("alg-p0", log 2 (verdict 2 { sh with transcript := some (hex 'f') }))) ==
  s!"alg-p0: its inner_transcript (the coin server's transcript, --rec-inner-transcript) is {hex 'f'}, not {hex '6'} as the first part's"
-- every part alike, but not in the compiled verifier's form
#guard [{ sh with recInner := some (hex 'f') }, { sh with recInner := none }, { sh with innerCircuit := none },
    { sh with innerPublic := none }, { sh with innerPartition := none }, { sh with innerCoins := some (hex 'f') },
    { sh with innerSession := none }, { sh with transcript := none }, { sh with innerStatement := none },
    { sh with innerStatementCoins := none }, { sh with innerTables := none }].all fun s =>
  !passes 2 [("L0", log 0 (verdict 0 s)), ("L1", log 1 (verdict 1 s))]
-- what the compiled verifier leaves optional may be absent, alike in every part
#guard passes 2 [("L0", log 0 (verdict 0 { sh with registration := none })),
  ("L1", log 1 (verdict 1 { sh with registration := none }))] (r := run 2 { sh with registration := none })
-- the composition's own inner shape
#guard !passes 4 honest (hex 'f')
-- a part checked against other rows
#guard !passes 4 (honest.set 2 ("alg-p0", log 2 (verdict 2 (digest := hex 'b'))))
#guard why 4 (honest.set 2 ("alg-p0", log 2 (verdict 2 (digest := hex 'b')))) ==
  s!"alg-p0: checked against the rows file {hex 'b'}, not the one staged rows file {rows}"
-- a reused coin key
#guard !passes 4 (honest.set 3 ("cap", log 3 (verdict 3 (key := "k0"))))
-- an identifier, a session without --zk, a refused session
#guard !passes 4 (honest.set 1 ("L1", log 1 (verdict 1 (ident := true))))
#guard !passes 4 (honest.set 1 ("L1", log 1 (verdict 1 (zk := false))))
#guard !passes 4 (honest.set 1 ("L1", log 1 (verdict 1 (ok := false))))
-- a VERDICT missing its rows digest, its shared inputs, its coin key, its zk or its rec_part
#guard ["rec_rows_sha512", "rec_shared", "coin_key", "zk", "rec_part"].all fun k =>
  let v := match Json.parse ((verdict 0).drop 8).toString with
    | .ok (.obj o) => "VERDICT " ++ (Json.obj (o.erase k)).compress
    | _ => ""
  !passes 1 [("L0", log 0 v)]
-- a part missing, a part twice, a part outside V*'s statements, one part too many
#guard !passes 4 (honest.take 3)
#guard !passes 4 (honest.set 3 ("cap", log 2 (verdict 2 (key := "k3"))))
#guard !passes 3 honest
#guard !passes 4 (honest ++ [("L0'", log 0 (verdict 0 (key := "k4")))])
-- no VERDICT, two VERDICTs in one log, a VERDICT that is not JSON, no part at all
#guard !passes 4 (honest.set 3 ("cap", "setup: refused\n"))
#guard !passes 1 [("L0", log 0 ++ verdict 0 (key := "k9"))]
#guard !passes 1 [("L0", "VERDICT {\"accepted\":true\n")]
#guard !passes 0 []

/-! ### One run: the challenger's interactive-run record -/

-- part 1 retried: two sessions opened for it, whichever of them is composed
#guard !passes 4 honest (r := runText (stageJ :: opens 2 ++ [openJ 1 "k1'"] ++ [openJ 2, openJ 3, closeJ]))
#guard why 4 honest (runText (stageJ :: opens 2 ++ [openJ 1 "k1'"] ++ [openJ 2, openJ 3, closeJ])) ==
  "part 1: 2 sessions opened in the run record, not one: a part is never retried"
#guard !passes 4 honest (r := runText (stageJ :: [openJ 0, openJ 1 "k1'", openJ 1, openJ 2, openJ 3, closeJ]))
-- a part's session not in the record, and a part with no session opened
#guard why 4 honest (runText (stageJ :: opens 3 ++ [closeJ])) ==
  "part 3: 0 sessions opened in the run record, not one: a part is never retried"
-- the one session opened for part 1 is not the one composed
#guard why 4 honest (runText (stageJ :: [openJ 0, openJ 1 "k9", openJ 2, openJ 3, closeJ])) ==
  "run record: session s1-k9 was opened for part 1, and it is not the session composed for that part"
-- a session opened for a part V* doesn't have
#guard why 4 honest (runText (stageJ :: opens 4 ++ [openJ 4, closeJ])) ==
  "run record: session s4-k4 was opened for part 4: V* has 4 statements"
-- the staging after the first Hello, a second staging, another rows file or shared set staged
#guard why 4 honest (runText (openJ 0 :: stageJ :: (opens 4).drop 1 ++ [closeJ])) ==
  "run record: its first event is not the staging: V*'s statements are staged before the first part's Hello"
#guard why 4 honest (runText (stageJ :: openJ 0 :: stageJ :: (opens 4).drop 1 ++ [closeJ])) ==
  "run record: a second staging: V*'s statements are staged once, before the first part's Hello"
#guard why 4 honest (runText (stageJ (hex 'b') :: opens 4 ++ [closeJ])) ==
  s!"run record: it staged the rows file {hex 'b'}, not the composition's {rows}"
#guard why 4 honest (run 4 { sh with transcript := some (hex 'f') }) ==
  "L0: its rec_shared is not the one the run record staged before the first Hello"
-- a record not closed, or going on past its close
#guard !passes 4 honest (r := runText (stageJ :: opens 4))
#guard why 4 honest (runText (stageJ :: opens 4 ++ [closeJ, openJ 1 "k1'"])) == "run record: an event after its close"
-- no record, not one, an event out of its form
#guard why 4 honest "" == "run record: not JSON"
#guard why 4 honest "{\"kind\":\"verity/rec-run/v0\",\"events\":[]}" == "run record: not a verity/rec-run/v1 record"
#guard why 4 honest "{\"kind\":\"verity/rec-run/v1\"}" == "run record: no list of events"
#guard why 4 honest (runText (stageJ :: openJ 0 :: Json.mkObj [("event", .str "open"), ("session", .str "s1")] ::
    (opens 4).drop 2 ++ [closeJ])) == "run record: its event 2 is not a stage, an open or a close in its form"

end Flock.Test.RecCompose
