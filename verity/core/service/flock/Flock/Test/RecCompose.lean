/-
`RecCompose.check`: honest parts compose, one log per V* statement with one `VERDICT`, each accepted, `--zk` and without
an identifier at the one staged rows file and inner shape, every part naming one set of shared inputs in the compiled
verifier's form and its own coin key. Each shared input changed in one part, a part checked against other rows or
another inner shape, a shared set missing an input or holding check 5's coins, a reused or missing coin key, an
identifier, a session without `--zk`, a refused session, a missing or repeated part, a log with no `VERDICT` or two, and a
`VERDICT` that is not JSON are refused.
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

def passes (n : Nat) (parts : List (String × String)) (at_ : String := inner) : Bool := (check n rows at_ parts).toBool

def why (n : Nat) (parts : List (String × String)) : String :=
  match check n rows inner parts with
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
  ("L1", log 1 (verdict 1 { sh with registration := none }))]
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

end Flock.Test.RecCompose
