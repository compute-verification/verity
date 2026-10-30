/-
RecCap, V*'s statements that hold each level's committed tops at fixed rows (`rec-cap/n<N>` at level 0,
`rec-cap/n<N>-l<l>` at a later level, `VStar.capUnit`): `RecVStar.checkCapListed` refuses a V* list without one at a level,
at another node count, twice, or past the last level; `RecVStar.checkTopsListed` refuses later levels' caps that are not
the reps' schedule's tops; `RecVStar.checkShared` refuses a port reading `rec-top-L<l>` (level `l`'s `out`, its RecCap's
`top`) without the verifier's shared rows registering it, or at another registration; `RecCoins.checkCap` refuses reads of
a level's tops other than node `i` in instance `i` of its `N` nodes of 64 words, and a table whose instances are not
labelled `0` to `n - 1` in order; `RecCoins.check` refuses a `rec-cap/` unit whose name `VStar.capParams` does not
read. The unit is the cap row's words past its node, then zeros, and lays out as V*'s file.

These are Flock's audit root for RecCap, each under a few seconds. V*'s whole list (`RecVStar.statements`) is about
20 minutes in the interpreter at the deployed shape and over 3 at the smallest, so here its list checks run on the list
it assembles at m 22, and `Flock.Test.RecCap`, which builds it at the deployed shape, runs after landing.
-/
import Flock.RecCoins

namespace Flock.Test.RecCapSmall

open Lean Flock Flock.RecVStar

#guard VStar.capName 0 128 == "rec-cap/n128"
#guard VStar.capName 3 64 == "rec-cap/n64-l3"
#guard VStar.capParams "rec-cap/n128" == some (0, 128)
#guard VStar.capParams "rec-cap/n64-l3" == some (3, 64)
#guard VStar.capParams "rec-cap/n0128" == none
#guard VStar.capParams "rec-cap/n64-l0" == none
#guard VStar.capParams "rec-cap/n64-l03" == none
#guard VStar.capParams "rec-cap/n64-l3-l4" == none
#guard VStar.capParams "rec-cap/n" == none
#guard VStar.capParams "rec-open/l16-h4" == none
#guard isVStar "rec-cap/n128" && isVStar "rec-cap/n64-l3"

-- `2^cap₀` at the fixture shapes: 128 nodes of `root_B` at m 22 through 35.
#guard (List.range 14).all fun d => ((fast100 (22 + d)).bind (·.levels[0]?)).map (2 ^ ·.cap) == some 128
#guard match VStar.capStatement { m := 25, kLog := 22, claims := 8 } with
  | .ok (.cap 0 128) => true
  | _ => false
-- The later levels at m 25 (caps 6, 6, 5): both reps' 2^cap_l nodes, 128, 128, 64.
#guard match VStar.levelCapStatements { m := 25, kLog := 22, claims := 8 } with
  | .ok caps => caps.toList.map (fun s => (s.name, s.instances, s.dir)) ==
      [("rec-cap/n128-l1", 128, "cap/L1"), ("rec-cap/n128-l2", 128, "cap/L2"), ("rec-cap/n64-l3", 64, "cap/L3")]
  | .error _ => false

def inner : Inner := { m := 25, kLog := 22, claims := 8, regions := 3 }

def listed (ss : Array VStar.Stmt) : Option String :=
  match checkCapListed inner ss with
  | .error e => some e
  | .ok () => none

def level0 : VStar.Stmt := .recOpen 0 64 8 436

def later : Array VStar.Stmt := #[.cap 1 128, .cap 2 128, .cap 3 64]

#guard listed (#[level0] ++ later ++ #[.cap 0 128]) == none
#guard listed (#[level0] ++ later) ==
  some "V*'s list has no RecCap statement (rec-cap/n128): the table's level-0 cap is held by none of its statements"
#guard listed (#[level0] ++ later ++ #[.cap 0 64]) ==
  some "V*'s list: rec-cap/n64 holds 64 nodes, and the table's level-0 cap has 128 (2^cap₀, cap₀ 7)"
#guard listed (#[.cap 0 128, level0] ++ later ++ #[.cap 0 128]) ==
  some "V*'s list has 2 RecCap statements, and holds the table's level-0 cap once"
#guard listed #[level0, .cap 1 128, .cap 3 64, .cap 0 128] ==
  some "V*'s list has no RecCap statement (rec-cap/n128-l2): level 2's tops are held by none of its statements"
#guard listed #[level0, .cap 1 128, .cap 2 64, .cap 3 64, .cap 0 128] ==
  some "V*'s list: rec-cap/n64-l2 holds 64 nodes, and level 2's tops have 128 (both reps' 2^cap_2, cap_2 6)"
#guard listed (#[level0] ++ later ++ #[.cap 2 128, .cap 0 128]) ==
  some "V*'s list has 2 RecCap statements at level 2, and holds its tops once"
#guard listed (#[level0] ++ later ++ #[.cap 4 64, .cap 0 128]) ==
  some "V*'s list: rec-cap/n64-l4 holds level 4's tops, and the schedule (m 25) has 4 levels"

def topsListed (i : Inner) (caps : Array VStar.Stmt) : Option String :=
  match checkTopsListed i caps with
  | .error e => some e
  | .ok () => none

-- The reps' schedule commits tops of 64, 64 and 32 nodes past `root_B`: the later levels' caps, both reps'.
#guard (repRounds 25 22 8).map topSizes == some [64, 64, 32]
#guard topsListed inner later == none
#guard topsListed { inner with holo := true } later == none
-- At every fast100 shape (m 22 to 35), dense and holo, the later levels' caps are the ones the reps' schedule commits.
#guard [22, 25, 27, 28, 29, 32, 35].all fun m => [false, true].all fun holo =>
  match VStar.levelCapStatements { m, kLog := 22, claims := 8 } with
  | .ok caps => topsListed { inner with m, holo } caps == none
  | .error _ => false
#guard topsListed inner #[.cap 1 128, .cap 2 128] ==
  some "V*'s list: the later levels' caps hold [128, 128] nodes, and the reps' schedule commits tops of [64, 64, 32] nodes a rep ([128, 128, 64] both reps')"
-- V*'s list at the smallest inner shape `statements` takes: m 22 (fast100's fewest levels, 3), k_log 6, claims 2, no region.
-- `statements small` lists the levels' RecOpen, one part, then the later levels' caps and level 0's, and takes 194 s in the
-- interpreter, 55 of them cutting the algebra into that one part (`VStar.parts`) and 76 counting the verifier's groups. So
-- the list is assembled here as `statements` assembles it, its part the algebra's spec uncut (whose message-round ports
-- are the part's), and held to the checks `statements` runs on it. `Flock.Test.RecCap` builds `statements` itself.
def small : Inner := { m := 22, kLog := 6, claims := 2, regions := 0 }
def smallSh : VStar.Algebra.Shape := { m := 22, kLog := 6, claims := 2 }

/-- The part at `small`, the spec uncut, without the row port named `drop`. -/
def smallPart (drop : String := "") : Except String VStar.Stmt := do
  let S ← VStar.Algebra.specOf smallSh false
  let S := { S with ports := S.ports.filter (·.1 != drop) }
  pure (.inner 0 "rec-residuals/m22-k6-c2-p0" { S, res := List.range S.gf.res.length, v := List.range S.gf.nv })

/-- `statements small`'s list with `parts` for its parts: `VStar.statements`' levels, then `parts`, then the caps. -/
def smallList (parts : Array VStar.Stmt) : Except String (Array VStar.Stmt) := do
  let s ← smallSh.sched
  let levels := (Array.range s.levels.size).map fun l =>
    let lv := s.levels[l]!
    VStar.Stmt.recOpen l (2 ^ lv.lanes) (lv.d - lv.cap) (2 * lv.q)
  let caps ← VStar.levelCapStatements smallSh
  checkTopsListed small caps
  pure (levels ++ parts ++ (← candidate3Statements small) ++ caps ++ #[← VStar.capStatement smallSh])

def listChecks (ss : Except String (Array VStar.Stmt)) : Option String :=
  match (do let ss ← ss; checkCandidate3 small ss; checkCapListed small ss; checkRoundsListed small ss) with
  | .error e => some e
  | .ok () => none

#guard match (do smallList #[← smallPart]) with
  | .ok ss => ss.toList.map (·.name) == ["rec-open/l64-h3", "rec-open/l16-h2", "rec-open/l16-h2",
      "rec-residuals/m22-k6-c2-p0", "rec-cap/n128-l1", "rec-cap/n64-l2", "rec-cap/n128"]
  | .error _ => false
#guard listChecks (do smallList #[← smallPart]) == none
#guard listChecks (do return (← smallList #[← smallPart]).filter (·.name != "rec-cap/n64-l2")) ==
  some "V*'s list has no RecCap statement (rec-cap/n64-l2): level 2's tops are held by none of its statements"
#guard listChecks (do return (← smallList #[← smallPart]).push (.cap 3 64)) ==
  some "V*'s list: rec-cap/n64-l3 holds level 3's tops, and the schedule (m 22) has 3 levels"
#guard topsListed small #[.cap 1 128] ==
  some "V*'s list: the later levels' caps hold [128] nodes, and the reps' schedule commits tops of [64, 32] nodes a rep ([128, 64] both reps')"

-- Every message round a rep absorbs is one of V*'s list's message-round ports at both reps' rows (`checkRoundsListed`):
-- the list passes it (above), a part without its last round's port is refused at that round, and a list without the parts
-- at round 0.
#guard (repRounds 22 6 2).map (commitRounds · |>.length) == some 44
#guard listChecks (do smallList #[← smallPart (drop := "m43")]) ==
  some "V*'s list: a rep's message round 43 (rec-c43) is a message-round port of none of its statements"
#guard listChecks (smallList #[]) ==
  some "V*'s list: a rep's message round 0 (rec-c0) is a message-round port of none of its statements"
#guard match checkRoundsListed inner (later ++ #[.cap 0 128]) with
  | .error e => e == "V*'s list: a rep's message round 0 (rec-c0) is a message-round port of none of its statements"
  | .ok () => false
#guard roundListed #[.sparse ⟨1, 6, 0, 0⟩ 7] 7 && !roundListed #[.sparse ⟨1, 6, 0, 0⟩ 7] 6

def port (name value : String) (root : UInt8 := 1) (leaves : Nat := 128) (reads : Array (Option Nat) := #[some 0]) :
    Registered.Port :=
  { name, value, root := ⟨#[root]⟩, leaves, words := some 64, wordBits := 16, domain := ⟨#[7]⟩, readsRel := reads }

def all : Array (Option Nat) := (Array.range 128).map some

def rows : Array Registered.Port := #[port "acc0" "rec-acc", port "rec-top-L0" "rec-top-L0" (reads := all)]

def shared (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : Option String :=
  match checkShared s registered rows with
  | .error e => some e
  | .ok () => none

-- Level 0's `out` and RecCap's `top` pass at the shared rows' one registration of `rec-top-L0`, each at its own reads.
#guard shared level0 #[port "out" "rec-top-L0" (reads := #[some 5, some 9])] (some rows) == none
#guard shared (.cap 0 128) #[port "top" "rec-top-L0" (reads := all)] (some rows) == none
#guard shared (.cap 0 128) #[port "top" "rec-top-L0" (root := 2) (reads := all)] (some rows) ==
  some "rec-cap/n128: its port top reads rec-top-L0, which the verifier's shared rows (--rec-rows) register as rec-top-L0: root 02, against 01"
#guard shared level0 #[port "out" "rec-top-L0" (leaves := 64)] (some rows) ==
  some "rec-open/l64-h8: its port out reads rec-top-L0, which the verifier's shared rows (--rec-rows) register as rec-top-L0: leaves 64, against 128"
#guard shared level0 #[port "out" "rec-top-L0"] (some #[port "acc0" "rec-acc"]) ==
  some "rec-open/l64-h8: its port out reads the table's tops rec-top-L0, which its level's RecCap holds; pass the verifier's shared rows (--rec-rows F), which register it"
#guard shared (.cap 0 128) #[port "top" "rec-top-L0" (reads := all)] none ==
  some "rec-cap/n128: its port top reads the table's tops rec-top-L0, which its level's RecCap holds; pass the verifier's shared rows (--rec-rows F), which register it"
-- A later level's tops need a shared entry too: its RecCap holds them.
#guard shared (.recOpen 1 16 7 212) #[port "out" "rec-top-L1"] (some #[port "acc0" "rec-acc"]) ==
  some "rec-open/l16-h7: its port out reads the table's tops rec-top-L1, which its level's RecCap holds; pass the verifier's shared rows (--rec-rows F), which register it"
#guard shared (.cap 1 128) #[port "top" "rec-top-L1" (reads := all)]
  (some (rows.push (port "rec-top-L1" "rec-top-L1" (reads := all)))) == none

def circuit : Circuit := { (default : Circuit) with nets := #[("rec-cap/n128", { (default : Net) with name := "rec-cap/n128" })] }

/-- A public file whose units header labels its instances `indices`. -/
def pubOf (indices : List Nat) : Public :=
  { (default : Public) with
    n := indices.length, header := Json.mkObj [("units", Json.mkObj [("indices", toJson indices)])] }

def cap (own : Array Registered.Port) (N : Nat := 128) (l : Nat := 0) (pub : Public := pubOf []) : Option String :=
  match RecCoins.checkCap circuit pub own l N with
  | .error e => some e
  | .ok () => none

#guard cap #[port "out" "rec-cap-zero" (reads := all)] ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port top: the cap's rows would be the prover's"
#guard cap #[port "top" "rec-top-L1" (reads := all)] ==
  some "rec-coins: rec-cap/n128: port top reads rec-top-L1, not level 0's tops rec-top-L0"
#guard cap #[port "top" "rec-top-L0" (reads := all)] (l := 2) ==
  some "rec-coins: rec-cap/n128: port top reads rec-top-L0, not level 2's tops rec-top-L2"
#guard cap #[port "top" "rec-top-L2" (reads := all)] (l := 2) ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port out: its rec-cap-zero would be the prover's"
#guard cap #[port "top" "rec-top-L0" (leaves := 64) (reads := all)] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: 64 rows of 64 words registered, the cap's 128 of 64"
#guard cap #[{ port "top" "rec-top-L0" (reads := all) with words := some 32 }] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: 128 rows of 32 words registered, the cap's 128 of 64"
-- Reads that follow a coin (node 3 twice, node 2 never) are refused: every node is held at its own row.
#guard cap #[port "top" "rec-top-L0" (reads := all.set! 2 (some 3))] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: the reads file does not read node i of the cap in instance i"
#guard cap #[port "top" "rec-top-L0" (reads := all.pop)] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: the reads file does not read node i of the cap in instance i"
-- Instance `i` reads node `i` only where the table labels its instances `0` to `n - 1` in order.
#guard cap #[port "top" "rec-top-L0" (reads := all)] (pub := pubOf [1, 0]) ==
  some ("rec-coins: rec-cap/n128: its instances are labelled #[1, 0], not 0 to 2 - 1 in order, so instance i would " ++
    "not read node i")
#guard cap #[port "top" "rec-top-L0" (reads := all)] (pub := pubOf [0, 1]) ==
  some ("rec-coins: rec-cap/n128: the verifier's registered reads hold no port out: its rec-cap-zero would be the " ++
    "prover's")
-- Past the reads, `out` must be V*'s zero rows `rec-cap-zero`.
#guard cap #[port "top" "rec-top-L0" (reads := all), port "out" "rec-residuals-zero" (reads := all)] ==
  some "rec-coins: rec-cap/n128: port out reads rec-residuals-zero, not the verifier's rec-cap-zero"
#guard cap #[port "top" "rec-top-L0" (reads := all)] ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port out: its rec-cap-zero would be the prover's"

def ok : Json := Json.mkObj [("ok", Json.arr #[])]

/-- Inner coins of no claims, at the empty record: `Source.load` passes, so `RecCoins.check` reaches the unit's name. -/
def coins : RecCoins.Source :=
  .coins (Json.mkObj [("format", Json.str RecCoins.COINS_FORMAT),
    ("record_sha512", Json.str (hexOf (Sha512.hash ByteArray.empty))), ("extras", ok),
    ("reps", Json.arr #[Json.mkObj [("rounds", ok)], Json.mkObj [("rounds", ok)]]),
    ("m", toJson (25 : Nat)), ("k_log", toJson (22 : Nat))]) ""

def named (name : String) : Option String :=
  let c : Circuit := { (default : Circuit) with nets := #[(name, { (default : Net) with name })] }
  match RecCoins.check { (default : Stmt) with c } (some #[]) (some coins) with
  | .error e => some e
  | .ok () => none

-- A `rec-cap/` unit is refused unless `capParams` reads its `(l, N)` back from its name.
#guard named "rec-cap/n0128" ==
  some "rec-coins: rec-cap/n0128: a rec-cap/ unit not named rec-cap/n<N> or rec-cap/n<N>-l<l>"
#guard named "rec-cap/n64-l0" ==
  some "rec-coins: rec-cap/n64-l0: a rec-cap/ unit not named rec-cap/n<N> or rec-cap/n<N>-l<l>"
#guard named "rec-cap/n128" ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port top: the cap's rows would be the prover's"
#guard named "rec-capx" == none

-- The unit: `out` is `top`'s words 32 … 63, then 32 zero words; no AND.
#guard (VStar.capUnit 0 128).st.rows.isEmpty
#guard (VStar.capUnit 0 128).inPorts == [64] && (VStar.capUnit 0 128).outPorts == [64]
#guard (VStar.capUnit 0 128).outs.size == 16 * 64
#guard ((VStar.capUnit 0 128).outs.toList.take (16 * 32)).map (·.s) == (List.range (16 * 32)).map fun v => [16 * 32 + v]
#guard ((VStar.capUnit 0 128).outs.toList.drop (16 * 32)).all fun f => f.s.isEmpty && !f.c
#guard (VStar.capUnit 2 128).name == "rec-cap/n128-l2"

def hex128 : String := "".pushn 'a' 128
def bindings : VStar.Bindings :=
  { partition := hex128, program := hex128, programDigests := Json.mkObj [("RecCap_v1", Json.str hex128)] }

#guard (circuitFile (.cap 0 128) bindings).toBool
#guard (circuitFile (.cap 2 128) bindings).toBool

end Flock.Test.RecCapSmall
