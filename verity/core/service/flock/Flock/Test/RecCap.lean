/-
RecCap, V*'s statement that holds the table's level-0 cap at fixed rows (`rec-cap/n<N>`, `VStar.capUnit`):
`RecVStar.checkCapListed` refuses a V* list without it, at another node count or twice; `RecVStar.checkShared` refuses a
port reading `rec-top-L0` (level 0's `out`, RecCap's `top`) without the verifier's shared rows registering it, or at
another registration; `RecCoins.checkCap` refuses reads of the cap other than node `i` in instance `i` of `2^cap₀` nodes of
64 words. The unit is the cap row's words past its node, then zeros, and lays out as V*'s file.
-/
import Flock.RecCoins

namespace Flock.Test.RecCap

open Lean Flock Flock.RecVStar

#guard VStar.capName 128 == "rec-cap/n128"
#guard VStar.capParams "rec-cap/n128" == some 128
#guard VStar.capParams "rec-cap/n0128" == none
#guard VStar.capParams "rec-cap/n" == none
#guard VStar.capParams "rec-open/l16-h4" == none
#guard isVStar "rec-cap/n128"

-- `2^cap₀` at the fixture shapes: 128 nodes of `root_B` at m 22 through 35.
#guard (List.range 14).all fun d => ((fast100 (22 + d)).bind (·.levels[0]?)).map (2 ^ ·.cap) == some 128
#guard match VStar.capStatement { m := 25, kLog := 22, claims := 8 } with
  | .ok (.cap 128) => true
  | _ => false

def inner : Inner := { m := 25, kLog := 22, claims := 8, regions := 3 }

def listed (ss : Array VStar.Stmt) : Option String :=
  match checkCapListed inner ss with
  | .error e => some e
  | .ok () => none

def level0 : VStar.Stmt := .recOpen 0 64 8 436

#guard listed #[level0, .cap 128] == none
#guard listed #[level0] ==
  some "V*'s list has no RecCap statement (rec-cap/n128): the table's level-0 cap is held by none of its statements"
#guard listed #[level0, .cap 64] ==
  some "V*'s list: rec-cap/n64 holds 64 nodes, and the table's level-0 cap has 128 (2^cap₀, cap₀ 7)"
#guard listed #[.cap 128, level0, .cap 128] ==
  some "V*'s list has 2 RecCap statements, and holds the table's level-0 cap once"

def port (name value : String) (root : UInt8 := 1) (leaves : Nat := 128) (reads : Array (Option Nat) := #[some 0]) :
    Registered.Port :=
  { name, value, root := ⟨#[root]⟩, leaves, words := some 64, wordBits := 16, domain := ⟨#[7]⟩, reads }

def all : Array (Option Nat) := (Array.range 128).map some

def rows : Array Registered.Port := #[port "acc0" "rec-acc", port "rec-top-L0" "rec-top-L0" (reads := all)]

def shared (s : VStar.Stmt) (registered : Array Registered.Port) (rows : Option (Array Registered.Port)) : Option String :=
  match checkShared s registered rows with
  | .error e => some e
  | .ok () => none

-- Level 0's `out` and RecCap's `top` pass at the shared rows' one registration of `rec-top-L0`, each at its own reads.
#guard shared level0 #[port "out" "rec-top-L0" (reads := #[some 5, some 9])] (some rows) == none
#guard shared (.cap 128) #[port "top" "rec-top-L0" (reads := all)] (some rows) == none
#guard shared (.cap 128) #[port "top" "rec-top-L0" (root := 2) (reads := all)] (some rows) ==
  some "rec-cap/n128: its port top reads rec-top-L0, which the verifier's shared rows (--rec-rows) register as rec-top-L0: root 02, against 01"
#guard shared level0 #[port "out" "rec-top-L0" (leaves := 64)] (some rows) ==
  some "rec-open/l64-h8: its port out reads rec-top-L0, which the verifier's shared rows (--rec-rows) register as rec-top-L0: leaves 64, against 128"
#guard shared level0 #[port "out" "rec-top-L0"] (some #[port "acc0" "rec-acc"]) ==
  some "rec-open/l64-h8: its port out reads the table's level-0 cap rec-top-L0, which RecCap holds; pass the verifier's shared rows (--rec-rows F), which register it"
#guard shared (.cap 128) #[port "top" "rec-top-L0" (reads := all)] none ==
  some "rec-cap/n128: its port top reads the table's level-0 cap rec-top-L0, which RecCap holds; pass the verifier's shared rows (--rec-rows F), which register it"
-- A later level's tops need no shared entry: only level 0's cap is RecCap's.
#guard shared (.recOpen 1 16 7 212) #[port "out" "rec-top-L1"] (some #[port "acc0" "rec-acc"]) == none

def circuit : Circuit := { (default : Circuit) with nets := #[("rec-cap/n128", { (default : Net) with name := "rec-cap/n128" })] }

def cap (own : Array Registered.Port) (N : Nat := 128) : Option String :=
  match RecCoins.checkCap circuit default own N with
  | .error e => some e
  | .ok () => none

#guard cap #[port "out" "rec-cap-zero" (reads := all)] ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port top: the cap's rows would be the prover's"
#guard cap #[port "top" "rec-top-L1" (reads := all)] ==
  some "rec-coins: rec-cap/n128: port top reads rec-top-L1, not the table's level-0 cap rec-top-L0"
#guard cap #[port "top" "rec-top-L0" (leaves := 64) (reads := all)] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: 64 rows of 64 words registered, the cap's 128 of 64"
#guard cap #[{ port "top" "rec-top-L0" (reads := all) with words := some 32 }] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: 128 rows of 32 words registered, the cap's 128 of 64"
-- Reads that follow a coin (node 3 twice, node 2 never) are refused: every node is held at its own row.
#guard cap #[port "top" "rec-top-L0" (reads := all.set! 2 (some 3))] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: the reads file does not read node i of the cap in instance i"
#guard cap #[port "top" "rec-top-L0" (reads := all.pop)] ==
  some "rec-coins: rec-cap/n128: rec-top-L0: the reads file does not read node i of the cap in instance i"
-- Past the reads, `out` must be V*'s zero rows `rec-cap-zero`.
#guard cap #[port "top" "rec-top-L0" (reads := all), port "out" "rec-residuals-zero" (reads := all)] ==
  some "rec-coins: rec-cap/n128: port out reads rec-residuals-zero, not the verifier's rec-cap-zero"
#guard cap #[port "top" "rec-top-L0" (reads := all)] ==
  some "rec-coins: rec-cap/n128: the verifier's registered reads hold no port out: its rec-cap-zero would be the prover's"

-- The unit: `out` is `top`'s words 32 … 63, then 32 zero words; no AND.
#guard (VStar.capUnit 128).st.rows.isEmpty
#guard (VStar.capUnit 128).inPorts == [64] && (VStar.capUnit 128).outPorts == [64]
#guard (VStar.capUnit 128).outs.size == 16 * 64
#guard ((VStar.capUnit 128).outs.toList.take (16 * 32)).map (·.s) == (List.range (16 * 32)).map fun v => [16 * 32 + v]
#guard ((VStar.capUnit 128).outs.toList.drop (16 * 32)).all fun f => f.s.isEmpty && !f.c

def hex128 : String := "".pushn 'a' 128
def bindings : VStar.Bindings :=
  { partition := hex128, program := hex128, programDigests := Json.mkObj [("RecCap_v1", Json.str hex128)] }

#guard (circuitFile (.cap 128) bindings).toBool

end Flock.Test.RecCap
