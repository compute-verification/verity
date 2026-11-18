module

public import Circuits.Wire
public import Flock.Registered
public import Flock.Test.RegProgram
meta import Circuits.Wire
meta import Flock.Registered
meta import Flock.Test.RegProgram
meta import Lean.Data.Json

/-!
Rows of in(S): a reader's row read by path in its writer template's output field's tree. On `Flock.Test.RegProgram`'s
program a unit's leaf in its template's trees (`HmLinks.treeLeaf`) is its rank among the template's units in root order,
counted from the nodes, and a unit's writer (`HmLinks.writerOf`) is the unit whose output it reads. A port of the
registered file states neither reads nor a tree (`PORT_KEYS`): `treeReads` derives a row's reads from the program, and
refuses a row committed under a root of its own, or another tree's (`fieldTree`), so a reader's input that is not its
writer's output does not verify. The honest reader's positions pass `badInstance` and a reader that refers to another
writer's row does not. On a registered tree of two real leaves, a reader's table rows open it only as committed: a row
with one bit flipped (the toy's `rb`), the writer's value under a fresh salt (`mc`) and the rows' paths swapped (`rd`)
are refused by `badRow`. A wire strided on the reader's side too (`Circuits.Wire.leafAt` with reader steps) gives each
reader unit the leaf of the one index reading it, and refuses a unit read twice, a cut, a row past 0 and steps that are
not one a digit.
-/

@[expose] public section

namespace Flock.Test.WiredPorts

open Lean (Json toJson)
open Flock Flock.Registered
open Flock.Test.RegProgram (plain)
open Flock.HmLinks (NodeUnits unitAt count)

def prog : Program.Prog := (Program.decode plain).toOption.getD default
def ns : List NodeUnits := HmLinks.nodes prog
def fnOf (v : Nat) : Option Nat := (unitAt ns v).map fun (k, _) => ((prog.defs[prog.root]!).body.nodes[k]!).fn

-- a unit's leaf is the number of earlier units of its template, its template its node's callee
#guard (List.range (count ns)).all fun v =>
  (HmLinks.treeLeaf prog ns v).map (·.2) == some (((List.range v).filter (fnOf · == fnOf v)).length)
#guard (List.range (count ns)).all fun v =>
  (HmLinks.treeLeaf prog ns v).map (·.1) == (fnOf v).map fun fn => (prog.defs[fn]!).key
#guard HmLinks.treeLeaf prog ns (count ns) == none
-- a unit that reads only circuit inputs has no writer; unit 13 reads another unit's output
#guard (List.range 64).all fun t => ((HmLinks.writerOf prog ns 0 t).toOption.join).isNone
#guard (List.range 64).any fun t => ((HmLinks.writerOf prog ns 13 t).toOption.join).isSome

def hex64 : Json := Json.str (String.ofList (List.replicate 128 'a'))

def portJ (value : String) (extra : List (String × Json) := []) : Json :=
  Json.mkObj ([("value", Json.str value), ("root", hex64), ("leaves", toJson 16), ("domain", hex64)] ++ extra)

def fileJ (ports : List (String × Json)) : Json := Json.mkObj [("format", Json.str FORMAT), ("ports", Json.mkObj ports)]

def refused (j : Json) : Bool := (ofJson j).toOption.isNone

-- the reader statement's units [8, 16), for `badInstance`
def units : Array Nat := (Array.range 8).map (· + 8)
/-- The template of the writer of unit `u`'s leaf 0, by C. -/
def wkey (u : Nat) : String :=
  (((HmLinks.writerOf prog ns u 0).toOption.join.bind (HmLinks.treeLeaf prog ns ·)).map (·.1)).getD ""

-- unit 19 (a `TSum4` of the root) reads at leaf 0 unit 5's output, a `TCol` of one output field
def x : Registered.Port := match ofJson (fileJ [("x", portJ (fieldTree (wkey 19) 0))]) with
  | .ok ps => ps.getD 0 default
  | .error _ => default

-- a port states no reads and names no tree: C gives both
#guard x.readsRel.isEmpty && x.value == fieldTree (wkey 19) 0 && wkey 19 != ""
#guard refused (fileJ [("x", portJ (fieldTree (wkey 19) 0) [("reads", toJson [0])])])
#guard refused (fileJ [("x", portJ (fieldTree (wkey 19) 0)
  [("tree", Json.mkObj [("template", Json.str (wkey 19)), ("field", Json.str "o0")])])])

-- every port's reads are derived from the verifier's program: without it every port is refused
#guard (withPositions default default none (some #[x])).toOption.isNone
#guard ((withPositions default default none none).toOption.map (·.isNone)) == some true

-- unit 19 reads leaf 0 of its writer's tree, derived from C
#guard ((treeReads prog #[19] 0 x).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 0], 19)
-- the crossing cheat: the reader's `x` committed under a root of its own is refused by name
#guard match treeReads prog #[19] 0 { x with value := "r/x" } with
  | .error e => e.startsWith s!"registered reads: row x of in(S) is committed as r/x, and C wires it from {fieldTree (wkey 19) 0}"
  | .ok _ => false
-- a row whose units read no writer is an ordinary input row, returned unread; one some of whose units do is refused
#guard (treeReads prog #[0] 0 x).toOption.map (·.readsRel.isEmpty) == some true
#guard (treeReads prog #[0, 19] 0 x).toOption.isNone
-- unit 13 reads a `TPair`, of two output fields: a row reads its writer's whole output, one field's tree
#guard match treeReads prog #[13] 0 x with
  | .error e => (e.splitOn "of 2 output fields").length == 2
  | .ok _ => false

def resolved : Registered.Port := { x with readsRel := (Array.range 8).map fun i => some (7 - i), readsFrom := 8 }
def tabs : Tables := { counts := #[8, 8], starts := #[0, 8], refs := (Array.range 8).flatMap fun i => #[i, i] }
#guard badInstance resolved tabs 2 0 8 8 ((Array.range 8).map (7 - ·)) units == none
#guard badInstance resolved tabs 2 0 8 8 (#[6] ++ (Array.range 7).map (6 - ·)) units == some 0

-- in(S)'s layout needs the block circuit's template in C
def inP : Flock.Port := { name := "x", words := 4, chunks := 1, nb := 0 }
def outP : Flock.Port := { name := "out", words := 4, chunks := 1, nb := 0 }
def block : Circuit := { (default : Circuit) with ports := #[inP, outP], outRowPorts := #[1] }
#guard match OneCircuit.io block prog with
  | .error e => e.startsWith "one-circuit: not derived yet: io : a block circuit that names no template"
  | .ok _ => false

def dom : ByteArray := ByteArray.mk (Array.replicate 64 7)
def val (k : UInt8) : ByteArray := ByteArray.mk (Array.replicate 32 k)
def salt (k : UInt8) : ByteArray := ByteArray.mk (Array.replicate 32 (k + 100))
def bc (k : UInt8) : ByteArray := Hm96.Default512.commitString (val k) (salt k)
def lf (pos : Nat) (row : ByteArray) : ByteArray := leafAs HmRow.ROW_SCHEMA dom pos (Hm96.Default512.treeLeaf row)
def reg2 : Registered.Port := { x with root := node dom 0 0 (lf 0 (bc 0)) (lf 1 (bc 1)), leaves := 2, domain := dom }
def tab2 : Tables := { counts := #[2], starts := #[0], refs := #[0, 1] }
def paths2 : Array ByteArray := #[lf 1 (bc 1), lf 0 (bc 0)]
def rows2 (rows : Array ByteArray) (paths : Array ByteArray := paths2) : Option Nat :=
  badRow reg2 { (default : Public) with n := 2, digests := rows } tab2 0 2 #[0, 1] paths
def flip (b : ByteArray) : ByteArray := b.set! 0 (b.get! 0 ^^^ 1)

#guard rows2 #[bc 0, bc 1] == none
-- rb: one bit of a row flipped
#guard rows2 #[bc 0, flip (bc 1)] == some 1
-- mc: the writer's value committed again under a fresh salt
#guard rows2 #[Hm96.Default512.commitString (val 0) (salt 9), bc 1] == some 0
-- rd: the first two paths swapped
#guard rows2 #[bc 0, bc 1] #[paths2[1]!, paths2[0]!] == some 0

-- index i (digits d0 d1) is reader unit d0 + 2 d1 reading leaf 3 + 4 d0 - 3 d1: units 0 1 2 3 read leaves 3 7 0 4
def sw : Circuits.Wire :=
  { reader := "r", port := "x", writer := "w", out := "o", start := 0, radix := [2, 2],
    base := 3, step := [4, -3], cut0 := 0, cuts := 1, rstep := [1, 2] }
#guard (List.range 4).map (sw.leafAt · 0) == [some 3, some 7, some 0, some 4]
#guard (List.range 4).map (fun u => { sw with start := 5 }.leafAt (u + 5) 0) == [some 3, some 7, some 0, some 4]
#guard sw.leafAt 4 0 == none && sw.leafAt 0 1 == none
#guard (List.range 4).map ({ sw with rstep := [] }.leafAt · 0) == [some 3, some 0, some 7, some 4]
#guard { sw with rstep := [0, 1] }.leafAt 0 0 == none
#guard { sw with rstep := [1] }.leafAt 0 0 == none
#guard { sw with cuts := 2 }.leafAt 0 0 == none
#guard { sw with base := -1 }.leafAt 2 0 == none

end Flock.Test.WiredPorts
