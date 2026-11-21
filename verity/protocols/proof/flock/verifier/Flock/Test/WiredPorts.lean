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
#guard ((treeReads prog #[19] 0 1 x).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 0], 19)
-- the crossing cheat: the reader's `x` committed under a root of its own is refused by name
#guard match treeReads prog #[19] 0 1 { x with value := "r/x" } with
  | .error e => e.startsWith s!"registered reads: row x of in(S) is committed as r/x, and C wires it from {fieldTree (wkey 19) 0}"
  | .ok _ => false
-- a run input node's tree's name is the node's alone: a row of in(S) under it is refused by that name
#guard match treeReads prog #[19] 0 1 { x with value := freshValue 3 } with
  | .error e => e == RESERVED_WHY "row of in(S)" "x" "input/n3"
  | .ok _ => false
-- a row whose units read no writer is an ordinary input row, returned unread; one some of whose units do is refused
#guard (treeReads prog #[0] 0 1 x).toOption.map (·.readsRel.isEmpty) == some true
#guard (treeReads prog #[0, 19] 0 1 x).toOption.isNone
-- the whole-output form beside it refuses unit 13's read of a `TPair`, of two output fields
#guard match treeReadsWhole prog #[13] 0 { x with value := fieldTree (wkey 13) 0 } with
  | .error e => (e.splitOn "of 2 output fields").length == 2
  | .ok _ => false
-- unit 13 (the scan's first step) reads at leaf 0 unit 0's `TPair` field 0 whole: its row opens `TPair_v1/o0`, at unit
-- 0's leaf, the tree C gives
def y : Registered.Port := { x with name := "y", value := fieldTree (wkey 13) 0 }
#guard wkey 13 == "TPair_v1"
#guard ((treeReads prog #[13] 0 1 y).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 0], 13)
-- a reader naming another tree: the writer's other field's, a field past its two, or another template's
#guard match treeReads prog #[13] 0 1 { y with value := fieldTree (wkey 13) 1 } with
  | .error e => e.startsWith "registered reads: row y of in(S) is committed as TPair_v1/o1, and C wires it from TPair_v1/o0"
  | .ok _ => false
#guard match treeReads prog #[13] 0 1 { y with value := fieldTree (wkey 13) 2 } with
  | .error e => e.startsWith "registered reads: row y of in(S) is committed as TPair_v1/o2, and C wires it from TPair_v1/o0"
  | .ok _ => false
#guard match treeReads prog #[13] 0 1 x with
  | .error e => e.startsWith s!"registered reads: row x of in(S) is committed as {fieldTree (wkey 19) 0}, and C wires it from TPair_v1/o0"
  | .ok _ => false
-- a row past the writer's field: unit 13's leaves [0, 2) are TPair's field 0 and a circuit input
#guard match treeReads prog #[13] 0 2 y with
  | .error e => e == FIELD_WHY "y" 13 0 "TPair_v1" 2
  | .ok _ => false
-- `fieldRead` on TPair's outputs: each field whole, at C's positions; both fields, or a reorder, are no one field
def og0 : Array Nat := (HmLinks.outGates prog 3 0).toOption.getD #[]
#guard og0.size == 2
#guard HmLinks.fieldRead prog og0 (og0.extract 0 1) 3 == some 0
#guard HmLinks.fieldRead prog og0 (og0.extract 1 2) 3 == some 1
#guard HmLinks.fieldRead prog og0 og0 3 == none
#guard HmLinks.fieldRead prog og0 og0.reverse 3 == none
#guard HmLinks.fieldRead prog og0 #[] 3 == none
-- two readers of one field's leaves under two roots: one value, one commitment
#guard splitValue? [y, { y with name := "z", root := ByteArray.mk #[1] }] == some (fieldTree (wkey 13) 0)
#guard splitValue? [y, { y with name := "z" }] == none

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

-- theta: a registered weight of 4 rows (node 0) read by T's two units (node 1) at rows 1 and 2, and by unit 2 (node 2)
-- from node 1's output, no registered input
def r16 : Program.Ty := .value 16
def rw2 : Program.Ty := .array 2 r16
def regP : Program.Prog :=
  let regIn : Program.Defn :=
    { key := "RegisteredInput16_v1", id := "RegisteredInput16_v1", prim := true, params := #[], ret := rw2,
      binding := some Program.REGISTRATION, gates := 1, paramLeaves := 0, outLeaves := 2 }
  let t : Program.Defn :=
    { key := "T", id := "T", prim := true, params := #[("w", rw2)], ret := r16, gates := 1, paramLeaves := 2,
      outLeaves := 1 }
  let nodes : Array Program.Node := #[{ form := .batch, fn := 0, args := #[], out := .array 4 rw2, n := 4 },
    { form := .batch, fn := 1, out := .array 2 r16, n := 2,
      args := #[{ ty := .array 2 rw2, seq := Program.Seq.ofParts #[.affine false 0 2 4], dims := some (2, 2) }] },
    { form := .call, fn := 1, out := r16, args := #[{ ty := rw2, seq := Program.Seq.ofParts #[.affine false 1 0 2] }] }]
  let root : Program.Defn :=
    { key := "R", id := "R", prim := false, params := #[], ret := r16, gates := 0, paramLeaves := 0, outLeaves := 1,
      body := { params := #[], retType := r16, nodes := nodes, ret := default, offsets := #[0, 0, 2] } }
  { defs := #[regIn, t, root], index := {}, root := 2, programFn := "", sha512 := .empty }
def theta : Registered.Port := { x with name := "w", value := "theta", leaves := 4 }

#guard ((OneCircuit.pos regP #[0, 1] theta).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 1, some 2], 0)
#guard ((OneCircuit.pos regP #[1] theta).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 2], 1)
-- row 1 of w (`name[i]`), one leaf a row: the weight's 8 rows, leaves 3 and 5
#guard ((OneCircuit.pos regP #[0, 1] { theta with name := "w1", param := "w[1]", leaves := 8 }).toOption.map
  (·.readsRel)) == some #[some 3, some 5]
-- a value of another row count than C's node, a unit reading no registered input, a parameter T lacks
#guard (OneCircuit.pos regP #[0, 1] { theta with leaves := 3 }).toOption.isNone
#guard (OneCircuit.pos regP #[2] theta).toOption.isNone
#guard (OneCircuit.pos regP #[0] { theta with name := "v" }).toOption.isNone

-- a fresh input: node 0 a run input, its rows read at the positions C gives from its one tree `input/n0`, the row count
-- held as a registered value's; a run input named as another value, and a registered one named as a fresh input, refused
def whyPos (P : Program.Prog) (units : Array Nat) (reg : Registered.Port) : String :=
  match OneCircuit.pos P units reg with
  | .error e => e
  | .ok _ => ""
def runP : Program.Prog :=
  { regP with defs := regP.defs.set! 0 { regP.defs[0]! with key := "Input16_v1", id := "Input16_v1", binding := some Program.RUN } }
def fresh : Registered.Port := { theta with value := freshValue 0 }
#guard ((OneCircuit.pos runP #[0, 1] fresh).toOption.map fun r => (r.readsRel, r.readsFrom)) == some (#[some 1, some 2], 0)
#guard (OneCircuit.pos runP #[0, 1] { fresh with leaves := 3 }).toOption.isNone
#guard (whyPos runP #[0, 1] theta).startsWith
  "registered reads: port w: C's node 0 is a run input, its rows committed once, in input/n0, and the port names theta"
#guard (whyPos runP #[0, 1] { fresh with value := freshValue 1 }).startsWith "registered reads: port w: C's node 0 is a run input"
#guard (whyPos regP #[0, 1] fresh).startsWith
  "registered reads: port w: C's node 0 is a registered input, read at its registered root, and the port names the fresh input input/n0"
#guard (OneCircuit.pos runP #[2] fresh).toOption.isNone
-- a value is one commitment: two ports of one value under two roots, domains or row counts
#guard splitValue? [theta, { theta with name := "w1" }] == none
#guard splitValue? [theta, { theta with name := "w1", root := ByteArray.mk #[1] }] == some "theta"
#guard splitValue? [fresh, theta, { fresh with name := "w1", leaves := 8 }] == some (freshValue 0)
#guard splitValue? [theta, { fresh with root := ByteArray.mk #[1] }] == none

-- a writer of fields of 2 and 1 leaves: a row is one field whole; a stretch inside a field, or across two, is none
#guard Program.fieldSpan (.tuple #[.array 2 r16, r16]) 0 == some (0, 2)
#guard Program.fieldSpan (.tuple #[.array 2 r16, r16]) 2 == some (1, 1)
#guard Program.fieldSpan (.tuple #[.array 2 r16, r16]) 1 == none
#guard Program.fieldSpan r16 0 == some (0, 1)
def wideP : Program.Prog := { regP with defs := regP.defs.set! 1 { regP.defs[1]! with ret := .tuple #[.array 2 r16, r16] } }
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[10, 11] 1 == some 0
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[12] 1 == some 1
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[10] 1 == none
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[11] 1 == none
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[11, 12] 1 == none
#guard HmLinks.fieldRead wideP #[10, 11, 12] #[11, 10] 1 == none

-- output rows: a block circuit of `regP`'s template T, whose one output field (16 bits) is its one output row `o0`
def oMeta (t : String) : Json := Json.mkObj [("program_digests", Json.mkObj [(t, Json.str "d")])]
def oPort (name : String) (words : Nat) : Flock.Port := { name, words, chunks := 1, nb := 0 }
def outC : Circuit :=
  { (default : Circuit) with
    ports := #[oPort "w" 2, oPort "o0" 1], outPorts := #[("o0", 1)], outRowPorts := #[1], metaJ := oMeta "T" }
def o0 : Registered.Port := { x with name := "o0", value := fieldTree "T" 0, leaves := 3 }
def outWhy (c : Circuit) (P : Program.Prog) (j : Nat) (units : Array Nat) (reg : Registered.Port) : String :=
  match outReads c P units j reg with
  | .error e => e
  | .ok _ => ""
def outRel (c : Circuit) (P : Program.Prog) (j : Nat) (units : Array Nat) (reg : Registered.Port) :
    Option (Array (Option Nat) × Nat) :=
  (outReads c P units j reg).toOption.map fun r => (r.readsRel, r.readsFrom)

-- (b) each unit reads its own leaf of T's output tree, the leaf `HmLinks.treeLeaf` gives (T's rank in root order)
#guard outRel outC regP 0 #[0, 1] o0 == some (#[some 0, some 1], 0)
#guard outRel outC regP 0 #[2] o0 == some (#[some 2], 2)
#guard (List.range 3).all fun u =>
  ((outRel outC regP 0 #[u] o0).bind (·.1[0]?)).join == (HmLinks.treeLeaf regP (HmLinks.nodes regP) u).map (·.2)
-- (a) the row opens T's output field's tree, never a value of the caller's naming, nor another field's tree
#guard (outWhy outC regP 0 #[0] { o0 with value := "o0" }).startsWith
  "registered reads: output row o0 is registered as o0, and it is T's output field 0, committed once in T/o0"
#guard (outWhy outC regP 0 #[0] { o0 with value := fieldTree "T" 1 }).startsWith "registered reads: output row o0 is registered as T/o1"
#guard outWhy outC regP 0 #[0] { o0 with value := freshValue 0 } == RESERVED_WHY "output row" "o0" "input/n0"
#guard isFreshValue "input/n3" && isFreshValue "input/n0" && !isFreshValue "input/n03" && !isFreshValue "input/n"
  && !isFreshValue "input/x3" && !isFreshValue "committed/input/n3" && !isFreshValue "input/n3/x"
-- a row of another width than its field's whole words, a circuit naming no template, or one C's root never calls
#guard (outWhy { outC with ports := #[oPort "w" 2, oPort "o0" 2], outPorts := #[("o0", 2)] } regP 0 #[0] o0).startsWith
  "registered reads: output row o0 is output words [0, 2), and T's output field 0 is bits [0, 16)"
#guard (outWhy { outC with metaJ := Json.null } regP 0 #[0] o0).startsWith
  "registered reads: output row o0: the circuit names no template of C"
#guard (outWhy { outC with metaJ := oMeta "U" } regP 0 #[0] o0).startsWith
  "registered reads: output row o0: C's root calls no U"

-- T returning two fields: one output row a field, each opening its own tree at the unit's leaf; one row for both refused
def pairP : Program.Prog := { regP with defs := regP.defs.set! 1 { regP.defs[1]! with ret := .tuple #[r16, r16] } }
def outC2 : Circuit :=
  { outC with
    ports := #[oPort "w" 2, oPort "o0" 1, oPort "o1" 1], outPorts := #[("o0", 1), ("o1", 1)], outRowPorts := #[1, 1] }
def o1 : Registered.Port := { o0 with name := "o1", value := fieldTree "T" 1 }
#guard outRel outC2 pairP 0 #[1] o0 == some (#[some 1], 1)
#guard outRel outC2 pairP 1 #[1] o1 == some (#[some 1], 1)
#guard (outWhy outC2 pairP 1 #[1] { o1 with value := fieldTree "T" 0 }).startsWith "registered reads: output row o1 is registered as T/o0"
#guard (outWhy { outC with ports := #[oPort "w" 2, oPort "o0" 2], outPorts := #[("o0", 2)] } pairP 0 #[1] o0).startsWith
  "registered reads: output row o0: T returns 2 output fields and the circuit has 1 output rows"

-- `withPositions` sends a port past the declared input rows (`outRow?`, META `input_ports`) to `outReads`; a row past
-- the circuit's is refused
#guard HmOut.nIn outC == 1 && HmOut.nIn outC2 == 1
def hidMeta (k : Nat) : Json := (oMeta "T").setObjVal! "input_ports" (Json.num k)
#guard outRow? { outC with metaJ := hidMeta 1 } 0 == none && outRow? { outC with metaJ := hidMeta 1 } 1 == some 0
#guard outRow? { outC2 with metaJ := hidMeta 1 } 2 == some 1
-- a public-output circuit (`HmRow.parse`: no `input_ports`, one output row counted per output port, none among its
-- ports): its last input, at `HmOut.nIn`, is an input row, not output row 0
def pubC : Circuit :=
  { (default : Circuit) with
    ports := #[oPort "x" 64, oPort "cs" 64], outPorts := #[("out", 64)], outRowPorts := #[1], metaJ := oMeta "T" }
#guard HmOut.nIn pubC == 1 && outRow? pubC 1 == none && outRow? pubC 0 == none
-- and so with a forged `input_ports`: 0 (not the rows before one output row each) or 1 (`cs` is no output port's row)
#guard [0, 1, 2].all fun k => [0, 1].all fun q => outRow? { pubC with metaJ := hidMeta k } q == none
#guard (outWhy outC regP 1 #[0] o0).startsWith "registered reads: output row o0: the circuit has 1 output rows"

-- (d) the derived position held at the instances: an instance at a row of another leaf than its unit's is refused
def oRes : Registered.Port := ((outReads outC regP #[0, 1] 0 o0).toOption).getD default
def otab : Tables := { counts := #[2, 2], starts := #[0, 2], refs := #[0, 0, 1, 1] }
#guard badInstance oRes otab 2 1 2 2 #[0, 1] #[0, 1] == none
#guard badInstance oRes otab 2 1 2 2 #[1, 0] #[0, 1] == some 0
-- and the output row's table rows open only the registered root: the honest block's rows under a wrong root are refused
def oReg (root : ByteArray) : Registered.Port := { reg2 with name := "o0", value := fieldTree "T" 0, root }
def oRows (root : ByteArray) : Option Nat :=
  badRow (oReg root) { (default : Public) with n := 2, digests := #[bc 0, bc 1] } tab2 0 2 #[0, 1] paths2
#guard oRows reg2.root == none
#guard oRows (node dom 0 0 (lf 0 (bc 2)) (lf 1 (bc 1))) == some 0

end Flock.Test.WiredPorts
