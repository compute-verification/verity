/-
`RegWired.check`: a statement's wired port `x`, its two instances reading writer `m/c0/o0`'s output rows 0 and 1, whose
`b ‖ c` they share (each wired row committed under its writer row's salt). The verifier's reads give each session
instance's writer root and position; the header gives one path per row of the population. Every instance's row opens its
writer's root, with or without row tables, and in a drawn session only the drawn instances' rows are read, a drawn
instance whose row is not its writer's refused; a swapped
position, another writer's root, a wrong path, a path or read short, a missing entry, a header port the verifier doesn't
read, a registered port, a `row/v1` port or a port of no circuit is refused; without the verifier's reads the check is a
no-op.
-/
import Flock.RegWired

namespace Flock.Test.RegWired

open Lean (Json)

def dom : ByteArray := Sha512.hash "verity/test/wired-domain".toUTF8
def v2 := HmRow.ROW_SCHEMA_V2

def bcs : Array ByteArray := #[ByteArray.mk (Array.replicate 32 1), ByteArray.mk (Array.replicate 32 2)]
def others : Array ByteArray := #[ByteArray.mk (Array.replicate 32 3), ByteArray.mk (Array.replicate 32 4)]

def leaves (xs : Array ByteArray) : Array ByteArray :=
  xs.mapIdx fun i bc => Registered.leafAs v2 dom i (Hm96.Default512.treeLeaf bc)

/-- The writer's root over its two output rows, as the window's registration holds it. -/
def root (xs : Array ByteArray := bcs) : ByteArray := FV3.root dom (leaves xs) Sha512.hash

def writer (name : String := "m/c0/o0") (r : ByteArray := root) : RegWired.Writer :=
  { name, root := r, domain := dom, leaves := 2 }

def xPort : Port := { name := "x", words := 32, chunks := 2, nb := 0, bits := 512, v2 := true }
def circuit (p : Port := xPort) : Circuit := { (default : Circuit) with ports := #[p] }

def paths : List Json := [Json.str (hexOf (leaves bcs)[1]!), Json.str (hexOf (leaves bcs)[0]!)]

/-- A public file of two instances of the one port `x`, its header's `wired` the given entries; with `tables`, a
shared-row file whose instance `i` reads table row `i`. -/
def pub (wired : List (String × Json) := [("x", Json.arr paths.toArray)]) (tables : Bool := false) : Public :=
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [0, 1])]), (RegWired.HEADER_KEY, Json.mkObj wired)],
    n := 2, digests := bcs, outputs := #[], sha := .empty,
    tables := if tables then some { counts := #[2], starts := #[0], refs := #[0, 1] } else none }

def drawJ : Json :=
  Json.mkObj [("law", "subset"), ("population", Lean.toJson (2 : Nat)), ("k", Lean.toJson (1 : Nat)),
    ("units", Lean.toJson [1])]

/-- The drawn session of unit 1 alone: one instance, its row the population's row 1 (by its ref, with tables). -/
def drawnPub (tables : Bool := false) : Public :=
  let p := pub (tables := tables)
  { p with header := p.header.setObjVal! "unit_draw" drawJ, n := 1,
           digests := if tables then bcs else #[bcs[1]!], population := some (2, bcs, #[]),
           tables := if tables then some { counts := #[2], starts := #[0], refs := #[1] } else none }

def reads (ps : List Nat := [0, 1]) (ws : List RegWired.Writer := [writer, writer]) : Array RegWired.Read :=
  (ws.zip ps).toArray.map fun (writer, pos) => { writer, pos }

def ports (rs : Array RegWired.Read := reads) (name : String := "x") : Array RegWired.Port := #[{ name, reads := rs }]

def why (ps : Array RegWired.Port := ports) (p : Public := pub) (c : Circuit := circuit)
    (registered : Option (Array Registered.Port) := none) : String :=
  match RegWired.check c p registered (some ps) with
  | .error e => e
  | .ok () => ""

-- each instance's row opens its writer's root at its position, with or without row tables
#guard why == ""
#guard why (p := pub (tables := true)) == ""

-- a drawn session reads only its drawn instance, at the population row it reads
#guard why (ports (reads [1] [writer])) (drawnPub) == ""
#guard why (ports (reads [1] [writer])) (drawnPub (tables := true)) == ""
#guard why (ports (reads [0] [writer])) (drawnPub) ==
  "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 0"

-- a drawn instance whose row is not its writer's row
#guard why (ports (reads [1] [writer])) ({ drawnPub with digests := #[others[1]!] }) ==
  "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 1"
#guard why (ports (reads [1] [writer])) ({ drawnPub (tables := true) with digests := #[bcs[0]!, others[1]!] }) ==
  "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 1"

-- the positions swapped: each row opens the root only at its own
#guard why (ports (reads [1, 0])) == "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 1"

-- another writer's root, which the rows' `b ‖ c` are not leaves of
#guard why (ports (reads [0, 1] [writer (r := root others), writer])) ==
  "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 0"
#guard why (ports (reads [0, 1] [writer, writer "m/c1/o0" (root others)])) ==
  "wired reads: port x: instance 1's row does not open m/c1/o0's root at position 1"

-- a wrong path, a path short, a read short
#guard why (p := pub [("x", Json.arr #[paths[1]!, paths[1]!])]) ==
  "wired reads: port x: instance 0's row does not open m/c0/o0's root at position 0"
#guard why (p := pub [("x", Json.arr #[paths[0]!])]) == "wired reads: port x: one path per row of the population (2)"
#guard why (ports (reads [0] [writer])) == "wired reads: port x: the verifier reads 1 rows, one per session instance (2)"

-- no entry for the port, a header port the verifier doesn't read, a registered port, a `row/v1` port, no such port
#guard why (p := pub []) ==
  "wired reads: the statement states no path for port x's rows: a wired row opens its writer's root"
#guard why (p := pub [("x", Json.arr paths.toArray), ("y", Json.arr paths.toArray)]) ==
  "wired reads: the statement names port y wired, and the verifier's wired reads do not"
#guard why (registered := some #[(writer "x").asPort v2]) ==
  "wired reads: port x is a registered port and wired: each port is one or the other"
#guard why (c := circuit { name := "x", words := 64, chunks := 2, nb := 0 }) ==
  "wired reads: port x is a hm96-sha512/row/v1 row, and a wired row is a bit row, as its writer's output row is"
#guard why (c := circuit { xPort with name := "z" }) == "wired reads: x is no port of the circuit"

-- without the verifier's wired reads the check is a no-op, whatever the header states
#guard (RegWired.check circuit (pub [("x", Json.arr #[])]) none none).toBool
#guard (RegWired.check circuit (pub []) none none).toBool

#print axioms RegWired.portOk_at
#print axioms RegWired.check_at

end Flock.Test.RegWired
