/-
`RegWired.check`: a statement's wired port `x`, its two instances reading writer `m/c0/o0`'s output rows 0 and 1, whose
`b ‖ c` they share (each wired row committed under its writer row's salt). The verifier's `--wired` file holds the writer's
root; the header states each table row's writer, position and path. Every table row opens its writer's root, with or
without row tables; a swapped position, another writer's root, a writer the file doesn't hold, a wrong path, a missing
entry, a header port the file doesn't hold, a `row/v1` port or a malformed file is refused; without the file the check is
a no-op.
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

def xPort : Port := { name := "x", words := 32, chunks := 2, nb := 0, bits := 512, v2 := true }
def circuit (p : Port := xPort) : Circuit := { (default : Circuit) with ports := #[p] }

def paths : List Json := [Json.str (hexOf (leaves bcs)[1]!), Json.str (hexOf (leaves bcs)[0]!)]

def entryJ (writers : List String := ["m/c0/o0", "m/c0/o0"]) (positions : List Nat := [0, 1])
    (ps : List Json := paths) : Json :=
  Json.mkObj [("writers", Lean.toJson writers), ("positions", Lean.toJson positions), ("paths", Json.arr ps.toArray)]

/-- A public file of two instances of the one port `x`, its header's `wired` the given entries; with `tables`, a
shared-row file whose instance `i` reads table row `i`. -/
def pub (wired : List (String × Json) := [("x", entryJ)]) (tables : Bool := false) : Public :=
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [0, 1])]), (RegWired.HEADER_KEY, Json.mkObj wired)],
    n := 2, digests := bcs, outputs := #[], sha := .empty,
    tables := if tables then some { counts := #[2], starts := #[0], refs := #[0, 1] } else none }

def writerJ (r : ByteArray) : Json :=
  Json.mkObj [("root", Json.str (hexOf r)), ("domain", Json.str (hexOf dom)), ("leaves", Lean.toJson (2 : Nat))]

/-- The verifier's `--wired` file: port `x`, the writers `ws`. -/
def file (ws : List (String × Json) := [("m/c0/o0", writerJ root)]) (top : List (String × Json) := []) : Json :=
  Json.mkObj ([("format", Json.str RegWired.FORMAT),
    ("ports", Json.mkObj [("x", Json.mkObj [("writers", Json.mkObj ws)])])] ++ top)

def run (f : Json := file) (p : Public := pub) (c : Circuit := circuit) : Except String Unit := do
  RegWired.check c p (some (← RegWired.ofJson f))

def why (f : Json := file) (p : Public := pub) (c : Circuit := circuit) : String :=
  match run f p c with
  | .error e => e
  | .ok () => ""

-- each instance's row opens the writer's root at its position, with or without row tables
#guard why == ""
#guard why (p := pub (tables := true)) == ""
#guard (List.range 2).all fun i => [false, true].all fun t =>
  decide ((pub (tables := t)).digest i 0 1 = RegWired.row (pub (tables := t)) 1 0 (RegWired.refOf (pub (tables := t)) 1 0 i))

-- the positions swapped: each row opens the root only at its own
#guard why (p := pub [("x", entryJ (positions := [1, 0]))]) ==
  "wired reads: port x: table row 0 does not open m/c0/o0's root at position 1"

-- another writer's root, which the rows' `b ‖ c` are not leaves of
#guard why (file [("m/c0/o0", writerJ (root others))]) ==
  "wired reads: port x: table row 0 does not open m/c0/o0's root at position 0"
#guard why (file [("m/c0/o0", writerJ root), ("m/c1/o0", writerJ (root others))])
    (pub [("x", entryJ (writers := ["m/c0/o0", "m/c1/o0"]))]) ==
  "wired reads: port x: table row 1 does not open m/c1/o0's root at position 1"

-- a writer the verifier's file holds no root for
#guard why (p := pub [("x", entryJ (writers := ["m/c0/o0", "m/c9/o0"]))]) ==
  "wired reads: port x: table row 1 names writer m/c9/o0, whose root the verifier's wired reads do not hold"

-- a wrong path, and one entry short
#guard why (p := pub [("x", entryJ (ps := [paths[1]!, paths[1]!]))]) ==
  "wired reads: port x: table row 0 does not open m/c0/o0's root at position 0"
#guard why (p := pub [("x", entryJ (positions := [0]))]) ==
  "wired reads: port x: one writer, position and path per table row (2)"

-- no entry for the port, a header port the file doesn't hold, a `row/v1` port
#guard why (p := pub []) ==
  "wired reads: the statement states no writer for port x's rows: a wired row opens its writer's root"
#guard why (p := pub [("x", entryJ), ("y", entryJ)]) ==
  "wired reads: the statement names port y wired, and the verifier's wired reads do not"
#guard why (c := circuit { name := "x", words := 64, chunks := 2, nb := 0 }) ==
  "wired reads: port x is a hm96-sha512/row/v1 row, and a wired row is a bit row, as its writer's output row is"
#guard why (c := circuit { xPort with name := "z" }) == "wired reads: x is no port of the circuit"

-- without the verifier's wired reads the check is a no-op, whatever the header states
#guard (RegWired.check circuit (pub [("x", entryJ (positions := [1, 0]))]) none).toBool
#guard (RegWired.check circuit (pub []) none).toBool

-- the file: its format, its keys, a writer of a 64-byte root and domain and at least one leaf
#guard !(RegWired.ofJson (file (top := [("window", Json.null)]))).toBool
#guard !(RegWired.ofJson (Json.mkObj [("format", Json.str "verity/wired-reads/v0"), ("ports", Json.mkObj [])])).toBool
#guard !(RegWired.ofJson (file [])).toBool
#guard !(RegWired.ofJson (file [("m/c0/o0", writerJ ((root).extract 0 32))])).toBool
#guard !(RegWired.ofJson (file [("m/c0/o0", Json.mkObj [("root", Json.str (hexOf root)),
  ("domain", Json.str (hexOf dom)), ("leaves", Lean.toJson (0 : Nat))])])).toBool
#guard ((RegWired.ofJson file).toOption.map fun ps => ps.map fun p => (p.name, p.writers.map (·.name))) ==
  some #[("x", #["m/c0/o0"])]

end Flock.Test.RegWired
