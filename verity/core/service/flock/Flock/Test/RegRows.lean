/-
`Registered.check` where a unit reads several rows of one registered value (H1's A: a tile's parameter `a` of R rows, its
circuit ports `a0 .. a{R-1}`, row `i` of the unit's rows read at port `a{i}`). Each port is its own `--registered` port
of the one value, with its own position per unit, and the check holds each separately: every table row opens the value's
root at its stated position, framed under the port's row format, and every instance reads, at each port, the row the
verifier's program reads for its unit at that port. A K = 64 A row is 64 int8 codes, a `hm96-sha512/row/v2` bit row of
512 bits: `words = K / 2 = 32` leaves of `word_bits = 16`, as `Port.fits` counts it.
-/
import Flock.Registered
import Flock.JsonKeys

namespace Flock.Test.RegRows

open Lean (Json)

def v2 := HmRow.ROW_SCHEMA_V2

def dom : ByteArray := Sha512.hash "verity/test/registered-domain/a".toUTF8

/-- Row `r`'s `b ‖ c` of value `tag` (`a`: 1, another value: 2). -/
def bc (tag r : Nat) : ByteArray := ByteArray.mk (Array.replicate 32 (UInt8.ofNat (16 * tag + r)))

/-- The four leaves of value `tag`'s rows, framed under `hm96-sha512/row/v2`. -/
def leaves (tag : Nat) : Array ByteArray :=
  (Array.range 4).map fun r => Registered.leafAs v2 dom r (Hm96.Default512.treeLeaf (bc tag r))

def root (tag : Nat) : ByteArray := FV3.root dom (leaves tag) Sha512.hash

/-- Row `pos`'s path in value `tag`'s tree of four leaves: its sibling leaf, then its pair's sibling node. -/
def path (tag pos : Nat) : ByteArray :=
  let ls := leaves tag
  let j := pos / 2 ^^^ 1
  ls[pos ^^^ 1]! ++ Registered.node dom 0 j ls[2 * j]! ls[2 * j + 1]!

/-- An A row of K = 64 int8 codes: a v2 bit row of 512 bits. -/
def aPort (name : String) : Port := { name, words := 32, chunks := 2, nb := 0, bits := 512, v2 := true }

def circuit : Circuit := { (default : Circuit) with ports := #[aPort "a0", aPort "a1"] }

/-- The statement: two units (0, 1), instance `i` unit `i`; port `a0`'s table the rows at `p0`, `a1`'s at `p1`, each of
value `tag` named `value`, instance `i` reading table row `i` of each. -/
def pub (p0 p1 : List Nat) (tag : Nat := 1) (value : String := "h1/a") : Public :=
  let entry (ps : List Nat) := Json.mkObj [("value", Json.str value), ("positions", Lean.toJson ps),
    ("paths", Json.arr (ps.map fun p => Json.str (hexOf (path tag p))).toArray)]
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [0, 1])]),
      ("registered", Json.mkObj [("a0", entry p0), ("a1", entry p1)])],
    n := 2, digests := ((p0 ++ p1).map (bc tag)).toArray, outputs := #[], sha := .empty,
    tables := some { counts := #[p0.length, p1.length], starts := #[0, p0.length], refs := #[0, 0, 1, 1] } }

/-- The verifier's `--registered` file: ports `a0` and `a1` both of `h1/a`, unit `u` reading rows `2u` and `2u + 1`. -/
def file (words : Nat := 32) : Json :=
  let port (reads : List Nat) := Json.mkObj [("value", Json.str "h1/a"), ("root", Json.str (hexOf (root 1))),
    ("leaves", Lean.toJson (4 : Nat)), ("words", Lean.toJson words), ("domain", Json.str (hexOf dom)),
    ("row_schema", Json.str v2), ("reads", Lean.toJson reads)]
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("a0", port [0, 2]), ("a1", port [1, 3])])]

def run (p : Public) (f : Json := file) : Except String Unit := do
  Registered.check circuit p (some (← Registered.ofJson f)) true

def passes (p : Public) (f : Json := file) : Bool := (run p f).toBool

def why (p : Public) (f : Json := file) : String :=
  match run p f with
  | .error e => e
  | .ok () => ""

-- the climb's own root is the tree's
#guard (List.range 4).all fun r => Registered.climb dom (leaves 1)[r]! r 2 (path 1 r) == root 1

-- each unit reads its two rows of A, one at each port, each opening A's root at its own position
#guard passes (pub [0, 2] [1, 3])

-- K = 64 codes are K / 2 = 32 u16 words: 64 words (a code a 16-bit leaf) is not the port's row
#guard !passes (pub [0, 2] [1, 3]) (file 64)

-- a wrong row index: the rows swapped between the ports, or a unit's rows swapped between units, each still opening the
-- root at its stated position, read where the verifier's program reads another row
#guard !passes (pub [1, 3] [0, 2])
#guard !passes (pub [2, 0] [1, 3])
#guard !passes (pub [0, 0] [1, 3])
#guard why (pub [1, 3] [0, 2]) ==
  "registered reads: port a0: unit 0 reads position 1, and the verifier's program reads 0"

-- a row of another value: its rows and paths open its own root, not A's; and a header naming the other value
#guard !passes (pub [0, 2] [1, 3] (tag := 2))
#guard why (pub [0, 2] [1, 3] (tag := 2)) ==
  "registered reads: port a0: table row 0 does not open h1/a's registered root at position 0"
#guard !passes (pub [0, 2] [1, 3] (value := "h1/b"))

/-! A port named twice: `Json.parse` keeps one of the two, so the file reads as naming it once (here as one port, `a1`'s
reads under `a0`); `flock-verify` refuses the text first (`JsonKeys.dupKey`). -/

/-- The file with port `a1` renamed `a0`: `a0` twice. -/
def twice : String := file.compress.replace "\"a1\":" "\"a0\":"

#guard ((Lean.Json.parse twice).toOption.bind fun j => (Registered.ofJson j).toOption).map (·.size) == some 1
#guard JsonKeys.dupKey twice == some "\"a0\""
#guard JsonKeys.dupKey file.compress == none
#guard JsonKeys.dupKey (file.pretty) == none
-- a key as `Json.parse` decodes it: an escape names the same key
#guard JsonKeys.dupKey "{\"a0\": 1, \"a\\u0030\": 2}" == some "\"a\\u0030\""
-- one key in two objects, keys inside strings, and values that are strings are no duplicate
#guard JsonKeys.dupKey "{\"a\": {\"x\": 1}, \"b\": {\"x\": 2}, \"c\": [{\"x\": 1}, {\"x\": 2}]}" == none
#guard JsonKeys.dupKey "{\"a\": \"a\", \"b\": \"}{,\\\"a\\\":\", \"c\": [\"a\", \"a\"]}" == none
#guard JsonKeys.dupKey "{\"r\": [{\"k\": 1, \"k\": 2}]}" == some "\"k\""
#guard JsonKeys.dupKey "{\"p\": {\"a\": {\"root\": \"00\", \"leaves\": 1, \"root\": \"01\"}}}" == some "\"root\""

end Flock.Test.RegRows
