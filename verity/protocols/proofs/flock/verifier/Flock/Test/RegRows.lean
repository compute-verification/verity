/-
`Registered.check` where a unit reads several rows of one registered value (H1's A: a tile's parameter `a` of R rows, its
circuit ports `a0 .. a{R-1}`, row `i` of the unit's rows read at port `a{i}`). Each port is its own `--registered` port
of the one value, with its own position per unit, and the check holds each separately: every table row opens the value's
root at its stated position, framed under the port's row format, and every instance reads, at each port, the row the
verifier's program reads for its unit at that port. A K = 64 A row is 64 int8 codes, a `hm96-sha512/row/v2` bit row of
512 bits: `words = K / 2 = 32` leaves of `word_bits = 16`, as `Port.fits` counts it.
-/
module

public import Flock.Registered
public import Flock.JsonKeys
public import Flock.JsonParse
meta import Flock.Registered
meta import Flock.JsonKeys
meta import Flock.JsonParse

@[expose] public section

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

/-! A port named twice: `JsonParse.parse` keeps one of the two, so the file reads as naming it once (here as one port, `a1`'s
reads under `a0`); `flock-verify` refuses the text first (`JsonKeys.dupKey`). -/

/-- The file with port `a1` renamed `a0`: `a0` twice. -/
def twice : String := file.compress.replace "\"a1\":" "\"a0\":"

#guard ((JsonParse.parse twice).toOption.bind fun j => (Registered.ofJson j).toOption).map (·.size) == some 1
#guard JsonKeys.dupKey twice == some "\"a0\""
#guard JsonKeys.dupKey file.compress == none
#guard JsonKeys.dupKey (file.pretty) == none
-- a key as `JsonParse.parse` decodes it: an escape names the same key
#guard JsonKeys.dupKey "{\"a0\": 1, \"a\\u0030\": 2}" == some "\"a\\u0030\""
-- one key in two objects, keys inside strings, and values that are strings are no duplicate
#guard JsonKeys.dupKey "{\"a\": {\"x\": 1}, \"b\": {\"x\": 2}, \"c\": [{\"x\": 1}, {\"x\": 2}]}" == none
#guard JsonKeys.dupKey "{\"a\": \"a\", \"b\": \"}{,\\\"a\\\":\", \"c\": [\"a\", \"a\"]}" == none
#guard JsonKeys.dupKey "{\"r\": [{\"k\": 1, \"k\": 2}]}" == some "\"k\""
#guard JsonKeys.dupKey "{\"p\": {\"a\": {\"root\": \"00\", \"leaves\": 1, \"root\": \"01\"}}}" == some "\"root\""

/-! A name a later port repeats is one `Registered.ofJson` refuses (`dupName?`), as `Json`'s type admits it though no
parse builds it. -/

def own : Array Registered.Port := (Registered.ofJson file).toOption.getD #[]

#guard own.size == 2 && Registered.dupName? own.toList == none
#guard Registered.dupName? (own.toList ++ [own[0]!]) == some "a0"

/-- `file` with its ports object holding `a0`'s entry twice, a tree no insertion builds. -/
def twicePorts : Json :=
  let e := (file.getObjValD "ports").getObjValD "a0"
  Json.mkObj [("format", Json.str Registered.FORMAT),
    ("ports", Json.obj ⟨⟨.inner 2 "a0" e .leaf (.inner 1 "a0" e .leaf .leaf)⟩⟩)]

#guard (Registered.portsOf twicePorts).toOption.map (·.size) == some 2
#guard (match Registered.ofJson twicePorts with | .error e => e | .ok _ => "") ==
  "registered reads: port a0 twice: a verdict reports one port per name, and each check reads a name's port"

/-! ### An entry that names its `table_rows`

Port `a0`'s table is all four rows of A, in order (row `r` at position `r`, as a shared table such as a RoPE table is),
and the sampled instances 0 and 1 (units 0 and 1) read only table rows 1 and 3. An entry naming `table_rows` gives a
position and a path for those rows alone; the rows no instance reads need none, and are not opened. -/

namespace Sampled

def circuit : Circuit := { (default : Circuit) with ports := #[aPort "a0"] }

/-- The verifier's `--registered` file: port `a0` of `h1/a`, unit 0 reading row 1 and unit 1 row 3. -/
def file : Json :=
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("a0", Json.mkObj [
    ("value", Json.str "h1/a"), ("root", Json.str (hexOf (root 1))), ("leaves", Lean.toJson (4 : Nat)),
    ("words", Lean.toJson (32 : Nat)), ("domain", Json.str (hexOf dom)), ("row_schema", Json.str v2),
    ("reads", Lean.toJson [1, 3])])])]

/-- Today's entry: a position and a path for each table row. -/
def full : Json :=
  Json.mkObj [("value", Json.str "h1/a"), ("positions", Lean.toJson [0, 1, 2, 3]),
    ("paths", Json.arr (#[0, 1, 2, 3].map fun p => Json.str (hexOf (path 1 p))))]

/-- An entry naming `table_rows` `rows`, at positions `ps`, with the paths of positions `via`. -/
def listing (rows ps : List Nat) (via : List Nat := ps) : Json :=
  Json.mkObj [("value", Json.str "h1/a"), (Registered.TABLE_ROWS_KEY, Lean.toJson rows), ("positions", Lean.toJson ps),
    ("paths", Json.arr (via.map fun p => Json.str (hexOf (path 1 p))).toArray)]

/-- The shared-row statement: table row `r` value `tags[r]`'s row `r`, instance `i` reading table row `2i + 1`, the
header's `registered` entry `e`, its `shared_rows` naming `a0`'s four rows unless `shared` is false. -/
def pub (e : Json) (tags : List Nat := [1, 1, 1, 1]) (shared : Bool := true) : Public :=
  { header := Json.mkObj ([("units", Json.mkObj [("indices", Lean.toJson [0, 1])]),
      ("registered", Json.mkObj [("a0", e)])] ++
      (if shared then [("shared_rows", Json.mkObj [("a0", Lean.toJson (4 : Nat))])] else [])),
    n := 2, digests := (tags.mapIdx fun r tag => bc tag r).toArray, outputs := #[], sha := .empty,
    tables := some { counts := #[4], starts := #[0], refs := #[1, 3] } }

def run (p : Public) : Except String Unit := do
  Registered.check circuit p (some (← Registered.ofJson file)) true

def passes (p : Public) : Bool := (run p).toBool

def why (p : Public) : String :=
  match run p with
  | .error e => e
  | .ok () => ""

-- today's full header, accepted as before
#guard passes (pub full)

-- a header giving only the sampled rows, accepted; the unread rows 0 and 2 are not opened, so rows of another value
-- there pass, where today's full header holding them is refused
#guard passes (pub (listing [1, 3] [1, 3]))
#guard passes (pub (listing [1, 3] [1, 3]) (tags := [2, 1, 2, 1]))
#guard !passes (pub full (tags := [2, 1, 2, 1]))
#guard why (pub full (tags := [2, 1, 2, 1])) ==
  "registered reads: port a0: table row 0 does not open h1/a's registered root at position 0"
-- naming every row is today's entry by another name
#guard passes (pub (listing [0, 1, 2, 3] [0, 1, 2, 3]))

-- a row an instance reads with no position and path: refused
#guard !passes (pub (listing [1] [1]))
#guard why (pub (listing [1] [1])) ==
  "registered reads: port a0: instance 1 reads table row 3, and the header gives no position and path for it"
#guard why (pub (listing [3] [3])) ==
  "registered reads: port a0: instance 0 reads table row 1, and the header gives no position and path for it"
#guard why (pub (listing [0, 2] [0, 2])) ==
  "registered reads: port a0: instance 0 reads table row 1, and the header gives no position and path for it"

-- a wrong path, or a listed row of another value: refused, as today
#guard !passes (pub (listing [1, 3] [1, 3] (via := [1, 1])))
#guard why (pub (listing [1, 3] [1, 3] (via := [1, 1]))) ==
  "registered reads: port a0: table row 3 does not open h1/a's registered root at position 3"
#guard why (pub (listing [1, 3] [1, 3]) (tags := [1, 2, 1, 1])) ==
  "registered reads: port a0: table row 1 does not open h1/a's registered root at position 1"
-- a listed row no instance reads is opened all the same
#guard why (pub (listing [0, 1, 3] [0, 1, 3]) (tags := [2, 1, 1, 1])) ==
  "registered reads: port a0: table row 0 does not open h1/a's registered root at position 0"

-- a listed row at a position the verifier's program does not read for its unit: refused, as today
#guard why (pub (listing [1, 3] [3, 1])) ==
  "registered reads: port a0: table row 1 does not open h1/a's registered root at position 3"
#guard why { pub (listing [1, 3] [3, 1]) with digests := #[bc 1 0, bc 1 3, bc 1 2, bc 1 1] } ==
  "registered reads: port a0: unit 0 reads position 3, and the verifier's program reads 1"

-- the listing itself: strictly increasing, inside the table, one position and one path per listed row
#guard why (pub (listing [3, 1] [3, 1])) ==
  "registered reads: port a0: its table_rows are not strictly increasing"
#guard why (pub (listing [1, 1, 3] [1, 1, 3])) ==
  "registered reads: port a0: its table_rows are not strictly increasing"
#guard why (pub (listing [1, 4] [1, 3])) ==
  "registered reads: port a0: its table_rows name a row past its table's 4"
#guard why (pub (listing [1, 3] [1])) ==
  "registered reads: port a0: one position and one path per row of its table_rows"
#guard why (pub (listing [1, 3] [1, 3] (via := [1]))) ==
  "registered reads: port a0: one position and one path per row of its table_rows"
#guard why (pub (listing [1, 3] [1, 3]) (shared := false)) ==
  "registered reads: a registered port is a table of a shared-row file"
-- a table longer than the header says: the spread entry is not one position per table row
#guard why { pub (listing [1, 3] [1, 3]) with tables := some { counts := #[5], starts := #[0], refs := #[1, 3] } } ==
  "registered reads: port a0: one position and one path per table row"

-- the rows an entry gives (`opened`, what a verdict reports), and `gives` agreeing with it at every row
#guard Registered.opened none 4 == [0, 1, 2, 3] && Registered.opened (some #[1, 3]) 4 == [1, 3]
#guard [none, some #[], some #[1, 3], some #[0, 1, 2, 3], some #[2], some #[0, 5, 9, 12, 40]].all fun sel =>
  (List.range 48).all fun r => Registered.gives sel r == (Registered.opened sel 48).contains r

end Sampled

end Flock.Test.RegRows
