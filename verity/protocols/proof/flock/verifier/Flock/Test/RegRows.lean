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
public import Flock.PortRoots
meta import Flock.Registered
meta import Flock.PortRoots
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

/-- The verifier's `--registered` file: ports `a0` and `a1` both of `h1/a`. -/
def file (words : Nat := 32) : Json :=
  let port := Json.mkObj [("value", Json.str "h1/a"), ("root", Json.str (hexOf (root 1))),
    ("leaves", Lean.toJson (4 : Nat)), ("words", Lean.toJson words), ("domain", Json.str (hexOf dom)),
    ("row_schema", Json.str v2)]
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("a0", port), ("a1", port)])]

/-- The reads `Registered.withPositions` derives from the verifier's program: unit `u` reads rows `2u` and `2u + 1`,
`a0` the first and `a1` the second. -/
def derived (own : Array Registered.Port) : Array Registered.Port :=
  own.mapIdx fun i q => { q with readsRel := #[some i, some (i + 2)] }

def run (p : Public) (f : Json := file) : Except String Unit := do
  Registered.check circuit p (some (derived (← Registered.ofJson f))) true

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
#guard why (pub [0, 2] [1, 3] (value := "h1/b")) == "registered reads: port a0 reads h1/a, not another value"

/-- Another domain than the verifier's file names for A. -/
def dom2 : ByteArray := Sha512.hash "verity/test/registered-domain/b".toUTF8

/-- Row `pos`'s path in a tree of A's own rows under `dom2`: the member's commitment of the right value at another root
than the verifier's. -/
def pathOther (pos : Nat) : ByteArray :=
  let ls := (Array.range 4).map fun r => Registered.leafAs v2 dom2 r (Hm96.Default512.treeLeaf (bc 1 r))
  let j := pos / 2 ^^^ 1
  ls[pos ^^^ 1]! ++ Registered.node dom2 0 j ls[2 * j]! ls[2 * j + 1]!

/-- `pub [0, 2] [1, 3]` with each path its row's in that tree. -/
def pubOther : Public :=
  let p := pub [0, 2] [1, 3]
  let entry (ps : List Nat) := Json.mkObj [("value", Json.str "h1/a"), ("positions", Lean.toJson ps),
    ("paths", Json.arr (ps.map fun q => Json.str (hexOf (pathOther q))).toArray)]
  { p with header := p.header.setObjVal! "registered" (Json.mkObj [("a0", entry [0, 2]), ("a1", entry [1, 3])]) }

-- A's own rows under a root of the member's, the verifier's file unchanged: no row opens the file's root
#guard FV3.root dom2 ((Array.range 4).map fun r => Registered.leafAs v2 dom2 r (Hm96.Default512.treeLeaf (bc 1 r)))
  Sha512.hash != root 1
#guard why pubOther == "registered reads: port a0: table row 0 does not open h1/a's registered root at position 0"
-- one word of a row changed, its path unchanged
#guard why { (pub [0, 2] [1, 3]) with digests := (pub [0, 2] [1, 3]).digests.set! 1 ((bc 1 2).set! 0 0xff) } ==
  "registered reads: port a0: table row 1 does not open h1/a's registered root at position 2"

/-! A port named twice: `JsonParse.parse` keeps one of the two, so the file reads as naming it once (here as one port, `a1`'s
reads under `a0`); `verity-verify-proof` refuses the text first (`JsonKeys.dupKey`). -/

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

/-! A shared tree: one template's output field `T/o0` of four rows of value `1`, member A's rows 0 and 1 and member B's
rows 2 and 3. Member B's public file states its table at port `o0`, its units 2 and 3 reading table rows 0 and 1, with a
position and a path per table row and a `frame_v3` root over its table; `Registered.check` holds the table to the shared
tree and `HmOut.checkPublic` holds it to the header, both reading the one `digests`. -/

def oPort : Port := aPort "o0"

def oCircuit : Circuit := { (default : Circuit) with ports := #[oPort] }

/-- `oCircuit` with its pinned dummy `b ‖ c` its padding row's under a zero salt, as a circuit file's is. -/
def oPinned : Circuit := { oCircuit with dummyDigests := #[HmOut.rowBcZeroSalt oPort (HmOut.dummyRow oCircuit 0)] }

def oId : ByteArray := ByteArray.mk (Array.replicate 64 7)

/-- A member's rows `rows` (of value `tag`) as a table at `o0`, its header's `frame_v3` over `hdr`'s rows of value `htag`
(default: the table's) and its `registered` entry stating `ps` with `paths`'s paths. -/
def shared (rows ps : List Nat) (tag : Nat := 1) (paths : List Nat := ps) (hdr : List Nat := rows) (htag : Nat := tag) :
    Public :=
  let tab (rs : List Nat) (t : Nat) : Public :=
    { header := Json.mkObj [("frame_v3", Json.mkObj [("domain_ids", Json.mkObj [("o0", Json.str (hexOf oId))])])],
      n := 2, digests := (rs.map (bc t)).toArray, outputs := #[], sha := .empty,
      tables := some { counts := #[rs.length], starts := #[0], refs := #[0, 1] } }
  let fv3 := Json.mkObj [("frame", Json.str "frame-v3-sha512"),
    ("row_key_sha512", Json.str (hexOf (Sha512.hash Hm96.Default512.key))),
    ("domain_ids", Json.mkObj [("o0", Json.str (hexOf oId))]),
    ("schemas", Json.mkObj [("o0", Json.str (PortRoots.schemaOf oCircuit 0))]),
    ("roots", Json.mkObj [("o0", Json.str (hexOf (PortRoots.pubRoot oCircuit (tab hdr htag) 0)))])]
  let entry := Json.mkObj [("value", Json.str "T/o0"), ("positions", Lean.toJson ps),
    ("paths", Json.arr (paths.map fun p => Json.str (hexOf (path 1 p))).toArray)]
  { tab rows tag with header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson [2, 3])]),
      ("registered", Json.mkObj [("o0", entry)]), ("frame_v3", fv3)] }

/-- The verifier's port `o0` of `T/o0`, from the registration, and the reads it derives from C: units 2 and 3, member
B's, read leaves 2 and 3. -/
def oReads : Array Registered.Port :=
  match Registered.ofJson (Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("o0",
      Json.mkObj [("value", Json.str "T/o0"), ("root", Json.str (hexOf (root 1))), ("leaves", Lean.toJson (4 : Nat)),
        ("words", Lean.toJson (32 : Nat)), ("domain", Json.str (hexOf dom)), ("row_schema", Json.str v2)])])]) with
  | .ok ps => ps.map fun q => { q with readsFrom := 2, readsRel := #[some 2, some 3] }
  | .error _ => #[]

def readsWhy (p : Public) : String :=
  match Registered.check oCircuit p (some oReads) true with
  | .error e => e
  | .ok () => ""

def headWhy (p : Public) : String := (PortRoots.headerWhy oPinned p).getD ""

-- member B's own rows at its own leaves pass both
#guard oReads.size == 1
#guard readsWhy (shared [2, 3] [2, 3]) == "" && headWhy (shared [2, 3] [2, 3]) == ""

-- the header's rows are not the rows opened at the shared tree, each set valid on its own: the rows the reads open
-- under a header root over other rows (valid over those rows), and those other rows under their own header (valid at
-- the header) opening nothing at the shared tree; either way one check reads the `digests` the other does
#guard readsWhy (shared [2, 3] [2, 3] (hdr := [0, 1]) (htag := 2)) == ""
#guard headWhy (shared [2, 3] [2, 3] (hdr := [0, 1]) (htag := 2)) ==
  s!"{PortRoots.HEADER_WHY} (row o0: the rows' b || c do not hash to the header's frame-v3-sha512 root)"
#guard headWhy (shared [0, 1] [2, 3] (tag := 2)) == ""
#guard readsWhy (shared [0, 1] [2, 3] (tag := 2)) ==
  "registered reads: port o0: table row 0 does not open T/o0's registered root at position 2"
#guard (List.range 2).all fun r =>
  match (shared [2, 3] [2, 3]).tables with
  | some t => Registered.tableRow (shared [2, 3] [2, 3]) t 0 r == (shared [2, 3] [2, 3]).digests[r]!
  | none => false

-- reads registered for all of the table's rows but one
#guard readsWhy (shared [2, 3] [2]) == "registered reads: port o0: one position and one path per table row"
#guard readsWhy (shared [2, 3] [2, 3] (paths := [2])) ==
  "registered reads: port o0: one position and one path per table row"

-- the member's own rows stated at another member's positions open nothing there; another member's rows at their own
-- positions open the tree, read where the verifier's program reads member B's leaves
#guard readsWhy (shared [2, 3] [0, 1] (paths := [2, 3])) ==
  "registered reads: port o0: table row 0 does not open T/o0's registered root at position 0"
#guard readsWhy (shared [2, 3] [0, 1]) ==
  "registered reads: port o0: table row 0 does not open T/o0's registered root at position 0"
#guard readsWhy (shared [0, 1] [0, 1]) ==
  "registered reads: port o0: unit 2 reads position 0, and the verifier's program reads 2"

end Flock.Test.RegRows
