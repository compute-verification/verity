/-
`Registered.check` at a port of each row format: a `--registered` port names its rows' leaf schema (`row_schema`, default
`hm96-sha512/row/v1`), and the check accepts it only at a circuit port of that schema (`portSchema`) and width
(`Port.fits`: a v1 row's u16 words in whole SHA-512 blocks, a bit row's exact bits), every table row opening the root
over leaves framed under that schema.
-/
module

public import Flock.Registered
meta import Flock.Registered

@[expose] public section

namespace Flock.Test.RegFormats

open Lean (Json)

def dom : ByteArray := Sha512.hash "verity/test/registered-domain".toUTF8

def bcs : Array ByteArray := #[ByteArray.mk (Array.replicate 32 1), ByteArray.mk (Array.replicate 32 2)]

def leaves (schema : String) : Array ByteArray :=
  bcs.mapIdx fun i bc => Registered.leafAs schema dom i (Hm96.Default512.treeLeaf bc)

/-- The value's root over its two rows, its leaves framed under `schema`, as the plain verifier roots a port's rows. -/
def root (schema : String) : ByteArray := FV3.root dom (leaves schema) Sha512.hash

def v1Port : Port := { name := "w", words := 64, chunks := 2, nb := 0 }
def v2Port : Port := { name := "w", words := 32, chunks := 2, nb := 0, bits := 512, v2 := true }
def segPort : Port := { name := "w", words := 32, chunks := 2, nb := 0, bits := 512, v2 := true, segs := some #[(512, 2)] }

def circuit (p : Port) : Circuit := { (default : Circuit) with ports := #[p] }

/-- A shared-row public file of two instances, instance `i` reading table row `i`, row `r` opened at position `r` through
the other row's leaf under `schema`. -/
def pub (schema : String) (units : List Nat := [0, 1]) : Public :=
  let paths := [Json.str (hexOf (leaves schema)[1]!), Json.str (hexOf (leaves schema)[0]!)]
  { header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson units)]),
      ("registered", Json.mkObj [("w", Json.mkObj [("value", Json.str "weights"), ("positions", Lean.toJson [0, 1]),
        ("paths", Json.arr paths.toArray)])])],
    n := 2, digests := bcs, outputs := #[], sha := .empty,
    tables := some { counts := #[2], starts := #[0], refs := #[0, 1] } }

/-- The verifier's `--registered` file of one port `w`: `row_schema` named only when `schema` is given. -/
def file (rootHex : String) (words : Option Nat) (wordBits : Nat := 16) (schema : Option String := none) : Json :=
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", Json.mkObj (
    [("value", Json.str "weights"), ("root", Json.str rootHex), ("leaves", Lean.toJson (2 : Nat)),
      ("domain", Json.str (hexOf dom))] ++
    (match words with | some w => [("words", Lean.toJson w)] | none => []) ++
    (if wordBits == 16 then [] else [("word_bits", Lean.toJson wordBits)]) ++
    (match schema with | some s => [("row_schema", Json.str s)] | none => [])))])]

/-- The verifier's ports `own` as `withPositions` hands them to the check: unit `i` reads row `i`, the reads
`OneCircuit.pos` derives from C for these two-unit programs. -/
def derivedReads (own : Array Registered.Port) : Array Registered.Port := own.map ({ · with readsRel := #[some 0, some 1] })

/-- `Registered.check` at the circuit with the one port `p`, the public file of `rows` (leaves framed under it), and the
verifier's file `f`, parsed by `Registered.ofJson` (`derivedReads`), the instances of the units `units`. -/
def run (p : Port) (rows : String) (f : Json) (units : List Nat := [0, 1]) : Except String Unit := do
  let own ← Registered.ofJson f
  Registered.check (circuit p) (pub rows units) (some (derivedReads own)) true

def passes (p : Port) (rows : String) (f : Json) (units : List Nat := [0, 1]) : Bool := (run p rows f units).toBool

def why (p : Port) (rows : String) (f : Json) (units : List Nat := [0, 1]) : String :=
  match run p rows f units with
  | .error e => e
  | .ok () => ""

def v1 := HmRow.ROW_SCHEMA
def v2 := HmRow.ROW_SCHEMA_V2
def seg := HmRow.ROW_SCHEMA_SEG

-- the circuit port's schema is the plain verifier's
#guard Registered.portSchema v1Port == v1 && Registered.portSchema v2Port == v2 && Registered.portSchema segPort == seg

-- v1, unchanged: no `row_schema`, rows of u16 words in whole blocks
#guard passes v1Port v1 (file (hexOf (root v1)) (some 32))
#guard passes v1Port v1 (file (hexOf (root v1)) none)
#guard passes v1Port v1 (file (hexOf (root v1)) (some 32) (schema := some v1))

-- v2 and segmented: accepted at their own schema and exact width (16 or 32-bit leaves)
#guard passes v2Port v2 (file (hexOf (root v2)) (some 32) (schema := some v2))
#guard passes v2Port v2 (file (hexOf (root v2)) (some 16) (wordBits := 32) (schema := some v2))
#guard passes segPort seg (file (hexOf (root seg)) (some 32) (schema := some seg))

-- a wrong root
#guard !passes v2Port v2 (file (hexOf (root v1)) (some 32) (schema := some v2))
#guard !passes segPort seg (file (hexOf (root v2)) (some 32) (schema := some seg))
#guard !passes v2Port v2 (file (hexOf (root v2 |>.set! 0 0)) (some 32) (schema := some v2))

-- a root over leaves framed under another schema: the verifier frames each row under the port's own
#guard !passes v2Port v1 (file (hexOf (root v1)) (some 32) (schema := some v2))
#guard !passes segPort v2 (file (hexOf (root v2)) (some 32) (schema := some seg))
#guard !passes v2Port seg (file (hexOf (root seg)) (some 32) (schema := some v2))

-- the wrong format: the registration's schema is not the circuit port's
#guard !passes v2Port v2 (file (hexOf (root v2)) (some 64))
#guard !passes v1Port v2 (file (hexOf (root v2)) (some 32) (schema := some v2))
#guard !passes segPort v2 (file (hexOf (root v2)) (some 32) (schema := some v2))
#guard !passes v2Port seg (file (hexOf (root seg)) (some 32) (schema := some seg))
#guard why v2Port v2 (file (hexOf (root v2)) (some 64)) ==
  "registered reads: port w is a hm96-sha512/row/v2 row, and weights's registered rows are hm96-sha512/row/v1 rows"

-- the wrong width: a bit row is exactly `words · word_bits` bits, no block padding
#guard !passes v2Port v2 (file (hexOf (root v2)) (some 31) (schema := some v2))
#guard !passes v2Port v2 (file (hexOf (root v2)) (some 33) (schema := some v2))
#guard !passes v2Port v2 (file (hexOf (root v2)) (some 32) (wordBits := 32) (schema := some v2))
#guard !passes segPort seg (file (hexOf (root seg)) (some 64) (schema := some seg))
#guard why v2Port v2 (file (hexOf (root v2)) (some 31) (schema := some v2)) ==
  "registered reads: port w's rows are 512 bits (32 u16 words), not weights's registered rows of 31 16-bit leaves; a reader of registered leaves at another width (the same bits cut again, low part first) is not read until the Program's Lean defines that cut"

-- the file: a bit row names its words, and `row_schema` is one of the three
#guard !(Registered.ofJson (file (hexOf (root v2)) none (schema := some v2))).toBool
#guard !(Registered.ofJson (file (hexOf (root v2)) (some 32) (schema := some "hm96-sha512/row/v3"))).toBool
#guard ((Registered.ofJson (file (hexOf (root v2)) (some 32) (schema := some v2))).toOption.map
  (·.map (·.schema))) == some #[v2]
#guard ((Registered.ofJson (file (hexOf (root v1)) (some 32))).toOption.map (·.map (·.schema))) == some #[v1]

-- a port names no position: C and the partition give them
#guard !(Registered.ofJson (Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [("w", Json.mkObj
  [("value", Json.str "weights"), ("root", Json.str (hexOf (root v1))), ("leaves", Lean.toJson (2 : Nat)),
    ("domain", Json.str (hexOf dom)), ("reads", Lean.toJson [some 0, some 1])])])])).toBool

end Flock.Test.RegFormats
