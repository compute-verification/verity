/-
`Registered.checkKnownAll` on a `--registered` file: a port named at known content (`known`: each row's u16 words and
its salt) passes only when its root is the frame-v3-sha512 root of those rows' `hm96-sha512/row/v2` leaves, as
`registered.Registered.commit` roots them; ports without `known` are untouched. And the tree a statement publishes
for such a port (`Registered.publishedRoot`, `Registered.knownTrees`): the value's own tree where the statement's
table is the value's rows at the value's domain, another root at another domain or layout, and an entry of the
verifier's list for each port at known content that the statement publishes a tree for.
-/
module

public import Flock.RegKnown
meta import Flock.RegKnown

@[expose] public section

namespace Flock.Test.RegKnown

open Lean (Json)

def dom : ByteArray := Sha512.hash "verity/test/registered-known-domain".toUTF8

def salt (i : Nat) : ByteArray := ByteArray.mk (Array.replicate HmRow.SALT_BYTES (i + 1).toUInt8)

/-- Two `ok` rows, each the one word `1`, under their own salts. -/
def ks : Array (Array Nat × ByteArray) := #[(#[1], salt 0), (#[1], salt 1)]

def reg : Registered.Port :=
  { name := "o1", value := "ok", root := .empty, leaves := 2, words := some 1, wordBits := 16, domain := dom,
    readsRel := #[some 0, some 1], schema := HmRow.ROW_SCHEMA_V2 }

def root (k : Array (Array Nat × ByteArray)) : ByteArray := FV3.root dom (Registered.knownLeaves reg k) Sha512.hash

def knownJ (k : Array (Array Nat × ByteArray)) : Json :=
  Json.mkObj [("rows", Json.arr (k.map fun (r, _) => Lean.toJson r)), ("salts", Json.arr (k.map fun (_, s) => Json.str (hexOf s)))]

/-- The verifier's file of one port `o1` of `ok` rows at `rootHex`, with its known content `known` when given. -/
def file (rootHex : String) (known : Option Json) (schema : String := HmRow.ROW_SCHEMA_V2) (name : String := "o1") :
    Json :=
  Json.mkObj [("format", Json.str Registered.FORMAT), ("ports", Json.mkObj [(name, Json.mkObj (
    [("value", Json.str "ok"), ("root", Json.str rootHex), ("leaves", Lean.toJson (2 : Nat)),
      ("domain", Json.str (hexOf dom)), ("reads", Lean.toJson [some 0, some 1]), ("words", Lean.toJson (1 : Nat)),
      ("row_schema", Json.str schema)] ++
    (match known with | some k => [(Registered.KNOWN_KEY, k)] | none => [])))])]

def run (f : Json) : Except String Unit := do
  let own ← Registered.ofJson f
  Registered.checkKnownAll own (← Registered.knownOf f)

def passes (f : Json) : Bool := (run f).toBool

-- a `verity/registered-reads/v1` port may name its known content
#guard Registered.PORT_KEYS.contains Registered.KNOWN_KEY
-- the root of the known rows
#guard passes (file (hexOf (root ks)) (some (knownJ ks)))
-- no known content: nothing to check
#guard passes (file (hexOf (root ks)) none)
#guard passes (file (hexOf (Sha512.hash .empty)) none)
-- a root that is not the known rows'
#guard !passes (file (hexOf (Sha512.hash .empty)) (some (knownJ ks)))
#guard !passes (file (hexOf (root #[(#[0], salt 0), (#[1], salt 1)])) (some (knownJ ks)))
-- the known rows are not the registered rows' shape
#guard !passes (file (hexOf (root ks)) (some (knownJ #[(#[1], salt 0)])))
#guard !passes (file (hexOf (root ks)) (some (knownJ #[(#[1, 0], salt 0), (#[1, 0], salt 1)])))
#guard !passes (file (hexOf (root ks)) (some (knownJ #[(#[65536], salt 0), (#[1], salt 1)])))
#guard !passes (file (hexOf (root ks)) (some (knownJ #[(#[1], ByteArray.mk #[1]), (#[1], salt 1)])))
-- known content is of row/v2 rows only
#guard !passes (file (hexOf (root ks)) (some (knownJ ks)) (schema := HmRow.ROW_SCHEMA))
-- one salt a row
#guard !passes (file (hexOf (root ks))
  (some (Json.mkObj [("rows", Json.arr #[Lean.toJson #[1], Lean.toJson #[1]]), ("salts", Json.arr #[Json.str (hexOf (salt 0))])])))
#guard (Registered.knownOf (file (hexOf (root ks)) (some (knownJ ks)))).toOption.map (·.map (·.1)) == some #["o1"]

-- a statement's table that is the value's rows, at the value's domain, is the value's tree
#guard Registered.publishedRoot reg dom ks #[0, 1] == root ks
-- and another tree at another domain, in another order, or over fewer rows
def pubDom : ByteArray := Sha512.hash "verity/test/published-domain".toUTF8
#guard Registered.publishedRoot reg pubDom ks #[0, 1] != root ks
#guard Registered.publishedRoot reg dom ks #[1, 0] != root ks
#guard Registered.publishedRoot reg dom ks #[1] != Registered.publishedRoot reg dom ks #[0]
-- an instance's row is the row its unit reads; two instances of one row, and a unit that reads none, are refused
#guard (Registered.tableRows reg #[1, 0]).toOption == some #[1, 0]
#guard (Registered.tableRows reg #[1]).toOption == some #[1]
#guard !(Registered.tableRows reg #[0, 0]).toBool
#guard !(Registered.tableRows reg #[0, 2]).toBool

/-- A statement's header of instances at the units `units`, publishing a tree at `pubDom` for each of `ports`. -/
def header (units : Array Nat) (ports : List String := ["o1"]) : Json :=
  Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson units)]),
    ("frame_v3", Json.mkObj [("domain_ids", Json.mkObj (ports.map fun p => (p, Json.str (hexOf pubDom))))])]

def trees (units : Array Nat) (ports : List String := ["o1"]) : Except String (Array (String × ByteArray)) :=
  Registered.knownTreesOf (file (hexOf (root ks)) (some (knownJ ks))) (header units ports)

-- the one port at known content: its published domain, row count, schema index and root
#guard (trees #[1, 0]).toOption == some #[("o1", pubDom ++ Flock.be64 2 ++ ByteArray.mk #[1] ++
  Registered.publishedRoot reg pubDom ks #[1, 0])]
#guard ((trees #[1]).toOption.map (·.map (·.2.size))) == some #[137]
-- a port the statement publishes no tree for has no entry, and the list of none is its count alone
#guard (trees #[0, 1] ["i0"]).toOption == some #[]
#guard Registered.knownList #[] == Flock.be32 0
#guard ((trees #[0, 1]).toOption.map fun ts => (Registered.knownList (ts.map (·.2))).size) == some (4 + 137)
-- a file whose root is not its known rows' lists nothing
#guard !(Registered.knownTreesOf (file (hexOf (Sha512.hash .empty)) (some (knownJ ks))) (header #[0, 1])).toBool
-- an instance at a unit in which the program reads no row of the value
#guard !(trees #[0, 2]).toBool

end Flock.Test.RegKnown
