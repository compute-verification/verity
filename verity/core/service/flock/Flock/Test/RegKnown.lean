/-
`Registered.checkKnownAll` on a `--registered` file: a port named at known content (`known`: each row's u16 words and
its salt) passes only when its root is the frame-v3-sha512 root of those rows' `hm96-sha512/row/v2` leaves, as
`registered.Registered.commit` roots them; ports without `known` are untouched.
-/
import Flock.RegKnown

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

end Flock.Test.RegKnown
