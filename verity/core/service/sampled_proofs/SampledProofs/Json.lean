/-
JSON as the Python reference reads it, over canonical JSON (integers only): Python's `==`, `bool(x)`, `x.get(k)` and its
integer tests. The one difference is deliberate: `jeq` tells `true` from `1`, which Python's `==` doesn't.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Hash

namespace SampledProofs

open Lean (Json)
open Flock (V)

/-- Python's `a == b` on canonical JSON values: the same canonical text. -/
def jeq (a b : Json) : Bool :=
  match Flock.canon a, Flock.canon b with
  | .ok x, .ok y => x == y
  | _, _ => false

/-- Python's `bool(x)`. -/
def truthy : Json → Bool
  | .null => false
  | .bool b => b
  | .num n => n.mantissa != 0
  | .str s => !s.isEmpty
  | .arr a => !a.isEmpty
  | .obj kvs => !kvs.toArray.isEmpty

/-- An integer that is not a boolean (Python's `isinstance(x, int) and not isinstance(x, bool)`). -/
def int? : Json → Option Int
  | .num n => if n.exponent == 0 then some n.mantissa else none
  | _ => none

def nat? (j : Json) : Option Nat :=
  match int? j with
  | some i => if 0 ≤ i then some i.toNat else none
  | none => none

def sortStrings (xs : Array String) : Array String := xs.qsort (· < ·)

/-- An object's keys, sorted; `none` for anything that isn't an object. -/
def keys? : Json → Option (Array String)
  | .obj kvs => some (sortStrings (kvs.toArray.map (·.1)))
  | _ => none

def field? (j : Json) (k : String) : Option Json := (j.getObjVal? k).toOption

/-- Python's `d.get(k)`: a missing key reads as `null`. -/
def getD (j : Json) (k : String) : Json := (field? j k).getD Json.null

def entries : Json → Array (String × Json)
  | .obj kvs => kvs.toArray
  | _ => #[]

/-- A failure code, kept once, in the order first found (`verdict._Collect`). -/
def addCode (cs : Array String) (code : String) : Array String := if cs.contains code then cs else cs.push code

def sha512Hex (b : ByteArray) : String := Flock.hexOf (Flock.Sha512.hash b)

/-- SHA-512 of the canonical JSON, hex. -/
def canonSha512 (j : Json) : V String := do return sha512Hex (← Flock.canon j).toUTF8

/-- `re.match(r"^[0-9a-f]+$", s)`, without Python's trailing newline. -/
def isHex (s : String) : Bool := !s.isEmpty && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

def natsJson (xs : Array Nat) : Json := Json.arr (xs.map fun x => Lean.toJson x)

end SampledProofs
