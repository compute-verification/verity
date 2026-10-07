/-
Canonical JSON: compact, object keys sorted bytewise, serde_json's string escaping (the form `compact-sorted-JSON` of
PROTOCOL.md §16 hashes). Only integers are accepted as numbers; anything else is refused rather than guessed.
-/
import Lean.Data.Json
import Flock.Bytes

namespace Flock

open Lean (Json JsonNumber)

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

def canonStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | c => if c.toNat == 8 then out ++ "\\b" else if c.toNat == 12 then out ++ "\\f"
             else if c.toNat < 0x20 then out ++ "\\u" ++ hex4 c.toNat else out.push c
  return out ++ "\""

partial def canon : Json → V String
  | Json.null => pure "null"
  | Json.bool b => pure (if b then "true" else "false")
  | Json.num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | Json.str s => pure (canonStr s)
  | Json.arr a => do
    let parts ← a.mapM canon
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | Json.obj kvs => do
    let items := (kvs.toArray.qsort (fun a b => a.1 < b.1))
    let parts ← items.mapM fun (k, v) => do pure (canonStr k ++ ":" ++ (← canon v))
    pure ("{" ++ ",".intercalate parts.toList ++ "}")

/- Total accessors over parsed JSON; every failure names the field. -/
namespace J

def get (j : Json) (k : String) (ctx : String) : V Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"{ctx}: {k} missing"

def has (j : Json) (k : String) : Bool :=
  match j.getObjVal? k with
  | .ok _ => true
  | .error _ => false

def asStr (j : Json) (ctx : String) : V String :=
  match j.getStr? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a string"

def asNat (j : Json) (ctx : String) : V Nat :=
  match j.getNat? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a natural number"

def asInt (j : Json) (ctx : String) : V Int :=
  match j.getInt? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not an integer"

def asArr (j : Json) (ctx : String) : V (Array Json) :=
  match j.getArr? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a list"

def asBool (j : Json) (ctx : String) : V Bool :=
  match j.getBool? with
  | .ok s => .ok s
  | .error _ => .error s!"{ctx}: not a boolean"

def str (j : Json) (k ctx : String) : V String := do asStr (← get j k ctx) s!"{ctx}.{k}"
def nat (j : Json) (k ctx : String) : V Nat := do asNat (← get j k ctx) s!"{ctx}.{k}"
def arr (j : Json) (k ctx : String) : V (Array Json) := do asArr (← get j k ctx) s!"{ctx}.{k}"
def bool (j : Json) (k ctx : String) : V Bool := do asBool (← get j k ctx) s!"{ctx}.{k}"

end J

end Flock
