import Lean.Data.Json

/-!
# The record's printer

`printJ`: the honest prover's JSON text, by structural recursion, so proofs read its bytes. It prints what
`Json.compress` prints: `null`, `true` and `false`, a number's `toString`, a string between quotes (`strJ`, with
`"`, `\`, newline and carriage return escaped and a character below `0x20` as `\u` and four hex digits, `escC`), and an
array's elements and an object's fields (in its tree's order, sorted by key) between commas.
-/

namespace FlockComplete

open Lean (Json)
open Std.DTreeMap.Internal (Impl)

/-- A character inside a printed string. -/
def escC (c : Char) : String :=
  if c = '"' then "\\\""
  else if c = '\\' then "\\\\"
  else if c = '\n' then "\\n"
  else if c = '\x0d' then "\\r"
  else if 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff then String.singleton c
  else
    let n := c.toNat
    "\\u" ++ String.ofList [Nat.digitChar (n / 4096), Nat.digitChar ((n % 4096) / 256), Nat.digitChar ((n % 256) / 16),
      Nat.digitChar (n % 16)]

/-- A string, printed. -/
def strJ (s : String) : String := "\"" ++ String.join (s.toList.map escC) ++ "\""

mutual
/-- **A JSON value, printed.** -/
def printJ : Json → String
  | .null => "null"
  | .bool b => toString b
  | .num n => toString n
  | .str s => strJ s
  | .arr ⟨l⟩ => "[" ++ ",".intercalate (itemsJ l) ++ "]"
  | .obj ⟨⟨t⟩⟩ => "{" ++ ",".intercalate (fieldsJ t) ++ "}"

/-- An array's elements, printed. -/
def itemsJ : List Json → List String
  | [] => []
  | j :: js => printJ j :: itemsJ js

/-- An object's fields, printed in its tree's order. -/
def fieldsJ : Impl String (fun _ => Json) → List String
  | .leaf => []
  | .inner _ k v l r => fieldsJ l ++ (strJ k ++ ":" ++ printJ v) :: fieldsJ r
end

end FlockComplete
