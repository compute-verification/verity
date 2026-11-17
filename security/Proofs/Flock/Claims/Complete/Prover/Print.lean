module

public import Lean.Data.Json.Basic

/-! `printJ` prints what `Json.compress` prints, by structural recursion, so proofs read its bytes. -/

@[expose] public section

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

def itemsJ : List Json → List String
  | [] => []
  | j :: js => printJ j :: itemsJ js

def fieldsJ : Impl String (fun _ => Json) → List String
  | .leaf => []
  | .inner _ k v l r => fieldsJ l ++ (strJ k ++ ":" ++ printJ v) :: fieldsJ r
end

end FlockComplete
