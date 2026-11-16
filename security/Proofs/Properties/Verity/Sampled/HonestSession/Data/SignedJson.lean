module

public import Lean.Data.Json.Basic

/-!
# A JSON value's text with signed numbers

`jsonCharsS` is `Flock.ZkServer.jsonChars` with each number's mantissa as `Int.repr` writes it, its sign included, where
`jsonChars` writes it as a natural. The circuit file's META line is its text, since that META holds the input leaf `-1`.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession

open Lean (Json)

mutual
/-- A value's text: no whitespace, each object's members in its tree's order, each number its mantissa (`Int.repr`). -/
def jsonCharsS : Json → List Char
  | .null => ['n', 'u', 'l', 'l']
  | .bool true => ['t', 'r', 'u', 'e']
  | .bool false => ['f', 'a', 'l', 's', 'e']
  | .num ⟨m, _⟩ => (Int.repr m).toList
  | .str s => '"' :: s.toList ++ ['"']
  | .arr ⟨a⟩ => '[' :: [','].intercalate (jsonCharsSList a) ++ [']']
  | .obj ⟨⟨t⟩⟩ => '{' :: [','].intercalate (jsonCharsSObj t) ++ ['}']
def jsonCharsSList : List Json → List (List Char)
  | [] => []
  | j :: js => jsonCharsS j :: jsonCharsSList js
def jsonCharsSObj : Std.DTreeMap.Internal.Impl String (fun _ => Json) → List (List Char)
  | .leaf => []
  | .inner _ k v l r => jsonCharsSObj l ++ [('"' :: k.toList ++ ['"', ':']) ++ jsonCharsS v] ++ jsonCharsSObj r
end

end Verity.Sampled.Instance.HonestSession
