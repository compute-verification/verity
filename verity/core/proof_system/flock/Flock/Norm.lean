/-
A parsed JSON value rebuilt: `Json.parse` builds an object's tree in the order its keys arrived, so where the verifier
keeps a parsed value (a draw) it keeps `norm` of it, whose trees depend only on each object's fields.
-/
import Lean.Data.Json

namespace Flock

open Lean (Json)

/-- An object rebuilt from its fields: inserted into an empty map, and that map's fields, now in key order, inserted
into another, so the tree is the same for any two lists of the same fields. -/
def normObj (fields : List (String × Json)) : Std.TreeMap.Raw String Json compare :=
  Std.TreeMap.Raw.ofList (Std.TreeMap.Raw.ofList fields compare).toList compare

mutual
/-- **A JSON value with every object rebuilt** (`normObj`): `Json.parse` builds a tree in the order the keys arrived,
which no proof can see into, so where the verifier keeps a parsed value (a draw) it keeps this instead, a well-formed
tree that is the same value for any two inputs with the same fields at each key. -/
def norm : Json → Json
  | .arr ⟨a⟩ => .arr ⟨normList a⟩
  | .obj ⟨⟨t⟩⟩ => .obj (normObj (normFields t))
  | j => j

/-- `norm` elementwise. -/
def normList : List Json → List Json
  | [] => []
  | a :: as => norm a :: normList as

/-- An object's fields with `norm` on their values, in the order of its tree. -/
def normFields : Std.DTreeMap.Internal.Impl String (fun _ => Json) → List (String × Json)
  | .leaf => []
  | .inner _ k v l r => normFields l ++ (k, norm v) :: normFields r
end

end Flock
