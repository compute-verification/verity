/-
A JSON value's canonical form (`rec_live.canonical`): its tokens as `Json.compress` writes them. `compress` and `Json`'s `==`
are `partial def`s, which no proof can unfold; `canonForm` is total. It is a module of its own so that `Tags.plainLeaves`
reads it without importing `Flock.Firewall`, whose `Schedule` would shadow `Flock.Schedule` in every `Flock.Firewall.*`
namespace that imports `Tags`.
-/
module

public import Lean.Data.Json

@[expose] public section

namespace Flock.Firewall

open Lean (Json)

/-- One token of a JSON value as `Json.compress` writes it: a number as its text, an array's or object's members between
its opening and `close`, each member of an object its `key` then its value, in key order. -/
inductive Tok where
  | null
  | bool (b : Bool)
  | num (text : String)
  | str (s : String)
  | arr
  | obj
  | key (k : String)
  | close
  deriving DecidableEq

mutual
/-- A value's canonical form, its tokens as `Json.compress` writes them, so that two values compress alike exactly when
their canonical forms are equal. `compress` is a `partial def`, which no proof can unfold; `canonForm` is total. -/
def canonForm : Json → List Tok
  | .null => [.null]
  | .bool b => [.bool b]
  | .num n => [.num (toString n)]
  | .str s => [.str s]
  | .arr ⟨a⟩ => .arr :: canonFormList a ++ [.close]
  | .obj ⟨⟨t⟩⟩ => .obj :: canonFormObj t ++ [.close]
def canonFormList : List Json → List Tok
  | [] => []
  | j :: js => canonForm j ++ canonFormList js
def canonFormObj : Std.DTreeMap.Internal.Impl String (fun _ => Json) → List Tok
  | .leaf => []
  | .inner _ k v l r => canonFormObj l ++ (.key k :: canonForm v) ++ canonFormObj r
end

end Flock.Firewall
