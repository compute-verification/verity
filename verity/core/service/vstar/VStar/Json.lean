import Lean.Data.Json
import VStar.InnerRepCheck

/-!
# V*'s structures as JSON

`GfResiduals`' S as `gf2k` writes it (`{"nw", "nv", "ops": [[term, ..], ..], "res": [{"mul": [[a, b], ..], "add": [term,
..]}, ..]}`, a term `["w", i]`, `["v", i]` or `["wt", i, u]`) and `InnerRepCheck`'s (`{"gf", "ports": [[name, n], ..]}`),
read strictly: a key missing or extra, or a number that is not a natural, is refused.
-/

namespace VStar

open Lean (Json)

def jNat (j : Json) (ctx : String) : Except String Nat :=
  match j with
  | .num n => if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat else throw s!"{ctx}: not a natural number"
  | _ => throw s!"{ctx}: not a natural number"

def jStr (j : Json) (ctx : String) : Except String String :=
  match j with
  | .str s => pure s
  | _ => throw s!"{ctx}: not a string"

def jArr (j : Json) (ctx : String) : Except String (Array Json) :=
  match j with
  | .arr a => pure a
  | _ => throw s!"{ctx}: not a list"

/-- An object with exactly the keys `keys`. -/
def jObj (j : Json) (keys : List String) (ctx : String) : Except String (String → Json) :=
  match j with
  | .obj kvs =>
    let got := kvs.toArray.map (·.1)
    if got.size == keys.length && keys.all got.contains then pure fun k => (j.getObjValD k)
    else throw s!"{ctx}: an object of the keys {keys}"
  | _ => throw s!"{ctx}: not an object"

def jNatAt (o : String → Json) (k ctx : String) : Except String Nat := jNat (o k) s!"{ctx}.{k}"

def Term.ofJson (j : Json) : Except String Term := do
  let a ← jArr j "term"
  let t ← jStr (a.getD 0 Json.null) "term"
  match t, a.size with
  | "w", 2 => pure (.w (← jNat (a.getD 1 Json.null) "term"))
  | "v", 2 => pure (.v (← jNat (a.getD 1 Json.null) "term"))
  | "wt", 3 => pure (.wt (← jNat (a.getD 1 Json.null) "term") (← jNat (a.getD 2 Json.null) "term"))
  | _, _ => throw "term: not w, v or wt"

def Term.toJson : Term → Json
  | .w i => Json.arr #["w", (i : Json)]
  | .v i => Json.arr #["v", (i : Json)]
  | .wt i u => Json.arr #["wt", (i : Json), (u : Json)]

def GfS.ofJson (j : Json) : Except String GfS := do
  let o ← jObj j ["nw", "nv", "ops", "res"] "S"
  let ops ← (← jArr (o "ops") "S.ops").mapM fun op => do (← jArr op "S.ops[k]").mapM Term.ofJson
  let res ← (← jArr (o "res") "S.res").mapM fun r => do
    let ro ← jObj r ["mul", "add"] "S.res[r]"
    let mul ← (← jArr (ro "mul") "S.res[r].mul").mapM fun ab => do
      match (← jArr ab "a product").toList with
      | [a, b] => pure ((← jNat a "a product"), (← jNat b "a product"))
      | _ => throw "a product is [a, b]"
    let add ← (← jArr (ro "add") "S.res[r].add").mapM Term.ofJson
    pure ({ mul := mul.toList, add := add.toList } : Res)
  pure { nw := ← jNatAt o "nw" "S", nv := ← jNatAt o "nv" "S", ops := (ops.map Array.toList).toList, res := res.toList }

def GfS.toJson (S : GfS) : Json := Json.mkObj [
  ("nw", (S.nw : Json)), ("nv", (S.nv : Json)),
  ("ops", Json.arr (S.ops.toArray.map fun o => Json.arr (o.toArray.map Term.toJson))),
  ("res", Json.arr (S.res.toArray.map fun r => Json.mkObj [
    ("mul", Json.arr (r.mul.toArray.map fun (a, b) => Json.arr #[(a : Json), (b : Json)])),
    ("add", Json.arr (r.add.toArray.map Term.toJson))]))]

def RepS.ofJson (j : Json) : Except String RepS := do
  let o ← jObj j ["gf", "ports"] "InnerRepCheck's S"
  let gf ← GfS.ofJson (o "gf")
  let ports ← (← jArr (o "ports") "S.ports").mapM fun p => do
    match (← jArr p "a row port").toList with
    | [nm, n] => pure ((← jStr nm "a row port's name"), (← jNat n "a row port's elements"))
    | _ => throw "a row port is [name, elements]"
  pure { gf, ports := ports.toList }

def RepS.toJson (S : RepS) : Json := Json.mkObj [
  ("gf", S.gf.toJson), ("ports", Json.arr (S.ports.toArray.map fun (nm, n) => Json.arr #[(nm : Json), (n : Json)]))]

end VStar
