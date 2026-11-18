module

public import Flock.Registered

/-!
A wired registered port on the shared-tree window: `2n` units of one template, reader unit `n + i` reading writer unit
`n - 1 - i`'s output as its `x` (`wireJ`). Its reads are its wire's in closed form (`Port.readAt`), the array the proofs read
agrees, the honest reader's positions pass `badInstance` and a reader that refers to another writer's row (`ra`) does
not; a link row passes `checkLinked` only as a wired port reading that row.
-/

@[expose] public section

namespace Flock.Test.WiredPorts

open Lean (Json toJson)
open Flock Flock.Registered

def hex64 : Json := Json.str (String.ofList (List.replicate 128 'a'))

def wireJ (n : Nat) (extra : List (String × Json) := []) : Json :=
  Json.mkObj ([("reader", Json.str "rope"), ("port", Json.str "x"), ("writer", Json.str "rope"), ("out", Json.str "out"),
    ("at", toJson n), ("radix", toJson [n]), ("base", toJson ((n : Int) - 1)), ("step", toJson [(-1 : Int)])] ++ extra)

def portJ (reads : List (Option Nat)) (wire : Option Json) : Json :=
  Json.mkObj ([("value", Json.str "w_out"), ("root", hex64), ("leaves", toJson 16), ("domain", hex64),
    ("reads", toJson reads)] ++ (wire.map fun w => [("wire", w)]).getD [])

def fileJ (ports : List (String × Json)) : Json := Json.mkObj [("format", Json.str FORMAT), ("ports", Json.mkObj ports)]

def x8 : Registered.Port := match ofJson (fileJ [("x", portJ [] (some (wireJ 8)))]) with
  | .ok ps => ps.getD 0 default
  | .error _ => default

def refused (j : Json) : Bool := (ofJson j).toOption.isNone

-- the toy's reader: units below 8 read no row, unit 8 + i reads writer unit 7 - i's leaf, units past 15 none
#guard x8.wire.isSome
#guard (List.range 20).all fun u =>
  x8.readAt u == if u < 8 then some none else if u < 16 then some (some (15 - u)) else none
#guard (List.range 20).all fun u => x8.readAt u == x8.reads[u]?
-- from a later population unit: `readsFrom` shifts the reader's units
#guard ({ x8 with readsFrom := 3 }).readAt 11 == some none && ({ x8 with readsFrom := 3 }).readAt 19 == some (some 4)

-- a wired port's reads are its wire's: its own reads, a cut row, or a malformed wire are refused
#guard refused (fileJ [("x", portJ [some 0] (some (wireJ 8)))])
#guard refused (fileJ [("x", portJ [] (some (wireJ 8 [("cut", toJson [0, 2])])))])
#guard refused (fileJ [("x", portJ [] (some (Json.mkObj [("reader", Json.str "rope")])))])
#guard !refused (fileJ [("x", portJ [] (some (wireJ 8 [("cut", toJson [0, 1])])))])

-- the reader statement (units [8, 16), instance i its row i, `x` port 0 of 2): the honest positions pass, `ra`'s don't
def tabs : Tables := { counts := #[8, 8], starts := #[0, 8], refs := (Array.range 8).flatMap fun i => #[i, i] }
def units : Array Nat := (Array.range 8).map (· + 8)
#guard badInstance x8 tabs 2 0 8 8 ((Array.range 8).map (7 - ·)) units == none
#guard badInstance x8 tabs 2 0 8 8 (#[6] ++ (Array.range 7).map (6 - ·)) units == some 0

-- a link row is a wired port reading that row
def inP : Flock.Port := { name := "x", words := 4, chunks := 1, nb := 0 }
def outP : Flock.Port := { name := "out", words := 4, chunks := 1, nb := 0 }
def linked : Circuit :=
  { (default : Circuit) with ports := #[inP, outP], outRowPorts := #[1],
    metaJ := Json.mkObj [("links", Json.arr #[Json.mkObj [("port", Json.str "x"), ("leaves", toJson [0, 4])]])] }
def ok? (r : V Unit) : Bool := r.toOption.isSome
#guard ok? (HmLinks.parse linked >>= fun _ => pure ())
#guard ok? (checkLinked linked #[x8])
#guard !ok? (checkLinked linked #[{ x8 with wire := none }])
#guard !ok? (checkLinked linked #[{ x8 with name := "y" }])
#guard !ok? (checkLinked linked #[{ x8 with wire := x8.wire.map ({ · with port := "y" }) }])
#guard ok? (checkLinked { linked with metaJ := Json.null } #[])

end Flock.Test.WiredPorts
