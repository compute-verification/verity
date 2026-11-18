module

public import Flock.Registered
public import Flock.Test.RegProgram
meta import Flock.Registered
meta import Flock.Test.RegProgram
meta import Lean.Data.Json

/-!
Tree ports: a link row read by path in its writer template's output field's tree. On `Flock.Test.RegProgram`'s program a
unit's leaf in its template's trees (`HmLinks.treeLeaf`) is its rank among the template's units in root order, counted
from the nodes, and a unit's writer (`HmLinks.writerOf`) is the unit whose output it reads. A tree port states no reads,
a statement's reads come from the program (`withTrees`), the honest reader's positions pass `badInstance` and a reader
that refers to another writer's row (the shared-tree toy's `ra`) does not; a link row passes `checkLinked` only as a tree
port.
-/

@[expose] public section

namespace Flock.Test.WiredPorts

open Lean (Json toJson)
open Flock Flock.Registered
open Flock.Test.RegProgram (plain)

def prog : Program.Prog := (Program.decode plain).toOption.getD default
def ns : List NodeUnits := HmLinks.nodes prog
def fnOf (v : Nat) : Option Nat := (unitAt ns v).map fun (k, _) => ((prog.defs[prog.root]!).body.nodes[k]!).fn

-- a unit's leaf is the number of earlier units of its template, its template its node's callee
#guard (List.range (count ns)).all fun v =>
  (HmLinks.treeLeaf prog ns v).map (·.2) == some (((List.range v).filter (fnOf · == fnOf v)).length)
#guard (List.range (count ns)).all fun v =>
  (HmLinks.treeLeaf prog ns v).map (·.1) == (fnOf v).map fun fn => (prog.defs[fn]!).key
#guard HmLinks.treeLeaf prog ns (count ns) == none
-- a unit that reads only circuit inputs has no writer; unit 13 reads another unit's output
#guard (List.range 64).all fun t => ((HmLinks.writerOf prog ns 0 t).toOption.join).isNone
#guard (List.range 64).any fun t => ((HmLinks.writerOf prog ns 13 t).toOption.join).isSome

def hex64 : Json := Json.str (String.ofList (List.replicate 128 'a'))

def portJ (reads : List (Option Nat)) (tree : Option Json) : Json :=
  Json.mkObj ([("value", Json.str "rope/out"), ("root", hex64), ("leaves", toJson 16), ("domain", hex64),
    ("reads", toJson reads)] ++ (tree.map fun t => [("tree", t)]).getD [])

def treeJ : Json := Json.mkObj [("template", Json.str "rope"), ("field", Json.str "out")]

def fileJ (ports : List (String × Json)) : Json := Json.mkObj [("format", Json.str FORMAT), ("ports", Json.mkObj ports)]

def x : Registered.Port := match ofJson (fileJ [("x", portJ [] (some treeJ))]) with
  | .ok ps => ps.getD 0 default
  | .error _ => default

def refused (j : Json) : Bool := (ofJson j).toOption.isNone

-- a tree port names its tree and states no reads
#guard x.tree == some ("rope", "out") && x.readsRel.isEmpty
#guard refused (fileJ [("x", portJ [some 0] (some treeJ))])
#guard refused (fileJ [("x", portJ [] (some (Json.mkObj [("template", Json.str "rope")])))])

-- without tree ports the registered reads are as given; a tree port needs the verifier's program
#guard ((withTrees default default none (some #[{ x with tree := none }])).toOption.join.map (·.size)) == some 1
#guard (withTrees default default none (some #[x])).toOption.isNone

-- the toy's reader statement (units [8, 16), instance i its row i, `x` port 0 of 2): unit 8 + i reads writer leaf 7 - i
def resolved : Registered.Port := { x with readsRel := (Array.range 8).map fun i => some (7 - i), readsFrom := 8 }
def tabs : Tables := { counts := #[8, 8], starts := #[0, 8], refs := (Array.range 8).flatMap fun i => #[i, i] }
def units : Array Nat := (Array.range 8).map (· + 8)
#guard badInstance resolved tabs 2 0 8 8 ((Array.range 8).map (7 - ·)) units == none
#guard badInstance resolved tabs 2 0 8 8 (#[6] ++ (Array.range 7).map (6 - ·)) units == some 0

-- a link row is a tree port
def inP : Flock.Port := { name := "x", words := 4, chunks := 1, nb := 0 }
def outP : Flock.Port := { name := "out", words := 4, chunks := 1, nb := 0 }
def links : Json := Json.mkObj [("links", Json.arr #[Json.mkObj [("port", Json.str "x"), ("leaves", toJson [0, 4])]])]
def linked : Circuit :=
  { (default : Circuit) with
    ports := #[inP, outP]
    outRowPorts := #[1]
    metaJ := links }
def ok? (r : V Unit) : Bool := r.toOption.isSome
#guard ok? (HmLinks.parse linked >>= fun _ => pure ())
#guard ok? (checkLinked linked #[x])
#guard !ok? (checkLinked linked #[{ x with tree := none }])
#guard !ok? (checkLinked linked #[{ x with name := "y" }])
#guard ok? (checkLinked { linked with metaJ := Json.null } #[])

end Flock.Test.WiredPorts
