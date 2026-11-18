module

public import Flock.Test.RegProgram
meta import Flock.Test.RegProgram

/-!
`HmLinks.check` reads each unit and each gate's unit from the root body's nodes (`unitAt`, `ownerAt`); on
`Flock.Test.RegProgram`'s program (a call, batches, a scan and primitives) those lookups are the unrolled units' (`unitRef`,
`ownerRef`) at every unit and gate, the check over the population is the reference walk's (`checkRef`), and a sample that
proves a unit reads only its wires. A link row reads a run of one writer unit's outputs from the offset the program gives:
unit 13 reads the first of unit 0's two outputs, at offset 0 and at no other, and a run that runs on into a circuit input
is refused.
-/

@[expose] public section

namespace Flock.Test.LinkUnits

open Lean (Json toJson)
open Flock Flock.HmLinks
open Flock.Test.RegComplete (lane pub)
open Flock.Test.RegProgram (plain)

def prog : Program.Prog := (Program.decode plain).toOption.getD default

def ns : List NodeUnits := nodes prog

def at_ (idx : List Nat) : Public :=
  { pub with header := Json.mkObj [("units", Json.mkObj [("indices", Lean.toJson idx)])] }

def res (r : Except String Nat) : String :=
  match r with
  | .ok n => s!"ok {n}"
  | .error e => e

-- the units, from the description: the root body's nodes that are not input nodes, and their members
#guard ns.map (fun (k, mm, _, _) => (k, mm)) == [(3, 1), (4, 2), (5, 2), (6, 4), (7, 4), (8, 4), (9, 1), (10, 1), (11, 1), (12, 1)]
#guard count ns == 21 && (unitsRef ns).length == 21

-- every unit and every gate's unit read from the nodes is the unrolled lookup's, past the last unit and gate too
#guard (List.range 24).all fun u => unitAt ns u == unitRef ns u
#guard (List.range 400).all fun g => ownerAt ns g == ownerRef ns g
#guard (List.range 400).any fun g => (ownerAt ns g).isSome

-- over the population, the check is the reference walk's: units that read only circuit inputs pass ...
#guard res (HmLinks.check lane (at_ [0, 1]) none plain) == "ok 0"
#guard res (HmLinks.checkRef lane (at_ [0, 1]) plain) == "ok 0"
-- ... and a scan iteration that reads another unit's output with no link row refuses, in both
#guard res (HmLinks.check lane (at_ [0, 13]) none plain) == res (HmLinks.checkRef lane (at_ [0, 13]) plain)
#guard (res (HmLinks.check lane (at_ [0, 13]) none plain)).startsWith "links: cut wire: unit 13's parameter leaf 0 reads unit 0's output"

-- a sample that proves only the first unit checks only its wires; one that proves the scan iteration refuses
#guard res (HmLinks.check lane (at_ [0, 13]) (some #[0]) plain) == "ok 0"
#guard res (HmLinks.check lane (at_ [0, 13]) (some #[1]) plain) == res (HmLinks.check lane (at_ [0, 13]) none plain)
#guard (res (HmLinks.check lane (at_ [0, 13]) (some #[2]) plain)).startsWith "links: unit 2 of the draw is outside"

-- unit 13's leaf 0 is unit 0's output 0 of 2: one link row x, the writer's whole row of 4 words, its leaves [lo, hi) from
-- word `off` (no `at` key when none)
def inP : Flock.Port := { name := "x", words := 4, chunks := 1, nb := 0 }
def outP : Flock.Port := { name := "out", words := 4, chunks := 1, nb := 0 }
def lk (lo hi : Nat) (off : Option Nat) : Circuit :=
  { (default : Circuit) with
    ports := #[inP, outP]
    outRowPorts := #[1]
    metaJ := Json.mkObj [("links", Json.arr #[Json.mkObj ([("port", Json.str "x"), ("leaves", toJson [lo, hi])] ++
      (off.map fun o => [("at", toJson o)]).getD [])])] }
def unit13 (c : Circuit) : String := res (HmLinks.check c (at_ [0, 13]) (some #[1]) plain)

#guard unit13 (lk 0 1 none) == "ok 1"
#guard unit13 (lk 0 1 (some 0)) == "ok 1"
-- wrong_offset: the program's offset is 0
#guard (unit13 (lk 0 1 (some 1))).startsWith "links: unit 13's leaves [0, 1) (row x) are not unit 0's outputs [1, 2) of its 2"
#guard (unit13 (lk 0 1 (some 2))).startsWith "links: unit 13's leaves [0, 1) (row x) are not unit 0's outputs [2, 3) of its 2"
-- a straddle: leaf 1 is a circuit input
#guard (unit13 (lk 0 2 (some 0))).startsWith "links: unit 13's leaves [0, 2) (row x) read unit 0's outputs and other leaves"
-- the row holds the read inside its words, and a link has no other key
#guard (unit13 (lk 0 1 (some 4))).startsWith "META links: x is not a hm96-sha512/row/v1 row holding its 1 leaves from word 4"
#guard (unit13 { lk 0 1 none with metaJ := Json.mkObj [("links", Json.arr #[Json.mkObj [("port", Json.str "x"),
    ("leaves", toJson [0, 1]), ("at", toJson 0), ("len", toJson 1)]])] }).startsWith "META links: a link is exactly"

end Flock.Test.LinkUnits
