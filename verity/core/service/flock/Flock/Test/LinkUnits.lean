/-
`HmLinks.check` reads each unit and each gate's unit from the root body's nodes (`unitAt`, `ownerAt`); on
`Flock.Test.RegProgram`'s program (a call, batches, a scan and primitives) those lookups are the unrolled units' (`unitRef`,
`ownerRef`) at every unit and gate, the check over the population is the reference walk's (`checkRef`), and a draw that
proves a unit reads only its wires.
-/
import Flock.Test.RegProgram

namespace Flock.Test.LinkUnits

open Lean (Json)
open Flock Flock.HmLinks
open Flock.Test.RegComplete (lane pub)
open Flock.Test.RegProgram (plain)

def prog : Program.Prog := (Program.decode plain).toOption.getD default

def ns : List Node := nodes prog

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

-- a draw that proves only the first unit checks only its wires; one that proves the scan iteration refuses
#guard res (HmLinks.check lane (at_ [0, 13]) (some #[0]) plain) == "ok 0"
#guard res (HmLinks.check lane (at_ [0, 13]) (some #[1]) plain) == res (HmLinks.check lane (at_ [0, 13]) none plain)
#guard (res (HmLinks.check lane (at_ [0, 13]) (some #[2]) plain)).startsWith "links: unit 2 of the draw is outside"

end Flock.Test.LinkUnits
