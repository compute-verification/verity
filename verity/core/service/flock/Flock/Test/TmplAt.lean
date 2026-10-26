/-
`Tmpl.unrollOk` against `Tmpl.unrollAt`: `Flock.Test.Tmpl`'s unit passes it under every layout mode, and `unrollAt` builds
its committed net (`FlockSoundness.TmplFold.unrollAt_ok`). Layouts it refuses, each one `unrollAt` refuses too: the pair's
copies placed over the root's own columns (`layApart`), a lane count below the copies (`placeOk`), and a layout of the pair
marked `ident` (`identOk`) placed at the input rows.
-/
import Flock.TmplAt
import Flock.Test.Tmpl

namespace Flock.Test.TmplAt

open Flock Flock.Test.Tmpl

def laid (mode : Tmpl.Mode) : Except String (Tmpl.Tmpl × Array Tmpl.Lay) := do
  let t ← Tmpl.parse tmplLines
  return (t, ← Tmpl.layoutM t mode)

/-- `unrollOk` and whether `unrollAt` succeeds, at `layoutM`'s layouts changed by `f`. -/
def both (mode : Tmpl.Mode) (f : Array Tmpl.Lay → Array Tmpl.Lay := id) : Option (Bool × Bool) :=
  (laid mode).toOption.map fun (t, lays) => (Tmpl.unrollOk t (f lays), (Tmpl.unrollAt t (f lays)).toOption.isSome)

/-- The root's place `i` changed by `g`. -/
def rootPlace (i : Nat) (g : Tmpl.Place → Tmpl.Place) (lays : Array Tmpl.Lay) : Array Tmpl.Lay :=
  lays.modify 2 fun L => { L with places := L.places.modify i g }

#guard [Tmpl.Mode.lenient, .strict, .hybrid 0, .hybrid 2].all fun m => both m == some (true, true)
-- the pair's copies at 0, over the root's input rows
#guard both .lenient (rootPlace 0 fun p => { p with nb := 0 }) == some (false, false)
#guard (laid .lenient).toOption.any fun (t, lays) => !Tmpl.layApart t (rootPlace 0 (fun p => { p with nb := 0 }) lays) 2
-- the pair's two copies in one lane
#guard both .lenient (rootPlace 0 fun p => { p with ms := 1 }) == some (false, false)
-- the pair marked `ident`, its copies at the input rows
#guard both .lenient (fun lays => rootPlace 0 (fun p => { p with nb := 0, cs := 3, ms := 1 })
  (lays.modify 1 fun L => { L with ident := true })) == some (false, false)

end Flock.Test.TmplAt
