module

public import Spot.Admitted
meta import Spot.Admitted

/-!
The refusals of `Spot.OneStageZk.admitted` and `admittedBlock` that no decoded descriptor reaches, each on the smallest
C_S or C, changed by hand, that only that refusal refuses, by its whole text.
-/

@[expose] public section

namespace Spot.Test.Admitted

open Lean (Json)
open Spot.OneStageZk

/-- C_S of `E16`'s unit 0, changed by `f`, under the query `q`. -/
def admitS (q : Json) (f : Flock.Program.Prog → Flock.Program.Prog := id) : Flock.V Admit := do
  let S ← Spot.Restrict.restrict (← Flock.Program.decode E16.toUTF8) #[0]
  admitted (Json.mkObj [("query", q)]) [] (f S)

/-- The query listing `E16`'s template. -/
def QE : Json := Json.mkObj [("name", "Q_template_instance"), ("params", Json.mkObj [("template", "E_v1")])]

/-- Whether `r` is refused with exactly `why`. -/
def refusedWith {α : Type} (r : Flock.V α) (why : String) : Bool :=
  match r with | .error e => e == why | .ok _ => false

-- a partition whose query is no template query: not derived, before C_S is read
#guard refusedWith (admitS (Json.mkObj [("name", "Q_all")]))
  (Flock.OneCircuit.notDerived
    "lower : C_S of a partition that is no template query (Q_template_instance, Q_template_instances) -> its Boolean block circuit")
-- C_S's root past its Definitions
#guard refusedBy (admitS QE fun S => { S with root := S.defs.size }) "one-template"
  "C_S's root is not one of its Definitions"
-- C_S's index without its template, then its index naming a Definition past its Definitions
#guard refusedBy (admitS QE fun S => { S with index := S.index.erase "E_v1" }) "listed"
  "C_S's template E_v1 has no entry in C_S's index"
#guard refusedBy (admitS QE fun S => { S with index := S.index.insert "E_v1" S.defs.size }) "listed"
  "C_S's template E_v1 is no Definition of C_S"
-- a lift that names no Definition of its program
#guard refusedBy (admitWith E16 (lift := fun S _ => pure (S, S.defs.size))) "leaves-are-bits"
  "C_S's template E_v1 has no lift to bits"

/-- `admittedBlock` on `AB16`'s units `us` at `A_v0`, C changed by `f`. -/
def formedP (us : Array Nat) (f : Flock.Program.Prog → Flock.Program.Prog) : Flock.V Unit := do
  admittedBlock (f (← Flock.Program.decode AB16.toUTF8)) us "A_v0"

-- C's root past its Definitions
#guard refusedBy (formedP #[0] fun P => { P with root := P.defs.size }) "no-recut" "C's root is not one of its Definitions"
-- unit 1's node (`B_v0`'s batch, root node 3) placed at gate 40, inside unit 0's gates [32, 48)
#guard refusedBy (formedP #[0, 1] fun P => { P with defs := P.defs.modify P.root fun d =>
    { d with body := { d.body with offsets := d.body.offsets.set! 3 40 } } })
  "disjoint-ranges" "units 0 and 1 of the block hold gates in common"

end Spot.Test.Admitted
