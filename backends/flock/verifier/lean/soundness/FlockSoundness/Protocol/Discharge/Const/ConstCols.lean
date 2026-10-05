import FlockSoundness.Protocol.Types.Program
import FlockSoundness.Protocol.Discharge.Const.Inst

/-!
# `hConst`: only a unit's constant column sits on the constant's gate

The headline's `hConst : (p.partition outs).ConstCols plan tab pp.derived.place ones` reads `pp` only through each unit's
instance: `(pp.derived.place S R u h).inst` is `pp.inst u` and `.R` is `p.rows u`. Neither the tables nor the statement
the verifier accepted enter. It holds at `ones = {one}` when every unit's instance wires its constant column to `one`
(`constCols_of_oneWire`): an instance shares a gate only between input columns, so no other column of a unit sits there
(`Partition.constCols_singleton`). Two facts about how `p` and `pp` are built give that:
* every unit's constant is on `one` (`Prog.OneGate p one`), which a program of units over circuit types has at its
  constant input (`UProg.toRows_oneGate`);
* `pp`'s instances wire each unit's constant where the program does, which the program's own instance does
  (`Prog.instC`, `Prog.instC_wire_one`). An instance chosen from `Prog.isRowsUnit` need not (`Inst.lean`).

`constCols_of_instC` is the form the headline takes. `{one}` is exactly the set of gates the units' constants read
(`image_oneWire`), as the certificate's `h1` needs at the committed values; a set holding another gate a unit's column
reads would make `ConstCols` false.
-/

namespace FlockSoundness.Discharge.Const

open FlockSoundness Audit

section places

variable {N n : ℕ} {p : FlockSoundness.Prog N n} {outs : Finset (Fin N)} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

/-- **`hConst` at a program's places, for one constant gate**: when each unit's instance wires its constant column to
`one`, only a unit's constant column sits on `one`. -/
theorem constCols_of_oneWire (pp : ProgPlaces p outs plan tab) (one : Fin N)
    (h : ∀ u, (pp.inst u).wire (p.rows u).one = one) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  (p.partition outs).constCols_singleton plan tab _ one fun _ _ u _ => h u

/-- **`hConst` at a program's places, from the program**: when every unit's constant is on `one` and the places'
instances wire each unit's constant where the program does. -/
theorem constCols_of_oneGate (pp : ProgPlaces p outs plan tab) {one : Fin N} (hp : Prog.OneGate p one)
    (hi : ∀ u, (pp.inst u).wire (p.rows u).one = Prog.oneAt p u) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  constCols_of_oneWire pp one fun u => (hi u).trans (hp u)

/-- **`hConst` at a program's places, with the program's own instances**: when every unit's constant is on `one` and
the places carry `Prog.instC`. -/
theorem constCols_of_instC (pp : ProgPlaces p outs plan tab) {one : Fin N} (hp : Prog.OneGate p one)
    (hinst : ∀ u, pp.inst u = Prog.instC p outs u) :
    (p.partition outs).ConstCols plan tab pp.derived.place {one} :=
  constCols_of_oneGate pp hp fun u => by rw [hinst u]; exact Prog.instC_wire_one p outs u

end places

/-! ## A program of units over circuit types -/

namespace UProg

variable {k one : ℕ} (hk : one < k)

end UProg

end FlockSoundness.Discharge.Const
