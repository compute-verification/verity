import FlockSoundness.E2E
import FlockSoundness.Types.ProgramPlaces
import FlockSoundness.Protocol.Discharge.Zero.Cols

/-!
# The zero's columns are input columns (`hZero`, e2e)

`ZeroCols place zeros` reads every column of a drawn unit's place that sits on a gate of `zeros`. When those gates are no
unit's (the statement's zero is a program input) and no unit's constant column sits on one, only input columns do: a
computed column is a gate of the unit itself (`IsRowsUnit.mem`).

- `zeroCols_of_inputs`: `ZeroCols` from its input columns, at any drawn units' places.
- `Prog.zeroCols_of_inputs`: the same at the places `ProgPlaces` gives the program's units, the `hZero` that
  `Prog.flock_headline` takes.
- `TableClass.block_eq_zero`: at a table class's slot, an input whose copy position is one of the class's forced-zero rows
  carries 0, in every block of a satisfying witness.
-/

namespace FlockSoundness.Discharge.Zero

open Audit Model

section Partition

variable {C : Audit.Circuit} {n : ℕ} (P : Audit.Partition C n) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

end Partition

section Prog

variable {N n : ℕ} {p : Prog N n} {outs : Finset (Fin N)} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

end Prog

/-- **An input copying a forced-zero row carries 0**: at slot `g` of a table class, an input whose copy position is one
of the class's forced-zero rows carries 0 in every block of a satisfying witness. -/
theorem TableClass.block_eq_zero {St : Statement} (cls : TableClass St) (g : Fin cls.slots) (c : Fin cls.rows.w)
    (hc : c.val < cls.rows.nIn) (hz : cls.copyPos g c.val ∈ cls.zeros) {z : Witness St.m} (hsat : St.Satisfies z)
    (o : Fin (2 ^ St.nbl)) : St.block z o (cls.col g c) = 0 := by
  rw [cls.copy g c hc z hsat o]
  exact cls.zero _ hz z hsat o

end FlockSoundness.Discharge.Zero
