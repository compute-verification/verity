import Proofs.Flock.Soundness.ExecTyped
import Proofs.Flock.Soundness.ExecTemplatePlace

/-!
# Satisfaction off a set of rows

A statement at public inputs (`HmIn.delta`) drops Δ's pairs into the unit slots' public leaf rows, so a satisfying
witness of it satisfies its flat twin at every row but those. The row nets' facts (`Hm/Flat`, `Hidden/Wide`) read rows
of the `sha512x3` and `hm96` slots, the pin and the regions only, so they hold at:

* `SatEx S D z`: every row of `S` off `D` holds in every block (`RowHolds`), the pin is 1 and the regions hold; a
  satisfying witness satisfies it at every `D` (`Satisfies.ex`);
* `UnitIn c x`: `x` is an input row of a unit slot, the rows `SatEx` leaves out at public inputs;
* `CopyRow.block_eq_ex`, `ZeroRow.block_eq_ex`: `CopyRow.block_eq` and `ZeroRow.block_eq` at a row off `D`.
-/

open Flock FlockLevel3

namespace FlockSoundness

/-- **Row `i`'s constraint holds in block `o`.** -/
abbrev RowHolds (S : Model.Statement) (z : Model.Witness S.m) (o : Fin (2 ^ S.nbl)) (i : Fin (2 ^ S.kLog)) : Prop :=
  (S.A₀.mulVec (S.block z o)) i * (S.B₀.mulVec (S.block z o)) i = S.block z o i

/-- **`Satisfies` off the rows `D`**: every row off `D` holds in every block, the pin is 1 and the regions hold. -/
def SatEx (S : Model.Statement) (D : ℕ → Prop) (z : Model.Witness S.m) : Prop :=
  (∀ o i, ¬ D i.val → RowHolds S z o i)
    ∧ (∀ o, S.block z o S.pin = 1)
    ∧ ∀ R ∈ S.regions, ∀ (o : Fin (2 ^ S.nbl)) (i : ℕ) (_hi : i < 2 ^ R.free.length),
        ∀ h : R.inBlock i < 2 ^ S.kLog, S.block z o ⟨R.inBlock i, h⟩ = R.bits o i

theorem Model.Statement.Satisfies.ex {S : Model.Statement} {z : Model.Witness S.m} (hz : S.Satisfies z)
    (D : ℕ → Prop) : SatEx S D z :=
  ⟨fun o i _ => hz.1 o i, hz.2.1, hz.2.2⟩

/-- `SatEx` off no row is `Satisfies`. -/
theorem SatEx.sat {S : Model.Statement} {z : Model.Witness S.m} (hz : SatEx S (fun _ => False) z) : S.Satisfies z :=
  ⟨fun o i => hz.1 o i not_false, hz.2.1, hz.2.2⟩

/-- **An input row of a unit slot**: column `off < inWords · WORD` of a slot of the unit net's range. -/
def UnitIn (c : Circuit) (x : ℕ) : Prop :=
  ∃ s off, AtSlot c c.unitNet s off x ∧ off < c.unit.inWords * WORD

section rows

variable {S : Model.Statement} {D : ℕ → Prop} {z : Model.Witness S.m}

/-- `CopyRow.block_eq` at a row off `D`. -/
theorem CopyRow.block_eq_ex {r src : Fin (2 ^ S.kLog)} (h : CopyRow S r src) (hz : SatEx S D z) (hr : ¬ D r.val)
    (o : Fin (2 ^ S.nbl)) : S.block z o r = S.block z o src := by
  have hr := hz.1 o r hr
  unfold RowHolds at hr
  rw [mulVec_single S.A₀ h.1, mulVec_single S.B₀ h.2] at hr
  rw [← hr]
  generalize S.block z o src = x
  fin_cases x <;> rfl

/-- `ZeroRow.block_eq` at a row off `D`. -/
theorem ZeroRow.block_eq_ex {r : Fin (2 ^ S.kLog)} (h : ZeroRow S r) (hz : SatEx S D z) (hr : ¬ D r.val)
    (o : Fin (2 ^ S.nbl)) : S.block z o r = 0 := by
  have hr := hz.1 o r hr
  unfold RowHolds at hr
  rw [mulVec_zero_row S.A₀ h.1, mulVec_zero_row S.B₀ h.2, zero_mul] at hr
  exact hr.symm

end rows

end FlockSoundness
