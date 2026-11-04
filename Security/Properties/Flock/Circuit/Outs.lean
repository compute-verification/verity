import Properties.Flock.Circuit.Keyed

/-!
# The program's own wiring of each unit, and the returned outputs' bits

`Const.Prog.rowsWireC p u`: unit `u`'s rows wired as `p` wires them. `Layout.outsOf c hU n`: the gates `keyProg`'s
units put on returned outputs' columns (`OutColAt`), the outputs of the circuit a registered circuit file describes.
Moved unchanged from `Proofs.Flock.Soundness.Discharge.Const.Inst` and
`Proofs.Flock.Soundness.Discharge.Layout.{KeyProg,OutPin,Outs}`; the lemmas about them stay there.
-/

namespace FlockSoundness.Discharge.Const.Prog

open FlockSoundness FlockSoundness.Prog

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

theorem rows_last : (snoc p R ins one hone).rows (Fin.last n₀) = R := by
  simp [FlockSoundness.Prog.rows]

theorem rows_castSucc (v : Fin n₀) : (snoc p R ins one hone).rows v.castSucc = p.rows v := by
  simp [FlockSoundness.Prog.rows, v.isLt]

end snoc

/-- A program-level instance, carried along an equality of rows. -/
def castW {N n : ℕ} {q : FlockSoundness.Prog N n} {u : Fin n} {R R' : Rows} (e : R = R') (W : q.RowsWire u R) :
    q.RowsWire u R' :=
  e ▸ W

/-- **Unit `u`'s rows, wired as the program wires them**: the last unit's by `Prog.rowsWire_last`, an earlier one's
lifted by `Prog.rowsWire_snoc`. -/
def rowsWireC : ∀ {N n : ℕ} (p : FlockSoundness.Prog N n) (u : Fin n), p.RowsWire u (p.rows u)
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R ins one hone, u =>
    Fin.lastCases (motive := fun u => (snoc p R ins one hone).RowsWire u ((snoc p R ins one hone).rows u))
      (castW (rows_last p R ins one hone).symm (rowsWire_last p R ins one hone))
      (fun v => castW (rows_castSucc p R ins one hone v).symm (rowsWire_snoc p R ins one hone (rowsWireC p v))) u

end FlockSoundness.Discharge.Const.Prog

namespace FlockSoundness.Discharge.Layout

open Flock Placed

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- The number of input columns of the circuit's template instance rows. -/
noncomputable abbrev keyB : ℕ := (Placed.circuitRows c hU).nIn

/-- The gates of `keyProg c hU n`. -/
noncomputable abbrev keyN : ℕ := Placed.gates (Placed.inGates n (keyB c hU)) (Placed.circuitRows c hU) n

/-- Column `16 j + t` of the unit's returned-output group, in the unit's rows. -/
def outCol (j t : ℕ) : ℕ := (c.unit.outGroups.getD 0 default).col * WORD + (16 * j + t)

/-- **Stacked column `x` of a template instance is a returned output's bit**: copy `u'`'s computed column `outCol j t`,
for output `j` of the copy's leaf and bit `t < 16`. -/
def OutColAt (x u' j t : ℕ) : Prop :=
  u' < c.upv ∧ j < (c.leavesOut.getD u' #[]).size ∧ t < 16 ∧ (Rows.ofNet hU).nIn ≤ outCol c j t ∧
    outCol c j t < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp ∧
    x = c.upv * (Rows.ofNet hU).nIn + (u' * ((Rows.ofNet hU).nComp + 1) + (1 + (outCol c j t - (Rows.ofNet hU).nIn)))

/-- **The returned outputs' bits**: the gates the units' instances put on returned outputs' columns (`OutColAt`). -/
noncomputable def outsOf : Finset (Fin (keyN c hU n)) :=
  @Finset.filter _ (fun g => ∃ u x, (Const.Prog.rowsWireC (keyProg c hU n) u).wire x = g ∧
    ∃ u' j t, OutColAt c hU x.val u' j t) (Classical.decPred _) Finset.univ

end FlockSoundness.Discharge.Layout
