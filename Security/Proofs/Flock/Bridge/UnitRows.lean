module

public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.VBridge.Unit
import Proofs.Core.Circuit
import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.Discharge.Placed.Wiring

/-!
# A unit of the parsed file holds its template's rows at its local evaluation

Unit `u` of the key program is an instance of the circuit's template instance rows, `units_per_vu` stacked copies of the
unit net's (`unitInst`), wired as the program wires it (`Const.Prog.rowsWireC`, the wiring `Layout.outsOf` reads;
`unitInst_wire`). In core's view of the parsed file (`circuitOf`, `partsOf`), copy `j` of every unit holds the unit
net's rows (`Rows.ofNet`) at the unit's local evaluation (`rows_localEval`): at column values that are the evaluation's
bits at every column of the copy (`copyGate`), with the constant the transcript's bit at the stack's constant
(`constGate`), a gate outside the unit. A correct unit's committed gates hold that evaluation (`Partition.Correct`).
No verifier path reads this file.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

theorem castW_wire {N m : ℕ} {q : FlockSoundness.Prog N m} {u : Fin m} {R R' : Rows} (e : R = R')
    (W : q.RowsWire u R) (c : Fin R'.w) :
    (Const.Prog.castW e W).wire c = W.wire (Fin.cast (congrArg Rows.w e).symm c) := by
  cases e
  rfl

/-- Every unit of the key program has the circuit's template instance rows. -/
theorem rows_keyProg (u : Fin n) : (Placed.keyProg D.unroll hU n).rows u = Placed.circuitRows D.unroll hU :=
  Placed.rows_circuitProg _ _ _ _ _ u

/-- **Unit `u` of the key program as an instance of its rows**: the circuit's template instance rows, wired as the
program wires them. -/
noncomputable def unitInst (u : Fin n) : (keyPartition D hU n).IsRowsUnit u (Placed.circuitRows D.unroll hU) :=
  (Const.Prog.castW (rows_keyProg D hU n u) (Const.Prog.rowsWireC (Placed.keyProg D.unroll hU n) u)).toIsRowsUnit _

/-- `unitInst`'s wires are the program's own (`Const.Prog.rowsWireC`). -/
theorem unitInst_wire (u : Fin n) (c : Fin (Placed.circuitRows D.unroll hU).w) :
    (unitInst D hU n u).wire c = (Const.Prog.rowsWireC (Placed.keyProg D.unroll hU n) u).wire
      (Fin.cast (congrArg Rows.w (rows_keyProg D hU n u)).symm c) :=
  castW_wire _ _ c

/-- **Copy `j`'s column `c`'s gate** in unit `u`: the gate on the stack's column `c` of copy `j`. -/
noncomputable def copyGate (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (c : Fin (Rows.ofNet hU).w) :
    Fin (circuitOf D hU n).N :=
  (unitInst D hU n u).wire ((Rows.ofNet hU).stackCol D.unroll.upv hj c)

/-- **Unit `u`'s constant gate**: the gate on the stack's constant column, which every copy's constant row reads. -/
noncomputable def constGate (u : Fin n) : Fin (circuitOf D hU n).N :=
  (unitInst D hU n u).wire (Placed.circuitRows D.unroll hU).one

/-- **A copy's input column is a gate outside its unit.** -/
theorem copyGate_input (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (c : Fin (Rows.ofNet hU).w)
    (hc : c.val < (Rows.ofNet hU).nIn) : (partsOf D hU n).unit (copyGate D hU n u hj c) ≠ some u := by
  have hlt : ((Rows.ofNet hU).stackCol D.unroll.upv hj c).val < D.unroll.upv * (Rows.ofNet hU).nIn := by
    rw [Rows.stackCol, dite_eq_left hc]
    exact Rows.mul_add_lt hj hc
  have hn : (Placed.circuitRows D.unroll hU).nIn = D.unroll.upv * (Rows.ofNet hU).nIn := rfl
  exact (unitInst D hU n u).outside _ fun i h => by
    have h2 : ((Rows.ofNet hU).stackCol D.unroll.upv hj c).val = (Placed.circuitRows D.unroll hU).nIn + i.val :=
      congrArg Fin.val h
    omega

/-- `partsOf`'s local evaluation is the key program's, on the transcript's bits. -/
theorem localEval_bits (tr : (circuitOf D hU n).Transcript) (u : Fin n) (g : Fin (circuitOf D hU n).N) :
    ((partsOf D hU n).localEval tr u g).getLsbD 0 =
      (keyPartition D hU n).localEval (fun g => (tr g).getLsbD 0) u g := by
  have hXs : tr = Verity.Circuit.ofBits (fun g => (tr g).getLsbD 0) := by
    funext g
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    obtain rfl : i = 0 := by change i < 1 at hi; omega
    show (tr g).getLsbD 0 = (BitVec.ofBool ((tr g).getLsbD 0)).getLsbD 0
    generalize (tr g).getLsbD 0 = b
    cases b <;> rfl
  show ((Verity.Partition.ofAudit (keyPartition D hU n)).localEval tr u g).getLsbD 0 = _
  conv_lhs => rw [hXs]
  rw [Verity.Partition.localEval_ofAudit]
  simp

/-- **A unit's copy holds the unit net's rows at the unit's local evaluation.** At unit `u` of `partsOf`, core's view
of the parsed file, every transcript `tr` gives, for each copy `j < units_per_vu` of the unit's stack, column values
`z` that satisfy the unit net's rows (`Rows.ofNet`), whose constant is `tr`'s bit at the unit's constant gate, and
which are the bits of `u`'s local evaluation of `tr` at every column of the copy. (`X` is a token here: ArkLib's global
`notation3 "X"` comes in through `CircuitOf`'s imports.) -/
theorem rows_localEval (tr : (circuitOf D hU n).Transcript) (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) :
    ∃ z : ℕ → Bool, (Rows.ofNet hU).Sat (fun c => z c.val) ∧
      z (Rows.ofNet hU).one.val = (tr (constGate D hU n u)).getLsbD 0 ∧
      ∀ c : Fin (Rows.ofNet hU).w, z c.val = ((partsOf D hU n).localEval tr u (copyGate D hU n u hj c)).getLsbD 0 := by
  obtain ⟨hs, hone⟩ := FlockVBridge.sat_copy (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u)
    (fun g => (tr g).getLsbD 0) hj
  refine ⟨FlockVBridge.copyVal (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u)
    (fun g => (tr g).getLsbD 0) hj, ?_, ?_, fun c => ?_⟩
  · intro i
    simp only [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
    exact hs i
  · simp only [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
    exact hone
  · simp only [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
    exact (localEval_bits D hU n tr u _).symm

end Flock.Bridge
