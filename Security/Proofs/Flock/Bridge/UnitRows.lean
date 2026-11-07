module

public import Proofs.Flock.Bridge.CircuitOf
public import Proofs.Flock.Bridge.Statement
public import Proofs.Flock.Soundness.Discharge.Const.Inst
public import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
public import Proofs.Flock.VBridge.Unit
import Proofs.Core.Circuit
import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Flock.Soundness.Discharge.Zero.Uniform

/-!
# A correct unit of the parsed file holds its template's rows

Unit `u` of the key program is an instance of the circuit's template instance rows, `units_per_vu` stacked copies of the
unit net's (`unitInst`), wired as the program wires it (`Const.Prog.rowsWireC`, the wiring `Layout.outsOf` reads;
`unitInst_wire`). In core's view of the parsed file (`circuitOf`, `partsOf`), copy `j` of a correct unit holds the unit
net's rows (`Rows.ofNet`) on the transcript's bits (`rows_of_correct`): at column values that are the transcript's bits
at every input column and every committed column (`copyGate`), with the constant the transcript's bit at the stack's
constant (`constGate`). The values are the unit's local evaluation; the converse direction is `correct_of_progCorrect`.
Where those gates sit: every unit's constant gate is the circuit's constant (`constGate_eq`), opened at the unit
(`constGate_mem_io`), and a copy's gate on a returned output's column is committed (`copyGate_mem_committed`).
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

/-- `partsOf`'s correct units are the key program's, on the transcript's bits. -/
theorem correct_bits {u : Fin n} {tr : (circuitOf D hU n).Transcript} (hX : (partsOf D hU n).Correct tr u) :
    (keyPartition D hU n).Correct (fun g => (tr g).getLsbD 0) u := by
  have hXs : tr = Verity.Circuit.ofBits (fun g => (tr g).getLsbD 0) := by
    funext g
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    obtain rfl : i = 0 := by change i < 1 at hi; omega
    show (tr g).getLsbD 0 = (BitVec.ofBool ((tr g).getLsbD 0)).getLsbD 0
    generalize (tr g).getLsbD 0 = b
    cases b <;> rfl
  have h1 : u ∉ (partsOf D hU n).wrong tr := by
    rw [Verity.Partition.mem_wrong]
    exact not_not.mpr hX
  rw [hXs, partsOf_wrong, FlockSoundness.Audit.Partition.mem_wrong] at h1
  exact not_not.mp h1

/-- **A correct unit's copy holds the unit net's rows on the transcript's bits.** At unit `u` of `partsOf`, core's
view of the parsed file, a transcript `tr` on which `u` is correct gives, for each copy `j < units_per_vu` of the
unit's stack, column values `z` that satisfy the unit net's rows (`Rows.ofNet`), whose constant is `tr`'s bit at the
unit's constant gate, and which are `tr`'s bits at every input column of the copy and at every column whose gate is
committed. (`X` is a token here: ArkLib's global `notation3 "X"` comes in through `CircuitOf`'s imports.) -/
theorem rows_of_correct {u : Fin n} {tr : (circuitOf D hU n).Transcript} (hX : (partsOf D hU n).Correct tr u)
    {j : ℕ} (hj : j < D.unroll.upv) :
    ∃ z : ℕ → Bool, (Rows.ofNet hU).Sat (fun c => z c.val) ∧
      z (Rows.ofNet hU).one.val = (tr (constGate D hU n u)).getLsbD 0 ∧
      ∀ c : Fin (Rows.ofNet hU).w,
        (c.val < (Rows.ofNet hU).nIn ∨ copyGate D hU n u hj c ∈ (partsOf D hU n).committed) →
          z c.val = (tr (copyGate D hU n u hj c)).getLsbD 0 := by
  have hc := correct_bits D hU n hX
  obtain ⟨hs, hone⟩ := FlockVBridge.sat_copy (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u)
    (fun g => (tr g).getLsbD 0) hj
  refine ⟨FlockVBridge.copyVal (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u)
    (fun g => (tr g).getLsbD 0) hj, ?_, ?_, fun c hcm => ?_⟩
  · intro i
    simp only [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
    exact hs i
  · simp only [FlockVBridge.copyVal, FlockVBridge.colVal_fin]
    exact hone
  · rcases hcm with hin | hm
    · exact FlockVBridge.copyVal_input (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u) _ hj hin
    · exact FlockVBridge.copyVal_committed (R := Rows.ofNet hU) (k := D.unroll.upv) (unitInst D hU n u) hc hj c
        ((Verity.Partition.mem_committed_ofAudit _ _).1 hm)

/-- Unit `u`'s constant gate is the gate the program wires its constant to (`Const.Prog.oneAt`). -/
theorem constGate_eq_oneAt (u : Fin n) :
    constGate D hU n u = Const.Prog.oneAt (Placed.keyProg D.unroll hU n) u := by
  rw [constGate, unitInst_wire, ← Const.Prog.rowsWireC_wire_one]
  congr 1
  exact Fin.ext (by simp [Rows.one, rows_keyProg D hU n u])

/-- Unit `u`'s constant gate is gate 0. -/
theorem constGate_val (u : Fin n) : (constGate D hU n u).val = 0 := by
  rw [constGate_eq_oneAt]
  exact Zero.oneAt_uniform n u

/-- **Unit `u`'s constant gate is the circuit's constant** (`Layout.oneG`), at its index. -/
theorem constGate_eq_oneG (u : Fin n) :
    (constGate D hU n u).val = (Layout.oneG (c := D.unroll) (hU := hU) (n := n)).val :=
  constGate_val D hU n u

/-- **Unit `u`'s constant gate is the circuit's constant** (`Layout.oneG`). -/
theorem constGate_eq (u : Fin n) :
    @Eq (Fin (Layout.keyN D.unroll hU n)) (constGate D hU n u) Layout.oneG :=
  Fin.ext (constGate_eq_oneG D hU n u)

/-- The constant wires hold unit `u`'s constant gate at 1. -/
theorem constWiresOf_constGate (u : Fin n) : constWiresOf D.unroll hU n (constGate D hU n u) = some true := by
  rw [constGate_eq D hU n u]
  simp [constWiresOf]

/-- The circuit's constants hold unit `u`'s constant gate at 1. -/
theorem constsOf_constGate (u : Fin n) : constsOf D.unroll hU n (constGate D hU n u) = some true := by
  rw [constGate_eq D hU n u]
  simp [constsOf]

/-- **Unit `u`'s constant gate is among its openings** (`Partition.io`): it is an input of the circuit, so committed,
and copy 0's constant row reads it. -/
theorem constGate_mem_io (u : Fin n) (hupv : 0 < D.unroll.upv) : constGate D hU n u ∈ (partsOf D hU n).io u := by
  have h0 : 0 < (Placed.circuitRows D.unroll hU).nComp := Nat.mul_pos hupv (Nat.succ_pos _)
  have hnone : (partsOf D hU n).unit (constGate D hU n u) = none := by
    show (Placed.keyProg D.unroll hU n).unit (constGate D hU n u) = none
    exact Layout.unit_uniform_input n _ (by rw [constGate_val]; show 0 < 2 + _; omega)
  have hcom : constGate D hU n u ∈ (partsOf D hU n).committed :=
    Verity.Partition.mem_committed_of_input _ (((partsOf D hU n).input_iff _).1 hnone)
  have hrow : (Placed.circuitRows D.unroll hU).one ∈ (Placed.circuitRows D.unroll hU).a ⟨0, h0⟩ := by
    show _ ∈ (Rows.ofNet hU).stackRow D.unroll.upv (Rows.ofNet hU).a ⟨0, h0⟩
    unfold Rows.stackRow
    split
    · exact List.mem_singleton.2 rfl
    · rename_i h
      exact absurd (Nat.zero_mod _) h
  have hread : ∀ g : Fin (keyCircuit D hU n).N, g = constGate D hU n u → g ∈ (coreOp ((keyCircuit D hU n).op
      ((unitInst D hU n u).wire ((Placed.circuitRows D.unroll hU).comp ⟨0, h0⟩)))).args.toFinset := by
    intro g hg
    rw [(unitInst D hU n u).op ⟨0, h0⟩]
    simp only [coreOp, Audit.Op.args, List.mem_toFinset, List.mem_append, List.mem_map]
    exact Or.inl ⟨_, hrow, hg.symm⟩
  rw [Verity.Partition.io, Finset.mem_filter]
  exact ⟨hcom, Or.inr ⟨_, (unitInst D hU n u).mem ⟨0, h0⟩, hread _ rfl⟩⟩

/-- **A copy's gate on a returned output's column is committed**: bit `t` of leaf `jj` of copy `j`'s returned outputs
(`Layout.outCol`) is a gate of the circuit's outputs (`Layout.outsOf`). -/
theorem copyGate_mem_committed (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (c : Fin (Rows.ofNet hU).w) {jj t : ℕ}
    (hjj : jj < (D.unroll.leavesOut.getD j #[]).size) (ht : t < 16) (hc : c.val = Layout.outCol D.unroll jj t)
    (hlo : (Rows.ofNet hU).nIn ≤ c.val) (hhi : c.val < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp) :
    copyGate D hU n u hj c ∈ (partsOf D hU n).committed := by
  have hx : ((Rows.ofNet hU).stackCol D.unroll.upv hj c).val = D.unroll.upv * (Rows.ofNet hU).nIn +
      (j * ((Rows.ofNet hU).nComp + 1) + (1 + (Layout.outCol D.unroll jj t - (Rows.ofNet hU).nIn))) := by
    unfold Rows.stackCol
    rw [dite_eq_right (show ¬ c.val < (Rows.ofNet hU).nIn by omega), dite_eq_left hhi, ← hc]
  have hout : ∀ g : Fin (Layout.keyN D.unroll hU n), g = copyGate D hU n u hj c →
      g ∈ Layout.outsOf D.unroll hU n := by
    intro g hg
    unfold Layout.outsOf
    refine (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2 ?_
    exact ⟨Finset.mem_univ _, u, Fin.cast (congrArg Rows.w (rows_keyProg D hU n u)).symm
      ((Rows.ofNet hU).stackCol D.unroll.upv hj c), (unitInst_wire D hU n u _).symm.trans hg.symm, j, jj, t, hj,
      hjj, ht, hc ▸ hlo, hc ▸ hhi, hx⟩
  exact Verity.Partition.mem_committed_of_output _ (hout _ rfl)

end Flock.Bridge
