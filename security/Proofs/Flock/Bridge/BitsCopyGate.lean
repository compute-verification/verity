module

public import Proofs.Flock.Bridge.UnitRows
public import Proofs.Flock.Bridge.BitsKeyProg
import Proofs.Flock.Soundness.Discharge.Placed.Keyed

/-!
# A copy's gate at a column is Bits' gate there

`copyGate D hU n u hj c` is the gate of the circuit `D` describes on column `c` of copy `j` of unit `u` (`UnitRows`).
`Spot.Bits.inGate` and `outGate` give a gate as a number, from a unit, an instance and a bit of an input port or of a
returned output. They are one gate:

* `copyGate_inGate`: the gate `Spot.Bits.inGate` gives on bit `t` of input port `p` of instance `j` of unit `u` is
  the copy's gate on the port's first column plus `t`, an input column.
* `copyGate_outGate`: the gate `Spot.Bits.outGate` gives on bit `t` of returned output `i` of instance `j` of unit `u`
  is the copy's gate on the output's column (`Spot.Bits.outCol`), a computed one (`outGate_col_lt`).

`unitInst_wire_input` and `unitInst_wire_comp` are a unit's wires in closed form: an input column on the gate the
program wires it to (`Const.Prog.insAt`), a computed column on its row's gate.
No verifier path reads this file.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge

variable (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ)

/-- **A unit's input column is on the gate the program wires it to** (`Const.Prog.insAt`). -/
theorem unitInst_wire_input (u : Fin n) (c : Fin (Placed.circuitRows D.unroll hU).w) {x : ℕ} (hx : c.val = x)
    (hc : x < (Placed.circuitRows D.unroll hU).nIn) :
    ((unitInst D hU n u).wire c).val = (Const.Prog.insAt (Placed.keyProg D.unroll hU n) u x).val := by
  subst hx
  exact (congrArg Fin.val (unitInst_wire D hU n u c)).trans (congrArg Fin.val
    (Const.Prog.rowsWireC_wire_input (Placed.keyProg D.unroll hU n) u
      (Fin.cast (congrArg Rows.w (rows_keyProg D hU n u)).symm c)
      (lt_of_lt_of_eq hc (congrArg Rows.nIn (rows_keyProg D hU n u)).symm)))

/-- **A unit's computed column `x` is on its row's gate**, `2 + n·B + u·nComp + (x − B)`, `B` the unit's inputs. -/
theorem unitInst_wire_comp (u : Fin n) (c : Fin (Placed.circuitRows D.unroll hU).w) {x : ℕ} (hx : c.val = x)
    (h1 : (Placed.circuitRows D.unroll hU).nIn ≤ x)
    (h2 : x < (Placed.circuitRows D.unroll hU).nIn + (Placed.circuitRows D.unroll hU).nComp) :
    ((unitInst D hU n u).wire c).val = 2 + n * (Placed.circuitRows D.unroll hU).nIn +
      u.val * (Placed.circuitRows D.unroll hU).nComp + (x - (Placed.circuitRows D.unroll hU).nIn) := by
  subst hx
  exact (congrArg Fin.val (unitInst_wire D hU n u c)).trans
    (rowsWireC_keyProg_comp (c := D.unroll) (hU := hU) u
      (Fin.cast (congrArg Rows.w (rows_keyProg D hU n u)).symm c) h1 h2)

/-- **Bits' gate on an input port's bit is the copy's gate on the bit's column.** Where `Spot.Bits.inGate` gives the
gate `g` on bit `t` of input port `p` of instance `j` of unit `u`, the copy's gate on the input column `col`, the
port's first column plus `t`, is `g`. -/
theorem copyGate_inGate (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (col : Fin (Rows.ofNet hU).w)
    (hc : col.val < (Rows.ofNet hU).nIn) {p t g : ℕ}
    (hcol : col.val = ((Flock.portCols D.unroll.unit.inGroups).getD p default).1 + t)
    (h : Spot.Bits.inGate D.unroll n u.val j p t = some g) : (copyGate D hU n u hj col).val = g := by
  have hxB : j * (Rows.ofNet hU).nIn + col.val < (Placed.circuitRows D.unroll hU).nIn := Rows.mul_add_lt hj hc
  have hsc : ((Rows.ofNet hU).stackCol D.unroll.upv hj col).val = j * (Rows.ofNet hU).nIn + col.val := by
    rw [Rows.stackCol, dite_eq_left hc]
  have hw : ((unitInst D hU n u).wire ((Rows.ofNet hU).stackCol D.unroll.upv hj col)).val =
      (Const.Prog.insAt (Placed.keyProg D.unroll hU n) u (j * (Rows.ofNet hU).nIn + col.val)).val :=
    unitInst_wire_input D hU n u _ hsc hxB
  have hv : (Const.Prog.insAt (Placed.keyProg D.unroll hU n) u (j * (Rows.ofNet hU).nIn + col.val)).val =
      (Placed.insOf n (Placed.circuitRows D.unroll hU).nIn
        (Aliased.keyIdx D.unroll (Placed.circuitRows D.unroll hU).nIn) u.val
        ⟨j * (Rows.ofNet hU).nIn + col.val, hxB⟩).val :=
    Placed.insAt_uniform n u _ hxB
  rw [Placed.insOf_of_lt (nU := n) (B := (Placed.circuitRows D.unroll hU).nIn)
    (src := Aliased.keyIdx D.unroll (Placed.circuitRows D.unroll hU).nIn) u.isLt] at hv
  unfold Spot.Bits.inGate at h
  obtain ⟨-, h⟩ := Option.ite_none_right_eq_some.1 h
  obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.1 h
  have hk : Spot.Bits.keyIdx D.unroll (Placed.circuitRows D.unroll hU).nIn
      (j * (Rows.ofNet hU).nIn + col.val) = some b :=
    (congrArg (fun x => Spot.Bits.keyIdx D.unroll (Placed.circuitRows D.unroll hU).nIn
      (j * (Rows.ofNet hU).nIn + x)) hcol).trans hb
  obtain ⟨b', hb', rfl⟩ := Option.map_eq_some_iff.1 ((keyIdx_eq D.unroll hxB).symm.trans hk)
  rw [hb'] at hv
  exact hw.trans hv

/-- Where `copyGate_outGate`'s rows sit: with `cI` and `cM` the inputs and the rows of `k` stacked copies of rows of
`rN` inputs and `rC` computed rows, copy `j`'s computed column `x` is a row of the stack, the one `j·(rC + 1) + 1 +
(x − rN)` rows past the stack's inputs. -/
theorem outGate_arith {n u j k rN rC x cI cM : ℕ} (hI : cI = k * rN) (hM : cM = k * (rC + 1)) (hj : j < k)
    (hlo : rN ≤ x) (hhi : x < rN + rC) :
    cI ≤ k * rN + (j * (rC + 1) + (1 + (x - rN))) ∧
      k * rN + (j * (rC + 1) + (1 + (x - rN))) < cI + cM ∧
      2 + n * cI + u * cM + (k * rN + (j * (rC + 1) + (1 + (x - rN))) - cI) =
        2 + n * cI + u * cM + (j * (rC + 1) + (1 + (x - rN))) := by
  subst hI hM
  have hjM := Nat.mul_le_mul_right (rC + 1) (show j + 1 ≤ k from hj)
  rw [Nat.succ_mul] at hjM
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- **A returned output's bit that has a gate is on a computed column** of the unit net's rows. -/
theorem outGate_col_lt {u j i t g : ℕ} (h : Spot.Bits.outGate D.unroll n u j i t = some g) :
    Spot.Bits.outCol D.unroll i t < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp :=
  ((outGate_eq_some D.unroll n).1 h).2.2.2.2.2.1

/-- **Bits' gate on a returned output's bit is the copy's gate on the output's column.** Where `Spot.Bits.outGate`
gives the gate `g` on bit `t` of returned output `i` of instance `j` of unit `u`, the copy's gate on the output's
column (`Spot.Bits.outCol`) is `g`. -/
theorem copyGate_outGate (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (col : Fin (Rows.ofNet hU).w) {i t g : ℕ}
    (hcol : col.val = Spot.Bits.outCol D.unroll i t)
    (h : Spot.Bits.outGate D.unroll n u.val j i t = some g) : (copyGate D hU n u hj col).val = g := by
  obtain ⟨-, -, -, -, hlo, hhi, rfl⟩ := (outGate_eq_some D.unroll n).1 h
  have hlo' : (Rows.ofNet hU).nIn ≤ col.val := le_of_le_of_eq hlo hcol.symm
  have hhi' : col.val < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp := lt_of_eq_of_lt hcol hhi
  obtain ⟨h1, h2, h3⟩ := outGate_arith (n := n) (u := u.val) (j := j) (k := D.unroll.upv)
    (rN := (Rows.ofNet hU).nIn) (rC := (Rows.ofNet hU).nComp) (x := col.val)
    (cI := (Placed.circuitRows D.unroll hU).nIn) (cM := (Placed.circuitRows D.unroll hU).nComp) rfl rfl hj hlo' hhi'
  have hsc : ((Rows.ofNet hU).stackCol D.unroll.upv hj col).val = D.unroll.upv * (Rows.ofNet hU).nIn +
      (j * ((Rows.ofNet hU).nComp + 1) + (1 + (col.val - (Rows.ofNet hU).nIn))) := by
    unfold Rows.stackCol
    rw [dite_eq_right (show ¬ col.val < (Rows.ofNet hU).nIn by omega), dite_eq_left hhi']
  have hw : ((unitInst D hU n u).wire ((Rows.ofNet hU).stackCol D.unroll.upv hj col)).val =
      2 + n * (Placed.circuitRows D.unroll hU).nIn + u.val * (Placed.circuitRows D.unroll hU).nComp +
        (j * ((Rows.ofNet hU).nComp + 1) + (1 + (col.val - (Rows.ofNet hU).nIn))) :=
    (unitInst_wire_comp D hU n u _ hsc h1 h2).trans h3
  exact hw.trans (congrArg (fun x => 2 + n * (Placed.circuitRows D.unroll hU).nIn +
    u.val * (Placed.circuitRows D.unroll hU).nComp +
      (j * ((Rows.ofNet hU).nComp + 1) + (1 + (x - (Rows.ofNet hU).nIn)))) hcol)

end Flock.Bridge
