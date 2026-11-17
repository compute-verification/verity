module

public import Properties.Flock.Circuit.Outs
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow

/-! A program's own instance of each unit's rows (`Prog.instC`), which wires each unit's constant where the program does; an instance chosen from `Prog.isRowsUnit` need not. -/

@[expose] public section

namespace FlockSoundness.Discharge.Const.Prog

open FlockSoundness FlockSoundness.Prog

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

theorem colWire_one : colWire R ins one R.one = Fin.castAdd R.nComp one := by
  apply Fin.ext
  rw [colWire_val]
  simp [Rows.one]

theorem colWire_input (c : Fin R.w) (hc : c.val < R.nIn) : colWire R ins one c = Fin.castAdd R.nComp (ins ⟨c.val, hc⟩) := by
  apply Fin.ext
  rw [colWire_val, dite_eq_left hc]
  rfl

end snoc

theorem castW_wire {N n : ℕ} {q : FlockSoundness.Prog N n} {u : Fin n} {R R' : Rows} (e : R = R')
    (W : q.RowsWire u R) (c : Fin R'.w) : (castW e W).wire c = W.wire (Fin.cast (by rw [e]) c) := by
  subst e
  rfl

theorem castW_wire_one {N n : ℕ} {q : FlockSoundness.Prog N n} {u : Fin n} {R R' : Rows} (e : R = R')
    (W : q.RowsWire u R) : (castW e W).wire R'.one = W.wire R.one := by
  subst e
  rfl

/-- **Unit `u`'s rows, wired as the program wires them**, as an instance in the program's circuit. -/
def instC {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n) :
    (p.partition outs).IsRowsUnit u (p.rows u) :=
  (rowsWireC p u).toIsRowsUnit outs

/-- The gate the program wires unit `u`'s constant to. -/
def oneAt : ∀ {N n : ℕ}, FlockSoundness.Prog N n → Fin n → Fin N
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R _ one _, u => Fin.lastCases (Fin.castAdd R.nComp one) (fun v => Fin.castAdd R.nComp (oneAt p v)) u

/-- The gate the program wires unit `u`'s input `i` to (its constant's gate past its inputs). -/
def insAt : ∀ {N n : ℕ}, FlockSoundness.Prog N n → Fin n → ℕ → Fin N
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R ins one _, u => Fin.lastCases
      (fun i => if h : i < R.nIn then Fin.castAdd R.nComp (ins ⟨i, h⟩) else Fin.castAdd R.nComp one)
      (fun v i => Fin.castAdd R.nComp (insAt p v i)) u

def OneGate {N n : ℕ} (p : FlockSoundness.Prog N n) (one : Fin N) : Prop := ∀ u, oneAt p u = one

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

theorem rowsWireC_last : rowsWireC (snoc p R ins one hone) (Fin.last n₀) =
    castW (rows_last p R ins one hone).symm (rowsWire_last p R ins one hone) := by
  rw [rowsWireC, Fin.lastCases_last]

theorem rowsWireC_castSucc (v : Fin n₀) : rowsWireC (snoc p R ins one hone) v.castSucc =
    castW (rows_castSucc p R ins one hone v).symm (rowsWire_snoc p R ins one hone (rowsWireC p v)) := by
  rw [rowsWireC, Fin.lastCases_castSucc]

theorem oneAt_last : oneAt (snoc p R ins one hone) (Fin.last n₀) = Fin.castAdd R.nComp one := by
  rw [oneAt, Fin.lastCases_last]

theorem oneAt_castSucc (v : Fin n₀) : oneAt (snoc p R ins one hone) v.castSucc = Fin.castAdd R.nComp (oneAt p v) := by
  rw [oneAt, Fin.lastCases_castSucc]

theorem insAt_last (i : ℕ) (hi : i < R.nIn) :
    insAt (snoc p R ins one hone) (Fin.last n₀) i = Fin.castAdd R.nComp (ins ⟨i, hi⟩) := by
  rw [insAt, Fin.lastCases_last, dite_eq_left hi]

theorem insAt_castSucc (v : Fin n₀) (i : ℕ) :
    insAt (snoc p R ins one hone) v.castSucc i = Fin.castAdd R.nComp (insAt p v i) := by
  rw [insAt, Fin.lastCases_castSucc]

theorem oneGate_snoc {o : Fin N₀} (hp : OneGate p o) (h : one = o) :
    OneGate (snoc p R ins one hone) (Fin.castAdd R.nComp o) := by
  intro u
  induction u using Fin.lastCases with
  | last => rw [oneAt_last, h]
  | cast v => rw [oneAt_castSucc, hp v]

end snoc

theorem rowsWireC_wire_one : ∀ {N n : ℕ} (p : FlockSoundness.Prog N n) (u : Fin n),
    (rowsWireC p u).wire (p.rows u).one = oneAt p u
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R ins one hone, u => by
    induction u using Fin.lastCases with
    | last =>
      rw [rowsWireC_last, castW_wire_one, oneAt_last]
      exact colWire_one R ins one
    | cast v =>
      rw [rowsWireC_castSucc, castW_wire_one, oneAt_castSucc]
      exact congrArg (Fin.castAdd R.nComp) (rowsWireC_wire_one p v)

theorem rowsWireC_wire_input : ∀ {N n : ℕ} (p : FlockSoundness.Prog N n) (u : Fin n) (c : Fin (p.rows u).w),
    c.val < (p.rows u).nIn → (rowsWireC p u).wire c = insAt p u c.val
  | _, _, .inputs _, u, _, _ => u.elim0
  | _, _, .snoc p R ins one hone, u, c, hc => by
    induction u using Fin.lastCases with
    | last =>
      have hc' : c.val < R.nIn := congrArg Rows.nIn (rows_last p R ins one hone) ▸ hc
      rw [rowsWireC_last, castW_wire, insAt_last p R ins one hone _ hc']
      exact colWire_input R ins one _ hc'
    | cast v =>
      have hc' : c.val < (p.rows v).nIn := congrArg Rows.nIn (rows_castSucc p R ins one hone v) ▸ hc
      rw [rowsWireC_castSucc, castW_wire, insAt_castSucc]
      exact congrArg (Fin.castAdd R.nComp) (rowsWireC_wire_input p v _ hc')

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n)

theorem instC_wire_one : (instC p outs u).wire (p.rows u).one = oneAt p u :=
  rowsWireC_wire_one p u

theorem instC_wire_input (c : Fin (p.rows u).w) (hc : c.val < (p.rows u).nIn) :
    (instC p outs u).wire c = insAt p u c.val :=
  rowsWireC_wire_input p u c hc

end FlockSoundness.Discharge.Const.Prog
