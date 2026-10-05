import FlockSoundness.Lowering
import FlockSoundness.Protocol.Discharge.Const.Inst

/-!
# A program's own instance of each unit's rows

`Prog.isRowsUnit` gives each unit of a program an instance of its rows only as `Nonempty`. An instance chosen from it
need not wire the unit's constant where the program does: a unit whose rows hold a row `[one]·[one]` (a VU's copy
constant) and a row `[x]·[x]` reading an input has an instance that swaps the two, with input `x` on the constant's gate
and the constant on `x`'s source. So the end-to-end theorem's `hConst` (`Partition.ConstCols`) is a fact about the
instance a `ProgPlaces` carries, and holds for the program's own wiring:

* `Prog.rowsWireC p u`, `Prog.instC p outs u`: unit `u`'s rows wired as `p` wires them (`Prog.rowsWire_last`, lifted
  through later units by `Prog.rowsWire_snoc`). For the unit `snoc p R ins one hone` adds, input `c` is on `ins c`, its
  computed rows on the new gates, its constant on `one`.
* `Prog.oneAt p u`: the gate `p` wires unit `u`'s constant to, which `instC` puts it on (`instC_wire_one`).
* `Prog.insAt p u i`: the gate `p` wires unit `u`'s input `i` to, which `instC` puts it on (`instC_wire_input`).
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

/-- A new unit's constant column is on the gate it is wired to. -/
theorem colWire_one : colWire R ins one R.one = Fin.castAdd R.nComp one := by
  apply Fin.ext
  rw [colWire_val]
  simp [Rows.one]

/-- A new unit's input column is on the gate it is wired to. -/
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

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n)

end FlockSoundness.Discharge.Const.Prog
