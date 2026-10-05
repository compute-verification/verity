import FlockSoundness.Protocol.Lowering

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

/-- **Unit `u`'s rows, wired as the program wires them**, as an instance in the program's circuit. -/
def instC {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n) :
    (p.partition outs).IsRowsUnit u (p.rows u) :=
  (rowsWireC p u).toIsRowsUnit outs

/-- **The gate the program wires unit `u`'s constant to**: for the unit `snoc p R ins one hone` adds, `one`. -/
def oneAt : ∀ {N n : ℕ}, FlockSoundness.Prog N n → Fin n → Fin N
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R _ one _, u => Fin.lastCases (Fin.castAdd R.nComp one) (fun v => Fin.castAdd R.nComp (oneAt p v)) u

/-- **The gate the program wires unit `u`'s input `i` to**: for the unit `snoc p R ins one hone` adds, `ins i` (its
constant's gate past its inputs). -/
def insAt : ∀ {N n : ℕ}, FlockSoundness.Prog N n → Fin n → ℕ → Fin N
  | _, _, .inputs _, u => u.elim0
  | _, _, .snoc p R ins one _, u => Fin.lastCases
      (fun i => if h : i < R.nIn then Fin.castAdd R.nComp (ins ⟨i, h⟩) else Fin.castAdd R.nComp one)
      (fun v i => Fin.castAdd R.nComp (insAt p v i)) u

/-- **Every unit's constant on one gate.** -/
def OneGate {N n : ℕ} (p : FlockSoundness.Prog N n) (one : Fin N) : Prop := ∀ u, oneAt p u = one

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

end snoc

/-- **`rowsWireC` puts unit `u`'s constant on the gate the program wires it to.** -/
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

/-- **`rowsWireC` puts unit `u`'s input `c` on the gate the program wires it to.** -/
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

/-- **`instC` puts unit `u`'s constant on the gate the program wires it to.** -/
theorem instC_wire_one : (instC p outs u).wire (p.rows u).one = oneAt p u :=
  rowsWireC_wire_one p u

/-- **`instC` puts unit `u`'s input `c` on the gate the program wires it to.** -/
theorem instC_wire_input (c : Fin (p.rows u).w) (hc : c.val < (p.rows u).nIn) :
    (instC p outs u).wire c = insAt p u c.val :=
  rowsWireC_wire_input p u c hc

end FlockSoundness.Discharge.Const.Prog
