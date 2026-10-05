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

end snoc

section snoc

variable {N₀ n₀ : ℕ} (p : FlockSoundness.Prog N₀ n₀) (R : Rows) (ins : Fin R.nIn → Fin N₀) (one : Fin N₀)
  (hone : ∀ c, ins c ≠ one)

theorem oneGate_snoc {o : Fin N₀} (hp : OneGate p o) (h : one = o) :
    OneGate (snoc p R ins one hone) (Fin.castAdd R.nComp o) := by
  intro u
  induction u using Fin.lastCases with
  | last => rw [oneAt_last, h]
  | cast v => rw [oneAt_castSucc, hp v]

end snoc

variable {N n : ℕ} (p : FlockSoundness.Prog N n) (outs : Finset (Fin N)) (u : Fin n)

end FlockSoundness.Discharge.Const.Prog
