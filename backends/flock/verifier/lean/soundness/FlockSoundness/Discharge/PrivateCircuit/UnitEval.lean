import FlockSoundness.ExecRows

/-!
# A template's rows, evaluated

`rowsEval R z one`: the column values of the rows `R` on the input bits `z` and the constant `one`, row by row. A unit
that is an instance of `R` (`IsRowsUnit`) computes them from the values at its input and constant gates
(`localEval_rows`), so it is wrong exactly when a committed computed gate misses them (`mem_wrong_iff_rows`). In `k`
stacked copies (`Rows.stack`), copy `u`'s columns hold the template's values on the copy's inputs
(`rowsEval_stackCol`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Audit

/-- **The rows' column values** on the input bits `z` and the constant `one`: an input column is its bit, the constant
column is `one`, and computed row `i` is the AND of the XORs of the columns it reads, which are earlier columns or the
constant (`Rows.topo`). -/
def rowsEval (R : FlockSoundness.Rows) (z : Fin R.nIn → Bool) (one : Bool) (c : Fin R.w) : Bool :=
  if h : c.val < R.nIn then z ⟨c.val, h⟩
  else if h' : c.val < R.nIn + R.nComp then
    xorSum (R.a ⟨c.val - R.nIn, by omega⟩) (fun d => if _hd : d.val < c.val then rowsEval R z one d else one) &&
      xorSum (R.b ⟨c.val - R.nIn, by omega⟩) (fun d => if _hd : d.val < c.val then rowsEval R z one d else one)
  else one
termination_by c.val
decreasing_by all_goals assumption

/-- `rowsEval` at a column given as a number, `false` past the columns. -/
def rowsEvalN (R : FlockSoundness.Rows) (z : Fin R.nIn → Bool) (one : Bool) (x : ℕ) : Bool :=
  if h : x < R.w then rowsEval R z one ⟨x, h⟩ else false

end FlockSoundness.Discharge.PrivateCircuit
