import Definitions.Accounting.Gates.Model

/-!
# The table circuit: a row of a registered table picked by a token (trusted)

A step that writes `w` int8 words, column `c`'s word at a token, from `w` registered columns of 256 words, as an
embedding lookup does. Its writes are `8·w` bits of run-time values, but the gate model charges it 9 units and 8
run-input bits whatever `w` is: its gathers share their index, so only the first pays the index's bits (`units`), and
its columns are registered. The encode bound then gives it at most `8 + 9·(n − 1)/E` bits of codec words
(`Guarantees.Cut.Table`), a share of about `1/w` of what it writes, for every table the prover picks.
-/

namespace Accounting.Gates

/-- Registered word `r` of the table, an int8. -/
def tableWord (r : ℕ) : Gate ℕ := ⟨.input .registered r, [], -128, 127, 0, "", .other, false⟩

/-- The token, a run input in `[0, 255]`, at gate `256·w`. -/
def tableToken (w : ℕ) : Gate ℕ := ⟨.input .run (256 * w), [], 0, 255, 0, "", .other, false⟩

/-- The `c`-th write: column `c`'s word at the token (gates `256·c … 256·c + 255`). -/
def tableGather (w c : ℕ) : Gate ℕ :=
  ⟨.gather none 0, 256 * w :: (List.range 256).map (256 * c + ·), -128, 127, 0, "", .other, false⟩

/-- **The table circuit** of `w` columns: the `256·w` registered words, the token, then the `w` writes. -/
def table (w : ℕ) : Dag ℕ :=
  (List.range (256 * w)).map tableWord ++ tableToken w :: (List.range w).map (tableGather w)

end Accounting.Gates
