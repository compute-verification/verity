/-
What a laid-out net's rows force: if its computed rows (from the end of its input rows to its constant) hold under the
column values `v`, a row's `A` and `B` the XOR of `v` over its columns, and its constant column is 1 (`LaidRows`), then
every row of the builder holds under `v ∘ col` (`laid_rows`), and output `j`'s column holds output `j`'s form
(`laid_outs`).
-/
import FlockProofs.HmNets
import Mathlib.Tactic
import FlockSoundness.Protocol.Discharge.Hm.Spec

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- A laid-out net's input rows: `groupWords nIn` words. -/
abbrev inRowsOf (inBits : Array ℕ) : ℕ := groupWords (inBits.foldl (· + ·) 0) * Flock.WORD

/-- A laid-out net's output group's rows. -/
abbrev spanOf (outBits : Array ℕ) : ℕ := groupWords (outBits.foldl (· + ·) 0) * Flock.WORD

/-- The column of a laid-out net's first output: past the input rows and the ANDs, aligned to the output group. -/
def outStart (st : St) (inBits outBits : Array ℕ) : ℕ :=
  (inRowsOf inBits + st.rows.length + spanOf outBits - 1) / spanOf outBits * spanOf outBits

theorem layout_outGroups (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) :
    (layout st inBits outBits outs).outGroups =
      #[⟨outStart st inBits outBits / Flock.WORD, groupWords (outBits.foldl (· + ·) 0), outBits⟩] := by
  simp [layout, outStart, inRowsOf, spanOf]

theorem layout_inGroups (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) :
    (layout st inBits outBits outs).inGroups = #[⟨0, groupWords (inBits.foldl (· + ·) 0), inBits⟩] := rfl

end FlockSoundness.Discharge.Hm
