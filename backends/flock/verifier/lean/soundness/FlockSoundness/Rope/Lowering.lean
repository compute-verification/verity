import FlockSoundness.Rope.Chunks

/-!
# RoPE's rows are its Boolean export's rows

The pinned `rope-head/neox-bf16/pair` unit (`flock-ir-unit/v2`, sha256 `933c4ef8…`,
`verity_flock.templates.rope_head.PINS`) and its gate circuit, the IR's Boolean export
(`verity_flock.boolean_export`: 64 inputs, 5,996 AND, 9,774 XOR and 1,733 NOT gates, 32 outputs),
are generated data (`Rope/Gates.lean`, `Pinned.lean`, `Checkpoints.lean`, `Chunks.lean`;
`verity_flock.lean_rows`). `Rope.sound` (`Chunks.lean`) is `UnitRows.Sound` for them; `rope_sound`
says it with RoPE's columns.
-/

namespace FlockSoundness.Rope

open GateRows UnitRows

theorem outs_length : circ.outs.length = 32 := by decide +kernel

theorem outCols_getD (o : ℕ) (ho : o < 32) : outCols.getD o 0 = 6144 + o := by
  have h : ∀ o < 32, outCols.getD o 0 = 6144 + o := by decide
  exact h o ho

/-- **RoPE's pinned rows compute its Boolean export.** Column values that satisfy the pinned rows,
with the constant (column 6272) at 1, carry on output column `6144 + o` the gate circuit's output
`o` on input columns `0 … 63`. -/
theorem rope_sound (z : ℕ → Bool) (hz : SatList pinned z) (h1 : z 6272 = true) (o : ℕ) (ho : o < 32) :
    z (6144 + o) = (circ.eval z)[o]'(by simpa [Circ.eval, outs_length] using ho) := by
  have h := sound z hz h1 o (by rw [outs_length]; exact ho)
  rw [outCols_getD o ho, List.getD_eq_getElem _ _ (by simpa [Circ.eval, outs_length] using ho)] at h
  exact h

end FlockSoundness.Rope
