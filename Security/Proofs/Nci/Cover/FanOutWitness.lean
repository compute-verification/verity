import Proofs.Nci.Cover.FanOutPlan
import Proofs.Nci.Cover.Witness

/-!
# The fan-out bounds: their hypotheses can hold together

`fanOut_witness`: one small circuit with a counted product and a keyed token structure, a root with two legs and a
lookup that a block holds, satisfying `UpdateHardness`, `ValueReadHardness` at `f = 2`, `ColumnsIncompressible` with
`k < bcol · V · (E + 1)` at `E = 1 < d`, and covered by inference blocks.
-/

namespace Nci.Cover.SecurityProofs

/-- **Non-vacuity of the fan-out bounds**: some circuit with a counted product and a keyed token structure satisfies
`UpdateHardness`, `ValueReadHardness` at `f = 2` and `ColumnsIncompressible` with `k < bcol · V · (E + 1)` and
`1 ≤ E < d`, and has a cover by inference blocks, one of which holds a lookup and one a root. -/
theorem fanOut_witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (d aOut c V : ℕ) (K : Keyed M Unit (Fin 3) d aOut c V) (s k E f : ℕ)
      (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ 2 ≤ f ∧ 1 ≤ E ∧ Assumptions.UpdateHardness M s b ∧
      Assumptions.ValueReadHardness K s f b ∧ Assumptions.ColumnsIncompressible K s bcol ∧
      (k : ℝ) < bcol * V * (E + 1) ∧ E < d ∧ (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 ∧
      (∃ I ∈ 𝓘, (K.lookups ∩ K.heldSet I).Nonempty) ∧ (∃ I ∈ 𝓘, (K.roots ∩ K.heldSet I).Nonempty) := by
  sorry

end Nci.Cover.SecurityProofs
