import Proofs.Nci.Cover.FanOutWitness
import Proofs.Nci.Cover.PartialSumKeyed

/-!
# The partial-sum bounds: their hypotheses can hold together

`partialSum_witness`: the fan-out witness (`FanOutWitness.fC`, its one counted product and keyed token structure) with a
cube of one entry, whose accumulation is the product's gate 14, satisfies the premises of `PartialSumKeyedCoverBound`
(`UpdateHardness`, `ColumnsIncompressible` with `k < bcol · V · (E + 1)` at `E = 1 < d = 2`, `Wide` at `n = 1`) and
those of `PartialSumCoverBound` at `f = a = 1` (`ActivationReads`, by `KeyedActivationReads`), with a cover by
inference blocks one of which computes the product whole.
-/

namespace Nci.Cover.SecurityProofs

namespace PartialSumWitness

open FanOutWitness

/-- The witness's cube: one matrix, row and column, and one entry, accumulated in the product's gate. -/
def fQ : Cube fK Unit Unit Unit where
  mat := fun _ => ()
  row := fun _ => ()
  col := fun _ => ()
  cell_inj := by
    intro p hp p' hp' _ _ _ _
    rw [Finset.mem_singleton.1 hp, Finset.mem_singleton.1 hp']
  stack := fun _ _ _ _ _ => ⟨rfl, rfl⟩
  acc := fun _ => {14}
  top := fun _ => 14
  acc_prod := by
    intro p hp
    obtain rfl : p = (1, 1) := Finset.mem_singleton.1 hp
    decide
  acc_disjoint := fun _ _ _ _ h => (h rfl).elim
  acc_computed := by
    intro _ _ g hg
    obtain rfl : g = 14 := Finset.mem_singleton.1 hg
    decide
  acc_top := by
    intro _ _ g hg
    obtain rfl : g = 14 := Finset.mem_singleton.1 hg
    exact reachesIn_refl (Finset.mem_singleton_self _)

theorem fQ_wide : fQ.Wide 1 := by
  intro p hp
  obtain rfl : p = (1, 1) := Finset.mem_singleton.1 hp
  exact Finset.card_pos.2 ⟨(1, 1), Finset.mem_filter.2 ⟨Finset.mem_singleton_self _, rfl⟩⟩

end PartialSumWitness

open FanOutWitness PartialSumWitness in
/-- **Non-vacuity of the partial-sum bounds**: some circuit with a counted product, a keyed token structure and a cube
satisfies `UpdateHardness`, `ColumnsIncompressible` with `k < bcol · V · (E + 1)` and `E < d`, `Wide` at `n ≥ 1`, and
`ActivationReads` at `f = 1` and `a = 1`, and has a cover by inference blocks one of which computes a counted product
whole (`FanOutWitness.fC`: `s = 6`, `k = 5`, `E = 1`, `d = 2`, `V = 3`, `b = bcol = 1`, four blocks). -/
theorem partialSum_witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (d aOut c V w : ℕ) (K : Keyed M Unit (Fin 3) d aOut c V w)
      (Q : Cube K Unit Unit Unit) (s k E n f a : ℕ) (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ 0 < a ∧ 0 < n ∧ Assumptions.UpdateHardness M s b ∧
      Assumptions.ActivationReads Q s k f a ∧ Assumptions.ColumnsIncompressible K s bcol ∧
      (k : ℝ) < bcol * V * (E + 1) ∧ E < d ∧ Q.Wide n ∧ (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧
      C.Covers 𝓘 ∧ ∃ I ∈ 𝓘, (M.whole I).Nonempty :=
  ⟨fC, fM, 2, 1, 1, 3, 2, fK, fQ, 6, 5, 1, 1, 1, 1, 1, 1, fI, ⟨(1, 1), by decide⟩, one_pos, one_pos, one_pos,
    fM_update 6,
    KeyedActivationReads fC fM 2 1 1 3 2 fK fQ 6 5 1 1 1 (fK_columns 6) (by norm_num) fQ_wide,
    fK_columns 6, by norm_num, by decide, fQ_wide, fI_blocks, fI_covers,
    ⟨{0, 1, 7, 12, 13, 14}, by decide, (1, 1), by decide⟩⟩

end Nci.Cover.SecurityProofs
