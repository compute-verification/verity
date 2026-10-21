import Models.Nci.Guarantees.Cover

/-!
# The cover theorem: its hypotheses can hold together

`witness`: one small circuit with a counted product that satisfies `UpdateHardness`, the upstream certificate with
`k < b D`, and has a cover by inference blocks, so the area bound's hypotheses are not contradictory.
`inference_sanity`: a circuit with no products, such as inference, satisfies `UpdateHardness` and the certificate at
every depth, and the bounds then say nothing, as they should.
-/

namespace Nci.Cover.SecurityProofs

/-- **Non-vacuity**: some circuit with a counted product satisfies `UpdateHardness` and the upstream certificate with
`k < b D`, and has a cover by inference blocks. -/
theorem witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (A : ℕ → Finset (Fin C.N)) (s k D : ℕ) (b : ℝ)
      (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ Assumptions.UpdateHardness M s b ∧ M.Upstream A D ∧ (k : ℝ) < b * D ∧
      (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 := by
  sorry

/-- **Honest inference**: with no products, `UpdateHardness` and the upstream certificate hold at every block size, rate
and depth. -/
theorem inference_sanity (C : Circuit) (M : MulBlocks C ℕ ℕ) (hM : M.Pall = ∅) (A : ℕ → Finset (Fin C.N))
    (s D : ℕ) (b : ℝ) : Assumptions.UpdateHardness M s b ∧ M.Upstream A D := by
  sorry

end Nci.Cover.SecurityProofs
