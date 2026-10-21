import Proofs.Nci.Cover.Charge

/-!
# The cover theorem: the area bound

`Intensity`: the counted products a block computes whole lie in `J × L`, with `#J ≤ k` (`ActivationCharge`) and
`b #L ≤ k` (`UpdateHardness` on them), so `b #whole ≤ k²`; the rest are at most `k` (`finished_sdiff_whole_card_le`).
`cover_sum` adds a per-block bound over a cover: every counted product's output gate is a non-input gate, so some block
finishes it. `CoverBound` is `cover_sum` on `Intensity`, and `Slowdown` is `CoverBound` multiplied out.
-/

namespace Nci.Cover.SecurityProofs

theorem Intensity : Guarantees.Intensity := by
  sorry

/-- A bound `c · #(finished I) ≤ X` on every block of a cover bounds the counted products: `c · #P ≤ #𝓘 · X`. -/
theorem cover_sum {C : Circuit} {J L : Type} (M : MulBlocks C J L)
    {𝓘 : Finset (Finset (Fin C.N))} (h𝓘 : C.Covers 𝓘) {c X : ℝ} (hc : 0 ≤ c)
    (hX : ∀ I ∈ 𝓘, c * (M.finished I).card ≤ X) : c * M.P.card ≤ 𝓘.card * X := by
  sorry

theorem CoverBound : Guarantees.CoverBound := by
  sorry

theorem Slowdown : Guarantees.Slowdown := by
  sorry

end Nci.Cover.SecurityProofs
