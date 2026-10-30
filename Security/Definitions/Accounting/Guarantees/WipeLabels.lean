import Definitions.Accounting.Assumptions.WipeLabels
import Definitions.Pous.Model.StreamedWipe

/-!
# The deployed wipe labels' PRF advantage (trusted statement)

`WipeLabelsAdvantage`: the wipe's label premise (`Accounting.Assumptions.WipeLabelsPrf`, `prf/sha-512`) bounds the PRF
advantage that the streamed wipe's survivor bound pays (`Pous.StreamedWipe.prfAdvantage`) at the labels the memory
challenger streams (`Accounting.Wipe.deployedLabels`). It is the guarantee that reads the premise, so the lock holds the
premise's record. Its proof is `Accounting.SecurityProofs.WipeLabelsAdvantage` (`Proofs.Accounting.WipeLabels`).
-/

namespace Accounting.Guarantees

open scoped ENNReal

/-- **The deployed labels' PRF advantage is at most `ε t N`.** At every hash `H`, bound `ε`, number of labels `N`,
distinguisher `D` with coins `ω` and count `t` of its calls: if `D` outputs true on the `N` deployed labels of a uniform
256-bit key at most `ε t N` more often than on `N` uniform labels (`WipeLabelsPrf`), then its PRF advantage against the
deployed labels, the term the streamed wipe pays, is at most `ε t N`. -/
def WipeLabelsAdvantage : Prop :=
  ∀ (H : Hash.Msg → Hash.Msg) (ε : ℕ → ℕ → ℝ≥0∞) (N : ℕ) (ω : Type) [Fintype ω]
    (D : ω × Accounting.Wipe.Labels N → Prop) (t : ℕ),
    Accounting.Assumptions.WipeLabelsPrf H ε D t →
      Pous.StreamedWipe.prfAdvantage (Accounting.Wipe.deployedLabels H N) D ≤ ε t N

end Accounting.Guarantees
