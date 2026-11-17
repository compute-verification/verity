module

public import Definitions.Poub.Guarantees.RemoteSketch

/-! The receipts' sketch bound over PoUS's P2 store (`SketchBoundP2`), drafted and not decided. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch
  ResourceAccounting.Remote.Sketch.Assumptions

/-- **The sketch bound at P2's receipts** (drafted). `SketchBoundRun` at a unit of four 2,056-byte blocks
(`m = 257`), 16 sign rows, SHA-512 tags and rank `r = 12`, for a prover of `Q` SHA-512 compressions over the session:
at most `Q + P` candidate seeds and `Q` receipt-hash queries in all, with SHA-512's keys at them within `(Q + P)²/2^257`
of uniform. -/
def SketchBoundP2 : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Remote.Sketch.Run β Hs P N 257 512 S L q 16 G) (T : Trace (Outcome β Hs N 257)),
    0 ≤ ε → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank 16 257 12 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    ResourceAccounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 257) → KeyBudget A qtot → Receipted A T →
    Present A T →
    SketchBound T ((Q : ℝ) / 2 ^ 96 + ((Q : ℝ) + P) * (ε + N / 2 ^ 512) + ((Q : ℝ) + P) ^ 2 / 2 ^ 257)

end ResourceAccounting.Guarantees.Remote
