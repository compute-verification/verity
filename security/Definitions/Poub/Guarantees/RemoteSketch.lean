module

public import Definitions.Poub.Assumptions.RateAuditDeployed
public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Poub.Assumptions.RemoteSketch

/-! The receipts' sketch bound, derived (`SketchBoundRun`, `SketchBoundDeployed`), the signs' rank tail evaluated
outside Lean. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.Remote

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sketch
  ResourceAccounting.Remote.Sketch.Assumptions

/-- **The sketch bound at every parameter.** On a trace over a session's outcomes whose receipts and obtained blocks
are the session devices' (`Receipted`, `Present`), with SHA-512's keys at the candidate seeds within `ε_ro` of
uniform (`SketchHashRandom`), within the per-pass hash budget `qtot` (`KeyBudget`) and with signs whose rank tail below
`r ≤ 32` is at most `ε` (`SketchRank`), `SketchBound` holds at
`(Σ_j G_j)·(ε + N/2^h) + (Σ_j qtot_j)/2^(8r) + ε_ro`. -/
def SketchBoundRun : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N m h S L q ks r : ℕ) (G qtot : Fin P → ℕ) (ε εro : ℝ)
    (A : Remote.Sketch.Run β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)),
    r ≤ 32 → 0 ≤ ε → 0 ≤ εro → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank ks m r (ENNReal.ofReal ε) →
    ResourceAccounting.Assumptions.SketchHashRandom A εro → KeyBudget A qtot → Receipted A T → Present A T →
    SketchBound T (((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r) + εro)

/-- **The sketch bound at the deployed receipts.** `SketchBoundRun` at 4096-byte blocks (`m = 128`), 64 sign rows,
SHA-512 tags and rank `r = 27`, for a prover of `Q` SHA-512 compressions over the session (its budget, `2^80` in
`audit.QUERIES_LOG2`, on any hardware it reaches): at most `Q + P` candidate seeds and `Q` receipt-hash queries in all,
with SHA-512's keys at them within the 512-bit sign keys' birthday `(Q + P)²/2^513` of uniform. -/
def SketchBoundDeployed : Prop :=
  ∀ (β Hs : Type) [Fintype β] [Fintype Hs] (P N S L q Q : ℕ) (G qtot : Fin P → ℕ) (ε : ℝ)
    (A : Remote.Sketch.Run β Hs P N 128 512 S L q 64 G) (T : Trace (Outcome β Hs N 128)),
    0 ≤ ε → T.passes ≤ P →
    ResourceAccounting.Assumptions.SketchRank 64 128 27 (ENNReal.ofReal ε) →
    ∑ j, G j ≤ Q + P → ∑ j, qtot j ≤ Q →
    ResourceAccounting.Assumptions.SketchHashRandom A (((Q : ℝ) + P) ^ 2 / 2 ^ 513) → KeyBudget A qtot → Receipted A T →
    Present A T →
    SketchBound T ((Q : ℝ) / 2 ^ 216 + ((Q : ℝ) + P) * (ε + N / 2 ^ 512) + ((Q : ℝ) + P) ^ 2 / 2 ^ 513)

end ResourceAccounting.Guarantees.Remote
