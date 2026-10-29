import Definitions.Pous.Game.Oracle

/-!
# The state after a history (sampled audits)

`seqState`, from `lean/submissions/b5/SampledAudit.lean`, verbatim: the state a sequential responder reaches after a
history of indices. The pinned `SampledAuditSeq` reads it.
-/

namespace Pous.Sampled

/-- The state a sequential responder reaches after answering the history `h` from round `j` in state `σ`. -/
def seqState {M : IdealModel} {ℓ S B : ℕ} (f : M.Qry → M.Ans)
    (step : ℕ → Bits S → Fin B → OracleAlgorithm M (Bits ℓ × Bits S)) : ℕ → List (Fin B) → Bits S → Bits S
  | _, [], σ => σ
  | j, i :: is, σ => seqState f step (j + 1) is ((step j σ i).run f).2

end Pous.Sampled
