module

public import Definitions.Poub.Remote.Held
public import Definitions.Poub.Remote.Sketch
public import Definitions.Assumptions.ResourceAccounting

/-! What deriving the receipts' sketch bound assumes (`SketchHashRandom`, `KeyBudget`, `Receipted`, `Present`); the
signs' rank tail is evaluated outside Lean. -/

@[expose] public section

namespace ResourceAccounting.Remote.Sketch.Assumptions

open ResourceAccounting.RateSketch ResourceAccounting.Remote.Held ResourceAccounting.Remote.Sketch

variable {β Hs : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- **The per-pass hash budget**: at every prover coin, earlier keys, candidate keys and store, pass `j`'s candidates'
receipt-hash queries total at most `qtot j`. Grinding a seed reads no block, so the prover may compute its hashes off
the node as well as on it: `Σ_j qtot j` is at most the SHA-512 compressions the prover can make over the session,
`Q`, its budget on any hardware it reaches (`timing.sketch_failure`'s `queries`, `2^80` in `audit.QUERIES_LOG2`). -/
def KeyBudget (A : Run β Hs P N m h S L q ks G) (qtot : Fin P → ℕ) : Prop :=
  ∀ b j η (κ : Fin (G j) → Key N m h ks) (D : Data N m 0 Sector),
    ∑ c, (((A.device b j η c).prog (κ c).1 D).queries (κ c).2 D).length ≤ qtot j

/-- **Receipted**: on every accepting outcome, each block whose receipt the trace counts correct in credited pass `j`
is one the kept candidate's device tagged right at the key SHA-512 gives at its seed. -/
def Receipted (A : Run β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ b ∈ T.correct ω (j : ℕ),
    ∃ t ∈ A.correctAt ω j (A.kept ω j), (t : ℕ) = b

/-- **Present**: on every accepting outcome, a block the kept candidate's device of pass `j` obtained whole was in node
memory or on a chip at some tick of the pass. -/
def Present (A : Run β Hs P N m h S L q ks G) (T : Trace (Outcome β Hs N m)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ t : Fin N, A.Whole ω j (A.kept ω j) t →
    ∀ b : Fin T.blocks, (b : ℕ) = t → ∃ τ, T.tick ω (j : ℕ) ≤ τ ∧ τ < T.tick ω ((j : ℕ) + 1) ∧ b ∈ T.present ω τ

end ResourceAccounting.Remote.Sketch.Assumptions
