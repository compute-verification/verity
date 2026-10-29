import Definitions.Pous.Model.Audit

/-!
# The sequential audit with free blocks (a sampled setup)

A sampled setup proves that `s` blocks of the committed code, drawn uniformly after its root, decode to `W`; so with
probability at least `1 − (1 − ε)^s` over the draw, all but at most `ε·B` committed blocks are the true ones. The
others are free: the prover commits anything there, stores nothing for them, and answers a challenge on one from its
commitment, so the audit's check on that round always passes. This game gives preprocessing that choice: besides its
`S`-bit state, it names at most `F` free blocks from `(W, pp, C)` and the whole primitive, and a round whose index is
free passes whatever the responder returns. With `F = 0` it is `seqAccepts`.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- The free blocks: at most `F` of them, chosen from `(W, pp, C)` and the whole primitive, as preprocessing is. -/
structure FreeSet (sch : Scheme M) (F : ℕ) : Type where
  free : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Finset (Fin sch.B)
  card_le : ∀ W pp C ω, (free W pp C ω).card ≤ F

/-- The sequential audit with `k` rounds and free blocks accepts: every round whose index is not free returns the
true block. -/
def seqAcceptsFree (sch : Scheme M) {S D Q F : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (Fr : FreeSet sch F) (A₂ : SeqResponder sch S D Q) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  let ans := seqAnswers (M.answer ω) (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω)
  ∀ j : Fin k, x.2.2 j ∉ Fr.free W pp C ω → ans[j.val]? = some (C (x.2.2 j))

/-- **The security target with free blocks.** For every `W`, every `S`-bit preprocessing, every choice of at most `F`
free blocks and every `(D, Q)`-bounded sequential responder, the `k`-round audit accepts with probability at most
`δ + ε`. -/
def AuditSecureFree (sch : Scheme M) (F S D Q k : ℕ) (δ ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (Fr : FreeSet sch F) (A₂ : SeqResponder sch S D Q),
    pr (seqAcceptsFree sch k W A₁ Fr A₂) ≤ δ + ε

end Pous
