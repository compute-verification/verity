import Definitions.Accounting.Remote.Held
import Definitions.Accounting.Remote.Sketch

/-!
# What deriving the receipts' sketch bound assumes (trusted)

`Accounting.Guarantees.Remote.SketchBoundSession` derives `Accounting.Remote.Held.Assumptions.SketchBound` on a
trace over a session's coins (`Accounting.Remote.Sketch.Coins`) from the session's devices
(`Accounting.Remote.Sketch.Session`) and these premises, each a named `Prop` taken as a hypothesis:

* `KeyBudget A qtot`: across pass `j`'s candidate seeds the receipt-hash queries total at most `qtot j`, the hashes the
  node computes in the pass (`SketchHashBudget`'s budget, per pass of a remote session).
* `Receipted A T`: a receipt the trace counts correct (`Trace.correct`) is one the kept candidate's device got right
  (`Session.correctAt`): the trace's verdict on a receipt is the model's.
* `Present A T`: a block the kept candidate's device obtained whole in pass `j`, held on chip or read in from node
  memory, a link or a rebuild, was in node memory or on a chip at some tick of the pass (`Trace.present`). The trace
  counts whole blocks, the device sectors: the premise reads a block present when all its sectors were obtained in the
  pass, not necessarily at one tick.

The signs' rank tail is `Accounting.Assumptions.SketchRank`, evaluated outside Lean (art:d3c891e4).
-/

namespace Accounting.Remote.Sketch.Assumptions

open Accounting.RateSketch Accounting.Remote.Held Accounting.Remote.Sketch

variable {β : Type} {P N m h S L q ks : ℕ} {G : Fin P → ℕ}

/-- **The per-pass hash budget**: at every prover coin, earlier keys, candidate keys and store, pass `j`'s candidates'
receipt-hash queries total at most `qtot j`. The node computes the hashes, so grinding a seed spends them from the same
budget: `Σ_j qtot j` is at most the node's SHA-512 compressions over the session (`timing.sketch_failure`'s `C`). -/
def KeyBudget (A : Session β P N m h S L q ks G) (qtot : Fin P → ℕ) : Prop :=
  ∀ b j η (κ : Fin (G j) → Key N m h ks) (D : Data N m 0 Sector),
    ∑ c, (((A.device b j η c).prog (κ c).1 D).queries (κ c).2 D).length ≤ qtot j

/-- **Receipted**: on every accepting outcome, each block whose receipt the trace counts correct in credited pass `j`
is one the kept candidate's device tagged right. -/
def Receipted (A : Session β P N m h S L q ks G) (T : Trace (Coins β P N m h ks G)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ b ∈ T.correct ω (j : ℕ),
    ∃ t ∈ A.correctAt ω j (A.kept ω j), (t : ℕ) = b

/-- **Present**: on every accepting outcome, a block the kept candidate's device of pass `j` obtained whole was in node
memory or on a chip at some tick of the pass. -/
def Present (A : Session β P N m h S L q ks G) (T : Trace (Coins β P N m h ks G)) : Prop :=
  ∀ ω, T.accepts ω → ∀ j : Fin P, ∀ t : Fin N, A.Whole ω j (A.kept ω j) t →
    ∀ b : Fin T.blocks, (b : ℕ) = t → ∃ τ, T.tick ω (j : ℕ) ≤ τ ∧ τ < T.tick ω ((j : ℕ) + 1) ∧ b ∈ T.present ω τ

end Accounting.Remote.Sketch.Assumptions
