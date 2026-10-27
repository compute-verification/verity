import Definitions.Accounting.TS.Receipt
import Definitions.Accounting.HMu

/-!
# (H-μ) at the receipt's operands (trusted)

`ReceiptGame.OperandPointMass μ` is HMU's (H-μ) (`Accounting.HMu.SkipRound.PointMass`) on the ideal game split at the
receipt, at word level. At every round, whatever the past, the round's share lies in `[0, 1]`, and the round is a
`SkipRound` on its coins with:

* `skipped ω`: the values of the round's deficient words;
* `committed o`: the cheater's values for them, from its view `o`;
* the honest receipt `honest o x = x`, the word level, so (Inj) holds with fibre 1;
* the share decided from the view;
* the deficient words all right only where the cheater's values are the skipped values.

The skipped value then has point mass at most `μ` given the view. Lemma B at `Z = 1` turns this into the operand luck
at `μ` (`ReceiptGame.OperandLuck`). It asks nothing of the cheater's play beyond (H-μ): any round is a `SkipRound` with
the view everything the cheater's skip and values are functions of, and (H-μ) is that the skipped values' point mass
given that view is at most `μ`. A cheater that reads a coin pinning the skipped value fails it.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity Accounting.HMu

namespace ReceiptGame

variable {Ω : Type}

/-- **(H-μ) at the operands, at `μ`**: at every round, whatever the past, its share lies in `[0, 1]`, and the round is
a word-level `SkipRound` (the deficient words' values skipped, the cheater's values for them committed from its view,
the share decided from the view, the words all right only where the two agree) whose skipped value has point mass at
most `μ` given the view (`SkipRound.PointMass`). -/
def OperandPointMass [Fintype Ω] (μ : ℝ) : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round d _ right _ next =>
      (∀ ω, 0 ≤ d ω ∧ d ω ≤ 1) ∧
        (∃ (O α : Type) (R : SkipRound Ω O α α), R.PointMass μ ∧ (∀ o x, R.honest o x = x) ∧
          (∀ ω, d ω = R.d (R.view ω)) ∧ ∀ ω, right ω → R.committed (R.view ω) = R.skipped ω) ∧
        ∀ ω, OperandPointMass μ (next ω)

end ReceiptGame

end Accounting.TS
