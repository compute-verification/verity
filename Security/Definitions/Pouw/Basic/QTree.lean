import Mathlib.Data.Nat.Notation

/-!
# Adaptive oracle programs (trusted)

A `QTree Q R α` returns a value, or queries the oracle at a point and continues with the answer. Any computation
may happen between queries; `calls` counts the queries on the path taken against a given oracle.
-/

namespace Pouw

/-- An adaptive oracle program with result type `α`: return a value, or query `H` at `q` and continue. -/
inductive QTree (Q R α : Type) where
  | done : α → QTree Q R α
  | query : Q → (R → QTree Q R α) → QTree Q R α

namespace QTree

variable {Q R α : Type}

/-- Run the program against the oracle `H`. -/
def eval (H : Q → R) : QTree Q R α → α
  | done a => a
  | query q k => eval H (k (H q))

/-- The number of queries the program makes against `H`. -/
def calls (H : Q → R) : QTree Q R α → ℕ
  | done _ => 0
  | query q k => calls H (k (H q)) + 1

end QTree

end Pouw
