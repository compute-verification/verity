import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Prod
import Mathlib.Basic.Real.Basic

/-!
# Uniform probability and adaptive oracle programs (trusted)

Every random quantity is uniform over a finite type: the oracle `H : Q → R` and the salt. `pr P` is the fraction of
the type on which `P` holds. Adversaries are deterministic; a randomized adversary is an average of deterministic ones
within the same budget, so a bound for every deterministic adversary covers randomized ones.

A `QTree Q R α` returns a value, or queries the oracle at a point and continues with the answer; `calls` counts the
queries on the path taken against a given oracle.
-/

namespace Pouw.PearlC

open Classical in
/-- The probability that a uniform `x : α` satisfies `P`: `|{x | P x}| / |α|`. -/
noncomputable def pr {α : Type} [Fintype α] (P : α → Prop) : ℝ :=
  ((Finset.univ.filter P).card : ℝ) / Fintype.card α

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

end Pouw.PearlC
