module

public import Mathlib.Data.Nat.Notation
public import Mathlib.Logic.Function.Basic

/-!
# Adaptive oracle programs (trusted)

A `QTree Q R α` returns a value, or queries the oracle at a point and continues with the answer. Any computation
may happen between queries; `calls` counts the queries on the path taken against a given oracle. `bind` sequences two
programs, `queryAll` queries a list in order, and `upTo` builds a transcript unit by unit.
-/

@[expose] public section

namespace Pouw

/-- An adaptive oracle program with result type `α`: return a value, or query `H` at `q` and continue. -/
inductive QTree (Q R α : Type) where
  | done : α → QTree Q R α
  | query : Q → (R → QTree Q R α) → QTree Q R α

namespace QTree

variable {Q R α β : Type}

/-- Run the program against the oracle `H`. -/
def eval (H : Q → R) : QTree Q R α → α
  | done a => a
  | query q k => eval H (k (H q))

/-- The number of queries the program makes against `H`. -/
def calls (H : Q → R) : QTree Q R α → ℕ
  | done _ => 0
  | query q k => calls H (k (H q)) + 1

/-- Run `t`, then the program `f` of its result. -/
def bind : QTree Q R α → (α → QTree Q R β) → QTree Q R β
  | done a, f => f a
  | query q k, f => query q fun r => bind (k r) f

/-- Query every point of a list, in order. -/
def queryAll : List Q → QTree Q R (List R)
  | [] => done []
  | q :: qs => query q fun r => (queryAll qs).bind fun rs => done (r :: rs)

/-- Units `0, …, n − 1` committed in order, unit `u` by the program `f u`; every other unit blank. -/
def upTo {C : Type} (f : ℕ → QTree Q R C) : ℕ → QTree Q R (ℕ → Option C)
  | 0 => done fun _ => none
  | n + 1 => (upTo f n).bind fun τ => (f n).bind fun c => done (Function.update τ n (some c))

end QTree

end Pouw
