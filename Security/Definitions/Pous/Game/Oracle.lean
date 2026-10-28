import Definitions.Pous.Game.Prob

/-!
# The idealized computational model

An `IdealModel` is an ideal primitive. `Ω` lists every possible primitive (every function `X → Y` for a random
oracle, every permutation for an ideal permutation), and setup draws one uniformly. The trusted encoder and the
unbounded preprocessor see the drawn `ω : Ω` whole; the decoder and the online responder see it only through
queries, `answer ω q`.

An `OracleAlgorithm` is a parallel oracle machine. Each round sends a list of queries at once and continues on the list of
answers. Local computation and working memory between rounds are free and unbounded. `Bounded D Q` means at most `D`
rounds and at most `Q` queries in total, on every primitive: the problem statement's `(D, Q)` time model, where a
step is one round of parallel queries and work counts queries. `RoundBounded D q` bounds each round by `q` queries
instead (the parallel-query model of depth-robust-graph proofs).
-/

namespace Pous

/-- An ideal primitive: drawn uniformly from `Ω`, queried through `answer`. -/
structure IdealModel : Type 1 where
  /-- Every possible primitive. -/
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  /-- Queries and answers of the public interface. -/
  Qry : Type
  Ans : Type
  answer : Ω → Qry → Ans

attribute [instance] IdealModel.fintypeΩ IdealModel.nonemptyΩ

/-- The plain model: no primitive and no queries. Every `OracleAlgorithm` is then `(0, 0)`-bounded while local computation
stays unbounded, so games in this model are information-theoretic. -/
def IdealModel.plain : IdealModel where
  Ω := Unit
  Qry := Empty
  Ans := Empty
  answer := fun _ q => q.elim

/-- A parallel oracle program with output `α`. -/
inductive OracleAlgorithm (M : IdealModel) (α : Type) : Type where
  /-- Stop with output `a`. -/
  | ret : α → OracleAlgorithm M α
  /-- One round: send the queries `qs` together, continue on their answers (in the same order). -/
  | round : List M.Qry → (List M.Ans → OracleAlgorithm M α) → OracleAlgorithm M α

namespace OracleAlgorithm

variable {M : IdealModel} {α : Type}

/-- The output, when queries are answered by `f`. -/
def run (f : M.Qry → M.Ans) : OracleAlgorithm M α → α
  | ret a => a
  | round qs k => run f (k (qs.map f))

/-- The number of rounds on the path taken when queries are answered by `f`. -/
def depth (f : M.Qry → M.Ans) : OracleAlgorithm M α → ℕ
  | ret _ => 0
  | round qs k => depth f (k (qs.map f)) + 1

/-- The number of queries on the path taken when queries are answered by `f`. -/
def work (f : M.Qry → M.Ans) : OracleAlgorithm M α → ℕ
  | ret _ => 0
  | round qs k => qs.length + work f (k (qs.map f))

/-- The largest number of queries in one round, on the path taken when queries are answered by `f`. -/
def width (f : M.Qry → M.Ans) : OracleAlgorithm M α → ℕ
  | ret _ => 0
  | round qs k => max qs.length (width f (k (qs.map f)))

/-- `(D, Q)`-bounded: at most `D` rounds and `Q` queries in total, against every primitive. -/
def Bounded (D Q : ℕ) (p : OracleAlgorithm M α) : Prop :=
  ∀ ω : M.Ω, p.depth (M.answer ω) ≤ D ∧ p.work (M.answer ω) ≤ Q

/-- At most `D` rounds of at most `q` queries each, against every primitive. -/
def RoundBounded (D q : ℕ) (p : OracleAlgorithm M α) : Prop :=
  ∀ ω : M.Ω, p.depth (M.answer ω) ≤ D ∧ p.width (M.answer ω) ≤ q

end OracleAlgorithm

/-- A random oracle `X → Y`: every function is equally likely, and a query at `x` returns its value. -/
def IdealModel.randomOracle (X Y : Type) [Fintype X] [DecidableEq X] [Fintype Y] [Nonempty Y] :
    IdealModel where
  Ω := X → Y
  Qry := X
  Ans := Y
  answer := fun ω x => ω x

end Pous
