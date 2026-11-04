import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Mathlib.Data.Fintype.Prod
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-!
# Plurality tables, off-plurality runs and conflicting pairs

The vocabulary of the rewinding lemma (`Proofs.Core.Game.Rewinding`). A prover's run with coins `ω` opens
position `p` to a value, or doesn't open it (`op`). A table is a **plurality** table when at each position no value is
opened in more runs (`IsPlurality`). A run is `bad` when it opens some position off the plurality table (`Off`), and a
pair of runs is a `conflict` when both open one position to different values (`ConflictAt`).
-/

open Finset

namespace Verity.Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq V]

-- `op ω p = some v`: the run with coins `ω` opens position `p` to `v` (with a path that
-- verifies); `none`: it does not open `p`.
variable (op : Ω → P → Option V)

/-- Runs that open `p` to `v`. -/
def openedTo (p : P) (v : V) : Finset Ω := univ.filter fun ω => op ω p = some v

/-- Runs that open `p`. -/
def opened (p : P) : Finset Ω := univ.filter fun ω => (op ω p).isSome

/-- `vstar` is a plurality table: at each position no value is opened in more runs. -/
def IsPlurality (vstar : P → V) : Prop := ∀ p v, #(openedTo op p v) ≤ #(openedTo op p (vstar p))

/-- The run opens `p` to a value other than `vstar p`. -/
abbrev Off (vstar : P → V) (ω : Ω) (p : P) : Prop := (op ω p).isSome ∧ op ω p ≠ some (vstar p)

/-- The two runs open `p` to different values. -/
abbrev ConflictAt (x : Ω × Ω) (p : P) : Prop :=
  (op x.1 p).isSome ∧ (op x.2 p).isSome ∧ op x.1 p ≠ op x.2 p

variable [Fintype P]

/-- Runs with an opening off the plurality table. -/
def bad (vstar : P → V) : Finset Ω := univ.filter fun ω => ∃ p, Off op vstar ω p

/-- Pairs of runs that open some position to different values. -/
def conflict : Finset (Ω × Ω) := univ.filter fun x => ∃ p, ConflictAt op x p

end Verity.Rewinding
