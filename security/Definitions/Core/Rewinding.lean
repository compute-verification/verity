module

public import Mathlib.Algebra.Order.Chebyshev
public import Mathlib.Combinatorics.Enumerative.DoubleCounting
public import Mathlib.Data.Fintype.Prod
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Order.ConditionallyCompleteLattice.Finset

/-! The rewinding lemma's vocabulary: plurality tables, runs that open off them, and conflicting pairs. -/

@[expose] public section

open Finset

namespace Verity.Rewinding

variable {Ω P V : Type} [Fintype Ω] [DecidableEq Ω] [DecidableEq V]

-- `op ω p = some v`: the run with coins `ω` opens position `p` to `v` (with a path that
-- verifies); `none`: it does not open `p`.
variable (op : Ω → P → Option V)

def openedTo (p : P) (v : V) : Finset Ω := univ.filter fun ω => op ω p = some v

def opened (p : P) : Finset Ω := univ.filter fun ω => (op ω p).isSome

def IsPlurality (vstar : P → V) : Prop := ∀ p v, #(openedTo op p v) ≤ #(openedTo op p (vstar p))

abbrev Off (vstar : P → V) (ω : Ω) (p : P) : Prop := (op ω p).isSome ∧ op ω p ≠ some (vstar p)

/-- The two runs open `p` to different values. -/
abbrev ConflictAt (x : Ω × Ω) (p : P) : Prop :=
  (op x.1 p).isSome ∧ (op x.2 p).isSome ∧ op x.1 p ≠ op x.2 p

variable [Fintype P]

def bad (vstar : P → V) : Finset Ω := univ.filter fun ω => ∃ p, Off op vstar ω p

def conflict : Finset (Ω × Ω) := univ.filter fun x => ∃ p, ConflictAt op x p

end Verity.Rewinding
