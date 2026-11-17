module

public import Mathlib.Data.BitVec
public import Mathlib.Data.Nat.Notation
public import Mathlib.Tactic.Attr.Core
public import Mathlib.Tactic.Basic
public import Mathlib.Tactic.ToAdditive
public import Mathlib.Tactic.ToDual
import Mathlib.Algebra.Group.Nat.Defs
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.Finiteness.Attr
import Mathlib.Tactic.SetLike

/-! `PouwH1Combine_v1{TILE}` on words; a leaf has at least two lanes and `nw ≥ 2`, so `tree`'s empty-list default is never read. -/

@[expose] public section

namespace Pouw.H1Zy

def xor32 (a b : BitVec 32) : BitVec 32 := a ^^^ b

def eq32 (x y : BitVec 32) : Bool := decide (x = y)

/-- `PouwAndBit_v1`, `PouwAndBit_v2`'s word view: `a and b`. -/
def andBit (a b : Bool) : Bool := a && b

/-- One pass of a pairwise tree: adjacent elements combined by `f` in order, an odd last one carried (the body of
`h1tile._xor`'s and `h1zy._and_all`'s `while` loop). -/
def pairs {α : Type} (f : α → α → α) : List α → List α
  | [] => []
  | [a] => [a]
  | a :: b :: rest => f a b :: pairs f rest

theorem length_pairs_lt {α : Type} (f : α → α → α) : ∀ (a b : α) (rest : List α),
    (pairs f (a :: b :: rest)).length < (a :: b :: rest).length
  | _, _, [] => by simp [pairs]
  | _, _, [_] => by simp [pairs]
  | _, _, c :: d :: rest => by
      have := length_pairs_lt f c d rest
      simp only [pairs, List.length_cons] at this ⊢
      omega

/-- **A pairwise tree** of `f` over a list (`h1tile._xor`, `h1zy._and_all`): `pairs` until at most one element is left,
then that element; `d` for the empty list. -/
def tree {α : Type} (f : α → α → α) (d : α) : List α → α
  | [] => d
  | [a] => a
  | a :: b :: rest => tree f d (pairs f (a :: b :: rest))
termination_by l => l.length
decreasing_by exact length_pairs_lt f a b rest

def combineOk {L nw : ℕ} (partials : Fin L → Fin nw → BitVec 32) (receipt : Fin nw → BitVec 32) : Bool :=
  tree andBit true (List.ofFn fun w => eq32 (tree xor32 0 (List.ofFn fun l => partials l w)) (receipt w))

end Pouw.H1Zy
