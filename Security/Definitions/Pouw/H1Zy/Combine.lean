import Mathlib.Data.BitVec

/-!
# `PouwH1Combine_v1{TILE}` on words (trusted)

A drawn H1 leaf's combine statement (`verity/core/protocols/pouw/circuit/h1zy.py`, `PouwH1Combine`) reads the leaf's
lanes' committed partials, `L` rows of `nw` u32 words (`nw = 2 (KS + m)`), and the leaf's committed receipt, `nw` words,
and returns one bit `ok`. Its body (`h1zy._combine_body`), on its primitives' word views:

* for each word `w`, the XOR of the lanes' words `w` in lane order, as `h1tile._xor` builds it: a tree of
  `PouwXor32_v2` calls (`xor32`), each pass XORing adjacent pairs and carrying an odd last word, until one is left
  (`tree`);
* each XOR against the receipt's word `w` (`PouwEq32_v2`, `eq32`);
* the comparisons' AND, as `h1zy._and_all` builds it: the same tree of `PouwAndBit_v2` calls (`andBit`).

Python's loops read the first word of a non-empty list; a leaf has at least two lanes and `nw ≥ 2`, so the empty
list's value (`tree`'s default: `0` for the XOR, `true` for the AND) is never read.
-/

namespace Pouw.H1Zy

/-- `PouwXor32_v1`, `PouwXor32_v2`'s word view: `a xor b`. -/
def xor32 (a b : BitVec 32) : BitVec 32 := a ^^^ b

/-- `PouwEq32_v1`, `PouwEq32_v2`'s word view (`segment.PouwEq32Word`): `1` exactly when `x = y`. -/
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

/-- **The combine's `ok`** (`h1zy._combine_body`) on its `L` lanes' partials and the receipt, `nw` u32 words each:
for every word `w`, the `PouwXor32` tree over the lanes' words `w` in lane order is the receipt's word `w`
(`PouwEq32`), and the comparisons' `PouwAndBit` tree. -/
def combineOk {L nw : ℕ} (partials : Fin L → Fin nw → BitVec 32) (receipt : Fin nw → BitVec 32) : Bool :=
  tree andBit true (List.ofFn fun w => eq32 (tree xor32 0 (List.ofFn fun l => partials l w)) (receipt w))

end Pouw.H1Zy
