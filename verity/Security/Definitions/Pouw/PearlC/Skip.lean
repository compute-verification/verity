import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Definitions.Pouw.PearlC.Defs

/-!
# Pearl-C's skip debit (definitions only)

The debit agreed for Pearl-C's credit: debit (A), the joint flags under G4, and what the unit's credit reads from them.
Skipping is in place: a skipped atom's incoming word passes through. The flags' properties (every atom of a set of one
group's atoms whose joint skip leaves the word's output unchanged is flagged) are proved in `Pouw.PearlC.Freeze`; this
module imports no proofs, so the game may read it.

* **The joint flags** (`jointFlags`): the atoms in some set of one group's atoms whose joint skip leaves the output
  unchanged, over all `2^|g|` masks of the group, each promoted and followed by the later promotions.
* **The word's flags** (`wordFlags k a b t`): atom `t` of the word over rows `a` of `A′` and `b` of `B̃`, in groups of
  four atoms (`Params.pi.G`).
* **The unit's debits** (`unitFlags`, `chainDebit`, `needed`, `neededSet`): the word flags over the unit's words, the
  flagged atoms at a credit per atom, and the `A′` elements some word reads in an unflagged atom.
-/

namespace Pouw.PearlC.Skip

open Pouw.Fp8Atom Pouw.PearlC

section Fold

variable {α β : Type}

/-- `f` folded over `xs` from `w`, the steps flagged `true` in `ms` passed through; past the end of `ms`, every step
runs. -/
def foldMask (f : β → α → β) : β → List α → List Bool → β
  | w, x :: xs, s :: ms => foldMask f (if s then w else f w x) xs ms
  | w, xs, [] => xs.foldl f w
  | w, [], _ :: _ => w

end Fold

/-- One atom. -/
def step (w : Nat) (x : AtomIn) : Nat := hopperStep w x.1 x.2

/-- A group's word promoted into the total. -/
def promoteGroup (t : Option ℚ) (g : List AtomIn) : Option ℚ := promote t (groupWord g)

/-- All `2^n` masks of length `n`. -/
def allMasks : Nat → List (List Bool)
  | 0 => [[]]
  | n + 1 => (allMasks n).flatMap fun m => [false :: m, true :: m]

/-- A group's word from `+0` with the atoms `m` flags skipped in place. -/
def wordIn (g : List AtomIn) (m : List Bool) : Nat := foldMask step 0 g m

/-- The promoted total before group `i`. -/
def inTotal (gs : List (List AtomIn)) (i : Nat) : Option ℚ := (gs.take i).foldl promoteGroup (some 0)

/-- The words of the groups after group `i`. -/
def laterWords (gs : List (List AtomIn)) (i : Nat) : List Nat := (gs.drop (i + 1)).map groupWord

/-- Whether skipping the atoms `m` flags in group `g` together leaves the output unchanged, given the total `t` before
the group and the later words `ws`. -/
def keeps (g : List AtomIn) (t : Option ℚ) (ws : List Nat) (m : List Bool) : Bool :=
  decide (ws.foldl promote (promote t (wordIn g m)) = ws.foldl promote (promote t (groupWord g)))

/-- **The joint flags** of a group: atom `j` is flagged when some set of the group's atoms containing it, skipped
together in place, leaves the output unchanged (`2^|g|` masked words of the group, each promoted, then the later
promotions). -/
def jointAt (g : List AtomIn) (t : Option ℚ) (ws : List Nat) : List Bool :=
  (List.range g.length).map fun j => (allMasks g.length).any fun m => m.getD j false && keeps g t ws m

/-- The joint flags of group `i`. -/
def jointFlags (gs : List (List AtomIn)) (i : Nat) : List Bool :=
  jointAt (gs.getD i []) (inTotal gs i) (laterWords gs i)

/-! ## The debit -/

/-- **The word's flags**, debit (A): atom `t` of the word over row `a` of `A′` and row `b` of `B̃` (its atoms
`rowAtoms a b`, the chain `ctChain` promotes every four) is in `jointFlags` of its group `t / 4`, at position `t % 4`;
`false` for `t ≥ k/32`. -/
def wordFlags (k : ℕ) (a b : Fin k → Code) (t : ℕ) : Bool :=
  (jointFlags ((rowAtoms a b).toChunks 4) (t / 4)).getD (t % 4) false

variable {m n k : Nat}

/-- **The unit's flags**: atom `t` of word `(i, j)`, the chain over row `i` of `A′` and row `j` of `B̃`; the type is
`Forming.Flags m n`. -/
def unitFlags (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) : Fin m → Fin n → ℕ → Bool :=
  fun i j t => wordFlags k (Aq i) (Bq j) t

/-- **The unit's chain debit**: the flagged atoms `t < k/32` over the unit's `m·n` words, each at `c`, the chain's
credit per atom; `c · Forming.flagCount k flag`. -/
def chainDebit (k : ℕ) (c : ℚ) (flag : Fin m → Fin n → ℕ → Bool) : ℚ :=
  c * ∑ i, ∑ j, ((Finset.range (k / 32)).filter fun t => flag i j t = true).card

/-- **Needed**: element `(i, l)` of `A′` is read by atom `l / 32` of some word `(i, j)` that `flag` leaves unflagged;
`Forming.Needed`, the same definition. -/
def needed (flag : Fin m → Fin n → ℕ → Bool) (i : Fin m) (l : Fin k) : Prop := ∃ j, flag i j ((l : ℕ) / 32) = false

instance instDecidableNeeded (flag : Fin m → Fin n → ℕ → Bool) (i : Fin m) (l : Fin k) : Decidable (needed flag i l) := by
  unfold needed; infer_instance

/-- The needed elements of `A′`. -/
def neededSet (flag : Fin m → Fin n → ℕ → Bool) : Finset (Fin m × Fin k) :=
  Finset.univ.filter fun p => needed flag p.1 p.2

end Pouw.PearlC.Skip
