import Mathlib

/-!
# Fan-in: a straight-line program's dataflow (trusted)

The counting object of the fan-in floor (ISA-FANIN): a straight-line program over fresh inputs `ι`, each node an
instruction (one lane of a lane instruction, or one warp-wide instruction such as an MMA or a shuffle) in program
order, reading the nodes `src` names and computing its value from theirs alone; its runs; which inputs feed a node;
when a set of outputs sees every input (the receipt is injective in each word given the others); and the thin fold,
whose receipt is.
-/

namespace Accounting.TS

/-- A straight-line program over the fresh inputs `ι`, values in `W` (a node's value is everything it writes: its
words, its predicates and, for a store, the memory): node `i` of `n`, in program order, reads the nodes `src i` (fresh
inputs and earlier nodes) and computes its value from theirs alone. -/
structure Dataflow (ι W : Type) where
  n : ℕ
  src : Fin n → Finset (ι ⊕ Fin n)
  earlier : ∀ i k, Sum.inr k ∈ src i → k < i
  op : Fin n → (ι ⊕ Fin n → W) → W
  reads : ∀ i v v', (∀ s ∈ src i, v s = v' s) → op i v = op i v'

namespace Dataflow

variable {ι W : Type} (D : Dataflow ι W)

/-- `v` is a run of `D` on the inputs `x`. -/
def Runs (x : ι → W) (v : ι ⊕ Fin D.n → W) : Prop :=
  (∀ j, v (.inl j) = x j) ∧ ∀ i, v (.inr i) = D.op i v

/-- Input `j` feeds node `s`: `s` is `j`, or reads a node `j` feeds. -/
inductive Feeds (j : ι) : ι ⊕ Fin D.n → Prop
  | input : Feeds j (.inl j)
  | read {i : Fin D.n} {s : ι ⊕ Fin D.n} : s ∈ D.src i → Feeds j s → Feeds j (.inr i)

/-- The outputs `O` see every input: on any two runs whose inputs differ in exactly one, some output differs. -/
def Sees (O : Finset (ι ⊕ Fin D.n)) : Prop :=
  ∀ j x x' v v', D.Runs x v → D.Runs x' v' → (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → ∃ o ∈ O, v o ≠ v' o

/-- The components node `i` merges into one: its sources less one. -/
def merges (i : Fin D.n) : ℕ := (D.src i).card - 1

/-- The fan-in of the issued instructions: the components the nodes issued as instruction `j` merge. -/
def issued {J : Type} [DecidableEq J] (ins : Fin D.n → J) (j : J) : ℕ :=
  ∑ i ∈ Finset.univ.filter (fun i => ins i = j), D.merges i

end Dataflow

/-- One step of the thin fold: two words into the state, `s ⊕ w ⊕ w'` (one LOP3) or `s + w + w'` (one IADD3). -/
def thinStep (xor : Bool) (s w w' : BitVec 32) : BitVec 32 := if xor then s ^^^ w ^^^ w' else s + w + w'

/-- The thin fold: a run of words absorbed two a step, the steps alternating XOR3 and IADD3 from `xor`. -/
def thinFold : Bool → BitVec 32 → List (BitVec 32 × BitVec 32) → BitVec 32
  | _, s, [] => s
  | xor, s, (w, w') :: ws => thinFold (!xor) (thinStep xor s w w') ws

end Accounting.TS
