module

public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Data.Fintype.Card

/-! Grids, occupancy, left-packing, fill and the FIFO certifier; padding (W3) is a hypothesis in `NetTiming.Assumptions`, not a type. -/

@[expose] public section

namespace NetTiming

open Finset

/-- One bucket's `S` slots on one link. -/
abbrev Row (S : ℕ) (F : Type) := Fin S → Option F

abbrev Grid (T S : ℕ) (F : Type) := Fin T → Row S F

def occ {T S : ℕ} {F : Type} (g : Grid T S F) : Fin T → Fin S → Bool := fun t s => (g t s).isSome

/-- A row's occupancy is left-packed: its occupied slots form a prefix. -/
def RowLeftPacked {S : ℕ} (r : Fin S → Bool) : Prop :=
  ∀ s s' : Fin S, s ≤ s' → r s' = true → r s = true

def LeftPacked {T S : ℕ} (P : Fin T → Fin S → Bool) : Prop :=
  ∀ t, RowLeftPacked (P t)

instance instDecidablePredForallFinBoolRowLeftPacked {S : ℕ} : DecidablePred (RowLeftPacked (S := S)) := fun r => by
  unfold RowLeftPacked; infer_instance

instance instDecidablePredForallFinForallBoolLeftPacked {T S : ℕ} : DecidablePred (LeftPacked (T := T) (S := S)) := fun P => by
  unfold LeftPacked; infer_instance

/-- The number of occupied slots of a row. -/
def rowCount {S : ℕ} {F : Type} (r : Row S F) : Fin (S + 1) :=
  ⟨(univ.filter fun s => (r s).isSome).card, Nat.lt_succ_of_le ((card_filter_le _ _).trans (by simp))⟩

def offset {T S : ℕ} (n : Fin T → Fin (S + 1)) (t : Fin T) : ℕ :=
  ∑ u ∈ univ.filter (fun u => u < t), (n u : ℕ)

def fill {T S : ℕ} {F : Type} (f : ℕ → F) (n : Fin T → Fin (S + 1)) : Grid T S F :=
  fun t s => if (s : ℕ) < (n t : ℕ) then some (f (offset n t + s)) else none

def queue (S : ℕ) (enq : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => queue S enq t + enq t - min (queue S enq t + enq t) S

def released (S : ℕ) (enq : ℕ → ℕ) (t : ℕ) : Fin (S + 1) :=
  ⟨min (queue S enq t + enq t) S, Nat.lt_succ_of_le (min_le_right _ _)⟩

def fifoGrid (T S : ℕ) {F : Type} (f : ℕ → F) (enq : ℕ → ℕ) : Grid T S F :=
  fill f (fun t : Fin T => released S enq t)

end NetTiming
