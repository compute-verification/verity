import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.BigOperators.Fin

/-!
# Grids, occupancy, left-packing, fill, and the FIFO certifier

Definitions only. `docs/network-transparency/timing-channel.md` §2.1 and §2.3.

A grid is one link's record for one window: `T` buckets × `S` slots, each slot `none` or holding one real frame. Under
constant-rate release (the doc's W4), `S` is the allocation `r` and `none` is certifier padding on the wire: every bucket
emits `S` frames, and occupancy counts the real ones.
A frame is a `List Bool`; padding (W3) is a hypothesis in `NetTiming.Assumptions`, not a type.
-/

namespace NetTiming

open Finset

/-- One bucket's `S` slots on one link. -/
abbrev Row (S : ℕ) (F : Type) := Fin S → Option F

/-- One link's grid for one window: `T` buckets × `S` slots. -/
abbrev Grid (T S : ℕ) (F : Type) := Fin T → Row S F

/-- Occupancy: which slots hold a frame. -/
def occ {T S : ℕ} {F : Type} (g : Grid T S F) : Fin T → Fin S → Bool := fun t s => (g t s).isSome

/-- A row's occupancy is left-packed: its occupied slots form a prefix. -/
def RowLeftPacked {S : ℕ} (r : Fin S → Bool) : Prop :=
  ∀ s s' : Fin S, s ≤ s' → r s' = true → r s = true

/-- Left-packed occupancy (the packing rule of W4): in every bucket the occupied slots form a prefix. -/
def LeftPacked {T S : ℕ} (P : Fin T → Fin S → Bool) : Prop :=
  ∀ t, RowLeftPacked (P t)

instance instDecidablePredForallFinBoolRowLeftPacked {S : ℕ} : DecidablePred (RowLeftPacked (S := S)) := fun r => by
  unfold RowLeftPacked; infer_instance

instance instDecidablePredForallFinForallBoolLeftPacked {T S : ℕ} : DecidablePred (LeftPacked (T := T) (S := S)) := fun P => by
  unfold LeftPacked; infer_instance

/-- The number of occupied slots of a row. -/
def rowCount {S : ℕ} {F : Type} (r : Row S F) : Fin (S + 1) :=
  ⟨(univ.filter fun s => (r s).isSome).card, Nat.lt_succ_of_le ((card_filter_le _ _).trans (by simp))⟩

/-- The number of frames released before bucket `t`, when bucket `u` releases `n u`. -/
def offset {T S : ℕ} (n : Fin T → Fin (S + 1)) (t : Fin T) : ℕ :=
  ∑ u ∈ univ.filter (fun u => u < t), (n u : ℕ)

/-- **Fill.** The `k`-th frame of the queue-order sequence `f` goes to the `k`-th occupied slot in index order
`tS + s`: bucket `t` holds `n t` frames, in slots `0 … n t − 1`. -/
def fill {T S : ℕ} {F : Type} (f : ℕ → F) (n : Fin T → Fin (S + 1)) : Grid T S F :=
  fun t s => if (s : ℕ) < (n t : ℕ) then some (f (offset n t + s)) else none

/-- **The FIFO left-packing certifier, its queue.** `enq t` frames become eligible in bucket `t` (they arrive before its
cutoff); bucket `t` releases `min (queue + enq t, S)` of them and the rest wait. -/
def queue (S : ℕ) (enq : ℕ → ℕ) : ℕ → ℕ
  | 0 => 0
  | t + 1 => queue S enq t + enq t - min (queue S enq t + enq t) S

/-- The FIFO certifier's release count in bucket `t`. -/
def released (S : ℕ) (enq : ℕ → ℕ) (t : ℕ) : Fin (S + 1) :=
  ⟨min (queue S enq t + enq t) S, Nat.lt_succ_of_le (min_le_right _ _)⟩

/-- The grid the FIFO certifier releases over buckets `0 … T − 1` when the sender enqueues `enq t` frames in bucket `t`,
the frames leaving in queue order `f 0, f 1, …`. -/
def fifoGrid (T S : ℕ) {F : Type} (f : ℕ → F) (enq : ℕ → ℕ) : Grid T S F :=
  fill f (fun t : Fin T => released S enq t)

end NetTiming
