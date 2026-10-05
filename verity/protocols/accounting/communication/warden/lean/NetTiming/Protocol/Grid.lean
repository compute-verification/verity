/-!
# Grids, occupancy, left-packing, fill, and the FIFO warden

Definitions only. `docs/network-transparency/timing-channel.md` §2.1 and §2.3.

A grid is one link's record for one window: `T` buckets × `S` slots, each slot `none` or holding one real frame. Under
constant-rate release (the doc's W4), `S` is the allocation `r` and `none` is warden padding on the wire: every bucket
emits `S` frames, and occupancy counts the real ones.
A frame is a `List Bool`; padding (W3) is a hypothesis in `NetTiming.Assumptions`, not a type.

Lean core only: counts and sums run over `List.finRange`. `NetTiming.SecurityProofs.Reference` restates `rowCount` and
`offset` over `Finset` and proves each equal to its form here.
-/

namespace NetTiming

/-- One bucket's `S` slots on one link. -/
abbrev Row (S : Nat) (F : Type) := Fin S → Option F

/-- One link's grid for one window: `T` buckets × `S` slots. -/
abbrev Grid (T S : Nat) (F : Type) := Fin T → Row S F

/-- Occupancy: which slots hold a frame. -/
def occ {T S : Nat} {F : Type} (g : Grid T S F) : Fin T → Fin S → Bool := fun t s => (g t s).isSome

/-- A row's occupancy is left-packed: its occupied slots form a prefix. -/
def RowLeftPacked {S : Nat} (r : Fin S → Bool) : Prop :=
  ∀ s s' : Fin S, s ≤ s' → r s' = true → r s = true

/-- Left-packed occupancy (the packing rule of W4): in every bucket the occupied slots form a prefix. -/
def LeftPacked {T S : Nat} (P : Fin T → Fin S → Bool) : Prop :=
  ∀ t, RowLeftPacked (P t)

instance {S : Nat} : DecidablePred (RowLeftPacked (S := S)) := fun r => by
  unfold RowLeftPacked; infer_instance

instance {T S : Nat} : DecidablePred (LeftPacked (T := T) (S := S)) := fun P => by
  unfold LeftPacked; infer_instance

/-- The number of occupied slots of a row. -/
def rowCount {S : Nat} {F : Type} (r : Row S F) : Fin (S + 1) :=
  ⟨(List.finRange S).countP fun s => (r s).isSome,
    Nat.lt_succ_of_le (Nat.le_trans List.countP_le_length (Nat.le_of_eq List.length_finRange))⟩

/-- The number of frames released before bucket `t`, when bucket `u` releases `n u`. -/
def offset {T S : Nat} (n : Fin T → Fin (S + 1)) (t : Fin T) : Nat :=
  (((List.finRange T).filter fun u => u < t).map fun u => (n u : Nat)).sum

/-- **Fill.** The `k`-th frame of the queue-order sequence `f` goes to the `k`-th occupied slot in index order
`tS + s`: bucket `t` holds `n t` frames, in slots `0 … n t − 1`. -/
def fill {T S : Nat} {F : Type} (f : Nat → F) (n : Fin T → Fin (S + 1)) : Grid T S F :=
  fun t s => if (s : Nat) < (n t : Nat) then some (f (offset n t + s)) else none

/-- **The FIFO left-packing warden, its queue.** `enq t` frames become eligible in bucket `t` (they arrive before its
cutoff); bucket `t` releases `min (queue + enq t, S)` of them and the rest wait. -/
def queue (S : Nat) (enq : Nat → Nat) : Nat → Nat
  | 0 => 0
  | t + 1 => queue S enq t + enq t - min (queue S enq t + enq t) S

/-- The FIFO warden's release count in bucket `t`. -/
def released (S : Nat) (enq : Nat → Nat) (t : Nat) : Fin (S + 1) :=
  ⟨min (queue S enq t + enq t) S, Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩

/-- The grid the FIFO warden releases over buckets `0 … T − 1` when the sender enqueues `enq t` frames in bucket `t`,
the frames leaving in queue order `f 0, f 1, …`. -/
def fifoGrid (T S : Nat) {F : Type} (f : Nat → F) (enq : Nat → Nat) : Grid T S F :=
  fill f (fun t : Fin T => released S enq t)

end NetTiming
