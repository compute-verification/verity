/-!
# The certifier program: rows, grids, fill and the FIFO certifier

The executable forms of `Security/Definitions/NetworkCertifier/Grid.lean` that the difftest runs, under the same names in
`NetworkCertifier`. Lean core only: `offset` sums over `List.finRange` where the reference sums over a `Finset`.
`Proofs.NetworkCertifier.Code` proves what the difftest computes from them equal to the reference's.
-/

namespace NetworkCertifier

/-- One bucket's `S` slots on one link. -/
abbrev Row (S : Nat) (F : Type) := Fin S → Option F

/-- One link's grid for one window: `T` buckets × `S` slots. -/
abbrev Grid (T S : Nat) (F : Type) := Fin T → Row S F

/-- The number of frames released before bucket `t`, when bucket `u` releases `n u`. -/
def offset {T S : Nat} (n : Fin T → Fin (S + 1)) (t : Fin T) : Nat :=
  (((List.finRange T).filter fun u => u < t).map fun u => (n u : Nat)).sum

/-- **Fill.** The `k`-th frame of the queue-order sequence `f` goes to the `k`-th occupied slot in index order
`tS + s`: bucket `t` holds `n t` frames, in slots `0 … n t − 1`. -/
def fill {T S : Nat} {F : Type} (f : Nat → F) (n : Fin T → Fin (S + 1)) : Grid T S F :=
  fun t s => if (s : Nat) < (n t : Nat) then some (f (offset n t + s)) else none

/-- **The FIFO left-packing certifier, its queue.** `enq t` frames become eligible in bucket `t` (they arrive before its
cutoff); bucket `t` releases `min (queue + enq t, S)` of them and the rest wait. -/
def queue (S : Nat) (enq : Nat → Nat) : Nat → Nat
  | 0 => 0
  | t + 1 => queue S enq t + enq t - min (queue S enq t + enq t) S

/-- The FIFO certifier's release count in bucket `t`. -/
def released (S : Nat) (enq : Nat → Nat) (t : Nat) : Fin (S + 1) :=
  ⟨min (queue S enq t + enq t) S, Nat.lt_succ_of_le (Nat.min_le_right _ _)⟩

/-- The grid the FIFO certifier releases over buckets `0 … T − 1` when the sender enqueues `enq t` frames in bucket `t`,
the frames leaving in queue order `f 0, f 1, …`. -/
def fifoGrid (T S : Nat) {F : Type} (f : Nat → F) (enq : Nat → Nat) : Grid T S F :=
  fill f (fun t : Fin T => released S enq t)

end NetworkCertifier
