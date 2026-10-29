import Definitions.NetworkCertifier.Run

/-!
# The Program's schedule: frames at their send ticks, one FIFO per execution block

The Program gives every real frame a **send tick**: the bucket, on the network certifier's bucket clock, at which B(a_j)
hands it over. An execution block's span is `g·T` consecutive buckets. Its frames are handed in canonical order, each
at `h(k) = max(send(k), h(k−1))` from the span's first bucket, and the certifier's own rule (FIFO at the allocation,
left-packed: `fifoGrid`) fixes every row. The FIFO starts empty at each span's first bucket, and a frame handed at or
after the span's end is never enqueued.

**What the Program reads.** `constantRate T g E frames` is an `EgressPrediction` whose advice is one value per execution
block, `Fin E → Adv`; the protocol's is a string of `|a|` bits, `Adv = Fin k → Bool`. Execution block `e`'s frames
(content, send tick, canonical order) are `frames ins e (a e) ℓ`: a function of the ingress and of `e`'s own advice
only, which is B's computation and is left abstract. So block `e`'s rows read no other block's advice.

**The program.** The certifier program (`NetworkCertifier.Schedule`, `schedule.py`) computes each frame's hand tick from
a declared clock sync per window (`NetworkCertifier.handingBuckets`). Its rows over its `W` windows are this schedule's
for one execution block of those windows, at send ticks its hand ticks (`Proofs.NetworkCertifier.Code`).

**One clock per spatial unit.** Every egress link reads the same advice. **The allocation is the slot count `S`**, as
in Python, where a row has `allocation` slots.
-/

namespace NetTiming

/-- A real frame as the Program fixes it: its content and its send tick. -/
structure LogicalFrame where
  payload : List Bool
  send : ℕ

/-- When each frame is handed, from the hand time `last` of the frame before: frame `f` at `max(f.send, last)`, so the
frames are handed in canonical order. -/
def handingFrom : ℕ → List LogicalFrame → List ℕ
  | _, [] => []
  | last, f :: fs =>
    let h := max f.send last
    h :: handingFrom h fs

/-- The enqueue counts over an execution block's span of `N` buckets from bucket `base`: its bucket `u` receives the
frames handed at `base + u`, the first handed no earlier than `base`. Frames handed at or after the span's end are
never enqueued. -/
def enqueued (N base : ℕ) (frames : List LogicalFrame) (u : ℕ) : ℕ :=
  if u < N then (handingFrom base frames).count (base + u) else 0

/-- The payloads in canonical order. The FIFO certifier never releases more frames than were enqueued
(`SumReleasedLe`), so the default past the end is never read. -/
def payloads (frames : List LogicalFrame) (k : ℕ) : List Bool :=
  (frames[k]?.map LogicalFrame.payload).getD []

/-- Row `u` of an execution block's span of `N` buckets from bucket `base`: the certifier's rule (FIFO at the allocation
`S`, left-packed) applied to the enqueue counts, the queue empty at the span's first bucket, with the frames leaving in
canonical order. -/
def expectedRow (S N base : ℕ) (frames : List LogicalFrame) (u : ℕ) : Row S (List Bool) :=
  fifoGrid (u + 1) S (payloads frames) (enqueued N base frames) (Fin.last u)

/-- **The constant-rate Program.** `E` execution blocks, each of `g·T` buckets, and block `e`'s frames
`frames ins e v ℓ` on each egress link `ℓ`, in canonical order, from the ingress `ins` and its advice `v`. With advice
`a`, link `ℓ`'s row at step `t` is `expectedRow` of block `t / (g·T)`'s frames under that block's advice, at bucket
`t mod (g·T)` of its span. Past the `E` blocks no slot holds a frame. Every link reads the same advice, one clock per
spatial unit (`Proofs/NetworkCertifier/README.md`, modelling choice 15). -/
def constantRate {L S : ℕ} {In Adv : Type} (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) : EgressPrediction L S In (Fin E → Adv) where
  rows t ins a ℓ :=
    if h : t / (g * T) < E then
      expectedRow S (g * T) (t / (g * T) * (g * T)) (frames ins ⟨t / (g * T), h⟩ (a ⟨t / (g * T), h⟩) ℓ)
        (t % (g * T))
    else fun _ => none

end NetTiming
