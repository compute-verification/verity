import Definitions.NetworkCertifier.Run

/-!
# The Program's logical schedule and the clock-sync advice: the constant-rate Program

Definitions only. They mirror `verity/core/protocols/certifier/schedule.py` one for one, with each Python name in its
docstring.

The Program gives every real frame an **anchor** and a **logical index**: the anchor is the recorded bucket, on the
global bucket clock, of the request it answers; the logical index λ is its position within its session, in the
Program's canonical gate order. A window declares one clock-sync value `ClockSync δ ρ` from the protocol's set
`SyncSet`. A frame is eligible at `anchor + δ + ⌊λ / ρ⌋`, with the sync of the window that holds its anchor. The
shaper hands frames to the certifier in canonical order, each at `h(k) = max(eligible(k), h(k−1))`, and the certifier's own
rule (FIFO at the allocation, left-packed: `fifoGrid`) fixes every row.

**What the Program reads.** `constantRate T W ss frames` is a `Program` whose advice is content advice `Ac` and one
value of `ss` per window, `Fin W → ss.Elem`. Its frames (content, anchor, index, canonical order) are `frames ins c ℓ`:
a function of the ingress and the content advice only, which is the Program's computation and is left abstract. The
sync advice enters only through `eligible`. Declaring a value of the set is the audit's `advice` check: a run whose
declared sync is outside `Σ_sync` is rejected, so the accepted runs' advice ranges over `ss.Elem`.

**One clock per spatial unit.** Every egress link reads the same window's sync (the note's §2.5 and decision A's
default). **The allocation is the slot count `S`**, as in Python, where a row has `allocation` slots.
-/

namespace NetTiming

/-- `ClockSync`: one window's clock-sync advice, the offset δ (buckets after the anchor) and the rate ρ (frames per
bucket within a session, a positive fraction). Python rejects `delta < 0` (here `ℕ`) and `rho ≤ 0`. -/
structure ClockSync where
  delta : ℕ
  rho : Rat
  rho_pos : 0 < rho

/-- `SyncSet`: `Σ_sync`, the values a window may declare, each δ of `deltas` with each ρ of `rhos`. The protocol fixes
it, from calibration. Python requires both lists non-empty, each value listed once (a `Finset`), every δ ≥ 0 (`ℕ`) and
every ρ > 0. -/
structure SyncSet where
  deltas : Finset ℕ
  rhos : Finset Rat
  deltas_nonempty : deltas.Nonempty
  rhos_nonempty : rhos.Nonempty
  rhos_pos : ∀ ρ ∈ rhos, 0 < ρ

namespace SyncSet

/-- A value of `Σ_sync`, a δ of the set and a ρ of the set: what `SyncSet.__iter__` yields and `__contains__` accepts. -/
abbrev Elem (ss : SyncSet) : Type := {d // d ∈ ss.deltas} × {ρ // ρ ∈ ss.rhos}

/-- The clock sync a value of the set names. -/
def sync (ss : SyncSet) (v : ss.Elem) : ClockSync := ⟨v.1, v.2, ss.rhos_pos v.2 v.2.2⟩

/-- `SyncSet.size`: `#Σ_sync`, so `log2 ss.size` is the timing advice per window (`SyncSet.bits`). -/
def size (ss : SyncSet) : ℕ := ss.deltas.card * ss.rhos.card

end SyncSet

/-- `LogicalFrame`: a real frame as the Program fixes it, its content, its anchor and its logical index. -/
structure LogicalFrame where
  payload : List Bool
  anchor : ℕ
  index : ℕ

/-- `eligible`: the frame's logical release bucket, `anchor + δ + ⌊λ / ρ⌋`, computed as Python does,
`anchor + delta + (index * rho.denominator) // rho.numerator`. -/
def eligible (f : LogicalFrame) (s : ClockSync) : ℕ :=
  f.anchor + s.delta + f.index * s.rho.den / s.rho.num.toNat

/-- The loop of `handing_buckets` from the hand time `last` of the frame before: frame `f`, anchored in window
`f.anchor / T`, is handed at `max(eligible f (syncs w), last)` if `w < W`, and otherwise (anchored past the last window
with advice) at `max(last, W·T, f.anchor)`, after the horizon. -/
def handingFrom (T W : ℕ) (syncs : Fin W → ClockSync) : ℕ → List LogicalFrame → List ℕ
  | _, [] => []
  | last, f :: fs =>
    let h := if hw : f.anchor / T < W then max (eligible f (syncs ⟨f.anchor / T, hw⟩)) last
      else max (max last (W * T)) f.anchor
    h :: handingFrom T W syncs h fs

/-- `handing_buckets`: when the shaper hands each frame to the certifier, `h(k) = max(eligible(k), h(k−1))` in canonical
order, starting from `last = 0`. `T` is the window's bucket count and `W` the number of windows with advice. -/
def handingBuckets (T W : ℕ) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) : List ℕ :=
  handingFrom T W syncs 0 frames

/-- The enqueue counts of `expected_grids`: bucket `t` of the horizon `W·T` receives the frames handed at `t`. Frames
handed at or after the horizon are never enqueued. -/
def enqueued (T W : ℕ) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ) : ℕ :=
  if t < W * T then (handingBuckets T W syncs frames).count t else 0

/-- The payloads in canonical order, `[f.payload for f in frames]`. The FIFO certifier never releases more frames than were
enqueued (`SumReleasedLe`), so the default past the end is never read. -/
def payloads (frames : List LogicalFrame) (k : ℕ) : List Bool :=
  (frames[k]?.map LogicalFrame.payload).getD []

/-- `expected_grids`, read at bucket `t`: the certifier's rule (FIFO at the allocation `S`, left-packed, the queue carried
across windows) applied to the enqueue counts, with the frames leaving in canonical order. For `t < W·T` this is row
`t mod T` of Python's grid for window `t / T`. Past the horizon, where Python has no grid, the queue drains with
nothing enqueued. The audit checks `W` windows, so it never reads these rows past `W·T`. -/
def expectedRow (S T W : ℕ) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (t : ℕ) : Row S (List Bool) :=
  fifoGrid (t + 1) S (payloads frames) (enqueued T W syncs frames) (Fin.last t)

/-- **The constant-rate Program.** Windows of `T` buckets, `W` of them with advice, the sync set `ss`, and the Program's
frames `frames ins c ℓ` on each egress link `ℓ`, in canonical order, from the ingress `ins` and the content advice `c`.
With advice `(c, σ)`, link `ℓ`'s row at step `t` is `expectedRow` of those frames under the declared syncs `σ`. Every
link reads the same `σ`, one clock per spatial unit (`README.md`, modelling choice 15), which the audit enforces. -/
def constantRate {L S : ℕ} {In Ac : Type} (T W : ℕ) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) : Program L S In (Ac × (Fin W → ss.Elem)) where
  rows t ins a ℓ := expectedRow S T W (fun w => ss.sync (a.2 w)) (frames ins a.1 ℓ) t

end NetTiming
