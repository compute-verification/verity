import NetworkCertifier.Run

/-!
# The certifier program: the logical schedule, the clock-sync advice and the constant-rate Program

The executable forms of `Security/Definitions/NetworkCertifier/Schedule.lean` that the difftest runs, under the same names
in `NetworkCertifier`. They mirror `schedule.py` one for one, with each Python name in its docstring. Lean core only:
`Σ_sync` is two duplicate-free lists, as in Python, where the reference has two `Finset`s.
-/

namespace NetworkCertifier

/-- `ClockSync`: one window's clock-sync advice, the offset δ (buckets after the anchor) and the rate ρ (frames per
bucket within a session, a positive fraction). Python rejects `delta < 0` (here `Nat`) and `rho ≤ 0`. -/
structure ClockSync where
  delta : Nat
  rho : Rat
  rho_pos : 0 < rho

/-- `SyncSet`: `Σ_sync`, the values a window may declare, each δ of `deltas` with each ρ of `rhos`. Python requires both
lists non-empty, every δ ≥ 0 (`Nat`), every ρ > 0, and each value listed once (`Nodup`). -/
structure SyncSet where
  deltas : List Nat
  rhos : List Rat
  deltas_nonempty : deltas ≠ []
  rhos_nonempty : rhos ≠ []
  rhos_pos : ∀ ρ ∈ rhos, 0 < ρ
  deltas_nodup : deltas.Nodup
  rhos_nodup : rhos.Nodup

namespace SyncSet

/-- A value of `Σ_sync`, a δ of the set and a ρ of the set: what `SyncSet.__iter__` yields and `__contains__` accepts. -/
abbrev Elem (ss : SyncSet) : Type := {d // d ∈ ss.deltas} × {ρ // ρ ∈ ss.rhos}

/-- The clock sync a value of the set names. -/
def sync (ss : SyncSet) (v : ss.Elem) : ClockSync := ⟨v.1, v.2, ss.rhos_pos v.2 v.2.2⟩

end SyncSet

/-- `LogicalFrame`: a real frame as the Program fixes it, its content, its anchor and its logical index. -/
structure LogicalFrame where
  payload : List Bool
  anchor : Nat
  index : Nat

/-- `eligible`: the frame's logical release bucket, `anchor + δ + ⌊λ / ρ⌋`, computed as Python does,
`anchor + delta + (index * rho.denominator) // rho.numerator`. -/
def eligible (f : LogicalFrame) (s : ClockSync) : Nat :=
  f.anchor + s.delta + f.index * s.rho.den / s.rho.num.toNat

/-- The loop of `handing_buckets` from the hand time `last` of the frame before: frame `f`, anchored in window
`f.anchor / T`, is handed at `max(eligible f (syncs w), last)` if `w < W`, and otherwise (anchored past the last window
with advice) at `max(last, W·T, f.anchor)`, after the horizon. -/
def handingFrom (T W : Nat) (syncs : Fin W → ClockSync) : Nat → List LogicalFrame → List Nat
  | _, [] => []
  | last, f :: fs =>
    let h := if hw : f.anchor / T < W then max (eligible f (syncs ⟨f.anchor / T, hw⟩)) last
      else max (max last (W * T)) f.anchor
    h :: handingFrom T W syncs h fs

/-- `handing_buckets`: when the shaper hands each frame to the certifier, `h(k) = max(eligible(k), h(k−1))` in canonical
order, starting from `last = 0`. -/
def handingBuckets (T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) : List Nat :=
  handingFrom T W syncs 0 frames

/-- The enqueue counts of `expected_grids`: bucket `t` of the horizon `W·T` receives the frames handed at `t`. Frames
handed at or after the horizon are never enqueued. -/
def enqueued (T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (t : Nat) : Nat :=
  if t < W * T then (handingBuckets T W syncs frames).count t else 0

/-- The payloads in canonical order, `[f.payload for f in frames]`. -/
def payloads (frames : List LogicalFrame) (k : Nat) : List Bool :=
  (frames[k]?.map LogicalFrame.payload).getD []

/-- `expected_grids`, read at bucket `t`: the certifier's rule (FIFO at the allocation `S`, left-packed, the queue carried
across windows) applied to the enqueue counts, with the frames leaving in canonical order. -/
def expectedRow (S T W : Nat) (syncs : Fin W → ClockSync) (frames : List LogicalFrame) (t : Nat) : Row S (List Bool) :=
  fifoGrid (t + 1) S (payloads frames) (enqueued T W syncs frames) (Fin.last t)

/-- **The constant-rate Program's egress prediction.** Windows of `T` buckets, `W` of them with advice, the sync set `ss`,
and the Program's frames `frames ins c ℓ` on each egress link `ℓ`; with advice `(c, σ)`, link `ℓ`'s row at step `t` is
`expectedRow` of those frames under the declared syncs `σ`, the same on every link. -/
def constantRate {L S : Nat} {In Ac : Type} (T W : Nat) (ss : SyncSet)
    (frames : List In → Ac → Fin L → List LogicalFrame) : EgressPrediction L S In (Ac × (Fin W → ss.Elem)) where
  rows t ins a ℓ := expectedRow S T W (fun w => ss.sync (a.2 w)) (frames ins a.1 ℓ) t

end NetworkCertifier
