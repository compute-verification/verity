module

public import Definitions.Certifier.Run

/-! The Program's schedule: frames at their send ticks, one FIFO per execution block. -/

@[expose] public section

namespace NetTiming

structure LogicalFrame where
  payload : List Bool
  send : ℕ

def handingFrom : ℕ → List LogicalFrame → List ℕ
  | _, [] => []
  | last, f :: fs =>
    let h := max f.send last
    h :: handingFrom h fs

def enqueued (N base : ℕ) (frames : List LogicalFrame) (u : ℕ) : ℕ :=
  if u < N then (handingFrom base frames).count (base + u) else 0

/-- The FIFO certifier never releases more frames than were enqueued (`SumReleasedLe`), so the default past the end is never read. -/
def payloads (frames : List LogicalFrame) (k : ℕ) : List Bool :=
  (frames[k]?.map LogicalFrame.payload).getD []

def expectedRow (S N base : ℕ) (frames : List LogicalFrame) (u : ℕ) : Row S (List Bool) :=
  fifoGrid (u + 1) S (payloads frames) (enqueued N base frames) (Fin.last u)

/-- Every link reads the same advice: one clock per spatial unit. -/
def constantRate {L S : ℕ} {In Adv : Type} (T g E : ℕ)
    (frames : List In → Fin E → Adv → Fin L → List LogicalFrame) : EgressPrediction L S In (Fin E → Adv) where
  rows t ins a ℓ :=
    if h : t / (g * T) < E then
      expectedRow S (g * T) (t / (g * T) * (g * T)) (frames ins ⟨t / (g * T), h⟩ (a ⟨t / (g * T), h⟩) ℓ)
        (t % (g * T))
    else fun _ => none

end NetTiming
