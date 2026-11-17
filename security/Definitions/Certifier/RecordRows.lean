module

public import Definitions.Certifier.Run
public import Definitions.Certifier.Cells
public import Mathlib.Algebra.GroupWithZero.Nat
public import Mathlib.Data.Fin.VecNotation

/-! The network attester's record as rows, a row a bucket and each frame naming its connection; a lane the Python attester
fails for its client's sake, before `adm c + D`, is not one `certifierCode` gives. -/

@[expose] public section

namespace NetTiming

variable {κ Cell : Type}

def laneFrames (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (c : κ) (b : ℕ) : List (CellView Cell) :=
  (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r .pad else []) ++
    ((v c).filter fun x => x.1 == b).map Prod.snd

def bucketFrames (conns : List κ) (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (b : ℕ) :
    List (κ × CellView Cell) :=
  conns.flatMap fun c => (laneFrames s v c b).map (c, ·)

def lanesOfRows [DecidableEq κ] (s : CertifierSchedule κ) (rows : ℕ → List (κ × CellView Cell)) (c : κ) :
    List (ℕ × CellView Cell) :=
  (List.range (s.cells c)).map fun j =>
    (s.bucket c j, (((rows (s.bucket c j)).filterMap fun f => if f.1 = c then some f.2 else none)[j % s.r]?).getD .pad)

def laneLoad (s : CertifierSchedule κ) (c : κ) (b : ℕ) : ℕ :=
  (if s.adm c ≤ b ∧ b < s.adm c + s.D then s.r else 0) +
    ((List.range (s.cells c)).filter fun j => s.bucket c j == b).length

structure RowsFormat (κ Cell : Type) (L K : ℕ) where
  conns : List κ
  enc : κ × CellView Cell → List Bool
  dec : List Bool → Option (κ × CellView Cell)
  link : Fin L
  T : ℕ
  jitter : CertifierSchedule κ → (κ → Arrivals Cell) → ℕ → ℕ
  status : CertifierSchedule κ → (κ → Arrivals Cell) → ℕ → Fin K → Status

namespace RowsFormat

variable {L K : ℕ} (F : RowsFormat κ Cell L K)

def Wf : Prop :=
  F.conns.Nodup ∧ (∀ c, c ∈ F.conns) ∧ ∀ x, F.dec (F.enc x) = some x

def record (S n : ℕ) (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) :
    Fin n → List (Obs L K S) :=
  fun i => (List.range F.T).map fun t =>
    { rows := fun ℓ σ => if ℓ = F.link then ((bucketFrames F.conns s v (i * F.T + t))[σ.val]?).map F.enc else none
      jitter := F.jitter s a (i * F.T + t)
      status := F.status s a (i * F.T + t) }

def frames {S n : ℕ} (o : Fin n → List (Obs L K S)) (b : ℕ) : List (κ × CellView Cell) :=
  if h : b / F.T < n then
    match (o ⟨b / F.T, h⟩)[b % F.T]? with
    | some x => (List.finRange S).filterMap fun σ => (x.rows F.link σ).bind F.dec
    | none => []
  else []

def read [DecidableEq κ] {S n : ℕ} (s : CertifierSchedule κ) (o : Fin n → List (Obs L K S)) :
    κ → List (ℕ × CellView Cell) :=
  lanesOfRows s (F.frames o)

def Fits (S n : ℕ) (s : CertifierSchedule κ) : Prop :=
  (∀ c j, j < s.cells c → s.bucket c j < n * F.T) ∧ ∀ b, (F.conns.map fun c => laneLoad s c b).sum ≤ S

end RowsFormat

/-- `fxQueue`: 3 connections; the Python network attester's egress record, buckets 0 … 9 (every later bucket empty). -/
def fxQueueSched : CertifierSchedule (Fin 3) :=
  { r := 2, D := 3, adm := ![1, 2, 6], cells := ![6, 2, 2] }

end NetTiming
