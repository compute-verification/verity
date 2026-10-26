import Definitions.NetworkCertifier.Tap
import Definitions.NetworkCertifier.Run

/-!
# The network certifier's record as rows: a row a bucket, each frame naming its connection

Definitions only. The record the deployed network certifier writes, as a `TapRecord` format
(`Definitions.NetworkCertifier.Tap`): the egress record of `verity/core/protocols/certifier/warden.py`, whose rows
`verity/core/protocols/certifier/warden_rows.py` commits. Bucket `b`'s row holds the frames of every connection live in
it, connection by connection in ordinal order (`conns`), each frame naming its connection (`bucketFrames`): `r` padding
frames a bucket from the connection's admission to `adm c + D`, then its lane's cells in their bucket, in order. The
frames fill the row's first slots, each encoded as a `List Bool` (`enc`), and a slot no connection holds is empty
(`none`: the Python record's `None`, `B` zero bytes in its rows). Window `i` of the monitoring period's `n` windows of
`T` buckets is buckets `i·T … i·T + T − 1`, on one egress link (`link`); a bucket's jitter class and statuses are the
network certifier's, from anything it saw.

Read back (`lanesOfRows`), connection `c`'s cell `j`, for `j < cells c`, is the `(j mod r)`-th of its frames in bucket
`adm c + D + ⌊j/r⌋`: `cells c` is the lane's length through its last bucket. A schedule fits the format
(`RowsFormat.Fits`) when its lanes' cells fall in the monitoring period's buckets and no bucket holds more than `S`
frames, which the network certifier's admission keeps (at most `lanes` connections a bucket, `r` frames each, and
`S = lanes·r`).

**What the Python records add.** A connection the network certifier fails for its client's sake (a malformed cell, or
ingress past its bound) fails at once, before `adm c + D` too; `wardenLane` fails a lane only at a late cell or a late
end, so such a lane is not one `wardenCode` gives. A lane whose server sends past `egress_cells` cells fails at the last
of them (limit): its arrivals are the ones whose cell there never arrives.
-/

namespace NetTiming

variable {κ Cell : Type}

/-- The frames connection `c` sends in bucket `b`, as its record holds them: `r` padding frames in each bucket from its
admission to `adm c + D`, then the cells `v c` of its lane in bucket `b`, in order. -/
def laneFrames (s : WardenSchedule κ) (v : κ → List (ℕ × CellView Cell)) (c : κ) (b : ℕ) : List (CellView Cell) :=
  (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r .pad else []) ++
    ((v c).filter fun x => x.1 == b).map Prod.snd

/-- **Bucket `b`'s row of frames**: each connection's frames in the bucket (`laneFrames`), connection by connection in
the order `conns` (the ordinals), each frame naming its connection. -/
def bucketFrames (conns : List κ) (s : WardenSchedule κ) (v : κ → List (ℕ × CellView Cell)) (b : ℕ) :
    List (κ × CellView Cell) :=
  conns.flatMap fun c => (laneFrames s v c b).map (c, ·)

/-- **The lanes read back** from rows of frames under the schedule: connection `c`'s cell `j`, for `j < cells c`, is the
`(j mod r)`-th of its frames in its bucket `adm c + D + ⌊j/r⌋`. -/
def lanesOfRows [DecidableEq κ] (s : WardenSchedule κ) (rows : ℕ → List (κ × CellView Cell)) (c : κ) :
    List (ℕ × CellView Cell) :=
  (List.range (s.cells c)).map fun j =>
    (s.bucket c j, (((rows (s.bucket c j)).filterMap fun f => if f.1 = c then some f.2 else none)[j % s.r]?).getD .pad)

/-- The number of frames connection `c` sends in bucket `b` on schedule `s`, whatever its cells hold. -/
def laneLoad (s : WardenSchedule κ) (c : κ) (b : ℕ) : ℕ :=
  (if s.adm c ≤ b ∧ b < s.adm c + s.D then s.r else 0) +
    ((List.range (s.cells c)).filter fun j => s.bucket c j == b).length

/-- **The rows format**: the connections in ordinal order (`conns`), a frame's encoding (`enc`) and its decoding
(`dec`), the egress link the rows are on (`link`), the buckets a window (`T`), and each bucket's jitter class and
statuses, from the schedule and what reached the network certifier. -/
structure RowsFormat (κ Cell : Type) (L K : ℕ) where
  conns : List κ
  enc : κ × CellView Cell → List Bool
  dec : List Bool → Option (κ × CellView Cell)
  link : Fin L
  T : ℕ
  jitter : WardenSchedule κ → (κ → Arrivals Cell) → ℕ → ℕ
  status : WardenSchedule κ → (κ → Arrivals Cell) → ℕ → Fin K → Status

namespace RowsFormat

variable {L K : ℕ} (F : RowsFormat κ Cell L K)

/-- The format is well formed: every connection is in `conns`, once, and `dec` reads back every frame `enc` writes. -/
def Wf : Prop :=
  F.conns.Nodup ∧ (∀ c, c ∈ F.conns) ∧ ∀ x, F.dec (F.enc x) = some x

/-- **The record** over `n` windows of `S` slots: window `i`'s step `t` is bucket `i·T + t`, whose row on the link holds
the bucket's frames (`bucketFrames`) in its first slots, encoded, and nothing in the rest; every other link's row is
empty. -/
def record (S n : ℕ) (s : WardenSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) :
    Fin n → List (Obs L K S) :=
  fun i => (List.range F.T).map fun t =>
    { rows := fun ℓ σ => if ℓ = F.link then ((bucketFrames F.conns s v (i * F.T + t))[σ.val]?).map F.enc else none
      jitter := F.jitter s a (i * F.T + t)
      status := F.status s a (i * F.T + t) }

/-- Bucket `b`'s frames read off a record: the decoded frames of its row on the link, in slot order. -/
def frames {S n : ℕ} (o : Fin n → List (Obs L K S)) (b : ℕ) : List (κ × CellView Cell) :=
  if h : b / F.T < n then
    match (o ⟨b / F.T, h⟩)[b % F.T]? with
    | some x => (List.finRange S).filterMap fun σ => (x.rows F.link σ).bind F.dec
    | none => []
  else []

/-- **The read**: the lanes read back (`lanesOfRows`) from the frames off the record. -/
def read [DecidableEq κ] {S n : ℕ} (s : WardenSchedule κ) (o : Fin n → List (Obs L K S)) :
    κ → List (ℕ × CellView Cell) :=
  lanesOfRows s (F.frames o)

/-- **A schedule fits the format** over `n` windows of `S` slots: every cell of every lane is in one of the monitoring
period's `n·T` buckets, and no bucket holds more than `S` frames. -/
def Fits (S n : ℕ) (s : WardenSchedule κ) : Prop :=
  (∀ c j, j < s.cells c → s.bucket c j < n * F.T) ∧ ∀ b, (F.conns.map fun c => laneLoad s c b).sum ≤ S

end RowsFormat

end NetTiming
