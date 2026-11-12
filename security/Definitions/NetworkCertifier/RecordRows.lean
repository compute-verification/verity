module

public import Definitions.NetworkCertifier.Run
public import Definitions.NetworkCertifier.Warden
public import Mathlib.Algebra.GroupWithZero.Nat
public import Mathlib.Data.Fin.VecNotation

/-!
# The network certifier's record as rows: a row a bucket, each frame naming its connection

Definitions only. The record the deployed network certifier writes, as a `RecordFormat` format
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

**The deployed format** (`cellRows`) is `warden.py`'s frames of `B` bytes: a kind u8, the connection's ordinal u32, the
payload's length u32, the payload and zero fill, the `c`-th of `N` connections at ordinal `c + 1`. It is well formed
(`RowsFormat.Wf`) when `B` and `N` are under 2^32.

**What the Python records add.** A connection the network certifier fails for its client's sake (a malformed cell, or
ingress past its bound) fails at once, before `adm c + D` too; `certifierLane` fails a lane only at a late cell or a late
end, so such a lane is not one `certifierCode` gives. A lane whose server sends past `egress_cells` cells fails at the last
of them (limit): its arrivals are the ones whose cell there never arrives.

`fxQueueSched` is the schedule of the fixture `fxQueue` (three connections on two lanes), one of the Python network
certifier's egress records that `Proofs.NetworkCertifier.RecordRows` holds the format to.
-/

@[expose] public section

namespace NetTiming

variable {κ Cell : Type}

/-- The frames connection `c` sends in bucket `b`, as its record holds them: `r` padding frames in each bucket from its
admission to `adm c + D`, then the cells `v c` of its lane in bucket `b`, in order. -/
def laneFrames (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (c : κ) (b : ℕ) : List (CellView Cell) :=
  (if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r .pad else []) ++
    ((v c).filter fun x => x.1 == b).map Prod.snd

/-- **Bucket `b`'s row of frames**: each connection's frames in the bucket (`laneFrames`), connection by connection in
the order `conns` (the ordinals), each frame naming its connection. -/
def bucketFrames (conns : List κ) (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (b : ℕ) :
    List (κ × CellView Cell) :=
  conns.flatMap fun c => (laneFrames s v c b).map (c, ·)

/-- **The lanes read back** from rows of frames under the schedule: connection `c`'s cell `j`, for `j < cells c`, is the
`(j mod r)`-th of its frames in its bucket `adm c + D + ⌊j/r⌋`. -/
def lanesOfRows [DecidableEq κ] (s : CertifierSchedule κ) (rows : ℕ → List (κ × CellView Cell)) (c : κ) :
    List (ℕ × CellView Cell) :=
  (List.range (s.cells c)).map fun j =>
    (s.bucket c j, (((rows (s.bucket c j)).filterMap fun f => if f.1 = c then some f.2 else none)[j % s.r]?).getD .pad)

/-- The number of frames connection `c` sends in bucket `b` on schedule `s`, whatever its cells hold. -/
def laneLoad (s : CertifierSchedule κ) (c : κ) (b : ℕ) : ℕ :=
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
  jitter : CertifierSchedule κ → (κ → Arrivals Cell) → ℕ → ℕ
  status : CertifierSchedule κ → (κ → Arrivals Cell) → ℕ → Fin K → Status

namespace RowsFormat

variable {L K : ℕ} (F : RowsFormat κ Cell L K)

/-- The format is well formed: every connection is in `conns`, once, and `dec` reads back every frame `enc` writes. -/
def Wf : Prop :=
  F.conns.Nodup ∧ (∀ c, c ∈ F.conns) ∧ ∀ x, F.dec (F.enc x) = some x

/-- **The record** over `n` windows of `S` slots: window `i`'s step `t` is bucket `i·T + t`, whose row on the link holds
the bucket's frames (`bucketFrames`) in its first slots, encoded, and nothing in the rest; every other link's row is
empty. -/
def record (S n : ℕ) (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) :
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
def read [DecidableEq κ] {S n : ℕ} (s : CertifierSchedule κ) (o : Fin n → List (Obs L K S)) :
    κ → List (ℕ × CellView Cell) :=
  lanesOfRows s (F.frames o)

/-- **A schedule fits the format** over `n` windows of `S` slots: every cell of every lane is in one of the monitoring
period's `n·T` buckets, and no bucket holds more than `S` frames. -/
def Fits (S n : ℕ) (s : CertifierSchedule κ) : Prop :=
  (∀ c j, j < s.cells c → s.bucket c j < n * F.T) ∧ ∀ b, (F.conns.map fun c => laneLoad s c b).sum ≤ S

end RowsFormat

/-- `fxQueue`: 3 connections; the Python network certifier's egress record, buckets 0 … 9 (every later bucket empty). -/
def fxQueueSched : CertifierSchedule (Fin 3) :=
  { r := 2, D := 3, adm := ![1, 2, 6], cells := ![6, 2, 2] }

/-! ## The deployed rows format -/

/-- A cell's content in the network certifier's frames of `P` payload bytes (`warden.py`): `DATA`'s bytes, or a
direction's last `0 … P` of them, `END`'s (`last`). -/
structure FrameCell (P : ℕ) where
  last : Bool
  payload : List (Fin 256)
  fits : payload.length ≤ P

/-- `x`'s low `w` bits, the most significant first. -/
def bitsBE (w x : ℕ) : List Bool :=
  List.ofFn fun i : Fin w => x.testBit (w - 1 - i)

/-- A frame's kind (`warden.py`: `PAD` 0, `DATA` 1, `END` 32, `FAIL` 33) and its payload. -/
def frameKind {P : ℕ} : CellView (FrameCell P) → ℕ × List (Fin 256)
  | .data x => (if x.last then 32 else 1, x.payload)
  | .pad => (0, [])
  | .fail => (33, [])

/-- **The network certifier's frame** of `B` bytes for connection `c` (`active.encode_frame`), as bits: its kind u8,
its connection's ordinal u32, `c + 1` (0 is a refusal's), its payload's length u32, the payload, and zero fill to `B`
bytes, past the 9-byte header. -/
def cellFrame {N : ℕ} (B : ℕ) (x : Fin N × CellView (FrameCell (B - 9))) : List Bool :=
  bitsBE 8 (frameKind x.2).1 ++ bitsBE 32 (x.1.val + 1) ++ bitsBE 32 (frameKind x.2).2.length ++
    (frameKind x.2).2.flatMap (fun b => bitsBE 8 b.val) ++
    List.replicate (8 * (B - 9 - (frameKind x.2).2.length)) false

open Classical in
/-- **The deployed rows format** (`warden_rows.py`'s rows of `warden.py`'s frames): `N` connections in ordinal order,
the `c`-th at ordinal `c + 1`; a frame is `B` bytes (`cellFrame`), and reads as the connection and cell whose frame it
is, and as nothing else; on the egress link `link`, `T` buckets a window, with the network certifier's jitter classes
and statuses. -/
noncomputable def cellRows {L K : ℕ} (B N : ℕ) (link : Fin L) (T : ℕ)
    (jitter : CertifierSchedule (Fin N) → (Fin N → Arrivals (FrameCell (B - 9))) → ℕ → ℕ)
    (status : CertifierSchedule (Fin N) → (Fin N → Arrivals (FrameCell (B - 9))) → ℕ → Fin K → Status) :
    RowsFormat (Fin N) (FrameCell (B - 9)) L K where
  conns := List.finRange N
  enc := cellFrame B
  dec f := if h : ∃ x, cellFrame B x = f then some (Classical.choose h) else none
  link := link
  T := T
  jitter := jitter
  status := status

end NetTiming
