import Mathlib.Data.Nat.Notation

/-!
# The warden: cells on a public schedule, the fail cell, and what the client sees

Definitions only (design v3). An active **warden** terminates both
connections, the client's and the server's, and re-frames every byte into cells of `B` bytes per direction, sent on a
public schedule: `r` cells per bucket and a latency of `D` buckets. A connection admitted at bucket `s` sends its egress
cell `j` (its bytes `B·j` to `B·j + B − 1`) in bucket `s + D + ⌊j/r⌋`, so its egress byte `i` goes out in bucket
`s + D + ⌊i/(B·r)⌋`. The admission bucket is the warden's decision on its own schedule, never anything the server times.

A cell past the connection's data is padding. A cell whose bytes haven't reached the warden by its bucket is late: the
warden fails the connection closed there, with one fail cell that carries no reason, and holds the lane to its scheduled
end with padding. The end of the data counts as a cell too: the server's end of stream must reach the warden by the
bucket of the cell after the data's last, or the connection fails there.

NCI's transfer gates `T` are these cells' bits (`transfers`, `Definitions.Nci`): their number per
window is the schedule's, so it is public.

**What is abstract.** A cell's `B` bytes are a value of `Cell`. Whatever the server does (its accept time, its send
timing, its ACKs and flow control) reaches the warden only as its connections' `Arrivals`: the data, in cells, and the
bucket by which each cell has arrived.
-/

namespace NetTiming

/-- One cell the client receives on a connection: `B` bytes of data, padding, or the fail cell, which carries no
reason. -/
inductive CellView (Cell : Type) where
  | data (x : Cell)
  | pad
  | fail

/-- **The warden's schedule**, public: `r` cells per bucket and a latency of `D` buckets for every connection, the
admission bucket `adm c` of each connection `c` (the warden's own decision), and the number of egress cells `cells c`
its lane has in the run. -/
structure WardenSchedule (κ : Type) where
  r : ℕ
  D : ℕ
  adm : κ → ℕ
  cells : κ → ℕ

/-- The bucket in which connection `c` sends its egress cell `j`: `adm c + D + ⌊j/r⌋`. -/
def WardenSchedule.bucket {κ : Type} (s : WardenSchedule κ) (c : κ) (j : ℕ) : ℕ :=
  s.adm c + s.D + j / s.r

/-- **What reaches the warden** from the server on one connection: its egress data in cells (`stream`), and for each
cell `j` the bucket by which all of it has arrived (`ready j`). `ready stream.length` is the bucket by which the end of
the stream has. A cell that never arrives has `ready` past every bucket of the run. -/
structure Arrivals (Cell : Type) where
  stream : List Cell
  ready : ℕ → ℕ

/-- Cell `j` of a connection whose data is `stream`, when it is neither late nor after a fail: its data, or padding
past the end. -/
def streamCell {Cell : Type} (stream : List Cell) (j : ℕ) : CellView Cell :=
  if h : j < stream.length then .data stream[j] else .pad

/-- **The warden's code on one lane**, from the lane's cell `j` with `k` cells left, each sent in its bucket `t j`.
Past the end of the data, padding. A cell, or the end of the stream, late for its bucket: the fail cell, then padding
to the lane's end. Otherwise the cell's data. -/
def wardenLane {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) : ℕ → ℕ → List (ℕ × CellView Cell)
  | _, 0 => []
  | j, k + 1 =>
    if a.stream.length < j then (t j, .pad) :: wardenLane a t (j + 1) k
    else if t j < a.ready j then (t j, .fail) :: (List.range' (j + 1) k).map fun i => (t i, .pad)
    else (t j, streamCell a.stream j) :: wardenLane a t (j + 1) k

/-- **The warden's code**: on schedule `s` and the arrivals `arr c` of every connection `c`, the cells the client
receives on each connection, each with its bucket. -/
def wardenCode {κ Cell : Type} (s : WardenSchedule κ) (arr : κ → Arrivals Cell) : κ → List (ℕ × CellView Cell) :=
  fun c => wardenLane (arr c) (s.bucket c) 0 (s.cells c)

/-- **The fail cell** of connection `c`: the first cell of its lane, among the data's cells and the end of the stream,
that reached the warden after its bucket, or none. -/
def failCell {κ Cell : Type} (s : WardenSchedule κ) (arr : κ → Arrivals Cell) (c : κ) : Option ℕ :=
  (List.range (s.cells c)).find? fun j => decide (j ≤ (arr c).stream.length ∧ s.bucket c j < (arr c).ready j)

/-- Cell `j` the client receives on a connection whose data is `stream` and whose fail cell is `fail`. -/
def cellView {Cell : Type} (stream : List Cell) : Option ℕ → ℕ → CellView Cell
  | none, j => streamCell stream j
  | some f, j => if j < f then streamCell stream j else if j = f then .fail else .pad

/-- **The client's view** of the server side, connection by connection, from the schedule `s`, each connection's data
and its fail cell or none: every cell of its lane, in its bucket. -/
def clientView {κ Cell : Type} (s : WardenSchedule κ) (data : κ → List Cell) (fail : κ → Option ℕ) :
    κ → List (ℕ × CellView Cell) :=
  fun c => (List.range (s.cells c)).map fun j => (s.bucket c j, cellView (data c) (fail c) j)

end NetTiming
