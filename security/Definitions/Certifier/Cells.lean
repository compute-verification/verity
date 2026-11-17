module

public import Mathlib.Data.Nat.Notation

/-! The network attester's cells on a public schedule, its fail cell, and what the client sees. -/

@[expose] public section

namespace NetTiming

/-- One cell the client receives on a connection: `B` bytes of data, padding, or the fail cell, which carries no
reason. -/
inductive CellView (Cell : Type) where
  | data (x : Cell)
  | pad
  | fail

/-- **The network attester's schedule**, public: `r` cells per bucket and a latency of `D` buckets for every
connection, the admission bucket `adm c` of each connection `c` (the network attester's own decision), and the number
of egress cells `cells c` its lane has in the run. -/
structure CertifierSchedule (κ : Type) where
  r : ℕ
  D : ℕ
  adm : κ → ℕ
  cells : κ → ℕ

/-- The bucket in which connection `c` sends its egress cell `j`: `adm c + D + ⌊j/r⌋`. -/
def CertifierSchedule.bucket {κ : Type} (s : CertifierSchedule κ) (c : κ) (j : ℕ) : ℕ :=
  s.adm c + s.D + j / s.r

/-- **What reaches the network attester** from the server on one connection: its egress data in cells (`stream`), and
for each cell `j` the bucket by which all of it has arrived (`ready j`). `ready stream.length` is the bucket by which
the end of the stream has. A cell that never arrives has `ready` past every bucket of the run. -/
structure Arrivals (Cell : Type) where
  stream : List Cell
  ready : ℕ → ℕ

/-- Cell `j` of a connection whose data is `stream`, when it is neither late nor after a fail: its data, or padding
past the end. -/
def streamCell {Cell : Type} (stream : List Cell) (j : ℕ) : CellView Cell :=
  if h : j < stream.length then .data stream[j] else .pad

/-- **The network attester's code on one lane**, from the lane's cell `j` with `k` cells left, each sent in its bucket
`t j`. Past the end of the data, padding. A cell, or the end of the stream, late for its bucket: the fail cell, then
padding to the lane's end. Otherwise the cell's data. -/
def certifierLane {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) : ℕ → ℕ → List (ℕ × CellView Cell)
  | _, 0 => []
  | j, k + 1 =>
    if a.stream.length < j then (t j, .pad) :: certifierLane a t (j + 1) k
    else if t j < a.ready j then (t j, .fail) :: (List.range' (j + 1) k).map fun i => (t i, .pad)
    else (t j, streamCell a.stream j) :: certifierLane a t (j + 1) k

/-- **The network attester's code**: on schedule `s` and the arrivals `arr c` of every connection `c`, the cells the
client receives on each connection, each with its bucket. -/
def certifierCode {κ Cell : Type} (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) : κ → List (ℕ × CellView Cell) :=
  fun c => certifierLane (arr c) (s.bucket c) 0 (s.cells c)

/-- **The fail cell** of connection `c`: the first cell of its lane, among the data's cells and the end of the stream,
that reached the network attester after its bucket, or none. -/
def failCell {κ Cell : Type} (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) (c : κ) : Option ℕ :=
  (List.range (s.cells c)).find? fun j => decide (j ≤ (arr c).stream.length ∧ s.bucket c j < (arr c).ready j)

def cellView {Cell : Type} (stream : List Cell) : Option ℕ → ℕ → CellView Cell
  | none, j => streamCell stream j
  | some f, j => if j < f then streamCell stream j else if j = f then .fail else .pad

def clientView {κ Cell : Type} (s : CertifierSchedule κ) (data : κ → List Cell) (fail : κ → Option ℕ) :
    κ → List (ℕ × CellView Cell) :=
  fun c => (List.range (s.cells c)).map fun j => (s.bucket c j, cellView (data c) (fail c) j)

end NetTiming
