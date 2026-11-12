module

public import Proofs.NetworkCertifier.RecordRows
public import Definitions.NetworkCertifier.PeriodConditions
meta import Definitions.NetworkCertifier.RecordRows
meta import Proofs.NetworkCertifier.RecordRows

/-!
# The network certifier's rows read back from the padding frames' count alone

`Proofs.NetworkCertifier.RecordRows` reads a connection's lane back from the rows given the whole schedule: the
admission and the lane's length say where each cell is (`lanesOfRows`). The rows hold more than that read uses. Each
frame names its connection; a connection's frames are `r` padding frames a bucket for the `D` buckets from its
admission, then its lane's cells, in buckets that never decrease. So a connection's frames in bucket order, less the
first `r · D`, are its lane's cells, each in its bucket (`lanesOfRowsPadded`), and the read takes the product `r · D`
and nothing else of the schedule.

* A lane with a cell has all `r · D` padding frames inside the monitoring period, before its first cell: its first
  cell's bucket is one of the period's (`RowsFormat.Fits`), and the padding's buckets are before it.
* A lane of no cells has at most `r · D` frames in the period, all padding, so nothing is left: the empty lane, whatever
  its admission, and whether or not the period's end cuts its padding.
* With `r = 0` or `D = 0` there is no padding frame and nothing is dropped.

So the rows format reads back under the pair `(r, D)` (`RowsReadPadded`, `RecordFormat.ReadsBy` with the parameters
`ℕ × ℕ`): the registered parameters of a period's schedules are one pair, where the format's own read takes each
schedule as its own parameter (`RecordFormat.readsBy_read`).

The `#guard`s hold the read to the code's cells on the Python network certifier's record `fxQueue` and at the edges: no
cell a bucket, no latency with two lanes admitted in one bucket, a lane of no cells whose padding the period's end
cuts, and one lane of one cell admitted at each of 31 buckets.
-/

@[expose] public section

namespace NetTiming

variable {κ Cell : Type}

/-- **The lanes read back from the padding frames' count**: connection `c`'s frames in the rows of the buckets before
`nb`, bucket by bucket in order, each with its bucket, less the first `pads`. -/
def lanesOfRowsPadded [DecidableEq κ] (pads nb : ℕ) (rows : ℕ → List (κ × CellView Cell)) (c : κ) :
    List (ℕ × CellView Cell) :=
  ((List.range nb).flatMap fun b =>
    ((rows b).filterMap fun f => if f.1 = c then some f.2 else none).map fun x => (b, x)).drop pads

namespace RowsFormat

variable {L K : ℕ}

/-- **The read from the padding frames' count**: the lanes read back (`lanesOfRowsPadded`) from the frames off the
record, over the monitoring period's `n · T` buckets. -/
def readPadded [DecidableEq κ] (F : RowsFormat κ Cell L K) {S n : ℕ} (pads : ℕ) (o : Fin n → List (Obs L K S)) :
    κ → List (ℕ × CellView Cell) :=
  lanesOfRowsPadded pads (n * F.T) (F.frames o)

end RowsFormat

/-- **The rows format reads back under its cells a bucket and its latency.** On every schedule with `r` cells a bucket
and a latency of `D` buckets that fits the rows format, the read from `r · D` padding frames (`RowsFormat.readPadded`)
of the record of the cells the network certifier's code sends is those cells: `RecordFormat.ReadsBy` at the parameters
`(r, D)`. -/
def RowsReadPadded : Prop :=
  ∀ (κ Cell : Type) [DecidableEq κ] (L K : ℕ) (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ),
    (F.recordFormat hF S n).ReadsBy (fun θ : ℕ × ℕ => F.readPadded (θ.1 * θ.2)) fun θ s => s.r = θ.1 ∧ s.D = θ.2

/-! ### The read at the edges -/

section Fixtures

/-- The lanes read back from `r · D` padding frames off the rows of the code's cells, over `nb` buckets, are the code's
cells. -/
def paddedAgrees {n : ℕ} (s : CertifierSchedule (Fin n)) (arr : Fin n → Arrivals (Bool × String)) (nb : ℕ) : Bool :=
  (List.finRange n).all fun c =>
    (lanesOfRowsPadded (s.r * s.D) nb (bucketFrames (List.finRange n) s (certifierCode s arr)) c).map pyCell ==
      (certifierCode s arr c).map pyCell

/-- Arrivals of three cells, the last the END, all in time: a longer lane's later cells are padding, which the read
must tell from the padding frames before the lane. -/
def edgeArr {n : ℕ} : Fin n → Arrivals (Bool × String) :=
  fun _ => { stream := [(false, "0a"), (false, "0b"), (true, "0c")], ready := fun _ => 0 }

-- The queue fixture's schedule and arrivals, over its 12 buckets.
#guard paddedAgrees fxQueueSched fxQueueArr 12

-- No cell a bucket (`r = 0`): no padding frame, and every cell of a lane in its first bucket.
#guard paddedAgrees (n := 2) { r := 0, D := 2, adm := ![1, 1], cells := ![3, 0] } edgeArr 4

-- No latency (`D = 0`), two lanes admitted in one bucket.
#guard paddedAgrees (n := 2) { r := 2, D := 0, adm := ![0, 0], cells := ![5, 1] } edgeArr 3

-- A lane of no cells whose padding the period's end cuts, beside a lane whose cells past its data are padding.
#guard paddedAgrees (n := 2) { r := 2, D := 3, adm := ![4, 0], cells := ![0, 4] } edgeArr 5

-- One lane of one cell, admitted at each bucket from 0 to 30 of 32: each admission reads back at the one pair.
#guard (List.range 31).all fun a => paddedAgrees (n := 1) { r := 1, D := 1, adm := ![a], cells := ![1] } edgeArr 32

end Fixtures

end NetTiming
