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

/-! ### The read back

A lane's frames in bucket order are its padding frames, then its cells (`flatMap_range_append`: no padding frame is in
a bucket from `adm c + D` on and no cell in a bucket before it). The cells' buckets never decrease, so bucket by bucket
they are the lane (`flatMap_range_filter`). The padding frames in the period number `r · D` when the lane has a cell and
at most `r · D` when it has none (`length_flatMap_padFrames`), so dropping `r · D` frames leaves the lane. -/

/-- Connection `c`'s padding frames in bucket `b`, each with its bucket. -/
def padFrames (s : CertifierSchedule κ) (c : κ) (b : ℕ) : List (ℕ × CellView Cell) :=
  if s.adm c ≤ b ∧ b < s.adm c + s.D then List.replicate s.r (b, .pad) else []

/-- A connection's frames in a bucket, each with its bucket: its padding frames, then its cells in the bucket. -/
theorem map_laneFrames (s : CertifierSchedule κ) (v : κ → List (ℕ × CellView Cell)) (c : κ) (b : ℕ) :
    (laneFrames s v c b).map (fun x => (b, x)) = padFrames s c b ++ (v c).filter fun x => x.1 == b := by
  have hcells : (((v c).filter fun x => x.1 == b).map Prod.snd).map (fun x => (b, x)) =
      (v c).filter fun x => x.1 == b := by
    rw [List.map_map]
    refine (List.map_congr_left fun x hx => ?_).trans (List.map_id _)
    have hb : x.1 = b := by simpa using (List.mem_filter.1 hx).2
    exact Prod.ext hb.symm rfl
  unfold laneFrames padFrames
  rw [List.map_append, hcells]
  congr 1
  split_ifs <;> simp

/-- The padding frames in the buckets before `nb` number `r` for each bucket of `adm c … adm c + D − 1` before `nb`. -/
theorem length_flatMap_padFrames (s : CertifierSchedule κ) (c : κ) :
    ∀ nb, ((List.range nb).flatMap (padFrames (Cell := Cell) s c)).length =
      s.r * (min nb (s.adm c + s.D) - min nb (s.adm c))
  | 0 => by simp
  | nb + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, length_flatMap_padFrames s c nb,
      List.flatMap_singleton]
    by_cases h : s.adm c ≤ nb ∧ nb < s.adm c + s.D
    · have e : min (nb + 1) (s.adm c + s.D) - min (nb + 1) (s.adm c) =
          (min nb (s.adm c + s.D) - min nb (s.adm c)) + 1 := by omega
      rw [e, Nat.mul_succ]
      simp [padFrames, h]
    · have e : min (nb + 1) (s.adm c + s.D) - min (nb + 1) (s.adm c) =
          min nb (s.adm c + s.D) - min nb (s.adm c) := by omega
      rw [e]
      simp [padFrames, h]

/-- Lists by bucket, `f`'s empty from bucket `m` on and `g`'s empty before it: bucket by bucket, `f`'s then `g`'s
are all of `f`'s, then all of `g`'s. -/
theorem flatMap_range_append {β : Type} (f g : ℕ → List β) (m : ℕ) (hf : ∀ b, m ≤ b → f b = [])
    (hg : ∀ b, b < m → g b = []) :
    ∀ nb, ((List.range nb).flatMap fun b => f b ++ g b) = (List.range nb).flatMap f ++ (List.range nb).flatMap g
  | 0 => by simp
  | nb + 1 => by
    simp only [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rw [flatMap_range_append f g m hf hg nb]
    rcases Nat.lt_or_ge nb m with h | h
    · have hG : (List.range nb).flatMap g = [] :=
        List.flatMap_eq_nil_iff.2 fun b hb => hg b (Nat.lt_trans (List.mem_range.1 hb) h)
      rw [hG, hg nb h]
      simp
    · rw [hf nb h]
      simp

/-- In a list whose buckets never decrease, the elements before bucket `nb`, then those in it, are those before
bucket `nb + 1`. -/
theorem filter_lt_append_filter_eq {α : Type} (nb : ℕ) :
    ∀ l : List (ℕ × α), (l.Pairwise fun x y => x.1 ≤ y.1) →
      (l.filter fun x => decide (x.1 < nb)) ++ (l.filter fun x => x.1 == nb) =
        l.filter fun x => decide (x.1 < nb + 1)
  | [], _ => rfl
  | x :: xs, h => by
    obtain ⟨hx, hxs⟩ := List.pairwise_cons.1 h
    have ih := filter_lt_append_filter_eq nb xs hxs
    rcases Nat.lt_trichotomy x.1 nb with hlt | heq | hgt
    · have h2 : x.1 ≠ nb := by omega
      have h3 : x.1 < nb + 1 := by omega
      have h4 : x.1 ≤ nb := by omega
      simpa [List.filter_cons, hlt, h2, h3, h4] using ih
    · subst heq
      have hnil : (xs.filter fun y => decide (y.1 < x.1)) = [] :=
        List.filter_eq_nil_iff.2 fun y hy => by
          have := hx y hy
          simp only [decide_eq_true_eq]
          omega
      simpa [List.filter_cons, hnil] using ih
    · have h1 : ¬ x.1 < nb := by omega
      have h2 : x.1 ≠ nb := by omega
      have h3 : ¬ x.1 < nb + 1 := by omega
      have h4 : ¬ x.1 ≤ nb := by omega
      simpa [List.filter_cons, h1, h2, h3, h4] using ih

/-- A list whose buckets never decrease, read bucket by bucket through the buckets before `nb`, is its elements
before bucket `nb`. -/
theorem flatMap_range_filter {α : Type} (l : List (ℕ × α)) (hl : l.Pairwise fun x y => x.1 ≤ y.1) :
    ∀ nb, ((List.range nb).flatMap fun b => l.filter fun x => x.1 == b) = l.filter fun x => decide (x.1 < nb)
  | 0 => (List.filter_eq_nil_iff.2 fun x _ => by simp).symm
  | nb + 1 => by
    rw [List.range_succ, List.flatMap_append, flatMap_range_filter l hl nb, List.flatMap_singleton]
    exact filter_lt_append_filter_eq nb l hl

/-- **A lane's frames in bucket order, less its `r · D` padding frames, are its cells**, when the cells' buckets never
decrease, none is before `adm c + D` and all are before `nb`. -/
theorem drop_flatMap_padFrames (s : CertifierSchedule κ) (c : κ) (lane : List (ℕ × CellView Cell)) (nb : ℕ)
    (hsorted : lane.Pairwise fun x y => x.1 ≤ y.1) (hlo : ∀ x ∈ lane, s.adm c + s.D ≤ x.1)
    (hhi : ∀ x ∈ lane, x.1 < nb) :
    ((List.range nb).flatMap fun b => padFrames s c b ++ lane.filter fun x => x.1 == b).drop (s.r * s.D) = lane := by
  have hsplit := flatMap_range_append (padFrames (Cell := Cell) s c) (fun b => lane.filter fun x => x.1 == b)
    (s.adm c + s.D) (fun b hb => if_neg (by omega))
    (fun b hb => List.filter_eq_nil_iff.2 fun x hx => by
      have := hlo x hx
      simp only [beq_iff_eq]
      omega) nb
  rw [hsplit, flatMap_range_filter lane hsorted nb, List.filter_eq_self.2 fun x hx => decide_eq_true (hhi x hx)]
  rcases lane with _ | ⟨x, xs⟩
  · rw [List.append_nil]
    refine List.drop_of_length_le ?_
    rw [length_flatMap_padFrames]
    exact Nat.mul_le_mul_left _ (by omega)
  · have h1 := hlo x List.mem_cons_self
    have h2 := hhi x List.mem_cons_self
    refine List.drop_left' ?_
    have e : min nb (s.adm c + s.D) - min nb (s.adm c) = s.D := by omega
    rw [length_flatMap_padFrames, e]

/-- The buckets of the code's cells on a lane never decrease. -/
theorem certifierCode_sorted (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (certifierCode s a c).Pairwise fun x y => x.1 ≤ y.1 := by
  have h : ((certifierCode s a c).map Prod.fst).Pairwise (· ≤ ·) := by
    rw [certifierCode_fst]
    exact List.pairwise_map.2 (List.pairwise_lt_range.imp fun hij =>
      Nat.add_le_add_left (Nat.div_le_div_right (Nat.le_of_lt hij)) _)
  exact List.pairwise_map.1 h

/-- A cell of the code's on a lane is in the bucket of one of the lane's cells. -/
theorem certifierCode_mem_fst (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) {x : ℕ × CellView Cell}
    (hx : x ∈ certifierCode s a c) : ∃ j < s.cells c, x.1 = s.bucket c j := by
  have h : x.1 ∈ (certifierCode s a c).map Prod.fst := List.mem_map.2 ⟨x, hx, rfl⟩
  rw [certifierCode_fst] at h
  obtain ⟨j, hj, e⟩ := List.mem_map.1 h
  exact ⟨j, List.mem_range.1 hj, e.symm⟩

namespace RowsFormat

variable {L K : ℕ}

/-- **The rows format reads back the code's cells from the padding frames' count** on every schedule that fits it. -/
theorem readPadded_record [DecidableEq κ] (F : RowsFormat κ Cell L K) (hF : F.Wf) (S n : ℕ)
    (s : CertifierSchedule κ) (hs : F.Fits S n s) (a : κ → Arrivals Cell) :
    F.readPadded (s.r * s.D) (F.record S n s a (certifierCode s a)) = certifierCode s a := by
  obtain ⟨hnd, hall, hdec⟩ := hF
  funext c
  have hrow : ∀ b ∈ List.range (n * F.T),
      (((F.frames (F.record S n s a (certifierCode s a)) b).filterMap fun f =>
        if f.1 = c then some f.2 else none).map fun x => (b, x)) =
        padFrames s c b ++ (certifierCode s a c).filter fun x => x.1 == b := by
    intro b hb
    rw [frames_record F hdec S n s a _ (List.mem_range.1 hb) ((length_bucketFrames _ _ _ _).trans_le (hs.2 _)),
      filterMap_bucketFrames hnd (hall c), map_laneFrames]
  have hflat : ((List.range (n * F.T)).flatMap fun b =>
      ((F.frames (F.record S n s a (certifierCode s a)) b).filterMap fun f =>
        if f.1 = c then some f.2 else none).map fun x => (b, x)) =
      (List.range (n * F.T)).flatMap fun b =>
        padFrames s c b ++ (certifierCode s a c).filter fun x => x.1 == b := by
    rw [List.flatMap_def, List.flatMap_def]
    exact congrArg List.flatten (List.map_congr_left hrow)
  show (((List.range (n * F.T)).flatMap fun b =>
      ((F.frames (F.record S n s a (certifierCode s a)) b).filterMap fun f =>
        if f.1 = c then some f.2 else none).map fun x => (b, x)).drop (s.r * s.D)) = certifierCode s a c
  rw [hflat]
  refine drop_flatMap_padFrames s c _ _ (certifierCode_sorted s a c) (fun x hx => ?_) fun x hx => ?_
  · obtain ⟨j, -, e⟩ := certifierCode_mem_fst s a c hx
    rw [e]
    exact Nat.le_add_right _ _
  · obtain ⟨j, hj, e⟩ := certifierCode_mem_fst s a c hx
    rw [e]
    exact hs.1 c j hj

end RowsFormat

/-- **The rows format reads back under its cells a bucket and its latency**: `RowsReadPadded`. -/
theorem rowsReadPadded : RowsReadPadded := by
  intro κ Cell _ L K F hF S n θ s hθ hfit a
  obtain ⟨hr, hD⟩ := hθ
  show F.readPadded (θ.1 * θ.2) (F.record S n s a (certifierCode s a)) = certifierCode s a
  rw [← hr, ← hD]
  exact F.readPadded_record hF S n s hfit a

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
