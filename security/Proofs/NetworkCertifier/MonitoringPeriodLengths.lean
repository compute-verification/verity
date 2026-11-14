module

public import Definitions.NetworkCertifier.MonitoringPeriodConditions
public import Proofs.NetworkCertifier.Warden
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# A lengths format reads back under the link's parameters

The network certifier's cells on a connection are a function of the link's `r` and `D`, the connection's admission
bucket, and three things of its lane: its length, the data it released and its fail cell (`lanesView_laneRecord`). So
a record format that writes those three for every connection (`LengthsRows`, a layout with its read) reads back under
a registered parameter that holds `r`, `D` and the rule that gives the admissions from the lanes' lengths
(`LengthsRows.readsBy`): the instance of `RecordFormat.ReadsBy` that `MonitoringPeriodLeakChecked` and the failure forms take
at the lengths rows.

What a layout has to give is `LengthsRows.read_write`: the three fields of every connection come back from the rows
it writes. No device's layout is modelled here (which execution phase's transfer blocks hold a connection's slot,
and the slot's bytes): `laneRows`, which writes the lane records as they are, shows the lengths layout's conditions
hold together and is no device's.
-/

@[expose] public section

namespace NetTiming

variable {κ Cell Ob : Type}

/-- The data a cell holds, where it is a data cell. -/
def cellData : CellView Cell → Option Cell
  | .data x => some x
  | _ => none

/-- **The data a lane released**, in order: its data cells' contents. -/
def laneData (l : List (ℕ × CellView Cell)) : List Cell :=
  l.filterMap fun x => cellData x.2

/-- Where a lane's data ends: at the end of the stream, or at its fail cell where that comes first. -/
def dataEnd (len : ℕ) : Option ℕ → ℕ
  | none => len
  | some f => min len f

theorem dataEnd_le (len : ℕ) (o : Option ℕ) : dataEnd len o ≤ len := by
  cases o with
  | none => exact le_rfl
  | some f => exact Nat.min_le_left _ _

/-- A lane's cell is a data cell exactly before the data's end, and then it is the stream's. -/
theorem cellData_cellView (stream : List Cell) (o : Option ℕ) (j : ℕ) :
    cellData (cellView stream o j) = if j < dataEnd stream.length o then stream[j]? else none := by
  cases o with
  | none =>
    by_cases h : j < stream.length
    · simp [cellView, streamCell, cellData, dataEnd, h]
    · simp [cellView, streamCell, cellData, dataEnd, h]
  | some f =>
    by_cases h1 : j < f
    · by_cases h : j < stream.length
      · simp [cellView, streamCell, cellData, dataEnd, h, h1]
      · simp [cellView, streamCell, cellData, dataEnd, h, h1]
    · have h2 : ¬ j < min stream.length f := fun h => h1 (lt_of_lt_of_le h (Nat.min_le_right _ _))
      by_cases h3 : j = f
      · simp [cellView, cellData, dataEnd, h3]
      · simp [cellView, cellData, dataEnd, h1, h2, h3]

/-- The first `N` cells' data are the stream up to the data's end. -/
theorem filterMap_range_take (stream : List Cell) {e : ℕ} (he : e ≤ stream.length) (N : ℕ) :
    ((List.range N).filterMap fun j => if j < e then stream[j]? else none) = stream.take (min N e) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [List.range_succ, List.filterMap_append, ih]
    by_cases h : N < e
    · have hN : N < stream.length := lt_of_lt_of_le h he
      rw [Nat.min_eq_left (Nat.le_of_lt h), Nat.min_eq_left (Nat.succ_le_of_lt h), List.take_add_one]
      simp [h, List.getElem?_eq_getElem hN]
    · rw [Nat.min_eq_right (Nat.le_of_not_lt h), Nat.min_eq_right (Nat.le_succ_of_le (Nat.le_of_not_lt h))]
      simp [h]

/-- A cell of the stream cut at `M` is the stream's before `M`, and padding from it. -/
theorem streamCell_take (stream : List Cell) (M j : ℕ) :
    streamCell (stream.take M) j = if j < M then streamCell stream j else .pad := by
  unfold streamCell
  by_cases h1 : j < M
  · by_cases h2 : j < stream.length
    · simp [h1, h2]
    · simp [h1, h2]
  · simp [h1]

/-- A lane's cell is the same from the data the lane released as from the whole stream. -/
theorem cellView_take (stream : List Cell) (o : Option ℕ) {N j : ℕ} (hj : j < N) :
    cellView (stream.take (min N (dataEnd stream.length o))) o j = cellView stream o j := by
  cases o with
  | none =>
    show streamCell (stream.take (min N stream.length)) j = streamCell stream j
    rw [streamCell_take]
    by_cases h : j < stream.length
    · simp [hj, h]
    · simp [streamCell, h]
  | some f =>
    show cellView (stream.take (min N (min stream.length f))) (some f) j = cellView stream (some f) j
    by_cases h1 : j < f
    · simp only [cellView, h1, ↓reduceIte]
      rw [streamCell_take]
      by_cases h : j < stream.length
      · simp [hj, h, h1]
      · simp [streamCell, h]
    · simp only [cellView, h1, ↓reduceIte]

/-- **A lane is its length, the data it released and its fail cell**: the client view of a schedule is the same from
each lane's released data as from the server's whole stream. -/
theorem clientView_laneData (s : CertifierSchedule κ) (data : κ → List Cell) (fail : κ → Option ℕ) :
    clientView s (fun c => laneData (clientView s data fail c)) fail = clientView s data fail := by
  funext c
  have hd : laneData (clientView s data fail c) =
      (data c).take (min (s.cells c) (dataEnd (data c).length (fail c))) := by
    simp only [laneData, clientView, List.filterMap_map]
    rw [← filterMap_range_take (data c) (dataEnd_le _ _)]
    exact List.filterMap_congr fun j _ => cellData_cellView (data c) (fail c) j
  show (List.range (s.cells c)).map
      (fun j => (s.bucket c j, cellView (laneData (clientView s data fail c)) (fail c) j)) =
    (List.range (s.cells c)).map fun j => (s.bucket c j, cellView (data c) (fail c) j)
  rw [hd]
  exact List.map_congr_left fun j hj => by rw [cellView_take _ _ (List.mem_range.1 hj)]

/-- **What the lengths and fails rows hold of one connection**: its lane's length, the data it released and its fail
cell or none. -/
structure LaneRecord (Cell : Type) where
  cells : ℕ
  data : List Cell
  fail : Option ℕ

/-- Connection `c`'s lane record, from the cells the certifier released and the fail cell it saw. -/
def laneRecord (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (v : κ → List (ℕ × CellView Cell)) (c : κ) :
    LaneRecord Cell :=
  ⟨(v c).length, laneData (v c), failCell s a c⟩

/-- **A link's registered parameters**: `r`, `D`, and the rule that gives each connection's admission bucket from the
lanes' lengths (a constant rule where the admissions are fixed). -/
structure LinkParams (κ : Type) where
  r : ℕ
  D : ℕ
  adm : (κ → ℕ) → κ → ℕ

/-- The schedule the link's parameters give at the lanes' lengths. -/
def LinkParams.sched (θ : LinkParams κ) (cells : κ → ℕ) : CertifierSchedule κ := ⟨θ.r, θ.D, θ.adm cells, cells⟩

/-- **The view read from the lane records** under the link's parameters. -/
def lanesView (θ : LinkParams κ) (x : κ → LaneRecord Cell) : κ → List (ℕ × CellView Cell) :=
  clientView (θ.sched fun c => (x c).cells) (fun c => (x c).data) fun c => (x c).fail

theorem certifierCode_length (s : CertifierSchedule κ) (a : κ → Arrivals Cell) (c : κ) :
    (certifierCode s a c).length = s.cells c := by
  rw [certifierCode_eq]
  simp [clientView]

/-- **The certifier's cells are the view of their lane records**, on a schedule the link's parameters give. -/
theorem lanesView_laneRecord (θ : LinkParams κ) (s : CertifierSchedule κ) (hs : s = θ.sched s.cells)
    (a : κ → Arrivals Cell) : lanesView θ (laneRecord s a (certifierCode s a)) = certifierCode s a := by
  have hc : (fun c => (laneRecord s a (certifierCode s a) c).cells) = s.cells :=
    funext fun c => certifierCode_length s a c
  have hv : lanesView θ (laneRecord s a (certifierCode s a)) =
      clientView s (fun c => laneData (certifierCode s a c)) (failCell s a) := by
    unfold lanesView
    rw [hc, ← hs]
    rfl
  rw [hv, certifierCode_eq, clientView_laneData]

/-- **A lengths layout**: how every connection's lane record is written in the record (`write`) and read from it
(`read`), on the schedules it holds (`fits`). -/
structure LengthsRows (κ Cell Ob : Type) where
  write : CertifierSchedule κ → (κ → LaneRecord Cell) → Ob
  read : Ob → κ → LaneRecord Cell
  fits : CertifierSchedule κ → Prop
  read_write : ∀ s, fits s → ∀ x : κ → LaneRecord Cell, (∀ c, (x c).cells = s.cells c) → read (write s x) = x

namespace LengthsRows

variable (F : LengthsRows κ Cell Ob)

/-- **The lengths format**: the record is the layout's rows of the lane records of the cells the certifier released,
and the read is their view under the schedule's `r`, `D` and admissions. -/
def format : RecordFormat κ Cell Ob where
  record s a v := F.write s (laneRecord s a v)
  read s ob := lanesView ⟨s.r, s.D, fun _ => s.adm⟩ (F.read ob)
  fits := F.fits
  read_record s hs a := by
    rw [F.read_write s hs _ fun c => certifierCode_length s a c]
    exact lanesView_laneRecord ⟨s.r, s.D, fun _ => s.adm⟩ s rfl a

/-- **The lengths format reads back under the link's parameters**: on every schedule they give at its own lanes'
lengths, the view is read from the record with nothing else of the schedule. -/
theorem readsBy : F.format.ReadsBy (fun θ ob => lanesView θ (F.read ob)) fun θ s => s = θ.sched s.cells := by
  intro θ s hs hfit a
  show lanesView θ (F.read (F.write s (laneRecord s a (certifierCode s a)))) = certifierCode s a
  rw [F.read_write s hfit _ fun c => certifierCode_length s a c]
  exact lanesView_laneRecord θ s hs a

end LengthsRows

/-- The layout that writes the lane records as they are: every lengths layout's condition holds at it. -/
def laneRows (κ Cell : Type) : LengthsRows κ Cell (κ → LaneRecord Cell) where
  write _ x := x
  read x := x
  fits _ := True
  read_write _ _ _ _ := rfl

end NetTiming
