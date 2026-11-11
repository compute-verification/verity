module

public import Definitions.NetworkCertifier.SampledAudit
public import Definitions.NetworkCertifier.Tap

/-!
# The layouts, openings and record format the network certifier's witnesses are stated at (trusted definitions; used
only by the satisfiability witnesses)

The nontrivial lemmas of `Guarantees/SampledAudit`, `Guarantees/SampledCharge`, `Guarantees/Checked` and `Period` state
their conclusions false at an audit that accepts everything, over these:

* `trivLay`: one link of one slot with no gate, at which `SampledAuditAccepted`'s conclusion fails;
* `notLay`: step `t`'s ingress is the input, and its egress slot's presence gate the input's negation, at which the
  leak bounds fail;
* `certOps`: the network certifier's openings, the input and its negation both `true`, at which `PeriodLeakAccepted`'s
  bound fails.

`PeriodLeakChecked`'s honest lemma states its period's premises over `noFormat`, the record format of a network
certifier with no connection over a monitoring period of no windows.

The lemma that drops its read-back is stated at a network certifier of one connection of one cell with no data
(`Cell := Empty`), over a monitoring period of one window of one step:

* `posObs`: the step's observation, a frame on the link when a bit is unset; `posDec` reads the bit back;
* `bitArr`: the server's arrivals, its end late for bucket 0 or not; `oneSched` and `admSched`: the schedules of one
  cell, admitted in bucket 0, or in bucket 1 at a bit;
* `bitFmt`: the record format that writes whether the connection's cell is the fail cell (`failedAt0`) as the window's
  frame, and reads it back under the whole schedule;
* `lateAt0`: whether a connection's first cell is past bucket 0, which the record doesn't hold.
-/

@[expose] public section

namespace NetTiming.Witness

/-- A layout of one link of one slot, with no gate. -/
def trivLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (0, []), fun _ => []⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- The network certifier's openings: the input and its negation, both `true`. -/
def certOps : List (Claims.Sampling.Opening Bool) := [⟨0, true, []⟩, ⟨1, true, []⟩]

/-- The record format of a network certifier with no connection, over a monitoring period of no windows: it holds every
schedule, its record is the empty one and no cell reads back from it, as the code sends none. -/
def noFormat : RecordFormat Empty Unit (Fin 0 → List (Obs 1 1 1)) :=
  ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩

/-! ## One connection of one cell with no data -/

/-- One step: a frame on the link when the bit is unset, every status complete. -/
def posObs (b : Bool) : Obs 1 1 1 := ⟨fun _ _ => if b then none else some [], 0, fun _ => .complete⟩

/-- The view is the records, and the secret is read off the record's frame. -/
def posDec (v : Fin 1 → List (Obs 1 1 1)) : Bool :=
  match v 0 with
  | [y] => (y.rows 0 0).isNone
  | _ => false

/-- Whether a connection's cells begin with the fail cell. -/
def failedAt0 {Cell : Type} : List (ℕ × CellView Cell) → Bool
  | (_, .fail) :: _ => true
  | _ => false

/-- One connection, admitted in bucket 0 with no latency, of one cell. -/
def oneSched : CertifierSchedule Unit := ⟨1, 0, fun _ => 0, fun _ => 1⟩

/-- The server's arrivals at the bit `b`: no data, its end late for bucket 0 exactly when `b` is set. -/
def bitArr (b : Bool) : Unit → Arrivals Empty := fun _ => ⟨[], fun _ => if b then 1 else 0⟩

/-- **The record format of one bit**: on a schedule of one cell a connection, the window's frame is there exactly
when the connection's cell is not the fail cell, and reads back as that cell in its bucket. -/
def bitFmt : RecordFormat Unit Empty (Fin 1 → List (Obs 1 1 1)) where
  record _ _ v := fun _ => [posObs (failedAt0 (v ()))]
  read s ob := fun _ => [(s.bucket () 0, if posDec ob then CellView.fail else CellView.pad)]
  fits s := s.cells () = 1
  read_record := by
    intro s hfits a
    have hs : s.cells () = 1 := hfits
    have hst : (a ()).stream = [] := by
      cases h : (a ()).stream with
      | nil => rfl
      | cons x _ => exact x.elim
    have hc : certifierCode s a () =
        [(s.bucket () 0, if s.bucket () 0 < (a ()).ready 0 then CellView.fail else CellView.pad)] := by
      show certifierLane (a ()) (s.bucket ()) 0 (s.cells ()) = _
      rw [hs]
      by_cases h : s.bucket () 0 < (a ()).ready 0
      · simp [certifierLane, hst, h]
      · simp [certifierLane, streamCell, hst, h]
    have hd : ∀ z : Bool, posDec (fun _ => [posObs z]) = z := fun z => by cases z <;> rfl
    funext c
    show [(s.bucket () 0, if posDec (fun _ => [posObs (failedAt0 (certifierCode s a ()))]) then CellView.fail
      else CellView.pad)] = certifierCode s a ()
    rw [hd, hc]
    by_cases h : s.bucket () 0 < (a ()).ready 0
    · simp only [h, ↓reduceIte]
      rfl
    · simp only [h, ↓reduceIte]
      rfl

/-- The schedule whose one connection, of one cell, is admitted in bucket 1 at `z` and in bucket 0 otherwise. -/
def admSched (z : Bool) : CertifierSchedule Unit := ⟨1, 0, fun _ => if z then 1 else 0, fun _ => 1⟩

/-- Whether a connection's first cell is past bucket 0. -/
def lateAt0 {Cell : Type} : List (ℕ × CellView Cell) → Bool
  | (b, _) :: _ => decide (0 < b)
  | _ => false

end NetTiming.Witness
