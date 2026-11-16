module

public import Definitions.Certifier.SampledAudit
public import Definitions.Certifier.Schedule
public import Definitions.Certifier.RecordRows
public import Definitions.Certifier.Tap

/-!
# The instances the network attester's honest and nontrivial lemmas are stated at (trusted definitions; used only by
the satisfiability witnesses)

Each guarantee's honest lemma states its premises at an instance built by hand, and its nontrivial lemma states its
conclusion false at an audit that accepts everything. Those statements read these definitions, and their lemmas are in
`Proofs.Certifier`:

* `trivLay`: one link of one slot with no gate, at which `SampledAuditAccepted`'s conclusion fails;
* `notLay`: step `t`'s ingress is the input, and its egress slot's presence gate the input's negation, at which the
  leak bounds fail;
* `certOps`: the network attester's openings, the input and its negation both `true`, at which `MonitoringPeriodLeakAccepted`'s
  bound fails;
* `obs0`, `obs1`: one step of one link of one slot, with no frame and with an empty frame;
* `unitOp`: the operator, which sends nothing;
* `acceptAll`, the verifier that accepts everything: `AccSeqsAt acceptAll` is `AccSeqs` with it in place of the
  Program's acceptance;
* `fxLeakI`: a sender whose secret is its every step's frame length, which `fxLeakDec` reads back from the last step of
  the wire or the record;
* `inI`: a certifier recording one frame a bucket (`oneRow`) in the first of two slots, every status complete;
* `sampProg`: two inputs and the negation of the first, with one proof block `{0, 1, 2}` (`sampPart`) sampled Bernoulli
  at rate `1/2` (`sampDesign`);
* `fxRows`: a rows format of the Python network attester's fixture `fxQueue` (three connections), one window of ten
  buckets, whose schedule is `fxQueueSched`;
* `noFormat`: the record format of a network attester with no connection over a monitoring period of no execution
  phases, at which `MonitoringPeriodLeakCharged`'s honest lemma states its period's premises;
* `frameFormat`: the record format of a network attester with no connection over a monitoring period of one execution
  phase, whose record of it is one step with a frame (`obs1`), at which `MonitoringPeriodLeakChecked`'s honest lemma
  states its period's premises and its execution phase's good event.

The lemma that drops its read-back is stated at a network attester of one connection of one cell with no data
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

/-- The network attester's openings: the input and its negation, both `true`. -/
def certOps : List (Claims.Sampling.Opening Bool) := [⟨0, true, []⟩, ⟨1, true, []⟩]

/-- The record format of a network attester with no connection, over a monitoring period of no windows: it holds every
schedule, its record is the empty one and no cell reads back from it, as the code sends none. -/
def noFormat : RecordFormat Empty Unit (Fin 0 → List (Obs 1 1 1)) :=
  ⟨fun _ _ _ => Fin.elim0, fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩

/-- One step: no frame on the link, every status complete. -/
def obs0 : Obs 1 1 1 := ⟨fun _ _ => none, 0, fun _ => .complete⟩

/-- One step: a frame on the link, every status complete. -/
def obs1 : Obs 1 1 1 := ⟨fun _ _ => some [], 0, fun _ => .complete⟩

/-- The record format of a network certifier with no connection, over a monitoring period of one window: it holds every
schedule, its record of the window is one step with a frame on the link (`obs1`), and no cell reads back from it, as
the code sends none. -/
def frameFormat : RecordFormat Empty Unit (Fin 1 → List (Obs 1 1 1)) :=
  ⟨fun _ _ _ _ => [obs1], fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩

/-! ## The operator -/

/-- The operator, which sends nothing. -/
def unitOp {L K S : ℕ} : ℕ → List (Obs L K S) → Unit := fun _ _ => ()

/-! ## The verifier that accepts everything -/

/-- The observation sequences of length `n` with jitter below `D` that the acceptance `acc` passes: `AccSeqs` with
`acc` in place of the Program's acceptance (`accSeqs_eq_at`). -/
def AccSeqsAt {L K S : ℕ} (acc : List (Obs L K S) → Prop) (n D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ acc o}

/-- The verifier that accepts everything. -/
def acceptAll {L K S : ℕ} (_ : List (Obs L K S)) : Prop := True

/-- **A sender whose secret shows**: its secret `z` is its every action, and the certifier records and puts on the wire
one frame of `z` bits in the first slot, jitter class 0, its one status complete. -/
def fxLeakI : Interaction 1 1 2 ℕ Unit ℕ (Obs 1 1 2) where
  certifier :=
    { obs := fun _ _ as => ⟨fun _ σ => if σ = 0 then some (List.replicate (as.getLast?.getD 0) true) else none, 0,
        fun _ => .complete⟩
      wire := fun _ _ as => ⟨fun _ σ => if σ = 0 then some (List.replicate (as.getLast?.getD 0) true) else none, 0,
        fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun z _ _ => z

/-- `fxLeakI`'s decoder: the length of the last step's first frame. -/
def fxLeakDec (w : List (Obs 1 1 2)) : ℕ := ((w.getLast?.bind fun o => o.rows 0 0).map List.length).getD 0

/-! ## Ingress -/

/-- One frame `[true]` in the first of two slots. -/
def oneRow : Row 2 (List Bool) := fun σ => if σ = 0 then some [true] else none

/-- **A certifier at one frame a bucket**: at every step it records and puts on the wire `oneRow` on its one link,
jitter class 0, its one status complete. -/
def inI : Interaction 1 1 2 Unit Unit Unit (Obs 1 1 2) where
  certifier :=
    { obs := fun _ _ _ => ⟨fun _ => oneRow, 0, fun _ => .complete⟩
      wire := fun _ _ _ => ⟨fun _ => oneRow, 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun _ _ _ => ()

/-! ## A sampled audit with a possible incorrect proof block -/

/-- Two inputs and the negation of the first, the negation returned. -/
def sampProg : Claims.Sampling.Prog := ⟨[.input, .not 0, .input], [1]⟩

/-- One class of one stratum of one proof block, `{0, 1, 2}`. -/
def sampPart : Claims.Sampling.Partition := ⟨[[[[0, 1, 2]]]]⟩

/-- The one class sampled Bernoulli at rate `1/2`. -/
def sampDesign : Claims.Sampling.Design := ⟨[.bernoulli 1 2]⟩

/-! ## The Python network attester's fixture -/

/-- A frame of the fixture's three connections, as a number below `9`: its connection's ordinal times three, plus its
kind. -/
def fxCode : Fin 3 × CellView Unit → ℕ
  | (c, .data _) => 3 * c.val
  | (c, .pad) => 3 * c.val + 1
  | (c, .fail) => 3 * c.val + 2

/-- The frame a number below `9` is. -/
def fxFrame (n : ℕ) : Option (Fin 3 × CellView Unit) :=
  if h : n < 9 then
    some (⟨n / 3, by omega⟩, if n % 3 = 0 then .data () else if n % 3 = 1 then .pad else .fail)
  else none

/-- **A rows format of the fixture `fxQueue`**: its three connections in order, a frame encoded as that many `true`
bits, one egress link and one status, ten buckets a window. -/
def fxRows : RowsFormat (Fin 3) Unit 1 1 where
  conns := List.finRange 3
  enc x := List.replicate (fxCode x) true
  dec l := fxFrame l.length
  link := 0
  T := 10
  jitter _ _ _ := 0
  status _ _ _ _ := .complete

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
