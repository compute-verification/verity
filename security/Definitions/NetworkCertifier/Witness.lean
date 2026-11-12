module

public import Definitions.NetworkCertifier.SampledAudit
public import Definitions.NetworkCertifier.Schedule
public import Definitions.NetworkCertifier.RecordRows
public import Definitions.NetworkCertifier.Tap

/-!
# The instances the network certifier's honest and nontrivial lemmas are stated at (trusted definitions; used only by
the satisfiability witnesses)

Each guarantee's honest lemma states its premises at an instance built by hand, and its nontrivial lemma states its
conclusion false at an audit that accepts everything. Those statements read these definitions, and their lemmas are in
`Proofs.NetworkCertifier`:

* `trivLay`: one link of one slot with no gate, at which `SampledAuditAccepted`'s conclusion fails;
* `notLay`: step `t`'s ingress is the input, and its egress slot's presence gate the input's negation, at which the
  leak bounds fail;
* `certOps`: the network certifier's openings, the input and its negation both `true`, at which `PeriodLeakAccepted`'s
  bound fails;
* `obs0`, `obs1`: one step of one link of one slot, with no frame and with an empty frame;
* `honestI P a`: the honest certifier at the Program `P` under the advice `a`, which records and puts on the wire `P`'s
  rows, and `fxHonestI`, it at `fxRate` under `fxAdv`;
* `fxRate`: the constant-rate Program over two execution blocks of two buckets (`T = 2`, `g = 1`), one link of two
  slots and one advice bit per block (`k = 1`), whose two frames, both at the block's first send tick, carry the
  block's advice bit and its negation;
* `acceptAll`, the verifier that accepts everything: `AccSeqsAt acceptAll` is `AccSeqs` with it in place of the
  Program's acceptance;
* `fxLeakI`: a sender whose secret is its every step's frame length, which `fxLeakDec` reads back from the last step of
  the wire or the record;
* `inI`: a certifier recording one frame a bucket (`oneRow`) in the first of two slots, every status complete;
* `sampProg`: two inputs and the negation of the first, with one proof block `{0, 1, 2}` (`sampPart`) sampled Bernoulli
  at rate `1/2` (`sampDesign`);
* `fxRows`: a rows format of the Python network certifier's fixture `fxQueue` (three connections), one window of ten
  buckets, whose schedule is `fxQueueSched`;
* `noFormat`: the record format of a network certifier with no connection over a monitoring period of no execution
  phases, at which `PeriodLeakChecked`'s honest lemma states its period's premises.
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

/-- One step: no frame on the link, every status complete. -/
def obs0 : Obs 1 1 1 := ⟨fun _ _ => none, 0, fun _ => .complete⟩

/-- One step: a frame on the link, every status complete. -/
def obs1 : Obs 1 1 1 := ⟨fun _ _ => some [], 0, fun _ => .complete⟩

/-! ## The honest certifier -/

/-- The operator, which sends nothing. -/
def unitOp {L K S : ℕ} : ℕ → List (Obs L K S) → Unit := fun _ _ => ()

/-- **The honest certifier at `P` under the advice `a`**: at each step it records and puts on the wire `P`'s row on
the ingress so far, jitter class 0 and every status complete, whatever the sender does. -/
def honestI {L K S : ℕ} {Adv : Type} (P : EgressPrediction L S Unit Adv) (a : Adv) :
    Interaction L K S Unit Unit Unit (Obs L K S) where
  certifier :=
    { obs := fun k xs _ => ⟨P.rows k xs a, 0, fun _ => .complete⟩
      wire := fun k xs _ => ⟨P.rows k xs a, 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun _ _ _ => ()

/-! ## The reference schedule -/

/-- Two frames on the one link at the execution block's first send tick: the block's advice bit, then its negation. -/
def fxFrames : List Unit → Fin 2 → (Fin 1 → Bool) → Fin 1 → List LogicalFrame :=
  fun _ _ v _ => [⟨[v 0], 0⟩, ⟨[!v 0], 0⟩]

/-- **The constant-rate Program** over two execution blocks of two buckets (`T = 2`, `g = 1`), one link of two slots,
one advice bit per block. -/
def fxRate : EgressPrediction 1 2 Unit (Fin 2 → Fin 1 → Bool) :=
  constantRate 2 1 2 fxFrames

/-- `fxRate`'s advice: the bit `true` in both execution blocks. -/
def fxAdv : Fin 2 → Fin 1 → Bool := fun _ _ => true

/-- The honest certifier at `fxRate` under `fxAdv`. -/
def fxHonestI : Interaction 1 1 2 Unit Unit Unit (Obs 1 1 2) := honestI (K := 1) fxRate fxAdv

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

/-! ## The Python network certifier's fixture -/

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

end NetTiming.Witness
