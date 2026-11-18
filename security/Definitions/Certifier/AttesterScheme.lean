module

public import Definitions.Core.Scheme
public import Definitions.Core.Partition
public import Definitions.Core.Challenger
public import Definitions.Core.Assumptions
public import Definitions.Core.Interaction
public import Definitions.Certifier.Tap
public import Mathlib.Data.Nat.Log
public import Mathlib.Data.Set.Card

/-!
# The network attester's scheme, at one execution phase

The network attester's instance of `Verity.Scheme`, in bits, at one execution phase: its blocks are the phase's proof
blocks, a block costs the bits a wrong instance of it can set, and a run's larger side is the bits of the records its
incorrect blocks explain. The period's statement is the pooled theorem's.
-/

@[expose] public section

namespace NetTiming.Attester

open Verity Finset
open scoped ENNReal

set_option linter.unusedVariables false

/-- An execution phase as the attester's scheme reads it: its circuit and partition over `k` steps, each step's ingress
gates (the operator's input) and egress gates (what the attester's rows carry, committed wires), and the registered
gates, among them the state carried in from earlier phases. No ingress of a step feeds the egress of an earlier one. -/
structure Phase where
  C : Circuit
  m : ℕ
  P : Partition C m
  k : ℕ
  ingress : Fin k → Finset (Fin C.N)
  egress : Fin k → Finset (Fin C.N)
  reg : Finset (Fin C.N)
  egress_committed : ∀ t, egress t ⊆ P.committed
  causal : ∀ t t' : Fin k, t'.val < t.val → ∀ g ∈ ingress t, ∀ e ∈ egress t', ¬ C.Feeds g e

namespace Phase

variable (ph : Phase)

/-- The bits a wrong instance of unit `u` can set: the widths of its committed gates. -/
def unitBits (u : Fin ph.m) : ℕ := ph.C.width (ph.P.committed.filter fun g => ph.P.unit g = some u)

open Classical in
/-- The input gates neither registered nor ingress, in bits. -/
noncomputable def freeBits : ℕ :=
  ph.C.width (univ.filter fun g => ph.C.isInput g = true ∧ g ∉ ph.reg ∧ ∀ t, g ∉ ph.ingress t)

/-- The values the attester's rows carry, step by step. -/
abbrev Rows := (t : Fin ph.k) → (g : ph.egress t) → BitVec (ph.C.w g.1)

/-- The rows a trace gives. -/
def rowsOf (X : ph.C.Trace) : ph.Rows := fun _ g => X g.1

open Classical in
/-- The incorrect blocks of a trace. -/
noncomputable def incorrectOf (X : ph.C.Trace) : Finset (Fin ph.m) := ph.P.incorrect X

/-- The operator's input at step `t`, from the channel and the rows. -/
abbrev Operator (Chan : Type) := Fin ph.k → Chan → ph.Rows → (g : Fin ph.C.N) → BitVec (ph.C.w g)

/-- The operator's input at step `t` reads only the rows of the steps before `t`. -/
def Operator.ReadsBefore {ph : Phase} {Chan : Type} (op : ph.Operator Chan) : Prop :=
  ∀ t c (r r' : ph.Rows), (∀ t' : Fin ph.k, t'.val < t.val → r t' = r' t') → op t c r = op t c r'

/-- The records, rows and channel, that committed values with the incorrect blocks `F` explain: registered values
`ρ`, ingress the operator's reaction, rows the values' rows. `ρ` and the operator may read the channel. -/
def Explainable {Chan : Type} (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan) (F : Finset (Fin ph.m)) :
    Set (ph.Rows × Chan) :=
  {rc | ∃ X : ph.C.Trace, ph.incorrectOf X = F ∧ (∀ g ∈ ph.reg, X g = ρ rc.2 g) ∧
    (∀ t, ∀ g ∈ ph.ingress t, X g = op t rc.2 (ph.rowsOf X) g) ∧ rc.1 = ph.rowsOf X}

/-- The scheme's charges, in bits: the free inputs and the channel. -/
noncomputable def charges (Chan : Type) [Fintype Chan] : ℕ := ph.freeBits + Nat.clog 2 (Fintype.card Chan)

/-- **The attester's charge inequality**: where the operator reads only the rows before, the records explainable
with the incorrect blocks `F` number at most two to the cost of `F` plus the charges. -/
def ChargeHolds {Chan : Type} [Fintype Chan] (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan) : Prop :=
  op.ReadsBefore → ∀ F : Finset (Fin ph.m),
    Nat.clog 2 (ph.Explainable ρ op F).ncard ≤ ∑ u ∈ F, ph.unitBits u + ph.charges Chan

end Phase

/-- One execution phase's run as the attester's scheme reads it: the committed values, the attester's record (rows
and channel), the audit's verdict and sample, the challenger's log, the SHA-512 evaluations (input, output) that check
the attester's openings and the proofs' openings of the egress gates, and the SHA-512 calls the run makes, its
prover's included. -/
structure PhaseRun (ph : Phase) (Chan Root Coin : Type) where
  X : ph.C.Trace
  rows : ph.Rows
  chan : Chan
  ok : Bool
  sample : Finset (Fin ph.m)
  log : Challenger.Log Root Coin
  calls : List (Claims.Msg × Claims.Msg)
  hashCalls : ℕ

/-- **The committed values carry the attester's rows, the registered values and the operator's ingress**: the premise
under which the scheme's incorrect blocks are the record's. -/
def RowsCommitted {ph : Phase} {Chan Root Coin : Type} (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan)
    (r : PhaseRun ph Chan Root Coin) : Prop :=
  r.rows = ph.rowsOf r.X ∧ (∀ g ∈ ph.reg, r.X g = ρ r.chan g) ∧ ∀ t, ∀ g ∈ ph.ingress t, r.X g = op t r.chan r.rows g

/-- **The network attester's scheme at one execution phase**, in bits: a run's larger side is the bits of the records
its incorrect blocks explain; its smaller side is none; a block costs the bits a wrong instance of it can set; the log
records every root `fixed` names before every coin; it breaks where the checked SHA-512 evaluations collide, which is
where the attester's committed rows can open two ways. -/
noncomputable def scheme (ph : Phase) {Chan : Type} [Fintype Chan] (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan)
    (hop : op.ReadsBefore) (Root Coin : Type) (Complete : Chan → Prop) (fixed : Root → Prop) (δtap : ℝ≥0∞)
    (hcharge : ph.ChargeHolds ρ op) : Scheme where
  Block := Fin ph.m
  cost := ph.unitBits
  charges := ph.charges Chan
  Run := PhaseRun ph Chan Root Coin
  accepts r := r.ok = true ∧ Complete r.chan
  over r := Nat.clog 2 (ph.Explainable ρ op (ph.incorrectOf r.X)).ncard
  under _ := 0
  logOK r := r.log.Cut fixed fun _ => True
  incorrect r := ph.incorrectOf r.X
  breaks r := (Claims.findColl r.calls).isSome
  δscheme := δtap
  charge r _ _ _ := by simpa using hcharge hop (ph.incorrectOf r.X)

/-- The collision finder a phase's runs on the coins `S` give: the first collision among a run's checked SHA-512
evaluations, at a cost of the run's SHA-512 calls. -/
def callsFinder {ph : Phase} {Chan Root Coin S : Type} [Fintype S] [Nonempty S] (play : S → PhaseRun ph Chan Root Coin) :
    Claims.Finder where
  S := S
  out s := Claims.findColl (play s).calls
  cost s := (play s).hashCalls

/-- **The scheme's break event is within its term**: with the scheme's term `ecrBound` of the runs' finder at the
budget `q`, and every evaluation a run checks an evaluation of `H`, the probability over the coins that a run breaks is
at most the term. -/
def BreaksWithinTerm : Prop :=
  ∀ (ph : Phase) (Chan : Type) [Fintype Chan] (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan) (hop : op.ReadsBefore)
    (Root Coin : Type) (Complete : Chan → Prop) (fixed : Root → Prop) (hcharge : ph.ChargeHolds ρ op)
    (H : Claims.Msg → Claims.Msg) (S : Type) [Fintype S] [Nonempty S] (play : S → PhaseRun ph Chan Root Coin) (q : ℝ),
    (∀ s, ∀ p ∈ (play s).calls, H p.1 = p.2) →
    let Sc := scheme ph ρ op hop Root Coin Complete fixed (Claims.ecrBound H (callsFinder play) q) hcharge
    ENNReal.ofReal (Claims.Pr fun s => Sc.breaks (play s)) ≤ Sc.δscheme

/-- **The secrets one execution phase's record singles out**: over the secrets whose phase's committed values the
design misses with probability above `δ`, at most the number of such block sets times two to the missable bits plus
the charges. -/
def PhaseRecordSinglesOut : Prop :=
  ∀ (ph : Phase) (Chan : Type) [Fintype Chan] (ρ : Chan → ph.C.Trace) (op : ph.Operator Chan) (hop : op.ReadsBefore)
    (Root Coin : Type) (Complete : Chan → Prop) (fixed : Root → Prop) (δtap : ℝ≥0∞) (hcharge : ph.ChargeHolds ρ op)
    (L : SamplingDesign (Fin ph.m)) (δ : ℝ≥0∞) {Z : Type} (Zs : Set Z) (runOf : Z → PhaseRun ph Chan Root Coin)
    (dec : ph.Rows × Chan → Z) (hdz : ∀ z ∈ Zs, dec ((runOf z).rows, (runOf z).chan) = z)
    (hrows : ∀ z ∈ Zs, RowsCommitted ρ op (runOf z)),
    {z | z ∈ Zs ∧ δ < L.missProb (ph.incorrectOf (runOf z).X)}.encard ≤
      (((univ.filter fun T : Finset (Fin ph.m) => δ < L.missProb T).card *
        2 ^ ((scheme ph ρ op hop Root Coin Complete fixed δtap hcharge).missable L δ + ph.charges Chan) : ℕ) : ℕ∞)

/-- A fail pattern: each connection's fail cell, below its cap, or none. -/
abbrev FailPattern {Conn : Type} (cap : Conn → ℕ) := (c : Conn) → Option (Fin (cap c))

open Classical in
/-- The fail patterns with at most `j` failed connections. -/
noncomputable def failAtMost {Conn : Type} [Fintype Conn] (cap : Conn → ℕ) (j : ℕ) : Finset (FailPattern cap) :=
  univ.filter fun φ => (univ.filter fun c => (φ c).isSome).card ≤ j

/-- **The client's views are the schedule's, the data's and a fail pattern's**: with the attester trusted
(`TrustedDevice`) and every schedule within the caps, each secret's view is `clientView` of its schedule, its
connections' data and a fail pattern. -/
def ClientViewReads : Prop :=
  ∀ {Conn Cell Srv Dev Ob Z : Type} [Fintype Conn] (f : RecordFormat Conn Cell Ob) (arr : Srv → Conn → Arrivals Cell)
    (tap : Dev) (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × Ob)
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Finset (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss),
    ∀ z ∈ Zs, ∃ φ : FailPattern cap,
      (run tap (schedOf z, srvOf z)).1 =
        clientView (schedOf z) (fun c => (arr (srvOf z) c).stream) fun c => (φ c).map Fin.val

/-- The channel of one execution phase at `j` failures: a schedule of `Ss`, a fail pattern with at most `j` failed
connections, and the jitter at each step. -/
abbrev ChanAt {Conn : Type} [Fintype Conn] (Ss : Finset (CertifierSchedule Conn)) (cap : Conn → ℕ) (j : ℕ)
    (ph : Phase) (Dj : ℕ) :=
  Ss × failAtMost cap j × (Fin ph.k → Fin Dj)

end NetTiming.Attester
