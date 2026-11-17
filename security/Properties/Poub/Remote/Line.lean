module

public import Definitions.Poub.Assumptions.RemoteHeld
public import Definitions.Poub.Assumptions.RemoteOutside
public import Definitions.Poub.Assumptions.RemoteDraw
public import Properties.Poub.Remote.Line.Record
public import Properties.Poub.Remote.Line.Premises
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
public import Definitions.Pouw.Basic.Prob

/-! The memory line of record over what the Lean verifier accepts (`LineHeld`), and the properties of the line's code
it reads. -/

@[expose] public section

namespace Properties.Poub.Line

open ResourceAccounting.Remote.Held ResourceAccounting.Remote.Held.Assumptions ResourceAccounting.Remote.Sample
  ResourceAccounting.Remote.Sample.Assumptions _root_.Poub.Line

/-- **LineChecked**: a line the code computes had a block of one or more bytes, a block's encode of one or more
squarings, nonnegative rates and bursts, and read a nonnegative wrong receipts' bound. -/
def LineChecked : Prop :=
  ∀ (gs : Array Group) (per blocks blockBytes chipBytes nodeBytes : ℕ) (windows : Array (ℕ × ℕ)) (δ : ℚ)
    (rates : Rates) (l : Line),
    line gs per blocks blockBytes chipBytes nodeBytes windows δ rates = .ok l →
    1 ≤ blockBytes ∧ 1 ≤ rates.encodeSquarings ∧ 0 ≤ rates.squaringsPerS ∧ 0 ≤ rates.diskBytesPerS ∧
      0 ≤ rates.linkBytesPerS ∧ 0 ≤ rates.rebuildBurstBytes ∧ 0 ≤ rates.diskBurstBytes ∧
      0 ≤ rates.linkBurstBytes ∧ 0 ≤ l.wrong

/-- **The verifier refuses a line that isn't registered.** A line `line_record` computes off a record given bare,
sampled `g` windows a coin (`lineOfRegistered`), is of a record of the registration given with it: the record names that
registration's digest, its registered blocks, which the line credits, are the registration's, and the line is the
record's (`lineOfRecordOn`). -/
def LineRegistered : Prop :=
  ∀ (onRecord : Bool) (g : ℕ) (reg : Poub.Registration) (digest : ByteArray)
    (sm : Poub.Session.Summary) (samples : ℕ) (seen : Array Seen) (per blockBytes chipBytes nodeBytes : ℕ)
    (δ : ℚ) (rates : Rates) (l : Line),
    lineOfRegistered onRecord g reg digest sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l →
    digest = reg.digest ∧ sm.blocks = reg.blocks ∧
      lineOfRecordOn onRecord g sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l

/-- **The arithmetic link.** Each stretch `x` of form `i` (time-held, link-charged) of a line the code computes: its
closing group is a group of the record, its span is `[spanStart, spanStop)` on the record's clock, nonempty, ending at
its paced session's `stop` and starting at its `ended 0` (the first stretch) or its `released 0` (the rest), its form's
charge is `charges`', its credit `span · (N − lost)` is at most the left-hand side of
`ResourceAccounting.Guarantees.Remote.HeldOverTimeLinks` at its paced session (`stretchPace`), that span, the chips' blocks
`chip / b` rounded up, the form's charge and the wrong receipts' bound the line read (`Line.wrong`), and its weights bytes
a tick are
`max(0, w − lost) · b`. -/
def StretchBound : Prop :=
  ∀ (gs : Array Group) (per blocks blockBytes chipBytes nodeBytes : ℕ) (windows : Array (ℕ × ℕ)) (δ : ℚ)
    (rates : Rates) (l : Line),
    line gs per blocks blockBytes chipBytes nodeBytes windows δ rates = .ok l →
    ∀ i < 2, ∀ f, l.forms[i]? = some f → ∀ x ∈ f.stretches,
      (x.stretch + 1) * per < gs.size ∧
      spanStart gs per x.stretch < spanStop gs per x.stretch ∧
      x.spanNs = spanStop gs per x.stretch - spanStart gs per x.stretch ∧
      spanStop gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).stop () ∧
      (spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).ended () 0 ∨
        spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).released () 0) ∧
      f.charge = (if i = 0 then (charges rates blockBytes).1 else (charges rates blockBytes).2) ∧
      (x.spanNs : ℝ) * ((blocks : ℝ) - x.time.lost) ≤
        (x.spanNs : ℝ) * ((blocks : ℝ) - ((chipBytes + blockBytes - 1) / blockBytes : ℕ))
          - 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest () *
              ((f.charge.rate : ℝ) *
                  ((x.spanNs : ℝ) + 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest ()) +
                f.charge.burst)
          - (stretchPace (fun _ : Unit => gs) per x.stretch).fill () blocks l.wrong ∧
      x.time.weights = max 0 (x.weights - x.time.lost) * (blockBytes : ℚ)

/-- What stretch `x` of a line over the record's groups `gs` says of the trace at `ω` over its span: at least
`span · (N − lost)` registered blocks in node memory summed over its ticks, and at least `span` times its weights
bytes a tick of any set of registered blocks as large as the weights its passes read on average. -/
def Credits {Ω : Type} (T : Trace Ω) (ω : Ω) (gs : Array Group) (per b : ℕ) (x : Stretch) : Prop :=
  x.spanNs = spanStop gs per x.stretch - spanStart gs per x.stretch ∧
  (x.spanNs : ℝ) * ((T.blocks : ℝ) - x.time.lost) ≤
    T.heldOver ω (spanStart gs per x.stretch) (spanStop gs per x.stretch) ∧
  ∀ R : Finset (Fin T.blocks), (x.weights : ℝ) ≤ R.card →
    (x.spanNs : ℝ) * x.time.weights ≤
      b * ∑ t ∈ Finset.Ico (spanStart gs per x.stretch) (spanStop gs per x.stretch), (T.mem ω t ∩ R).card

/-- Every stretch of form `form` of the record's line, if the code computes one, credits what the trace held at `ω`. -/
def HeldAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) (form : ℕ) : Prop :=
  ∀ l, lineOf node r = .ok l → ∀ f, l.forms[form]? = some f → ∀ x ∈ f.stretches,
    Credits T ω (r.groups node) node.per node.blockBytes x

/-- What the record's line credits read at `ω`, in bits: if the code computes a line, the passes of the record's groups
(`firstPass` over all of them, `P`) read every one of the node's `N` registered blocks but the W wrong receipts the
line read (`Line.wrong`), `b` bytes a block: `8·b·(P·N − W)` is at most the bits of the blocks they read on the node
(`Trace.reads`). -/
def ReadAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) : Prop :=
  ∀ l, lineOf node r = .ok l →
    (8 * node.blockBytes : ℝ) * (((firstPass (r.groups node) (r.groups node).size * node.blocks : ℕ) : ℝ) - l.wrong) ≤
      (8 * node.blockBytes : ℝ) * T.reads ω (firstPass (r.groups node) (r.groups node).size)

/-- **The memory line of record, in bits read.** On a trace over a session's coins, one for each of its samples
(`Node.samples`, a window each at g = 1), the prover's records read through the coin commitment (`recordCommitted`,
each record's opening sample at its `samples` s) and the record of each outcome the one played holding the deployed
commitment (`played`), whose trace of record (`ofRecord`: over the registration's blocks, its verdict the Lean
verifier's on the record) meets the line's premises at the record's groups, the network attesters' coverage of the
node's uplinks (`Covered`) and count of what enters (`IngressCounted` at `δnet`), the probability that the Lean
verifier accepts the record's session, the certifiers sign nothing but the verifier's messages entering the
node over the credited passes, and the record's line credits more bits read than the passes read is at most
`δs + δnet + (δ + Σ_c s / 2^256 + Σ_c adv_c + Σ_c hide_c + bind)`, `δ` the line's, `adv_c` the advantage on `prf c` of
sample `c`'s distinguisher at it with leaf `c` of the coin commitment committing zeros, `hide_c` sample `c`'s hiding
term and `bind` the receipts' binding term at `R`, over the session's samples. -/
def LineHeld : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.samples → Pous.Bits κ → Blocks receiptMax)
    (rest : SampleCoins node.samples κ X → Fin node.samples → ℕ → ℕ) (recs : Cm → SampleCoins node.samples κ X → Record)
    (correct : Cm → Fin node.samples → SampleCoins node.samples κ X → Finset ℕ)
    (com : SampleCoins node.samples κ X → Cm)
    (T : Trace (SampleCoins node.samples κ X)) (R : Receipts node.samples κ X) (δs δnet : ℝ),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ (O : Outside (ofRecord T node r)) (I : Ingress (SampleCoins node.samples κ X)),
    Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → Covered (ofRecord T node r) O I →
    IngressCounted (ofRecord T node r) I δnet → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Verified node (r ω) ∧
        ¬ (Quiet (ofRecord T node r) I.counted ω → ReadAt (ofRecord T node r) ω node (r ω))) ≤
      δs + δnet + ((node.delta : ℝ) + ((∑ _t : Fin node.samples, node.params.samples : ℕ) : ℝ) / 2 ^ 256 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end Properties.Poub.Line
