module

public import Aesop.BuiltinRules
public import Mathlib.Tactic.Attr.Core
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

/-! An auditor party's assumptions `Faithful`, `Secret`, `Confined` and `WitnessPathPrivate`, the last needed because
witness messages travel unencrypted. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.Parties

/-- Where a declared channel's messages go: to the auditor in the clear, or to the prover only, as witness. -/
inductive Dest
  | clear
  | witness
  deriving DecidableEq

/-- An auditor party's code: its state from its own coins, and one step: from its state and an input (a message
delivered to it, a reading of its clock, the coins it flips), its next state and the messages it sends, each on
one of its declared channels. -/
structure Code (Ω S I Ch M : Type*) where
  init : Ω → S
  step : S → I → S × List (Ch × M)

variable {Ω S I Ch M Wi A : Type*}

/-- The code on the inputs `xs` from the state `s`: the state each step starts from, then the last state, and every
message it sends, in order. -/
def Code.trace (c : Code Ω S I Ch M) : S → List I → List S × List (Ch × M)
  | s, [] => ([s], [])
  | s, x :: xs => let r := c.step s x; let t := c.trace r.1 xs; (s :: t.1, r.2 ++ t.2)

/-- The code's messages on the inputs `xs`, from the state its coins `ω` give it. -/
def Code.messages (c : Code Ω S I Ch M) (ω : Ω) (xs : List I) : List (Ch × M) := (c.trace (c.init ω) xs).2

/-- One run of an auditor party as it happened: its coins; at each step its true input (what was delivered to it, its
clock's true reading, the coins it flipped) and the input it took; the state each step started from, then its last;
the messages it sent on its declared channels; everything it emitted on any wire, each with where it went; the prover's
accesses (reads and writes) to its secret state; and the messages on its path to the prover that someone other than
the party and the prover observed. -/
structure Run (Ω S I Ch M Wi A : Type*) where
  coins : Ω
  truth : List I
  taken : List I
  states : List S
  sent : List (Ch × M)
  emitted : List (Dest × Wi)
  access : List A
  overheard : List (Ch × M)

/-- **`faithful`**: the party runs its code `c` on its true inputs, its clock's and its coins included, and
each step starts from the state the last one left (no rollback): what it took is what was true, and its states and
messages are the code's on those inputs from the state its coins give it. -/
def Faithful (c : Code Ω S I Ch M) (r : Run Ω S I Ch M Wi A) : Prop :=
  r.taken = r.truth ∧ (r.states, r.sent) = c.trace (c.init r.coins) r.taken

/-- **`secret`**: the prover never reads or sets the party's secret state (its signing keys, its LMS state, its coins
not yet released). -/
def Secret (r : Run Ω S I Ch M Wi A) : Prop := r.access = []

/-- **`confined`**: all the party emits is its declared channels' messages, each as `wire` puts it on the wire, to
where its channel goes (`dest`). -/
def Confined (dest : Ch → Dest) (wire : Ch × M → Wi) (r : Run Ω S I Ch M Wi A) : Prop :=
  r.emitted = r.sent.map fun m => (dest m.1, wire m)

/-- **`witness-path-private`**: the path between the party and the prover carries no observer but its endpoints:
nobody else observed a message on it. -/
def WitnessPathPrivate (r : Run Ω S I Ch M Wi A) : Prop := r.overheard = []

/-- What the auditor (and anyone but the prover) sees of the party: what it emitted in the clear, and what was overheard
on its path to the prover. -/
def auditorView (wire : Ch × M → Wi) (r : Run Ω S I Ch M Wi A) : List Wi :=
  (r.emitted.filter (·.1 = .clear)).map (·.2) ++ r.overheard.map wire

/-- What the prover sees of the party: all it emitted, and its accesses to the party's secret state. -/
def proverView (r : Run Ω S I Ch M Wi A) : List Wi × List A := (r.emitted.map (·.2), r.access)

end ResourceAccounting.Assumptions.Parties
