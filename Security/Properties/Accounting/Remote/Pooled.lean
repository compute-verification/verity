module

public import Definitions.Accounting.Remote.Draw
public import RemoteAudit.Line
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
# The verifier's pooled bound covers the draw's (`RemoteAudit.Pooled`)

`RemoteAudit.Pooled.pooled` (`verity/core/protocols/remote_bandwidth/lean/RemoteAudit/Pooled.lean`, which `pooled.py`
is held to by the difftest, and whose W the memory line reads: `RemoteAudit.Line.line`) computes a session's pooled
bound from its windows' receipts `U_t` and draws `s_t` at `δ`, in exact rationals: the budget from above by a certified
rational root, and W by the fractional knapsack's greedy. The draw's model reads the bound as a supremum
(`Accounting.Remote.Draw.pooled`, which `Accounting.Guarantees.Remote.PooledSample` reads).

* `PooledCovers`: whenever the code computes a bound, its W is at least `Draw.pooled` of the same windows at the same
  `δ`: its budget is at least `S·(1 − δ^(1/S))`, and its greedy reaches the supremum over every share vector the draw
  allows.
* `PooledCoversSession`: the same at a session's sizes and draw counts (`Draw.Session.pooled`), the windows given as
  `(U_t, min(s_t, U_t))`.
* `LineCoversSession`: whenever the memory line gives a line for those windows, the W it reads (`Line.wrong`, every
  stretch's) is at least the session's pooled bound.

They take no premise. That the windows the line reads are the session's (each window's receipts and its draws, as the
verifier's record holds them, `audit.Record`) is the record's, outside this statement.
-/

@[expose] public section

open RemoteAudit

namespace Properties.RemoteAudit

/-- Whenever `pooled` gives a bound for windows of sizes `U` sampling `d` at `δ`, its wrong receipts are at least the
draw's pooled bound of the same windows (`Accounting.Remote.Draw.pooled`) at `δ`. -/
def PooledCovers : Prop :=
  ∀ (n : ℕ) (U d : Fin n → ℕ) (δ : ℚ) (b : Pooled.Bound),
    Pooled.pooled (Array.ofFn fun t => (U t, d t)) δ = .ok b →
      Accounting.Remote.Draw.pooled n U d δ ≤ b.wrong

/-- Whenever `pooled` gives a bound for a session's windows at outcome `ω`, each window's size and draw count
(`Draw.Session.samples`, `min(s_t, U_t)`), its wrong receipts are at least the session's pooled bound at `δ`
(`Draw.Session.pooled`). -/
def PooledCoversSession : Prop :=
  ∀ (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) (D : Accounting.Remote.Draw.Session n m s κ X)
    (ω : Accounting.Remote.Draw.Coins n κ X) (δ : ℚ) (b : Pooled.Bound),
    Pooled.pooled (Array.ofFn fun t => ((D.size t ω : ℕ), D.samples t ω)) δ = .ok b →
      D.pooled δ ω ≤ b.wrong

/-- Whenever the memory line (`RemoteAudit.Line.line`) gives a line for a session's windows at outcome `ω` at `δ`, the
wrong receipts it reads are at least the session's pooled bound at `δ` (`Draw.Session.pooled`). -/
def LineCoversSession : Prop :=
  ∀ (n m : ℕ) (s : Fin n → ℕ) (κ : ℕ) (X : Type) (D : Accounting.Remote.Draw.Session n m s κ X)
    (ω : Accounting.Remote.Draw.Coins n κ X) (δ : ℚ) (gs : Array Line.Group)
    (per blocks blockBytes chipBytes nodeBytes : ℕ) (rates : Line.Rates) (l : Line.Line),
    Line.line gs per blocks blockBytes chipBytes nodeBytes (Array.ofFn fun t => ((D.size t ω : ℕ), D.samples t ω)) δ
      rates = .ok l →
      D.pooled δ ω ≤ l.wrong

end Properties.RemoteAudit
