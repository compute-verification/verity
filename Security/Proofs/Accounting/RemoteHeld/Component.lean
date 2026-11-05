module

public import Properties.RemoteAudit.Component
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
import Proofs.Accounting.RemoteHeld.Line
import Proofs.Pouw.Prob
import Proofs.Properties.RemoteAudit.Component

/-!
# The memory line of record at the bandwidth component, proved (`Properties.RemoteAudit.Component.LineHeldAtComponent`)

Where the component accepts bytes that play the record (`Plays`), the Lean verifier accepts the record's session
(`plays_verified`, from `acceptIsVerified`), so the event is inside `LineHeld`'s (`Pouw.pr_mono`), whose bound
`lineHeld` gives.
-/

@[expose] public section

namespace Properties.RemoteAudit.Component.SecurityProofs

open Properties.RemoteAudit.Line

theorem lineHeldAtComponent : LineHeldAtComponent := by
  intro node κ X Cm _ prf rest recs correct com T R δs a b e r F O hL hS hD hC hR hP hCv hSk hN hLive hRef hCnt
  exact le_trans (Pouw.pr_mono fun ω h => ⟨plays_verified h.1, h.2⟩)
    (Properties.RemoteAudit.Line.SecurityProofs.lineHeld node κ X Cm prf rest recs correct com T R δs O hL hS hD hC hR hP hCv hSk hN hLive
      hRef hCnt)

end Properties.RemoteAudit.Component.SecurityProofs
