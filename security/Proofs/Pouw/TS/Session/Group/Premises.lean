module

public import Proofs.Pouw.TS.Session.Group.Assumptions
public import Proofs.Pouw.TS.Session.Assumptions
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

/-! The premises a leaf proved as a group of Flock sessions takes, each a named assumption at every session of every
proof, in the module `shape.premises` in `security/lean-audit.json` names. -/

@[expose] public section

namespace ResourceAccounting.Premises.TS

open ResourceAccounting.TS Flock.SoundAtBridge
open Pouw.RelativeSecurity (Paths)

/-- **`RowsSession.CR` at every session of every proof**: `EndToEndRegisteredRows`' named assumptions. -/
abbrev SessionsCR {Ω : Type} {S nπ : ℕ} {ι : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession) : Prop :=
  ∀ x w i k, (sess x w i k).CR

/-- **`ResourceAccounting.Assumptions.TS.ExecutionPhaseDesign` at every statement of the group**. -/
abbrev ExecutionPhaseDesigns {Ω : Type} [Fintype Ω] {S nπ : ℕ} {ι : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession)
    (out : ∀ x w i k, (sess x w i k).Out) : Prop :=
  ∀ k, ResourceAccounting.Assumptions.TS.ExecutionPhaseDesign (fun x w i => sess x w i k) (fun x w i => out x w i k)

/-- **`ResourceAccounting.Assumptions.TS.GroupRealizes` at every proof**. -/
abbrev GroupRealizes {Ω : Type} {S nπ : ℕ} {ι V : Type} (sess : Paths Ω S → Fin S → Fin nπ → ι → RowsSession)
    (known : Paths Ω S → Fin S → Fin nπ → ι → ResourceAccounting.TS.Lanes.Known)
    (full : Paths Ω S → Fin S → Fin nπ → ι → Array String)
    (want : Paths Ω S → Fin S → Fin nπ → ι → List _root_.Flock.Registered.Port)
    (gate : Paths Ω S → Fin S → Fin nπ → Prop) (opened : Paths Ω S → Fin S → Fin nπ → V)
    (rel : Paths Ω S → Fin S → Fin nπ → V → Prop) : Prop :=
  ∀ x w i, ResourceAccounting.Assumptions.TS.GroupRealizes (sess x w i) (known x w i) (full x w i) (want x w i) (gate x w i)
    (opened x w i) (rel x w i)

end ResourceAccounting.Premises.TS
