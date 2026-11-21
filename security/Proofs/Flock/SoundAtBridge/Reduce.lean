module

public import Proofs.Flock.SoundAtBridge.Statement
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
import Proofs.Flock.Soundness.Game

/-!
# `SoundAtBridge` from `EndToEndRegisteredRows`

* `RowsSession.prob_badU_le` (gap 3): at a session with one registrant's opening per unit, the per-unit event has
  probability at most `n` times `EndToEndRegisteredRows`' bound: that guarantee at each unit's opening, and the union
  bound over the units (`game_prob_exists_le`);
* `openedPlur_zero` (gap 2 at `δo = 0`): a layer whose accepted opening is the value layer's reading meets
  `OpenedPlurAt` at `0`;
* `soundAtBridge_of_rows`: `SoundAtBridge` from `EndToEndRegisteredRows`, through `soundAt_of_sessions`.
-/

@[expose] public section

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.ZkReg ResourceAccounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

namespace RowsSession

end RowsSession

variable {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type}

/-- **Gap 2 at `δo = 0`**: a layer whose accepted opening is its session's value layer read as a layer value meets
`OpenedPlurAt` at `0`. -/
theorem openedPlur_zero (pf : ProofLayer Ω S nπ V) (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V)
    (h : ∀ x w i, pf.accepted x w i → pf.opened x w i = rows (sess x w i) (sess x w i).Xv) :
    OpenedPlurAt pf sess rows 0 := by
  intro w i
  have h0 : (fun x => pf.accepted x w i ∧ pf.opened x w i ≠ rows (sess x w i) (sess x w i).Xv) = fun _ => False :=
    funext fun x => propext ⟨fun ⟨ha, hne⟩ => hne (h x w i ha), False.elim⟩
  rw [h0]
  unfold Pouw.RelativeSecurity.prob Claims.prUniform
  simp

/-- **The proof layer a family of sessions gives**: proof `i` of round `w` is accepted exactly where verity-verify-proof
accepts its session's outcome, its opening is the session's value layer read as a layer value, and its relation holds
there when no sampled proof block of the session is wrong or off at its own reads. -/
noncomputable def rowsLayer (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) :
    ProofLayer Ω S nπ V where
  accepted x w i := LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)
  opened x w i := rows (sess x w i) (sess x w i).Xv
  rel x w i _ := ∀ u ∈ (out x w i).2.1, ¬ (sess x w i).IncorrectAt u ∧ (sess x w i).RowsAt (out x w i) u
  right := right

/-- **The layer's premises hold together, at verity-verify-proof's acceptance** (non-vacuity of `AcceptedLive`,
`OpenedPlurAt` and `RelOfRows`): at any family of sessions, `rowsLayer` accepts exactly where verity-verify-proof accepts, and
meets `AcceptedLive`, `OpenedPlurAt` at `δo = 0` and `RelOfRows`. `ExecutionPhaseSamplingDesign` and `RowsSession.CR` are premises of the
sessions, not of the layer. -/
theorem rowsLayer_premises (sess : Paths Ω S → Fin S → Fin nπ → RowsSession) (out : ∀ x w i, (sess x w i).Out)
    (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V) (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) :
    (∀ x w i, (rowsLayer sess out rows right).accepted x w i ↔
      LiveAcceptsZKR (sess x w i).I (sess x w i).own (out x w i)) ∧
    AcceptedLive (rowsLayer sess out rows right) sess out ∧ OpenedPlurAt (rowsLayer sess out rows right) sess rows 0 ∧
    RelOfRows (rowsLayer sess out rows right) sess out rows :=
  ⟨fun _ _ _ => Iff.rfl, fun _ _ _ h => h, openedPlur_zero _ sess rows fun _ _ _ _ => rfl, fun _ _ _ h => h⟩

end Flock.SoundAtBridge
