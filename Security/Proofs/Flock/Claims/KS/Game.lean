module

public import Proofs.Flock.Claims.KS.Fork
public import Proofs.Flock.Claims.Bridge.OfGame
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
import VCVio.Prelude.Core

/-!
# Escapes at a game's verifier

At the public-coin verifier of a game (`Bridge.ofGame`), the verifier's `H` log is the decision's calls on the game's
outcome (`Bridge.run_ofGame`), so an escape event is tied to the verifier's log (`Escapes.tied`, `Escapes.tied₁`) as
soon as it gives a collision among the decision's calls: on the outcomes of two runs forked at `j`
(`tiedColl_ofGame`), or of one run (`tiedColl_ofGame₁`).
-/

@[expose] public section

namespace Claims

open FlockSoundness FlockSoundness.Game Bridge

theorem tiedColl_ofGame {β Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (k : Msg → ℕ) (g : Msg → Game β) (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) (x : Msg) (ω : Ω)
    (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) (j : Fin (k x))
    (r r' : Fin (k x) → Coin)
    (h : TiedColl H (((d x (outcome H P k g I x ω r)).eval H).2 ++
      ((d x (outcome H P k g I x ω (forkCoins j r r'))).eval H).2)) :
    TiedColl H ((run H P (ofGame k g I d) x ω r).vLog ++
      (run H P (ofGame k g I d) x ω (forkCoins j r r')).vLog) := by
  have h2 : (run H P (ofGame k g I d) x ω (forkCoins j r r')).vLog =
      ((d x (outcome H P k g I x ω (forkCoins j r r'))).eval H).2 :=
    (run_ofGame H P k g I d x ω _ hR).2.1
  simp only [(run_ofGame H P k g I d x ω r hR).2.1, h2]
  exact h

theorem tiedColl_ofGame₁ {β Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) {Ω : Type} (P : Prover Ω)
    (k : Msg → ℕ) (g : Msg → Game β) (I : ∀ x, Iface Coin (g x)) (d : Msg → β → OC Bool) (x : Msg) (ω : Ω)
    (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) (r : Fin (k x) → Coin)
    (h : TiedColl H ((d x (outcome H P k g I x ω r)).eval H).2) :
    TiedColl H (run H P (ofGame k g I d) x ω r).vLog := by
  rw [(run_ofGame H P k g I d x ω r hR).2.1]
  exact h

end Claims
