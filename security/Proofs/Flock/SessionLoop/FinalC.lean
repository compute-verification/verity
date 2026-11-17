module

public import Proofs.Flock.SessionLoop.TakenVals
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

/-! `Loop.FinalCFixed`, the premise over `vs` that its `final_c` is the c_eval of its round-1c values, holds at the honest values. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked

namespace Loop

def FinalCFixed (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin) (τ : UE.Tape) (j : Fin (jOf x)),
    (∀ (r : Fin 2), ∀ y ∈ (tabC PE x j).rOuter (czExe PE x c j) r, y ≠ 1) → ∀ (r : Fin 2),
    vsWord vs x w ν c τ j r (layE x j).fc =
      _root_.Flock.dot (_root_.Flock.lagrange _root_.Flock.nodesL _root_.Flock.den6Inv (zcE c j r).z)
        ((Array.range 64).map fun i => vsWord vs x w ν c τ j r ((layE x j).r1c i))

theorem vsWord_honestVals (x : Msg) (w : Flock.Wit) (ν : CoinLayer.Nonce) (c : Fin (refK x) → FlockDeploy.Coin)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) {k : ℕ} (hk : k < (tabC PE x j).z.nE) :
    vsWord (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w ν c τ j r k =
      FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt (zkTab x j).table (stAt x j) (czExe PE x c j) r
        (FlockSoundness.Discharge.Composed.zWith (stAt x j) (wtOf PE x w j)
          (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1)
        ((((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r) k) := by
  unfold vsWord
  rw [dite_eq_left hk]
  rfl

/-- Holds at every coins; the outer coins' restriction is unused. -/
theorem finalCFixed_honestVals (x : Msg) (w : Flock.Wit) :
    FinalCFixed (CoinLayer.session Loop.loopNr Masked.dataE).honestVals x w := by
  intro ν c τ j _ r
  have hE : (tabC PE x j).z.nE = (layE x j).hab := by
    show (layE x j).len - 4 = (layE x j).hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hr1c : ∀ i < 64, (layE x j).r1c i < (tabC PE x j).z.nE := fun i _ => by
    rw [hE]
    exact lt_hab (layE x j) (by
      simp only [layE, _root_.Flock.Zk.PadLayout.new, _root_.Flock.Zk.PadLayout.r1c, _root_.Flock.Zk.PadLayout.fc,
        _root_.Flock.Zk.PadLayout.fa]
      omega)
  rw [vsWord_honestVals x w ν c τ j r (k := (layE x j).fc) (by rw [hE]; exact lt_hab (layE x j) le_rfl)]
  refine (congrArg FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt_fc (zkTab x j).table
    (stAt x j) (czExe PE x c j) r _ _)).trans ?_
  show (FlockSoundness.Discharge.Composed.zcH (zkTab x j).table (stAt x j) (czExe PE x c j) r _).finalC = _
  unfold FlockSoundness.Discharge.Composed.zcH
  rw [← FlockSoundness.Discharge.Composed.Zerocheck.c_eval]
  refine congrArg₂ _root_.Flock.dot rfl (Array.ext ?_ fun i h1 h2 => ?_)
  · simp [FlockSoundness.Discharge.Composed.Zerocheck.honest]
  · have hi : i < 64 := by simpa using h2
    simp only [Array.getElem_map, Array.getElem_range]
    rw [vsWord_honestVals x w ν c τ j r (hr1c i hi), ← getElem!_pos _ i h1]
    exact (congrArg FlockLevel3.GF128.val (FlockSoundness.Discharge.Composed.honestAt_r1c (zkTab x j).table
      (stAt x j) (czExe PE x c j) r _ _ i hi)).symm

end Loop

end Claims.ZKServer
