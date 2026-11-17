module

public import Proofs.Flock.Claims.ZK.Split
public import Proofs.Flock.Claims.ZK.Blind
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

/-! The switched prover's rep source. -/

@[expose] public section

theorem FlockSoundness.ZK.Table.swPre_masked {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.Rand)
    (hr : T.onVals.RowsAt (w, v) ω.1 ω.2.1) : (T.swPre w v ω).masked = (T.onVals.view (w, v) ω).masked := by
  rw [FlockSoundness.ZK.Table.swPre, if_pos hr]
  rfl

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

theorem swCutAt_masked {J : ℕ} (C : Fin J → Composed.CoinedSetup) (cz : ∀ j, Composed.CoinsZK (C j).Rest)
    (wt : ∀ j, FlockSoundness.Model.Witness (C j).S.m) (v : ∀ j, ((C j).T (cz j)).tab.Vals)
    (ω : ∀ j, ZkView.Base (C j).z (C j).Hi) (lv : ZkView.Leaves (fun j => (C j).z) → ByteArray) (j : Fin J) :
    (swCutAt C cz wt v ω lv j).masked =
      (((C j).T (cz j)).tab.swPre (wt j) (v j) (((C j).T (cz j)).randOf (ω j) 0)).masked := rfl

namespace Masked

/-- **The switched prover's cut at the opened coins `c`**, on `w`'s table witnesses, the workers' values `v` and the
tape `τ` (`RefSession.swCutOf`), as `cutE` has the honest prover's. -/
noncomputable def swCutE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) :
    ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Cut :=
  swCutAt (tabC PE x) (czExe PE x c) (wtOf PE x w) v (UE.tape x τ).1
    (swExeOf (tabC PE x) (czExe PE x c) (wtOf PE x w) v (UE.tape x τ).1 (UE.tape x τ).2.1)

theorem swCutOf_eq (nr : Msg → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : (∀ j, ZkView.Seen (tabC PE x j).z) → ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) :
    (session nr (tableData PE QE UE)).swCutOf x c (wtOf PE x w) v τ =
      swCutE x c w (v ((session nr (tableData PE QE UE)).seenOf x τ)) τ := rfl

/-- **What the deployed session reads from the switched cut** (`tabSrcCut`), as `srcE` reads `cutE`. -/
noncomputable def swSrcE (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) : ℕ → FlockComplete.TabSrc :=
  tabSrcCut PE x (czExe PE x c) (swCutE x c w v τ)

theorem swCutE_masked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swCutE x c w v τ j).masked =
      (((tabC PE x j).T (czExe PE x c j)).tab.swPre (wtOf PE x w j) (v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0)).masked := by
  unfold swCutE
  exact swCutAt_masked _ _ _ _ _ _ j

theorem swCutE_masked_of_rowsAt (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x))
    (hr : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1) :
    (swCutE x c w v τ j).masked =
      (((tabC PE x j).T (czExe PE x c j)).tab.onVals.view (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0)).masked := by
  rw [swCutE_masked]
  exact Table.swPre_masked _ _ _ _ hr

end Masked
end Claims.ZKMal
