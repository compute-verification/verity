module

public import Proofs.Flock.Claims.ZK.FinishSwMk
public import Proofs.Flock.Claims.ZK.FrameTape
public import Flock.ZkServer.Finish
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
import Proofs.Flock.Claims.Complete.Zk.PadAlgebra

/-! `hmk` at the ZK server's own request. -/

@[expose] public section

theorem Flock.ZkServer.Finish.padsOf_h (d : _root_.Flock.ZkServer.Frame.Draws) :
    (_root_.Flock.ZkServer.Finish.padsOf d).h = d.h := rfl

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem pads_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.ZkServer.Finish.RepIn)
    (hq : q.draws = L.samplesAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.ZkServer.Frame.sampleBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfSamples t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k) :
    ∀ k < (layE x j).hab, (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j)).h[k]! =
      (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[k]! := by
  intro k hk
  have hlen : (layE x j).len = (layE x j).tw + 1 := rfl
  have htw : (layE x j).tw = (layE x j).hab + 3 := rfl
  rw [FlockComplete.padsOf_get (UE.tapeZk τ) j.val r.val (layE x j) k (by omega) (by omega) (by omega), htape k hk,
    ← FlockComplete.padsOf_get (Claims.ZKServer.tapeOfSamples t₀ (L.lay x) (L.draws ρ) rep) j.val r.val (layE x j) k
      (by omega) (by omega) (by omega),
    ← hlay, Claims.ZKServer.padsOf_samplesAt L x ρ t₀ rep j.val r.val hz, hq]
  rfl

theorem swSrcE_hmk_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.ZkServer.Finish.RepIn)
    (hA : A j r = q.vals) (hq : q.draws = L.samplesAt x ρ (rep j.val r.val))
    (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.ZkServer.Frame.sampleBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfSamples t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1) :
    ∀ k < (layE x j).zp 64, ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[k]! := by
  intro k hk
  rw [swSrcE_hmk x c w A B τ j r hrows k hk, hA,
    pads_frame L ρ t₀ rep x τ j r q hq hlay hz htape k (lt_trans hk (zp64_lt_hab x j))]

theorem swSrcE_l0_frame {W R : Type} (L : Claims.ZKServer.LeanRelease W R) (ρ : R) (t₀ : FlockComplete.TapeZk)
    (rep : ℕ → ℕ → ℕ) (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2) (q : _root_.Flock.ZkServer.Finish.RepIn)
    (hq : q.draws = L.samplesAt x ρ (rep j.val r.val)) (hlay : L.lay x (rep j.val r.val) = layE x j)
    (hz : (L.draws ρ (rep j.val r.val)).size = _root_.Flock.ZkServer.Frame.sampleBytes (L.lay x (rep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfSamples t₀ (L.lay x) (L.draws ρ) rep).h j.val r.val k)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1) :
    ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).masked k =
      B j r k + (_root_.Flock.ZkServer.Finish.padsOf q.draws).h[k]! := by
  intro k hk0 hk
  rw [swSrcE_l0 x c w A B τ j r hrows k hk0 hk, pads_frame L ρ t₀ rep x τ j r q hq hlay hz htape k hk]

end Masked
end Claims.ZKMal
