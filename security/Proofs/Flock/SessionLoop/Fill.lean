module

public import Proofs.Flock.SessionLoop.FinishParts
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

/-! `FillRoundEq` at the honest worker, whose frame checks read the history only through its decoded coins, the release's `cross` being Rust's own checks. -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open Claims.ZKMal.CoinLayer Claims.ZKMal.Masked
open _root_.Flock (V F128)
open _root_.Flock.ZkServer.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace LeanRelease

theorem roundOk_of_refusesCoins {W R : Type} (L : LeanRelease W R) (x : Msg) (w : W) (ρ : R) (g : List (Msg × Msg))
    (hr : L.refusesCoins x ρ g = true) : L.roundOk x w ρ g = false := by
  have ht : L.takes x ρ g (L.devOf x w ρ g) = false := by
    unfold refusesCoins at hr
    unfold takes step
    cases ha : L.answered x ρ g with
    | ok m => rw [ha] at hr; cases hr
    | error e => rfl
  unfold roundOk
  rw [ht, Bool.and_false]

theorem refusesCoins_withDevice {W R : Type} (L : LeanRelease W R)
    (d : Msg → W → R → List (Msg × Msg) → (String → Option ByteArray) → Msg) (x : Msg) (ρ : R)
    (g : List (Msg × Msg)) : (L.withDevice d).refusesCoins x ρ g = L.refusesCoins x ρ g :=
  rfl

end LeanRelease

namespace Loop

/-- Completeness, which Lean does not model for the deployed device; the honest session of
`verity/protocols/proofs/flock/prover/agreement/zk_server_controls.sh` checks it under `FC_ZK_SERVER=1`. -/
def RoundsPass (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) : Prop :=
  ∀ (ν : finS.Nonce) (t : finS.Tape) (g : List (Msg × Msg)), 2 ≤ g.length → g.length ≤ refK x →
    L.refusesCoins x (ν, t) g = false → L.roundOk x w (ν, t) g = true

def HonestRoundsPass (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit) : Prop :=
  RoundsPass (LeanRelease.ofLoopLink L₀ src real) x w

theorem live_honestWk (L : LeanRelease Flock.Wit Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : RoundsPass L x w) (ν : finS.Nonce) (t : finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (hc : ∀ i, 2 ≤ i → i ≤ ((m₀, ρ₀) :: ops).length → L.refusesCoins x (ν, t) (((m₀, ρ₀) :: ops).take i) = false) :
    L.toSrv.live L.honestWk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true := by
  refine L.toSrv_live_honest x w (ν, t) κ _ (by rw [List.length_cons]; omega) fun i h2 hi => ?_
  have hi' : i ≤ ops.length + 1 := by simpa using hi
  exact hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) (hc i h2 hi)

theorem live_honest (L₀ : LeanRelease Flock.Wit Tp)
    (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : HonestRoundsPass L₀ src real x w)
    (ν : finS.Nonce) (t : finS.Tape) (κ : (LeanRelease.ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg)) (hlt : ops.length < refK x)
    (hc : ∀ i, 2 ≤ i → i ≤ ((m₀, ρ₀) :: ops).length →
      (LeanRelease.ofLoopLink L₀ src real).refusesCoins x (ν, t) (((m₀, ρ₀) :: ops).take i) = false) :
    (LeanRelease.ofLoopLink L₀ src real).toSrv.live (LeanRelease.ofLoopLink L₀ src real).honestWk x w (ν, t) κ
      ((m₀, ρ₀) :: ops) = true := by
  have hlast : (LeanRelease.ofLoopLink L₀ src real).last x = refK x + 1 :=
    LeanRelease.loop_last loopNr Masked.dataE _ x
  exact live_honestWk _ x w (by omega) hr ν t κ m₀ ρ₀ ops hlt hc

end Loop

namespace LeanRelease

theorem fillRoundEq_live (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (wk : L.toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : L.toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      L.refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false → L.toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH wk x w :=
  fun ν t κ m₀ ρ₀ ops hp hlt hf _ hr => (Bool.false_ne_true (hf.symm.trans (hl ν t κ m₀ ρ₀ ops hp hlt hr))).elim

theorem fillRoundEq_of_live (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)
    (hl : ∀ (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk) (m₀ ρ₀ : Msg)
      (ops : List (Msg × Msg)), 0 < ops.length → ops.length < refK x →
      (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false →
      (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH wk x w :=
  fillRoundEq_live vs _ wk x w hl

theorem fillRoundEq_honestWk (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.RoundsPass L x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH L.honestWk x w :=
  fillRoundEq_live vs L _ x w fun ν t κ m₀ ρ₀ ops _ hlt hc =>
    Loop.live_honestWk L x w hlast hr ν t κ m₀ ρ₀ ops hlt fun i _ _ =>
      Bool.eq_false_of_ne_true fun h => Bool.false_ne_true (hc.symm.trans (L.refusesCoins_of_take x (ν, t) _ i h))

theorem fillRoundEq_honest (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (x : Msg) (w : Flock.Wit)
    (hr : Loop.HonestRoundsPass L₀ src real x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs (ofLoopLink L₀ src real) CoinOpen.shaH
      (ofLoopLink L₀ src real).honestWk x w := by
  have hlast : (ofLoopLink L₀ src real).last x = refK x + 1 := loop_last Loop.loopNr Masked.dataE _ x
  exact fillRoundEq_honestWk vs _ x w (by omega) hr

theorem fillRoundEq_valsWk (vs vs' : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.RoundsPass (L.withDevice (Loop.valsDev vs' L)) x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH (Loop.valsWk vs' L) x w :=
  fillRoundEq_live vs L _ x w fun ν t κ m₀ ρ₀ ops _ hlt hc =>
    Loop.live_valsWk vs' L x w (ν, t) κ _ (by rw [List.length_cons]; omega) fun i h2 hi => by
      have hi' : i ≤ ops.length + 1 := by simpa using hi
      refine hr ν t _ (by rw [List.length_take]; omega) (by rw [List.length_take]; omega) ?_
      rw [refusesCoins_withDevice]
      exact Bool.eq_false_of_ne_true fun h =>
        Bool.false_ne_true (hc.symm.trans (L.refusesCoins_of_take x (ν, t) _ i h))

theorem fillRoundEq_honestVals (vs : (CoinLayer.session Loop.loopNr Masked.dataE).SwVals)
    (L : LeanRelease Flock.Wit Loop.Tp) (x : Msg) (w : Flock.Wit) (hlast : refK x < L.last x)
    (hr : Loop.RoundsPass
      (L.withDevice (Loop.valsDev (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L)) x w) :
    FillRoundEq (CoinLayer.session Loop.loopNr Masked.dataE) vs L CoinOpen.shaH
      (Loop.valsWk (CoinLayer.session Loop.loopNr Masked.dataE).honestVals L) x w :=
  fillRoundEq_valsWk vs _ L x w hlast hr

end LeanRelease

end Claims.ZKServer
