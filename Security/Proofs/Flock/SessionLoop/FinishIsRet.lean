module

public import Proofs.Flock.SessionLoop.FinRepLanesFw
public import Proofs.Flock.SessionLoop.FinRepRows
public import Proofs.Flock.SessionLoop.FinishIsOf
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
import Proofs.Flock.SessionLoop.FinRepMsgsFw

/-!
# `FinishIs` with the clear values and level 0 read off the returns

`finishIs_of_consumed` takes four facts `FinRepRows` consumes and level 0's openings. Two of them hold at its premises
with no rows: below `zp 64` the workers' clear values are the request's (`Loop.finRepMsgs_ret`), and the returned
proof's level 0 is the clear values plus the request's pads (`Loop.finRepLanes_ret`).

**`Loop.finishIs_of_ret`**: `FinishIs` from the ring switch's target, the restart and level 0's openings alone.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinishIs` from the target, the restart and level 0's openings**: `finishIs_of_consumed` at its premises, its
clear values below `zp 64` and its level 0 values from the returns (`finRepMsgs_ret`, `finRepLanes_ret`). -/
theorem finishIs_of_ret (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEqOwn finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausalOn (Loop.SwOk Masked.dataE) vs)
    (hT : FinRepTarget vs L₀ src (realOf src) wk x w) (hTP : FinRepRestart vs L₀ src (realOf src) wk x w)
    (hL : FinRepL0 vs L₀ src (realOf src) wk x w) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w :=
  finishIs_of_consumed vs L₀ src wk x w hsc hR hServe hL0 hSrc hE hc
    (finRepMsgs_ret L₀ src wk x w vs hsc hR hServe hSrc.2 hSrc.1 hE hc)
    (finRepLanes_ret L₀ src wk x w vs hsc hR hServe hL0 hSrc.2 hSrc.1 hE hc) hT hTP hL

end Loop

end Claims.ZKServer
