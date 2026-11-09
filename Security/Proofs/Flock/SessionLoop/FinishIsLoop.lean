module

public import Proofs.Flock.SessionLoop.FinRepL0
public import Proofs.Flock.SessionLoop.FinRepRestart
public import Proofs.Flock.SessionLoop.FinRepTarget
public import Proofs.Flock.SessionLoop.FinishIsRet
public import Proofs.Flock.SessionLoop.FinishParts
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
# `FinishIs` at `SessionLoopZKDeployed`'s premises

`finishIs_of_ret` takes three facts beyond `SessionLoopZKDeployed`'s premises: the ring switch's target
(`Loop.finRepTarget`), the restart (`Loop.finRepRestart`) and level 0's openings (`Loop.finRepL0`). Each holds in the
masked scope with a witness where the loop samples level 0 from the tape (`Loop.Src`'s level-0 half).

**`Loop.finishIsOn`**: `FinishIs` at `SessionLoopZKDeployed`'s premises, with no other hypothesis; `Loop.finishIs` is
it at causal switched values.
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src (realOf src)).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinishIs` at `SessionLoopZKDeployed`'s premises**: in the masked scope, with a witness, at a statement the loop
serves whose inner round 0 is the loop's and whose inputs from its host are the tape's, at live rounds the loop's and
switched values causal at `Loop.SwOk`'s coins, the `Finish` the loop runs is the session's (`finishIs_of_ret`, at
`finRepTarget`, `finRepRestart` and `finRepL0`). -/
theorem finishIsOn (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEqOwn finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausalOn (Loop.SwOk Masked.dataE) vs) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w :=
  finishIs_of_ret vs L₀ src wk x w hsc hR hServe hL0 hSrc hE hc
    (finRepTarget vs L₀ src (realOf src) wk x w hsc hR hServe hSrc.1)
    (finRepRestart vs L₀ src (realOf src) wk x w hsc hR hSrc.1)
    (finRepL0 vs L₀ src (realOf src) wk x w hsc hR hSrc.1)

/-- **`finishIsOn` at causal switched values** (`swCausalOn_of_swCausal`). -/
theorem finishIs (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Serves x)
    (hL0 : _root_.Flock.Guarantees.LoopInnerRound0 Masked.dataE L₀ (realOf src) x w) (hSrc : Loop.Src L₀ src x w)
    (hE : LeanRelease.LiveRoundEq finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w)
    (hc : finS.SwCausal vs) :
    LeanRelease.FinishIs finS vs (LeanRelease.ofLoopLink L₀ src (realOf src)) CoinOpen.shaH wk x w :=
  finishIsOn vs L₀ src wk x w hsc hR hServe hL0 hSrc hE.own (finS.swCausalOn_of_swCausal vs _ hc)

end Loop

end Claims.ZKServer
