module

public import Definitions.Flock.Accounting.Padded
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.MeasureTheory.Integral.Bochner.Basic
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
import VCVio.Prelude.Core

/-!
# The level-0 padding geometry of `CoinedOK`

`geometry_of_padOK`: at `cols` level 0's message columns (`logCols`) and `t_pad` its padding, the padded level 0's
conditions (`Accounting.PadOK`) and rate `1/2` (`logLen = logCols + 1`) give C-Flock ZK's geometry: `cols ≤ 40`,
`t_pad ≤ 2^cols`, and opened positions inside the codeword are inside `2^(cols+1)`.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Accounting

theorem geometry_of_padOK (l : Level) (t nS : ℕ) (pos : Fin nS → ℕ) (hp : PadOK l t)
    (hrate : l.logLen = l.logCols + 1) (hS : nS ≤ t) (hinj : Function.Injective pos)
    (hlt : ∀ q, pos q < 2 ^ l.logLen) :
    l.logCols ≤ 40 ∧ t ≤ 2 ^ l.logCols ∧ nS ≤ t ∧ Function.Injective pos ∧ ∀ q, pos q < 2 ^ (l.logCols + 1) :=
  ⟨by have := hp.hd; omega, hp.ht, hS, hinj, fun q => hrate ▸ hlt q⟩

end FlockSoundness.Discharge.Composed
