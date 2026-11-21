module

public import Proofs.Flock.EndToEnd.Blocks.Statement
public import Proofs.Flock.EndToEnd.Period
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Game

/-!
# Proof of the integrity audit per server-time block, and pooled over a monitoring period's blocks

* `Flock.SecurityProofs.integrity_on`: `IntegrityPooled` on a set `I` of the period's sampling blocks. The probability
  that every sampling block of `I` is accepted and `I`'s incorrect proof blocks total at least `M` is at most
  `(1 − q)^M + Σ_{t ∈ I} (ε_ks,t + ε_bind,t)`. It is `IntegrityPooled`'s proof with every sampling block outside `I`
  weighing nothing: `monitoringPeriod_union` over the sampling blocks of `I` accepted with a sampled proof block incorrect, and
  `monitoringPeriod_horizon` with weight `|W_t|` at a sampling block of `I` and `0` elsewhere, where the sampling block's miss
  probability `≤ 1 = (1 − q)^0`.
* `IntegrityBlock`: `integrity_on` at block `b`'s sessions, with `M = ⌈ln(1/δ)/q⌉`.
* `IntegrityBlocks`: the union bound over the blocks (`value_exists_finset_le`), each block's by `IntegrityBlock`.
* `IntegrityPooledBlocks`: `IntegrityPooled` over the monitoring period's `start B` sessions, whose terms regroup by
  block (`sum_range_start`).
-/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- The incorrect proof blocks of the sampling blocks of `I`, as a sum over every sampling block weighing `0` outside `I`. -/
theorem sum_filter_incorrect (I : Finset ℕ) (l : List SamplingBlockRec) :
    ((l.filter fun r => r.idx ∈ I).map SamplingBlockRec.incorrect).sum =
      (l.map fun r => if r.idx ∈ I then r.incorrect else 0).sum := by
  induction l with
  | nil => rfl
  | cons r l ih =>
      by_cases h : r.idx ∈ I
      · simp [h, ih]
      · simp [h, ih]

end Flock.SecurityProofs
