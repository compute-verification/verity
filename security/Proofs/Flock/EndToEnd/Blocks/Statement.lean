module

public import Proofs.Flock.EndToEnd.Period.Statement
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

/-!
# A monitoring period of server-time blocks (trusted)

A server's life is cut into server-time blocks by certified wipes. Over a monitoring period, its `--zk` sessions are
numbered in serve order, session `t` a `SamplingBlock` `W t` with a fresh live subset sample, and block `b` is the
sessions from `start b` up to `start (b + 1)` (`ServerBlocks`): a wipe separates block `b`'s last session from block
`b + 1`'s first. One prover strategy plays every session of every block (`ServerBlocks.game`, `Period.game` over the
monitoring period's `T` sessions), adaptively, so a block's sessions may depend on everything the prover saw in the
blocks before. The state a block carries across a wipe is `EndToEnd/Carry/Statement.lean`'s.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

/-- **A server's sessions over a monitoring period, cut at wipes into `B` server-time blocks.** Session `t` is `W t`,
in serve order; block `b` is the sessions from `start b` up to `start (b + 1)`, so the blocks partition the monitoring
period's first `start B` sessions, and a wipe sits before session `start b` for each `0 < b < B`. -/
structure ServerBlocks where
  W : ℕ → SamplingBlock
  B : ℕ
  start : ℕ → ℕ
  start_zero : start 0 = 0
  mono : Monotone start

namespace ServerBlocks

variable (S : ServerBlocks)

/-- The monitoring period's number of sessions. -/
def T : ℕ := S.start S.B

/-- The monitoring period's game: every block's sessions in serve order, against one prover. -/
noncomputable def game : Game Unit := Period.game S.W S.T

end ServerBlocks

end Flock.Guarantees
