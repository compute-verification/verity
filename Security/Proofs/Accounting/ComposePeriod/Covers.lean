import Proofs.Accounting.Compose.ServiceCovers
import Definitions.Accounting.ComposePeriod

/-!
# The period layer's covers: `ServiceCovers` with a window index

`PeriodCoversStmt` is the one piece of `CircuitWorkPeriod`'s proof (`Proofs.Accounting.ComposePeriod`) stated here for
training's Lean worker to prove, as `theorem periodCovers : PeriodCoversStmt` in this file. The assembly reads it as a
hypothesis (`ComposePeriod.circuitWorkPeriod_of_covers`) until that theorem lands.

It is `CoversProof.serviceCovers`' argument with a window index. Every position, cut and writer of window `w` of the
period layer (`Declared.periodLayer`) is window `w`'s, so a late edge (`DrawnLayer.Late`) stays in its window and
lies in the IR's late cone (`Declared.Reaches`) under `Declared.Wf` and `LayerCommitted`, as `reaches_of_late` shows
for one window. A credited window's live tiles have its own run's modes under `periodMode`, and its dead tiles are
skipped whole. So a live tile of words that isn't good is reached by a tile its window's run didn't pay for: either a
live one, whose `liveCone` the window's round charges and which holds the tile's live words, or a dead one, which
`DeadConeDead` says reaches no live words.
-/

namespace Accounting.SecurityProofs.ComposePeriod

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

open Classical in
/-- **The period layer's covers on a credited window's live tiles** (for training's Lean worker to prove): the period
layer's writers read their positions (`DrawnLayer.WriterReads`), and on every path `x`, in every credited window `w`
whose dead tiles' cones hold no live words (`DeadConeDead`), the live words of the tiles that aren't good under the
period's modes (`periodMode`) are at most `N_w` times the `liveCone` of the tiles window `w`'s run didn't pay for, at
its rounds' coins. The right side is window `w`'s round's deficiency in `periodGame` at weights `liveCone` and costs
`1, 1`. -/
def PeriodCoversStmt : Prop :=
  ∀ {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (pr : PeriodRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V),
    Cd.Wf → LayerCommitted Cd →
    (Cd.periodLayer hr hK pr).WriterReads ∧
      ∀ (x : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n), pr.Credited x w →
        DeadConeDead Sc Cd Np (Wp.liveAt x w) →
        (∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t then (0 : ℝ)
          else (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ)) ≤
          (∑ u : Fin Np, if (Wp.windowAt x w).modeAt (tileStage Cd Np) (coinAt x w).1 u = .paid then 0
            else liveCone Sc Cd Np (Wp.liveAt x w) u) * (wordsOf Sc Cd : ℝ)

end Accounting.SecurityProofs.ComposePeriod
