module

public import Proofs.Flock.Recursive.ForkLast
public import Proofs.Flock.Soundness.PairTrial.PickS
import Proofs.Flock.Recursive.ForkInner
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.Soundness.Game

/-!
# Trial: the recursion's two picks with nothing chosen by the picker

A scratch, not to merge. `clashPickCL` and `clashPickHL` hand `collOf` the proposition "the two transcripts play to
outcomes that clash", and `collOf_congr` lets them ignore the last round's coin. Here each pick reads the two outcomes
from `outCL` (`outHL`) and hands them to a pick of two outcomes (`pairC`, `pairH`: the search of the outcomes' own
lists, `PairTrial.SessionS`). The picker takes no witness by choice, and no congruence lemma is needed: the pick reads
a transcript only through `outCL`. What `outCL` itself is (the played outcome, taken by choice among the outcomes the
steps play to, of which there is one) is unchanged.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section PicksS

open Model Accounting Exec

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

/-- `clashPickCL` with the outcomes read, not chosen: on two inner transcripts whose steps play the compiled table,
the pick of the two outcomes. -/
noncomputable def clashPickCLS (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (pairC : OutCL F D Sl → OutCL F D Sl → Pair)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  (outCL T CT A H E Lf cls regs sch M t).bind fun o =>
    (outCL T CT A H E Lf cls regs sch M t').bind fun o' => pairC o o'

/-- **`clashPickCLS` ignores the last round's coin.** -/
theorem clashPickCLS_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (pairC : OutCL F D Sl → OutCL F D Sl → Pair) :
    PickLastFree (r + 1) (clashPickCLS T CT A H E Lf cls regs sch M pairC) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outCL T CT A H E Lf cls regs sch M t = outCL T CT A H E Lf cls regs sch M t' := fun _ _ h => by
    unfold outCL
    rw [h]
  unfold clashPickCLS
  rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)]

/-- `clashPickHL` with the outcomes read, not chosen. -/
noncomputable def clashPickHLS (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (pairH : OutHL F D Sl → OutHL F D Sl → Pair)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  (outHL T CT A H E Lf e x t).bind fun o => (outHL T CT A H E Lf e x t').bind fun o' => pairH o o'

/-- **`clashPickHLS` ignores the last round's coin.** -/
theorem clashPickHLS_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (pairH : OutHL F D Sl → OutHL F D Sl → Pair) :
    PickLastFree (r + 1) (clashPickHLS T CT A H E Lf e x pairH) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outHL T CT A H E Lf e x t = outHL T CT A H E Lf e x t' := fun _ _ h => by
    unfold outHL
    rw [h]
  unfold clashPickHLS
  rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)]

end PicksS

end FlockSoundness.Discharge.Recursion
