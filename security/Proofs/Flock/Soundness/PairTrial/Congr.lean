module

public import Proofs.Flock.Recursive.ForkLast
public import Proofs.Flock.Soundness.PairTrial.Pick
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
import Proofs.Flock.Recursive.ForkInner
import Proofs.Flock.Recursive.ForkRuns
import Proofs.Flock.Soundness.Game

/-!
# Trial: `collOf_congr` and its two users, over a pair function

A scratch, not to merge. `collOf_congr` says that `collOf` reads its condition only up to `↔`, and the two picks of
the recursion (`clashPickCL`, `clashPickHL`) use it to ignore the last round's coin. Here the picks are `collOfP` at the
two outcomes as the witness, and `collOfP_congr` reads the condition up to `↔` at each witness. The pair of two
outcomes that clash is the parameter `pairC` (`pairH`): the two lemmas hold for any function of the two outcomes, since
it does not read the transcripts.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section PicksP

open Model Accounting Exec

/-- **`collOfP` reads its condition only up to `↔`, witness by witness.** -/
theorem collOfP_congr {W : Type} {p q : W → Prop} {pr : W → List UInt8 × List UInt8} (e : ∀ w, p w ↔ q w) :
    collOfP p pr = collOfP q pr := by
  have hpq : p = q := funext fun w => propext (e w)
  subst hpq
  rfl

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

/-- `clashPickCL` over a pair function: on two inner transcripts whose steps play the compiled table to outcomes that
clash, the pair of the two outcomes. -/
noncomputable def clashPickCLP (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (pairC : OutCL F D Sl → OutCL F D Sl → List UInt8 × List UInt8)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  collOfP (fun w : OutCL F D Sl × OutCL F D Sl => outCL T CT A H E Lf cls regs sch M t = some w.1 ∧
      outCL T CT A H E Lf cls regs sch M t' = some w.2 ∧ ClashCL H E Lf sch w.1 w.2)
    fun w => pairC w.1 w.2

/-- **`clashPickCLP` ignores the last round's coin.** -/
theorem clashPickCLP_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) (pairC : OutCL F D Sl → OutCL F D Sl → List UInt8 × List UInt8) :
    PickLastFree (r + 1) (clashPickCLP T CT A H E Lf cls regs sch M pairC) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outCL T CT A H E Lf cls regs sch M t = outCL T CT A H E Lf cls regs sch M t' := fun _ _ h => by
    unfold outCL
    rw [h]
  unfold clashPickCLP
  exact collOfP_congr fun w => by
    rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)]

/-- `clashPickHL` over a pair function: on two inner transcripts whose steps play candidate 3's table to outcomes that
clash against each other, the pair of the two outcomes. -/
noncomputable def clashPickHLP (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (pairH : OutHL F D Sl → OutHL F D Sl → List UInt8 × List UInt8)
    (t t' : List (List (SMsg T) × ((k : Fin r) → CT k))) : Pair :=
  collOfP (fun w : OutHL F D Sl × OutHL F D Sl => outHL T CT A H E Lf e x t = some w.1 ∧
      outHL T CT A H E Lf e x t' = some w.2 ∧ ClashRunHL H E Lf e.sch e.rsch.sch x.cap w.1 w.2)
    fun w => pairH w.1 w.2

/-- **`clashPickHLP` ignores the last round's coin.** -/
theorem clashPickHLP_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D)
    (pairH : OutHL F D Sl → OutHL F D Sl → List UInt8 × List UInt8) :
    PickLastFree (r + 1) (clashPickHLP T CT A H E Lf e x pairH) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outHL T CT A H E Lf e x t = outHL T CT A H E Lf e x t' := fun _ _ h => by
    unfold outHL
    rw [h]
  unfold clashPickHLP
  exact collOfP_congr fun w => by
    rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)]

end PicksP

end FlockSoundness.Discharge.Recursion
