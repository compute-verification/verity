module

public import Proofs.Flock.C3.VStarList.Default
public import Proofs.Flock.Recursive.HoloExec
public import Proofs.Flock.Recursive.VStarCheck
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
# Piece (h): V*'s list and R's shape

The verifier's own list of V*'s statements (`RecVStar.statements`) has `vstarCount` entries (`vstarCount_eq`). At
candidate 3's inner shape the list is circuits' gap 2's eleven statements, in the writer's order: four `rec-open`, two
`rec-residuals`, one `rec-sparse`, R's three levels (`rec-open-stacked`) and one `rec-opening`.

R's shape is derived, not a hypothesis: each `rec-opening/m<m>-c<c>-p<j>` unit the verifier checks opens R at META's
`stacked` entry (`stacked_of_openingShape`, `checkOpeningShape`), and that entry opens exactly `STACKED_CLAIMS` claims
(`claims_of_stackedShape`, `checkStackedShape`), whose rounds are `fast100`'s at its `m`. The link to the bucket's
`table_m` is transitive, through the outer program's `rec-opening` units, whose names the verifier compiles; the list's
own `rec-opening` statements are at `mReg` (`mReg_of_list`, `Shape.lean`).
-/

@[expose] public section

namespace FlockC3.VStarList

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Spot.Sampling Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- **The count is the list's size** where the verifier lists V*'s statements. -/
theorem vstarCount_eq {ish : Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : Flock.RecVStar.statements ish = .ok ss) : vstarCount ish = ss.size := by
  unfold vstarCount
  rw [h]

theorem stackedShape_beq_eq {a b : Flock.RecVStar.StackedShape} (h : (a == b) = true) : a = b := by
  obtain ⟨am, ac⟩ := a
  obtain ⟨bm, bc⟩ := b
  have e : (({ m := am, claims := ac } : Flock.RecVStar.StackedShape) == { m := bm, claims := bc }) =
      (am == bm && ac == bc) := rfl
  rw [e, Bool.and_eq_true, beq_iff_eq, beq_iff_eq] at h
  rw [h.1, h.2]

theorem optStacked_eq_of_not_bne {a b : Option Flock.RecVStar.StackedShape} (h : (a != b) = false) : a = b := by
  have hb : (a == b) = true := by
    rw [bne, Bool.not_eq_false'] at h
    exact h
  cases a with
  | none => cases b with
    | none => rfl
    | some _ => cases hb
  | some x => cases b with
    | none => cases hb
    | some y => exact congrArg some (stackedShape_beq_eq hb)

/-- **An R opening unit opens R at META's shape**: `checkOpeningShape` passing at a `rec-opening` unit of `m` and `c`
claims means the inner commitments' `stacked` entry is `(m, c)`. -/
theorem stacked_of_openingShape {unit : String} {cs : Flock.RecVStar.InnerCommits} {m c j : ℕ}
    (hp : Flock.RecOpening.params unit = some (m, c, j)) (h : Flock.RecVStar.checkOpeningShape unit cs = .ok ()) :
    cs.stacked = some { m := m, claims := c } := by
  unfold Flock.RecVStar.checkOpeningShape at h
  rw [hp] at h
  dsimp only at h
  cases hne : (cs.stacked != some { m := m, claims := c })
  · exact optStacked_eq_of_not_bne hne
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hne)] at h
    cases h

/-- **R's stream opens `STACKED_CLAIMS` claims**: `checkStackedShape` passing means META's shape has the two reps'
claims. -/
theorem claims_of_stackedShape {s : String} {sh : Flock.RecVStar.StackedShape} {rs : Array Lean.Json}
    (h : Flock.RecVStar.checkStackedShape s sh rs = .ok ()) : sh.claims = Flock.RecVStar.STACKED_CLAIMS := by
  unfold Flock.RecVStar.checkStackedShape at h
  dsimp only at h
  cases hne : (sh.claims != Flock.RecVStar.STACKED_CLAIMS)
  · simpa using hne
  · rw [ite_eq_left_of_eq_true _ _ (eq_true hne)] at h
    cases h

end FlockC3.VStarList
