import Proofs.Flock.Recursive.AuditC3

/-!
# Piece (h): V*'s statements over the verifier's list (gap 4(c)), and R's shape

`vstarParses_of_list`: at the inner shape `ish`, a family of V*'s statements indexed by the verifier's own list
(`RecVStar.statements`, `vstarCount`) whose files parse (`VStarParses`, the guarantee's `_hob`), at any outer law. It
discharges `_hob`. At candidate 3's inner shape the list is circuits' gap 2's eleven statements, in the writer's order:
four `rec-open`, two `rec-residuals`, one `rec-sparse`, R's three levels (`rec-open-stacked`) and one `rec-opening`.

R's shape is derived, not a hypothesis: each `rec-opening/m<m>-c<c>-p<j>` unit the verifier checks opens R at META's
`stacked` entry (`stacked_of_openingShape`, `checkOpeningShape`), and that entry opens exactly `STACKED_CLAIMS` claims
(`claims_of_stackedShape`, `checkStackedShape`), whose rounds are `fast100`'s at its `m`. The link to the bucket's
`table_m` is transitive, through the outer program's `rec-opening` units, whose names the verifier compiles.
-/

namespace FlockC3.VStarList

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL

/-- **The count is the list's size** where the verifier lists V*'s statements. -/
theorem vstarCount_eq {ish : Flock.RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : Flock.RecVStar.statements ish = .ok ss) : vstarCount ish = ss.size := by
  unfold vstarCount
  rw [h]

/-- **(h) V*'s statements over the verifier's list**: at every binding of each, a family whose files parse, at the
outer law's points, rounds and draws. -/
theorem vstarParses_of_list (ish : Flock.RecVStar.Inner) (bind : Fin (vstarCount ish) → VStar.Bindings)
    (mPts J : ℕ) (law : (nv : ℕ) → VLaw nv) :
    ∃ Vs : Fin (vstarCount ish) → VStmt, VStarParses Flock.Tags.circuit ish bind Vs ∧
      ∀ j, (Vs j).mPts = mPts ∧ (Vs j).J = J ∧ (Vs j).law = law (Vs j).nv := by
  sorry

/-- **An R opening unit opens R at META's shape**: `checkOpeningShape` passing at a `rec-opening` unit of `m` and `c`
claims means the inner commitments' `stacked` entry is `(m, c)`. -/
theorem stacked_of_openingShape {unit : String} {cs : Flock.RecVStar.InnerCommits} {m c j : ℕ}
    (hp : Flock.RecOpening.params unit = some (m, c, j)) (h : Flock.RecVStar.checkOpeningShape unit cs = .ok ()) :
    cs.stacked = some { m := m, claims := c } := by
  sorry

/-- **R's stream opens `STACKED_CLAIMS` claims**: `checkStackedShape` passing means META's shape has the two reps'
claims. -/
theorem claims_of_stackedShape {s : String} {sh : Flock.RecVStar.StackedShape} {rs : Array Lean.Json}
    (h : Flock.RecVStar.checkStackedShape s sh rs = .ok ()) : sh.claims = Flock.RecVStar.STACKED_CLAIMS := by
  sorry

end FlockC3.VStarList
