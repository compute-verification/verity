import Proofs.Flock.Recursive.AuditC3
import Proofs.Flock.C3.VStarList.Default
import Proofs.Flock.C3.VStarList.Parse

/-!
# Piece (h): V*'s statements over the verifier's list (gap 4(c)), and R's shape

`vstarParses_of_list`: at the inner shape `ish`, a family of V*'s statements indexed by the verifier's own list
(`RecVStar.statements`, `vstarCount`) whose files parse (`VStarParses`, the guarantee's `_hob`), at any outer law. It
discharges `_hob`. At candidate 3's inner shape the list is circuits' gap 2's eleven statements, in the writer's order:
four `rec-open`, two `rec-residuals`, one `rec-sparse`, R's three levels (`rec-open-stacked`) and one `rec-opening`.
Statement `j`'s circuit is the parse of V*'s file for it where the verifier lists it and the file is written
(`circuitFile_parses`, `Parse.lean`), and `defaultCircuit` (`Default.lean`) elsewhere.

R's shape is derived, not a hypothesis: each `rec-opening/m<m>-c<c>-p<j>` unit the verifier checks opens R at META's
`stacked` entry (`stacked_of_openingShape`, `checkOpeningShape`), and that entry opens exactly `STACKED_CLAIMS` claims
(`claims_of_stackedShape`, `checkStackedShape`), whose rounds are `fast100`'s at its `m`. The link to the bucket's
`table_m` is transitive, through the outer program's `rec-opening` units, whose names the verifier compiles; the list's
own `rec-opening` statements are at `mReg` (`mReg_of_list`, `Shape.lean`).
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
  have key : ∀ j : Fin (vstarCount ish), ∃ q : Flock.Circuit × ℕ, FlockSoundness.Built q.1.unit ∧
      FlockSoundness.Discharge.Layout.scopeOk q.1 = true ∧
      ∀ ss s cf, Flock.RecVStar.statements ish = .ok ss → ss[(j : ℕ)]? = some s →
        Flock.RecVStar.circuitFile s (bind j) = .ok cf →
        Flock.HmRow.parse (Flock.Zk.tags (Flock.Tags.circuit.withCoins
          (.tree (Flock.Zk.coinSpec #[Flock.Tags.circuit.table]))) (Flock.Zk.maskLogOf cf) 1) cf [] = .ok q.1 ∧
          q.2 = s.instances := by
    intro j
    by_cases hw : ∃ ss s cf, Flock.RecVStar.statements ish = .ok ss ∧ ss[(j : ℕ)]? = some s ∧
        Flock.RecVStar.circuitFile s (bind j) = .ok cf
    · obtain ⟨ss, s, cf, hss, hs, hcf⟩ := hw
      obtain ⟨c, hc, hsc⟩ := FlockC3.VStarList.circuitFile_parses hss (Array.mem_of_getElem? hs) hcf
      refine ⟨(c, s.instances), FlockC3.VStarList.built_of_parse hc, hsc, fun ss' s' cf' hss' hs' hcf' => ?_⟩
      rw [hss] at hss'
      cases hss'
      rw [hs] at hs'
      cases hs'
      rw [hcf] at hcf'
      cases hcf'
      exact ⟨hc, rfl⟩
    · exact ⟨(FlockC3.VStarList.defaultCircuit, 0), FlockC3.VStarList.defaultCircuit_built,
        FlockC3.VStarList.defaultCircuit_scope, fun ss s cf hss hs hcf => absurd ⟨ss, s, cf, hss, hs, hcf⟩ hw⟩
  refine ⟨fun j => ⟨(Classical.choose (key j)).2, mPts, J, (Classical.choose (key j)).1,
      (Classical.choose_spec (key j)).1, (Classical.choose_spec (key j)).2.1, law _⟩,
    fun j ss s cf hss hs hcf => (Classical.choose_spec (key j)).2.2 ss s cf hss hs hcf, fun j => ⟨rfl, rfl, rfl⟩⟩

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
