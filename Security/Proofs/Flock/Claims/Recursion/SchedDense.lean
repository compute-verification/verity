import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Claims.Recursion.InnerAccepts
import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.C3.VStarList.Shape

/-!
# `SchedMatches` at the dense form, and at the sparse form vacuously

`SchedMatches` (`DeployLayout.lean`) asks that the deployed plain verifier take `2 + 2R` rounds at the inner statement,
`R` the rounds of the verifier's own inner shape's rep schedule (`repSched`).

* **Sparse.** The verifier's list refuses a RecSparse shape at public circuits (`RecVStar.checkSparseHolo`, through
  `checkCandidate3`: `FlockC3.VStarList.holo_of_list`), so no statement the reader reads is at the sparse form
  (`not_parse_sparse`), and `SchedMatches` holds there vacuously (`schedMatches_sparse`).
* **Dense.** The verifier's setup holds its inner shape's `m`, `k_log`, regions and claims to the inner statement's as
  its files set it up, and the inner statement to the files' setup (`RecSetup.checkShape`, `RecSetup.checkForm`, in
  `RecScope.accepts`: `checkShape_files`), so Flock's plain verifier sets the inner statement up as one table at those
  numbers (`innerSetup_files`), its rounds are `2 + 2R` at the plain schedule (`k_plain`), and `SchedMatches` holds at
  the dense form (`schedMatches_dense`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **What `parse` read**: a statement `parse` accepts is `readX`'s reading, which the setup's checks accept
(`RecScope.accepts`), made readable (`RecScope.toStmt`). -/
theorem parse_read {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ∃ r, readX x = some r ∧ RecScope.accepts (readOf r) = true ∧ st = RecScope.toStmt (readOf r) := by
  obtain ⟨r', hr', rfl⟩ := Option.map_eq_some_iff.1 hst
  obtain ⟨hd, ha⟩ := Option.filter_eq_some_iff.1 hr'
  obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 hd
  exact ⟨r, hr, ha, rfl⟩

/-- **No statement the reader reads is at the sparse form**: the verifier's list refuses a RecSparse shape at public
circuits (`holo_of_list`). -/
theorem not_parse_sparse {x : Msg} {st : Claims.Recursion.Stmt} (hst : parse x = some st) :
    ¬ Claims.Recursion.Form.Sparse st.form := by
  obtain ⟨r, -, ha, rfl⟩ := parse_read hst
  rintro ⟨hh, hs⟩
  change r.shape.holo = false at hh
  change (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)).isSome = true at hs
  rw [Option.isSome_map] at hs
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, -⟩, -⟩ := ha
  have h1 : r.shape.holo = true := FlockC3.VStarList.holo_of_list (i := r.shape) hss hs
  exact Bool.false_ne_true (hh.symm.trans h1)

/-- **`SchedMatches` at the sparse form**, vacuously: no statement the reader reads is at it (`not_parse_sparse`). -/
theorem schedMatches_sparse : SchedMatches Claims.Recursion.Form.Sparse :=
  fun _ _ hst hf => absurd hf (not_parse_sparse hst)

/-- **The inner shape the verifier's list accepts has a rep schedule**: `RecVStar.checkTopsListed` refuses one without. -/
theorem repSched_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ rs, repSched i = some rs := by
  obtain ⟨_, _, _, _, _, -, -, -, ht, -, -, -⟩ := FlockC3.VStarList.statements_ok h
  unfold RecVStar.checkTopsListed at ht
  unfold repSched
  cases hrs : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims with
  | none => rw [hrs] at ht; cases ht
  | some rs => exact ⟨rs, rfl⟩

/-- **`SchedMatches` at the dense form**: Flock's plain verifier sets the inner statement up as one table at the
files' numbers (`innerSetup_files`), the setup holds the inner shape to them (`checkShape_files`), and the plain
verifier's rounds are `2 + 2R` there (`k_plain`, at `k_log ≥ 6`, which V*'s list needs: `RecScope.statements_ok`). -/
theorem schedMatches_dense : SchedMatches Claims.Recursion.Form.Dense := by
  intro x st hst hf
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpts, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨-, -, hk6, -, -⟩ := RecScope.statements_ok hss
  obtain ⟨rs, hrs⟩ := repSched_of_statements hss
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  refine ⟨r, rs, hr, hrs, ?_⟩
  rw [hx]
  exact k_plain hset (e1.trans hm.symm) (e2.trans hk.symm) (by rw [e4, hpts, hreg]) (by omega) hrs'

end RecDeploy
