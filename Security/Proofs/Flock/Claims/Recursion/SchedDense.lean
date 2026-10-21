import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Claims.Recursion.InnerAccepts
import Proofs.Flock.C3.VStarList.Shape

/-!
# `SchedMatches` at the dense form, and at the sparse form vacuously

`SchedMatches` (`DeployLayout.lean`) asks that the deployed plain verifier take `2 + 2R` rounds at the inner statement,
`R` the rounds of the verifier's own inner shape's rep schedule (`repSched`).

* **Sparse.** The verifier's list refuses a RecSparse shape at public circuits (`RecVStar.checkSparseHolo`, through
  `checkCandidate3`: `FlockC3.VStarList.holo_of_list`), so no statement the reader reads is at the sparse form
  (`not_parse_sparse`), and `SchedMatches` holds there vacuously (`schedMatches_sparse`).
* **Dense.** The verifier's setup holds its inner shape's `m`, `k_log`, regions and claims to the inner statement's as
  its files set it up (`RecSetup.checkShape`, in `RecScope.accepts`: `checkShape_files`). Where Flock's plain verifier
  sets the inner statement up as one table at those numbers (`InnerSetupIsFiles`, a named premise), its rounds are
  `2 + 2R` at the plain schedule (`k_plain`), so `SchedMatches` holds at the dense form (`schedMatches_dense`).

`InnerSetupIsFiles` is not a property of the code: the compiled verifier reads the inner statement's files and record,
never the tags, table count or draw the plain verifier sets the inner statement up at, so nothing it checks rules out
an inner statement at two tables, or whose drawn table is smaller than the files' one.
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
  obtain ⟨⟨⟨ss, hss⟩, -⟩, -⟩ := ha
  have h1 : r.shape.holo = true := FlockC3.VStarList.holo_of_list (i := r.shape) hss hs
  exact Bool.false_ne_true (hh.symm.trans h1)

/-- **`SchedMatches` at the sparse form**, vacuously: no statement the reader reads is at it (`not_parse_sparse`). -/
theorem schedMatches_sparse : SchedMatches Claims.Recursion.Form.Sparse :=
  fun _ _ hst hf => absurd hf (not_parse_sparse hst)

/-- **The setup's shape check**: where `RecSetup.checkShape` passes on the inner statement's files, the inner shape's
`m`, `k_log`, regions and claims are the ones the files set the inner statement up at. -/
theorem checkShape_files {i : RecVStar.Inner} {f : RecCoins.Files} {u : Unit} {spec : SessionSpec} {st : Stmt}
    (h : RecSetup.checkShape (some i) (some (.files f)) = .ok u) (hf : f.stmt "the inner session" = .ok (spec, st)) :
    i.m = st.m ∧ i.kLog = st.c.kLog ∧ i.regions = st.regions.size ∧
      i.claims = 2 + spec.points * st.regions.size := by
  unfold RecSetup.checkShape at h
  simp only [hf, bind, Except.bind] at h
  by_cases hc : (i.m != st.m || i.kLog != st.c.kLog || i.regions != st.regions.size ||
      i.claims != 2 + spec.points * st.regions.size) = true
  · simp [hc, throw, throwThe, MonadExceptOf.throw] at h
  · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
    exact ⟨hc.1.1.1, hc.1.1.2, hc.1.2, hc.2⟩

/-- **The inner shape the verifier's list accepts has a rep schedule**: `RecVStar.checkTopsListed` refuses one without. -/
theorem repSched_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ rs, repSched i = some rs := by
  obtain ⟨_, _, _, _, _, -, -, -, ht, -, -, -⟩ := FlockC3.VStarList.statements_ok h
  unfold RecVStar.checkTopsListed at ht
  unfold repSched
  cases hrs : (if i.holo then RecVStar.holoRepRounds else RecVStar.repRounds) i.m i.kLog i.claims with
  | none => rw [hrs] at ht; cases ht
  | some rs => exact ⟨rs, rfl⟩

/-- **Flock's plain verifier sets the inner statement up as its files do** (a named premise): at every compiled
statement the reader reads at public circuits, the verifier holds the inner statement's files (`filesOf`), and Flock's
plain verifier (`FlockDeploy.setupOf false`) sets the inner statement up as one table at the `m`, `k_log` and region
count, and with the points, that the files set it up at (`RecCoins.Files.stmt`, which `RecSetup.checkShape` holds the
inner shape to). -/
def InnerSetupIsFiles : Prop :=
  ∀ x r, readX x = some r → r.shape.holo = false →
    ∃ f spec' st' s spec st, filesOf r = some (.files f) ∧ f.stmt "the inner session" = .ok (spec', st') ∧
      FlockDeploy.setupOf false r.w.inner = some (s, spec, #[st]) ∧
      st.m = st'.m ∧ st.c.kLog = st'.c.kLog ∧ st.regions.size = st'.regions.size ∧ spec.points = spec'.points

/-- **`SchedMatches` at the dense form**, under `InnerSetupIsFiles`: the setup holds the inner shape to the files'
numbers (`checkShape_files`), the plain verifier sets the inner statement up as one table at them, and its rounds are
`2 + 2R` there (`k_plain`, at `k_log ≥ 6`, which V*'s list needs: `RecScope.statements_ok`). -/
theorem schedMatches_dense (hI : InnerSetupIsFiles) : SchedMatches Claims.Recursion.Form.Dense := by
  intro x st hst hf
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpts⟩ := hI x r hr hh
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨e1, e2, -, e4⟩ := checkShape_files hsh hstm
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
