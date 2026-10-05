import FlockSoundness.Discharge.Composed.DefsJ
import FlockSoundness.Discharge.ZkHidden.SoundBind
import FlockSoundness.Discharge.ZkHidden.SoundRuns
import FlockSoundness.Protocol.Discharge.ZkHidden.DefsJ

set_option Elab.async false

/-!
# One hidden-output `--zk` sampled session of `J` tables: the pieces `zk_session_soundHJ` reads

`Composed/DefsJ.lean`'s pieces at `J` tables, under hidden outputs (`Stmt.setupHidden`, the default `--statement`), each
the one-table hidden-output piece (`ZkHidden/Defs.lean`, `SoundBind.lean`, `SoundRuns.lean`) at table `j`:

* `setupTables_partsH`: `Tables.setupTables_parts` under hidden outputs. At `count > 1`, table `j` is `setupHidden`'s
  statement under `Tags.tableIdentity count j` at the subset draw of part `j`; `setupOf_tableJH` is `setupOf_tableJ`'s
  counterpart.
* `parseH_tagsAt`, `loadPublicH_tagsAt`: `HmOut.parse` (untyped) and `HmOut.loadPublic` read no identity.
* `DrawSetupZHJ`, `tableZHJ`, `planZHJ`, `tabZHJ`: `DrawSetupZJ`, `tableZJ`, `planZJ`, `tabZJ` with a `DrawSetupH` at
  table `j`'s inputs (`inputsJ I j`), and the refused table of the reference setup `y₀ : DrawSetupZHJ` at table 0's
  inputs.
* `tableShHJ`, `planShHJ`, `tabShHJ`, `sitesShHJ`: the public hidden-output tables (`HiddenExec.TableH`) at the same
  draws, each at the inputs it is set up on (table `j`'s, or table 0's, `y₀`'s, at a refused table), and their sites.
* `placeZHJ`, `layZHJ`, `rsZHJ`, `rsZHJ_computes`: `keyPlaceH`, `keyLayoutH` and `keyRowsH` at those sites, moved to the
  `--zk` tables (`SoundBind.lean`'s `placeZH`, `layZH`, `rsZH` at `J` tables).
* `TagsOkZHJ`, `DrawOkZKHJ`, `opnAtZHJ`, `runsZCJH_live`: `TagsOkZJ` with hidden outputs, `DrawOkZKJ` and `opnAtZJ` with
  `setupHidden`, and `runsZCJ_live` under hidden outputs.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## flock-verify's tables at `count > 1` under hidden outputs -/

section Parts

open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- **`setupTables`'s tables at `count > 1` under hidden outputs** (`Tables.setupTables_parts` with `hh = true`): the
same split, and table `j` is `setupHidden`'s statement under `Tags.tableIdentity count j` at part `j`'s subset draw. -/
theorem setupTables_partsH (hh : tags.hiddenOutputs = true) (hc1 : 1 < count)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure =
      .ok (spec, sts)) :
    tags.manyTables.isSome ∧ count ≤ Tags.MAX_TABLES ∧
      ∃ (population : ℕ) (units : Array ℕ),
        (∀ d, draw = some d → ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧ (∃ k', u.law = .subset k') ∧
          u.population = population ∧ u.units = units) ∧
        (draw = none → units = Array.range population) ∧
        0 < units.size ∧ units.size % count = 0 ∧ sts.size = count ∧
        ∀ (j : ℕ) (hj : j < sts.size),
          Stmt.setupHidden { tags with identity := tags.tableIdentity count j } cf pf coins tables
            (some (Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population),
              ("k", Lean.toJson (units.size / count)),
              ("units", Json.arr ((units.extract (j * (units.size / count))
                (j * (units.size / count) + units.size / count)).map Lean.toJson))]))
            partition program = .ok sts[j] := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  simp only [throw_bind', pure_bind, ite_eq_left hh] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
  all_goals try omega
  all_goals try contradiction
  all_goals try exact (Tables.no_some (by assumption)).elim
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    obtain ⟨hpos, hmod⟩ := Tables.splits_ok (by assumption)
  · obtain ⟨hD1, hD2⟩ := Tables.draw_subset (by assumption) (by assumption) (by assumption)
    exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, hD1, hD2, hpos, hmod, hsz, hall⟩
  · have hdn := Tables.draw_none (by assumption)
    exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, fun d hd => by simp [hdn] at hd,
      fun _ => rfl, hpos, hmod, hsz, hall⟩

end Parts

/-- **`verify --zk` sets table `j` up under its tags at its draw, under hidden outputs** (`setupOf_tableJ` with
`setupHidden`): at the record's draw `d`, `setupTables` at `zkInputs I` has `nTab I` tables, table `j` setupHidden's
statement under `tagsAt` at `drawJ` (`setupTables_okH` at one table, `setupTables_partsH` at several). -/
theorem setupOf_tableJH {I : Inputs} {record : String} {d : Lean.Json} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = true) (hd : recordDraw record = some d)
    (hs : setupOf (zkInputs I) record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  by_cases hc : I.count ≤ 1
  · rcases setupTables_okH hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt (zkTags I) I.count 0 = zkTags I := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by change ¬ 1 < I.count; omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := setupTables_partsH hh' hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partDraw population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : tagsAt (zkTags I) I.count j = { zkTags I with identity := (zkTags I).tableIdentity I.count j } := by
      simp [tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

/-! ## Table `j`'s tags under hidden outputs -/

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

namespace DrawSetupZHJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJ I' mPts S d)

/-- A table's `--zk` table is the same at every `DrawSetupZHJ` of it. -/
theorem table_eq (y₁ y₂ : DrawSetupZHJ I' mPts S d) : y₁.table = y₂.table := by
  obtain ⟨⟨st₁, h₁, D₁, hD₁, hS₁, hg₁, hb₁, m₁, k₁, a₁, b₁, r₁, e₁, hs₁, hm₁, hp₁, w₁⟩, sch₁, t₁, hz₁, _, _, _, _, _, _⟩ :=
    y₁
  obtain ⟨⟨st₂, h₂, D₂, hD₂, hS₂, hg₂, hb₂, m₂, k₂, a₂, b₂, r₂, e₂, hs₂, hm₂, hp₂, w₂⟩, sch₂, t₂, hz₂, _, _, _, _, _, _⟩ :=
    y₂
  obtain rfl : st₁ = st₂ := Except.ok.inj (h₁.symm.trans h₂)
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz₁.symm.trans hz₂))
  rfl

theorem table_seam : ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem table_lay : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end DrawSetupZHJ

/-- A `DrawSetupH` whose statement passes the `--zk` shape check, with `Zk.schedule`'s schedule at its `m`, is a
`DrawSetupZHJ` (`ZkSchedOK`). -/
theorem drawSetupZHJ_of {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupH I' mPts S d)
    (hshape : Flock.Zk.shapeOk x.st = .ok ()) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZHJ I' mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := zk_schedOK x.st _ _ x.wf.1 x.wf.2 x.hm hshape hz
  exact ⟨⟨x, sch, tPad, hz, h1, h2, h3, h4, h5, h6 _, hshape⟩, rfl, rfl, rfl⟩

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)

theorem planZHJ_length (S : Finset (Fin n)) (R : Reg) : (planZHJ (Reg := Reg) dj y₀ S R).length = nTab I :=
  List.length_ofFn

/-! ## The public shadow, the sites, the layout and the rows -/

variable (ht : I.tags.typed = false)

variable {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)

variable (hscope : Layout.scopeOk c = true)

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

theorem rsShHJ_computes : (rsShHJ dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

/-- **`hHm` at the hidden-output `--zk` tables' rows, at `J` tables** (`rsShHJ_computes`, moved along). -/
theorem rsZHJ_computes : (rsZHJ dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) _
    (rsShHJ_computes dj y₀ ht hc hscope hU hpub hpt)

end Shadow

/-! ## The tags, the coin server's draws, and the openings -/

/-- **The verify command's tags at any number of tables, under hidden outputs** (`TagsOkZJ` with hidden outputs):
hidden outputs, retained rounds, `hm96-sha512/v1` leaves. -/
def TagsOkZHJ (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1"

/-! ## The message half at the live hidden-output tables -/

/-- **`RunsZC` at the live hidden-output `--zk` tables of a session of `J` tables, at the law's draws**
(`runsZCJ_live` under hidden outputs): on untyped inputs under the tags `TagsOkZHJ` fixes, whose circuit file
`HmOut.parse` accepts as a circuit whose unit returns the outputs, with `wr` the coin server's record of the live
session and each of the law's draws setting each table up at `mPts` link points, `RunsOnZCJ` holds at the tables
`planZHJ` under `hm96-sha512/v1`, with the decoder `decAtZJ`, the openings `opnAtZHJ`, no tail and table `j`'s cap
from `root_B[j]`. -/
theorem runsZCJH_live (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, hshape, hv⟩ := hacc
  have hpts2 := setupTables_pointsAny hs
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hh hd hs
  have hjn : j.val < nTab I := planZHJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, cap, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨sch₀, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := drawSetupH_of_setupHidden (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hcJ ho hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZHJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZHJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZHJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZHJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st y.x.wf.1 y.x.wf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupHidden_hm96 (((tagsAt_merkleLeaf _ _ _).trans (zkTags_merkleLeaf I)).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ cap y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZHJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZHJ (planZHJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZHJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    exact outZC_of_accZ hacc'

end FlockSoundness.Discharge.ZkHidden
