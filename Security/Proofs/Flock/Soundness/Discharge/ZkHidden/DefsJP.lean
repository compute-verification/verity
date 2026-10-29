import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Discharge.Composed.DefsJA
import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
import Proofs.Flock.Soundness.Discharge.HiddenExec.RowsAtHP
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

set_option Elab.async false

/-!
# One hidden-output `--zk` session of `J` tables at the verifier's public inputs

`ZkHidden/DefsJ.lean` at the verifier's public inputs `own` (`flock-verify verify --zk --public-inputs F`, which
`Main.buildSessionOf` passes to `Stmt.setupTables`). `HmIn.checkOwn` holds the public file's public input words to `own`
and changes nothing else `Stmt.setupHidden` returns, but at `own = none` it refuses a circuit with public input
ports, so the acceptance is its own event (`AcceptsZKP`, which at `none` is `AcceptsZK`: `acceptsZKP_none`):

* `AcceptsZKP`, `LiveAcceptsZKP`: `ZkExec.AcceptsZK` at `Exec.setupOfP (zkInputs I) own`.
* `setupTables_partsHP`, `setupOf_tableJHP`: `setupTables_partsH`, `setupOf_tableJH` at `own`.
* `DrawSetupZHJP`, `tableZHJP`, `planZHJP`, `tabZHJP`: `DrawSetupZHJ`, … with a `DrawSetupHP` at table `j`'s inputs.
* `tableShHJP`, `planShHJP`, `tabShHJP`, `sitesShHJP`: the public tables (`HiddenExec.TableHP`) and their sites
  (`SitesHP`), at the session's statement circuit `c` (`HmOut.parseStmt`, a templated file's committed circuit
  unrolled).
* `placeZHJP`, `layZHJP`, `rsZHJP`, `rsZHJP_computes`: `keyPlaceHP`, `keyLayoutHP` and `keyRowsHP` (the program
  `Placed.keyProgP`, its outputs `Layout.outsOfP`, the registration the population's public file `HmIn.loadPublic`,
  with or without shared rows), moved to the `--zk` tables.
* `DrawOkZKHJP`, `opnAtZHJP`: `DrawOkZKHJ` and `opnAtZHJ` with `setupHidden … own`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The acceptance -/

/-- **flock-verify `--zk` accepts the transcript at the verifier's public inputs `own`** (`AcceptsZK` with
`Exec.setupOfP`). -/
def AcceptsZKP (I : Inputs) (own : Option ByteArray) (t : Exec.Transcript) : Prop :=
  ∃ spec sts, setupOfP (zkInputs I) own t.record = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
    Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

theorem acceptsZKP_none (I : Inputs) (t : Exec.Transcript) : AcceptsZKP I none t ↔ AcceptsZK I t := Iff.rfl

/-- flock-verify `--zk` accepts the live audit's transcript at `own`. -/
def LiveAcceptsZKP {n : ℕ} {Reg : Type} (I : Inputs) (own : Option ByteArray)
    (o : Reg × Finset (Fin n) × Exec.Transcript) : Prop :=
  AcceptsZKP I own o.2.2

/-! ## flock-verify's tables at `count > 1` at `own` -/

section Parts

open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program own : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- **`setupTables`'s tables at `count > 1` under hidden outputs at `own`** (`setupTables_partsH`). -/
theorem setupTables_partsHP (hh : tags.hiddenOutputs = true) (hc1 : 1 < count)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) :
    tags.manyTables.isSome ∧ count ≤ Tags.MAX_TABLES ∧
      ∃ (population : ℕ) (units : Array ℕ),
        (∀ d, draw = some d → ∃ u, Draw.UnitDraw.ofJson d = .ok u ∧
          ((∃ k', u.law = .subset k') ∨ ∃ K st, u.law = .stratified K st) ∧
          u.population = population ∧ u.units = units) ∧
        (draw = none → units = Array.range population) ∧
        0 < units.size ∧ units.size % count = 0 ∧ sts.size = count ∧
        ∀ (j : ℕ) (hj : j < sts.size),
          Stmt.setupHidden { tags with identity := tags.tableIdentity count j } cf pf coins tables
            (some (Json.mkObj [("law", Json.str "subset"), ("population", Lean.toJson population),
              ("k", Lean.toJson (units.size / count)),
              ("units", Json.arr ((units.extract (j * (units.size / count))
                (j * (units.size / count) + units.size / count)).map Lean.toJson))]))
            partition program own = .ok sts[j] := by
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
  all_goals first
    | (have hdn : draw = none := Tables.draw_none ‹draw.mapM Draw.UnitDraw.ofJson = .ok none›
       exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, fun d hd => by simp [hdn] at hd,
         fun _ => rfl, hpos, hmod, hsz, fun j hj => FlockSoundness.Discharge.Exec.ok_of_refuse (hall j hj)⟩)
    | (have hdu : draw.mapM Draw.UnitDraw.ofJson = .ok (some _) := ‹_›
       obtain ⟨hD1, hD2⟩ := Tables.draw_split hdu
         (by first | exact .inl ⟨_, by assumption⟩ | exact .inr ⟨_, _, by assumption⟩) (by assumption)
       exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, hD1, hD2, hpos, hmod, hsz,
         fun j hj => FlockSoundness.Discharge.Exec.ok_of_refuse (hall j hj)⟩)

end Parts

/-- **`verify --zk --public-inputs` sets table `j` up under its tags at its draw** (`setupOf_tableJH` at `own`). -/
theorem setupOf_tableJHP {I : Inputs} {own : Option ByteArray} {record : String} {d : Lean.Json}
    {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = true) (hd : recordDraw record = some d)
    (hs : setupOfP (zkInputs I) own record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program own = .ok sts[j] := by
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  by_cases hc : I.count ≤ 1
  · rcases setupTables_okHP hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt (zkTags I) I.count 0 = zkTags I := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by change ¬ 1 < I.count; omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := setupTables_partsHP hh' hc1 hs
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

/-- `HmIn.loadPublic` reads no identity, so it agrees at every table's tags. -/
theorem loadPublicIn_tagsAt (tags : Tags) (count j : ℕ) (c : Flock.Circuit) (pi : HmIn.Ports) (b : ByteArray) :
    HmIn.loadPublic (tagsAt tags count j) c pi b = HmIn.loadPublic tags c pi b := by
  unfold tagsAt
  split_ifs <;> rfl

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

/-- **A hidden-output table `verify --zk` sets up at `own`**, on table `j`'s inputs `I'`: `DrawSetupZHJ` at a
`DrawSetupHP`. -/
structure DrawSetupZHJP (I' : Inputs) (own : Option ByteArray) (mPts : ℕ) (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetupHP I' own mPts S d
  sch : Flock.Schedule
  tPad : ℕ
  hz : Zk.schedule x.st.m = .ok (sch, tPad)
  hshape : 2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
    (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
    levelsChained (schedOf sch).levels (schedOf sch).ext = true
  hmsch : (schedOf sch).m = x.st.m
  hpad : Accounting.PadOK (schedOf sch).level₀ tPad
  hN : innerN (layOf x.st sch) < 2 ^ 64
  hL : (layOf x.st sch).len + Zk.K_PAD ≤ innerN (layOf x.st sch)
  hpos : ∀ D : Finset (Fin (innerN (layOf x.st sch))),
    prCoin (fun w => innerPos _ (lt_of_lt_of_le (by simp [Zk.K_PAD]) hL) w ∈ D) ≤
      (D.card : ℝ≥0∞) / innerN (layOf x.st sch)
  hzk : Flock.Zk.shapeOk x.st = .ok ()

namespace DrawSetupZHJP

variable {I' : Inputs} {own : Option ByteArray} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJP I' own mPts S d)

theorem innerN_pos : 0 < innerN (layOf y.x.st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64`. -/
def innerDom : Fin (innerN (layOf y.x.st y.sch)) ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin (innerN (layOf y.x.st y.sch)), j.val + 1 < 2 ^ 64 := fun j => by
      have := y.hN
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- **The `--zk` table of a hidden-output table `verify --zk` sets up at `own`**, at statement regions `regs`
(`DrawSetupZHJ.tabAt`). -/
noncomputable def tabAt (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabZK 𝔽 mPts where
  S := stmtOf y.x.st y.x.wf.1 regs
  sch := schedOf y.sch
  hshape := y.hshape
  hmsch := y.hmsch
  hm := y.x.hm
  t := y.tPad
  hpad := y.hpad
  c₀ := fOf Zk.PAD_C
  e := 2
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    have h := (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).2
    rw [y.x.hmPts] at h
    exact h⟩
  np := (layOf y.x.st y.sch).len
  pads := padsOfLay (layOf y.x.st y.sch)
  padsA := padsAOfLay (layOf y.x.st y.sch)
  tr := tripleOfLay (layOf y.x.st y.sch)
  N := innerN (layOf y.x.st y.sch)
  dom := y.innerDom
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ y.innerN_pos
  hpos := y.hpos
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code y.innerDom ((layOf y.x.st y.sch).len + Zk.K_PAD) : Set (Fin (innerN (layOf y.x.st y.sch)) → 𝔽))
  hL := y.hL
  hδ := le_rfl

/-- The table's `--zk` table: setupHidden's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused table's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

/-- A table's `--zk` table is the same at every `DrawSetupZHJP` of it. -/
theorem table_eq (y₁ y₂ : DrawSetupZHJP I' own mPts S d) : y₁.table = y₂.table := by
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

end DrawSetupZHJP

/-- A `DrawSetupHP` whose statement passes the `--zk` shape check, with `Zk.schedule`'s schedule at its `m`, is a
`DrawSetupZHJP` (`drawSetupZHJ_of`). -/
theorem drawSetupZHJP_of {I' : Inputs} {own : Option ByteArray} {S : Finset (Fin n)} {d : Lean.Json}
    (x : DrawSetupHP I' own mPts S d) (hshape : Flock.Zk.shapeOk x.st = .ok ()) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZHJP I' own mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := zk_schedOK x.st _ _ x.wf.1 x.wf.2 x.hm hshape hz
  exact ⟨⟨x, sch, tPad, hz, h1, h2, h3, h4, h5, h6 _, hshape⟩, rfl, rfl, rfl⟩

section Shadow

variable {I : Inputs} {own : Option ByteArray} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`, at `own`** (`tableZHJ`): the table `verify --zk` sets up at table `j`'s inputs and draw
when it sets one up, and otherwise the reference setup `y₀`'s refused table. -/
noncomputable def tableZHJP (S : Finset (Fin n)) (j : Fin (nTab I)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table
  else y₀.refused

/-- **The hidden-output plan of `J` tables at `own`**. -/
noncomputable def planZHJP : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => List.ofFn (tableZHJP dj y₀ S)

theorem planZHJP_length (S : Finset (Fin n)) (R : Reg) : (planZHJP (Reg := Reg) dj y₀ S R).length = nTab I :=
  List.length_ofFn

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabZHJP : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZHJP (Reg := Reg) dj y₀ S R).length :=
  fun S R u => Fin.cast (planZHJP_length dj y₀ S R).symm (partOf I (dj S) u.val)

theorem tabOfZHJP_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u = tableZHJP dj y₀ S (partOf I (dj S) u.val) := by
  simp only [tabOfZ, planZHJP, tabZHJP, List.get_ofFn]
  rfl

open Classical in
/-- **The public hidden-output table at table `j`'s draw, at `own`**, with the inputs it is set up on. -/
noncomputable def tableShHJP (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (I' : Inputs) × TableHP I' own mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    ⟨inputsJ I j, (Classical.choice h).x.table⟩
  else ⟨inputsJ I 0, y₀.x.refused _⟩

theorem tableShHJP_set {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJP dj y₀ S j = ⟨inputsJ I j, (Classical.choice h).x.table⟩ := by
  unfold tableShHJP
  rw [dite_eq_left_of_eq_true (eq_true h)]

theorem tableShHJP_refused {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : ¬ Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJP dj y₀ S j = ⟨inputsJ I 0, y₀.x.refused _⟩ := by
  unfold tableShHJP
  rw [dite_eq_right_of_eq_false (eq_false h)]

theorem tableZHJP_S (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (tableZHJP dj y₀ S j).S = (tableShHJP dj y₀ S j).2.T.S := by
  by_cases h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJP_set dj y₀ h]
    unfold tableZHJP
    rw [dite_eq_left_of_eq_true (eq_true h)]
    rfl
  · rw [tableShHJP_refused dj y₀ h]
    unfold tableZHJP
    rw [dite_eq_right_of_eq_false (eq_false h)]
    rfl

/-- The public plan at the same draws. -/
noncomputable def planShHJP : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tableShHJP dj y₀ S j).2.T

noncomputable def tabShHJP :
    (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShHJP (Reg := Reg) dj y₀ S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfShHJP_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planShHJP dj y₀) (tabShHJP dj y₀) S R u = (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.T := by
  simp only [Partition.tabOf, planShHJP, tabShHJP, List.get_ofFn]
  rfl

theorem tableShHJP_tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.T.S =
      (tabOfZ (planZHJP (Reg := Reg) dj y₀) (tabZHJP dj y₀) S R u).S := by
  rw [tabOfZHJP_eq, tableZHJP_S dj y₀]

theorem planZHJP_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S =
      (Partition.tabOf (planShHJP dj y₀) (tabShHJP dj y₀) S R u).S :=
  fun S R u => (tableShHJP_tabOfZ dj y₀ S R u).symm.trans (congrArg TabSpec.S (tabOfShHJP_eq dj y₀ S R u)).symm

variable (ht : I.tags.typed = false)

include ht in
theorem tableShHJP_typed (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (tableShHJP dj y₀ S j).1.tags.typed = false := by
  by_cases h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJP_set dj y₀ h]
    exact typedJ ht j
  · rw [tableShHJP_refused dj y₀ h]
    exact typedJ ht 0

variable {c : Flock.Circuit} (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)

include ht hcS in
/-- **One statement circuit per session, at every table** (`tableShHJ_parseStmt`). -/
theorem tableShHJP_parse (S : Finset (Fin n)) (j : Fin (nTab I)) :
    HmOut.parseStmt (tableShHJP dj y₀ S j).1.tags (tableShHJP dj y₀ S j).1.circuitFile
      (tableShHJP dj y₀ S j).1.tables = .ok c := by
  by_cases h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJP_set dj y₀ h]
    exact (parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hcS
  · rw [tableShHJP_refused dj y₀ h]
    exact (parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hcS

/-- A hidden-output acceptance at `own` at `T`, as one at `T' = T`. -/
def recastAccHP {T T' : Model.Statement} (a : AcceptedHP T) (e : T = T') : AcceptedHP T' :=
  { a with hT := a.hT.trans e }

variable (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites at `own`**: each drawn unit's acceptance, VU and block are its table's public table's. -/
noncomputable def sitesShHJP : SitesHP (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) where
  acc S R u _ := recastAccHP ((tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.acc
    (tableShHJP_typed dj y₀ ht S _) (tableShHJP_parse dj y₀ ht hcS S _)
      (rowsFit_of_stmt (zkTyped ht) hcS hscope))
    (congrArg TabSpec.S (tabOfShHJP_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJP_eq dj y₀ S R u)).symm
    ((tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One statement circuit per session**: every site's committed circuit unrolls to the session's `c`. -/
theorem sitesShHJP_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesShHJP dj y₀ ht hcS hscope).acc S R u h).st.c = c :=
  fun S _ u _ => (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.unrolled_eq (tableShHJP_typed dj y₀ ht S _)
    (tableShHJP_parse dj y₀ ht hcS S _)

variable (hU : Built c.unit) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pop : Public}
  (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pop)

omit ht in
theorem bcSiteHP_recast {T T' : Model.Statement} (a : AcceptedHP T) (e : T = T') (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : BcSiteHP a pop o g u) :
    BcSiteHP (recastAccHP a e) pop (Fin.cast h' o) g u :=
  hs

include hU hpi hpub in
/-- `BcSiteHP` at a table set up at table `j`'s inputs at `own` (`tableHP_bcSite` at `inputsJ I j`). -/
theorem tableShHJP_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableHP I' own mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parseStmt Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupHP (inputsJ I j) own mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteHP (Tb.2.acc p₁ p₂ (rowsFit_of_stmt p₁ p₂ hscope)) pop (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableHP_bcSite p₁ p₂ hscope (fun _ => d) hU hpi ((loadPublicIn_tagsAt _ _ _ _ _ _).trans hpub) x h

include hU hpi hpub in
/-- **`BcSiteHP` at every satisfiable public site** at `J` tables at `own` (`sitesShHJ_bcSite`). -/
theorem sitesShHJP_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJP dj y₀) (tabShHJP dj y₀) S R u),
      Binding.Sat (planShHJP dj y₀) (tabShHJP dj y₀) S R u M →
        BcSiteHP ((sitesShHJP dj y₀ ht hcS hscope).acc S R u h) pop ((sitesShHJP dj y₀ ht hcS hscope).o S R u h)
          ((sitesShHJP dj y₀ ht hcS hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteHP_recast _ (congrArg TabSpec.S (tabOfShHJP_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJP_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJP (inputsJ I (partOf I (dj S) u.val)) own mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJP_bcSite hscope hU hpi hpub _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJP_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJP_eq, tableShHJP_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan at `own`**: `keyLayoutHP` at `sitesShHJP`. -/
noncomputable abbrev layShHJP :
    Binding.Layout ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planShHJP (Reg := Reg) dj y₀)
      (tabShHJP dj y₀)
      (((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)).placeDecoder (planShHJP dj y₀) (tabShHJP dj y₀)
        (keyPlaceHP (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi))) :=
  keyLayoutHP (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi) (fun _ => pop)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt (zkTyped ht) hcS hscope) (Layout.outsOutP_outsOfP c hU n pi)

include ht hcS hscope in
/-- **`KeysRowP` from `scopeOkH`** at the session's statement circuit (`Hidden.keysRow_stmt`). -/
theorem keysRowP_of_scopeZ : Layout.KeysRowP c hU pi (Layout.portRowW c) :=
  Layout.keysRowP_of_keysRow (Hidden.keysRow_stmt (zkTyped ht) hcS (Layout.scopeOkH_spec hU hscope).1
    (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`rs` at the public plan at `own`**: `keyRowsHP` at `sitesShHJP`, with `hsite` from `sitesShHJP_bcSite`. -/
noncomputable abbrev rsShHJP : Binding.HmRowsX (layShHJP dj y₀ ht hcS hscope hU hpi (pop := pop) (Reg := Reg)) :=
  keyRowsHP (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi) (fun _ => pop)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt (zkTyped ht) hcS hscope) (Layout.outsOutP_outsOfP c hU n pi)
    (sitesShHJP_bcSite dj y₀ ht hcS hscope hU hpi hpub)
    (keysRowP_of_scopeZ ht hcS hscope hU)

theorem rsShHJP_computes : (rsShHJP dj y₀ ht hcS hscope hU hpi hpub (Reg := Reg)).Computes :=
  keyRowsHP_computes (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi) (fun _ => pop)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_stmt (zkTyped ht) hcS hscope) (Layout.outsOutP_outsOfP c hU n pi)
    (sitesShHJP_bcSite dj y₀ ht hcS hscope hU hpi hpub)
    (keysRowP_of_scopeZ ht hcS hscope hU)

/-- **Each drawn unit's place in its hidden-output `--zk` table at `own`**: `keyPlaceHP`'s place in the public table,
along `planZHJP_S`. -/
noncomputable def placeZHJP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi))
      (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S u :=
  placeCast (planZHJP_S dj y₀ S R u)
    (keyPlaceHP (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHJP :
    DecAgree (P := (keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZHJP dj y₀ ht hcS hscope hU hpi))
      (decS := ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)).placeDecoder (planShHJP dj y₀)
        (tabShHJP dj y₀) (keyPlaceHP (sitesShHJP_parse dj y₀ ht hcS hscope) hpi (Layout.outsOfP c hU n pi))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJP, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables at `own`**: `layShHJP` moved along `planZHJP_S`. -/
noncomputable abbrev layZHJP :
    LayoutZ execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP (Reg := Reg) dj y₀)
      (tabZHJP dj y₀)
      (decodeOfPlace execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZHJP dj y₀ ht hcS hscope hU hpi)) :=
  LayoutZ.ofShadow (planZHJP_S dj y₀) (layShHJP dj y₀ ht hcS hscope hU hpi (pop := pop))
    (decAgreeHJP dj y₀ ht hcS hscope hU hpi)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables at `own`**: `rsShHJP` moved along `planZHJP_S`. -/
noncomputable abbrev rsZHJP : HmRowsZX (layZHJP dj y₀ ht hcS hscope hU hpi (pop := pop) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planZHJP_S dj y₀) (decAgreeHJP dj y₀ ht hcS hscope hU hpi)
    (rsShHJP dj y₀ ht hcS hscope hU hpi hpub)

/-- **`hHm` at the hidden-output `--zk` tables' rows at `own`** (`rsShHJP_computes`, moved along). -/
theorem rsZHJP_computes : (rsZHJP dj y₀ ht hcS hscope hU hpi hpub (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planZHJP_S dj y₀) (decAgreeHJP dj y₀ ht hcS hscope hU hpi) _
    (rsShHJP_computes dj y₀ ht hcS hscope hU hpi hpub)

end Shadow

/-! ## The coin server's draws, and the openings, at `own` -/

/-- **The coin server's draw at `S`, at `J` hidden-output tables at `own`** (`DrawOkZKHJ` with `setupHidden … own`). -/
def DrawOkZKHJP (I : Inputs) (own : Option ByteArray) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ)
    (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j)) I.partition I.program own = .ok st → st.mPts = mPts

/-- **The `--zk` openings at hidden-output table `j` at `own`** (`opnAtZHJ` with `setupHidden … own`). -/
noncomputable def opnAtZHJP (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
    (own : Option ByteArray) (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg)
    (proofs : Array ByteArray) (j : Fin (planZ S R).length) : OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (tagsAt (zkTags I) I.count j.val) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program own with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZJ ((planZ S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | .error _ => default

end FlockSoundness.Discharge.ZkHidden
