import Proofs.Flock.Soundness.Discharge.Composed.DefsJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundRuns

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
  `setupHidden`, and `runsZCJ_live` under hidden outputs. It holds from the sessions alone (`runsZCJH_liveU`), so
  `runsZCJH_liveSt` and `runsZCJH_liveISt` state it at the statement circuit (`HmOut.parseStmt`).
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
  all_goals first
    | (have hdn : draw = none := Tables.draw_none ‹draw.mapM Draw.UnitDraw.ofJson = .ok none›
       exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, fun d hd => by simp [hdn] at hd,
         fun _ => rfl, hpos, hmod, hsz, fun j hj => ok_of_refuse (hall j hj)⟩)
    | (have hdu : draw.mapM Draw.UnitDraw.ofJson = .ok (some _) := ‹_›
       obtain ⟨hD1, hD2⟩ := Tables.draw_split hdu
         (by first | exact .inl ⟨_, by assumption⟩ | exact .inr ⟨_, _, by assumption⟩) (by assumption)
       exact ⟨Option.isSome_iff_exists.2 ⟨_, by assumption⟩, by omega, _, _, hD1, hD2, hpos, hmod, hsz,
         fun j hj => ok_of_refuse (hall j hj)⟩)

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

theorem parseH_identity {tags : Tags} (ht : tags.typed = false) (x : Lean.Json) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse { tags with identity := x } b t = HmOut.parse tags b t := by
  simp only [HmOut.parse, parse_identity, Tags.isTmpl]
  simp only [ht, Bool.not_false, ite_true]

/-- An untyped `HmOut.parse` reads no identity, so it agrees at every table's tags. -/
theorem parseH_tagsAt {tags : Tags} (ht : tags.typed = false) (count j : ℕ) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse (tagsAt tags count j) b t = HmOut.parse tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · exact parseH_identity ht _ _ _

/-- `HmOut.loadPublic` reads no identity, so it agrees at every table's tags. -/
theorem loadPublicH_tagsAt (tags : Tags) (count j : ℕ) (c : Flock.Circuit) (b : ByteArray) :
    HmOut.loadPublic (tagsAt tags count j) c b = HmOut.loadPublic tags c b := by
  unfold tagsAt
  split_ifs <;> rfl

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

/-- **A hidden-output table `verify --zk` sets up**, on table `j`'s inputs `I'` (`inputsJ I j`): `DrawSetupZKH`'s
fields at a `DrawSetupH` of `I'`, and the statement passes the `--zk` shape check (`Zk.shapeOk`). -/
structure DrawSetupZHJ (I' : Inputs) (mPts : ℕ) (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetupH I' mPts S d
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

namespace DrawSetupZHJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJ I' mPts S d)

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

/-- **The `--zk` table of a hidden-output table `verify --zk` sets up**, at statement regions `regs`
(`DrawSetupZKH.tabAt`). -/
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

open Classical in
/-- **Table `j` at draw `S`, under hidden outputs** (`tableZJ` with `setupHidden`): the table `verify --zk` sets up at
table `j`'s inputs and draw when it sets one up, and otherwise the reference setup `y₀`'s refused table (a setup at
table 0's inputs, which every session `verify --zk` accepts at a draw `DrawOkZKHJ` admits has: `setupZHJ_of_acceptsZK`). -/
noncomputable def tableZHJ (S : Finset (Fin n)) (j : Fin (nTab I)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table
  else y₀.refused

/-- **The hidden-output plan of `J` tables**: at draw `S`, table `j` is `tableZHJ … S j`, for every registration. -/
noncomputable def planZHJ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => List.ofFn (tableZHJ dj y₀ S)

theorem planZHJ_length (S : Finset (Fin n)) (R : Reg) : (planZHJ (Reg := Reg) dj y₀ S R).length = nTab I :=
  List.length_ofFn

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabZHJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZHJ (Reg := Reg) dj y₀ S R).length :=
  fun S R u => Fin.cast (planZHJ_length dj y₀ S R).symm (partOf I (dj S) u.val)

theorem tabOfZHJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u = tableZHJ dj y₀ S (partOf I (dj S) u.val) := by
  simp only [tabOfZ, planZHJ, tabZHJ, List.get_ofFn]
  rfl

/-! ## The public shadow, the sites, the layout and the rows -/

open Classical in
/-- **The public hidden-output table at table `j`'s draw**, with the inputs it is set up on: its `DrawSetupZHJ`'s
table at table `j`'s inputs, or `y₀`'s refused one at table 0's inputs. -/
noncomputable def tableShHJ (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (I' : Inputs) × TableH I' mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    ⟨inputsJ I j, (Classical.choice h).x.table⟩
  else ⟨inputsJ I 0, y₀.x.refused _⟩

theorem tableShHJ_set {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJ dj y₀ S j = ⟨inputsJ I j, (Classical.choice h).x.table⟩ := by
  unfold tableShHJ
  rw [dite_eq_left_of_eq_true (eq_true h)]

theorem tableShHJ_refused {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : ¬ Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJ dj y₀ S j = ⟨inputsJ I 0, y₀.x.refused _⟩ := by
  unfold tableShHJ
  rw [dite_eq_right_of_eq_false (eq_false h)]

theorem tableZHJ_S (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (tableZHJ dj y₀ S j).S = (tableShHJ dj y₀ S j).2.T.S := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    unfold tableZHJ
    rw [dite_eq_left_of_eq_true (eq_true h)]
    rfl
  · rw [tableShHJ_refused dj y₀ h]
    unfold tableZHJ
    rw [dite_eq_right_of_eq_false (eq_false h)]
    rfl

/-- The public plan at the same draws. -/
noncomputable def planShHJ : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tableShHJ dj y₀ S j).2.T

noncomputable def tabShHJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShHJ (Reg := Reg) dj y₀ S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfShHJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planShHJ dj y₀) (tabShHJ dj y₀) S R u = (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.T := by
  simp only [Partition.tabOf, planShHJ, tabShHJ, List.get_ofFn]
  rfl

theorem tableShHJ_tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.T.S =
      (tabOfZ (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀) S R u).S := by
  rw [tabOfZHJ_eq, tableZHJ_S dj y₀]

theorem planZHJ_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S = (Partition.tabOf (planShHJ dj y₀) (tabShHJ dj y₀) S R u).S :=
  fun S R u => (tableShHJ_tabOfZ dj y₀ S R u).symm.trans (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm

variable (ht : I.tags.typed = false)

include ht in
theorem tableShHJ_typed (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableShHJ dj y₀ S j).1.tags.typed = false := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    exact typedJ ht j
  · rw [tableShHJ_refused dj y₀ h]
    exact typedJ ht 0

variable {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)

include ht hc in
/-- **One circuit per session, at every table**: each public table's inputs parse the session's circuit file to `c`
(table `j`'s tags differ from the session's in the identity alone, which an untyped `HmOut.parse` does not read). -/
theorem tableShHJ_parse (S : Finset (Fin n)) (j : Fin (nTab I)) :
    HmOut.parse (tableShHJ dj y₀ S j).1.tags (tableShHJ dj y₀ S j).1.circuitFile (tableShHJ dj y₀ S j).1.tables =
      .ok c := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  · rw [tableShHJ_refused dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc

/-- A hidden-output acceptance at `T`'s twin, as one at `T' = T`'s. -/
def recastAccH {T T' : Model.Statement} {A B : Matrix (Fin (2 ^ T.kLog)) (Fin (2 ^ T.kLog)) (ZMod 2)}
    (a : AcceptedH (HiddenExec.twin T A B)) (e : T = T') :
    AcceptedH (HiddenExec.twin T' (HiddenExec.castM e A) (HiddenExec.castM e B)) :=
  { a with hT := a.hT.trans (HiddenExec.twin_cast e A B) }

variable (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites**: each drawn unit's acceptance, VU and block are its table's public table's. -/
noncomputable def sitesShHJ : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  A S R u _ := HiddenExec.castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinA
  B S R u _ := HiddenExec.castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinB
  sat S R u _ := HiddenExec.twin_sat_cast (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.sat (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _))
  pin S R u _ := HiddenExec.pin_cast (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.pin (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _))
  acc S R u _ := recastAccH ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.acc
    (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)
      (rowsFit_of_scopeH (zkTyped ht) hc hscope))
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session**: every site's statement circuit is the session's one circuit `c`. -/
theorem sitesShHJ_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesShHJ dj y₀ ht hc hscope).acc S R u h).st.c = c :=
  fun S _ u _ => (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.acc_unrolled (tableShHJ_typed dj y₀ ht S _)
    (tableShHJ_parse dj y₀ ht hc S _) (rowsFit_of_scopeH (zkTyped ht) hc hscope)

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

omit ht in
theorem bcSiteH_recast {T T' : Model.Statement} {A B : Matrix (Fin (2 ^ T.kLog)) (Fin (2 ^ T.kLog)) (ZMod 2)}
    (a : AcceptedH (HiddenExec.twin T A B)) (e : T = T') (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : BcSiteH a pop o g u) :
    BcSiteH (recastAccH a e) pop (Fin.cast h' o) g u :=
  hs

include hU hpub hpt in
/-- `BcSiteH` at a table set up at table `j`'s inputs (`tableH_bcSite` at `inputsJ I j`). -/
theorem tableShHJ_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.acc p₁ p₂ (rowsFit_of_scopeH p₁ p₂ hscope)) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSite p₁ p₂ hscope (fun _ => d) hU ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) hpt x h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables (`sitesShH_bcSite`): at a table `verify --zk` sets up
by `tableH_bcSite` at table `j`'s inputs; at a refused one no message satisfies the table. -/
theorem sitesShHJ_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJ dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHJ dj y₀ ht hc hscope).o S R u h)
          ((sitesShHJ dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSite hscope hU hpub hpt _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan**: `keyLayoutH` at `sitesShHJ`. -/
noncomputable abbrev layShHJ :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsH` at `sitesShHJ`, with `hsite` from `sitesShHJ_bcSite`. -/
noncomputable abbrev rsShHJ : Binding.HmRowsX (layShHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShHJ_computes : (rsShHJ dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **Each drawn unit's place in its hidden-output `--zk` table at `J` tables**: `keyPlaceH`'s place in the public
table, along `planZHJ_S`. -/
noncomputable def placeZHJ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHJ :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJ dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJ, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables**: `layShHJ` moved along `planZHJ_S`. -/
noncomputable abbrev layZHJ :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJ dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJ dj y₀ ht hc hscope hU (pub := pub)) (decAgreeHJ dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables**: `rsShHJ` moved along `planZHJ_S`. -/
noncomputable abbrev rsZHJ : HmRowsZX (layZHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) (rsShHJ dj y₀ ht hc hscope hU hpub hpt)

/-- **`hHm` at the hidden-output `--zk` tables' rows, at `J` tables** (`rsShHJ_computes`, moved along). -/
theorem rsZHJ_computes : (rsZHJ dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) _
    (rsShHJ_computes dj y₀ ht hc hscope hU hpub hpt)

end Shadow

/-! ## The tags, the coin server's draws, and the openings -/

/-- **The verify command's tags at any number of tables, under hidden outputs** (`TagsOkZJ` with hidden outputs):
hidden outputs, retained rounds, `hm96-sha512/v1` leaves. -/
def TagsOkZHJ (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1"

/-- **The coin server's draw at `S`, at `J` hidden-output tables** (`DrawOkZKJ` with `setupHidden` for `setupH`): each
table's draw proves the drawn units it is given (`partS`), and setupHidden sets each table up at it with `mPts` link
points. -/
def DrawOkZKHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j)) I.partition I.program = .ok st → st.mPts = mPts

/-- **The `--zk` openings at hidden-output table `j`** (`opnAtZJ` with `setupHidden`): `opnZ` at the statement
setupHidden sets up under table `j`'s tags at its draw, and its `--zk` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnAtZHJ (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (planZ S R).length) : OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (tagsAt (zkTags I) I.count j.val) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZJ ((planZ S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | .error _ => default

/-! ## The message half at the live hidden-output tables -/

/-- **`RunsZC` at the live hidden-output `--zk` tables of a session of `J` tables, at the law's draws**
(`runsZCJ_live` under hidden outputs): on untyped inputs under the tags `TagsOkZHJ` fixes, whose circuit file
`HmOut.parse` accepts, with `wr` the coin server's record of the live session and each of the law's draws setting each
table up at `mPts` link points, `RunsOnZCJ` holds at the tables `planZHJ` under `hm96-sha512/v1`, with the decoder
`decAtZJ`, the openings `opnAtZHJ`, no tail and table `j`'s cap from `root_B[j]`. `runsZCJH_liveU` is it from the
sessions alone: each table's draw setup reads only its own session's committed circuit (`drawSetupH_of_setupHiddenU`). -/
theorem runsZCJH_liveU (I : Inputs) (ht : I.tags.typed = false)
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
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := drawSetupH_of_setupHiddenU (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
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

set_option linter.unusedVariables false in
/-- `runsZCJH_liveU` on every circuit `HmOut.parse` accepts, wherever the outputs are returned. -/
theorem runsZCJH_liveI (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L :=
  runsZCJH_liveU I ht dj y₀ wr L hT hRec hDraw

set_option linter.unusedVariables false in
/-- `runsZCJH_liveI` where the unit returns the outputs. -/
theorem runsZCJH_live (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L :=
  runsZCJH_liveU I ht dj y₀ wr L hT hRec hDraw

set_option linter.unusedVariables false in
/-- `runsZCJH_liveI` at the statement circuit (`HmOut.parseStmt`, a templated file's template unrolled). -/
theorem runsZCJH_liveISt (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L :=
  runsZCJH_liveU I ht dj y₀ wr L hT hRec hDraw

set_option linter.unusedVariables false in
/-- `runsZCJH_live` at the statement circuit (`HmOut.parseStmt`, a templated file's template unrolled). -/
theorem runsZCJH_liveSt (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L :=
  runsZCJH_liveU I ht dj y₀ wr L hT hRec hDraw

end FlockSoundness.Discharge.ZkHidden
