import Proofs.Flock.Soundness.Discharge.ZkPubIn.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

set_option Elab.async false

/-!
# The `J` tables of a hidden-output `--zk` session with the verifier's public inputs

ZkHidden's tables at `J` tables (`ZkHidden/DefsJ.lean`) with `Stmt.setupTables … (some own)` and
`Stmt.setupHidden … (some own)`, `own` the same at every table, as `Stmt.setupTables` passes it, field for field:

* `setupTables_partsHP`, `setupOfP_tableJH`: at `count > 1`, table `j` is `setupHidden … (some own)`'s statement
  under `Tags.tableIdentity count j` at part `j`'s subset draw;
* `DrawSetupZHJP I' own mPts S d`: `DrawSetupZHJ` at a `DrawSetupHP`, with `.table`, `.refused`, `table_eq`,
  `table_seam`, `table_lay` and `drawSetupZHJP_of`;
* `tableZHJP`, `planZHJP`, `tabZHJP` and the public shadow `tableShHJP`, `planShHJP`, `tabShHJP` (`TableHP` for
  `HiddenExec.TableH`), with `planZHJP_S`, `tableShHJP_typed` and `tableShHJP_parse`.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## flock-verify's tables at `count > 1` with the verifier's public inputs -/

section Parts

open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}
  {own : ByteArray}

set_option maxHeartbeats 4000000 in
/-- **`setupTables`'s tables at `count > 1` with the verifier's public inputs** (`ZkHidden.setupTables_partsH` at
`some own`): the same split, and table `j` is `setupHidden … (some own)`'s statement under
`Tags.tableIdentity count j` at part `j`'s subset draw. -/
theorem setupTables_partsHP (hh : tags.hiddenOutputs = true) (hc1 : 1 < count)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure (some own) =
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
            partition program (some own) = .ok sts[j] := by
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

/-- **`verify --zk --public-inputs F` sets table `j` up under its tags at its draw** (`ZkHidden.setupOf_tableJH` at
`some own`): at the record's draw `d`, `setupTables` at `zkInputs I` and `own` has `nTab I` tables, table `j`
setupHidden's statement with `own` under `tagsAt` at `drawJ`. -/
theorem setupOfP_tableJH {I : Inputs} {own : ByteArray} {record : String} {d : Lean.Json} {spec : SessionSpec}
    {sts : Array Flock.Stmt} (hh : I.tags.hiddenOutputs = true) (hd : recordDraw record = some d)
    (hs : setupOfP (zkInputs I) own record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program (some own) = .ok sts[j] := by
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

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

/-- **A hidden-output table `verify --zk --public-inputs F` sets up**, on table `j`'s inputs `I'` (`inputsJ I j`):
`DrawSetupZHJ`'s fields at a `DrawSetupHP` of `I'` and `own`. -/
structure DrawSetupZHJP (I' : Inputs) (own : ByteArray) (mPts : ℕ) (S : Finset (Fin n)) (d : Lean.Json) where
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

variable {I' : Inputs} {own : ByteArray} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJP I' own mPts S d)

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

/-- **The `--zk` table of a hidden-output table `verify --zk --public-inputs F` sets up**, at statement regions
`regs` (`DrawSetupZHJ.tabAt`). -/
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

/-- The table's `--zk` table: setupHidden's statement with `own` at its regions. -/
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
`DrawSetupZHJP` (`ZkSchedOK`). -/
theorem drawSetupZHJP_of {I' : Inputs} {own : ByteArray} {S : Finset (Fin n)} {d : Lean.Json}
    (x : DrawSetupHP I' own mPts S d) (hshape : Flock.Zk.shapeOk x.st = .ok ()) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZHJP I' own mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := zk_schedOK x.st _ _ x.wf.1 x.wf.2 x.hm hshape hz
  exact ⟨⟨x, sch, tPad, hz, h1, h2, h3, h4, h5, h6 _, hshape⟩, rfl, rfl, rfl⟩

section Shadow

variable {I : Inputs} {own : ByteArray} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`, with the verifier's public inputs** (`tableZHJ` with `own`): the table
`verify --zk --public-inputs F` sets up at table `j`'s inputs and draw when it sets one up, and otherwise the reference
setup `y₀`'s refused table. -/
noncomputable def tableZHJP (S : Finset (Fin n)) (j : Fin (nTab I)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table
  else y₀.refused

/-- **The plan of `J` tables with the verifier's public inputs**: at draw `S`, table `j` is `tableZHJP … S j`, for
every registration. -/
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
/-- **The public hidden-output table at table `j`'s draw, with the verifier's public inputs**, with the inputs it is
set up on: its `DrawSetupZHJP`'s table at table `j`'s inputs, or `y₀`'s refused one at table 0's inputs. -/
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

noncomputable def tabShHJP : (S : Finset (Fin n)) → (R : Reg) → Fin n →
    Fin (planShHJP (Reg := Reg) dj y₀ S R).length :=
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
theorem tableShHJP_typed (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableShHJP dj y₀ S j).1.tags.typed = false := by
  by_cases h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJP_set dj y₀ h]
    exact typedJ ht j
  · rw [tableShHJP_refused dj y₀ h]
    exact typedJ ht 0

variable {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)

include ht hc in
/-- **One circuit per session, at every table**: each public table's inputs parse the session's circuit file to `c`
(table `j`'s tags differ from the session's in the identity alone, which an untyped `HmOut.parse` does not read). -/
theorem tableShHJP_parse (S : Finset (Fin n)) (j : Fin (nTab I)) :
    HmOut.parse (tableShHJP dj y₀ S j).1.tags (tableShHJP dj y₀ S j).1.circuitFile (tableShHJP dj y₀ S j).1.tables =
      .ok c := by
  by_cases h : Nonempty (DrawSetupZHJP (inputsJ I j) own mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJP_set dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  · rw [tableShHJP_refused dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc

end Shadow

end FlockSoundness.Discharge.ZkPubIn
