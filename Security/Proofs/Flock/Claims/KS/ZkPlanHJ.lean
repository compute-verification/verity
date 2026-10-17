import Proofs.Flock.Claims.KS.ZkPlanJ
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# The old hidden-output `--zk` plan at the deployed masked statement is `zkTabs`

`ZkPlanJ` under hidden outputs (`Stmt.setupHidden`, `ZkHidden/DefsJ.lean`): the old hidden-output `--zk` theorems run
their decoders on the plan `ZkHidden.planZHJ`, whose table `j` is the table `verify --zk` sets up at table `j`'s inputs
and draw (`DrawSetupZHJ.table`). Where `verify --zk` sets a record up and its tables pass the `--zk` shape check, that
plan is `zkTabs` too, so the old hidden-output decoders are right at `zkTabs`:

* `tableH_eq_zkTab`: a `DrawSetupZHJ`'s table is `zkTab` at its statement, at every layout and link layout (`Layout`
  is a `Prop`, and `Layout.model` is `stmtOf` at setupHidden's statement: `Integrate.model_eq` at its parse facts and
  default leaf, `Hidden.setupHidden_spec'`).
* `setupZHJ_of_setupOf`: where `verify --zk` sets the record up at the draw `dj S`, `DrawOkZKHJ` holds and table `j`
  passes the shape check and has `Zk.schedule`'s schedule, table `j` has a `DrawSetupZHJ` at its statement.
* `tableZHJ_eq_zkTabs`, `planZHJ_eq_tabsJ`: so the plan there is `tabsJ` of `zkTabs`, at every layout.
* `drawOkZKHJ_empty`: at no drawn units, `DrawOkZKHJ` is the tables' link points (`setupTables_pointsAny`).
* `decodesOnZKHJ_wrD`: `Custody.decodesOnZKJ_wrD` under hidden outputs (`ZkHidden.runsZCJH_liveI` at `wrD`).
* `decodesTableH_zkTabs`: so at the empty draw the deployed hidden-output `--zk` decoders are right at `zkTabs`, on
  the outcomes `verify --zk` accepts at `wrD` (`DecodesTable`, `ZkLiveAtJ`'s `hdec` at the verdict `accepted`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (nTab inputsJ drawJ partS outJ repJ decAtZJ)
open FlockSoundness.Discharge.ZkHidden (DrawSetupZHJ DrawOkZKHJ TagsOkZHJ tableZHJ planZHJ opnAtZHJ)
open FlockSoundness.Discharge.Tables (DecodesTable)

local notation "𝔽" => FlockLevel3.GF128

/-- **A `DrawSetupZHJ`'s table is `zkTab`** at its statement, at every layout and link layout: the same schedule and
`t_pad`, `stmtOf` the layout's model statement (`Integrate.model_eq`: setupHidden's circuit has `HmRow.parse`'s facts
and its leaf is the default one), and the same padding constant, pads and inner code. -/
theorem tableH_eq_zkTab {n mPts : ℕ} {I' : Exec.Inputs} {S : Finset (Fin n)} {d : Lean.Json}
    (ht : I'.tags.typed = false) (y : DrawSetupZHJ I' mPts S d) {st : Stmt} (hst : y.x.st = st)
    (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL mPts) :
    y.table = zkTab L hz hk hlay := by
  obtain ⟨x, sch', tPad', hz', _, _, _, _, _, _, _⟩ := y
  subst hst
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz'.symm.trans hz))
  have s := Hidden.setupHidden_spec' x.h
  have hS : Refine.stmtOf x.st x.wf.1 (Refine.regionsOf x.st) = L.model (Refine.regionsOf x.st) :=
    (Integrate.model_eq (parse_facts (HiddenExec.parse_untyped ht s.parse).1) L s.leaf x.wf.1 _).symm
  unfold DrawSetupZHJ.table DrawSetupZHJ.tabAt zkTab
  congr 1

/-- **Hidden-output table `j` is set up where `verify --zk` sets the record up**: at the record's draw `dj S`, on
untyped inputs whose circuit file `HmOut.parse` accepts, where `DrawOkZKHJ` holds, the tables pass the `--zk` shape
check and table `j` has `Zk.schedule`'s schedule, table `j` has a `DrawSetupZHJ` at its inputs, its draw's units of
`S` and its draw, at its statement and schedule. -/
theorem setupZHJ_of_setupOf {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hshape : sts.forM Flock.Zk.shapeOk = .ok ()) (hd : Exec.recordDraw record = some (dj S))
    (hok : DrawOkZKHJ I dj mPts S) (j : Fin (nTab I)) (hj : j.val < sts.size) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule sts[j.val].m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j),
      y.x.st = sts[j.val] ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨-, hstJ⟩ := ZkHidden.setupOf_tableJH hh hd hs
  have hst := hstJ j.val hj
  obtain ⟨_, hf⟩ := ZkHidden.fast100_of_schedule hz
  have ht' : (inputsJ I j.val).tags.typed = false := Composed.typedJ ht j.val
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (ZkHidden.parseH_tagsAt ((ZkExec.zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨_, -, hdr⟩ := HiddenExec.setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := HiddenExec.drawSetupH_of_setupHiddenI (I := inputsJ I j.val) (S := partS I (dj S) S j)
    (mPts := mPts) ht' hcJ hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, hyx, hys, hyt⟩ := ZkHidden.drawSetupZHJ_of x (by rw [hx]; exact ZkExec.array_forM_ok hshape j.val hj)
    (sch := sch) (tPad := tPad) (by rw [hx]; exact hz)
  exact ⟨y, by rw [hyx, hx], hys, hyt⟩

/-- **The hidden-output plan's table `j` is `zkTabs`'s** where `verify --zk` sets the record up at the draw `dj S`,
the tables pass the shape check and `DrawOkZKHJ` holds. -/
theorem tableZHJ_eq_zkTabs {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hshape : sts.forM Flock.Zk.shapeOk = .ok ()) (hd : Exec.recordDraw record = some (dj S))
    (hok : DrawOkZKHJ I dj mPts S) (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts)
    (j : Fin (nTab I)) (hj : j.val < sts.size) :
    tableZHJ dj y₀ S j = zkTabs h0 L hz hk hlay j.val := by
  obtain ⟨y, hyst, -, -⟩ := setupZHJ_of_setupOf hh ht hc dj hs hshape hd hok j hj (hz j.val hj)
  unfold tableZHJ
  rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩), DrawSetupZHJ.table_eq _ y, zkTabs_of_lt h0 L hz hk hlay hj]
  exact tableH_eq_zkTab (Composed.typedJ ht j.val) y hyst (L j.val hj) (hz j.val hj) (hk j.val hj) (hlay j.val hj)

/-- **The hidden-output plan is `tabsJ` of `zkTabs`** where `verify --zk` sets the record up at the draw `dj S`, the
tables pass the shape check and `DrawOkZKHJ` holds, at every registration: the setup has the plan's `nTab I` tables
(`setupOf_tableJH`), each `zkTabs`'s. -/
theorem planZHJ_eq_tabsJ {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
    {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hshape : sts.forM Flock.Zk.shapeOk = .ok ()) (hd : Exec.recordDraw record = some (dj S))
    (hok : DrawOkZKHJ I dj mPts S) (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (R : Reg) :
    planZHJ (Reg := Reg) dj y₀ S R = tabsJ sts.size (zkTabs h0 L hz hk hlay) := by
  have hsz := (ZkHidden.setupOf_tableJH hh hd hs).1
  refine List.ext_get (by rw [ZkHidden.planZHJ_length, tabsJ_length, hsz]) fun i h1 h2 => ?_
  have hi : i < nTab I := by rwa [ZkHidden.planZHJ_length] at h1
  simp only [planZHJ, tabsJ, List.get_eq_getElem, List.getElem_ofFn]
  exact tableZHJ_eq_zkTabs hh ht hc dj y₀ hs hshape hd hok h0 L hz hk hlay ⟨i, hi⟩ (by omega)

/-- **`DrawOkZKHJ` at no drawn units** is the tables' link points, which are the session's (`setupTables_pointsAny`). -/
theorem drawOkZKHJ_empty {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = true) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {record : String} {spec : SessionSpec} {sts : Array Stmt}
    (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts)) (hd : Exec.recordDraw record = some (dj ∅)) :
    DrawOkZKHJ I dj spec.mPts ∅ := by
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH hh hd hs
  have hpts := (HiddenExec.setupTables_pointsAny hs).2
  intro j
  refine ⟨fun _ _ u hu => by simp [partS] at hu, fun st hst => ?_⟩
  have hj : j.val < sts.size := by have := j.isLt; omega
  obtain rfl : st = sts[j.val] := Except.ok.inj (hst.symm.trans (hstJ j.val hj))
  exact hpts j.val hj

/-- **The deployed hidden-output `--zk` decoders are right at `wrD`** (`Custody.decodesOnZKJ_wrD` under hidden
outputs): `wrD` records the live session (`recordsLiveZKJ_wrD`), so the old hidden-output discharge
(`runsZCJH_liveI`) needs only untyped inputs under `TagsOkZHJ` whose circuit file `HmOut.parse` accepts, and every draw
setting each table up at `mPts` link points (`DrawOkZKHJ`). -/
theorem decodesOnZKHJ_wrD {n mPts : ℕ} {Reg : Type} (I : Exec.Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (L : Audit.Law n) (hT : TagsOkZHJ I)
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts (L.draw ω)) :
    Composed.DecodesOnZKJ execArith ZkExec.hs512 (planZHJ (Reg := Reg) dj y₀) I
      (Claims.Bridge.Custody.wrD I dj fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length)
      (opnAtZHJ (planZHJ dj y₀) I dj) (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => Exec.capOfRoot rs[j.val]!) L :=
  Composed.decodesOnZKJ_of_runsOnZCJ execArith ZkExec.hs512 _ I _ _ _ _ _ _
    (ZkHidden.runsZCJH_liveI I ht hc dj y₀ _ L hT (Claims.Bridge.Custody.recordsLiveZKJ_wrD I dj _) hDraw)

/-- **The deployed hidden-output `--zk` decoders are right at `zkTabs`**: where `verify --zk` sets the record up at
the empty draw `dj ∅` and its tables pass the shape check, on untyped inputs under `TagsOkZHJ` whose circuit file
`HmOut.parse` accepts, at every layout and link layout at the session's link points, the old hidden-output openings
and decoder at the plan, moved to `tabsJ` of `zkTabs` (`planZHJ_eq_tabsJ`), are right on the outcomes `verify --zk`
accepts at `wrD` (`decodesOnZKHJ_wrD` at the law of that one draw, `drawOkZKHJ_empty`, table 0's setup as the
reference). -/
theorem decodesTableH_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {record : String} {spec : SessionSpec} {sts : Array Stmt}
    (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts)) (hshape : sts.forM Flock.Zk.shapeOk = .ok ())
    (hd : Exec.recordDraw record = some (dj ∅))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {Reg : Type} (R : Reg) :
    ∃ (opn : Array ByteArray → Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length →
        OpensZ 𝔽 Refine.Dig (List UInt8))
      (dec : Fin (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length → ℕ → ByteArray → List Msg),
      ∀ i, DecodesTable (outJ ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) (fun _ o => o.accepted = true)
        (fun o => ZkExec.AcceptsZK I
          (Claims.Bridge.Custody.wrD I dj (fun _ _ => (tabsJ sts.size (zkTabs h0 L hz hk hlay)).length) ∅ R o))
        opn dec (fun _ _ => true) (fun _ _ _ => []) (fun rs j => Exec.capOfRoot rs[j.val]!)
        (repJ execArith ZkExec.hs512 (tabsJ sts.size (zkTabs h0 L hz hk hlay))) i := by
  have hok := drawOkZKHJ_empty hT.1 dj hs hd
  obtain ⟨y₀, -, -, -⟩ :=
    setupZHJ_of_setupOf hT.1 ht hc dj hs hshape hd hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := decodesOnZKHJ_wrD (Reg := Reg) I ht hc dj y₀ { Ω := Unit, draw := fun _ => ∅ } hT (fun _ => hok) () R
  rw [← planZHJ_eq_tabsJ hT.1 ht hc dj y₀ hs hshape hd hok h0 L hz hk hlay R]
  exact ⟨_, _, h⟩

end FlockDeploy.KS
