import Proofs.Flock.Claims.KS.ZkLiveJ
import Proofs.Flock.Claims.Bridge.Custody

/-!
# The old `--zk` plan at the deployed masked statement is `zkTabs`

The old `--zk` theorems run their decoders on a plan of `J` tables (`Composed.planZJ`): at a draw, table `j` is the
table `verify --zk` sets up at table `j`'s inputs and draw (`DrawSetupZJ.table`), and a reference setup's refused table
where it sets none up. `ZkLiveJ` reads C-Flock ZK's padded tables of the deployed setup (`zkTabs`). Where `verify --zk`
sets a record up, the two are one list, so the old decoders are right at `zkTabs`:

* `table_eq_zkTab`: a `DrawSetupZJ`'s table is `zkTab` at its statement, at every layout and link layout (`Layout` is a
  `Prop`, and `Layout.model` is `stmtOf`: `Integrate.model_eq_stmtOf`).
* `setupZJ_of_setupOf`: where `verify --zk` sets the record up (`Exec.setupOf`) at the draw `dj S`, `DrawOkZKJ` holds and
  table `j` has `Zk.schedule`'s schedule, table `j` has a `DrawSetupZJ` at its statement (`setupZJ_of_acceptsZK` without
  the verdict).
* `tableZJ_eq_zkTabs`, `planZJ_eq_tabsJ`: so the plan there is `tabsJ` of `zkTabs`, at every layout.
* `drawOkZKJ_empty`: at no drawn units, `DrawOkZKJ` is the tables' link points, which `setupTables` makes the session's
  (`setupTables_points`).
* `decodesTable_zkTabs`: so at the empty draw the deployed `--zk` decoders are right at `zkTabs`, on the outcomes
  `verify --zk` accepts at `wrD` (`DecodesTable`, `ZkLiveAtJ`'s `hdec` at the verdict `accepted`), from
  `Custody.decodesOnZKJ_wrD` at the law of that one draw, with table 0's setup as the reference.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Composed (DrawSetupZJ DrawOkZKJ nTab inputsJ drawJ partS tableZJ planZJ outJ repJ)
open FlockSoundness.Discharge.Tables (DecodesTable)

local notation "𝔽" => FlockLevel3.GF128

/-- **A `DrawSetupZJ`'s table is `zkTab`** at its statement, at every layout and link layout: the same schedule and
`t_pad` (`Zk.schedule` is a function), `stmtOf` the layout's model statement (`Integrate.model_eq_stmtOf`), and the same
padding constant, pads and inner code. -/
theorem table_eq_zkTab {n mPts : ℕ} {I' : Exec.Inputs} {S : Finset (Fin n)} {d : Lean.Json}
    (ht : I'.tags.typed = false) (y : DrawSetupZJ I' mPts S d) {st : Stmt} (hst : y.x.st = st)
    (L : Layout st) {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule st.m = .ok (sch, tPad))
    (hk : st.c.kLog ≤ 27) (hlay : (L.model (Refine.regionsOf st)).LinkLayout PT_LOCAL mPts) :
    y.table = zkTab L hz hk hlay := by
  obtain ⟨x, sch', tPad', hz', _, _, _, _, _, _⟩ := y
  subst hst
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz'.symm.trans hz))
  have hS : Refine.stmtOf x.st x.wf.1 (Refine.regionsOf x.st) = L.model (Refine.regionsOf x.st) :=
    (Integrate.model_eq_stmtOf ht x.h x.wf.1 _).symm
  unfold DrawSetupZJ.table DrawSetupZJ.tabAt zkTab
  congr 1

/-- **Table `j` is set up where `verify --zk` sets the record up**: at the record's draw `dj S`, where `DrawOkZKJ` holds
and table `j` has `Zk.schedule`'s schedule, table `j` has a `DrawSetupZJ` at its inputs, its draw's units of `S` and its
draw, at its statement and schedule. -/
theorem setupZJ_of_setupOf {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hd : Exec.recordDraw record = some (dj S)) (hok : DrawOkZKJ I dj mPts S) (j : Fin (nTab I))
    (hj : j.val < sts.size) {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule sts[j.val].m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j),
      y.x.st = sts[j.val] ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨-, hstJ⟩ := Composed.setupOf_tableJ hh hd hs
  have hst := hstJ j.val hj
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz
  have ht' : (inputsJ I j.val).tags.typed = false := Composed.typedJ ht j.val
  obtain ⟨_, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, hyx, hys, hyt⟩ := Composed.drawSetupZJ_of ht' x (sch := sch) (tPad := tPad) (by rw [hx]; exact hz)
  exact ⟨y, by rw [hyx, hx], hys, hyt⟩

/-- **The plan's table `j` is `zkTabs`'s** where `verify --zk` sets the record up at the draw `dj S` and `DrawOkZKJ`
holds. -/
theorem tableZJ_eq_zkTabs {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hd : Exec.recordDraw record = some (dj S)) (hok : DrawOkZKJ I dj mPts S) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts)
    (j : Fin (nTab I)) (hj : j.val < sts.size) :
    tableZJ dj y₀ S j = zkTabs h0 L hz hk hlay j.val := by
  obtain ⟨y, hyst, -, -⟩ := setupZJ_of_setupOf hh ht dj hs hd hok j hj (hz j.val hj)
  unfold tableZJ
  rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩), DrawSetupZJ.table_eq _ y, zkTabs_of_lt h0 L hz hk hlay hj]
  exact table_eq_zkTab (Composed.typedJ ht j.val) y hyst (L j.val hj) (hz j.val hj) (hk j.val hj) (hlay j.val hj)

/-- **The plan is `tabsJ` of `zkTabs`** where `verify --zk` sets the record up at the draw `dj S` and `DrawOkZKJ`
holds, at every registration: the setup has the plan's `nTab I` tables (`setupOf_tableJ`), each `zkTabs`'s. -/
theorem planZJ_eq_tabsJ {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    {n mPts : ℕ} {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀) {S : Finset (Fin n)} {record : String} {spec : SessionSpec}
    {sts : Array Stmt} (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts))
    (hd : Exec.recordDraw record = some (dj S)) (hok : DrawOkZKJ I dj mPts S) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL mPts) (R : Reg) :
    planZJ (Reg := Reg) dj y₀ S R = tabsJ sts.size (zkTabs h0 L hz hk hlay) := by
  have hsz := (Composed.setupOf_tableJ hh hd hs).1
  refine List.ext_get (by rw [Composed.planZJ_length, tabsJ_length, hsz]) fun i h1 h2 => ?_
  have hi : i < nTab I := by rwa [Composed.planZJ_length] at h1
  simp only [planZJ, tabsJ, List.get_eq_getElem, List.getElem_ofFn]
  exact tableZJ_eq_zkTabs hh ht dj y₀ hs hd hok h0 L hz hk hlay ⟨i, hi⟩ (by omega)

/-- **`DrawOkZKJ` at no drawn units** is the tables' link points, which are the session's (`setupTables_points`). -/
theorem drawOkZKJ_empty {I : Exec.Inputs} (hh : I.tags.hiddenOutputs = false) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {record : String} {spec : SessionSpec} {sts : Array Stmt}
    (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts)) (hd : Exec.recordDraw record = some (dj ∅)) :
    DrawOkZKJ I dj spec.mPts ∅ := by
  obtain ⟨hsz, hstJ⟩ := Composed.setupOf_tableJ hh hd hs
  have hh' : (ZkExec.zkTags I).hiddenOutputs = false := (ZkExec.zkTags_hiddenOutputs I).trans hh
  have hpts := (Exec.setupTables_points hh' hs).2
  intro j
  refine ⟨fun _ _ u hu => by simp [partS] at hu, fun st hst => ?_⟩
  have hj : j.val < sts.size := by have := j.isLt; omega
  obtain rfl : st = sts[j.val] := Except.ok.inj (hst.symm.trans (hstJ j.val hj))
  exact hpts j.val hj

/-- **The deployed `--zk` decoders are right at `zkTabs`**: where `verify --zk` sets the record up at the empty draw
`dj ∅`, on untyped inputs under `TagsOkZJ`, at every layout and link layout at the session's link points, the old
openings and decoder at the plan, moved to `tabsJ` of `zkTabs` (`planZJ_eq_tabsJ`), are right on the outcomes
`verify --zk` accepts at `wrD` (`Custody.decodesOnZKJ_wrD` at the law of that one draw, `drawOkZKJ_empty`, table 0's
setup as the reference). -/
theorem decodesTable_zkTabs {I : Exec.Inputs} (ht : I.tags.typed = false) (hT : Composed.TagsOkZJ I) {n : ℕ}
    (dj : Finset (Fin n) → Lean.Json) {record : String} {spec : SessionSpec} {sts : Array Stmt}
    (hs : Exec.setupOf (ZkExec.zkInputs I) record = .ok (spec, sts)) (hd : Exec.recordDraw record = some (dj ∅))
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
  have hok := drawOkZKJ_empty hT.1 dj hs hd
  obtain ⟨y₀, -, -, -⟩ := setupZJ_of_setupOf hT.1 ht dj hs hd hok ⟨0, Composed.nTab_pos I⟩ h0 (hz 0 h0)
  have h := Claims.Bridge.Custody.decodesOnZKJ_wrD (Reg := Reg) I ht dj y₀ { Ω := Unit, draw := fun _ => ∅ } hT
    (fun _ => hok) () R
  rw [← planZJ_eq_tabsJ hT.1 ht dj y₀ hs hd hok h0 L hz hk hlay R]
  exact ⟨_, _, h⟩

end FlockDeploy.KS
