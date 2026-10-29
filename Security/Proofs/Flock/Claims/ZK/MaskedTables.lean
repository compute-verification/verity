import Proofs.Flock.Claims.ZK.Session
import Proofs.Flock.Claims.Scope.Split
import Proofs.Flock.Claims.Scope.StmtOf
import Proofs.Flock.Soundness.Discharge.ZkHidden.Theorems

/-!
# The deployed masked session's tables: `Tables`, `DummyReg` and `Complete`

The masked session's tables are `setupOf true x`'s (`stsOf`). Each is a `--zk` table built from its statement alone
(`ZkHidden.ZkTab`, `ZkTab.table`: the plan of the single-input-group form, which the tree had only at a hidden-output
draw, `DrawSetupZHJ.table`), so every table is a table the session sets up (`setupTable_zkTab`).

* `ZkHidden.schedule_ok_of_shapeOk`: a statement the `--zk` shape check passes has a `--zk` schedule (`queries0` puts
  `m` in `[25, 35]`, where `Zk.schedule` accepts, `scheduleOkB_all`).
* `Readable.tableAt_zk`: each table the masked verifier parses passes the shape check, is a `MaskSetup` and has
  whole-word regions (`setupH_whole` at the scope's `outNet = unitNet` where the outputs are not hidden, or
  `setupHidden_whole`).
* `Readable.tableLowers`: the relation at a parse makes each table's share satisfy the table's lowered statement
  (`stmtOf`). `OneTableLowers` is its one-table case (`oneTableLowers`), the converse of `good_of_committed_plain`'s
  reading; it needs `Parses p`, an untyped statement and no public input ports, under which the readable table's
  matrices are `stmtOf`'s (`tableAt_model`), and none of `SetupTable`'s other side conditions.
* `Readable.parsed_zk`: what the masked scope and the relation give the parse (`FlockComplete.ZkTables.Case.parsed`
  without the case, the verifier's parse of `x`, and the scope's `outNet = unitNet` where the outputs are not hidden).
* `Masked.Live`: the statements in the masked scope each of whose tables has a `ZkTab`; `jOf` is their table count
  (`0` elsewhere), `tabC` their tables, `dummy` a witness holding the regions (`RegionsHold`), and `wtOf` the shares.
  `live_of`: every statement of the scope with a witness is `Live`. So `Tables` (`setupTable_tabC`) and `DummyReg`
  (`dummyReg_tabC`, `coinedOf_reg_of`) hold at every statement, and `Complete` (`complete_tabC`) at every statement with
  a witness.
-/

/-! ## A `--zk` table from its statement alone -/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec Composed
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- `queries0` accepts `m` only from 25 to 35. -/
theorem queries0_range {m q : ℕ} (h : Flock.Zk.queries0 m = .ok q) : 25 ≤ m ∧ m ≤ 35 := by
  unfold Flock.Zk.queries0 at h
  refine ⟨?_, ?_⟩ <;> by_contra hc
  · simp [show m ≤ 24 by omega, throw, throwThe, MonadExceptOf.throw] at h
  · simp [show ¬ m ≤ 24 by omega, show m ≠ 25 by omega, show m ≠ 26 by omega, show m ≠ 27 by omega,
      show ¬ m ≤ 31 by omega, show ¬ m ≤ 35 by omega, throw, throwThe, MonadExceptOf.throw] at h

/-- Whether `Zk.schedule` accepts `m`. -/
def scheduleOkB (m : ℕ) : Bool :=
  match Flock.Zk.schedule m with
  | .ok _ => true
  | .error _ => false

set_option maxRecDepth 100000 in
theorem scheduleOkB_all : ∀ m, 25 ≤ m → m ≤ 35 → scheduleOkB m = true := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- **A statement the `--zk` shape check passes has a `--zk` schedule**: `queries0` puts `m` in `[25, 35]`, where
`Zk.schedule` accepts. -/
theorem schedule_ok_of_shapeOk {st : Flock.Stmt} (h : Flock.Zk.shapeOk st = .ok ()) :
    ∃ p, Flock.Zk.schedule st.m = .ok p := by
  unfold Flock.Zk.shapeOk at h
  obtain ⟨_, hq, -⟩ := except_bind_ok h
  obtain ⟨h1, h2⟩ := queries0_range hq
  have hb := scheduleOkB_all st.m h1 h2
  unfold scheduleOkB at hb
  split at hb
  · exact ⟨_, by assumption⟩
  · cases hb

/-- **A table `verify --zk` sets up, from its statement alone**: well formed, passing the `--zk` shape check, a
`MaskSetup` of whole-word regions that some witness holds, and its `--zk` schedule. -/
structure ZkTab (st : Flock.Stmt) where
  wf : StmtWF st ∧ RegionsWF st
  hzk : Flock.Zk.shapeOk st = .ok ()
  mask : MaskSetup st
  whole : WholeWords st
  regs : ∃ z, RegionsHold (stmtOf st wf.1 (regionsOf st)) z
  sch : Flock.Schedule
  tPad : ℕ
  hz : Flock.Zk.schedule st.m = .ok (sch, tPad)

namespace ZkTab

variable {st : Flock.Stmt} (y : ZkTab st)

include y in
theorem hm : 13 ≤ st.m := by
  have := (schedule_range y.hz).1
  omega

theorem hN : innerN (layOf st y.sch) < 2 ^ 64 :=
  (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).2.2.2.1

theorem hL : (layOf st y.sch).len + Zk.K_PAD ≤ innerN (layOf st y.sch) :=
  (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).2.2.2.2.1

theorem innerN_pos : 0 < innerN (layOf st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64`. -/
def innerDom : Fin (innerN (layOf st y.sch)) ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin (innerN (layOf st y.sch)), j.val + 1 < 2 ^ 64 := fun j => by
      have := y.hN
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- **The table's `--zk` table**: `stmtOf` at its regions, its schedule, and the inner proof's layout
(`DrawSetupZHJ.table`'s fields, read off the statement and its schedule, `zk_schedOK`). -/
noncomputable def table : TabZK 𝔽 st.mPts where
  S := stmtOf st y.wf.1 (regionsOf st)
  sch := schedOf y.sch
  hshape := (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).1
  hmsch := (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).2.1
  hm := y.hm
  t := y.tPad
  hpad := (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).2.2.1
  c₀ := fOf Zk.PAD_C
  e := 2
  ptLocal := PT_LOCAL
  hlay := stmtOf_linkLayout st y.wf.1 y.wf.2
  np := (layOf st y.sch).len
  pads := padsOfLay (layOf st y.sch)
  padsA := padsAOfLay (layOf st y.sch)
  tr := tripleOfLay (layOf st y.sch)
  N := innerN (layOf st y.sch)
  dom := y.innerDom
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ y.innerN_pos
  hpos := (zk_schedOK st y.sch y.tPad y.wf.1 y.wf.2 y.hm y.hzk y.hz).2.2.2.2.2 y.innerN_pos
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code y.innerDom ((layOf st y.sch).len + Zk.K_PAD) : Set (Fin (innerN (layOf st y.sch)) → 𝔽))
  hL := y.hL
  hδ := le_rfl

end ZkTab

end FlockSoundness.Discharge.ZkHidden

/-! ## What the masked scope gives each table -/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge Exec

/-- Under `--zk`, every table `Read.tablesOf` accepts passes `Zk.shapeOk`. -/
theorem tablesOf_shapeOk {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (hz : r.zk = true) :
    sts.forM Zk.shapeOk = .ok () := by
  unfold Read.tablesOf at hs
  split at hs
  · simp only [hz, ↓reduceIte] at hs
    split at hs
    · rename_i h
      cases hs
      exact h
    · cases hs
  · cases hs

/-- **Each table the masked verifier parses is a `--zk` table it sets up**: untyped, without public input ports, at a
circuit whose output is its unit's where its outputs are not hidden (the scope's `scopeOk`), each table passes the shape
check, is a `MaskSetup` and has whole-word regions. -/
theorem tableAt_zk {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (hz : r.zk = true)
    (ht : r.tags.typed = false) (hpi : r.publicInputs = none) {c : Circuit}
    (hc : ProvedScope.circuitOf r.tags r.I.circuitFile r.I.tables = .ok c)
    (ho : r.tags.hiddenOutputs = false → c.outNet = c.unitNet) {j : ℕ} (hj : j < sts.size) :
    Zk.shapeOk sts[j] = .ok () ∧ Composed.MaskSetup sts[j] ∧ Composed.WholeWords sts[j] := by
  refine ⟨array_forM_ok (tablesOf_shapeOk hs hz) j hj, ?_⟩
  obtain ⟨spec, hset⟩ := setup_of_tablesOf hs
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt r.tags r.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs] at hd
  cases hh : r.tags.hiddenOutputs
  · rw [hh] at hd
    simp only [Bool.false_eq_true, ↓reduceIte] at hd
    have hp := (setupH_spec ht' hd).1
    have hc' : c = sts[j].c := by
      rw [Composed.parse_tagsAt] at hp
      simp only [ProvedScope.circuitOf, hh, Bool.false_eq_true, ↓reduceIte, hp] at hc
      exact (Except.ok.inj hc).symm
    exact ⟨Composed.maskSetup_of_setupH ht' hd, Composed.setupH_whole hd (hc' ▸ ho hh) (parse_facts hp).typed⟩
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    exact ⟨ZkHidden.maskSetup_of_setupHidden ht' hd, Composed.setupHidden_whole hd⟩

/-- **The relation at a parse makes each table's share satisfy the table's lowered statement**: at a parse of an untyped
statement without public input ports, a witness in the readable relation is, at table `j` (`zOf` at `j`), a witness of
`stmtOf` at the table's own regions. -/
theorem tableLowers {p : Parsed} (hp : Parses p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none)
    {w : Claims.Flock.Wit} (hR : Claims.Flock.RelS (toReadable relation p) w) {j : ℕ} {st : Stmt}
    (hst : p.sts[j]? = some st) (hwf : Refine.StmtWF st) :
    (Refine.stmtOf st hwf (Refine.regionsOf st)).Satisfies (zOf p.sts.size w j st.m) := by
  obtain ⟨hj, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨hf, hsat⟩ := (relS_relation p w).1 hR j hj
  obtain ⟨-, -, hmat⟩ := tableAt_model (tablesOf_of_parses hp) ht hpi hj
  have e : p.sts[j]! = p.sts[j] := getElem!_pos p.sts j hj
  revert hf hsat
  rw [e]
  rintro hf ⟨h1, h2, h3⟩
  have hA : ((Tab.ofStmt p.sts[j]).toModel hf.2.2).A₀ = (Refine.stmtOf p.sts[j] hwf (Refine.regionsOf p.sts[j])).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Tab.ofStmt p.sts[j]).toModel hf.2.2).B₀ = (Refine.stmtOf p.sts[j] hwf (Refine.regionsOf p.sts[j])).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  refine ⟨fun o i => ?_, h2, h3⟩
  have := h1 o i
  rw [hA, hB] at this
  exact this

/-- **The relation at a one-table parse makes the witness satisfy the table's lowered statement**: at a parse of an
untyped statement without public input ports whose one table is `st`, a witness in the readable relation is, as its
model witness (`zOf`), a witness of `stmtOf st` at its own regions. The converse of `good_of_committed_plain`'s
reading, which `RefSession.Complete` needs at each table's share (`Readable.tableWit`, `zOf_tableWit`). -/
def OneTableLowers : Prop :=
  ∀ (p : Parsed) (st : Stmt) (w : Claims.Flock.Wit) (hwf : Refine.StmtWF st), Parses p → p.tags.typed = false →
    p.publicInputs = none → p.sts = #[st] → Claims.Flock.RelS (toReadable relation p) w →
      (Refine.stmtOf st hwf (Refine.regionsOf st)).Satisfies (zOf 1 w 0 st.m)

theorem oneTableLowers : OneTableLowers := by
  intro p st w hwf hp ht hpi hpe hR
  have h := tableLowers hp ht hpi hR (j := 0) (by simp [hpe]) hwf
  rwa [show p.sts.size = 1 by simp [hpe]] at h

/-- **What the masked scope and the relation give the parse**: at a statement of the masked record's scope with a
witness in its relation, whose setup has the tables `sts`, the verifier's parse is masked, untyped, without public input
ports, has the tables `sts`, the witness is in its readable relation, and the circuit's output is its unit's where its
outputs are not hidden. -/
theorem parsed_zk {rel : Parsed → Relation} {x : Claims.Msg} {w : Claims.Flock.Wit} {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hsc : Claims.Flock.Scope (FlockDeploy.deployZk rel) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk rel).parse x w)
    (hset : FlockDeploy.setupOf true x = some (s, spec, sts)) :
    ∃ p : Parsed, Parses p ∧ p.zk = true ∧ p.tags.typed = false ∧ p.publicInputs = none ∧ p.sts = sts ∧
      (FlockDeploy.deployZk rel).parse x = some (toReadable rel p) ∧ Claims.Flock.RelS (toReadable rel p) w ∧
      ∃ c, ProvedScope.circuitOf p.tags p.I.circuitFile p.I.tables = .ok c ∧
        (p.tags.hiddenOutputs = false → c.outNet = c.unitNet) := by
  obtain ⟨p, hp, hsc'⟩ := (scope_iff (FlockDeploy.deployZk rel) rel (FlockDeploy.readOf true) rfl
    (fun _ _ h => FlockDeploy.readOf_zk h) x).1 hsc
  have hparse : (FlockDeploy.deployZk rel).parse x = some (toReadable rel p) := by
    change (parseBytes (FlockDeploy.readOf true) x).map (toReadable rel) = _
    rw [hp]
    rfl
  obtain ⟨s', hs', hR'⟩ := hR
  change (parseBytes (FlockDeploy.readOf true) x).map (toReadable rel) = some s' at hs'
  rw [hp] at hs'
  simp only [Option.map_some, Option.some.injEq] at hs'
  subst hs'
  have hi := (FlockDeploy.setupOf_some hset).1
  unfold parseBytes FlockDeploy.readOf at hp
  rw [hi] at hp
  have ht := toRead_of_parse (r := s.read) hp
  have hps := parses_of_parse hp
  have hz : p.zk = true := by
    have h1 : FlockDeploy.readOf true x = some s.read := by simp [FlockDeploy.readOf, hi]
    show p.toRead.zk = true
    rw [ht]
    exact FlockDeploy.readOf_zk h1
  obtain ⟨hc, -⟩ := (scope_ok_iff p).1 hsc'
  obtain ⟨hRu, c, pub, hcirc, -, -, hok⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  obtain ⟨spec', hset'⟩ := setup_of_tablesOf (tablesOf_of_parses hps)
  rw [ht] at hset'
  have hpe : p.sts = sts := by
    have := hset'.symm.trans (FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some hset).2)
    simp only [Except.ok.injEq, Prod.mk.injEq] at this
    exact this.2
  have ho : p.tags.hiddenOutputs = false → c.outNet = c.unitNet := fun hh => by
    unfold FormOk at hok
    split at hok
    · simp only [ProvedScope.scopeOkT, Bool.and_eq_true, beq_iff_eq] at hok
      first | exact hok.1.1.2 | simp_all
    obtain ⟨-, hok⟩ := hok
    split at hok
    · rename_i hh'
      cases hh'.symm.trans hh
    · simp only [ProvedScope.scopeOk, Bool.and_eq_true, beq_iff_eq] at hok
      exact hok.1.1.1.2
  exact ⟨p, hps, hz, hRu.2.1, hpi, hpe, hparse, hR', c, hcirc, ho⟩

end Flock.Readable

/-! ## The masked session's tables -/

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.Discharge

/-- **The masked session's tables at statement `x`**: `setupOf true x`'s, none where it does not set up. -/
def stsOf (x : Msg) : Array _root_.Flock.Stmt :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) => sts
  | none => #[]

theorem setupOf_of_pos {x : Msg} (h : 0 < (stsOf x).size) :
    ∃ s spec, FlockDeploy.setupOf true x = some (s, spec, stsOf x) := by
  unfold stsOf at h ⊢
  revert h
  cases FlockDeploy.setupOf true x with
  | none => intro h; simp at h
  | some v =>
    intro _
    obtain ⟨s, spec, sts⟩ := v
    exact ⟨s, spec, rfl⟩

/-- **The statements whose tables the masked session proves**: in the masked scope, each table a `--zk` table from its
statement alone (`ZkTab`). Every statement of the scope with a witness is one (`live_of`). -/
def Live (x : Msg) : Prop :=
  Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x ∧
    ∀ j (hj : j < (stsOf x).size), Nonempty (ZkHidden.ZkTab (stsOf x)[j])

open Classical in
/-- **The masked session's table count at `x`**: `setupOf true x`'s at a `Live` statement, `0` elsewhere. -/
noncomputable def jOf (x : Msg) : ℕ := if Live x then (stsOf x).size else 0

theorem jOf_le (x : Msg) : jOf x ≤ (stsOf x).size := by
  unfold jOf
  split_ifs <;> omega

theorem live_of_lt {x : Msg} (j : Fin (jOf x)) : Live x := by
  by_contra h
  have := j.isLt
  simp only [jOf, h, ↓reduceIte] at this
  omega

theorem lt_of_jOf {x : Msg} (j : Fin (jOf x)) : j.val < (stsOf x).size := lt_of_lt_of_le j.isLt (jOf_le x)

/-- Table `j` of the masked session at `x`. -/
def stAt (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Stmt := (stsOf x)[j.val]'(lt_of_jOf j)

/-- Table `j`'s `ZkTab`. -/
noncomputable def zkTab (x : Msg) (j : Fin (jOf x)) : ZkHidden.ZkTab (stAt x j) :=
  Classical.choice ((live_of_lt j).2 j.val (lt_of_jOf j))

/-- **Table `j` at `x` as `Coined` tables**: `coinedOf` at its `--zk` table (`ZkTab.table`) and its statement. -/
noncomputable def tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) : Composed.CoinedSetup :=
  Composed.coinedOf P (zkTab x j).table (stAt x j)

/-- **A `ZkTab`'s `Coined` tables are a table the session sets up**: `ZkTab`'s fields are `SetupTable`'s side
conditions, with mask avoidance from the shape check (`avoidsMask_of_shapeOk`). -/
theorem setupTable_zkTab (P : Composed.LigProver) {st : _root_.Flock.Stmt} (y : ZkHidden.ZkTab st) :
    SetupTable (Composed.coinedOf P y.table st) :=
  ⟨st.mPts, P, y.table, st, y.wf.1, y.wf.2, rfl, y.mask, FailClosed.avoidsMask_of_shapeOk y.wf.2 y.hzk, y.whole, rfl⟩

/-- **Every table of the masked session is a table the session sets up.** -/
theorem setupTable_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) : SetupTable (tabC P x j) :=
  setupTable_zkTab P (zkTab x j)

open Classical in
/-- **The dummy witness**: one holding the statement's regions (`RegionsHold`), where one exists. -/
noncomputable def dummy (S : FlockSoundness.Model.Statement) : FlockSoundness.Model.Witness S.m :=
  if h : ∃ z, Composed.RegionsHold S z then h.choose else fun _ => 0

theorem dummy_holds {S : FlockSoundness.Model.Statement} (h : ∃ z, Composed.RegionsHold S z) :
    Composed.RegionsHold S (dummy S) := by
  rw [dummy, dite_eq_left h]
  exact h.choose_spec

/-- **The dummy's table witnesses carry the public region data**, at every table and coin (`coinedOf_reg_of`). -/
theorem dummyReg_tabC (P : Composed.LigProver) (x : Msg) (j : Fin (jOf x)) (c : Composed.CoinsZK (tabC P x j).Rest) :
    ((tabC P x j).T c).tab.sReg (dummy (tabC P x j).S) = ((tabC P x j).T c).tab.pubReg :=
  Composed.coinedOf_reg_of P (zkTab x j).table (stAt x j) (zkTab x j).wf.1 (zkTab x j).wf.2 rfl (zkTab x j).whole c _
    (dummy_holds (zkTab x j).regs)

/-- **The witness's table witnesses**: each table's share (`Readable.tableWit`) as its model witness (`Readable.zOf`). -/
noncomputable def wtOf (P : Composed.LigProver) (x : Msg) (w : Claims.Flock.Wit) (j : Fin (jOf x)) :
    FlockSoundness.Model.Witness (tabC P x j).S.m :=
  _root_.Flock.Readable.zOf 1 (_root_.Flock.Readable.tableWit (stsOf x).size w j.val) 0 (tabC P x j).S.m

/-- **The lowering's completeness at the masked session's tables**: at a statement with a witness in the deployed
relation, each table's share is valid (`tableLowers`, `zOf_tableWit`). -/
theorem complete_tabC (P : Composed.LigProver) {x : Msg} {w : Claims.Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (j : Fin (jOf x)) :
    (tabC P x j).Valid (wtOf P x w j) := by
  have hj := lt_of_jOf j
  obtain ⟨s, spec, hset⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨p, hp, -, ht, hpi, hpe, -, hRp, -⟩ := _root_.Flock.Readable.parsed_zk (live_of_lt j).1 hR hset
  have h := _root_.Flock.Readable.tableLowers hp ht hpi hRp (j := j.val) (st := stAt x j)
    (by rw [hpe]; exact Array.getElem?_eq_getElem hj) (zkTab x j).wf.1
  rw [hpe, ← _root_.Flock.Readable.zOf_tableWit hj] at h
  exact h

/-- **Every statement of the masked scope with a witness is `Live`**: each of its tables is a `--zk` table it sets up
(`tableAt_model`, `tableAt_zk`), with a schedule (`schedule_ok_of_shapeOk`), whose regions the witness's share holds
(`tableLowers`). -/
theorem live_of {x : Msg} {w : Claims.Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) : Live x := by
  refine ⟨hsc, fun j hj => ?_⟩
  obtain ⟨s, spec, hset⟩ := setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨p, hp, hz, ht, hpi, hpe, -, hRp, c, hc, ho⟩ := _root_.Flock.Readable.parsed_zk hsc hR hset
  have hs : p.toRead.tablesOf = some (stsOf x) := by
    rw [← hpe]
    exact _root_.Flock.Readable.tablesOf_of_parses hp
  obtain ⟨hwf, hrw, -⟩ := _root_.Flock.Readable.tableAt_model hs ht hpi hj
  obtain ⟨hzk, hmask, hwhole⟩ := _root_.Flock.Readable.tableAt_zk hs hz ht hpi hc ho hj
  obtain ⟨⟨sch, tPad⟩, hsch⟩ := ZkHidden.schedule_ok_of_shapeOk hzk
  have hsat := _root_.Flock.Readable.tableLowers hp ht hpi hRp (by rw [hpe]; exact Array.getElem?_eq_getElem hj) hwf
  exact ⟨⟨⟨hwf, hrw⟩, hzk, hmask, hwhole, ⟨_, Composed.RegionsHold.of_satisfies hsat⟩, sch, tPad, hsch⟩⟩

/-- At a statement with a witness, `Live` is the masked scope. -/
theorem live_iff {x : Msg} {w : Claims.Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) :
    Live x ↔ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x :=
  ⟨fun h => h.1, fun h => live_of h hR⟩

end Claims.ZKMal.Masked
