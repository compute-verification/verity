module

public import Proofs.Properties.Verity.Recursive.BaseSetup
public import Proofs.Properties.Verity.Sampled.Record.PartSame
public import Proofs.Flock.Claims.Deploy

/-! v2's verdict on its one session is the record's decision at that session. -/

@[expose] public section

namespace Verity.Sampled.Record

open Lean (Json)
open Flock (V SessionSpec Stmt Setup)
open Flock.PartCong (PartSame)
open FlockDeploy (Coin Final Ins)
open Verity.Recursive.BaseSetup (bind_ok forIn_mem body_ok body_yield noIdentifier_of_check)

private theorem ok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

private theorem mapError_ok {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

private theorem forIn_ok_of_all {α : Type} {xs : Array α} {b : α → PUnit → V (ForInStep PUnit)}
    (hb : ∀ a ∈ xs, ∀ u, b a u = .ok (.yield PUnit.unit)) : forIn xs PUnit.unit b = .ok PUnit.unit := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray]
  induction l with
  | nil => rfl
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind]
    rw [hb c (by simp) PUnit.unit]
    exact ih fun a ha u => hb a (by simp_all) u

structure SessionMatch (si : Ins) (I : Flock.Witness.Inputs) (rd : Spot.OneStageZk.Reads) : Prop where
  zk : si.read.zk = true
  stmtTags : si.stmtTags = I.tags
  tags : si.read.I.tags = I.tags
  circuitFile : si.read.I.circuitFile = I.circuitFile
  publicFile : si.read.I.publicFile = I.publicFile
  count : si.read.I.count = I.count
  tables : si.read.I.tables = I.tables
  partition : PartSame si.read.I.partition I.partition
  program : si.read.I.program = I.program
  work : si.read.I.work = none
  stratified : si.read.I.stratified = none
  closure : si.read.I.closure = none
  publicInputs : si.read.publicInputs = none
  registered : si.registered = rd.registered
  readRegistered : si.read.registered = rd.registered.isSome

section Match

variable {si : Ins} {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads}

theorem SessionMatch.read_tags (hm : SessionMatch si I rd) : si.read.tags = Flock.Witness.zkTags I := by
  unfold Flock.Readable.Read.tags
  rw [ite_eq_left hm.zk]
  unfold FlockSoundness.Discharge.ZkExec.zkTags Flock.Witness.zkTags
  rw [hm.tags, hm.circuitFile, hm.count]

theorem SessionMatch.read_coins (hm : SessionMatch si I rd) : si.read.coins = Flock.Witness.zkCoins I := by
  unfold Flock.Readable.Read.coins
  rw [ite_eq_left hm.zk]
  unfold FlockSoundness.Discharge.ZkExec.zkCoins Flock.Witness.zkCoins
  rw [hm.tags, hm.count]

theorem SessionMatch.read_setup (hm : SessionMatch si I rd) {draw : Option Json} (hdraw : si.read.draw = draw) :
    si.read.setup = Flock.Stmt.setupTables (Flock.Witness.zkTags I) I.circuitFile I.publicFile
      (Flock.Witness.zkCoins I) I.count I.tables draw I.partition I.program := by
  unfold Flock.Readable.Read.setup
  rw [hm.read_tags, hm.read_coins, hm.circuitFile, hm.publicFile, hm.count, hm.tables, hdraw, hm.program, hm.work,
    hm.stratified, hm.closure, hm.publicInputs]
  exact Flock.PartCong.setupTables_same _ _ _ _ _ _ _ hm.partition _ _ _ _ _

theorem SessionMatch.read_check (hm : SessionMatch si I rd) :
    si.read.check = Flock.ProvedScope.check (Flock.Witness.zkTags I) true I.circuitFile I.publicFile I.tables I.count
      false rd.registered.isSome := by
  unfold Flock.Readable.Read.check
  rw [hm.read_tags, hm.zk, hm.circuitFile, hm.publicFile, hm.tables, hm.count, hm.publicInputs, hm.readRegistered]
  all_goals rfl

end Match

def scopeOf (I : Flock.Witness.Inputs) (rd : Spot.OneStageZk.Reads) : V Unit := do
  Flock.ProvedScope.checkOf (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)
    (Flock.Witness.zkTags I) true I.publicFile I.count false rd.registered.isSome
  Spot.OneStageZk.templateOk (← Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)

theorem setupTables_of_build {tags : Flock.Tags} {cB pB : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : ℕ} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {spec : SessionSpec} {sts : Array Stmt}
    (hb : Spot.OneStageZk.buildSessionOf (Flock.ProvedScope.circuitOf tags cB tables) tags cB pB coins tables
      count draw partition program registered = .ok (spec, sts)) :
    tags.hm96Rows = true ∧
      Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok (spec, sts) ∧
      (∀ st ∈ sts, Flock.checkRegistered st registered = .ok () ∧ FlockDeploy.noIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull registered draw = .ok () := by
  unfold Spot.OneStageZk.buildSessionOf at hb
  have hh : tags.hm96Rows = true := by
    refine Classical.byContradiction fun hne => ?_
    simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hb
  simp only [hh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind] at hb
  have hstep : ∃ x : SessionSpec × Array Stmt,
      Flock.Stmt.setupTables tags cB pB coins count tables draw partition program = .ok x ∧
      (∀ st ∈ x.2, Flock.checkRegistered st registered = .ok () ∧
        Spot.OneStageZk.checkIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull registered draw = .ok () ∧ (x.1, x.2) = (spec, sts) := by
    split at hb
    · next hhid =>
      obtain ⟨x, hs, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hloop, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hfull, hb⟩ := bind_ok hb
      refine ⟨x, ?_, fun st hst => (body_ok (A := fun st => Flock.checkRegistered st registered)
        (B := Spot.OneStageZk.checkIdentifier)).1 (forIn_mem body_yield hloop st hst), hfull,
        Except.ok.inj hb⟩
      rw [Flock.Stmt.setupTables_eq]
      simpa only [Flock.ProvedScope.circuitOf, hhid, eq_self_iff_true, ↓reduceIte] using hs
    · obtain ⟨x, hs, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hloop, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hfull, hb⟩ := bind_ok hb
      exact ⟨x, hs, fun st hst => (body_ok (A := fun st => Flock.checkRegistered st registered)
        (B := Spot.OneStageZk.checkIdentifier)).1 (forIn_mem body_yield hloop st hst), hfull,
        Except.ok.inj hb⟩
  obtain ⟨⟨spec', sts'⟩, hs, hpt, hfull, hx⟩ := hstep
  simp only [Prod.mk.injEq] at hx
  obtain ⟨rfl, rfl⟩ := hx
  exact ⟨hh, hs, fun st hst => ⟨(hpt st hst).1, noIdentifier_of_check (hpt st hst).2⟩, hfull⟩

theorem sessionAt_ok {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {draw : Option Json}
    {spec : SessionSpec} {sts : Array Stmt}
    (h : Spot.OneStageZk.sessionAt I rd draw = .ok (spec, sts)) :
    Flock.Registered.checkProgram I.program rd.registered.isSome = .ok () ∧
      Flock.RecSetup.checkInner none none (pure (none, none)) = .ok () ∧
      Spot.OneStageZk.buildSessionOf
          (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables) (Flock.Witness.zkTags I)
          I.circuitFile I.publicFile (Flock.Witness.zkCoins I) I.tables I.count draw I.partition I.program
          rd.registered = .ok (spec, sts) ∧
      Spot.OneStageZk.setupChecks I rd sts = .ok () ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
      Spot.OneStageZk.recChecks rd sts = .ok () := by
  unfold Spot.OneStageZk.sessionAt at h
  dsimp only at h
  obtain ⟨⟨⟩, hprog, h⟩ := bind_ok h
  obtain ⟨⟨⟩, hinner, h⟩ := bind_ok h
  obtain ⟨r, hb, h⟩ := bind_ok h
  obtain ⟨⟨⟩, hsc, h⟩ := bind_ok h
  obtain ⟨⟨⟩, hsh, h⟩ := bind_ok h
  obtain ⟨⟨⟩, hrc, h⟩ := bind_ok h
  obtain rfl : r = (spec, sts) := Except.ok.inj h
  exact ⟨hprog, hinner, hb, hsc, hsh, hrc⟩

theorem setupAt_ok {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {draw : Option Json}
    {spec : SessionSpec} {tabs : Array Setup} {p o : Json}
    (h : Spot.OneStageZk.setupAt I rd draw = .ok (spec, tabs, p, o)) :
    ∃ sts, Spot.OneStageZk.sessionAt I rd draw = .ok (spec, sts) ∧ tabs = FlockDeploy.tabsOf sts ∧
      (if draw.isSome then Flock.ProvedScope.sampleDesign true (Flock.Witness.zkTags I) false draw else pure () : V Unit) =
        .ok () := by
  unfold Spot.OneStageZk.setupAt at h
  obtain ⟨⟨spec', sts⟩, hs, h⟩ := bind_ok h
  dsimp only at h
  obtain ⟨⟨⟩, hlaw, h⟩ := bind_ok h
  have h := Except.ok.inj h
  simp only [Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, -, -⟩ := h
  exact ⟨sts, mapError_ok hs, rfl, hlaw⟩

theorem setupChecks_checks {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {sts : Array Stmt}
    (h : Spot.OneStageZk.setupChecks I rd sts = .ok ()) :
    Flock.RecSetup.checks none none I.tags I.tables sts I.circuitFile I.partition I.program rd.registered none none
      none = .ok () := by
  unfold Spot.OneStageZk.setupChecks at h
  rcases hr : rd.registered with _ | own
  all_goals
    simp only [hr] at h
    obtain ⟨⟨⟩, -, h⟩ := bind_ok h
    obtain ⟨⟨⟩, -, h⟩ := bind_ok h
    first
      | (obtain ⟨⟨⟩, hc, -⟩ := bind_ok h
         exact hc)
      | (obtain ⟨⟨⟩, -, h⟩ := bind_ok h
         obtain ⟨⟨⟩, hc, -⟩ := bind_ok h
         exact hc)

theorem setup_of_setupAt {si : Ins} {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads}
    {draw : Option Json} {spec : SessionSpec} {tabs : Array Setup} {p o : Json} (hm : SessionMatch si I rd)
    (hdraw : si.read.draw = draw) (h : Spot.OneStageZk.setupAt I rd draw = .ok (spec, tabs, p, o)) :
    ∃ sts, si.setup = .ok (spec, sts) ∧ tabs = FlockDeploy.tabsOf sts := by
  obtain ⟨sts, hsa, rfl, hlaw⟩ := setupAt_ok h
  obtain ⟨-, hci, hb, hsc, hsh, hrc⟩ := sessionAt_ok hsa
  obtain ⟨hh, hs0, hregs, hfull⟩ := setupTables_of_build hb
  have hchecks := setupChecks_checks hsc
  unfold Spot.OneStageZk.recChecks at hrc
  obtain ⟨⟨⟩, hcoins, hrc⟩ := bind_ok hrc
  obtain ⟨⟨⟩, hpub, -⟩ := bind_ok hrc
  refine ⟨sts, ?_, rfl⟩
  have hci' : Flock.RecSetup.checkInner none none (Except.ok (none, none)) = .ok () := hci
  have hpart := fun tags tables sts circuitB program registered =>
    Flock.PartCong.checks_none tags tables sts circuitB si.read.I.partition I.partition program registered
  unfold FlockDeploy.Ins.setup
  simp only [hm.read_tags, hm.read_setup hdraw, hdraw, hm.zk, hm.registered, hm.stmtTags, hm.tables,
    hm.circuitFile, hpart, hm.program]
  simp only [hh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hci', hs0, bind, Except.bind, pure, Except.pure]
  rw [forIn_ok_of_all fun a ha u => by simp only [(hregs a ha).1, (hregs a ha).2]]
  rcases draw with _ | d
  · simp only [hfull, hchecks, hsh, hcoins, hpub, ↓reduceIte, Option.isSome_none, Bool.false_eq_true]
  · have hl : Flock.ProvedScope.sampleDesign true (Flock.Witness.zkTags I) false (some d) = .ok () := by
      simpa only [Option.isSome_some, eq_self_iff_true, ↓reduceIte] using hlaw
    simp only [hfull, hchecks, hsh, hcoins, hpub, ↓reduceIte, Option.isSome_some, hl]

theorem decide_of_verdictOf {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {bs : Core.Blobs}
    {ksFiles population : V Unit} {coinKeys : Array String} {record : String} {q : Spot.OneStageZk.RunFiles}
    {coinKey : Option String} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {si : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final}
    (hv : Spot.OneStageZk.verdictOf I (scopeOf I rd) ksFiles population coinKeys record
      (Flock.Witness.recordDraw record) (Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record))
      (Spot.OneStageZk.filesOf bs q
        (Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record))) coinKey = .ok ())
    (hact : FlockDeploy.act true x as cs = .decide si spec sts link c₁ mids f) (hm : SessionMatch si I rd)
    (hdraw : si.read.draw = Flock.Witness.recordDraw record) (hstmt : f.stmt = x) (hrec : f.record = record)
    (hfiles : Spot.OneStageZk.filesOf bs q
      (Spot.OneStageZk.setupAt I rd (Flock.Witness.recordDraw record)) = .ok f.proofs)
    (hlive : (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : V Unit) = .ok ()) :
    ((FlockDeploy.next true x as cs).eval FlockDeploy.sha512).1 = .acc := by
  rw [FlockDeploy.next_decide_sha512 hact]
  unfold Spot.OneStageZk.verdictOf at hv
  obtain ⟨proofs, hpf, hv⟩ := bind_ok hv
  rw [hfiles] at hpf
  obtain rfl : f.proofs = proofs := Except.ok.inj hpf
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨⟨spec0, tabs, p0, o0⟩, hst, hv⟩ := bind_ok hv
  dsimp only at hv
  obtain ⟨⟨⟩, hzv, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hsc, hv⟩ := bind_ok hv
  unfold scopeOf at hsc
  obtain ⟨⟨⟩, hsc, -⟩ := bind_ok hsc
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hdl, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hrt, -⟩ := bind_ok hv
  obtain ⟨sts0, hsa, rfl, -⟩ := setupAt_ok hst
  obtain ⟨-, -, hb, -, -, -⟩ := sessionAt_ok hsa
  obtain ⟨-, hs0, -, -⟩ := setupTables_of_build hb
  have hs := FlockDeploy.Ins.setup_read (FlockDeploy.setupOf_some (FlockDeploy.act_decide hact)).2
  rw [hm.read_setup hdraw, hs0] at hs
  have hs := Except.ok.inj hs
  simp only [Prod.mk.injEq] at hs
  obtain ⟨rfl, rfl⟩ := hs
  have hds : (Flock.Witness.recordDraw record).isSome = true := by
    cases hd : Flock.Witness.recordDraw record with
    | none =>
      simp only [hd, Option.isNone_none, ↓reduceIte] at hdl
      all_goals cases hdl
    | some _ => rfl
  refine ⟨?_, ?_⟩
  · rw [ite_eq_left (rfl : true = true), hrec]
    exact hzv
  · unfold FlockDeploy.checks
    refine ok_bind (a := ()) ?_ ?_
    · simp [Flock.Record.check, hstmt]
      all_goals rfl
    refine ok_bind (a := ()) ?_ ?_
    · have hre : FlockSoundness.Discharge.Exec.recordDraw f.record = Flock.Witness.recordDraw record := by
        rw [hrec]
        all_goals rfl
      rw [hre, hdraw, FlockDeploy.sampleEq_refl]
      all_goals rfl
    obtain ⟨lv, hlv, hcu⟩ := bind_ok hlive
    refine ok_bind hlv ?_
    refine ok_bind hcu ?_
    refine ok_bind (a := ()) ?_ ?_
    · rw [hm.read_check]
      exact hsc
    refine ok_bind (a := ()) ?_ ?_
    · obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 hds
      rw [hdraw, hd]
      all_goals rfl
    rw [hrec]
    exact hrt

end Verity.Sampled.Record
