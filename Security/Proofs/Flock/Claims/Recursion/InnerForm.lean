import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.VStarCross
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-!
# Flock's plain verifier sets the inner statement up as its files do

The compiled verifier reads the inner statement through its files (`RecCoins.Files`): its circuit, public file and
partition, and how Flock's plain verifier is told to set it up (`RecCoins.InnerForm`: its statement id, coins, lookup
tables and draw), which each staged V* statement hands it from the inner statement `x` carries (`partRaw`,
`innerFormOf`). Its setup refuses an inner shape off the files' numbers and an inner statement set up otherwise than
the files set it up (`RecSetup.checkShape`, `RecSetup.checkForm`), and its readers refuse the files in part
(`sourceOf`), so `RecScope.accepts` holds both.

`innerSetup_files`: where the setup's checks accept, Flock's plain verifier (`FlockDeploy.setupOf false`) sets the inner
statement up as one table at the `m`, `k_log` and region count, and with the points, that the files set it up at.

* `checkForm_ok`: the inner statement is a `Tags.circuitPlainLeaves` statement on the coin server's coins, with no
  lookup table, at a draw that proves every one of the files' units in order.
* `ins_plainLeaves`, `check_plain`: Flock's plain verifier reads it so (`Raw.ins`), at one table and without public
  inputs (`ProvedScope.check`, which `innerOk` runs).
* `files_stmt`: both are one `Stmt.setupHidden` table (`HiddenExec.setupTables_okH`), the files' at no draw.
* `drawn_full`: a full draw keeps `n`, so `m`, and the regions, whose count is the circuit's ports
  (`regions_size`).
-/

namespace RecDeploy

open Flock FlockSoundness.Refine
open Claims (Msg)

/-! ## The inner statement's form -/

/-- A file read as JSON, where its text parses. -/
theorem jsonOf_of {b : ByteArray} {t : String} {d : Lean.Json} {g : String → String}
    (ht : String.fromUTF8? b = some t) (hd : (Lean.Json.parse t).mapError g = .ok d) (what : String) :
    FlockDeploy.jsonOf what b = .ok d := by
  unfold FlockDeploy.jsonOf
  simp only [ht]
  cases hp : Lean.Json.parse t with
  | error e => rw [hp] at hd; cases hd
  | ok j => rw [hp] at hd; cases hd; rfl

/-- **What `checkForm` takes**: a `Tags.circuitPlainLeaves` statement on the coin server's coins, with no lookup table,
at a draw (`jsonOf`, as Flock's plain verifier reads it) that proves every one of `st`'s units in order. -/
theorem checkForm_ok {fm : RecCoins.InnerForm} {st : Stmt} {u : Unit}
    (h : RecSetup.checkForm (some fm) st = .ok u) :
    fm.statement = Tags.circuitPlainLeaves.name ∧ fm.os = true ∧ fm.tables = [] ∧
      ∃ b d D, fm.draw = some b ∧ FlockDeploy.jsonOf "draw" b = .ok d ∧
        Draw.UnitDraw.ofJson (Flock.norm d) = .ok D ∧ D.proved = Array.range st.n := by
  unfold RecSetup.checkForm at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    exact ⟨by simpa using ‹¬(fm.statement != Tags.circuitPlainLeaves.name) = true›,
      by simpa using ‹¬(!fm.os) = true›, by simpa using ‹¬(!fm.tables.isEmpty) = true›,
      _, _, _, ‹fm.draw = some _›, jsonOf_of ‹String.fromUTF8? _ = some _› ‹Except.mapError _ _ = .ok _› "draw",
      ‹Draw.UnitDraw.ofJson _ = .ok _›, by simpa using ‹¬(_ != Array.range st.n) = true›⟩

/-- **The setup's shape check, whole**: where `RecSetup.checkShape` passes on the inner statement's files, they set it
up, the inner shape's `m`, `k_log`, regions and claims are the ones they set it up at, its shared-row tables are within
`RecBudget.innerRows`, and `RecSetup.checkForm` passes there. -/
theorem checkShape_spec {i : RecVStar.Inner} {f : RecCoins.Files} {u : Unit}
    (h : RecSetup.checkShape (some i) (some (.files f)) = .ok u) :
    ∃ spec st, f.stmt "the inner session" = .ok (spec, st) ∧ i.m = st.m ∧ i.kLog = st.c.kLog ∧
      i.regions = st.regions.size ∧ i.claims = 2 + spec.points * st.regions.size ∧
      RecBudget.innerRows st = .ok () ∧ RecSetup.checkForm f.form st = .ok u := by
  unfold RecSetup.checkShape at h
  rcases hf : f.stmt "the inner session" with e | ⟨spec, st⟩
  · simp [hf, bind, Except.bind] at h
  simp only [hf, bind, Except.bind] at h
  by_cases hc : (i.m != st.m || i.kLog != st.c.kLog || i.regions != st.regions.size ||
      i.claims != 2 + spec.points * st.regions.size) = true
  · simp [hc, throw, throwThe, MonadExceptOf.throw] at h
  · simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hc
    obtain ⟨⟨⟨e1, e2⟩, e3⟩, e4⟩ := hc
    rcases hr : RecBudget.innerRows st with e | v
    · simp [e1, e2, e3, e4, hr, pure, Except.pure] at h
    · cases v
      refine ⟨spec, st, rfl, e1, e2, e3, e4, hr, ?_⟩
      simpa [e1, e2, e3, e4, hr, pure, Except.pure] using h

/-- **The setup's shape check**: where `RecSetup.checkShape` passes on the inner statement's files, they set it up, the
inner shape's `m`, `k_log`, regions and claims are the ones they set it up at, and `RecSetup.checkForm` passes there. -/
theorem checkShape_files {i : RecVStar.Inner} {f : RecCoins.Files} {u : Unit}
    (h : RecSetup.checkShape (some i) (some (.files f)) = .ok u) :
    ∃ spec st, f.stmt "the inner session" = .ok (spec, st) ∧ i.m = st.m ∧ i.kLog = st.c.kLog ∧
      i.regions = st.regions.size ∧ i.claims = 2 + spec.points * st.regions.size ∧
      RecSetup.checkForm f.form st = .ok u := by
  obtain ⟨spec, st, hf, e1, e2, e3, e4, -, hform⟩ := checkShape_spec h
  exact ⟨spec, st, hf, e1, e2, e3, e4, hform⟩

/-- **The inner statement's shared-row tables are within the cap**: where `RecSetup.checkShape` passes on the inner
statement's files, each shared-row table of the statement they set up holds at most `RecBudget.MAX_INNER_ROWS` rows. -/
theorem checkShape_rows {i : RecVStar.Inner} {f : RecCoins.Files} {u : Unit}
    (h : RecSetup.checkShape (some i) (some (.files f)) = .ok u) :
    ∃ spec st, f.stmt "the inner session" = .ok (spec, st) ∧
      ∀ t, st.pub.tables = some t → ∀ r ∈ t.counts, r ≤ RecBudget.MAX_INNER_ROWS := by
  obtain ⟨spec, st, hf, -, -, -, -, hr, -⟩ := checkShape_spec h
  refine ⟨spec, st, hf, fun t ht r hrt => ?_⟩
  unfold RecBudget.innerRows at hr
  rw [ht] at hr
  by_cases ha : (t.counts.any (RecBudget.MAX_INNER_ROWS < ·)) = true
  · simp [ha, throw, throwThe, MonadExceptOf.throw] at hr
  · refine Nat.le_of_not_lt fun hlt => ha (Array.any_eq_true'.2 ⟨r, hrt, ?_⟩)
    exact decide_eq_true hlt

/-! ## Flock's plain verifier on the inner statement -/

/-- **Flock's own checks on the inner statement**: its plain verifier sets it up, and its statement check passes. -/
theorem innerOk_ok {inner : Msg} (h : innerOk inner = .ok ()) :
    ∃ s spec sts, FlockDeploy.setupOf false inner = some (s, spec, sts) ∧ s.read.check = .ok () := by
  unfold innerOk at h
  rcases hs : FlockDeploy.setupOf false inner with _ | ⟨s, spec, sts⟩
  · simp [hs] at h
  · refine ⟨s, spec, sts, rfl, ?_⟩
    simp only [hs] at h
    cases hc : s.read.check with
    | ok u => rfl
    | error e => simp [hc, bind, Except.bind] at h

/-- **The plain verifier's statement check** (`ProvedScope.check`) takes one table and no public inputs. -/
theorem check_plain {r : Readable.Read} (h : r.check = .ok ()) (hz : r.zk = false) :
    r.I.count ≤ 1 ∧ r.publicInputs = none := by
  unfold Readable.Read.check ProvedScope.check at h
  rw [hz] at h
  cases hf : ProvedScope.firstFailure (ProvedScope.statementRules r.tags false r.I.count r.publicInputs.isSome) with
  | some why => simp [hf] at h
  | none =>
    unfold ProvedScope.firstFailure at hf
    have hn : (ProvedScope.statementRules r.tags false r.I.count r.publicInputs.isSome).find? (fun q => !q.1) = none := by
      cases hfind : (ProvedScope.statementRules r.tags false r.I.count r.publicInputs.isSome).find? (fun q => !q.1) with
      | none => rfl
      | some q => simp [hfind] at hf
    have hall := List.find?_eq_none.1 hn
    have hm : ∀ i (hi : i < (ProvedScope.statementRules r.tags false r.I.count r.publicInputs.isSome).length),
        ((ProvedScope.statementRules r.tags false r.I.count r.publicInputs.isSome)[i]).1 = true := fun i hi => by
      simpa using hall _ (List.getElem_mem hi)
    have h5 := hm 4 (by simp [ProvedScope.statementRules])
    have h6 := hm 5 (by simp [ProvedScope.statementRules])
    simp [ProvedScope.statementRules] at h5 h6
    exact ⟨h5, by simpa using h6⟩

/-- **Flock's plain verifier reads a `Tags.circuitPlainLeaves` statement on the coin server's coins** at the files'
tags (`RecCoins.innerTags`) and coins, with the statement's files, table count, lookup tables, partition, draw and
public inputs. -/
theorem ins_plainLeaves {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) (hz : w.zk = false)
    (hs : w.statement = Tags.circuitPlainLeaves.name) (ho : w.osCoins = true) :
    s.read.zk = false ∧ s.read.tags = RecCoins.innerTags ∧ s.read.coins = Coins.os ∧
      s.read.I.circuitFile = w.circuit ∧ s.read.I.publicFile = w.publicFile ∧ s.read.I.count = w.count ∧
      s.read.I.tables = w.tables ∧ w.partition.mapM (FlockDeploy.jsonOf "partition") = .ok s.read.I.partition ∧
      (∃ dr, w.draw.mapM (FlockDeploy.jsonOf "draw") = .ok dr ∧ s.read.draw = dr.map Flock.norm) ∧
      s.read.publicInputs = w.publicInputs := by
  have hf : Tags.find? w.statement = some Tags.circuitPlainLeaves := by rw [hs]; rfl
  unfold FlockDeploy.Raw.ins at h
  simp only [hf, hz, ho, Bool.false_and, Bool.false_eq_true, ↓reduceIte, bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, by assumption, ⟨_, by assumption, rfl⟩, rfl⟩)

/-! ## A full draw keeps the files' numbers -/

/-- `HmOut.regions` has one region per port, whatever the public file. -/
theorem regions_size {c : Circuit} {pub : Public} {rs : Array Region} (h : HmOut.regions c pub = .ok rs) :
    rs.size = c.ports.size := by
  unfold HmOut.regions at h
  exact (FlockSoundness.Discharge.Exec.range_mapM_ok h).1

/-- **A full draw keeps `m` and the regions**: a hidden-output statement set up at a draw that proves every one of the
undrawn statement's units, in order, has its `n`, hence its `m`, its `k_log` and its region count. -/
theorem drawn_full {T : Tags} {cf pf : ByteArray} {co co' : Coins} {tb : List (String × ByteArray)}
    {pa pa' : Option Lean.Json} {pr pr' : Option ByteArray} {d : Lean.Json} {D : Draw.UnitDraw} {st st' : Stmt}
    (h' : Stmt.setupHidden T cf pf co tb none pa pr = .ok st')
    (h : Stmt.setupHidden T cf pf co' tb (some d) pa' pr' = .ok st)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hp : D.proved = Array.range st'.n) :
    st.m = st'.m ∧ st.c.kLog = st'.c.kLog ∧ st.regions.size = st'.regions.size := by
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' h
  have s' := FlockSoundness.Discharge.Hidden.setupHidden_spec' h'
  have hc : st.c = st'.c := Except.ok.inj (s.parse.symm.trans s'.parse)
  obtain ⟨pop, -, -, hd⟩ := s.pub
  have hdr : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hd d rfl
  obtain ⟨D', hD', -, hn, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  have hN : st.n = st'.n := by rw [s.n, hn, hp, Array.size_range]
  have hnbl : st.nbl = st'.nbl := by rw [s.nbl, s'.nbl, s.blocksReal, s'.blocksReal, hN, hc]
  refine ⟨by rw [s.m, s'.m, hnbl, hc], by rw [hc], ?_⟩
  rw [regions_size s.regions, regions_size s'.regions, hc]

/-- **The files set the inner statement up as one hidden-output table at no draw** (`RecCoins.Files.stmt`). -/
theorem files_stmt {f : RecCoins.Files} {spec : SessionSpec} {st : Stmt}
    (h : f.stmt "the inner session" = .ok (spec, st)) :
    Stmt.setupHidden RecCoins.innerTags f.circuit f.pub Coins.os [] none (some f.partition) none = .ok st ∧
      spec = st.spec := by
  unfold RecCoins.Files.stmt at h
  have hh : RecCoins.innerTags.hiddenOutputs = true := rfl
  cases hs : Stmt.setupTables RecCoins.innerTags f.circuit f.pub Coins.os 1 (partition := some f.partition) with
  | error e => simp [hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  | ok q =>
    obtain ⟨spec0, sts0⟩ := q
    rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH hh hs with ⟨-, st0, hst0, rfl, rfl⟩ | ⟨hc, -⟩
    · simp [hs, bind, Except.bind, pure, Except.pure] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨hst0, rfl⟩
    · omega

/-! ## The inner statement, set up -/

/-- **Flock's plain verifier sets the inner statement up as its files do**: where the setup's checks accept what the
verifier reads of a compiled statement (`RecScope.accepts`), it holds the inner statement's files (`filesOf`), and
Flock's plain verifier (`FlockDeploy.setupOf false`) sets the inner statement up as one table at the `m`, `k_log` and
region count, and with the points, that the files set it up at (`RecCoins.Files.stmt`), reading it at the files' tags
(`RecCoins.innerTags`) and with no public inputs: `RecSetup.checkForm` holds the
inner statement to a `Tags.circuitPlainLeaves` statement on the coin server's coins with no lookup table, at a draw of
every one of the files' units in order (`checkForm_ok`); the plain verifier's statement check gives one table and no
public inputs (`check_plain`); so both are one hidden-output table (`files_stmt`), and the full draw keeps the files'
numbers (`drawn_full`). -/
theorem innerSetup_files {x : Msg} {r : XR} (hr : readX x = some r) (ha : RecScope.accepts (readOf r) = true) :
    ∃ f spec' st' s spec st, filesOf r = some (.files f) ∧ f.stmt "the inner session" = .ok (spec', st') ∧
      FlockDeploy.setupOf false r.w.inner = some (s, spec, #[st]) ∧
      st.m = st'.m ∧ st.c.kLog = st'.c.kLog ∧ st.regions.size = st'.regions.size ∧ spec.points = spec'.points ∧
      s.read.tags = RecCoins.innerTags ∧ s.read.publicInputs = none := by
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨-, ⟨u, hsh⟩⟩, hheld⟩ := ha
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  have hheld : (filesOf r).isSome = true := hheld
  obtain ⟨p, hp⟩ : ∃ p, r.partition = some p := by
    cases hq : r.partition with
    | none => simp [filesOf, hq] at hheld
    | some p => exact ⟨p, rfl⟩
  have hfs : filesOf r = some (.files ⟨r.innerRaw.circuit, r.innerRaw.publicFile, p, "", none,
      some (innerFormOf r.innerRaw)⟩) := by rw [filesOf, hp]; rfl
  rw [hfs] at hsh
  obtain ⟨spec', st', hstm, -, -, -, -, hform⟩ := checkShape_files hsh
  obtain ⟨hst', hspec'⟩ := files_stmt hstm
  obtain ⟨hs1, hs2, hs3, b, d, D, hb, hjd, hD, hpr⟩ := checkForm_ok hform
  have hs1 : r.innerRaw.statement = Tags.circuitPlainLeaves.name := hs1
  have hs2 : r.innerRaw.osCoins = true := hs2
  have hs3 : r.innerRaw.tables = [] := hs3
  have hb : r.innerRaw.draw = some b := hb
  obtain ⟨hdec, -, hok⟩ := readX_inner hr
  obtain ⟨s, spec, sts, hset, hchk⟩ := innerOk_ok hok
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hw : r.innerRaw.zk = false ∧ r.innerRaw.ins = .ok s := by
    unfold FlockDeploy.insOf at hins
    obtain ⟨w, hw0, hw⟩ := Option.bind_eq_some_iff.1 hins
    rw [hdec] at hw0
    cases hw0
    split_ifs at hw with hz
    · cases hw
    change r.innerRaw.ins.toOption = some s at hw
    exact ⟨not_not.1 hz, toOption_eq_some hw⟩
  obtain ⟨z1, z2, z3, z4, z5, -, z7, z8, z9, -⟩ := ins_plainLeaves hw.2 hw.1 hs1 hs2
  obtain ⟨hcount, hpi⟩ := check_plain hchk z1
  have hpart : s.read.I.partition = some p := by
    have hpa := (readX_spec hr).2.1
    rw [z8] at hpa
    simp only [Except.toOption, Option.some.injEq] at hpa
    rw [hpa, hp]
  have hdraw : s.read.draw = some (Flock.norm d) := by
    obtain ⟨dr, hz9, he⟩ := z9
    rw [hb] at hz9
    simp only [Option.mapM_some, hjd, Functor.map, Except.map, Except.ok.injEq] at hz9
    rw [he, ← hz9]
    rfl
  have hsetup := FlockDeploy.Ins.setup_read hsu
  unfold Readable.Read.setup at hsetup
  rw [z2, z3, z4, z5, z7, hs3, hdraw, hpart, hpi] at hsetup
  have hh : RecCoins.innerTags.hiddenOutputs = true := rfl
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH hh hsetup with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc, -⟩
  · obtain ⟨e1, e2, e3⟩ := drawn_full hst' hst hD hpr
    exact ⟨_, spec', st', s, _, st, hfs, hstm, hset, e1, e2, e3, by rw [hspec']; rfl, z2, hpi⟩
  · omega

end RecDeploy
