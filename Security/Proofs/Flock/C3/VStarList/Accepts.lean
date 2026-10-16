import Flock.RecSetup
import Proofs.Flock.C3.VStarList.Range
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

/-!
# Piece (h): `AcceptsC3` from the verifier's acceptance

Both of `FlockVerify.lean`'s setups of a candidate-3 session, `verify`'s `setupOf` and `statement`'s, first run
candidate 3's checks on the inner statement's files (`RecSetup.checkInner`: check 5's sparse range on the statement they
set up, `RecCoins.checkAtC3`, and META's binding, `checkMeta`), hold the inner transcript against R's registration with
`RecSetup.commits`, and check the session with `RecSetup.checks`: each statement as V*'s (`RecVStar.checkStmt`) and the
covers check (`RecVStar.checkCovers`). So:

* `c3_of_checks`: where they pass at a candidate-3 shape (`ish.sparse`) and the inner statement `st` sets up from the
  inner files' circuit and public file, `AcceptsC3` holds at `st`, check 5's verdict included (`sparseRangeOk_of_check`
  at `st`: one circuit and public file set up one `c`, `m` and regions, whatever the partition and program,
  `setupHidden_same`), its inner shape `checkInnerStmt` among them.
* `acceptsC3_of_verify`: `AcceptsC3`, at `verify`'s setup (`Exec.setupOf`).
* Every candidate-3 session either setup accepts is one these read: the verifier holds the inner files
  (`files_of_checkInner`), the inner transcript (`commits_of_checks`) and R's registration, both JSON
  (`held_of_commits`).
* `sparse_of_acceptsC3`: `AcceptsC3` names a RecSparse shape (`checkInnerStmt`), so `hcoins_of_acceptsC3`: at an
  accepted candidate-3 session check 5's source is the files of the record the verifier holds the inner transcript
  against, never `--rec-inner-coins`'s record, which `checkAtC3` refuses there.
-/

namespace FlockC3.VStarList

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (ebind_ok)
open Flock
open Lean (Json)

variable {ish : RecVStar.Inner} {cs : RecVStar.InnerCommits} {cf : ByteArray} {f : RecCoins.Files}
  {obj partition : Option Json} {prog program : Option ByteArray} {spec : SessionSpec} {st : Stmt} {tags : Tags}
  {tables : List (String × ByteArray)} {bound : Except String (Option String × Option RecVStar.StackedShape)}
  {sts : Array Stmt} {circuitB : ByteArray} {registered rows : Option (Array Registered.Port)} {part : Option ℕ}
  {bindings : Option VStar.Bindings} {tt rt record : String} {tr reg : Json}

theorem boundKey_inj {a b : Option String × Option RecVStar.StackedShape}
    (h : RecSetup.boundKey a = RecSetup.boundKey b) : a = b := by
  obtain ⟨a1, _ | ⟨am, ac⟩⟩ := a <;> obtain ⟨b1, _ | ⟨bm, bc⟩⟩ := b <;> cases h <;> rfl

/-- **META's check**: the inner circuit parses, and its META binds what its header's does. -/
theorem checkMeta_ok (h : RecSetup.checkMeta cf bound = .ok ()) :
    ∃ c, HmOut.parse RecVStar.innerTags cf = .ok c ∧ RecSetup.metaBound c.metaJ = bound := by
  unfold RecSetup.checkMeta at h
  obtain ⟨c, hc, h⟩ := ebind_ok h
  obtain ⟨b, hb, h⟩ := ebind_ok h
  obtain ⟨b', hb', h⟩ := ebind_ok h
  refine ⟨c, ?_, ?_⟩
  · cases hp : HmOut.parse RecVStar.innerTags cf with
    | error e => rw [hp] at hc; cases hc
    | ok c' => rw [hp] at hc; cases hc; rfl
  · split at h
    · cases h
    · rename_i hk
      rw [hb, hb', boundKey_inj (Decidable.of_not_not hk)]

theorem metaBound_ok {m : Json} {d : Option String} {sh : Option RecVStar.StackedShape}
    (h : RecSetup.metaBound m = .ok (d, sh)) :
    RecVStar.stackedShapeOf m = .ok sh ∧
      d = (match m.getObjVal? "registration" with | .ok (.str d) => some d | _ => none) := by
  unfold RecSetup.metaBound at h
  obtain ⟨sh', hsh, h⟩ := ebind_ok h
  cases hr : m.getObjVal? "registration" with
  | error e => rw [hr] at h; cases h; exact ⟨hsh, rfl⟩
  | ok v =>
    rw [hr] at h
    cases v <;> (try cases h)
    exact ⟨hsh, rfl⟩

/-- **A candidate-3 session holds the inner statement's files**: `checkInner` refuses one without them. -/
theorem files_of_checkInner {src : Option RecCoins.Source} (hsp : ish.sparse.isSome = true)
    (h : RecSetup.checkInner (some ish) src bound = .ok ()) : ∃ f, src = some (.files f) := by
  unfold RecSetup.checkInner at h
  obtain ⟨⟨⟩, h1, -⟩ := ebind_ok h
  obtain ⟨f, -, -, hf, -⟩ := checkAtC3_spec h1 hsp
  exact ⟨f, hf⟩

/-- **Candidate 3's acceptance names a RecSparse shape**: `AcceptsC3` holds the inner shape to the inner statement's
(`checkInnerStmt`), and `checkInnerShape` refuses an inner shape with no RecSparse shape. -/
theorem sparse_of_acceptsC3 (h : AcceptsC3 ish st tr record reg) : ish.sparse.isSome = true := by
  obtain ⟨⟨⟩, hi⟩ := ok_of_isOk h.2.2.2.2
  exact (checkInnerStmt_ok hi).1

/-- **Check 5's source at an accepted candidate-3 session is the files of its record**, `_hcoins`'s form at the record
`r`: `AcceptsC3` at `r` names a RecSparse shape (`sparse_of_acceptsC3`), where `checkInner` refuses every source but
the inner statement's files (`files_of_checkInner`: `--rec-inner-coins` is refused there), and the verifier holds the
inner transcript against its source's record (`hrec`: `verify` reads it as `recCoins.map (·.record)`), so the files'
record is `r`. -/
theorem hcoins_of_acceptsC3 {coins : String → Option RecCoins.Source} {r : String}
    (hA : AcceptsC3 ish st tr r reg) (hin : RecSetup.checkInner (some ish) (coins r) bound = .ok ())
    (hrec : ∀ s, coins r = some s → s.record = r) :
    ∀ s, coins r = some s → ∃ f, s = .files f ∧ f.record = r := by
  intro s hs
  obtain ⟨f, hf⟩ := files_of_checkInner (sparse_of_acceptsC3 hA) hin
  rw [hf] at hs
  cases hs
  exact ⟨f, rfl, hrec _ hf⟩

theorem checkInner_ok (hsp : ish.sparse.isSome = true)
    (h : RecSetup.checkInner (some ish) (some (.files f)) bound = .ok ()) :
    ∃ spec st, f.stmt "the inner session" = .ok (spec, st) ∧ RecCoins.checkInnerRange ish st = .ok () ∧
      RecSetup.checkMeta f.circuit bound = .ok () := by
  unfold RecSetup.checkInner at h
  obtain ⟨⟨⟩, h1, h⟩ := ebind_ok h
  obtain ⟨_, spec, st, hf, hst, hr⟩ := checkAtC3_spec h1 hsp
  cases hf
  simp only [hsp, ↓reduceIte] at h
  exact ⟨spec, st, hst, hr, h⟩

theorem checks_ok
    (h : RecSetup.checks (some ish) (some cs) tags tables sts circuitB partition program registered part bindings rows =
      .ok ()) :
    sts.forM (RecVStar.checkStmt (some ish) tags tables · circuitB partition program registered part bindings rows) =
      .ok () ∧ RecVStar.checkCovers ish cs = .ok () := by
  unfold RecSetup.checks at h
  simp only [Option.isNone_some, Bool.and_false, Bool.false_eq_true, ↓reduceIte] at h
  obtain ⟨⟨⟩, h1, h⟩ := ebind_ok h
  exact ⟨h1, h⟩

/-- **A candidate-3 session holds the inner transcript**: `checks` refuses one without it. -/
theorem commits_of_checks {commits : Option RecVStar.InnerCommits} (hsp : ish.sparse.isSome = true)
    (h : RecSetup.checks (some ish) commits tags tables sts circuitB partition program registered part bindings rows =
      .ok ()) : ∃ cs, commits = some cs := by
  cases commits with
  | some cs => exact ⟨cs, rfl⟩
  | none =>
    simp only [RecSetup.checks, Option.bind_some, hsp, Option.isNone_none, Bool.and_self, ↓reduceIte, throw,
      throwThe, MonadExceptOf.throw, bind, Except.bind, reduceCtorEq] at h

theorem commits_ok (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (h : RecSetup.commits (some ish) (.ok tt) (.ok (some rt)) bound record = .ok cs) :
    ∃ d sh, bound = .ok (d, sh) ∧ RecVStar.innerCommits tr record (some reg) d sh = .ok cs := by
  unfold RecSetup.commits at h
  cases hb : bound with
  | error e => simp [hb, bind, Except.bind, pure, Except.pure, hreg] at h
  | ok b =>
    obtain ⟨d, sh⟩ := b
    refine ⟨d, sh, rfl, ?_⟩
    simpa [hb, bind, Except.bind, pure, Except.pure, hreg, htr] using h

/-- **A candidate-3 session holds R's registration**, and both files are JSON: `commits` refuses one without it. -/
theorem held_of_commits {t : Except String String} {r : Except String (Option String)}
    (hsp : ish.sparse.isSome = true) (h : RecSetup.commits (some ish) t r bound record = .ok cs) :
    ∃ tt tr rt reg, t = .ok tt ∧ Json.parse tt = .ok tr ∧ r = .ok (some rt) ∧ Json.parse rt = .ok reg := by
  unfold RecSetup.commits at h
  replace h := ebind_ok h
  obtain ⟨_ | rt, rfl, h⟩ := h
  · simp only [Option.bind_some, hsp, ↓reduceIte, throw, throwThe, MonadExceptOf.throw, bind, Except.bind,
      reduceCtorEq] at h
  dsimp only at h
  rcases hreg : Json.parse rt with e | reg <;> rw [hreg] at h
  · cases h
  rcases bound with e | _
  · cases h
  rcases t with e | tt
  · cases h
  rcases htr : Json.parse tt with e | tr
  · dsimp only [pure, Except.pure, bind, Except.bind] at h
    rw [htr] at h
    cases h
  exact ⟨tt, tr, rt, reg, rfl, htr, rfl, hreg⟩

/-- **A statement checked as V*'s builds V*'s list**, whatever the verifier's reads. -/
theorem statements_of_checkStmt
    (h : RecVStar.checkStmt (some ish) tags tables st circuitB partition program registered part bindings rows =
      .ok ()) : (RecVStar.statements ish).isOk = true := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  repeat' split at h
  all_goals first
    | (obtain ⟨_, h1, -⟩ := ebind_ok h; cases h1)
    | cases h
    | skip
  all_goals
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨s, hs, -⟩ := ebind_ok h
    obtain ⟨ss, hss, -⟩ := check_ok (checkCircuit_ok hs).1
    simp [hss, Except.isOk, Except.toBool]

/-- **The inner files' statement** (`Files.stmt`) is `setupHidden`'s on the files, with no program. -/
theorem stmt_setupHidden {u : String} {spec' : SessionSpec} {st' : Stmt} (h : f.stmt u = .ok (spec', st')) :
    Stmt.setupHidden RecVStar.innerTags f.circuit f.pub Coins.os [] none (some f.partition) none = .ok st' := by
  unfold RecCoins.Files.stmt at h
  split at h
  · rename_i x hx
    obtain ⟨spec'', sts'⟩ := x
    rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hx with ⟨-, st'', hst, -, rfl⟩ | ⟨hc, -⟩
    · simp only [pure, Except.pure, bind, Except.bind, List.getElem?_toArray, List.getElem?_cons_zero] at h
      cases h
      exact hst
    · omega
  · cases h

/-- **One circuit and public file set up one statement's `c`, `m` and regions**, whatever the partition and program:
`Stmt.setupHidden` reads them off the circuit and the public file alone (`Hidden.Spec`). -/
theorem setupHidden_same {pf : ByteArray} {coins : Coins} {p p' : Option Json} {q q' : Option ByteArray}
    {st' : Stmt} (h : Stmt.setupHidden tags cf pf coins [] none p q = .ok st)
    (h' : Stmt.setupHidden tags cf pf coins [] none p' q' = .ok st') :
    st.c = st'.c ∧ st.m = st'.m ∧ st.regions = st'.regions := by
  have s := Hidden.setupHidden_spec h
  have s' := Hidden.setupHidden_spec h'
  have hc : st.c = st'.c := Except.ok.inj (s.parse.symm.trans s'.parse)
  obtain ⟨pop, hl, hp, -⟩ := s.pub
  obtain ⟨pop', hl', hp', -⟩ := s'.pub
  have hpub : st.pub = st'.pub := by
    rw [hp rfl, hp' rfl, hc] at *
    exact Except.ok.inj (hl.symm.trans hl')
  have hr := s.regions
  rw [hc, hpub] at hr
  refine ⟨hc, ?_, Except.ok.inj (hr.symm.trans s'.regions)⟩
  rw [s.m, s'.m, s.nbl, s'.nbl, s.blocksReal, s'.blocksReal, s.n, s'.n, hc, hpub]

/-- Check 5's range check reads the statement's circuit, `m` and region count alone. -/
theorem checkInnerRange_congr {st' : Stmt} (hc : st.c = st'.c) (hm : st.m = st'.m)
    (hr : st.regions = st'.regions) : RecCoins.checkInnerRange ish st = RecCoins.checkInnerRange ish st' := by
  have hu : RecCoins.unitRangeOf st = RecCoins.unitRangeOf st' := by unfold RecCoins.unitRangeOf; rw [hc]
  simp only [RecCoins.checkInnerRange, RecVStar.checkInnerStmt, hu, hc, hm, hr]

/-- **The bridge, at both setups**: where the verifier's candidate-3 checks on the inner files `f` pass
(`RecSetup.checkInner`), it holds the inner transcript `tt` against R's registration `rt` (`RecSetup.commits`, at their
parses `tr` and `reg`) to `cs`, and its checks of a session's statements `sts` pass there (`RecSetup.checks`), then at
the inner statement `st` set up from the files' circuit and public file, as `AcceptsStmt` sets it up, `AcceptsC3`
holds, check 5's verdict included (its sparse refusals and its inner shape). -/
theorem c3_of_checks (hsp : ish.sparse.isSome = true)
    (hpub : Stmt.setupTables RecVStar.innerTags f.circuit f.pub Coins.os 1 (partition := obj) (program := prog) =
      .ok (spec, #[st])) (hzk : ∃ p, Zk.schedule st.m = .ok p)
    (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (hcs : RecSetup.commits (some ish) (.ok tt) (.ok (some rt)) bound record = .ok cs)
    (hin : RecSetup.checkInner (some ish) (some (.files f)) bound = .ok ())
    (hck : RecSetup.checks (some ish) (some cs) tags tables sts circuitB partition program registered part bindings
      rows = .ok ()) (hsts : 0 < sts.size) :
    AcceptsC3 ish st tr record reg := by
  obtain ⟨hst, hcov⟩ := checks_ok hck
  obtain ⟨_, st', hst', hr, hm⟩ := checkInner_ok hsp hin
  obtain ⟨c, hc, hmb⟩ := checkMeta_ok hm
  rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hpub with ⟨-, st₀, hst₀, -, hsts₀⟩ | ⟨hc1, -⟩
  · obtain rfl : st = st₀ := by simpa using hsts₀
    obtain ⟨hcc, hmm, hrr⟩ := setupHidden_same hst₀ (stmt_setupHidden hst')
    rw [HiddenExec.setupHidden_parseC hst₀] at hc
    cases hc
    obtain ⟨d, sh, hb, hic⟩ := commits_ok htr hreg hcs
    obtain ⟨hsh, hd⟩ := metaBound_ok (hmb.trans hb)
    have hmr : metaRegistration st = d := by subst hd; rfl
    have hat : innerCommitsAt tr record st reg = .ok cs := by
      unfold innerCommitsAt
      rw [hsh, hmr]
      exact hic
    obtain ⟨hsr, hshape⟩ := sparseRangeOk_of_check (classOk_of_accepts ⟨⟨_, _, _, _, _, hpub⟩, hzk⟩)
      ((checkInnerRange_congr hcc hmm hrr).trans hr)
    unfold AcceptsC3
    refine ⟨statements_of_checkStmt (ZkExec.array_forM_ok hst 0 hsts), by simp [hat, Except.isOk, Except.toBool],
      fun cs' h => ?_, hsr, hshape⟩
    rw [hat] at h
    cases h
    simp [hcov, Except.isOk, Except.toBool]
  · omega

/-- **`verify` accepting a candidate-3 session gives `AcceptsC3`**, at its setup of the session (`Exec.setupOf`, as
`flock_verify_sound` reads it). -/
theorem acceptsC3_of_verify {I : Exec.Inputs} {r : String} {spec' : SessionSpec}
    (hset : Exec.setupOf I r = .ok (spec', sts)) (hsp : ish.sparse.isSome = true)
    (hpub : Stmt.setupTables RecVStar.innerTags f.circuit f.pub Coins.os 1 (partition := obj) (program := prog) =
      .ok (spec, #[st])) (hzk : ∃ p, Zk.schedule st.m = .ok p)
    (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (hcs : RecSetup.commits (some ish) (.ok tt) (.ok (some rt)) bound record = .ok cs)
    (hin : RecSetup.checkInner (some ish) (some (.files f)) bound = .ok ())
    (hck : RecSetup.checks (some ish) (some cs) tags tables sts circuitB partition program registered part bindings
      rows = .ok ()) :
    AcceptsC3 ish st tr record reg :=
  c3_of_checks hsp hpub hzk htr hreg hcs hin hck (HiddenExec.setupTables_setupOne hset).1

end FlockC3.VStarList
