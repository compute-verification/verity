import Flock.RecSetup
import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup

/-!
# Piece (h): `AcceptsC3` from the verifier's acceptance

Both of `FlockVerify.lean`'s setups of a candidate-3 session, `verify`'s `setupOf` and `statement`'s, hold the inner
transcript with `RecSetup.commits` and check the session with `RecSetup.checks`: META's binding (`checkMeta`), each
statement as V*'s (`RecVStar.checkStmt`), and the covers check (`RecVStar.checkCovers`). So:

* `c3_of_checks`: where both pass at a candidate-3 shape (`ish.sparse`) and the inner statement `st` sets up from the
  checked circuit, V*'s list builds, the verifier's hold of the inner transcript is `innerCommitsAt`'s, and the covers
  check passes at the commits it returns. It reads a session with R's registration (`--rec-inner-registration`).
* `acceptsC3_of_verify`: at `verify`'s setup (`Exec.setupOf`), that is `AcceptsC3` but `SparseRangeOk`, check 5's
  verdict, which the event keeps as its own conjunct (gap list).
-/

namespace FlockC3.VStarList

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockSoundness.Refine (ebind_ok)
open Flock
open Lean (Json)

variable {ish : RecVStar.Inner} {cs : RecVStar.InnerCommits} {cf pf : ByteArray} {obj partition : Option Json}
  {prog program : Option ByteArray} {spec : SessionSpec} {st : Stmt} {tags : Tags} {tables : List (String × ByteArray)}
  {bound : Except String (Option String × Option RecVStar.StackedShape)} {sts : Array Stmt} {circuitB : ByteArray}
  {registered rows : Option (Array Registered.Port)} {part : Option ℕ} {bindings : Option VStar.Bindings}
  {tt rt record : String} {tr reg : Json}

theorem boundKey_inj {a b : Option String × Option RecVStar.StackedShape}
    (h : RecSetup.boundKey a = RecSetup.boundKey b) : a = b := by
  obtain ⟨a1, _ | ⟨am, ac⟩⟩ := a <;> obtain ⟨b1, _ | ⟨bm, bc⟩⟩ := b <;> cases h <;> rfl

/-- **META's check**: the inner circuit parses, and its META binds what its header's does. -/
theorem checkMeta_ok (h : RecSetup.checkMeta (some cf) bound = .ok ()) :
    ∃ c, HmOut.parse RecVStar.innerTags cf = .ok c ∧ RecSetup.metaBound c.metaJ = bound := by
  unfold RecSetup.checkMeta at h
  dsimp only at h
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

theorem checks_ok (hsp : ish.sparse.isSome = true)
    (h : RecSetup.checks (some ish) (some cs) (some cf) bound tags tables sts circuitB partition program registered part
      bindings rows = .ok ()) :
    RecSetup.checkMeta (some cf) bound = .ok () ∧
      sts.forM (RecVStar.checkStmt (some ish) tags tables · circuitB partition program registered part bindings rows) =
        .ok () ∧ RecVStar.checkCovers ish cs = .ok () := by
  unfold RecSetup.checks at h
  simp only [hsp, ite_true] at h
  obtain ⟨⟨⟩, h1, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, h2, h⟩ := ebind_ok h
  exact ⟨h1, h2, h⟩

theorem commits_ok (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (h : RecSetup.commits (.ok tt) (.ok (some rt)) bound record = .ok cs) :
    ∃ d sh, bound = .ok (d, sh) ∧ RecVStar.innerCommits tr record (some reg) d sh = .ok cs := by
  unfold RecSetup.commits at h
  cases hb : bound with
  | error e => simp [hb, bind, Except.bind, pure, Except.pure, hreg] at h
  | ok b =>
    obtain ⟨d, sh⟩ := b
    refine ⟨d, sh, rfl, ?_⟩
    simpa [hb, bind, Except.bind, pure, Except.pure, hreg, htr] using h

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

/-- **The bridge, at both setups**: where the verifier holds the inner transcript `tt` against R's registration `rt`
(`RecSetup.commits`, at their parses `tr` and `reg`) to `cs`, and its candidate-3 checks of a session's statements `sts`
pass (`RecSetup.checks`) at the inner circuit `cf` that the inner statement `st` sets up from, as `AcceptsPublic`
does, V*'s list builds, `innerCommitsAt` returns `cs`, and the covers check passes there. -/
theorem c3_of_checks (hsp : ish.sparse.isSome = true)
    (hpub : Stmt.setupTables RecVStar.innerTags cf pf Coins.os 1 (partition := obj) (program := prog) =
      .ok (spec, #[st]))
    (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (hcs : RecSetup.commits (.ok tt) (.ok (some rt)) bound record = .ok cs)
    (hck : RecSetup.checks (some ish) (some cs) (some cf) bound tags tables sts circuitB partition program registered
      part bindings rows = .ok ()) (hsts : 0 < sts.size) :
    (RecVStar.statements ish).isOk = true ∧ innerCommitsAt tr record st reg = .ok cs ∧
      (RecVStar.checkCovers ish cs).isOk = true := by
  obtain ⟨hm, hst, hcov⟩ := checks_ok hsp hck
  obtain ⟨c, hc, hmb⟩ := checkMeta_ok hm
  rcases HiddenExec.setupTables_okH innerTags_hiddenOutputs hpub with ⟨-, st', hst', -, hsts'⟩ | ⟨hc1, -⟩
  · obtain rfl : st = st' := by simpa using hsts'
    rw [HiddenExec.setupHidden_parseC hst'] at hc
    cases hc
    obtain ⟨d, sh, hb, hic⟩ := commits_ok htr hreg hcs
    obtain ⟨hsh, hd⟩ := metaBound_ok (hmb.trans hb)
    have hmr : metaRegistration st = d := by subst hd; rfl
    refine ⟨statements_of_checkStmt (ZkExec.array_forM_ok hst 0 hsts), ?_, by simp [hcov, Except.isOk, Except.toBool]⟩
    unfold innerCommitsAt
    rw [hsh, hmr]
    exact hic
  · omega

/-- **`verify` accepting a candidate-3 session gives `AcceptsC3`**, at its setup of the session (`Exec.setupOf`, as
`flock_verify_sound` reads it), but for check 5's verdict (`SparseRangeOk`, `hsr`), which the bridge does not
conclude. -/
theorem acceptsC3_of_verify {I : Exec.Inputs} {r : String} {spec' : SessionSpec}
    (hset : Exec.setupOf I r = .ok (spec', sts)) (hsp : ish.sparse.isSome = true)
    (hpub : Stmt.setupTables RecVStar.innerTags cf pf Coins.os 1 (partition := obj) (program := prog) =
      .ok (spec, #[st]))
    (htr : Json.parse tt = .ok tr) (hreg : Json.parse rt = .ok reg)
    (hcs : RecSetup.commits (.ok tt) (.ok (some rt)) bound record = .ok cs)
    (hck : RecSetup.checks (some ish) (some cs) (some cf) bound tags tables sts circuitB partition program registered
      part bindings rows = .ok ()) (hsr : SparseRangeOk ish st) :
    AcceptsC3 ish st tr record reg := by
  obtain ⟨h1, h2, h3⟩ := c3_of_checks hsp hpub htr hreg hcs hck (HiddenExec.setupTables_setupOne hset).1
  unfold AcceptsC3
  refine ⟨h1, by simp [h2, Except.isOk, Except.toBool], fun cs' h => ?_, hsr⟩
  rw [h2] at h
  cases h
  exact h3

end FlockC3.VStarList
