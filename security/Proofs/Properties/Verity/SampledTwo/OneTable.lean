module

public import Proofs.Flock.SpotFlock.Accepting
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
public import Proofs.Properties.Verity.SampledTwo.SessionCoins

/-!
# A member session sets up one table

The sessions' layout (`SessionCoins.slotStream`, `roundOf`) places a session's rounds at one table: the link's, then
each of the table's two rep streams with `SessionCoins.repsOf` rounds. That is the session Flock's masked verifier
runs only where the session's statement sets up one table. Where the held copies' two-stage design evaluates
(`TwoHolds`) and each session statement is in scope (`ScopeHeld`), it does:

* `oneTable_of_shape`: a statement in Flock's masked scope (`SpotFlock.zkOk`) whose list of table sizes
  (`SpotFlock.shapeOf`) has one entry sets up one table.
* `oneTable_of_sessionOk`: where the design evaluates, any statement in template `t`'s scope sets up one table, at a
  table count of at most one. The design evaluates only where each template's reference statement sets up one
  (`SpotFlock.ref_one_table`), a statement in scope has its reference statement's table sizes (`SpotFlock.sessionOk`'s
  shape clause), and the masked verifier sets a count above one up as that many tables
  (`HiddenExec.setupTables_okHP`).
* `oneTable_of_scope`: so template `t`'s session statement sets up one table.
* `rounds_of_oneTable`: a statement that sets up one table runs `2 + 2 · repRounds` rounds, so `SessionCoins.repsOf` of
  its round count is the rounds of each of its two reps.
* `reps_of_scope`: both at template `t`'s round count `SpotFlock.k₀`, where each session statement has its template's
  round count (`RoundsHeld`).
-/

@[expose] public section

namespace Verity.SampledTwo

open SpotFlock.RunsAs.Round2 (TwoHolds RoundsHeld ScopeHeld two_of_twoHolds)

/-- **A statement in Flock's masked scope whose list of table sizes has one entry sets up one table.** -/
theorem oneTable_of_shape {x : Claims.Msg} (hz : SpotFlock.zkOk x = true) (h1 : (SpotFlock.shapeOf x).length = 1) :
    ∃ s spec st, FlockDeploy.setupOf true x = some (s, spec, #[st]) := by
  obtain ⟨s, spec, sts, hset, -⟩ := SpotFlock.zkOk_sound hz
  have hsz : sts.size = 1 := by simpa [SpotFlock.shapeOf, hset] using h1
  obtain ⟨st, rfl⟩ := Array.size_eq_one_iff.1 hsz
  exact ⟨s, spec, st, hset⟩

/-- **A statement in a template's scope sets up one table, at a table count of at most one**: where the held copies'
two-stage design evaluates, a statement that passes template `t`'s scope check has the one table size of the
template's reference statement, and the masked verifier sets a table count above one up as that many tables. -/
theorem oneTable_of_sessionOk {h : SpotFlock.Held} {s d} (hd : SpotFlock.two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) {xS : Claims.Msg} (hs : SpotFlock.sessionOk h t xS = true) {ins : FlockDeploy.Ins}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts)) : sts.size = 1 ∧ ins.read.I.count ≤ 1 := by
  have h1 : (SpotFlock.shapeOf (h.xS₀ t)).length = 1 := (SpotFlock.ref_one_table h hd ht).1
  simp only [SpotFlock.sessionOk, Bool.and_eq_true] at hs
  have e : SpotFlock.shapeOf xS = SpotFlock.shapeOf (h.xS₀ t) := eq_of_beq hs.1.1.2
  rw [← e] at h1
  have hsz : sts.size = 1 := by simpa [SpotFlock.shapeOf, hset] using h1
  refine ⟨hsz, ?_⟩
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hread : FlockDeploy.readOf true xS = some ins.read := by simp [FlockDeploy.readOf, hins]
  have hh : ins.read.tags.hiddenOutputs = true := by simpa [SpotFlock.hiddenOf, hread] using hs.2
  have hrs := FlockDeploy.Ins.setup_read hsu
  unfold Flock.Readable.Read.setup at hrs
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hrs with ⟨hc, -⟩ | ⟨hc, hn, -⟩
  · exact hc
  · omega

/-- **A member session sets up one table**: where the held copies' two-stage design evaluates and each session
statement is in scope, template `t`'s session statement sets up exactly one table. -/
theorem oneTable_of_scope {h : SpotFlock.Held} {m₂ : Claims.Msg} (hT : TwoHolds h) (hS : ScopeHeld h m₂) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ s spec st, FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st]) := by
  obtain ⟨_, _, hd⟩ := two_of_twoHolds hT
  have hs := hS t ht
  have hz : SpotFlock.zkOk (SpotFlock.stmtAt h m₂ t) = true := by
    have hs' := hs
    simp only [SpotFlock.sessionOk, Bool.and_eq_true] at hs'
    exact hs'.1.1.1
  obtain ⟨s, spec, sts, hset, -⟩ := SpotFlock.zkOk_sound hz
  obtain ⟨st, rfl⟩ := Array.size_eq_one_iff.1 (oneTable_of_sessionOk hd ht hs hset).1
  exact ⟨s, spec, st, hset⟩

/-- **A statement that sets up one table runs `2 + 2 · repRounds` rounds**: the link's, each of the table's two
reps' and the last message's (`FlockDeploy.V`), so each rep has `SessionCoins.repsOf` of the statement's round
count. -/
theorem rounds_of_oneTable {x : Claims.Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    (hset : FlockDeploy.setupOf true x = some (s, spec, #[st])) :
    (FlockDeploy.V true).k x = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionCoins.repsOf ((FlockDeploy.V true).k x) = FlockDeploy.repRounds true spec st := by
  have hk : (FlockDeploy.V true).k x = 2 + FlockDeploy.rounds true spec #[st] := by simp only [FlockDeploy.V, hset]
  have hr : FlockDeploy.rounds true spec #[st] = 2 * FlockDeploy.repRounds true spec st := by
    simp [FlockDeploy.rounds]
  rw [hr] at hk
  refine ⟨hk, ?_⟩
  rw [hk]
  unfold SessionCoins.repsOf
  omega

/-- **Each rep of a member session has `SessionCoins.repsOf` of its template's round count**: where the held
copies' two-stage design evaluates, each session statement has its template's round count and each is in scope,
template `t`'s session statement sets up one table, and its `SpotFlock.k₀` rounds are the link's, the last message's
and each of the table's two reps'. -/
theorem reps_of_scope {h : SpotFlock.Held} {m₂ : Claims.Msg} (hT : TwoHolds h) (hR : RoundsHeld h m₂)
    (hS : ScopeHeld h m₂) {t : ℕ} (ht : t < h.ms.length) :
    ∃ s spec st, FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st]) ∧
      SpotFlock.k₀ h t = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionCoins.repsOf (SpotFlock.k₀ h t) = FlockDeploy.repRounds true spec st := by
  obtain ⟨s, spec, st, hset⟩ := oneTable_of_scope hT hS ht
  obtain ⟨hk, hreps⟩ := rounds_of_oneTable hset
  rw [hR t ht] at hk hreps
  exact ⟨s, spec, st, hset, hk, hreps⟩

end Verity.SampledTwo
