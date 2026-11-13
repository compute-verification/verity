module

public import Proofs.Properties.Verity.SampledTwo.OneTable
public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Properties.Verity.Sampled.Layout
public import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP

/-!
# A member session's one table has its template's table name

The sessions' layout names a session's rep streams by its table (`SessionCoins.slotStream` at `tab t`). The table a
member session sets up is named by the held copies alone:

* `refTable h t`: template `t`'s table name, the table tag of the statement its reference session statement
  (`Held.xS₀`) names.
* `tables_of_pin`: the one table template `t`'s session statement sets up is `refTable h t`, where each session
  statement is in scope (`ScopeHeld`) and is its template's reference statement but for the run's fields (`PinHeld`).
  The masked verifier sets a statement with hidden outputs (`SpotFlock.sessionOk`'s tag) up as one table under its own
  tags (`HiddenExec.setupTables_okHP`, `Hidden.setupHidden_tagsP`), whose table tag is the statement's
  (`Verity.Sampled.zkTags_table`); the statement's tags are the ones its name names (`SpotFlock.ins_fields`), and the
  pin holds that name to the reference statement's.
* `session_of_held`: with `reps_of_scope`, template `t`'s session is one table named `refTable h t`, each of whose two
  reps has `SessionCoins.repsOf` of the template's round count.
-/

@[expose] public section

namespace Verity.SampledTwo

open SpotFlock.RunsAs.Round2 (TwoHolds RoundsHeld ScopeHeld PinHeld)
open FlockSoundness.Discharge

/-- **Template `t`'s table name**: the table tag of the statement its reference session statement names; empty where
the reference statement does not decode or names no statement the verifier knows. -/
def refTable (h : SpotFlock.Held) (t : ℕ) : String :=
  (((FlockDeploy.decode (h.xS₀ t)).bind fun w₀ => Flock.Tags.byName? w₀.statement).map fun tags => tags.table).getD ""

/-- `refTable` where the reference statement decodes and names a statement the verifier knows. -/
theorem refTable_eq {h : SpotFlock.Held} {t : ℕ} {w₀ : FlockDeploy.Raw} {tags : Flock.Tags}
    (hw₀ : FlockDeploy.decode (h.xS₀ t) = some w₀) (htags : Flock.Tags.byName? w₀.statement = some tags) :
    refTable h t = tags.table := by
  simp [refTable, hw₀, htags]

/-- **A member session's one table has its template's table name**: where each session statement is in scope and is
its template's reference statement but for the run's fields, the session template `t`'s session statement sets up
with one table has the tables `#[refTable h t]`. -/
theorem tables_of_pin {h : SpotFlock.Held} {m₂ : Claims.Msg} (hS : ScopeHeld h m₂) (hP : PinHeld h m₂) {t : ℕ}
    (ht : t < h.ms.length) {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    (hset : FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st])) :
    spec.tables = #[refTable h t] := by
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hread : FlockDeploy.readOf true (SpotFlock.stmtAt h m₂ t) = some s.read := by
    simp [FlockDeploy.readOf, hins]
  have hz : s.read.zk = true := FlockDeploy.readOf_zk hread
  have hs := hS t ht
  simp only [SpotFlock.sessionOk, Bool.and_eq_true] at hs
  have hh : s.read.tags.hiddenOutputs = true := by simpa [SpotFlock.hiddenOf, hread] using hs.2
  have hins' := hins
  unfold FlockDeploy.insOf at hins'
  obtain ⟨w, hw, -⟩ := Option.bind_eq_some_iff.1 hins'
  obtain ⟨s', hs', hrd⟩ := SpotFlock.ins_of_readOf hw hread
  obtain ⟨htag, -⟩ := SpotFlock.ins_fields hs'
  rw [hrd] at htag
  have hp := hP t ht w hw
  unfold SpotFlock.pinOk at hp
  cases hw₀ : FlockDeploy.decode (h.xS₀ t) with
  | none => simp [hw₀] at hp
  | some w₀ =>
    simp only [hw₀, decide_eq_true_eq] at hp
    have e1 : w.statement = w₀.statement := (congrArg FlockDeploy.Raw.statement hp :)
    rw [e1] at htag
    rw [refTable_eq hw₀ htag]
    have hT : s.read.tags = ZkExec.zkTags s.read.I := by
      unfold Flock.Readable.Read.tags
      rw [ite_eq_left hz]
    have hrs := FlockDeploy.Ins.setup_read hsu
    unfold Flock.Readable.Read.setup at hrs
    rcases HiddenExec.setupTables_okHP hh hrs with ⟨-, st', hst, rfl, -⟩ | ⟨h1, hsz, -⟩
    · show #[st'.tags.table] = #[s.read.I.tags.table]
      rw [Hidden.setupHidden_tagsP hst, hT, Verity.Sampled.zkTags_table]
    · have h2 : (#[st] : Array Flock.Stmt).size = 1 := rfl
      omega

/-- **A member session is one table of its template's name, each of whose two reps has `SessionCoins.repsOf` of the
template's round count**: where the held copies' two-stage design evaluates, and each session statement has its
template's round count, is in scope and is its template's reference statement but for the run's fields. -/
theorem session_of_held {h : SpotFlock.Held} {m₂ : Claims.Msg} (hT : TwoHolds h) (hR : RoundsHeld h m₂)
    (hS : ScopeHeld h m₂) (hP : PinHeld h m₂) {t : ℕ} (ht : t < h.ms.length) :
    ∃ s spec st, FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st]) ∧
      spec.tables = #[refTable h t] ∧
      SpotFlock.k₀ h t = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionCoins.repsOf (SpotFlock.k₀ h t) = FlockDeploy.repRounds true spec st := by
  obtain ⟨s, spec, st, hset, hk, hreps⟩ := reps_of_scope hT hR hS ht
  exact ⟨s, spec, st, hset, tables_of_pin hS hP ht hset, hk, hreps⟩

end Verity.SampledTwo
