import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP

/-!
# flock-verify's acceptance at the verifier's public inputs

`Exec.setupOf` sets the session up with no `--public-inputs`; `Exec.setupOfP I own` passes `own`, the verifier's own
recomputation of the public input words (`HmIn.checkOwn`), and `Exec.AcceptsP` is `Exec.Accepts` at it. At `own = none`
both are the deployed ones (`setupOfP_none`, `acceptsP_none`).

`setupTables_okHP` is `setupTables_okH` at any `own`, by the same walk; `setupTables_specP` puts it together with
`Hidden.setupHidden_specP`: every table of a hidden-output session accepted at `own` is its `SpecP`, at its circuit's
public input ports.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock

/-- The session's statement flock-verify sets up for a record with `own` as `--public-inputs`. -/
def setupOfP (I : Inputs) (own : Option ByteArray) (record : String) : Flock.V (Flock.SessionSpec × Array Flock.Stmt) :=
  Flock.Stmt.setupTables I.tags I.circuitFile I.publicFile I.coins I.count I.tables (recordDraw record) I.partition I.program
    I.work I.stratified I.closure own

/-- **flock-verify accepts the transcript at the verifier's public inputs `own`**. -/
def AcceptsP (I : Inputs) (own : Option ByteArray) (t : Transcript) : Prop :=
  ∃ spec sts, setupOfP I own t.record = .ok (spec, sts) ∧
    Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok ()

theorem setupOfP_none (I : Inputs) (record : String) : setupOfP I none record = setupOf I record := rfl

theorem acceptsP_none (I : Inputs) (t : Transcript) : AcceptsP I none t ↔ Accepts I t := Iff.rfl

end FlockSoundness.Discharge.Exec

namespace FlockSoundness.Discharge.HiddenExec

open Flock Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program own : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

set_option maxHeartbeats 4000000 in
/-- **`setupTables` sets up each table with `setupHidden` at the verifier's public inputs** under a tag with hidden
outputs (`setupTables_okH` at any `own`). -/
theorem setupTables_okHP (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) :
    (count ≤ 1 ∧ ∃ st, Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st ∧
        spec = st.spec ∧ sts = #[st]) ∨
      (1 < count ∧ sts.size = count ∧ (∀ (j : ℕ) (hj : j < sts.size), ∃ d,
        Stmt.setupHidden { tags with identity := tags.tableIdentity count j } cf pf coins tables (some d) partition
          program own = .ok sts[j]) ∧ (∀ (j : ℕ) (hj : j < sts.size), sts[j].mPts = spec.mPts) ∧
          spec.points = 2) := by
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
  · exact Or.inl ⟨by assumption, _, ok_of_refuse (by assumption), rfl, rfl⟩
  all_goals
    obtain ⟨hsz, hall⟩ := range_mapM_ok (by assumption)
    exact Or.inr ⟨by omega, hsz, fun j hj => ⟨_, ok_of_refuse (hall j hj)⟩, mPts_of_any (by assumption), rfl⟩

/-- **Every table of a hidden-output session accepted at the verifier's public inputs is its `SpecP`**: under the
table's tags (`tagsAt`), at some draw (the session's, for one table), and at its circuit's public input ports. -/
theorem setupTables_specP (hh : tags.hiddenOutputs = true)
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure own =
      .ok (spec, sts)) :
    0 < sts.size ∧ ∀ (j : ℕ) (hj : j < sts.size), ∃ d pi,
      Hidden.SpecP (tagsAt tags count j) cf pf coins tables d own pi sts[j] ∧ (count ≤ 1 → d = draw) := by
  rcases setupTables_okHP hh h with ⟨hc, st, hst, -, rfl⟩ | ⟨hc, hsz, hall, -⟩
  · refine ⟨by simp, fun j hj => ?_⟩
    have hj0 : j = 0 := by simpa using hj
    subst hj0
    obtain ⟨pi, hp⟩ := Hidden.setupHidden_specP hst
    have ht : tagsAt tags count 0 = tags := ite_eq_left hc
    exact ⟨draw, pi, by rw [ht]; exact hp, fun _ => rfl⟩
  · refine ⟨by omega, fun j hj => ?_⟩
    obtain ⟨d, hd⟩ := hall j hj
    obtain ⟨pi, hp⟩ := Hidden.setupHidden_specP hd
    have ht : tagsAt tags count j = { tags with identity := tags.tableIdentity count j } :=
      ite_eq_right (by omega)
    exact ⟨some d, pi, by rw [ht]; exact hp, fun h1 => absurd h1 (by omega)⟩

/-- **flock-verify's acceptance at the verifier's public inputs, table by table**: under hidden outputs, when
`AcceptsP I own t`, every table's statement is its `SpecP` at `own`. -/
theorem acceptsP_specP {I : Inputs} {own : Option ByteArray} {t : Exec.Transcript} (hh : I.tags.hiddenOutputs = true)
    (h : AcceptsP I own t) :
    ∃ spec sts, setupOfP I own t.record = .ok (spec, sts) ∧
      Flock.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ d pi,
        Hidden.SpecP (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables d own pi sts[j] ∧
          (I.count ≤ 1 → d = recordDraw t.record) := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  exact ⟨spec, sts, hs, hv, setupTables_specP hh hs⟩

end FlockSoundness.Discharge.HiddenExec
