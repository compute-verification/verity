import FlockSoundness.Discharge.ZkExec.Setup
import FlockSoundness.Discharge.HiddenExec.Accepts

/-!
# flock-verify `--zk`'s acceptance and the statement, hidden outputs included (#793's N9)

`Setup.lean`'s three lemmas without `hh`: under any tags, `Stmt.setupTables` sets each table up with
`HiddenExec.setupOne` (`Stmt.setupHidden` under hidden outputs, `Stmt.setupH` otherwise), at any table count and any
draw ([wave2-hidden-exec]'s `setupTables_okAny`, `setupTables_setupOne` and `setupTables_retainAny`, at `zkInputs I`).

* `acceptsZK_setupOne : AcceptsZKSetupOne`: `AcceptsZKSetupH` with `setupOne` for `setupH`, no `hh`.
* `acceptsZK_oneAny : AcceptsZKOneAny`: `AcceptsZKOne` with `setupOne` for `setupH`, no `hh`.
* `acceptsZK_retainedAny : AcceptsZKRetainedAny`: `AcceptsZKRetained`, verbatim, no `hh`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec HiddenExec

/-- **Each table of a `--zk` acceptance is `setupOne`'s statement**, under any tags: `AcceptsZKSetupH` without `hh`,
with `setupOne` (`setupHidden` under hidden outputs) for `setupH`. -/
def AcceptsZKSetupOne : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), AcceptsZK I t →
    ∃ spec sts, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧
      Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) t.record t.proofs = .ok () ∧ 0 < sts.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), Flock.Zk.shapeOk sts[j] = .ok () ∧ ∃ d,
        setupOne (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables d I.partition
          I.program = .ok sts[j] ∧ (I.count ≤ 1 → d = recordDraw t.record)

/-- **A one-table `--zk` acceptance is one `setupOne` statement's**, under any tags: `AcceptsZKOne` without `hh`, with
`setupOne` for `setupH`. -/
def AcceptsZKOneAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.count ≤ 1 → AcceptsZK I t →
    ∃ st, setupOne (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record) I.partition
        I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      Flock.Zk.verify st.spec #[Flock.Setup.ofCircuit st] t.record t.proofs = .ok ()

/-- **A `--zk` acceptance under retaining tags has every round's bytes**, under any tags: `AcceptsZKRetained` without
`hh`. -/
def AcceptsZKRetainedAny : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.retainRounds = true → AcceptsZK I t →
    ∃ spec sts sess, setupOf (zkInputs I) t.record = .ok (spec, sts) ∧ Record.decode spec t.record = .ok sess ∧
      ∀ s ∈ sess.streams.toList, AllRetained s.rounds

theorem acceptsZK_setupOne : AcceptsZKSetupOne := by
  intro I t h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨hpos, hall⟩ := setupTables_setupOne hs
  exact ⟨spec, sts, hs, hv, hpos, fun j hj => ⟨array_forM_ok hshape j hj, hall j hj⟩⟩

theorem acceptsZK_oneAny : AcceptsZKOneAny := by
  intro I t hc h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  rcases setupTables_okAny hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using array_forM_ok hshape 0 (by simp), by simpa using hv⟩
  · change 1 < I.count at hc'
    omega

theorem acceptsZK_retainedAny : AcceptsZKRetainedAny := by
  intro I t hr h
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  obtain ⟨-, sess, -, hd, -⟩ := zk_verify_ok _ _ _ _ hv
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hr
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

end FlockSoundness.Discharge.ZkExec
