import Proofs.Flock.Soundness.Discharge.ZkExec.Verify

/-!
# flock-verify `--zk`'s acceptance sets up each table with `Stmt.setupH`

`AcceptsZK I t` sets up the session with `Stmt.setupTables` at `zkInputs I`, whose tags differ from `--statement`'s
only in the identity (`zkTags_eq`), so `Exec`'s facts about `setupTables` apply at `zkInputs I` (under `hh`, as
there):

* `acceptsZK_setupH : AcceptsZKSetupH`: every table is `setupH`'s statement under `tagsAt (zkTags I) J j` and the coin
  tree, and passes `Zk.shapeOk`.
* `acceptsZK_one : AcceptsZKOne`: for one table, at the record's draw, with `Zk.verify` on `#[Setup.ofCircuit st]`.
* `acceptsZK_retained : AcceptsZKRetained`: under tags that retain rounds, every stream of the record retained every
  round's bytes.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec

theorem list_forM_ok {α ε : Type} {f : α → Except ε PUnit} :
    ∀ (L : List α), forM L f = .ok () → ∀ a ∈ L, f a = .ok ()
  | [], _ => by simp
  | a :: L, h => by
      rw [List.forM_cons] at h
      obtain ⟨u, hu, h⟩ := ebind_ok h
      intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hu
      · exact list_forM_ok L h b hb

/-- Each element passes a check that `Array.forM` accepted. -/
theorem array_forM_ok {α ε : Type} {f : α → Except ε PUnit} {xs : Array α} (h : xs.forM f = .ok ()) :
    ∀ (j : ℕ) (hj : j < xs.size), f xs[j] = .ok () := by
  rcases xs with ⟨l⟩
  have hl : forM l f = .ok () := by simpa using h
  intro j hj
  exact list_forM_ok l hl _ (List.getElem_mem (by simpa using hj))

theorem acceptsZK_setupH : AcceptsZKSetupH := by
  intro I t hh h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  obtain ⟨hpos, hall⟩ := setupTables_setupH hh' hs
  exact ⟨spec, sts, hs, hv, hpos, fun j hj => ⟨array_forM_ok hshape j hj, hall j hj⟩⟩

theorem acceptsZK_one : AcceptsZKOne := by
  intro I t hh hc h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  rcases setupTables_ok hh' hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, by simpa using array_forM_ok hshape 0 (by simp), by simpa using hv⟩
  · change 1 < I.count at hc'
    omega

theorem acceptsZK_retained : AcceptsZKRetained := by
  intro I t hh hr h
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  obtain ⟨-, sess, -, hd, -⟩ := zk_verify_ok _ _ _ _ hv
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hr
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

end FlockSoundness.Discharge.ZkExec
