import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Soundness.Discharge.Integrate.Bridge

/-!
# Each table the verifier parses is `stmtOf`'s

`tableAt_model`: each table the verifier parses (untyped, without public input ports) is well formed (`StmtWF`,
`RegionsWF`), and its table's matrices (`placedA`, `placedB`, what `Tab.toModel` reads) are `stmtOf`'s (`matA`, `matB`;
`placedA_eq`, on the public-output path and the hidden one).
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge Exec

/-- **Each table the verifier parses is `stmtOf`'s**: well formed, and its matrices are `stmtOf`'s. -/
theorem tableAt_model {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) {j : ℕ} (hj : j < sts.size) :
    Refine.StmtWF sts[j] ∧ Refine.RegionsWF sts[j] ∧
      ∀ a b : Fin (2 ^ sts[j].c.kLog), FlockLevel3.placedA sts[j] a b = Refine.matA sts[j] a.val b.val ∧
        FlockLevel3.placedB sts[j] a b = Refine.matB sts[j] a.val b.val := by
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
    obtain ⟨hwf, hrw⟩ := Refine.setupH_wf hd
    have P := parse_facts (setupH_spec ht' hd).1
    exact ⟨hwf, hrw, fun a b => ⟨Integrate.placedA_eq P (setupH_layout ht' hd) (Integrate.setupH_leaf hd) a b,
      Integrate.placedB_eq P (setupH_layout ht' hd) (Integrate.setupH_leaf hd) a b⟩⟩
  · rw [hh] at hd
    simp only [↓reduceIte] at hd
    obtain ⟨hwf, hrw⟩ := Hidden.setupHidden_wf hd
    have s := Hidden.setupHidden_spec' hd
    have hC := Hidden.committed_of_commit s.parse
    have hW := hC.netWidths ht'
    have hK := hC.netKinds ht'
    have L := Hidden.hidden_layout' ht' hd (HiddenExec.setupTables_pinnedH hh hset j hj)
    exact ⟨hwf, hrw, fun a b => ⟨Integrate.placedA_eqW hW hK L s.leaf a b, Integrate.placedB_eqW hW hK L s.leaf a b⟩⟩

end Flock.Readable
