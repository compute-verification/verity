import Proofs.Flock.Claims.Complete.Replay.Trace
import Proofs.Flock.Claims.Scope.StmtOf

/-!
# The deployed table as the statement the soundness proof plays

At the deployed relation the case's one table `st` is `Refine.stmtOf st hwf (regionsOf st)`, the statement whose
matrices (`matA`, `matB`) the circuit fold reads (`stmtOf_fold`), and the witness's bits satisfy it.

* `Readable.tableAt_model` (`Scope/StmtOf.lean`): each table the verifier parses (untyped, without public input ports) is
  well formed (`StmtWF`, `RegionsWF`), and its table's matrices (`placedA`, `placedB`, what `Tab.toModel` reads) are
  `stmtOf`'s (`matA`, `matB`).
* `Case.model`: at a good case of the deployed relation, `stmtOf st hwf (regionsOf st)` is satisfied by the witness's
  bits (`zOf 1 w 0`), from `RelS` (`relS_relation`).
-/

namespace FlockComplete.Replay

open Flock FlockDeploy FlockSoundness.Refine

variable {H : Claims.Msg → Claims.Msg}

/-- **The case's statement, satisfied**: at a good case of the deployed relation, the one table `st` is well formed,
its table's matrices are `stmtOf`'s, and the witness's bits satisfy `stmtOf st hwf (regionsOf st)`. -/
theorem _root_.FlockComplete.Case.model (c : Case H Readable.relation) :
    ∃ st, c.sts = #[st] ∧ ∃ hwf : StmtWF st, RegionsWF st ∧
      (∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = matA st a.val b.val ∧
        FlockLevel3.placedB st a b = matB st a.val b.val) ∧
      (stmtOf st hwf (regionsOf st)).Satisfies (Readable.zOf 1 c.w 0 st.m) := by
  obtain ⟨st, hsts, -, -, -⟩ := c.table
  obtain ⟨p, hp, hpe, hR⟩ := c.parsed
  obtain ⟨p', hp', hsc⟩ := (Readable.scope_iff (deployPlain Readable.relation) Readable.relation (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  rw [hp] at hp'
  obtain rfl : p = p' := Option.some.inj hp'
  have hP : Readable.Parses p := by
    unfold Readable.parseBytes at hp
    obtain ⟨r, -, hrp⟩ := Option.bind_eq_some_iff.1 hp
    exact Readable.parses_of_parse hrp
  obtain ⟨hc, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hRu := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  rw [hsts] at hpe
  have h0 : 0 < p.sts.size := by simp [hpe]
  obtain ⟨hf, hsat⟩ := (Readable.relS_relation p c.w).1 hR 0 h0
  have e0 : p.sts[0] = st := by simp [hpe]
  have e0' : p.sts[0]! = st := by simp [hpe]
  have e1 : p.sts.size = 1 := by simp [hpe]
  have key := Readable.tableAt_model (Readable.tablesOf_of_parses hP) hRu.2.1 hpi h0
  revert key hsat hf
  rw [e0, e0', e1]
  rintro hf hsat ⟨hwf, hrw, hmat⟩
  refine ⟨st, hsts, hwf, hrw, hmat, ?_⟩
  obtain ⟨h1, h2, h3⟩ := hsat
  have hA : ((Readable.Tab.ofStmt st).toModel hf.2.2).A₀ = (stmtOf st hwf (regionsOf st)).A₀ :=
    funext fun a => funext fun b => (hmat a b).1
  have hB : ((Readable.Tab.ofStmt st).toModel hf.2.2).B₀ = (stmtOf st hwf (regionsOf st)).B₀ :=
    funext fun a => funext fun b => (hmat a b).2
  refine ⟨fun o i => ?_, h2, h3⟩
  have := h1 o i
  rw [hA, hB] at this
  exact this

end FlockComplete.Replay
