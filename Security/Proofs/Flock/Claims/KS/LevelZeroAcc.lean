import Proofs.Flock.Claims.KS.LevelZero
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Soundness.Discharge.Exec.Verdict
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-!
# Level 0's request first, at flock-verify's acceptance

`accepts_req0`: when `Accepts I t`, whether or not the outputs are hidden, at every table `j` and rep `r`, every
result of the rep's compiled game played with its proof file's messages and its stream's coins requests level 0 first,
at the positions of the coins of the stream's round `plainQueryRound` (`rep_refines_req0` at `verifyRep`'s acceptance;
a run has one result, `plays_det`), where the proof opens one level-0 row per position (`merkleCheck_rows`), and every
request has its level's lanes and path length in the model's schedule `schedOf sch` (`ReqOK`, `lvOf_schedOf`).
-/

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model Game

local notation "𝔽" => FlockLevel3.GF128

/-- A level of an executable schedule is the model schedule's. -/
theorem lvOf_schedOf {s : Flock.Schedule} {i : ℕ} (hi : i < s.levels.size) :
    lvOf (schedOf s) i = levelOf (s.levels.getD i default) := by
  simp [lvOf, schedOf, Array.getD, hi, List.getD_eq_getElem?_getD]

/-- **An accepted rep requests level 0 first** (§9–§14). When `Accepts I t`, at every table `j`,
its `fast100` schedule and rep `r < 2`, every result of the rep's compiled game,
played with its proof file's messages and its stream's coins, has level 0's request first, at the positions of the
coins of the stream's round `plainQueryRound` at the table's `m`, `kLog`, `k0` and claims, where the proof opens one
level-0 row per position, and every request has its level's lanes and path length in the model's schedule. -/
theorem accepts_req0 {I : Inputs} {t : Transcript} (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (r : ℕ), r < 2 →
        ∀ (s : StreamRec) (p : Proof) (pend : PendL 𝔽 Dig),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) =
            some s →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r]! = .ok p →
          Plays fOf (Model.repC execArith r (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) pend →
          (∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
            R.positions = (Flock.positions (sch.levels.getD 0 default)
              (s.rounds[Claims.Flock.plainQueryRound sts[j].m sts[j].c.kLog sch.k0
                (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length)]!).coins).toList ∧
            p.ligerito.initialProof.rows.size = R.positions.length) ∧
          ∀ R ∈ pend.reqs, R.lanes = (lvOf (schedOf sch) R.level).lanes ∧
            R.depth = Model.pathLen (lvOf (schedOf sch) R.level) := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, -, -, -, hall⟩ := verify_ok_tables hv
  obtain ⟨hp2, hmp⟩ := HiddenExec.setupTables_pointsAny hs
  have hpts := decode_points hsess
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  refine ⟨schs[j], hf, fun hwf hr hm r hr2 s p pend hsr hp hpl => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp j hj).symm)
  have h15 : 15 ≤ sts[j].m := by have : 22 ≤ sts[j].m := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims _ schs[j] hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < schs[j].levels.size, Accounting.ceilLog2 (schs[j].levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hrep : verifyRep spec (Setup.ofCircuit sts[j]) schs[j] sess j r t.proofs[2 * j + r]! = .ok c := by
    rcases r with _ | _ | r
    · exact h0
    · exact h1
    · omega
  obtain ⟨s', p', hs'', hp', -, -, -, -, hk⟩ := rep_refines_req0 (D := Dig) spec (Setup.ofCircuit sts[j])
    (stmtOf sts[j] hwf (regionsOf sts[j])) hm h15 schs[j] rfl rfl (ofCircuit_fold sts[j] hwf _) _ sess hext hdim hlv
    hα1 hsm (decode_roundsWF hsess) j r _ _ hrep
  obtain rfl : s' = s := Option.some.inj (hs''.symm.trans hsr)
  obtain rfl : p' = p := Except.ok.inj (hp'.symm.trans hp)
  obtain ⟨pend', hpl', -, hok, -, R, rest, hRe, hRl, hRp⟩ :=
    hk capDig (fun _ ℓ k => (((openAt p'.ligerito (schs[j].levels.size - 1) ℓ).1)[k]!.toList.map fOf, []))
      (fun _ _ _ _ => rfl)
  obtain rfl := ZkExec.plays_det hpl hpl'
  have hmc := (hok R (by rw [hRe]; exact List.mem_cons_self)).2.2.2
  rw [hRl] at hmc
  simp only [openAt, ↓reduceIte] at hmc
  refine ⟨⟨R, rest, hRe, hRl, hRp, by simpa using (merkleCheck_rows hmc).1⟩, fun R' hR' => ?_⟩
  obtain ⟨hlt, hl, hd, -⟩ := hok R' hR'
  rw [lvOf_schedOf hlt]
  exact ⟨hl, hd⟩

end FlockSoundness.Discharge.Exec
