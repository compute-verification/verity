import Proofs.Flock.Claims.KS.PlainEvent
import Proofs.Flock.Claims.Sampling.Deployed

/-!
# An accepted run of the deployed `--zk` verifier is flock-verify `--zk`'s acceptance

The claims form's Flock verifier is `V true` (`deployZk`'s and `deployZkTrim`'s `V`, and so the sessions' verifier of
the sampling record `SamplingDeploy.deploy`). Where its run accepts against SHA-512 (`zkAccepts_of_run`), it decided
at the statement's inputs `s` (`insOf true x = some s`, what flock-verify reads of the statement), on the last message's
record and proof files `f`, and:

* the record's draw is the statement's as a value (`Custody.recordDraw_of_checks`: the verifier rebuilds both,
  `Flock.norm`);
* flock-verify `--zk` accepts that transcript at the statement's inputs (`AcceptsZK s.read.I ⟨f.record, f.proofs⟩`, and
  so `LiveAcceptsZK` at every outcome carrying it).

This is the `--zk` side of `plainVerdict_of_run`. Equality of the draw as a value is what the setup reads: `Exec.setupOf`
sets the record up at the record's own draw (`Custody.setupOf_record`).
-/

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody
open Lean (Json)

theorem deployZkTrim_V (rel : Readable.Parsed → Readable.Relation) : (FlockDeploy.deployZkTrim rel).V = V true := rfl

theorem deploy_flock_V (rel : Readable.Parsed → Readable.Relation) (c : SamplingDeploy.Code) :
    (SamplingDeploy.deploy rel c).flock.V = V true := rfl

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance**: against SHA-512, the run decided at the
statement's inputs `s` on its last message's record and proof files `f`; the record's draw is the statement's, and
`verify --zk` accepts `⟨f.record, f.proofs⟩` at `s.read.I`. -/
theorem zkAccepts_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      insOf true x = some s ∧
      Exec.recordDraw f.record = s.read.draw ∧ ZkExec.AcceptsZK s.read.I ⟨f.record, f.proofs⟩ := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, -⟩ := acc_decide h
  have hacc := run_acc_next sha512 P (V true) x ω r h
  replace hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc
  simp only [ite_true] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide hact)
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hdraw := Claims.Bridge.Custody.recordDraw_of_checks hins hc
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, hins, hdraw, ?_⟩
  exact ⟨spec, sts, Claims.Bridge.Custody.setupOf_zkInputs (Ins.setup_read hs) hzk
    (Claims.Bridge.Custody.publicInputs_none_of_checks hc) hdraw, Claims.Bridge.Custody.setup_shapeOk hs hzk, hv⟩

/-- **`zkAccepts_of_run` as the live event**: `LiveAcceptsZK` at the statement's inputs, at every outcome whose
transcript is the last message's record and proof files. -/
theorem liveAcceptsZK_of_run {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : FlockDeploy.Ins) (f : FlockDeploy.Final), insOf true x = some s ∧
      Exec.recordDraw f.record = s.read.draw ∧ ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript)) := by
  obtain ⟨s, -, -, -, -, -, f, -, hins, hdraw, hacc⟩ := zkAccepts_of_run h
  exact ⟨s, f, hins, hdraw, fun _ _ => hacc⟩

end FlockDeploy.KS
