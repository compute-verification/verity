import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Claims.Sampling.Reference.Complete

/-!
# Probe: the deployed sampling record's completeness off one table

`SamplingDeploy.deploy`'s `δc` is the sessions' count times the largest of Flock's masked error `δcZk`, capped at `1`,
over the statements of the reference prover's sessions at honest runs (`Reference.sessErrAt`). `δcZk` is `1` at a
masked statement whose setup has other than one table, as `readyZk` asks for one. So wherever an honest run's checked
unit has its session at such a statement, the record's `δc` is at least `1` and the sampling `Complete` asks nothing
there (`δc_deploy_tables`).
-/

namespace SamplingDeploy

open Claims Claims.Sampling

/-- **Probe: an honest run's session statement of other than one table puts the deployed sampling record's `δc` at `1`
or more.** -/
theorem δc_deploy_tables (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)
    (H : Msg → Msg) {x xS : Msg} {w ω r} {u : ℕ × ℕ} {vS : Seen} (hw : Rhon H (commit c) c.parse c.partition x w)
    (hu : u ∈ c.checked x r)
    (hsess : c.session x u (run H (Reference.refP c.step c.parse c.reads (commit c) c.wit w) c.V x ω r).seen =
      some (xS, vS))
    {s spec sts} (hs : FlockDeploy.setupOf true xS = some (s, spec, sts)) (h : sts.size ≠ 1) (hn : 1 ≤ c.sessions x) :
    1 ≤ (deploy rel c).δc H x := by
  have hz : FlockComplete.δcZk H xS = 1 := by
    have hr : FlockComplete.readyZk H xS ≠ true := by
      intro h'
      unfold FlockComplete.readyZk at h'
      rw [hs] at h'
      simp only [Bool.and_eq_true] at h'
      exact h (beq_iff_eq.1 h'.1.1.1.1.1.1)
    unfold FlockComplete.δcZk
    rw [if_neg hr]
  have he : min 1 ((deploy rel c).flock.δc H xS) ≤ Reference.sessErr H (deploy rel c) x :=
    Reference.le_sessErr H (deploy rel c) hw hu hsess
  have hz' : (deploy rel c).flock.δc H xS = 1 := hz
  rw [hz', min_self] at he
  show 1 ≤ (c.sessions x : ℝ) * Reference.sessErr H (deploy rel c) x
  have hn' : (1 : ℝ) ≤ c.sessions x := by exact_mod_cast hn
  nlinarith [hn', he]

end SamplingDeploy
