import Proofs.Flock.Claims.Sampling.Deployed

/-!
# Probe: the deployed sampling record's completeness off one table

`SamplingDeploy.deploy`'s `δc` is the proof units' count times the largest of Flock's masked error `δcZk`, capped at
`1`, over the sessions' statements (`Reference.sessErrAt`). `δcZk` is `1` at a masked statement whose setup has other
than one table, as `readyZk` asks for one. So at a code with a unit whose session's statement is such a statement, the
record's `δc` is at least `1` and the sampling `Complete` asks nothing there (`δc_deploy_tables`).
-/

namespace SamplingDeploy

open Claims Claims.Sampling

/-- **Probe: one session statement of other than one table puts the deployed sampling record's `δc` at `1` or more.** -/
theorem δc_deploy_tables (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code)
    (H : Msg → Msg) {x xS : Msg} {u : ℕ × ℕ} {v vS : Seen} (hsess : c.session x u v = some (xS, vS)) {s spec sts}
    (hs : FlockDeploy.setupOf true xS = some (s, spec, sts)) (h : sts.size ≠ 1) (hu : 1 ≤ units c x) :
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
  have hb : BddAbove (Set.range fun xS : {xS : Msg // ∃ u v vS, c.session x u v = some (xS, vS)} =>
      min 1 (FlockComplete.δcZk H xS.1)) := ⟨1, by rintro _ ⟨i, rfl⟩; exact min_le_left _ _⟩
  have he : 1 ≤ Reference.sessErrAt c.session (FlockComplete.δcZk H) x := by
    unfold Reference.sessErrAt
    refine le_trans ?_ (le_ciSup hb ⟨xS, u, v, vS, hsess⟩)
    simp [hz]
  show 1 ≤ (units c x : ℝ) * Reference.sessErrAt c.session (FlockComplete.δcZk H) x
  have hu' : (1 : ℝ) ≤ units c x := by exact_mod_cast hu
  nlinarith [hu', he]

end SamplingDeploy
