import Proofs.Flock.Claims.Complete.ZkTables.Ready

/-!
# Completeness at any number of tables gives it at one

`honestP` and `honestPZk` send the same messages on every statement whose setup has one table (`honestP_one`), so
against any verifier their runs there are the same (`run_congr`). Where `readyZk` holds the setup has one table and
`δc H` is `δcZk H` (`δc_eq_δcZk`); elsewhere `δcZk H` is `1`. So `honestP`'s completeness with error `δc H` gives
`honestPZk`'s with error `δcZk H` (`completeZk_of`): the bound at several tables loses nothing at one. (The record's
prover, `(deployZk rel).P`, is `honestPR`, which runs `honestP` at the tape's mask words.)
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **Rounds depend on the prover only through its next messages at the statement and tape.** -/
theorem stepWith_congr {Ω : Type} (H : Claims.Msg → Claims.Msg) {P P' : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {vnext : List Claims.Msg → Claims.OC Claims.Out} (h : ∀ hist, P.next x hist ω = P'.next x hist ω) :
    ∀ n hist as t, Claims.stepWith H P x ω vnext n hist as t = Claims.stepWith H P' x ω vnext n hist as t
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
    simp only [Claims.stepWith, h hist, stepWith_congr H h n]

/-- **A run depends on the prover only through its next messages at the statement and tape.** -/
theorem run_congr {Ω Coin : Type} [Inhabited Coin] (H : Claims.Msg → Claims.Msg) {P P' : Claims.Prover Ω}
    (V : Claims.Verifier Coin) {x : Claims.Msg} {ω : Ω} (h : ∀ hist, P.next x hist ω = P'.next x hist ω)
    (r : Fin (V.k x) → Coin) : Claims.run H P V x ω r = Claims.run H P' V x ω r :=
  stepWith_congr H h _ _ _ _

/-- **Where `readyZk` holds the setup has one table.** -/
theorem readyZk_one {H : Claims.Msg → Claims.Msg} {x : Claims.Msg} (h : readyZk H x = true) :
    ∃ s spec sts, setupOf true x = some (s, spec, sts) ∧ sts.size = 1 := by
  unfold readyZk at h
  split at h
  · rename_i s spec sts hs
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨s, spec, sts, hs, h.1.1.1.1.1.1⟩
  · cases h

/-- **Completeness at any number of tables gives it at one**: if the honest masked prover on each table's share
(`honestP`) is complete with error `δc H` (`deployZk`'s), then `honestPZk` is complete with error `δcZk H`. -/
theorem completeZk_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h : CompleteWith H (deployZk rel) honestP (δc H)) : CompleteWith H (deployZk rel) honestPZk (δcZk H) := by
  intro h64 x w hs hr ω
  by_cases hz : readyZk H x = true
  · obtain ⟨s, spec, sts, hset, h1⟩ := readyZk_one hz
    rw [← δc_eq_δcZk hz]
    have hrun := fun r => run_congr H (deployZk rel).V (fun hist => honestP_one hset h1 w hist ω) r
    have h' : 1 - δc H x ≤ Claims.Pr fun r : Fin ((deployZk rel).V.k x) → Coin =>
        (Claims.run H (honestP w) (deployZk rel).V x ω r).out = .acc := h h64 x w hs hr ω
    simpa only [hrun] using h'
  · simp only [δcZk, hz, Bool.false_eq_true, ↓reduceIte, sub_self]
    exact Claims.Pr_nonneg _

end FlockComplete.ZkTables
