import Proofs.Flock.Soundness.LigeritoOOD

/-!
# A level's folds against a list (PROTOCOL.md §13.2, §13.3 step 4)

`foldRounds` folds the lowest variable first, one `K` challenge per round, and the running claim
is a sum `Σ_y u(y)·v(y)` of the level's (folded) table `u` against the claim's weight `v`. For a
list of tables (the codewords near the level's commitment), a claim that is false for every one
of them stays false for every one after `k` rounds, except with probability `k·2L/|K|`
(`value_foldRounds_sum_le`): per round, each list element's true quadratic differs from the
verifier's, which sums to the claim by construction (§13.2). Together with the rounds' MCA events
(`value_foldRounds_mca_le`) this is the accounting's `Level.foldError`.
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- Fold the lowest variable by each challenge in turn (`foldRounds`' order). -/
def foldLoL {L : Type} [Field L] : List L → (ℕ → L) → (ℕ → L)
  | [], u => u
  | r :: rs, u => foldLoL rs (foldLo r u)

/-- The true quadratic of a round over `Σ_{x < 2^(n+1)} u(x)·v(x)`, lowest variable first. -/
noncomputable def roundPolyLo {L : Type} [Field L] (n : ℕ) (u v : ℕ → L) : L[X] :=
  ∑ y ∈ Finset.range (2 ^ n), (C (u (2 * y)) + C (u (2 * y + 1) + u (2 * y)) * Polynomial.X) *
    (C (v (2 * y)) + C (v (2 * y + 1) + v (2 * y)) * Polynomial.X)

theorem roundPolyLo_natDegree_le {L : Type} [Field L] (n : ℕ) (u v : ℕ → L) :
    (roundPolyLo n u v).natDegree ≤ 2 := by
  unfold roundPolyLo
  refine natDegree_sum_le_of_forall_le _ _ fun y _ => ?_
  have hlin : ∀ a b : L, (C a + C b * Polynomial.X : L[X]).natDegree ≤ 1 := fun a b =>
    (natDegree_add_le _ _).trans (max_le (by simp) ((natDegree_C_mul_le _ _).trans (by simp)))
  have h1 := hlin (u (2 * y)) (u (2 * y + 1) + u (2 * y))
  have h2 := hlin (v (2 * y)) (v (2 * y + 1) + v (2 * y))
  exact natDegree_mul_le.trans (by omega)

theorem roundPolyLo_eval {L : Type} [Field L] (n : ℕ) (u v : ℕ → L) (t : L) :
    (roundPolyLo n u v).eval t =
      ∑ y ∈ Finset.range (2 ^ n), foldLo t u y * foldLo t v y := by
  simp only [roundPolyLo, eval_finset_sum, eval_mul, eval_add, eval_C, eval_X, foldLo]
  exact Finset.sum_congr rfl fun y _ => by ring

theorem roundPolyLo_sum {L : Type} [Field L] [CharP L 2] (n : ℕ) (u v : ℕ → L) :
    (roundPolyLo n u v).eval 0 + (roundPolyLo n u v).eval 1 =
      ∑ x ∈ Finset.range (2 ^ (n + 1)), u x * v x := by
  rw [roundPolyLo_eval, roundPolyLo_eval, sum_range_pairs, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  have h1 : ∀ w : ℕ → L, foldLo 1 w y = w (2 * y + 1) := fun w => by
    simp only [foldLo, one_mul]
    rw [add_comm (w (2 * y + 1)), ← add_assoc, Rounds.add_self, zero_add]
  have h0 : ∀ w : ℕ → L, foldLo 0 w y = w (2 * y) := fun w => by simp [foldLo]
  rw [h1 u, h1 v, h0 u, h0 v]

/-- **A level's folds against a list.** If the incoming quadratic sums to the claim and the claim
is false for every table in the list, the folds end with a true claim for some list element with
probability at most `k·2L/|K|`, whatever the prover sends. -/
theorem value_foldRounds_sum_le (A : Arith F K) (hA : A.Correct) {ι : Type} (S : Finset ι)
    (u : ι → ℕ → K) (v : ℕ → K) :
    ∀ (k N : ℕ) (s : K) (q : Quad K), q.eval 0 + q.eval 1 = s →
      (∀ M ∈ S, s ≠ ∑ y ∈ Finset.range (2 ^ (k + N)), u M y * v y) →
      value (fun out : List K × K × Quad K => ∃ M ∈ S,
          out.2.1 = ∑ y ∈ Finset.range (2 ^ N), foldLoL out.1 (u M) y * foldLoL out.1 v y)
        (foldRounds A k s q) ≤ ((2 * k * S.card : ℕ) : ℝ≥0∞) / Fintype.card K
  | 0, N, s, q, _, hs => by
      show value _ (Game.ret ([], s, q)) ≤ _
      rw [value_ret, if_neg]
      · exact zero_le
      · rintro ⟨M, hM, h⟩
        exact hs M hM (by simpa [foldLoL] using h)
  | k + 1, N, s, q, hq, hs => by
      have := hA.charTwoK
      rw [foldRounds_succ]
      refine (value_drawK_le A hA _ _ (fun r => ∃ M ∈ S,
          q.eval r = (roundPolyLo (k + N) (u M) v).eval r)
          ((2 * S.card : ℕ) / Fintype.card K) (((2 * k * S.card : ℕ) : ℝ≥0∞) / Fintype.card K)
          ?_ fun r hr => ?_).trans ?_
      · -- one round: each list element's true quadratic differs from the verifier's
        calc prCoin (fun r => ∃ M ∈ S, q.eval r = (roundPolyLo (k + N) (u M) v).eval r)
            ≤ ∑ M ∈ S, prCoin (fun r => q.eval r = (roundPolyLo (k + N) (u M) v).eval r) :=
              prCoin_exists_le _ _
          _ ≤ ∑ _M ∈ S, (2 : ℝ≥0∞) / Fintype.card K := Finset.sum_le_sum fun M hM => by
              simp_rw [← quadPoly_eval]
              refine Rounds.sumRound_sound s _ _ (quadPoly_natDegree_le q)
                (roundPolyLo_natDegree_le _ _ _) (by rw [quadPoly_eval, quadPoly_eval, hq]) ?_
              rw [roundPolyLo_sum, show k + N + 1 = k + 1 + N by ring]
              exact hs M hM
          _ = ((2 * S.card : ℕ) : ℝ≥0∞) / Fintype.card K := by
              rw [Finset.sum_const, nsmul_eq_mul, ← mul_div_assoc]
              push_cast
              ring_nf
      · rw [value_send]
        refine iSup_le fun msg => ?_
        rw [value_bind_ret]
        refine (value_mono (fun out h => ?_) _).trans
          (value_foldRounds_sum_le A hA S (fun M => foldLo r (u M)) (foldLo r v) k N
            (q.eval r) (fromMsg msg (q.eval r)) (fromMsg_sum msg _) fun M hM h => ?_)
        · obtain ⟨M, hM, h⟩ := h
          exact ⟨M, hM, by simpa only [foldLoL] using h⟩
        · exact hr ⟨M, hM, by rw [h, roundPolyLo_eval]⟩
      · rw [ENNReal.div_add_div_same]
        gcongr
        push_cast
        ring_nf
        exact le_rfl

/-- The folds keep the quadratic summing to the claim. -/
theorem value_foldRounds_sum_invariant (A : Arith F K) (hA : A.Correct) :
    ∀ (k : ℕ) (s : K) (q : Quad K), q.eval 0 + q.eval 1 = s →
      value (fun out : List K × K × Quad K => out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1)
        (foldRounds A k s q) = 0
  | 0, s, q, hq => by
      show value _ (Game.ret ([], s, q)) = 0
      simp [hq]
  | k + 1, s, q, hq => by
      have := hA.charTwoK
      rw [foldRounds_succ]
      refine le_antisymm ((value_drawK_le A hA _ _ (fun _ => False) 0 0 (by simp [prCoin, avg])
        fun r _ => ?_).trans (by simp)) zero_le
      rw [value_send]
      refine iSup_le fun msg => ?_
      rw [value_bind_ret]
      exact (value_foldRounds_sum_invariant A hA k _ _ (fromMsg_sum msg _)).le

end FlockSoundness
