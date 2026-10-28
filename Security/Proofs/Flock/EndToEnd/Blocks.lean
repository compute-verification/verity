import Proofs.Flock.EndToEnd.Blocks.Statement
import Proofs.Flock.EndToEnd.Period

/-!
# Proof of the integrity audit per server-time block, and over a horizon of blocks

* `Flock.SecurityProofs.integrity_on`: `IntegrityHorizon` on a set `I` of the period's sampling blocks. The probability
  that every sampling block of `I` is accepted and `I`'s wrong units total at least `M` is at most
  `(1 − q)^M + Σ_{t ∈ I} (ε_ks,t + δ_link,t)`. It is `IntegrityHorizon`'s proof with every sampling block outside `I`
  weighing nothing: `period_union` over the sampling blocks of `I` accepted with a drawn unit wrong, and
  `period_horizon` with weight `|W_t|` at a sampling block of `I` and `0` elsewhere, where the sampling block's miss
  probability `≤ 1 = (1 − q)^0`.
* `IntegrityBlock`: `integrity_on` at block `b`'s sessions, with `M = ⌈ln(1/δ)/q⌉`.
* `IntegrityBlocks`: the union bound over the blocks (`value_exists_finset_le`), each block's by `IntegrityBlock`.
* `IntegrityHorizonBlocks`: `IntegrityHorizon` over the horizon's `start B` sessions, whose terms regroup by block
  (`sum_range_start`).
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

/-- The wrong units of the sampling blocks of `I`, as a sum over every sampling block weighing `0` outside `I`. -/
theorem sum_filter_wrong (I : Finset ℕ) (l : List WindowRec) :
    ((l.filter fun r => r.idx ∈ I).map WindowRec.wrong).sum =
      (l.map fun r => if r.idx ∈ I then r.wrong else 0).sum := by
  induction l with
  | nil => rfl
  | cons r l ih =>
      by_cases h : r.idx ∈ I
      · simp [h, ih]
      · simp [h, ih]

/-- **`IntegrityHorizon` on a set `I` of the period's sampling blocks**, at `M` wrong units. -/
theorem integrity_on (W : ℕ → SamplingBlock) (T : ℕ) (I : Finset ℕ) {q : ℝ} (hq : 0 < q)
    (hqt : ∀ t ∈ I, q ≤ ((W t).kd : ℝ) / (W t).n)
    (hA3 : ∀ t ∈ I, Assumptions.UniformRandomBytes (W t).Lb (W t).src)
    (s : Strategy (Period.game W T)) (hs : PeriodGood (fun t => (W t).game) (fun t => (W t).CR) 0 T s) (M : ℕ) :
    value (fun l => (∀ r ∈ l, r.idx ∈ I → r.accepted) ∧
        M ≤ ((l.filter fun r => r.idx ∈ I).map WindowRec.wrong).sum) (Period.run W T s) ≤
      ENNReal.ofReal (1 - q) ^ M + ∑ t ∈ I, ((W t).ks + (W t).link) := by
  set x : ℝ≥0∞ := ENNReal.ofReal (1 - q)
  have hx : x ≤ 1 := ENNReal.ofReal_le_one.2 (by linarith)
  -- some sampling block of `I` accepted with a drawn unit wrong
  have hA := period_union (fun t => (W t).game) (fun t => (W t).record t)
    (fun r => r.idx ∈ I ∧ r.accepted ∧ r.drawnWrong) (fun t τ => (W t).CR τ)
    (fun t => if t ∈ I then (W t).ks + (W t).link else 0)
    (fun t τ hτ => by
      by_cases ht : t ∈ I
      · rw [ite_eq_left_of_eq_true _ _ (eq_true ht)]
        exact (prob_mono (fun o ho => ho.2) _ _).trans ((W t).accepted_drawnWrong_le τ hτ)
      · rw [ite_eq_right_of_eq_false _ _ (eq_false ht)]
        exact (prob_mono (q := fun _ => False) (fun o ho => ht ho.1) _ _).trans_eq (prob_false _ _)) T 0 s hs
  -- every draw of `I` misses, with at least `M` wrong units in `I`
  have hB := period_horizon (fun t => (W t).game) (fun t => (W t).record t)
    (fun r => r.idx ∈ I → ¬ r.drawnWrong) (fun r => if r.idx ∈ I then r.wrong else 0)
    (fun t τ => (W t).CR τ) x hx
    (fun t τ _ => by
      by_cases ht : t ∈ I
      · refine ⟨((W t).wrongSet τ).card, fun _ => ite_eq_left_of_eq_true _ _ (eq_true ht), ?_⟩
        refine (prob_mono (q := fun o => ¬ ∃ u ∈ o.2.1, u ∈ (W t).wrongSet τ) (fun o ho => ho ht) _ _).trans ?_
        exact ((W t).drawMiss_le (hA3 t ht) _ τ).trans
          (pow_le_pow_left₀ zero_le ((W t).one_sub_le hq (hqt t ht)) _)
      · exact ⟨0, fun _ => ite_eq_right_of_eq_false _ _ (eq_false ht), (prob_le_one _ _ _).trans_eq (pow_zero x).symm⟩) T 0 s M hs
  calc value (fun l => (∀ r ∈ l, r.idx ∈ I → r.accepted) ∧
        M ≤ ((l.filter fun r => r.idx ∈ I).map WindowRec.wrong).sum) (Period.run W T s)
      ≤ value (fun l => (∃ r ∈ l, r.idx ∈ I ∧ r.accepted ∧ r.drawnWrong) ∨
          ((∀ r ∈ l, r.idx ∈ I → ¬ r.drawnWrong) ∧
            M ≤ (l.map fun r => if r.idx ∈ I then r.wrong else 0).sum)) (Period.run W T s) := by
        refine value_mono (fun l hl => ?_) _
        by_cases hd : ∃ r ∈ l, r.idx ∈ I ∧ r.drawnWrong
        · obtain ⟨r, hr, hrI, hdr⟩ := hd
          exact Or.inl ⟨r, hr, hrI, hl.1 r hr hrI, hdr⟩
        · push Not at hd
          exact Or.inr ⟨hd, (sum_filter_wrong I l) ▸ hl.2⟩
    _ ≤ value (fun l => ∃ r ∈ l, r.idx ∈ I ∧ r.accepted ∧ r.drawnWrong) (Period.run W T s) +
          value (fun l => (∀ r ∈ l, r.idx ∈ I → ¬ r.drawnWrong) ∧
            M ≤ (l.map fun r => if r.idx ∈ I then r.wrong else 0).sum) (Period.run W T s) :=
        value_or_le _ _ _
    _ ≤ (∑ t ∈ Finset.range T, if t ∈ I then (W t).ks + (W t).link else 0) + x ^ M := by
        gcongr
        · simpa [Period.run] using hA
        · exact hB
    _ ≤ x ^ M + ∑ t ∈ I, ((W t).ks + (W t).link) := by
        rw [add_comm, ← Finset.sum_filter]
        exact add_le_add le_rfl (Finset.sum_le_sum_of_subset fun t ht => (Finset.mem_filter.1 ht).2)

/-- The horizon's sessions, regrouped by block. -/
theorem sum_range_start (S : ServerBlocks) (f : ℕ → ℝ≥0∞) :
    ∀ B, ∑ t ∈ Finset.range (S.start B), f t = ∑ b ∈ Finset.range B, ∑ t ∈ S.sessions b, f t
  | 0 => by simp [S.start_zero]
  | B + 1 => by
      rw [Finset.sum_range_succ, ← sum_range_start S f B, ServerBlocks.sessions,
        Finset.sum_range_add_sum_Ico _ (S.mono (Nat.le_add_right B 1))]

/-- **The integrity audit of one server-time block.** -/
theorem IntegrityBlock : Flock.Guarantees.IntegrityBlock := by
  intro S b q δ hq hδ hqt hA3 s hT hL
  have hs : PeriodGood (fun t => (S.W t).game) (fun t => (S.W t).CR) 0 S.T s :=
    periodGood_of_plays _ _ S.T 0 s fun t τ hp => And.intro (hT t τ hp) (hL t τ hp)
  refine (value_mono (fun l hl => ⟨hl.1, Nat.ceil_le.2 hl.2⟩) _).trans
    ((integrity_on S.W S.T (S.sessions b) hq hqt hA3 s hs ⌈Real.log (1 / δ) / q⌉₊).trans ?_)
  exact add_le_add (one_sub_pow_le_of_log hq hδ (Nat.le_ceil _)) le_rfl

/-- **The integrity audit per server-time block, over a horizon of `B` blocks.** -/
theorem IntegrityBlocks : Flock.Guarantees.IntegrityBlocks := by
  intro S q δ hq hδ hqt hA3 s hT hL
  have hin : ∀ b < S.B, ∀ t ∈ S.sessions b, t < S.T := fun b hb t ht =>
    (Finset.mem_Ico.1 ht).2.trans_le (S.mono (Nat.succ_le_of_lt hb))
  calc value (fun l => ∃ b < S.B, S.BlockEscapes q δ b l) (S.run s)
      = value (fun l => ∃ b ∈ Finset.range S.B, S.BlockEscapes q δ b l) (S.run s) :=
        value_congr (fun l => by simp only [Finset.mem_range]) _
    _ ≤ ∑ b ∈ Finset.range S.B, value (S.BlockEscapes q δ b) (S.run s) := value_exists_finset_le _ _ _
    _ ≤ ∑ b ∈ Finset.range S.B, (ENNReal.ofReal δ + S.err b) := Finset.sum_le_sum fun b hb =>
        IntegrityBlock S b q δ hq hδ (fun t ht => hqt t (hin b (Finset.mem_range.1 hb) t ht))
          (fun t ht => hA3 t (hin b (Finset.mem_range.1 hb) t ht)) s hT hL

/-- **The integrity audit over a horizon of `B` server-time blocks, pooled.** -/
theorem IntegrityHorizonBlocks : Flock.Guarantees.IntegrityHorizonBlocks := by
  intro S q δ hq hδ hqt hA3 s hT hL
  refine (IntegrityHorizon S.W S.T q δ hq hδ hqt hA3 s hT hL).trans_eq ?_
  congr 1
  exact sum_range_start S (fun t => (S.W t).ks + (S.W t).link) S.B

end Flock.SecurityProofs
