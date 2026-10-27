import Definitions.Nci.Guarantees.Fmt2
import Proofs.Nci.Fmt

/-!
# Proofs of NCI's guarantees with each value charged once over its copies

A thread's counted multiplications read each source at most `reuse` times over its copies, so they number at most
`reuse` per source `R`'s weights come from (`card_threadMults_le_sources`), and Theorem 1's counting
(`threaded_count_of`) runs on the sources as it ran on the weight gates. The fan-out bound and the distance from the
baseline are what `WeightsSupportFmt2` asks of the circuit before it claims anything; the counting reads neither. The
containment and the state carry over as in `Proofs.Nci.Fmt`.
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- A thread's counted multiplications in `R` number at most `reuse` per source of the weights `R` uses. -/
theorem card_threadMults_le_sources {c : Declaration} (T : Threading c) (Src : T.Sources)
    (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    (T.threadMults R t).card ≤ T.reuse * (Src.sourcesUsed R).card := by
  classical
  have hsub : T.threadMults R t ⊆ (Src.sourcesUsed R).biUnion fun s =>
      T.M.filter fun g => Src.src (T.weight g) = s ∧ T.thread (T.act g) = some t := by
    intro g hg
    rw [Threading.threadMults, mem_filter] at hg
    exact mem_biUnion.2 ⟨Src.src (T.weight g), mem_image_of_mem Src.src (mem_image_of_mem T.weight hg.1),
      mem_filter.2 ⟨(mem_inter.1 hg.1).1, rfl, hg.2⟩⟩
  calc (T.threadMults R t).card ≤ ∑ s ∈ Src.sourcesUsed R,
        (T.M.filter fun g => Src.src (T.weight g) = s ∧ T.thread (T.act g) = some t).card :=
        (card_le_card hsub).trans card_biUnion_le
    _ ≤ ∑ _s ∈ Src.sourcesUsed R, T.reuse := sum_le_sum fun s _ => Src.reuse_le s t
    _ = T.reuse * (Src.sourcesUsed R).card := by rw [sum_const, smul_eq_mul, mul_comm]

/-- **Theorem 1 with reads counted by information, each value charged once over its copies.** -/
theorem threaded_reads_info_fmt2 (c : Declaration) (θv : c.D.Transcript) (T : Threading c) (Src : T.Sources)
    (Γ : ℕ) (S b_w b_a : ℝ) (d e n k₀ : ℕ) (hbw : 0 < b_w) (hba : 0 < b_a) (hd : 0 < d) (he : 0 < e)
    (hF : Src.FanOutBound b_a e n) (hk : Src.OffFresh k₀)
    (hW : Assumptions.WeightsSupportFmt2 c θv T Src Γ S b_w b_a d e n k₀)
    (R : Finset (Fin c.D.N)) (hR : SmallInfo c θv Γ S R) :
    b_a * (T.mults R).card ≤ novelInfo c θv R * T.bound S b_w d e :=
  threaded_count_of T R (T.work d e R) _ (card_threadMults_le_sources T Src R) (fun t => work_nonneg T d e R t)
    hbw hba hd (fun t h => work_threaded T d e he R t h) ((le_novelInfo_iff c θv R _).2 (hW hF hk R hR)) hR.2.2

theorem ThreadedReadsInfoFmt2 : Guarantees.ThreadedReadsInfoFmt2 := threaded_reads_info_fmt2

theorem ThreadedContainedInfoFmt2 : Guarantees.ThreadedContainedInfoFmt2 := by
  intro c d M θc θv T Src Γ K b_w b_a dd e n k₀ hθ hv hbw hba hdd he hF hk hW W hWc hΓ hK
  calc b_a * ((T.mults (univ.filter fun g => M.emb g ∈ W)).card : ℝ)
      ≤ novelInfo c θc (univ.filter fun g => M.emb g ∈ W) * T.bound K b_w dd e :=
        threaded_reads_info_fmt2 c θc T Src Γ K b_w b_a dd e n k₀ hbw hba hdd he hF hk hW _
          (smallInfo_piece M hθ hv hWc hΓ hK)
    _ ≤ novelInfo d θv W * T.bound K b_w dd e :=
        mul_le_mul_of_nonneg_right (novelInfo_le_of_containment M hθ hv hWc) (le_max_of_le_left (Nat.cast_nonneg _))

theorem ThreadedOfStateFmt2 : Guarantees.ThreadedOfStateFmt2 := by
  intro c d M θc θv T Src Γ b_w b_a dd e n k₀ W Sst Sin k f hθ hv hbw hba hdd he hF hk hW hWc hΓ hin hf
  have hK := (InfoOfState d θv W Sst Sin k f hWc hin hf).2
  exact (ThreadedContainedInfoFmt2 c d M θc θv T Src Γ _ b_w b_a dd e n k₀ hθ hv hbw hba hdd he hF hk hW W hWc hΓ
    hK).trans (mul_le_mul_of_nonneg_right hK (le_max_of_le_left (Nat.cast_nonneg _)))

end Nci.SecurityProofs
