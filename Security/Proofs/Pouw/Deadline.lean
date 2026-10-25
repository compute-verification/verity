import Models.Pouw.Guarantees.Deadline
import Proofs.Pouw.Game
import Proofs.Pouw.Prob
import Proofs.Pouw.PearlC.DeviceCapRev1Gamma
import Proofs.Pouw.PearlC.ServedMixedRev1Gamma

/-!
# Proofs of the deadline's guarantees

The honest clock names a program that outputs the adversary's transcript wherever it is received by the deadline, after
the same preprocessing, at a cost within the run's spending between the salt and the deadline. That program is within
the budget `EndToEnd` (or `GγSampled`) reads, and its accepted transcripts include every one the verifier accepts by the
deadline.
-/

namespace Pouw.SecurityProofs

theorem WorkBetweenSaltAndDeadline : Pouw.Guarantees.WorkBetweenSaltAndDeadline := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ₀ hΩ hW hTT U hU A run td hclock T hT q hq hB t εs hεs
  obtain ⟨A', hpre, hA'⟩ := hclock
  have hb : A'.Bounded U T q := by
    intro H s
    refine ⟨(hA' H s).2.1.trans (hB H s).1, ?_⟩
    have h1 := (hA' H s).2.2
    have h2 := (hB H s).2
    rw [hpre] at h1 ⊢
    omega
  refine le_trans (pr_mono fun x hx => ?_) (EndToEnd Q R S CM P D γ₀ Ωmax ε hγ₀ hΩ hW hTT U hU T hT q hq A' hb t εs hεs)
  obtain ⟨⟨htr, hacc⟩, hlt⟩ := hx
  rw [(hA' x.1 x.2.1).1 htr]
  exact ⟨hacc, hlt⟩

theorem SpareCapacityAtDeadline : Pouw.Guarantees.SpareCapacityAtDeadline := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hγ₀ hΩ hW hTT U hU A run td hclock C Y hY q hq hB t εs hεs
  have hb : A.BoundedBy U run td (C * (td - run.ts) - Y) q := by
    intro H s
    exact ⟨by linarith [(hB H s).1], (hB H s).2⟩
  refine le_trans (pr_mono fun x hx => ⟨hx.1, by linarith [hx.2]⟩)
    (WorkBetweenSaltAndDeadline Q R S CM P D γ₀ Ωmax ε hγ₀ hΩ hW hTT U hU A run td hclock (C * (td - run.ts) - Y)
      (by linarith) q hq hb t εs hεs)

theorem PearlCSampledByDeadline : Pouw.Guarantees.PearlCSampledByDeadline := by
  intro Q R S _ _ _ _ CM P TR D γ η hG U hU A run td hclock T hT q hq hB w κ hκ hw t εs hεs
  obtain ⟨A', hpre, hA'⟩ := hclock
  have hb : A'.Bounded U T q := by
    intro H s
    refine ⟨(hA' H s).2.1.trans (hB H s).1, ?_⟩
    have h1 := (hA' H s).2.2
    have h2 := (hB H s).2
    rw [hpre] at h1 ⊢
    omega
  refine le_trans (pr_mono fun x hx => ?_) (hG U hU T hT q hq A' hb w κ hκ hw t εs hεs)
  obtain ⟨⟨htr, hacc⟩, hlt⟩ := hx
  rw [(hA' x.1 x.2.1).1 htr]
  exact ⟨hacc, hlt⟩

theorem PearlCSm120Rev1Cap1000ByDeadline_8192 : Pouw.Guarantees.PearlCSm120Rev1Cap1000ByDeadline_8192 :=
  fun Q R S _ _ _ _ CM sem hTT =>
    PearlCSampledByDeadline Q R S CM _ _ _ _ _ (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192 CM sem hTT)

theorem PearlCSm120Rev1Cap1000ByDeadline_16384 : Pouw.Guarantees.PearlCSm120Rev1Cap1000ByDeadline_16384 :=
  fun Q R S _ _ _ _ CM sem hTT =>
    PearlCSampledByDeadline Q R S CM _ _ _ _ _ (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_16384 CM sem hTT)

theorem PearlCSm120Rev1Cap1000ByDeadline_qwen3_8b : Pouw.Guarantees.PearlCSm120Rev1Cap1000ByDeadline_qwen3_8b :=
  fun Q R S _ _ _ _ CM sem hTT =>
    PearlCSampledByDeadline Q R S CM _ _ _ _ _ (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_qwen3_8b CM sem hTT)

theorem PearlCSm120Rev1Cap1000ByDeadline_llama31_8b : Pouw.Guarantees.PearlCSm120Rev1Cap1000ByDeadline_llama31_8b :=
  fun Q R S _ _ _ _ CM sem hTT =>
    PearlCSampledByDeadline Q R S CM _ _ _ _ _
      (PearlC.pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b CM sem hTT)

end Pouw.SecurityProofs
