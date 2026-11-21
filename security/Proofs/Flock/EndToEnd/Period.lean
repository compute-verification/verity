module

public import Proofs.Flock.EndToEnd.Period.Statement
public import Proofs.Flock.EndToEnd.Period.Union
public import Proofs.Flock.EndToEndHidden
public import Proofs.Spot.Sampling.Pooled
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Proofs.Spot.Sampling.SamplingDesign
import Proofs.Flock.Soundness.Game

/-!
# Proof of the integrity audit over a period, per sampling block and pooled

* `Flock.SecurityProofs.IntegrityMonitoringPeriod`: `monitoringPeriod_union` over the sampling blocks, each sampling block's bound
  `Flock.SecurityProofs.EndToEndHidden` at the strategy that plays it (`SamplingBlock.accepted_incorrect_le`), with its
  knowledge term at the worst registration.
* `Flock.SecurityProofs.IntegrityPooled`: a run in which every sampling block is accepted and the incorrect proof blocks total at
  least `M` either has a sampling block accepted with a sampled proof block incorrect, at most `Σ_w (ε_ks,w + ε_bind,w)` by
  `monitoringPeriod_union` over `Flock.SecurityProofs.EndToEndHiddenSampled` (`SamplingBlock.accepted_drawnIncorrect_le`), or every
  sampling block's sample misses its incorrect proof blocks with at least `M` of them in all, at most `(1 − q)^M` by
  `monitoringPeriod_horizon`: sampling block `w`'s sample misses its `|W|` incorrect proof blocks with probability at most
  `C(n − |W|, kd)/C(n, kd) ≤ (1 − kd/n)^{|W|}` under A3 (`SamplingBlock.sampleMiss_le`; `SamplingDesign.execOS_subset_missProb_le`,
  `SamplingDesign.choose_div_le_one_sub_pow`). At `M = ⌈ln(1/δ)/q⌉`, `(1 − q)^M ≤ e^{−qM} ≤ δ`.
-/

@[expose] public section

namespace Flock.SecurityProofs

open scoped ENNReal

end Flock.SecurityProofs

namespace Flock.Guarantees.SamplingBlock

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open scoped ENNReal

variable (w : SamplingBlock)

private theorem prob_const {α : Type} (Q : Prop) [Decidable Q] (g : Game α) (s : Strategy g) :
    prob (fun _ => Q) g s = if Q then 1 else 0 := by
  by_cases hQ : Q
  · simp only [hQ, if_true]
    exact prob_true g s
  · simp only [hQ, if_false]
    exact prob_false g s

open Classical in
/-- **A sampling block's sample misses its incorrect proof blocks** with the acceptance probability of its live sample. -/
theorem prob_sampleMiss (B : Finset (Fin w.n)) (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ = w.L.missProb B := by
  show avg (fun ω => prob _ ((liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr (w.L.draw ω) τ.1).map
    fun t => (τ.1, w.L.draw ω, t)) (τ.2 ω)) = prCoin fun ω => Disjoint (w.L.draw ω) B
  unfold prCoin
  refine congrArg avg (funext fun ω => ?_)
  rw [prob_map, prob_const (¬ ∃ u ∈ w.L.draw ω, u ∈ B)]
  beta_reduce
  have hiff : (¬ ∃ u ∈ w.L.draw ω, u ∈ B) ↔ Disjoint (w.L.draw ω) B := by simp [Finset.disjoint_left]
  by_cases hd : Disjoint (w.L.draw ω) B
  · rw [if_pos (hiff.2 hd), if_pos hd]
  · rw [if_neg (mt hiff.1 hd), if_neg hd]

open Classical in
/-- **A sampling block's sample misses its incorrect proof blocks** with probability at most `(1 − kd/n)^{|W|}`, under A3. -/
theorem sampleMiss_le (hA3 : FlockSoundness.Assumptions.UniformRandomBytes w.Lb w.src) (B : Finset (Fin w.n))
    (τ : Strategy w.game) :
    prob (fun o => ¬ ∃ u ∈ o.2.1, u ∈ B) w.game τ ≤ (1 - (w.kd : ℝ≥0∞) / w.n) ^ B.card := by
  rw [prob_sampleMiss]
  refine (SamplingDesign.execOS_subset_missProb_le w.n w.kd w.hkd w.Lb w.src hA3 B).trans ?_
  rw [Verity.SamplingDesign.subset_missProb]
  exact SamplingDesign.choose_div_le_one_sub_pow w.hkd (by simpa using B.card_le_univ)

/-- A sample covering a fraction `q` of the sampling block's units: `1 − kd/n ≤ 1 − q`. -/
theorem one_sub_le {q : ℝ} (hq : 0 < q) (hqk : q ≤ (w.kd : ℝ) / w.n) :
    1 - (w.kd : ℝ≥0∞) / w.n ≤ ENNReal.ofReal (1 - q) := by
  have hn : (0 : ℝ) < w.n := by
    rcases Nat.eq_zero_or_pos w.n with h0 | h0
    · rw [h0, Nat.cast_zero, div_zero] at hqk
      linarith
    · exact_mod_cast h0
  rw [show (w.kd : ℝ≥0∞) / w.n = ENNReal.ofReal ((w.kd : ℝ) / w.n) by
    rw [ENNReal.ofReal_div_of_pos hn, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]]
  rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_sub _ (div_nonneg (Nat.cast_nonneg _) hn.le)]
  exact ENNReal.ofReal_le_ofReal (by linarith)

end Flock.Guarantees.SamplingBlock

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game Flock.Guarantees
open scoped ENNReal

end Flock.SecurityProofs
