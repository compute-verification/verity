module

public import Proofs.Flock.Bridge.Rate
public import Proofs.Flock.EndToEnd.Carry.Statement
public import Proofs.Flock.EndToEnd.Period.Statement
import Proofs.Flock.EndToEnd.Carry
import Proofs.Flock.EndToEnd.Period

/-!
# The integrity audit pooled over a period, at a rate

Seven statements about a period of `SamplingBlock`s take one share `q` for every window as a hypothesis,
`q ≤ kd_t / n_t`: `IntegrityPeriodLog`, `IntegrityHorizon`, `IntegrityBlock`, `IntegrityBlocks`,
`IntegrityHorizonBlocks`, `IntegrityBlockCarried` and `IntegrityHorizonCarried`. A window's `n_t` is the registration's,
which the prover sizes, so under a fixed count `kd_t` the largest `q` that holds is the prover's. A `SamplingBlock`
samples a uniform `kd`-subset of its `n` units (`SamplingBlock.L`), the design `subset kd`, which is what the rule
`rate:NUM/DEN[:T]` names at `kd = Spot.Rate.count NUM DEN T n` (`Spot.Select.Rule.over`; `Properties.Spot.RuleAtRate`),
and that count is at least `NUM/DEN` of `n` at every `n` (`rateOn_subset`). So under that rule the hypothesis holds at
`q = NUM/DEN`, whatever each window registers:

* `samplingBlock_rate`: a window with a unit whose design `subset kd` is `RateOn num den` has `num/den ≤ kd / n`;
* `pooledIntegrity_rate`: `IntegrityHorizon` with every window's design `RateOn num den` in the hypothesis' place, at
  `q = num/den`;
* `pooledIntegrityCarried_rate`: `IntegrityHorizonCarried`, likewise.

The other five follow the same way, from `samplingBlock_rate`. None of the seven is about a window under
`stratified-rate:` or `work-rate:`, whose sample is no uniform subset: `rateOn_count` gives such a window's share, and
the period statement of the stratified design, `JointStratifiedHorizon`, takes no share as a hypothesis.

Each statement fixes every window's population before the period. A prover that picks a window's population after
seeing earlier windows' coins is outside them. Under a rate the bound does not depend on the population, so the
adaptive statement should hold; it is a different theorem and is not proved.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind
  FlockSoundness.Binding FlockSoundness.LinkFinder Flock.Guarantees
open scoped ENNReal

/-- A window with a unit, whose design `subset kd` samples at least `num/den` of its `n` units (`RateOn`), meets the
pooled statements' hypothesis `q ≤ kd / n` at `q = num/den`. -/
theorem samplingBlock_rate (w : SamplingBlock) {num den : ℕ} (hn : 0 < w.n)
    (hr : RateOn num den (.subset w.kd) w.n) (hd : 0 < den) : (num : ℝ) / den ≤ (w.kd : ℝ) / w.n :=
  rate_le_share hr hd hn

/-- **`IntegrityHorizon` at a rate.** Over `T` windows as in `IntegrityHorizon`, each with a unit and its design
`subset kd_t` sampling at least `num/den` of the `n_t` units it registers (`RateOn`, which the count of a
`rate:NUM/DEN[:T]` rule meets at every `n_t`: `rateOn_subset`), the probability that every window is accepted and the
incorrect proof blocks total at least `ln(1/δ) · den/num` is at most `δ + Σ_t (ε_ks,t + δ_link,t)`. Its named
assumptions are `IntegrityHorizon`'s. -/
theorem pooledIntegrity_rate (W : ℕ → SamplingBlock) (T num den : ℕ) (δ : ℝ) (hnum : 0 < num) (hden : 0 < den)
    (hδ : 0 < δ) (hn : ∀ t < T, 0 < (W t).n) (hrate : ∀ t < T, RateOn num den (.subset (W t).kd) (W t).n)
    (hA3 : ∀ t < T, Assumptions.UniformRandomBytes (W t).Lb (W t).src)
    (s : Strategy (Period.game W T))
    (hT : ∀ t (τ : Strategy (W t).game), Period.Plays W T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (W t).dj (W t).y₀ ((W t).L.draw ω) (reg ((W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (W t).dj (W t).y₀) (cont ((W t).sim τ) ω)) (W t).qF (W t).qS)
    (hL : ∀ t (τ : Strategy (W t).game), Period.Plays W T s t τ →
      LinkCRZC execArith hs512 (W t).P (planZHJ (W t).dj (W t).y₀) (tabZHJ (W t).dj (W t).y₀) (W t).vb
        (reg ((W t).sim τ)) (cont ((W t).sim τ)) (W t).k (W t).Rw (W t).M (W t).t') :
    value (fun l => (∀ r ∈ l, r.accepted) ∧
        Real.log (1 / δ) / ((num : ℝ) / den) ≤ ((l.map WindowRec.wrong).sum : ℝ)) (Period.run W T s) ≤
      ENNReal.ofReal δ + ∑ t ∈ Finset.range T, ((W t).ks + (W t).link) :=
  Flock.SecurityProofs.IntegrityHorizon W T ((num : ℝ) / den) δ (div_pos (Nat.cast_pos.2 hnum) (Nat.cast_pos.2 hden))
    hδ (fun t ht => samplingBlock_rate (W t) (hn t ht) (hrate t ht) hden) hA3 s hT hL

/-- **`IntegrityHorizonCarried` at a rate.** In a monitoring period of server-time blocks with carried leaves
(`CarriedBlocks`), each session with a unit and its design `subset kd_t` sampling at least `num/den` of the `n_t`
units it registers (`RateOn`, as in `pooledIntegrity_rate`), the probability that every session is accepted and the
incorrect proof blocks over the monitoring period total at least `ln(1/δ) · den/num` is at most
`IntegrityHorizonCarried`'s bound. Its named assumptions are `IntegrityHorizonCarried`'s. -/
theorem pooledIntegrityCarried_rate (S : CarriedBlocks) (num den : ℕ) (δ : ℝ) (hnum : 0 < num) (hden : 0 < den)
    (hδ : 0 < δ) (hn : ∀ t < S.T, 0 < (S.W t).n)
    (hrate : ∀ t < S.T, RateOn num den (.subset (S.W t).kd) (S.W t).n)
    (hA3 : ∀ t < S.T, Assumptions.UniformRandomBytes (S.W t).Lb (S.W t).src)
    (s : Strategy S.game)
    (hT : ∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ → ∀ ω j,
      TableCRZ execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀ ((S.W t).L.draw ω) (reg ((S.W t).sim τ))) j
        (zcStrat execArith hs512 (planZHJ (S.W t).dj (S.W t).y₀) (cont ((S.W t).sim τ) ω)) (S.W t).qF (S.W t).qS)
    (hL : ∀ t (τ : Strategy (S.W t).game), Period.Plays S.W S.T s t τ →
      LinkCRZC execArith hs512 (S.W t).P (planZHJ (S.W t).dj (S.W t).y₀) (tabZHJ (S.W t).dj (S.W t).y₀) (S.W t).vb
        (reg ((S.W t).sim τ)) (cont ((S.W t).sim τ)) (S.W t).k (S.W t).Rw (S.W t).M (S.W t).t')
    (hA2 : ∀ x, Assumptions.SHA512CRExpected H512 (finderG (S.mix s x) (S.leaf x).M)
      (finderStrat (S.mix s x) (S.mixS s x) (S.leaf x).M)
      (outOf (rdSig fun _ => (S.leaf x).rd) (S.leaf x).op (S.leaf x).col)
      (finderCost (rdSig fun _ => (S.leaf x).rd) (S.leaf x).costMix)) :
    value (fun l => (∀ r ∈ l, r.accepted) ∧
        Real.log (1 / δ) / ((num : ℝ) / den) ≤ ((l.map WindowRec.wrong).sum : ℝ)) (S.run s) ≤
      ENNReal.ofReal δ + ∑ b ∈ Finset.range S.B, S.err b + ∑ x, (S.leaf x).carry :=
  Flock.SecurityProofs.IntegrityHorizonCarried S ((num : ℝ) / den) δ
    (div_pos (Nat.cast_pos.2 hnum) (Nat.cast_pos.2 hden)) hδ
    (fun t ht => samplingBlock_rate (S.W t) (hn t ht) (hrate t ht) hden) hA3 s hT hL hA2

end Flock.Bridge
