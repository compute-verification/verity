import Definitions.Nci.Dampener
import Definitions.Nci.Assumptions.Threads
import Definitions.Nci.Assumptions.Baseline

/-!
# The compute dampener: the guarantees

The training-efficiency bound `η ≤ s + min(r, μ) + ν + χ` (`benchmarks/nci/tiers.py`), composed from one bound per
source of counted multiplications. Each source's bound is a hypothesis over real numbers, the reading of a mechanism's
guarantee or of the model's accounting:

* unmonitored hardware trains at full efficiency, `M_U ≤ κ U/2`: the model's accounting, which no mechanism bounds;
* the monitored compute residual trains at full efficiency, `M_r ≤ κ (C − W_cert)/2`: the reading, outside its error,
  of the compute audit's covert residual (`Accounting.Guarantees.CovertResidual`, `CovertResidualPerRun`), whose other
  accounts stay within `F·T − W_cert`;
* the declared computation runs at `ρ` declared gates per counted multiplication, `M_d ≤ C/ρ`: NCI's thread bound,
  `Nci.Guarantees.ThreadedThrottled`, which `DampenerThreaded` takes as proved rather than as a hypothesis;
* faulty proof blocks skip or inject at most the cheating share, `M_χ ≤ χ κ (C + U)/2`: what the sampled proofs' draw
  establishes (the cumulative bound, `Population.Exfil.tail_cumPrev`, at a budget);
* the memory cap bounds the residual and the declared computation on the monitored servers together,
  `M_r + M_d ≤ μ κ C/2`: `DampenerWindows` derives it, with `μ` the memory cap and one window of rounding, from the
  servers' windows, each holding at most `K Λ(K)/b_a` counted multiplications: the memory cap over threads
  (`Accounting.Guarantees.Actual.ThreadedRuntime`) at a whole-tick window `ω = L`, when one threading of the servers'
  computation counts the residual and the declared computation together.

`Dampener` takes all five as hypotheses; `DampenerWindows` derives the memory cap's from the windows; `DampenerThreaded`
also discharges the declared computation's with `ThreadedThrottled`, and `DampenerThreadedSlots` with
`ThreadedThrottledSlots`, weights charged per parameter slot.
-/

namespace Nci.Guarantees

open Verity Nci.Dampener

/-- **The compute dampener.** A developer spends `C > 0` FLOP on monitored hardware and `U ≥ 0` on unmonitored
hardware and computes `M` counted multiplications, each by one of four sources (`M ≤ M_U + M_r + M_d + M_χ`):
unmonitored hardware at full efficiency (`M_U ≤ κ U/2`), the monitored compute residual outside the compute certificate
`W_cert` at full efficiency (`M_r ≤ κ (C − W_cert)/2`), the declared computation at `ρ` declared gates per counted
multiplication (`M_d ≤ C/ρ`), and faulty proof blocks within the cheating share `χ` (`M_χ ≤ χ κ (C + U)/2`). The memory
cap `μ` bounds the residual and the declared computation together (`M_r + M_d ≤ μ κ C/2`). Then the compute efficiency
`η = (2/κ) M/(C + U)` is at most `s + min(r, μ) + ν + χ`, where `s = U/(C + U)`, `r = 1 − W_cert/C` and
`ν = (1 − s) · 2/(κ ρ)`. -/
def Dampener : Prop :=
  ∀ (κ C U W ρ μ χ : ℝ) (M Mu Mr Md Mx : ℕ),
    0 < κ → 0 < C → 0 ≤ U → M ≤ Mu + Mr + Md + Mx →
    (Mu : ℝ) ≤ κ * U / 2 → (Mr : ℝ) ≤ κ * (C - W) / 2 → (Md : ℝ) ≤ C / ρ → (Mx : ℝ) ≤ κ * χ * (C + U) / 2 →
    (Mr : ℝ) + Md ≤ κ * μ * C / 2 →
      efficiency κ M C U ≤ unmonitoredShare C U + min (residualShare W C) μ + declaredShare κ ρ C U + χ

/-- **The compute dampener with the memory cap from windows.** As `Dampener`, with the memory cap's hypothesis replaced
by the servers' windows. `n` servers of `F` gates a tick run for `H > 0` ticks, so that `n F H ≤ C`. Cut each server's
horizon into `⌈H/L⌉` windows of `L = R/(αF)` ticks (`α F L = R`), each starting from at most `R` bits of state (what a
wipe leaves) and taking in at most `αF` bits a tick, so that it reads at most `K = R + αF L = 2R` bits. When each window
holds at most `K Λ(K)/b_a` counted multiplications of the residual and the declared computation, summed
`b_a (M_r + M_d) ≤ n ⌈H/L⌉ K Λ(K)`, and `Λ(2R)` is in its quadratic regime (`e ≤ reuse · 2R/(4 d b_w)`), then
`η ≤ s + min(r, μ (1 + L/H)) + ν + χ` with `μ = 2 reuse · R α/(κ d b_w b_a)`: the factor `1 + L/H` is the last window,
cut at `H`, and is 1 + R/(αFH). -/
def DampenerWindows : Prop :=
  ∀ (κ C U W ρ χ : ℝ) (M Mu Mr Md Mx n : ℕ) (F H L R α reuse e d b_w b_a : ℝ),
    0 < κ → 0 < C → 0 ≤ U → 0 < H → 0 < L → 0 ≤ R → 0 ≤ α → 0 ≤ reuse → 0 < d → 0 < b_w → 0 < b_a →
    M ≤ Mu + Mr + Md + Mx →
    (Mu : ℝ) ≤ κ * U / 2 → (Mr : ℝ) ≤ κ * (C - W) / 2 → (Md : ℝ) ≤ C / ρ → (Mx : ℝ) ≤ κ * χ * (C + U) / 2 →
    n * F * H ≤ C → α * F * L = R → e ≤ reuse * (2 * R) / (4 * d * b_w) →
    b_a * ((Mr : ℝ) + Md) ≤ n * ⌈H / L⌉₊ * ((R + α * F * L) * Lambda reuse e d b_w (R + α * F * L)) →
      efficiency κ M C U ≤
        unmonitoredShare C U + min (residualShare W C) (memoryCap reuse κ R α d b_w b_a * (1 + L / H)) +
          declaredShare κ ρ C U + χ

/-- **The compute dampener over NCI's thread bound.** As `DampenerWindows`, with the declared computation's bound proved:
the declared circuit `d` contains the reference training circuit `c` (`Mc`), threaded by `T` and with incompressible
weights, and complies with NCI at `(Γ, α)`, with its θ registered (`ThetaRegistered`), so its counted multiplications `|T.M|` number at most `size(d)/ρ_T`
(`Nci.Guarantees.ThreadedThrottled`), `ρ_T = b_a/(α Λ(αΓ))`, and `size(d) ≤ C`. The windows' bound is stated with
`T`'s `Λ = T.bound`, as the memory cap over threads states it. Then `η ≤ s + min(r, μ (1 + L/H)) + ν_T + χ` with
`ν_T = (1 − s) · 2/(κ ρ_T)` and `μ = 2 reuse · R α/(κ d b_w b_a)` at `T`'s reuse. -/
def DampenerThreaded : Prop :=
  ∀ (κ C U W χ : ℝ) (M Mu Mr Mx n Γ : ℕ) (F H L R α b_w b_a : ℝ) (c d : Registered) (T : Threading c) (dd e : ℕ)
    (Mc : Containment c.D d.D),
    0 < κ → 0 < C → 0 ≤ U → 0 < H → 0 < L → 0 ≤ R → 0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd →
    Complies Γ α d → Assumptions.ThetaRegistered c d (PartialContainment.ofContainment Mc) →
    T.Threaded Γ (α * Γ) dd e →
    Assumptions.WeightsIncompressible c T Γ (α * Γ) b_w b_a → (d.D.size d.D.computed : ℝ) ≤ C →
    M ≤ Mu + Mr + T.M.card + Mx →
    (Mu : ℝ) ≤ κ * U / 2 → (Mr : ℝ) ≤ κ * (C - W) / 2 → (Mx : ℝ) ≤ κ * χ * (C + U) / 2 →
    n * F * H ≤ C → α * F * L = R → (e : ℝ) ≤ T.reuse * (2 * R) / (4 * dd * b_w) →
    b_a * ((Mr : ℝ) + T.M.card) ≤ n * ⌈H / L⌉₊ * ((R + α * F * L) * T.bound (R + α * F * L) b_w dd e) →
      efficiency κ M C U ≤
        unmonitoredShare C U + min (residualShare W C) (memoryCap T.reuse κ R α dd b_w b_a * (1 + L / H)) +
          declaredShare κ (rhoT α b_a (T.bound (α * Γ) b_w dd e)) C U + χ

/-- **The compute dampener over NCI's thread bound, per parameter slot**: `DampenerThreaded` with the reference
circuit's weights incompressible per parameter slot of a slot map `σ` (`Assumptions.WeightsIncompressibleSlots`), its
declared computation's bound from `Nci.Guarantees.ThreadedThrottledSlots`. -/
def DampenerThreadedSlots : Prop :=
  ∀ (κ C U W χ : ℝ) (M Mu Mr Mx n Γ : ℕ) (F H L R α b_w b_a : ℝ) (c d : Registered) (T : Threading c)
    (σ : T.Slots) (dd e : ℕ) (Mc : Containment c.D d.D),
    0 < κ → 0 < C → 0 ≤ U → 0 < H → 0 < L → 0 ≤ R → 0 ≤ α → 0 < b_w → 0 < b_a → 0 < dd →
    Complies Γ α d → Assumptions.ThetaRegistered c d (PartialContainment.ofContainment Mc) →
    T.Threaded Γ (α * Γ) dd e →
    Assumptions.WeightsIncompressibleSlots c T σ Γ (α * Γ) b_w b_a → (d.D.size d.D.computed : ℝ) ≤ C →
    M ≤ Mu + Mr + T.M.card + Mx →
    (Mu : ℝ) ≤ κ * U / 2 → (Mr : ℝ) ≤ κ * (C - W) / 2 → (Mx : ℝ) ≤ κ * χ * (C + U) / 2 →
    n * F * H ≤ C → α * F * L = R → (e : ℝ) ≤ T.reuse * (2 * R) / (4 * dd * b_w) →
    b_a * ((Mr : ℝ) + T.M.card) ≤ n * ⌈H / L⌉₊ * ((R + α * F * L) * T.bound (R + α * F * L) b_w dd e) →
      efficiency κ M C U ≤
        unmonitoredShare C U + min (residualShare W C) (memoryCap T.reuse κ R α dd b_w b_a * (1 + L / H)) +
          declaredShare κ (rhoT α b_a (T.bound (α * Γ) b_w dd e)) C U + χ

end Nci.Guarantees
