import Mathlib.Order.Interval.Finset.Nat
import Models.Accounting.Guarantees
import Definitions.Accounting.Ledger.Pins
import Proofs.Accounting.Ledger.Device

/-!
# The device model's guarantees as pins

Each guarantee of the device model (`Models.Accounting.Guarantees`) is a pin of its audit's experiment
(`Accounting.Run.computeExperiment` and its companions), by `Accounting.Run.computeSound_iff` and its two companions:

* `ComputeOfPouw` is PoUW's work pin (`Experiment.PouwWork`) on the cores over the compute window.
* `BandwidthOfAudit` is a budget-form pin of device memory's crossing outward over the bandwidth window, at the bandwidth
  audit's certificate. The game's guarantee (`Bandwidth.Guarantees.AuditRO`) holds per run, and the restatement over a
  ledger keeps that reading (`Experiment.BandwidthRead`); the device model states every audit in budget form.
* `CapacityOfStateBound` is PoUS's graded pin (`Experiment.PousHeldGraded`) on device memory over every tick of the
  capacity window.
-/

namespace Accounting.Ledger

open Accounting Pouw

variable {ι : Type} [Fintype ι] [DecidableEq ι]

theorem pouwWork_of_computeOfPouw (hG : Guarantees.ComputeOfPouw) {Q R S : Type} [Fintype Q] [DecidableEq Q]
    [Fintype R] [Fintype S] {CM : CostModel Q R S} {P : Protocol Q R S} {D : Layout → Prop} {γ₀ Ωmax : ℝ}
    {ε : ℕ → ℕ → ℝ} (hγ : γ₀ < 1) (hΩ : 0 < Ωmax)
    (hW : ∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ))
    (hTT : Assumptions.TT CM P D γ₀ ε) {U : Workload} (hU : U.InDomain D) {t : ℕ} {εs : ℝ} (hεs : εs ≤ 1)
    (hne : Nonempty (PouwCoins P U t)) {α ΩQ ΩS : Type} [Fintype α] [Fintype ΩQ] [Fintype ΩS] {d : Device ι}
    (r : Run d α (PouwCoins P U t) ΩQ ΩS) (p : Price) (hq : d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64)
    (hr : RealizesPouw CM P U r p) :
    r.computeExperiment.PouwWork ⟨{.cores}, (r.win .compute).ticks⟩ p (totalWref P U.layout)
      (reductionGamma γ₀ Ωmax) εs ((1 - εs) ^ t + ε (d.rate .oracle * (r.win .compute).len) U.layout.N) :=
  (r.computeSound_iff _ _ _).mp (hG Q R S CM P D γ₀ Ωmax ε hγ hΩ hW hTT U hU t εs hεs hne ι α ΩQ ΩS d r p hq hr)

theorem crossing_of_bandwidthOfAudit (hG : Guarantees.BandwidthOfAudit) {N b h s G k lam : ℕ}
    {α ΩW ΩS : Type} [Fintype α] [Fintype ΩW] [Fintype ΩS] {d : Device ι}
    (r : Run d α ΩW (BandwidthCoins N b h s) ΩS) (hN : 0 < N) (hr : RealizesBandwidth G r) :
    r.bandwidthExperiment.PinBudget ⟨{.mem}, (r.win .bandwidth).ticks⟩ (Meter.entry .crossOut)
      (((k * b : ℕ) : ℝ) - d.chip - ((∑ i, d.cap i * (r.win .bandwidth).len : ℕ) : ℝ) - ((G * b : ℕ) : ℝ) - lam)
      ((((k - 1 : ℕ) : ℝ) / N) ^ s + (s : ℝ) / 2 ^ h +
        (((d.rate .oracle * (r.win .bandwidth).len).choose k : ℕ) : ℝ) / 2 ^ lam) :=
  (r.bandwidthSound_iff _ _).mp (hG N b h s G k lam ι α ΩW ΩS d r hN hr)

theorem pousHeldGraded_of_capacityOfStateBound (hG : Guarantees.CapacityOfStateBound) {α ΩW ΩQ ΩS : Type}
    [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] {d : Device ι} (r : Run d α ΩW ΩQ ΩS) (Gm : StateGame ΩS)
    (round S₀ : ℕ) {S ε : ℝ} (hε : 0 ≤ ε) (hpass : ∀ n ≤ S₀, ∀ A : Gm.Server n, Pouw.pr (Gm.Passes A) ≤ ε)
    (hr : RealizesState Gm r round)
    (hS : S * (r.win .capacity).len + d.chip + ((∑ i, d.cap i * round : ℕ) : ℝ) ≤ S₀) :
    r.capacityExperiment.PousHeldGraded {.mem} (r.win .capacity).ticks S ε := by
  have h := (r.capacitySound_iff S ε).mp (hG ι α ΩW ΩQ ΩS d r Gm round S₀ S ε hε hpass hr hS)
  have hcard : (r.win .capacity).ticks.card = (r.win .capacity).len := by simp [Window.ticks]
  unfold Experiment.PousHeldGraded
  rwa [hcard]

end Accounting.Ledger
