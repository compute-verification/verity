import Proofs.Accounting.PerRunPriced
import Proofs.Pouw.PearlC.W1Priced
import Proofs.Pouw.PearlC.W1Stop

/-!
# Proof of the per-run guarantees from the γ-gap property per coin

The machine that stops a run at its budget (`Pouw.PearlC.CostModel.withStop`) is a proof device, not a deployment
requirement. On it every cost model with nonnegative costs truncates (`Pouw.PearlC.W1Stop.withStop_budgetAbort`), and
its γ-gap property is the γ-gap property per coin (`Pouw.PearlC.W1Stop.gγ_withStop_iff`), so the priced per-run
guarantees hold there from `GγPerCoin` alone. Its adversary may choose no budget, so a run that realizes Pearl-C's audit
at `CM` realizes it there (`RealizesPearlCPriced.withStop`): the guarantees cover a device that never stops a run.

At W1 priced at positive prices, the costs are nonnegative (`Pouw.PearlC.W1Stop.w1Priced_nonneg`), so
`w1Stop_computeSoundPerRunPriced` is the compute audit's guarantee from `GγPerCoin (w1Priced dv salt)`, for a run that
realizes `W1.w1Stop`, or `w1Priced` through the lift. A run of `w1Priced` has its cost on the priced sum by the cost's
split (`Pouw.PearlC.W1Priced.w1Priced_cost`: its arithmetic within the compute charge, its first fetches within the
bandwidth charge); `w1Stop_cost_le` is the same split for a device that does stop a run, whose budget must then be
within the two.
-/

namespace Accounting

open Pouw

variable {Q R S : Type}

/-- **A run that realizes Pearl-C's audit at `CM` realizes it at `CM.withStop`**: the same adversary with no budget
commits the same transcript at the same cost and calls. -/
def RealizesPearlCPriced.withStop {CM : PearlC.CostModel Q R S} {P : PearlC.Protocol Q R S} {U : PearlC.Workload}
    {w : ℕ → ℕ} {t : ℕ} {ι : Type} {d : Device ι} {α ΩQ ΩS : Type} [Fintype α]
    {r : Run d α (PearlCCoins Q R S U w t) ΩQ ΩS} {p : Price} {pRead : ℝ} (h : RealizesPearlCPriced CM P U w r p pRead) :
    RealizesPearlCPriced CM.withStop P U w r p pRead :=
  ⟨fun o => ⟨(h.adv o).pre, ((h.adv o).online, none), (h.adv o).act⟩, h.accepts_iff, h.cost_le, h.calls_le⟩

/-- **The cost's split for a device that stops a run**: a run of `W1.w1Stop` costs at most `work + pRead·bits` when, if
the machine doesn't stop it, its arithmetic (`final`'s `cost`) is within the compute charge `work` and `fetch` per word
it first fetches (`w1Read`) within the bandwidth charge `pRead·bits`, and, if the machine stops it at budget `b`, `b` is
within the two. -/
theorem w1Stop_cost_le (dv : PearlC.W1.W1Device) (salt : S → List ℕ) (prog : PearlC.W1.Prog × Option ℝ) (H : Q → R)
    (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (σ : PearlC.State) (s : S) {work bits pRead : ℝ}
    (hrun : (∀ b, prog.2 = some b → (PearlC.W1.w1Priced dv salt : PearlC.CostModel Q R S).cost prog.1 H U act σ s ≤ b) →
      ((PearlC.W1.final dv (salt s) prog.1 U act σ).cost : ℝ) ≤ work ∧
        (dv.prices.fetch : ℝ) * (PearlC.W1.w1Read dv (salt s) prog.1 U act σ : ℝ) ≤ pRead * bits)
    (hstop : ∀ b, prog.2 = some b → b < (PearlC.W1.w1Priced dv salt : PearlC.CostModel Q R S).cost prog.1 H U act σ s →
      max b 0 ≤ work + pRead * bits) :
    (PearlC.W1.w1Stop dv salt : PearlC.CostModel Q R S).cost prog H U act σ s ≤ work + pRead * bits := by
  obtain ⟨p, o⟩ := prog
  have hsplit := PearlC.W1Priced.w1Priced_cost (Q := Q) (R := R) dv salt p H U act σ s
  cases o with
  | none =>
    obtain ⟨hc, hf⟩ := hrun fun b hb => by cases hb
    change (PearlC.W1.w1Priced dv salt : PearlC.CostModel Q R S).cost p H U act σ s ≤ _
    rw [hsplit]
    linarith
  | some b =>
    change (if (PearlC.W1.w1Priced dv salt : PearlC.CostModel Q R S).cost p H U act σ s ≤ b then
      (PearlC.W1.w1Priced dv salt : PearlC.CostModel Q R S).cost p H U act σ s else max b 0) ≤ _
    split_ifs with hc
    · obtain ⟨h1, h2⟩ := hrun fun b' hb' => by cases hb'; exact hc
      rw [hsplit]
      linarith
    · exact hstop b rfl (lt_of_not_ge hc)

namespace SecurityProofs

theorem PearlCStopEndToEndPerRun : Accounting.Guarantees.PearlCStopEndToEndPerRun := by
  intro Q R S _ _ _ _ CM hnn P D γ η hγ hG U hU w κ hκ hw q hq A hA t εs hεs
  exact PearlCEndToEndPerRun Q R S CM.withStop P D γ η hγ ((PearlC.W1Stop.gγ_withStop_iff CM P D hγ.le η).2 hG)
    ((budgetAbortPearlC_iff _).2 (PearlC.W1Stop.withStop_budgetAbort CM hnn)) U hU w κ hκ hw q hq A hA t εs hεs

theorem ComputeOfPearlCStopPerRunPriced : Accounting.Guarantees.ComputeOfPearlCStopPerRunPriced := by
  intro Q R S _ _ _ _ CM hnn P D γ η hγ hG U hU w κ hκ hw t εs hεs hne ι α ΩQ ΩS _ _ _ d r p pRead hq real
  exact ComputeOfPearlCPerRunPriced Q R S CM.withStop P D γ η hγ
    ((PearlC.W1Stop.gγ_withStop_iff CM P D hγ.le η).2 hG)
    ((budgetAbortPearlC_iff _).2 (PearlC.W1Stop.withStop_budgetAbort CM hnn)) U hU w κ hκ hw t εs hεs hne
    ι α ΩQ ΩS d r p pRead hq real

end SecurityProofs

/-- **At W1 priced**, at positive prices: the compute audit's per-run guarantee on the priced sum from the γ-gap
property per coin over `w1Priced`, for a run that realizes `W1.w1Stop` (one that realizes `w1Priced`, through
`RealizesPearlCPriced.withStop`). -/
theorem w1Stop_computeSoundPerRunPriced [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : PearlC.W1.W1Device)
    (hdv : PearlC.W1BudgetAbort.Positive dv) (salt : S → List ℕ) {P : PearlC.Protocol Q R S}
    {D : PearlC.Layout → Prop} {γ : ℝ} {η : ℕ → ℕ → ℝ} (hγ : γ < 1)
    (hG : PearlC.GγPerCoin (PearlC.W1.w1Priced dv salt) P D γ η) {U : PearlC.Workload} (hU : U.InDomain P D)
    {w : ℕ → ℕ} {κ : ℝ} (hκ : 0 ≤ κ) (hw : ∀ u < U.layout.N, P.Wref U.layout u = κ * w u) {t : ℕ} {εs : ℝ}
    (hεs : εs ≤ 1) [Nonempty (PearlCCoins Q R S U w t)] {ι α ΩQ ΩS : Type} [Fintype α] [Fintype ΩQ] [Fintype ΩS]
    {d : Device ι} (r : Run d α (PearlCCoins Q R S U w t) ΩQ ΩS) {p : Price} {pRead : ℝ}
    (hq : d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64)
    (real : RealizesPearlCPriced (PearlC.W1.w1Stop dv salt) P U w r p pRead) :
    r.ComputeSoundPerRunPriced p pRead ((1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
      ((1 - εs) ^ t + η (d.rate .oracle * (r.win .compute).len) U.layout.N) :=
  SecurityProofs.ComputeOfPearlCStopPerRunPriced Q R S (PearlC.W1.w1Priced dv salt)
    (PearlC.W1Stop.w1Priced_nonneg dv hdv salt) P D γ η hγ hG U hU w κ hκ hw t εs hεs inferInstance ι α ΩQ ΩS d r p
    pRead hq real

end Accounting
