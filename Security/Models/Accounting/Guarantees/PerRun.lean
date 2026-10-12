import Definitions.Accounting.PerRun
import Definitions.Accounting.Pouw
import Definitions.Pouw.Game.Params
import Definitions.Pouw.PearlC.W1Stop
import Models.Pouw.Assumptions.Game

/-!
# The per-run guarantees (trusted)

`Accounting.Guarantees.Joint` and `CovertResidual` take each audit's guarantee in the shape of PoUW's `EndToEnd`, a
sure bound on the account's charge over the audit's own coins, so they bound "accepted, and the worst run below the
certificate". A device that is cheap on a few coins and honest on the rest escapes them. These state the per-run
forms: the probability that every audit accepts and the realized charge *on that run* is below a certificate, or the
other accounts' realized use beyond a residual, is at most the summed error (`JointPerRun`, `CovertResidualPerRun`),
from each audit's per-run guarantee (`Accounting.Run.ComputeSoundPerRun` and the two beside it).

The compute audit's per-run guarantee comes from PoUW by truncation (`PouwEndToEndPerRun`, `ComputeOfPouwPerRun`),
under one more hypothesis on PoUW's cost model, that it truncates (`Accounting.BudgetAbort`); without it the per-run
form does not follow from `Pouw.Assumptions.TT` (`Accounting.PerRunNecessity.tt_not_perRun`). The bandwidth audit's guarantee, `Bandwidth.Guarantees.AuditRO`, is
already per-run: its event reads the ingress of the run itself. PoUS's `EndToEndRing` bounds a server's state by its
type, not a charge that varies with the coins, so its per-run form is a matter of how a run's device memory maps to a
server's state, in the capacity embedding.

**Priced.** The machine charges a PoUW run's arithmetic, at W1's instruction prices with no fetch charge and no refund,
to the compute account, and its first fetches of pre-salt or supplied bytes, deduplicated within the run, to the
bandwidth account at the device's price of a bit read (Daniel's decision 9, 11 Oct). What the compute audit certifies is
the priced sum: the compute account's work plus `pRead` per bit it read from device memory
(`Accounting.Run.ComputeSoundPerRunPriced`). Tables cite the work and the bits separately. Pearl-C's game gives that
guarantee by the same truncation, from its γ-gap property at a priced cost model that truncates
(`PearlCEndToEndPerRun`, `ComputeOfPearlCPerRunPriced`). The machine that stops a run at its budget
(`Pouw.PearlC.CostModel.withStop`) is a proof device, not a deployment requirement: on it every cost model with
nonnegative costs truncates, and the γ-gap property over it is the γ-gap property per coin (`Pouw.PearlC.GγPerCoin`),
so `PearlCStopEndToEndPerRun` and `ComputeOfPearlCStopPerRunPriced` rest on `GγPerCoin` alone. Its adversary may
choose no budget, which runs, costs and calls as at `CM`, so a run that realizes Pearl-C's audit at `CM` realizes it on
the stopping machine (`Accounting.RealizesPearlCPriced.withStop`), and `ComputeOfPearlCStopPerRunPriced` covers every
run `ComputeOfPearlCPerRunPriced` covers, whether or not the device ever stops one. At W1 priced (`w1Priced`) the one
open hypothesis is `GγPerCoin`, a conjecture (claim `pouw-gamma-per-coin`). PoUW's reads and the bandwidth audit's window reads are different device-model
accounts' over one window (`Accounting.Run.ReadsAttributed`), so no bit read is certified twice (`JointPerRunPriced`),
and the covert residual is one priced budget, `(F + pRead·BW)·T − (W + pRead·Q)` (`CovertResidualPerRunPriced`). The
unpriced statements stay: a compute guarantee on the work alone gives the priced one at any `pRead ≥ 0`
(`Accounting.Run.computeSoundPerRunPriced_of_perRun`).

Each `def G : Prop` is proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G`
(`Proofs.Accounting.PerRunPouw`, `Proofs.Accounting.PerRunJoint`, `Proofs.Accounting.PerRunPriced` and
`Proofs.Accounting.PerRunStop`).
-/

namespace Accounting.Guarantees

open Accounting Pouw

/-- **PoUW's end-to-end guarantee, per run.** Under EndToEnd's hypotheses (TT among them), and when the cost model
truncates (`BudgetAbort`), for every in-domain workload, oracle bound `q ≤ 2^64` and adversary that makes at most `q`
oracle calls (with no bound on its cost), the audit with `t` work-weighted samples accepts while the adversary's online
cost on that run is below `(1 − γ)(1 − ε_s)·W` with probability at most `(1 − ε_s)^t + ε(q, N)`, with
`γ = 1 − (1 − γ₀)/Ω_max`. -/
def PouwEndToEndPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε → BudgetAbort CM →
    ∀ U : Workload, U.InDomain D → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, CallsWithin A U q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          onlineCost A U x.1 x.2.1 < (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **PoUW's per-run guarantee is the compute audit's per-run guarantee.** Under `PouwEndToEndPerRun`'s hypotheses, for
any run on any device whose compute audit realizes PoUW's game at price `p`, with at least one draw of the game's coins
and the device's oracle queries over the compute window at most `2^64`, the compute audit has its per-run guarantee at
certificate `(1 − γ)(1 − εs)·W` and error `(1 − εs)^t + ε(q, N)`, where `q` is the device's oracle rate times the compute
window's length. -/
def ComputeOfPouwPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε → BudgetAbort CM →
    ∀ U : Workload, U.InDomain D → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 → Nonempty (PouwCoins P U t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PouwCoins P U t) ΩQ ΩS) (p : Price),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → RealizesPouw CM P U r p →
      r.ComputeSoundPerRun p ((1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ((1 - εs) ^ t + ε (d.rate .oracle * (r.win .compute).len) U.layout.N)

/-- **The joint guarantee, per run.** On any device and any run of a joint audit on it whose compute, bandwidth and
capacity audits have their per-run guarantees at certificates `W`, `Q` and `S` with errors `εW`, `εQ` and `εS`, the
probability over the three audits' coins that every audit accepts and, on that run, the compute account's work is below
`W`, the bandwidth account read fewer than `Q` bits, or the capacity account held fewer than `S` bits on average, is at
most `εW + εQ + εS`. -/
def JointPerRun : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ]
    [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSoundPerRun p W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    pr (fun ω => r.Accepted ω ∧
        ((r.work p ω : ℝ) < W ∨ (r.read ω : ℝ) < Q ∨ (r.held ω : ℝ) < S * (r.win .capacity).len))
      ≤ εW + εQ + εS

/-- **The covert residual, per run.** Under `JointPerRun`'s hypotheses, the probability that every audit accepts and,
on that run, the other accounts did more than `F·T − W` of work over the compute window, read more than `BW·T − Q` bits
over the bandwidth window, or held more than `M − S` bits of device memory on average over the capacity window, is at
most `εW + εQ + εS`. -/
def CovertResidualPerRun : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSoundPerRun p W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    pr (fun ω => r.Accepted ω ∧
        (((d.priced p * (r.win .compute).len : ℕ) : ℝ) - W < r.othersWork p ω ∨
          ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) - Q < r.othersRead ω ∨
          ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω))
      ≤ εW + εQ + εS

/-- **Pearl-C's γ-gap property, audited by work-weighted tickets, per run.** For Pearl-C's game at any cost model that
truncates (`BudgetAbortPearlC`) and has the γ-gap property at `γ < 1` with error `η`, every in-domain workload whose
units' `W_ref` are `κ ≥ 0` times their ticket weights `w`, every oracle bound `q ≤ 2^64` and every adversary that makes
at most `q` oracle calls (with no bound on its cost), the audit with `t` tickets accepts while the adversary's online
cost on that run is below `(1 − γ)(1 − ε_s)·∑_u W_ref(u)` with probability at most `(1 − ε_s)^t + η(q, N)`. At W1 with
fetches priced (`w1Priced`), the cost is the run's instruction prices plus its first fetches priced. -/
def PearlCEndToEndPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : PearlC.CostModel Q R S) (P : PearlC.Protocol Q R S) (D : PearlC.Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ),
    γ < 1 → PearlC.Gγ CM P D γ η → BudgetAbortPearlC CM →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (w : ℕ → ℕ) (κ : ℝ), 0 ≤ κ →
    (∀ u < U.layout.N, P.Wref U.layout u = κ * w u) →
    ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : PearlC.Adversary CM, CallsWithinPearlC A U q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : PearlCCoins Q R S U w t =>
          AuditAccepts (PearlC.correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          onlineCostPearlC A U x.1 x.2.1 < (1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
        ≤ (1 - εs) ^ t + η q U.layout.N

/-- **Pearl-C's per-run guarantee is the compute audit's per-run guarantee on the priced sum.** Under
`PearlCEndToEndPerRun`'s hypotheses, for any run on any device whose compute audit realizes Pearl-C's game with the
adversary's cost covered by the compute account's work at price `p` plus `pRead` per bit it read
(`RealizesPearlCPriced`), with at least one draw of the game's coins and the device's oracle queries over the compute
window at most `2^64`, the compute audit has its per-run guarantee on the priced sum at certificate
`(1 − γ)(1 − ε_s)·∑_u W_ref(u)` and error `(1 − ε_s)^t + η(q, N)`, where `q` is the device's oracle rate times the
compute window's length. Tables cite the work and the bits read separately; the certificate is on their priced sum. -/
def ComputeOfPearlCPerRunPriced : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : PearlC.CostModel Q R S) (P : PearlC.Protocol Q R S) (D : PearlC.Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ),
    γ < 1 → PearlC.Gγ CM P D γ η → BudgetAbortPearlC CM →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (w : ℕ → ℕ) (κ : ℝ), 0 ≤ κ →
    (∀ u < U.layout.N, P.Wref U.layout u = κ * w u) → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
    Nonempty (PearlCCoins Q R S U w t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PearlCCoins Q R S U w t) ΩQ ΩS) (p : Price) (pRead : ℝ),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → RealizesPearlCPriced CM P U w r p pRead →
      r.ComputeSoundPerRunPriced p pRead ((1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
        ((1 - εs) ^ t + η (d.rate .oracle * (r.win .compute).len) U.layout.N)

/-- **Pearl-C's per-run guarantee from the γ-gap property per coin.** For any Pearl-C cost model `CM` with nonnegative
costs and the γ-gap property per coin (`PearlC.GγPerCoin`) at `γ < 1` and error `η`: for every in-domain workload whose
units' `W_ref` are `κ ≥ 0` times their ticket weights `w`, every oracle bound `q ≤ 2^64` and every adversary within `q`
oracle calls (with no bound on its cost) on `CM.withStop`, the audit with `t` tickets accepts while the adversary's
online cost on that run is below the certified `(1 − γ)(1 − ε_s)·∑_u W_ref(u)` with probability at most
`(1 − ε_s)^t + η(q, N)`. `CM.withStop`, `CM` with an optional budget at which a run is stopped, is a proof device: an
adversary of `CM` is one of `CM.withStop` with no budget, at the same transcript, cost and calls. At
`CM = W1.w1Priced dv salt`, the cost is the run's arithmetic, charged to the compute account, plus its first fetches,
charged to the bandwidth account. -/
def PearlCStopEndToEndPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : PearlC.CostModel Q R S),
    (∀ (prog : CM.Prog) (H : Q → R) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (σ : PearlC.State) (s : S),
      0 ≤ CM.cost prog H U act σ s) →
    ∀ (P : PearlC.Protocol Q R S) (D : PearlC.Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ),
    γ < 1 → PearlC.GγPerCoin CM P D γ η →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (w : ℕ → ℕ) (κ : ℝ), 0 ≤ κ →
    (∀ u < U.layout.N, P.Wref U.layout u = κ * w u) →
    ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : PearlC.Adversary CM.withStop, CallsWithinPearlC A U q →
    ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : PearlCCoins Q R S U w t =>
          AuditAccepts (PearlC.correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          onlineCostPearlC A U x.1 x.2.1 < (1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
        ≤ (1 - εs) ^ t + η q U.layout.N

/-- **The compute audit's per-run guarantee on the priced sum, from the γ-gap property per coin.** Under
`PearlCStopEndToEndPerRun`'s hypotheses, for any run on any device whose compute audit realizes Pearl-C's game on
`CM.withStop` with the adversary's cost covered by the compute charge, the compute account's work at price `p`, plus the
bandwidth charge, `pRead` per bit it read (`RealizesPearlCPriced CM.withStop`), with at least one draw of the game's
coins and the device's oracle queries over the compute window at most `2^64`, the compute audit has its per-run
guarantee on the priced sum at the certified `(1 − γ)(1 − ε_s)·∑_u W_ref(u)` and error `(1 − ε_s)^t + η(q, N)`, where
`q` is the device's oracle rate times the compute window's length. The device need not stop a run: a run that realizes
the game at `CM` realizes it on `CM.withStop` (`RealizesPearlCPriced.withStop`, the adversary choosing no budget), so
this covers every run `ComputeOfPearlCPerRunPriced` covers, from `GγPerCoin` in place of `Gγ` and `BudgetAbortPearlC`. -/
def ComputeOfPearlCStopPerRunPriced : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : PearlC.CostModel Q R S),
    (∀ (prog : CM.Prog) (H : Q → R) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (σ : PearlC.State) (s : S),
      0 ≤ CM.cost prog H U act σ s) →
    ∀ (P : PearlC.Protocol Q R S) (D : PearlC.Layout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ),
    γ < 1 → PearlC.GγPerCoin CM P D γ η →
    ∀ U : PearlC.Workload, U.InDomain P D → ∀ (w : ℕ → ℕ) (κ : ℝ), 0 ≤ κ →
    (∀ u < U.layout.N, P.Wref U.layout u = κ * w u) → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
    Nonempty (PearlCCoins Q R S U w t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PearlCCoins Q R S U w t) ΩQ ΩS) (p : Price) (pRead : ℝ),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → RealizesPearlCPriced CM.withStop P U w r p pRead →
      r.ComputeSoundPerRunPriced p pRead ((1 - γ) * (1 - εs) * ∑ u ∈ Finset.range U.layout.N, P.Wref U.layout u)
        ((1 - εs) ^ t + η (d.rate .oracle * (r.win .compute).len) U.layout.N)

/-- **The joint guarantee, per run, with the compute certificate on the priced sum.** On any device and any run of a
joint audit on it whose reads are attributed (`Run.ReadsAttributed`: PoUW's reads and the bandwidth audit's window reads
are different device-model accounts' over one window), whose compute audit has its per-run guarantee on the priced sum at
certificate `W` and read price `pRead`, and whose bandwidth and capacity audits have theirs at `Q` and `S`, with errors
`εW`, `εQ` and `εS`: on every run, each bit read over the window is the bandwidth account's, the compute account's or
another's, so no bit is credited to both certificates; and the probability that every audit accepts and, on that run,
the compute account's work plus `pRead` per bit it read is below `W`, the bandwidth account read fewer than `Q` bits, or
the capacity account held fewer than `S` bits on average, is at most `εW + εQ + εS`. -/
def JointPerRunPriced : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (pRead W Q S εW εQ εS : ℝ),
    r.ReadsAttributed →
    r.ComputeSoundPerRunPriced p pRead W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    (∀ ω, r.read ω + r.pouwRead ω + r.covertRead ω = r.totalRead ω) ∧
      pr (fun ω => r.Accepted ω ∧
          ((r.work p ω : ℝ) + pRead * r.pouwRead ω < W ∨ (r.read ω : ℝ) < Q ∨
            (r.held ω : ℝ) < S * (r.win .capacity).len))
        ≤ εW + εQ + εS

/-- **The covert residual, per run, as one priced budget.** Under `JointPerRunPriced`'s hypotheses and a read price
`pRead ≥ 0`, the probability that every audit accepts and, on that run, the work of the accounts other than the compute
account plus `pRead` per bit read by the accounts other than the compute and bandwidth accounts exceeds
`(F + pRead·BW)·T − (W + pRead·Q)`, or the other accounts held more than `M − S` bits of device memory on average over
the capacity window, is at most `εW + εQ + εS`. `F` is the device's priced work per tick, `BW` its bits read from device
memory per tick and `T` the window's length. -/
def CovertResidualPerRunPriced : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (pRead W Q S εW εQ εS : ℝ),
    0 ≤ pRead → r.ReadsAttributed →
    r.ComputeSoundPerRunPriced p pRead W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    pr (fun ω => r.Accepted ω ∧
        (((d.priced p * (r.win .compute).len : ℕ) : ℝ) + pRead * ((d.bw * (r.win .compute).len : ℕ) : ℝ) -
            (W + pRead * Q) < r.othersWork p ω + pRead * r.covertRead ω ∨
          ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω))
      ≤ εW + εQ + εS

end Accounting.Guarantees
