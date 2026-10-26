import Definitions.Accounting.JointPeriod

/-!
# The joint guarantee over a period audited by every protocol, and over a horizon of periods (trusted)

A period of a node's run (`Accounting.PeriodRun`: one circuit, one device, one coin space) is decided by five audits,
each separately on its own coins: sampled proofs for integrity, PoUW for work, PoUS for space, PoUB for bandwidth and
the network certifier for communication (`Accounting.Audit`). Each audit's per-period guarantee
(`PeriodRun.Sound`: at any value of the other coins, it accepts while its claim fails with probability at most its
error over its own coins) is a hypothesis. Then, over the period's coins, the probability that some audit accepts while
its claim fails is at most the five errors summed (`JointPeriod`). Over a horizon of periods on one coin space, a month
of them say, the probability that some audit of some period accepts while its claim fails is at most every period's
errors summed (`JointHorizon`).

Each audit's guarantee is to come from its protocol's own per-period statement, for a run that realizes the
protocol's audit, each against every prover strategy with no random oracle (PoUW's named random-oracle premises
excepted):

* PoUW: the per-component bound CMP-145 is stating for PoUW, which a `PeriodRun.Sound` at `.pouw` carries into; it is
  not on `main` yet. Until then `Properties.Accounting.JointBridge.WorkOfH1CheckedCreditPeriod` gives PoUW's from H1's
  checked credit over the period's accepted prefix, at `δ_s + n·(δ_bind + δ_ks)`, through a credit of words as work;
* PoUB: the remote bandwidth audit's session, through `Accounting.Guarantees.Bridge.BandwidthOfRemoteLinePeriod`, at the
  bits-read form of its line (`Properties.RemoteAudit.Line.LineHeld`, MEM-97 item 4), which is not stated yet;
* PoUS: `Pous.Guarantees.ContinuousAudit`, the freed share of memory over the period's challenges, at
  `βe + n·(β + 1)·π`, or the remote audit's memory line (`Properties.RemoteAudit.Line.LineHeld`);
* the network certifier: the leak through the network certifiers over a period of sampled audits, each window at
  `δ + ε_ks + ε_bind`, whose count of secrets gives `K`;
* sampled proofs: `Accounting.Guarantees.Bridge.IntegrityOfIntegrityPeriod`, from C-Flock's integrity audit over a
  horizon of windows, a bound on the committed transcript's wrong units.

The bandwidth audit v0's `Bandwidth.Guarantees.AuditRO` is in the random-oracle model, so it is no source here. Given
the five per-period guarantees, both statements are proved (`Accounting.SecurityProofs.JointPeriod` and
`JointHorizon`), by a union bound over the product of the audits' coins.
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee over a period.** On any device and any period of a node's run over a circuit on one coin
space whose five audits (sampled proofs, PoUW, PoUS, PoUB and the network certifier) have their per-period guarantees
at certificates `c` (integrity `ε`, work `W`, space `S`, bandwidth `Q`, communication `K`), price `p` and errors `δ`,
the probability over the period's coins that some audit accepts while its claim fails on that run (more than an `ε`
share of the circuit's units wrong, less than `W` of work, fewer than `S` bits held on average, fewer than `Q` bits
read, or more than `K` bits through the network certifiers) is at most `∑ δ`. The audits are decided separately:
each accepted audit's claim holds, whatever the other audits decided. -/
def JointPeriod : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (r : PeriodRun d α C Ω) (c : Certificates) (p : Price)
    (δ : Audit → ℝ),
    (∀ a, r.Sound c p a (δ a)) →
    Pouw.pr (fun ξ => ∃ a, r.accepts a ξ ∧ ¬ r.Holds c p a ξ) ≤ ∑ a, δ a

/-- **The joint guarantee over a horizon of periods.** On any device, for any `m` periods of a node's run over one
circuit on one coin space (a month of periods, each period's audits on their own coordinates, each period's run a
function of every coin, so a period may adapt to the earlier periods' coins) whose audits have their per-period
guarantees, period `j`'s at certificates `c j`, price `p` and errors `δ j`: the probability that some audit of some
period accepts while its claim fails on that run is at most `∑_j ∑_a δ j a`. Outside it, every accepted audit of every
period holds its claim, so the horizon's work, bits held and bits read are at least the sums of the certificates of
the periods whose PoUW, PoUS and PoUB audits accepted, and its bits through the network certifiers at most the sum of
those whose network certifier accepted. -/
def JointHorizon : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (m : ℕ) (r : Fin m → PeriodRun d α C Ω)
    (c : Fin m → Certificates) (p : Price) (δ : Fin m → Audit → ℝ),
    (∀ j a, (r j).Sound (c j) p a (δ j a)) →
    Pouw.pr (fun ξ => ∃ j a, (r j).accepts a ξ ∧ ¬ (r j).Holds (c j) p a ξ) ≤ ∑ j, ∑ a, δ j a

end Accounting.Guarantees
