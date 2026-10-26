import Definitions.Accounting.JointPeriod

/-!
# The joint guarantee over a horizon of blocks with extra events (trusted)

`JointHorizon` over `B` blocks (each a `PeriodRun`, all on one coin space), with `W` more events on the same coins,
each bounded over the whole product: some event happens, or some audit of some block accepts while its claim fails,
with probability at most the events' bounds plus every block's errors summed (`JointHorizonBlocksAbstract`).

In the demo the events are the parts of the run its audits don't cover: each wipe's survivor bound failing, and each
block's memory term failing. Their bounds are hypotheses here, and none of the bridges from the guarantees below to an
event on this coin space is stated yet (`Pouw.pr_restrict_le` carries a bound proved over a block of the coordinates to
the whole product). Proof: `Accounting.SecurityProofs.JointHorizonBlocksAbstract`, by `Pouw.union_bound_horizon_with`.
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee over a horizon of blocks, with extra events.** On any device, for any `B` blocks of a node's
run over one circuit on one coin space (each block a period whose audits are on their own coordinates and whose run is
a function of every coin, so a block may adapt to the others' coins), whose audits have their per-period guarantees,
block `b`'s at certificates `c b`, price `p` and errors `δ b`, and any `W` events `F w` on the same coins, each of
probability at most `ε w`: the probability that some event happens or some audit of some block accepts while its claim
fails is at most `∑_w ε w + ∑_b ∑_a δ b a`.

In the demo each `F w` is one of:

* a wipe's survivor bound failing: the prover keeps more than the wipe's guarantee allows through its spot reads,
  `Pous.Guarantees.SecureErasureMeetsFamily` at the `N` the wipe's fill reaches, at `2^-128`;
* a block's memory term failing: at one of the block's challenge times the undeclared memory exceeds
  `0.07·U + S_chip + c`, except with `δ_PoUS` (Daniel's ASK-132).

The state a block carries across a wipe is outside the integrity bound (PRF-167): the sampled audit bounds a block's
wrong units, not what survives a wipe into the next block, which only the wipe's event bounds. -/
def JointHorizonBlocksAbstract : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (C : Verity.Circuit) (B W : ℕ) (r : Fin B → PeriodRun d α C Ω)
    (c : Fin B → Certificates) (p : Price) (δ : Fin B → Audit → ℝ) (F : Fin W → ((i : I) → Ω i) → Prop)
    (ε : Fin W → ℝ),
    (∀ b a, (r b).Sound (c b) p a (δ b a)) →
    (∀ w, Pouw.pr (F w) ≤ ε w) →
    Pouw.pr (fun ξ => (∃ w, F w ξ) ∨ ∃ b a, (r b).accepts a ξ ∧ ¬ (r b).Holds (c b) p a ξ) ≤
      ∑ w, ε w + ∑ b, ∑ a, δ b a

end Accounting.Guarantees
