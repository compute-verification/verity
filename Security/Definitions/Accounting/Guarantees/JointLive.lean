import Definitions.Accounting.JointLive

/-!
# The joint guarantee over a causal game: a period, a horizon of periods, and server-time blocks (trusted)

Soundness's R5 (26 Oct, LEAN-62): a period is one causal game (`Verity.Game`), each device release drawn after the
bytes it answers and the evidence last, and its claims are read off the run it ends in (`Accounting.PeriodLive`), the
sampled-proofs claim off the commitment (R4: `PeriodLive.Holds .sampled`, with two openings of one position the
period's `Break`).

Each subprotocol's per-period guarantee (`PeriodLive.Sound`) bounds, at every strategy of the whole game, the
probability that it accepts while its claim fails with no break. That is the adaptive form: the subprotocols' coins
interleave in the device's order, and the prover may adapt every byte to every coin released before it. Given those
guarantees, at every strategy, the probability that some subprotocol accepts while its claim fails, with no break, is
at most the errors summed:

* `JointPeriodLive`: one period, the five subprotocols (sampled proofs, PoUW, PoUS, the bandwidth audit and the network
  certifier);
* `JointHorizonLive`: `m` periods read off one game's run, each over its own circuit, each period's parameters and
  commitment sent after the coins of the periods before it, so a period may adapt to them;
* `JointHorizonBlocksLive`: `B` server-time blocks read off one game's run, with the wipe that opens each block and each
  block's memory term (`Accounting.Guarantees.JointHorizonBlocks`' terms, on the run).

The proofs are `Accounting.SecurityProofs.JointPeriodLive`, `JointHorizonLive` and `JointHorizonBlocksLive`, union
bounds over the one game at the strategy. These replace `JointPeriod`, `JointHorizon` and `JointHorizonBlocks`, whose
fixed-other-coins form over a product of coins reads a committed transcript that is a field of the run, once each
subprotocol's guarantee is restated over the causal game.
-/

namespace Accounting.Guarantees

open Accounting

/-- **The joint guarantee over a period played as a causal game.** On any device and any period read off the run of a
causal game `G` (each device release drawn after the bytes it answers, the evidence last), over a circuit, whose five
subprotocols (sampled proofs, PoUW, PoUS, the bandwidth audit and the network certifier) have their per-period
guarantees over `G` at certificates `c` (integrity `ε`, work `W`, space `S`, bandwidth `Q`, communication `K`), price
`p` and errors `δ`: at every strategy, the probability that some subprotocol accepts while its claim fails (more than an
`ε` share of the circuit's units unbound by the commitment's openings, less than `W` of work, fewer than `S` bits held on
average, fewer than `Q` bits read, or more than `K` bits through the network certifiers), with no two openings of one
position, is at most `∑ δ`. -/
def JointPeriodLive : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (Cm O : Type) (r : PeriodLive d α C Cm O)
    (G : Verity.Game O) (c : Certificates) (p : Price) (δ : Audit → ℝ),
    (∀ a, r.Sound G c p a (δ a)) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => (∃ a, r.accepts a o ∧ ¬ r.Holds c p G s a o) ∧ ¬ r.Break G s o) G s ≤
        ∑ a, ENNReal.ofReal (δ a)

/-- **The joint guarantee over a horizon of periods played as one causal game.** On any device, for any `m` periods
read off the run of one causal game `G`, period `j` over the circuit `C j` (each period's parameters and commitment sent
after the coins of the periods before it, so a period may adapt to them), whose subprotocols have their per-period
guarantees over `G`, period `j`'s at certificates `c j`, price `p` and errors `δ j`: at every strategy, the probability
that some subprotocol of some period accepts while its claim fails, with no two openings of one position under that
period's commitment, is at most `∑_j ∑_a δ j a`. -/
def JointHorizonLive : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (m : ℕ) (C : Fin m → Verity.Circuit) (Cm O : Type)
    (r : (j : Fin m) → PeriodLive d α (C j) Cm O) (G : Verity.Game O) (c : Fin m → Certificates) (p : Price)
    (δ : Fin m → Audit → ℝ),
    (∀ j a, (r j).Sound G (c j) p a (δ j a)) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => ∃ j a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) p G s a o ∧ ¬ (r j).Break G s o)
          G s ≤
        ∑ j, ∑ a, ENNReal.ofReal (δ j a)

/-- **The joint guarantee over a horizon of `B` server-time blocks and the wipes that open them, played as one causal
game.** On any device and any `B` blocks read off the run of one causal game `G`, block `b` over the circuit `C b`,
suppose, over `G` at every strategy:

* each subprotocol `a` of block `b` has its per-period guarantee at certificates `c b`, price `p` and error `δ b a`;
* the wipe that opens block `b` is accepted while more than `f · fill + 2λ` bits of foreign content plus the unfilled
  state survive beyond the committed carried state with probability at most `ε_wipe b`;
* block `b`'s PoUS accepts while, at some of its challenge times, its undeclared memory exceeds
  `0.07 · U + S_chip + c` with probability at most `δ_PoUS b`.

Then at every strategy the probability that some block's accepted subprotocol's claim fails with no break, some
accepted wipe's survivor bound fails, or some block's memory term fails while its PoUS accepted is at most
$$\sum_b \sum_a \delta_{b,a} + \sum_b \varepsilon_{\mathrm{wipe},b} + \sum_b \delta_{\mathrm{PoUS},b}.$$
The gap `JointHorizonBlocks` states (PRF-167, the state a block carries across a wipe) is unchanged. -/
def JointHorizonBlocksLive : Prop :=
  ∀ (ι α : Type) [Fintype ι] [Fintype α] (d : Device ι) (B : ℕ) (C : Fin B → Verity.Circuit) (Cm O : Type)
    (S : BlocksLive d α B C Cm O) (G : Verity.Game O) (c : Fin B → Certificates) (p : Price) (f : ℝ)
    (δ : Fin B → Audit → ℝ) (εWipe δPous : Fin B → ℝ),
    (∀ b a, (S.block b).Sound G (c b) p a (δ b a)) →
    (∀ b (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives f o) G s ≤ ENNReal.ofReal (εWipe b)) →
    (∀ b (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm b o) G s ≤ ENNReal.ofReal (δPous b)) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o =>
          (∃ b a, (S.block b).accepts a o ∧ ¬ (S.block b).Holds (c b) p G s a o ∧ ¬ (S.block b).Break G s o) ∨
          (∃ b, (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives f o) ∨
          (∃ b, (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm b o)) G s ≤
        ∑ b, ∑ a, ENNReal.ofReal (δ b a) + ∑ b, ENNReal.ofReal (εWipe b) + ∑ b, ENNReal.ofReal (δPous b)

end Accounting.Guarantees
