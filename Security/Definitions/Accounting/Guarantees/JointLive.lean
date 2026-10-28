import Definitions.Accounting.JointLive

/-!
# The joint guarantee over a causal game: a period, a horizon of periods, and server-time blocks (trusted)

In the one theorem's form, a period is one causal game (`Verity.Game`), each device release
drawn after the bytes it answers and the evidence last; its claims are read off the run it ends in
(`Accounting.PeriodLive`), the sampled-proofs claim on the fixed extractor's values (`PeriodLive.extract`), PoUW's in
gates, and its break is the probability that a fixed finder built from the strategy outputs a collision
(`PeriodLive.finder`, `PeriodLive.collides`).

Each subprotocol's per-period guarantee (`PeriodLive.Sound`) bounds, at every strategy of the whole game, the
probability that it accepts while its claim fails by its error at that strategy (`PeriodLive.error`: its `δ` plus its
reduction's loss at its finder's success; or `Accounting.condError`: its `δ` where its finders meet a quantitative
collision bound, `⊤` elsewhere). That is the adaptive form: the subprotocols' coins interleave in the device's
order, and the prover may adapt every byte to every coin released before it. Given those guarantees, at every strategy,
the probability that some subprotocol accepts while its claim fails is at most the errors summed:

* `JointPeriodLive`: one period, the five subprotocols (sampled proofs, PoUW, PoUS, the bandwidth audit and the network
  certifier);
* `JointPeriodLiveECR`: the same in the one adversary form (`PeriodLive.error`) with `ecr/sha-512` asked of each finder, so the
  bound is numbers;
* `JointHorizonLive`: `m` periods read off one game's run, each over its own circuit;
* `JointHorizonBlocksLive`: `B` server-time blocks read off one game's run, with the wipe that opens each block and each
  block's memory term (`Accounting.BlocksRun`'s wipes and memory terms, on the run).

Two statements say the bound is not vacuous: an honest prover meets every claim on every run
(`JointPeriodLive_honest`), and some prover fails a claim with positive probability at an accepting verdict
(`JointPeriodLive_nontrivial`).

The proofs are `Accounting.SecurityProofs.JointPeriodLive` and its siblings.
-/

namespace Accounting.Guarantees

open Accounting
open scoped ENNReal

/-- **The joint guarantee over a period played as a causal game.** On any device and any period read off the run of a
causal game `G`, over a circuit, whose five subprotocols (sampled proofs, PoUW, PoUS, the bandwidth audit and the
network certifier) have their per-period guarantees over `G` at certificates `c` (integrity `ε`, work `W`, space `S`,
bandwidth `Q`, communication `K`) and errors `err`, each a function of the strategy (`PeriodLive.error`'s
`δ + loss(Pr[B^s finds a collision])`, or `condError`'s): at every strategy `s`, the probability that some subprotocol
accepts while its claim fails (more than an `ε` share of the circuit's units unbound in the extractor's transcript,
fewer than `W` gates computed after their coin, fewer than `S` bits held on average, fewer than `Q` bits read, or more
than `K` bits through the network certifiers) is at most `∑_a err_a(s)`. -/
def JointPeriodLive : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (err : Audit → Verity.Game.Strategy G → ℝ≥0∞),
    (∀ a, r.Sound c a (err a)) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s ≤ ∑ a, err a s

/-- **The joint guarantee over a period in the one adversary form, with `ecr/sha-512` asked of the finders only.** As
`JointPeriodLive`, with each subprotocol's error `δ_a(s) + loss_a(Pr[B_a^s finds a collision of H])`
(`PeriodLive.error`), and suppose each subprotocol's loss is monotone and, at every strategy `s`, its finder meets
`ecr/sha-512` at `H` (`Claims.ExpectedCR`: it outputs a collision with probability at most its expected calls over
`2^256`) within `q_a(s)` expected calls. Then at every strategy the probability that some subprotocol accepts while its
claim fails is at most `∑_a (δ_a(s) + loss_a(q_a(s) / 2^256))`. -/
def JointPeriodLiveECR : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (H : Claims.Msg → Claims.Msg) (c : Certificates)
    (δ : Audit → Verity.Game.Strategy G → ℝ≥0∞) (loss : Audit → ℝ≥0∞ → ℝ≥0∞)
    (q : Audit → Verity.Game.Strategy G → ℝ),
    (∀ a, r.Sound c a (fun s => r.error H a (δ a s) (loss a) s)) →
    (∀ a, Monotone (loss a)) →
    (∀ a (s : Verity.Game.Strategy G), Claims.ExpectedCR H (r.finder a s)) →
    (∀ a (s : Verity.Game.Strategy G), Claims.Ex (fun x => ((r.finder a s).cost x : ℝ)) ≤ q a s) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s ≤
        ∑ a, (δ a s + loss a (ENNReal.ofReal (q a s / 2 ^ (256 : ℝ))))

/-- **The honest prover meets every claim.** On any device and any period of a causal game `G`, at a strategy `s`
whose extractor reads an evaluation of the circuit on every run (the honest prover commits the circuit's evaluation on
its inputs) which the commitment opens at every gate, whose run computes at least `W` gates after their coin, whose
world holds `S` bits on average and reads `Q` bits, and whose network certifiers see at most `K` bits on every run, at
a nonnegative integrity share `ε`: every subprotocol's claim holds on every run, so the probability that some
subprotocol accepts while its claim fails is `0`. -/
def JointPeriodLive_honest : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (s : Verity.Game.Strategy G),
    0 ≤ c.ε →
    (∀ o, ∃ x : C.Transcript, r.extract s o = C.eval x) →
    (∀ o g, r.opens g (r.extract s o g)) →
    (∀ o, c.W ≤ (C.size (r.computed s o) : ℝ)) →
    (∀ o, c.S * (r.world s o).win.len ≤
      ((r.world s o).schedule.held (r.world s o).win ((r.world s o).acct .capacity) : ℝ)) →
    (∀ o, c.Q ≤ ((r.world s o).schedule.read (r.world s o).win ((r.world s o).acct .bandwidth) : ℝ)) →
    (∀ o, (r.world s o).commWithin c.K) →
    (∀ a o, r.Holds c s a o) ∧
      Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s = 0

/-- **The bound is not vacuous**: on some device, for some period of some causal game over a circuit, at some
certificates and strategy, a subprotocol accepts while its claim fails with positive probability, at an extractor
whose transcript is wrong at every unit. -/
def JointPeriodLive_nontrivial : Prop :=
  ∃ (ι α : Type) (_ : Fintype α) (d : Device ι) (C : Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : PeriodLive d α C G) (c : Certificates) (s : Verity.Game.Strategy G),
    0 < Verity.Game.prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s

/-- **The joint guarantee over a horizon of periods played as one causal game.** On any device, for any `m` periods
read off the run of one causal game `G`, period `j` over the circuit `C j` (each period's parameters and commitment sent
after the coins of the periods before it, so a period may adapt to them), whose subprotocols have their per-period
guarantees over `G`, period `j`'s at certificates `c j` and errors `err j`, each a function of the strategy: at every
strategy `s`, the probability that some subprotocol of some period accepts while its claim fails is at most
`∑_j ∑_a err_{j,a}(s)`. -/
def JointHorizonLive : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (m : ℕ) (C : Fin m → Verity.Circuit) (O : Type) (G : Verity.Game O)
    (r : (j : Fin m) → PeriodLive d α (C j) G) (c : Fin m → Certificates)
    (err : Fin m → Audit → Verity.Game.Strategy G → ℝ≥0∞),
    (∀ j a, (r j).Sound (c j) a (err j a)) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o => ∃ j a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) s a o) G s ≤
        ∑ j, ∑ a, err j a s

/-- **The joint guarantee over a horizon of `B` server-time blocks and the wipes that open them, played as one causal
game.** On any device and any `B` blocks read off the run of one causal game `G`, block `b` over the circuit `C b`,
suppose, over `G` at every strategy `s`:

* each subprotocol `a` of block `b` has its per-period guarantee at certificates `c b` and error `err b a`;
* the wipe that opens block `b` is accepted while more than `f · fill + 2λ` bits of foreign content plus the unfilled
  state survive beyond the committed carried state with probability at most `ε_wipe b s`;
* block `b`'s PoUS accepts while, at some of its challenge times, its undeclared memory exceeds
  `σ_b · M + S_chip + c` (`σ_b` the share PoUS certifies for block `b`, `M` the device's memory) with probability at
  most `δ_PoUS b s`.

Then at every strategy `s` the probability that some block's accepted subprotocol's claim fails, some accepted wipe's
survivor bound fails, or some block's memory term fails while its PoUS accepted is at most
$$\sum_b \sum_a \mathrm{err}_{b,a}(s) + \sum_b \varepsilon_{\mathrm{wipe},b}(s) + \sum_b \delta_{\mathrm{PoUS},b}(s).$$
**Gap:** the state a block carries across a wipe (the KV cache, committed before the wipe and re-read after
it) is outside the integrity bound. -/
def JointHorizonBlocksLive : Prop :=
  ∀ (ι α : Type) [Fintype ι] [Fintype α] (d : Device ι) (B : ℕ) (C : Fin B → Verity.Circuit) (O : Type)
    (G : Verity.Game O) (S : BlocksLive d α B C G) (c : Fin B → Certificates) (fw : ℝ) (σ : Fin B → ℝ)
    (err : Fin B → Audit → Verity.Game.Strategy G → ℝ≥0∞) (εWipe δPous : Fin B → Verity.Game.Strategy G → ℝ≥0∞),
    (∀ b a, (S.block b).Sound (c b) a (err b a)) →
    (∀ b (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives fw o) G s ≤ εWipe b s) →
    (∀ b (s : Verity.Game.Strategy G),
      Verity.Game.prob (fun o => (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm (σ b) s b o) G s ≤ δPous b s) →
    ∀ s : Verity.Game.Strategy G,
      Verity.Game.prob (fun o =>
          (∃ b a, (S.block b).accepts a o ∧ ¬ (S.block b).Holds (c b) s a o) ∨
          (∃ b, (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives fw o) ∨
          (∃ b, (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm (σ b) s b o)) G s ≤
        ∑ b, ∑ a, err b a s + ∑ b, εWipe b s + ∑ b, δPous b s

end Accounting.Guarantees
