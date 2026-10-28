import Definitions.Accounting.JointBlocks
import Definitions.Core.Game
import Definitions.Claims

/-!
# A period as a causal game, its claims on a fixed extractor's values, its break a fixed finder's (trusted)

The one theorem's form:

* **A causal game.** A period is one `Verity.Game` (`Definitions.Core.Game`) played after its parameters and its
  commitment: each device release is a `coin` drawn after the bytes it answers (a `send`), in the device's order, and
  the evidence is the last `send`. The game ends in the run `O`, what the verifier of record reads. A prover's strategy
  (`Verity.Game.Strategy`) sees each coin only once it is drawn.
* **One adversary form.** For every prover `τ`,
  $$\Pr[\text{accept} \wedge \text{the claim fails on } E^\tau\text{'s values}] \le
    \delta + f(\Pr[B^\tau \text{ finds a SHA-512 collision}]),$$
  with the extractor `E` and the finder `B` fixed constructions. The sampled-proofs claim reads the transcript the
  period's extractor `E` (`PeriodLive.extract`) reads off the strategy: at most an `ε` share of the circuit's units are
  unbound in it (`unbound`: wrong in it, or with a committed value the commitment doesn't open to its value there).
  Under zero knowledge a run shows no value, so the values a claim reads are an extractor's; the deployed one is
  C-Flock's plurality over the strategy's runs (`XplurZC`). The break is no event of the run: it is the probability
  that the subprotocol's finder `B` (`PeriodLive.finder`), built from the strategy, outputs a collision of the hash
  (`Claims.Finder.Pr`), so `ecr/sha-512` (`Claims.ExpectedCR`, stated of a `Claims.Finder`) is asked of `B` only.
  A subprotocol's error at a strategy is `PeriodLive.error` (`δ` plus its reduction's loss at that probability) or,
  while its reduction takes its finders' collision resistance as a premise, `condError` (`δ` where the finders meet a
  quantitative bound, `⊤` elsewhere); either way `δ` may read the strategy's cost.

`E` and `B` are fields of the period, fixed before the strategy is: a statement over `PeriodLive` holds at every `E`
and `B`, and the statement that instantiates it (the one theorem's vocabulary, `Verity.Vocab`) names them. At an `E`
that reads an honest evaluation whatever the strategy, the integrity claim holds trivially, so what an instance's
guarantee says rests on the `E` it names.

A period's view of the run (`PeriodLive`) is the device model's period (`Accounting.PeriodRun`) with every field read
off the strategy and the run rather than off a product of coins: its partition and the commitment's openings, its
extractor and finders, the circuit's gates the run computed after their coin (PoUW's claim, in gates), the world the
run realized (`PeriodWorld`: accounts, window, schedule and the network certifier's claim), and each subprotocol's
verdict. A subprotocol's per-period guarantee (`PeriodLive.Sound`) bounds its failure over the whole game at every
strategy: its coins may come before, between or after the other subprotocols', and the prover may adapt every byte to
every coin released before it. The joint guarantees are union bounds over the one game
(`Accounting.Guarantees.JointPeriodLive`, `JointHorizonLive`, `JointHorizonBlocksLive`).

The period's game may be a horizon's: one game, several periods, each read off the same run, each period's parameters
and commitment sent after the coins of the periods before it, so a period's extractor may read the run's prefix up to
its commitment (`extract` reads the run).
-/

namespace Accounting

open Verity
open scoped ENNReal

open Classical in
/-- **The units `X` doesn't bind**: those wrong in `X`, and those that read or write a committed value (`Partition.io`)
that `opens` doesn't open to `X`'s value there. -/
noncomputable def unbound {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
    (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) (X : C.Transcript) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ P.Correct X u ∨ ∃ g ∈ P.io u, ¬ opens g (X g)

/-- **The integrity claim on a transcript**, at share `ε`: at most an `ε` share of `P`'s units are unbound in `X` under
the openings `opens`. -/
def IntegrityHolds {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
    (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) (X : C.Transcript) (ε : ℝ) : Prop :=
  ((unbound P opens X).card : ℝ) ≤ ε * n

/-- **The world a run realized** on device `d` with accounts `α`: whose account each resource is charged to, the window,
the device schedule, which fits the device, and the network certifier's claim at `K` (what the network certifiers saw
the node move is within `K` bits). -/
structure PeriodWorld {ι : Type} (d : Device ι) (α : Type) [Fintype α] where
  acct : Resource → α
  win : Window
  schedule : Schedule ι α
  fits : schedule.Fits d
  commWithin : ℝ → Prop

/-- **A period's view of the run `O` the causal game `G` ends in**, on device `d` with accounts `α`, over the circuit
`C`: the partition whose units the integrity claim counts, the commitment's openings, the extractor whose values the
integrity claim reads and each subprotocol's collision finder, both built from the strategy, the gates the run
computed after their coin, the world the run realized, and each subprotocol's verdict. -/
structure PeriodLive {ι : Type} (d : Device ι) (α : Type) [Fintype α] (C : Verity.Circuit) {O : Type} (G : Verity.Game O) where
  /-- the number of units of the partition -/
  units : ℕ
  /-- the partition of the circuit's computed gates into units -/
  parts : Verity.Partition C units
  /-- the commitment opens gate `g` to `v` -/
  opens : (g : Fin C.N) → BitVec (C.w g) → Prop
  /-- `E`: the transcript the period's extractor reads off the strategy, on the run -/
  extract : Verity.Game.Strategy G → O → C.Transcript
  /-- `B`: a subprotocol's collision finder, built from the strategy -/
  finder : Audit → Verity.Game.Strategy G → Claims.Finder
  /-- the circuit's gates the run computed after their window's coin, over the period's windows -/
  computed : Verity.Game.Strategy G → O → Finset (Fin C.N)
  /-- the world the run realized -/
  world : Verity.Game.Strategy G → O → PeriodWorld d α
  /-- a subprotocol's verdict on the run -/
  accepts : Audit → O → Prop

namespace PeriodLive

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {O : Type} {G : Verity.Game O}
  (r : PeriodLive d α C G)

/-- **The claim a subprotocol certifies** at certificates `c`, on the run `o` of the strategy `s`: sampled proofs' on
the extractor's values (at most an `ε` share of the units unbound), PoUW's in gates (at least `W` gates computed after
their coin), and the others' on the world the run realized (at least `S` bits of device memory held on average, at
least `Q` bits read, at most `K` bits through the network certifiers). -/
def Holds (c : Certificates) (s : Verity.Game.Strategy G) : Audit → O → Prop
  | .sampled, o => IntegrityHolds r.parts r.opens (r.extract s o) c.ε
  | .pouw, o => c.W ≤ (C.size (r.computed s o) : ℝ)
  | .pous, o => c.S * (r.world s o).win.len ≤
      ((r.world s o).schedule.held (r.world s o).win ((r.world s o).acct .capacity) : ℝ)
  | .poub, o => c.Q ≤ ((r.world s o).schedule.read (r.world s o).win ((r.world s o).acct .bandwidth) : ℝ)
  | .network, o => (r.world s o).commWithin c.K

/-- **The probability that subprotocol `a`'s finder, built from the strategy `s`, outputs a collision of `H`.** -/
noncomputable def collides (H : Claims.Msg → Claims.Msg) (a : Audit) (s : Verity.Game.Strategy G) : ℝ≥0∞ :=
  ENNReal.ofReal ((r.finder a s).Pr H)

/-- **Subprotocol `a`'s error at the strategy `s`**: its error `δ`, plus its reduction's loss at the probability that
its finder, built from `s`, outputs a collision of `H`. -/
noncomputable def error (H : Claims.Msg → Claims.Msg) (a : Audit) (δ : ℝ≥0∞) (loss : ℝ≥0∞ → ℝ≥0∞)
    (s : Verity.Game.Strategy G) : ℝ≥0∞ :=
  δ + loss (r.collides H a s)

/-- **A subprotocol's per-period guarantee over the causal game**, at certificates `c` and the error `err s` at each
strategy `s`: at every strategy `s`, the probability over the coins the game draws that subprotocol `a` accepts while
its claim fails on the extractor's values is at most `err s`. The error is a function of the strategy, as `error` and
`condError` are: its `δ` may read the strategy's cost, and its finder term the finder built from it. -/
def Sound (c : Certificates) (a : Audit) (err : Verity.Game.Strategy G → ℝ≥0∞) : Prop :=
  ∀ s : Verity.Game.Strategy G,
    Verity.Game.prob (fun o => r.accepts a o ∧ ¬ r.Holds c s a o) G s ≤ err s

end PeriodLive

open Classical in
/-- **An error conditional on its finders' collision resistance**: `δ s` at a strategy `s` where `P s` holds, `⊤`
(no bound) elsewhere. With `P s` a quantitative bound on the probability that the finders built from `s` output a
collision (`Claims.Finder.Pr`), this is the conditional form of the one theorem (the 21:55Z ruling); with `P` true it is
`δ`. -/
noncomputable def condError {O : Type} {G : Verity.Game O} (P : Verity.Game.Strategy G → Prop)
    (δ : Verity.Game.Strategy G → ℝ≥0∞) (s : Verity.Game.Strategy G) : ℝ≥0∞ :=
  if P s then δ s else ⊤

/-- The wipe that opens a server-time block, read off the run: the bits of device memory its fill covers, the memory
challenger's verdict, and `kept`, the bits of foreign content beyond the committed carried state that the block can
read back from what the wipe leaves (`Accounting.BlockWipe`'s fields, on the run rather than on a product of coins). -/
structure WipeLive {ι : Type} (d : Device ι) (O : Type) where
  /-- the bits of device memory the fill covers -/
  fill : ℕ
  fill_le : fill ≤ d.mem
  /-- the memory challenger's verdict on the wipe -/
  accepts : O → Prop
  /-- the bits of foreign content beyond the committed carried state that survive the wipe into the block -/
  kept : O → ℕ

/-- **The wipe's survivor bound** at fraction `f`, on the run `o`: `BlockWipe.Survives`' bound. -/
def WipeLive.Survives {ι : Type} {d : Device ι} {O : Type} (w : WipeLive d O) (f : ℝ) (o : O) : Prop :=
  (w.kept o : ℝ) ≤ f * w.fill + 2 * 128 + (d.chip + (d.mem - w.fill))

/-- A horizon of `B` server-time blocks read off one run of the game `G`: each block's period, over its own circuit,
the wipe that opens it, its PoUS challenge times on the run, and the read deadline. -/
structure BlocksLive {ι : Type} (d : Device ι) (α : Type) [Fintype α] (B : ℕ) (C : Fin B → Verity.Circuit)
    {O : Type} (G : Verity.Game O) where
  /-- block `b`'s period -/
  block : (b : Fin B) → PeriodLive d α (C b) G
  /-- the wipe that opens block `b` -/
  wipe : Fin B → WipeLive d O
  /-- block `b`'s PoUS challenge times on the run -/
  chal : Fin B → O → Finset ℕ
  /-- the read deadline, in ticks (T_read) -/
  readDeadline : ℕ

/-- **Block `b`'s memory term at share `σ`** on the run `o` of the strategy `s` (`BlocksRun.MemoryTerm`'s): at each of
its PoUS challenge times, the bits of device memory its capacity account doesn't hold, `M` less what it holds, are at
most `σ · M + S_chip + ∑_i cap_i · T_read`, `M` the device's memory. -/
def BlocksLive.MemoryTerm {ι : Type} [Fintype ι] {d : Device ι} {α : Type} [Fintype α] {B : ℕ}
    {C : Fin B → Verity.Circuit} {O : Type} {G : Verity.Game O} (S : BlocksLive d α B C G) (σ : ℝ)
    (s : Verity.Game.Strategy G) (b : Fin B) (o : O) : Prop :=
  ∀ t ∈ S.chal b o, (d.mem : ℝ) -
      ((((S.block b).world s o).schedule t).mem (((S.block b).world s o).acct .capacity) : ℝ) ≤
    σ * d.mem + d.chip + (∑ i, d.cap i * S.readDeadline : ℕ)

end Accounting
