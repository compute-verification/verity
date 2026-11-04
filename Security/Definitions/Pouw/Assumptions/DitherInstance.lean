import Definitions.Pouw.DitherInstance

/-!
# The headline on the cards: the auditor's devices, causality, and the held inventory premise (trusted)

* `CertifierSeesImports`: the network certifier's counts take in every exchanged bit, class L (the auditor's devices,
  trust base item 3: the certifier runs on a machine the prover doesn't control).
* `RootAfterCoin`, `CommitAfterWork` and `AnswerAfterWork`: each commitment and each selected answer arrives no
  sooner than the run that computed it, after the coins that run reads, physical causality (the node is within the
  model). The commitment is the block's root after its dither coins, the operand root after the previous block's work
  (its commit phase), and the answer after its block's dither coins.

  **Why the verdict can't check them**: each bounds a time the auditor's devices sign by a program's
  wall time in the model, and the verdict reads the transcript and the signed times, never the prover's programs. What
  they say among the signed times alone the verdict does check (`Dither.Timed`: each operand root in before its
  block's coins, each root before the next block's coins, each check coin after the block's roots, every deadline
  inside the period). What is left is that the node is within the model: its runs take at least the model's
  wall time, from the coins they read to what the certifier sees arrive.

  `Definitions.Pouw.Guarantees.DitherInstance.WorkSoundAt` reads all four.
* `PeriodInventory`, the period's devices are its inventory. It is stated and held: no guarantee reads it
  until it is decided whether the node's host CPUs and DPUs are devices of the period (the recommendation) or the
  protocol binds hashing to the cards instead. The fleet's transfer from the model to the devices reads it beside
  `Assumptions.CardWithinAccel`.
-/

namespace Pouw.Assumptions

open Pouw.Dither

/-- `pouw-certifier-sees-imports` (proposed; class L, the auditor's devices, trust base item 3). **The network
certifier counts every exchanged bit**: it watches the replica's boundary, every link of the period's devices, on a
machine the prover doesn't control, so the bits a prover's sampling blocks' work and answers exchange with outside
machines (`Prover.ext`, in either direction, other than the challenger's coin stream) are at most the bits it signs
as entering (`cin`) and leaving (`cout`) the devices, as the verdict reads them from its readings `dev`.

**The coin stream is the coins and nothing else**, the one thing `Accel.ext` leaves out: a sampling
block's dither and check coins are its `Draw` and `Chk`, inputs of its programs, not exchanged bits.
* Under the MVP the challenger is the replica's certifier device (`CoinsConfidential`), which makes
  each coin at the boundary, so no bit that crosses a link from outside is left out.
* A challenger on a device of its own is left out message by message: only a coin message that verifies under its key
  and carries its expected (period, replica, sampling block, kind, sequence number), once, at its expected length.
  Every other byte counts: a malformed, unverifiable, duplicate, replayed or late message as entering, and the
  stream's reverse traffic (acknowledgements, flow control) as leaving. A stream told apart by its addresses or headers
  would let forged packets on it carry imports uncounted.

**A replica on several nodes has one boundary**: a certifier on each node's links, the traffic between
its nodes carried certifier to certifier on the auditor's private links, never through the prover's network, and one
count `cin`, `cout` that the replica's certifiers keep together. Everything else crossing between its nodes counts.
Its cards are one pool (`poolCard`), so the traffic between them is the devices' own and not exchanged. Counting it
would fail honest replicas, and leaving it out by address would let an outside machine join as a node. -/
def CertifierSeesImports (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (dev : Readings G P k Obs)
    (cin cout : Obs → ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), A.ext o x ≤ cin (dev A o x) + cout (dev A o x)

/-- `pouw-root-after-coin` (proposed; physical causality, the node within the model). **A commitment arrives no
sooner than the work that computed it**: sampling block `b`'s work reads the dither coins the challenger issues at
`t ob b` (its signed issue time, as the verdict reads it from the readings `ob = dev A o x`), so the network
certifier, watching the replica's boundary, can't see the block's root arrive (`arrive ob b`, its signed time)
before `t ob b` plus the work's wall time in the model. The challenger's and the certifier's clocks are the
auditor's devices. That the node takes at least the model's wall time is the node's being within the model, not a
property of the clocks. -/
def RootAfterCoin (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t arrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb),
    t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) ≤ arrive (dev A o x) b

/-- `pouw-commit-after-work` (proposed; physical causality, the node within the model). **An operand root arrives
no sooner than the commit phase that computed it**: sampling block `b + 1`'s commit phase reads the state after
block `b`'s work, which reads the dither coins the challenger issues at `t ob b`, so the network certifier,
watching the replica's boundary, can't see block `b + 1`'s operand root arrive (`opArrive ob (b + 1)`, its signed
time) before `t ob b` plus the work's and the commit phase's wall times in the model. The first block's commit
phase starts no sooner than the period, at time 0: what a prover computes before the period is its start state,
fixed before it. The clocks are the auditor's devices, and the node's taking at least the model's wall time is the
node's being within the model.

Both pieces are wall time: the verdict checks the order of the signed times (`opArrive b ≤ t b`, `arrive b ≤ t b'`),
but not that the commit phase between the previous block's work and the operand root took its wall time, which only
the prover's programs determine. `TimedWorkFits` needs both: the first to keep block 0's commit phase inside the
period, the second to keep each commit phase after the previous block's work. -/
def CommitAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t opArrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk),
    (∀ b : Fin nsb, (b : ℕ) = 0 → G.wall (A.commit b) (A.commitIn o x b) ≤ opArrive (dev A o x) b) ∧
    ∀ b b' : Fin nsb, (b : ℕ) + 1 = b' →
      t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) + G.wall (A.commit b') (A.commitIn o x b') ≤
        opArrive (dev A o x) b'

/-- `pouw-answer-after-work` (proposed; physical causality, the node within the model). **A selected answer
arrives no sooner than its block's dither coins plus its own run**: sampling block `b`'s answer reads the state
after the block's work, which reads the dither coins issued at `t ob b`, so the network certifier can't see a
selected block's answer arrive (`ansAt ob b`) before `t ob b` plus the answer's wall time in the model. It isn't
anchored at the check coin: the answer's run may start at the block's root, before the check coin is issued, and a
deadline from the check coin bounds it only through this. An unselected block's answer never runs.

It is the one piece of the answer's timing that is wall time: the verdict checks that the dither coins were issued
inside the period and the answer arrived within `τ_ans` of the check coin, before the period's end (`Timed`), but not
how long the answer's program ran, which only the prover's programs determine. That is all `AnswersFit` reads: the
work's wall time before the answer isn't needed, so it isn't assumed. -/
def AnswerAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (sel : P.Chk → Prop) (t ansAt : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb), sel (x b).2 →
    t (dev A o x) b + G.wall (A.answer b) (A.answerIn o x b) ≤ ansAt (dev A o x) b

/-- `pouw-period-inventory` (proposed, and held until it is decided whether host CPUs and DPUs are devices of the
period; class 1, the accelerator model at θ, extended to the period's other processors). **The period's devices are
its inventory**: every processor the period's cards reach without crossing a network certifier (the cards, host CPUs
over PCIe or NVLink-C2C, DPUs) is a device of the period (`Devices`: the cards, and `Others` at their θ rows), and
what those processors together compute within `t` seconds (`can F t`, on any code), some program of `G` computes on
every input within `t` seconds, exchanging nothing. With `Assumptions.CardWithinAccel` for the cards, it is what
carries the headline from the model's provers to the fleet's. -/
def PeriodInventory (G : Accel) (can : ∀ {I O : Type}, (I → O) → ℝ → Prop) : Prop :=
  ∀ {I O : Type} (F : I → O) (t : ℝ), can F t →
    ∃ f : G.Mc.Alg I O, ∀ i, G.Mc.eval f i = F i ∧ G.wall f i ≤ t ∧ G.ext f i = 0

end Pouw.Assumptions
