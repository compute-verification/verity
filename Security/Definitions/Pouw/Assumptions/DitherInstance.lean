import Definitions.Pouw.DitherInstance

/-!
# The headline on the cards: the auditor's devices, causality, and the premise held pending Daniel (trusted)

* `CertifierSeesImports`: the network certifier's counts take in every exchanged bit, class L (the auditor's devices,
  trust base item 3: the certifier runs on a machine the prover doesn't control).
* `RootAfterCoin`, `CommitAfterWork` and `AnswerAfterWork`: each commitment and each selected answer arrives no
  sooner than the work that computed it, physical causality (PHYS-INVENTORY's minimal-set item A3, `NodeWithinModel`).
  The commitment is the block's root after its dither coins, the operand root after the previous block's work (its
  commit phase, RT-POOL RP23, RP26), and the answer after its block's work (RT-POOL RP20).

  `Definitions.Pouw.Guarantees.DitherInstance.WorkSoundAt` reads all four.
* `PeriodInventory`, the period's devices are its inventory (RT-PROTO P9). It is stated and held: no guarantee reads it
  until Daniel decides whether the node's host CPUs and DPUs are devices of the period (the recommendation) or the
  protocol binds hashing to the cards instead. The fleet's transfer from the model to the devices reads it beside
  `Assumptions.CardWithinAccel`.
-/

namespace Pouw.Assumptions

open Pouw.Dither

/-- `pouw-certifier-sees-imports` (proposed; class L, the auditor's devices, trust base item 3; RT-PROTO P2, RT-LEAN
L32, CMP-154). **The network certifier counts every exchanged bit**: it watches the replica's boundary, every link of
the period's devices, on a machine the prover doesn't control, so the bits a prover's sampling blocks' work and
answers exchange with outside machines (`Prover.ext`, in either direction, other than the challenger's signed coin
stream) are at most the bits it signs as entering (`cin`) and leaving (`cout`) the devices, as the verdict reads them
from its readings `dev`. -/
def CertifierSeesImports (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (dev : Readings G P k Obs)
    (cin cout : Obs → ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), A.ext o x ≤ cin (dev A o x) + cout (dev A o x)

/-- `pouw-root-after-coin` (proposed; physical causality, PHYS-INVENTORY's minimal-set item A3 `NodeWithinModel`;
RT-PROTO P5, RT-LEAN L33, CMP-154). **A commitment arrives no sooner than the work that computed it**: sampling block
`b`'s work reads the dither coins the challenger issues at `t ob b` (its signed issue time, as the verdict reads it from
the readings `ob = dev A o x`), so the network certifier, watching the replica's boundary, can't see the block's root
arrive (`arrive ob b`, its signed time) before `t ob b` plus the work's wall time in the model. The challenger's and the
certifier's clocks are the auditor's devices. That the node takes at least the model's wall time is the node's being
within the model (A3), not a property of the clocks. -/
def RootAfterCoin (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t arrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb),
    t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) ≤ arrive (dev A o x) b

/-- `pouw-commit-after-work` (proposed; physical causality, PHYS-INVENTORY's minimal-set item A3 `NodeWithinModel`;
RT-POOL RP23, RP26, CMP-154). **An operand root arrives no sooner than the commit phase that computed it**: sampling
block `b + 1`'s commit phase reads the state after block `b`'s work, which reads the dither coins the challenger
issues at `t ob b`, so the network certifier, watching the replica's boundary, can't see block `b + 1`'s operand
root arrive (`opArrive ob (b + 1)`, its signed time) before `t ob b` plus the work's and the commit phase's wall times
in the model. The first block's commit phase starts no sooner than the period, at time 0: what a prover computes
before the period is its start state, fixed before it. The clocks are the auditor's devices, and the node's taking at
least the model's wall time is the node's being within the model (A3). -/
def CommitAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t opArrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk),
    (∀ b : Fin nsb, (b : ℕ) = 0 → G.wall (A.commit b) (A.commitIn o x b) ≤ opArrive (dev A o x) b) ∧
    ∀ b b' : Fin nsb, (b : ℕ) + 1 = b' →
      t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) + G.wall (A.commit b') (A.commitIn o x b') ≤
        opArrive (dev A o x) b'

/-- `pouw-answer-after-work` (proposed; physical causality, PHYS-INVENTORY's minimal-set item A3 `NodeWithinModel`;
RT-POOL RP20, CMP-154). **A selected answer arrives no sooner than its block's work and its own run**: sampling block
`b`'s answer reads the state after the block's work, which reads the dither coins issued at `t ob b`, so the network
certifier can't see a selected block's answer arrive (`ansAt ob b`) before `t ob b` plus the work's and the answer's
wall times in the model. It isn't anchored at the check coin: the answer's run may start at the block's root, before
the check coin is issued, and a deadline from the check coin bounds it only through this. An unselected block's
answer never runs. -/
def AnswerAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (sel : P.Chk → Prop) (t ansAt : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb), sel (x b).2 →
    t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) + G.wall (A.answer b) (A.answerIn o x b) ≤
      ansAt (dev A o x) b

/-- `pouw-period-inventory` (proposed, held pending Daniel; RT-PROTO P9, RT-UNITS U9, O-REDTEAM NF4; class 1, the
accelerator model at θ, extended to the period's other processors). **The period's devices are its inventory**: every
processor the period's cards reach without crossing a network certifier (the cards, host CPUs over PCIe or
NVLink-C2C, DPUs) is a device of the period (`Devices`: the cards, and `Others` at their θ rows), and what those
processors together compute within `t` seconds (`can F t`, on any code), some program of `G` computes on every input
within `t` seconds, exchanging nothing. With `Assumptions.CardWithinAccel` for the cards, it is what carries the
headline from the model's provers to the fleet's. -/
def PeriodInventory (G : Accel) (can : ∀ {I O : Type}, (I → O) → ℝ → Prop) : Prop :=
  ∀ {I O : Type} (F : I → O) (t : ℝ), can F t →
    ∃ f : G.Mc.Alg I O, ∀ i, G.Mc.eval f i = F i ∧ G.wall f i ≤ t ∧ G.ext f i = 0

end Pouw.Assumptions
