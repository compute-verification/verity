import Definitions.Pouw.DitherInstance

/-!
# The headline on the cards: the auditor's devices, causality, and the premise held pending Daniel (trusted)

* `CertifierSeesImports`: the network certifier's counts take in every exchanged bit, class L (the auditor's devices,
  trust base item 3: the certifier runs on a machine the prover doesn't control).
* `RootAfterCoin`: a commitment arrives no sooner than the work that computed it, physical causality (PHYS-INVENTORY's
  minimal-set item A3, `NodeWithinModel`).

  `Definitions.Pouw.Guarantees.DitherInstance.WorkSoundAt` reads both.
* `PeriodInventory`, the period's devices are its inventory (RT-PROTO P9). It is stated and held: no guarantee reads it
  until Daniel decides whether the node's host CPUs and DPUs are devices of the period (the recommendation) or the
  protocol binds hashing to the cards instead. The fleet's transfer from the model to the devices reads it beside
  `Assumptions.CardWithinAccel`.
-/

namespace Pouw.Assumptions

open Pouw.Dither

/-- `pouw-certifier-sees-imports` (proposed; class L, the auditor's devices, trust base item 3; RT-PROTO P2, RT-LEAN
L32). **The network certifier counts every exchanged bit**: it sits on every link of the period's devices, on a machine
the prover doesn't control, so the bits a prover's rounds exchange with outside machines (`Prover.ext`, its work's and
answers', in either direction, other than the certifier's own coin stream) are at most the bits it signs as entering
(`cin`) and leaving (`cout`) the devices, as the verdict reads them from its readings `dev`. -/
def CertifierSeesImports (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (dev : Readings G P k Obs)
    (cin cout : Obs → ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), A.ext o x ≤ cin (dev A o x) + cout (dev A o x)

/-- `pouw-root-after-coin` (proposed; physical causality, PHYS-INVENTORY's minimal-set item A3 `NodeWithinModel`;
RT-PROTO P5, RT-LEAN L33). **A commitment arrives no sooner than the work that computed it**: round `w`'s work reads the
dither coin the challenger draws at `t w`, so the network certifier can't see its commitment arrive (`arrive`, its
signed time as the verdict reads it from its readings `dev`) before `t w` plus the work's wall time in the model. The
challenger's and the certifier's clocks are the auditor's devices; that the node takes at least the model's wall time is
the node's being within the model (A3), not a property of the clocks. -/
def RootAfterCoin (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (dev : Readings G P k Obs) (t : Fin k → ℝ)
    (arrive : Obs → Fin k → ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k),
    t w + G.wall (A.work w) (A.workIn o x w) ≤ arrive (dev A o x) w

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
