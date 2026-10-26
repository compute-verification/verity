import Definitions.Pouw.DitherInstance

/-!
# The headline on the cards: the premise held pending Daniel (trusted)

`PeriodInventory`, the period's devices are its inventory (RT-PROTO P9). It is stated and held: no guarantee reads it
until Daniel decides whether the node's host CPUs and DPUs are devices of the period (the recommendation) or the
protocol binds hashing to the cards instead. The fleet's transfer from the model to the devices reads it beside
`Assumptions.CardWithinAccel`.
-/

namespace Pouw.Assumptions

open Pouw.Dither

/-- `pouw-period-inventory` (proposed, held pending Daniel; RT-PROTO P9, RT-UNITS U9, O-REDTEAM NF4; class 1, the
accelerator model at θ, extended to the period's other processors). **The period's devices are its inventory**: every
processor the period's cards reach without crossing a network certifier (the cards, host CPUs over PCIe or
NVLink-C2C, DPUs) is a device of the period (`Devices`: the cards, and `Others` at their θ rows), and what those
processors together compute within `t` seconds (`can F t`, on any code), some program of `G` computes on every input
within `t` seconds, importing nothing. With `Assumptions.CardWithinAccel` for the cards, it is what carries the
headline from the model's provers to the fleet's. -/
def PeriodInventory (G : Accel) (can : ∀ {I O : Type}, (I → O) → ℝ → Prop) : Prop :=
  ∀ {I O : Type} (F : I → O) (t : ℝ), can F t →
    ∃ f : G.Mc.Alg I O, ∀ i, G.Mc.eval f i = F i ∧ G.wall f i ≤ t ∧ G.ext f i = 0

end Pouw.Assumptions
