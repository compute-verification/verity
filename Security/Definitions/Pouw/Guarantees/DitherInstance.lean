import Definitions.Pouw.DitherInstance
import Definitions.Pouw.Guarantees.Dither
import Definitions.Core.Assumptions

/-!
# The headline on the H200 and the RTX PRO 6000 at θ (trusted)

`WorkSoundAt c` is `Definitions.Pouw.Guarantees.Dither.WorkSound` read on the card table `c`, at every open parameter
meeting θ's floors with the product form, on every period's devices (`Devices`: `n` cards and the other devices) and
every accelerator `G` that is those devices (`OnCard`), with the network term from the certifier's counts and the
challenger's clock in the verdict. `WorkSoundH200` and `WorkSoundSm120` are its two instances. `WorkSound` gives each
directly (`Proofs.Pouw.DitherInstance`).
-/

namespace Pouw.Guarantees.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period on the card table `c` at θ** (CMP-136;
RT-LEAN L4, L9; RT-MODEL M12; RT-PROTO P2, P5, P9). For every open parameter `x` meeting the table's floors
(`c.floors`) with the product form (`AccelProductForm`), every period's devices `D` (`n` cards pooled, at `f` ticks a
second, and the other devices), every round protocol `P` and layout `L`, every accelerator `G` that is `D` at `x`
(`OnCard`), every service verdict and period `Pd` with its dual price `μ` at `k` credited rounds: with `WorkSound`'s
hypotheses, and

* the network term from the network certifier's counts: `K = E_in + E_out`, the bits it counted into and out of the
  period's devices (egress counted as well as ingress), and `β = n_max H_mm / macs`, `n_max` MACs' honest cost a bit,
  `macs` the honest round's MACs: a bit across the certifier carries the coin of at most `n_max` MACs when one coin side
  is streamed by the certifier (RT-PROTO P2 (i), O-IO P11 (i)), so a unit outsourced costs its coin's egress and the
  units outsourced are at most `n_max E_out` MACs. Without a streamed side, about 1 kbit carries a whole round, and
  `SAKMatmul` at this `β` is false;
* the deadline: the verdict is the challenger's (`Timed`), every round's coin drawn at least `Δ` before the period's
  end on the schedule `t`, and a commitment later than `Δ` after its coin refused (RT-PROTO P5),

for every prover, the probability over the oracle and the verifier's coins that the timed verdict accepts its period
and other work fits on the devices beside its rounds' use with `k rv + W_other > OPT_honest + ε C` is at most
`δ + κ_S + κ_x`, at `C = k H` and
`ε = (H_mm (1 − c_mm) + H_sha (1 − c_sha) + H_o)/H + ln(1/δ)/(η k) + n_max (H_mm/macs) (E_in + E_out)/(k H)`.

The other devices are resources of `G` at their θ rates (RT-PROTO P9): the honest optimum and the capacity count them,
and `SAKMatmul` and `ShaSpeed` hold at the dual price over them too. That they are all of the processors the cards
reach is `Assumptions.PeriodInventory`, held pending Daniel and not read here. -/
def WorkSoundAt (c : Verity.Accel.Card) : Prop :=
  ∀ (x : Verity.Accel.Open), c.floors x → Verity.Assumptions.AccelProductForm x →
  ∀ (D : Devices c) (P : RoundProto) (L : Layout (D.machine x) P) (G : Accel), OnCard G c x D P L →
  ∀ (k : ℕ) (verdict : Verdict G P k) (Pd : Period G) (μ : G.Res → ℝ)
    (pubmm pubsha Hmm Hsha Ho η δ nmax macs Ein Eout κS κx Δ : ℝ) (t : Fin k → ℝ),
    Pd.DualOpt μ k → 0 < Hmm → 0 < Hsha → 0 ≤ Ho → Hmm + Hsha + Ho = ∑ r, μ r * Pd.hon r →
    pubmm ≤ Hmm → pubsha ≤ Hsha → 0 < η → 0 < δ → δ ≤ 1 → 0 ≤ nmax → 0 < macs → 0 ≤ Ein → 0 ≤ Eout →
    G.G1 → G.G2 → G.P1 → G.P2 →
    SAKMatmul G P μ (pubmm / Hmm) η (nmax * Hmm / macs) Hmm →
    ShaSpeed G P μ (pubsha / Hsha) η (nmax * Hmm / macs) Hsha →
    ServiceSound G P k verdict κS → ImportWithinCertifier G P k verdict (Ein + Eout) → Extract G P k κx →
    FreedSound G P k (Timed verdict Pd Δ t) Pd k
      (epsWorkSound (pubmm / Hmm) (pubsha / Hsha) Hmm Hsha Ho η δ (nmax * Hmm / macs) (Ein + Eout) k)
      (∑ r, μ r * (k * Pd.hon r)) (δ + κS + κx)

/-- **The headline on the H200 SXM at θ**: `WorkSoundAt` on `Verity.Accel.H200.card`, whose floors are `sync ≥ 1`
(the data sheet's tensor peak is the core's, whichever instruction issues it), `b1 ≥ 1/8` (CALIB's BGMMA measurement)
and `madHi ≥ 2` (`IMAD.HI` at 32 lanes a SM-clock, CPG Table 7; RT-MODEL M12). -/
def WorkSoundH200 : Prop := WorkSoundAt Verity.Accel.H200.card

/-- **The headline on the RTX PRO 6000 (sm_120) at θ**: `WorkSoundAt` on `Verity.Accel.Sm120.card`, whose floor is
`madHi ≥ 2` and which has no native `b1` MMA (ptxas emulates it; RT-MODEL M13). -/
def WorkSoundSm120 : Prop := WorkSoundAt Verity.Accel.Sm120.card

end Pouw.Guarantees.DitherInstance
