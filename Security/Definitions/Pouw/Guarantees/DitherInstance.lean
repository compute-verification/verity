import Definitions.Pouw.DitherInstance
import Definitions.Pouw.Assumptions.DitherInstance
import Definitions.Pouw.Guarantees.Dither
import Definitions.Core.Assumptions

/-!
# The headline on the H200 and the RTX PRO 6000 at θ (trusted)

`WorkSoundAt c pb` is `Definitions.Pouw.Guarantees.Dither.WorkSound` read on the card table `c` with its public best
methods `pb`, at every open parameter meeting θ's floors with the product form, on every period's devices (`Devices`:
`n` cards and the other devices) and every accelerator `G` that is those devices (`OnCard`), with the network
certifier's counts and the auditor's clocks in the verdict. `WorkSoundH200` and `WorkSoundSm120` are its two
instances. `WorkSound` gives each (`Proofs.Pouw.DitherInstance`).
-/

namespace Pouw.Guarantees.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period on the card table `c` at θ** (CMP-136;
RT-LEAN L3, L4, L9, L31, L32, L33; RT-MODEL M12; RT-PROTO P2, P5, P9). For every open parameter `x` meeting the table's
floors (`c.floors`) with the product form (`AccelProductForm`), every period's devices `D` (`n` cards pooled, at `f`
ticks a second, and the other devices), every round protocol `P` and layout `L`, every accelerator `G` that is `D` at
`x` under the resource match `e` (`OnCard`), every service verdict and period `Pd` with its dual price `μ` at `k`
credited rounds: with `WorkSound`'s hypotheses, and

* the public bests fixed by the card: `pub_mm` and `pub_sha` are `pb`'s units priced at `μ` (`pubCost`);
* the network term from the network certifier's counts: the verdict refuses a period whose signed counts into or out
  of the devices pass the caps `E_in`, `E_out` (`Counted`), the certifier counts every imported bit
  (`CertifierSeesImports`), and `K = E_in + E_out` (egress counted as well as ingress), with `β = n_max H_mm / macs`,
  `n_max` MACs' honest cost a bit, `macs` the honest round's MACs: a bit across the certifier carries the coin of at
  most `n_max` MACs when one coin side is streamed by the certifier (RT-PROTO P2 (i), O-IO P11 (i)), so the units
  outsourced are at most `n_max E_out` MACs. Without a streamed side, about 1 kbit carries a whole round, and
  `SAKMatmul` at this `β` is false;
* the deadline from the auditor's clocks: every round's coin drawn at least `Δ` before the period's end on the
  challenger's schedule `t`, the period refused if the network certifier saw a commitment arrive later than `Δ` after
  its coin (`Timed`, RT-PROTO P5), a commitment arriving no sooner than its work (`RootAfterCoin`), and the period's
  boundary `k Δ ≤ T`,

for every prover, the probability over the oracle and the verifier's coins that the verdict accepts its period and
other work fits on the devices beside its rounds' use with `k rv + W_other > OPT_honest + ε C` is at most
`δ + κ_S + κ_x`, at `C = k H` and
`ε = (H_mm (1 − c_mm) + H_sha (1 − c_sha) + H_o)/H + ln(1/δ)/(η k) + n_max (H_mm/macs) (E_in + E_out)/(k H)`;
and every period the verdict accepts did its rounds' work within the period's capacity (`TimedWorkFits`), which is
what the deadline buys.

The other devices are resources of `G` at their θ rates (RT-PROTO P9): the honest optimum and the capacity count them,
and `SAKMatmul` and `ShaSpeed` hold at the dual price over them too. That they are all of the processors the cards
reach is `Assumptions.PeriodInventory`, held pending Daniel and not read here. -/
def WorkSoundAt (c : Verity.Accel.Card) (pb : PubBest c) : Prop :=
  ∀ (x : Verity.Accel.Open), c.floors x → Verity.Assumptions.AccelProductForm x →
  ∀ (D : Devices c) (P : RoundProto) (L : Layout (D.machine x) P) (G : Accel)
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res), OnCard G c x D P L e →
  ∀ (k : ℕ) (verdict : Verdict G P k) (Pd : Period G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho η δ nmax macs Ein Eout κS κx Δ : ℝ) (t : Fin k → ℝ) (cin cout : Count G P k)
    (arrive : Arrival G P k),
    Pd.DualOpt μ k → 0 < Hmm → 0 < Hsha → 0 ≤ Ho → Hmm + Hsha + Ho = ∑ r, μ r * Pd.hon r →
    pubCost e μ pb.mm ≤ Hmm → pubCost e μ pb.sha ≤ Hsha → 0 < η → 0 < δ → δ ≤ 1 → 0 ≤ nmax → 0 < macs →
    0 ≤ Ein → 0 ≤ Eout → k * Δ ≤ Pd.T →
    G.G1 → G.G2 → G.P1 → G.P2 →
    SAKMatmul G P μ (pubCost e μ pb.mm / Hmm) η (nmax * Hmm / macs) Hmm →
    ShaSpeed G P μ (pubCost e μ pb.sha / Hsha) η (nmax * Hmm / macs) Hsha →
    ServiceSound G P k verdict κS → CertifierSeesImports G P k cin → RootAfterCoin G P k t arrive →
    ExtractAdv G P k κx →
    FreedSound G P k (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) Pd k
      (epsWorkSound (pubCost e μ pb.mm / Hmm) (pubCost e μ pb.sha / Hsha) Hmm Hsha Ho η δ (nmax * Hmm / macs)
        (Ein + Eout) k)
      (∑ r, μ r * (k * Pd.hon r)) (δ + κS + κx) ∧
    TimedWorkFits (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) Pd

/-- **The headline on the H200 SXM at θ**: `WorkSoundAt` on `Verity.Accel.H200.card`, whose floors are `sync ≥ 1`
(the data sheet's tensor peak is the core's, whichever instruction issues it), `b1 ≥ 1/8` (CALIB's BGMMA measurement)
`madHi ≥ 2` (`IMAD.HI` at 32 lanes a SM-clock, CPG Table 7; RT-MODEL M12) and `slack ≥ 1` (the kernel idles a tick
after every 20 `wgmma`s; RT-MODEL M1), with the H200's public bests (the
placeholder until D5). -/
def WorkSoundH200 : Prop := WorkSoundAt Verity.Accel.H200.card pubBestH200Placeholder

/-- **The headline on the RTX PRO 6000 (sm_120) at θ**: `WorkSoundAt` on `Verity.Accel.Sm120.card`, whose floor is
`madHi ≥ 2` and which has no native `b1` MMA (ptxas emulates it; RT-MODEL M13), with its public bests (the
placeholder until D5). -/
def WorkSoundSm120 : Prop := WorkSoundAt Verity.Accel.Sm120.card pubBestSm120Placeholder

end Pouw.Guarantees.DitherInstance
