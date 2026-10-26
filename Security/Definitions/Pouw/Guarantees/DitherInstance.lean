import Definitions.Pouw.DitherInstance
import Definitions.Pouw.Assumptions.DitherInstance
import Definitions.Pouw.Guarantees.Dither
import Definitions.Core.Assumptions.Accel

/-!
# The headline on the H200 and the RTX PRO 6000 at θ (trusted)

`WorkSoundAtX c pb hr x` is `Definitions.Pouw.Guarantees.Dither.WorkSound` read on the card table `c` at the open
parameters `x`, with its public best methods `pb` and its node's hosts `hr`, on every period's devices (`Devices`: `n`
cards and the other devices) and every accelerator `G` that is those devices (`OnCard`) with its oracle calls at a
public compression (`CallsAtPub`), with the network certifier's counts and the auditor's clocks in the verdict.

* `WorkSoundAt c pb hr`: at every `x` meeting θ's floors with the product form; `WorkSoundH200`, `WorkSoundSm120`.
* `WorkSoundOnCard c pb hr`: at the card's own `x`, the one `Assumptions.CardWithinAccel` gives with the card's runs as
  its model's (`CardRunsAt`); `WorkSoundOnCardH200`, `WorkSoundOnCardSm120`. This is where trust base item 1 (the card
  can do no more than its model) and the product form enter the headline.

`WorkSound` gives each (`Proofs.Pouw.DitherInstance`).
-/

namespace Pouw.Guarantees.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period on the card table `c` at the open
parameters `x`** (CMP-136; RT-LEAN L3, L4, L9, L31, L32, L33; RT-LEAN-2 L27, L28, L39; RT-MODEL M12; RT-PROTO P2, P5,
P9; RT-PROTO-2 P18). For every period's devices `D` (`n` cards pooled, at `f` ticks a second, and the other devices),
every round protocol `P` and layout `L`, every accelerator `G` that is `D` at `x` under the resource match `e`
(`OnCard`) whose oracle-call step is a public compression on the card (`CallsAtPub`), every set `Reg` of registrable
instances and registered instances `reg`, every service verdict on the transcript and the devices' readings `dev`,
and every period `Pd` with its dual price `μ` at the registered rounds' honest use: with `WorkSound`'s hypotheses, and

* the public bests fixed by the card: instance `a`'s `pub_mm` and `pub_sha` are the cheapest of `pb`'s named programs at
  the period's dual price `μ`, for its honest round's `macs a` MACs and `comps a` compressions (`pubCost`);
* the cards' hosts in the period's devices at the row `hr`'s AMX and SHA-512 rates (`HostsCounted`), so the capacity
  `C` counts them (RT-UNITS U9);
* the network term from the network certifier's counts: the verdict refuses a period whose signed counts into or out
  of the devices pass the caps `E_in`, `E_out` (`Counted`), the certifier counts every exchanged bit
  (`CertifierSeesImports`), so `K = E_in + E_out`; and an exchanged bit replaces at most `β = n_max H_mac` of priced
  work (`OutsourceWithinEgress`), `H_mac` the honest round's cost a MAC at every registrable instance
  (`H(a) ≤ H_mac · macs a`, `H = H_mm + H_sha + H_o`: an outside machine that returns the root saves the round's
  hashing too, RT-PROTO-2 P18). A bit across the certifier carries the coin of at most `n_max` MACs when the certifier
  streams the dither bits `b` (RT-PROTO P2, the coordinator's 06:20Z decision; `n_max = N` for a static call, `d_v` for
  PV). The counts exclude the certifier's own coin stream, and the caps are set at served traffic (P18). Without a
  streamed side, about 1 kbit carries a whole round, and `OutsourceWithinEgress` at this `β` is false;
* the deadline from the auditor's clocks: every round's coin drawn at least `Δ` before the period's end on the
  challenger's schedule `t`, the period refused if the network certifier saw a commitment arrive later than `Δ` after
  its coin (`Timed`, RT-PROTO P5), a commitment arriving no sooner than its work (`RootAfterCoin`), and the period's
  boundary `k Δ ≤ T`,

for every prover `A`, the probability over the oracle and the verifier's coins that the verdict accepts its period and
other work `z` fits on the devices beside its rounds' work with `rv + W_other(z) > OPT_honest + ε_A C` is at most
`δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm(B_adv(A)) + κ_dsha(B_adv(A)))`, at `C = μ · hon = Σ_w H(reg w)` and
`ε_A = (Σ_w (H − pub_mm − pub_sha)(reg w) + ln(1/δ)/η + n_max H_mac (E_in + E_out))/C + μ · ξ_A/C`; and every period
the verdict accepts did its rounds' work within the period's capacity (`TimedWorkFits`), which is what the deadline buys.

The other devices are resources of `G` at their θ rates (RT-PROTO P9): the honest optimum and the capacity count them,
and `SAKMatmul` and `ShaSpeed` hold at the dual price over them too. That they are all of the processors the cards
reach is `Assumptions.PeriodInventory`, held pending Daniel and not read here. `Q` calls use at least the capacity of
one card resource `r` a public compression uses (`cap_r ≤ Q callUse_r`, `CallsAtPub`), whatever the price. -/
def WorkSoundAtX (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) : Prop :=
  ∀ (D : Devices c) (P : RoundProto) (L : Layout (D.machine x) P) (G : Accel)
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res), OnCard G c x D P L e → HostsCounted D hr → CallsAtPub G e pb →
  ∀ (k : ℕ) (Reg : Set P.Inst) (reg : Fin k → P.Inst) (Obs : Type) (verdict : Verdict P k Obs)
    (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ) (Hmm Hsha Ho macs comps : P.Inst → ℝ)
    (η δ nmax Hmac Ein Eout κS Δ Q : ℝ) (κdmm κdsha : ℕ → ℝ) (Badv : Prover G P k → ℕ)
    (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ) (t : Fin k → ℝ) (cin cout : Obs → ℝ)
    (arrive : Obs → Fin k → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubCost e μ pb.mm (macs a) ≤ Hmm a ∧
      pubCost e μ pb.sha (comps a) ≤ Hsha a ∧ Hmm a + Hsha a + Ho a ≤ Hmac * macs a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ nmax → 0 ≤ Hmac → 0 ≤ Ein → 0 ≤ Eout → k * Δ ≤ Pd.T → (∀ A r, 0 ≤ ξ A r) →
    (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) → (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    G.G1 → G.G2 → G.G3 → G.P1 → G.P2 →
    SAKMatmul G P μ Reg (fun a => pubCost e μ pb.mm (macs a)) (fun a => η * (Hmm a + Hsha a + Ho a) / Hmm a)
      κdmm →
    ShaSpeed G P k reg μ (fun a => pubCost e μ pb.sha (comps a)) (fun a => η * (Hmm a + Hsha a + Ho a) / Hsha a)
      κdsha κsha →
    OutsourceWithinEgress G μ (nmax * Hmac) → ServiceSound G P k reg verdict dev κS →
    CertifierSeesImports G P k dev cin cout → RootAfterCoin G P k dev t arrive → ExtractAdv G P k reg Q Badv ξ κx →
    FreedSound G P k (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) dev Pd 1
      (fun A => epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w))
          (fun w => pubCost e μ pb.mm (macs (reg w))) (fun w => pubCost e μ pb.sha (comps (reg w))) η δ
          (nmax * Hmac) (Ein + Eout) +
        (∑ r, μ r * ξ A r) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun A => δ + κS + κx A + κsha A + k * (κdmm (Badv A) + κdsha (Badv A))) ∧
    TimedWorkFits G P k (Timed (Counted verdict cin cout Ein Eout) Pd Δ t arrive) dev Pd

/-- **The headline on the card table `c` at θ**: `WorkSoundAtX` at every open parameter `x` meeting the table's floors
(`c.floors`) with the product form (`AccelProductForm`). -/
def WorkSoundAt (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) : Prop :=
  ∀ (x : Verity.Accel.Open), c.floors x → Verity.Assumptions.AccelProductForm x → WorkSoundAtX c pb hr x

/-- **The headline on the card itself** (RT-LEAN L10, L31; RT-LEAN-2's fix): if the card can do no more than its
accelerator model (`Assumptions.CardWithinAccel c cap oracle can`, trust base item 1, for the card's computations
`can`), then at some open parameters `x` both hold: `x` meets θ's floors with the product form and the card's runs are
`c.machine x`'s programs (`CardRunsAt`), and the headline holds at `x` (`WorkSoundAtX`). So the `x` the headline is
read at is the card's own, the one its runs refine to, and the trust base's model premise is a hypothesis of the
theorem and in the lock.

Open: `G` is still any accelerator that is the devices at `x` (`OnCard`), not the canonical card-level accelerator
whose programs are exactly the card's; the witness that `SAKMatmul` and `ShaSpeed` hold at such a `G`, with free
guesses (RT-LEAN-2 L38), is WORKSOUND-WIT's. A round's commitment reads the oracle, which the card has no instruction
for (ABSORB-THM's realization, R-ACCEL-7). -/
def WorkSoundOnCard (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) : Prop :=
  ∀ {ι : Type} (cap : ι → ℕ) (oracle : ℕ) (can : (c.Mem → c.Mem) → ℕ → (c.Mem → Prop) → Prop),
    Verity.Assumptions.CardWithinAccel c cap oracle can →
    ∃ x, c.floors x ∧ Verity.Assumptions.AccelProductForm x ∧ CardRunsAt c x cap oracle can ∧ WorkSoundAtX c pb hr x

/-- **The headline on the H200 SXM at θ**: `WorkSoundAt` on `Verity.Accel.H200.card`, whose floors are `sync ≥ 1`
(the data sheet's tensor peak is the core's, whichever instruction issues it), `b1 ≥ 1/8` (CALIB's BGMMA measurement)
`madHi ≥ 2` (`IMAD.HI` at 32 lanes a SM-clock, CPG Table 7; RT-MODEL M12) and `slack ≥ 1` (the kernel idles a tick
after every 20 `wgmma`s; RT-MODEL M1), with D5 v2's public bests on the H200 and the H200 node's hosts. -/
def WorkSoundH200 : Prop := WorkSoundAt Verity.Accel.H200.card pubBestH200 hostH200

/-- **The headline on the RTX PRO 6000 (sm_120) at θ**: `WorkSoundAt` on `Verity.Accel.Sm120.card`, whose floor is
`madHi ≥ 2` and which has no native `b1` MMA (ptxas emulates it; RT-MODEL M13), with D5 v2's public bests on the
RTX PRO 6000 and its node's hosts. -/
def WorkSoundSm120 : Prop := WorkSoundAt Verity.Accel.Sm120.card pubBestSm120 hostSm120

/-- **The headline on the H200 SXM itself**: `WorkSoundOnCard` on `Verity.Accel.H200.card`, at the open parameters
`card-within-accel(H200)` gives, with D5 v2's public bests on the H200 and the H200 node's hosts. -/
def WorkSoundOnCardH200 : Prop := WorkSoundOnCard Verity.Accel.H200.card pubBestH200 hostH200

/-- **The headline on the RTX PRO 6000 (sm_120) itself**: `WorkSoundOnCard` on `Verity.Accel.Sm120.card`, at the open
parameters `card-within-accel(RTX PRO 6000)` gives, with D5 v2's public bests on it and its node's hosts. -/
def WorkSoundOnCardSm120 : Prop := WorkSoundOnCard Verity.Accel.Sm120.card pubBestSm120 hostSm120

end Pouw.Guarantees.DitherInstance
