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

* `WorkSoundAt c pb hr`: at every `x` meeting (P1) and (P2); `WorkSoundH200`, `WorkSoundSm120`.
* `WorkSoundOnCard c pb hr`: at the card's own `x`, the one `Assumptions.CardWithinAccel` gives with the card's runs as
  its model's (`CardRunsAt`); `WorkSoundOnCardH200`, `WorkSoundOnCardSm120`. This is where the card premise (the card
  can do no more than its model) and the product form enter the headline.

`WorkSound` gives each (`Proofs.Pouw.DitherInstance`).
-/

open scoped NNRat

namespace Pouw.Guarantees.DitherInstance

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period on the card table `c` at the open parameters
`x`**. For every period's devices `D` (`n` cards pooled, at the table's rates, and the other devices), every
sampling-block protocol `P` and round plan `L`, every accelerator `G` of the machine model 𝓜 that is `D` at `x` under the
resource match `e` (`OnCard`) whose oracle-call step is a public compression on the card (`CallsAtPub`), every number
`nsb` of sampling blocks, every sampling block's set `Reg b` of registrable instances, every service verdict on the
transcript and the devices' readings `dev`, and every period `Pd` with its dual price `μ` at the sampling blocks' honest
use: with `WorkSound`'s hypotheses, and

* the public bests fixed by the card: block `b`'s `pub_mm` and `pub_sha` are the cheapest of `pb`'s named programs at
  the period's dual price `μ`, for its honest sampling block's `macs b` MACs and `comps b` compressions (`pubCost`),
  its shape public and the same at every instance of `Reg b`;
* the cards' hosts in the period's devices at the row `hr`'s AMX and SHA-512 rates (`HostsCounted`), so the honest
  optimum and the period's capacity count them;
* the network term from the network certifier's counts: it watches the replica's boundary, and the verdict refuses a
  period whose signed counts into or out of the devices pass the caps `E_in`, `E_out` (`Counted`). The certifier counts
  every exchanged bit (`CertifierSeesImports`), so `E = E_in + E_out`, and an exchanged bit replaces at most
  `β = n_max W_mac` of priced work (`OutsourceWithinEgress`). `W_mac` is the honest sampling block's cost a MAC
  (`W_b ≤ W_mac · macs b`, `W_b = W_mm + W_sha + W_ans`: an outside machine that returns the root saves the block's
  hashing too). A bit across the boundary carries the coin of at most `n_max` MACs when the challenger issues the dither
  bits as a signed coin stream (`n_max = N` for a static call, `d_v` for PV). The counts exclude the challenger's coin
  stream and nothing else, as `CertifierSeesImports` defines it, a replica on several nodes has one pair of counts, and
  the caps are set at served traffic. Without a streamed side, about 1 kbit carries a whole sampling block, and
  `OutsourceWithinEgress` at this `β` is false;
* the deadlines from the auditor's clocks (`Timed`), in each sampling block's commit, challenge and response
  phases:
  - the challenger issues each sampling block's dither coins at `t b`, after its operand root is in
    (`opArrive b ≤ t b`), at least `Δ` before the period's end, and the next block's only after every
    root of this one is in;
  - every block's check coin, which carries whether it is selected, is issued at `tc b`, after every root of the
    block is in;
  - only selected sampling blocks (`sel` of the check coin) are answered, each selected block's check coin at least
    `τ_ans` before the end;
  - the period is refused if the network certifier saw a root arrive later than `Δ` after its block's coins, or a
    selected block's answer later than `τ_ans` after its check coin. An answer may arrive after later blocks' coins;
  - an operand root arrives no sooner than its commit phase after the previous block's work (`CommitAfterWork`), a
    root no sooner than its work after its coins (`RootAfterCoin`), a selected answer no sooner than its own run after
    its block's dither coins (`AnswerAfterWork`), and the period's length `T` is at least 0. These three are the
    node's wall time in the model, which no verdict reads; the orderings among the signed times are `Timed`'s checks.

  No count of sampling blocks or of selected ones is a hypothesis: the blocks' commit phases and work run one block at
  a time, so they fit in `T` at any `nsb`. The answers run beside later blocks, and no answer's use is counted, only
  its exchanged bits and its oracle calls, under the cap (`Prover`). So a served period of about 1e8–1e9 sampling
  blocks a week is covered. `q (Δ + τ_ans) ≤ T` over the `q` selected blocks is the honest replica's schedule (its
  answers through the firewall one at a time), not a premise of this bound,

for every prover `A`, the probability over the oracle and the verifier's coins that the verdict accepts its period and
other work `z` fits on the devices beside its sampling blocks' timed work with
`rv + W_other(z) > OPT_honest + ε_A W_reg` is at most
`δ + κ_S + κ_x(A) + κ_sha(A) + nsb (κ_dmm(B_adv(A)) + κ_dsha(B_adv(A)))`, at `W_reg = μ · hon = Σ_b W_b` and
`ε_A = (Σ_b (W_b − pub_mm − pub_sha) + ln(1/δ)/η_w + n_max W_mac E)/W_reg + μ · ξ_A/W_reg`. Here `W_ans` holds the
selected blocks' honest answers and operand roots at full weight, and `η_w` is the rate per priced cost,
`η q/W_reg`. Every period the verdict accepts did its sampling blocks' commit phases and work within the period's
capacity (`TimedWorkFits`) and ran its selected answers within the period (`AnswersFit`), which is what the deadlines
buy.

The other devices are resources of `G` at their rates in 𝓜: the honest optimum and the capacity count
them, and `SAKMatmul` and `SAKSha` hold at the dual price over them too. That they are all of the processors the
cards reach is `Assumptions.PeriodInventory`, held until it is decided whether they are devices of the period, and not
read here. `Q` calls use at least the capacity of one card resource `r` a public compression uses
(`cap_r ≤ Q callUse_r`, `CallsAtPub`), whatever the price. -/
def WorkSoundAtX (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) (x : Verity.Accel.Open) : Prop :=
  ∀ (D : Devices c) (P : RoundProto) (L : RoundPlan (D.machine x) P) (G : Accel)
    (e : G.Res ≃ Verity.Accel.Res ⊕ D.oth.Res), OnCard G c x D P L e → HostsCounted D hr → CallsAtPub G e pb →
  ∀ (nsb : ℕ) (Reg : Fin nsb → Set P.Inst) (Obs : Type) (verdict : Verdict P nsb Obs)
    (dev : Readings G P nsb Obs) (Pd : Jobs G) (μ : G.Res → ℝ) (Wmm Wsha Wans macs comps : Fin nsb → ℝ)
    (ηw δ nmax Wmac Ein Eout κS Δ Q : ℝ) (κdmm κdsha : ℕ → ℝ) (Badv : Prover G P nsb → ℕ)
    (ξ : Prover G P nsb → G.Res → ℝ) (κx κsha : Prover G P nsb → ℝ) (cin cout : Obs → ℝ)
    (opArrive t arrive : Obs → Fin nsb → ℝ) (sel : P.Chk → Prop) (τans : ℝ) (tc ansAt : Obs → Fin nsb → ℝ),
    Pd.DualOpt μ 1 →
    (∀ b, 0 < Wmm b ∧ 0 < Wsha b ∧ 0 ≤ Wans b ∧ pubCost e μ pb.mm (macs b) ≤ Wmm b ∧
      pubCost e μ pb.sha (comps b) ≤ Wsha b ∧ Wmm b + Wsha b + Wans b ≤ Wmac * macs b) →
    ∑ b, (Wmm b + Wsha b + Wans b) = ∑ r, μ r * Pd.hon r →
    0 < ηw → 0 < δ → δ ≤ 1 → 0 ≤ nmax → 0 ≤ Wmac → 0 ≤ Ein → 0 ≤ Eout → 0 ≤ Pd.T →
    (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    G.G1 → G.G2 → G.G3 → G.P1 → G.P2 →
    (∀ b, SAKMatmul G P μ (Reg b) (fun _ => pubCost e μ pb.mm (macs b))
      (fun _ => ηw * (Wmm b + Wsha b + Wans b) / Wmm b) κdmm) →
    SAKSha G P nsb Reg μ (fun b => pubCost e μ pb.sha (comps b))
      (fun b => ηw * (Wmm b + Wsha b + Wans b) / Wsha b) κdsha κsha →
    OutsourceWithinEgress G μ (nmax * Wmac) → ProofSystemSound G P nsb Reg verdict dev κS →
    CertifierSeesImports G P nsb dev cin cout → CommitAfterWork G P nsb dev t opArrive →
    RootAfterCoin G P nsb dev t arrive → AnswerAfterWork G P nsb dev sel t ansAt →
    ExtractAdv G P nsb Reg sel Q Badv ξ κx →
    FreedSound G P nsb (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt) dev Pd 1
      (fun A => epsWorkSound (fun b => Wmm b + Wsha b + Wans b)
          (fun b => pubCost e μ pb.mm (macs b)) (fun b => pubCost e μ pb.sha (comps b)) ηw δ
          (nmax * Wmac) (Ein + Eout) +
        (∑ r, μ r * ξ A r) / ∑ b, (Wmm b + Wsha b + Wans b))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun A => δ + κS + κx A + κsha A + nsb * (κdmm (Badv A) + κdsha (Badv A))) ∧
    TimedWorkFits G P nsb (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt) dev
      Pd ∧
    AnswersFit G P nsb (Timed (Counted verdict cin cout Ein Eout) Pd Δ opArrive t arrive sel τans tc ansAt) dev sel Pd

/-- **The headline on the card table `c` at θ**: `WorkSoundAtX` at every semantics `x` meeting (P1), the table's
exactness range (`c.Exact`), and (P2), the product form (`AccelProductForm`). -/
def WorkSoundAt (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) : Prop :=
  ∀ (x : Verity.Accel.Open), c.Exact x → Verity.Assumptions.AccelProductForm x → WorkSoundAtX c pb hr x

/-- **The headline on the card itself**: if the card can do no more than its accelerator model (the card premise,
`Assumptions.CardWithinAccel c can`, for the card's computations `can`), then
at some semantics `x` both hold: `x` meets (P1) and (P2) and the card's runs are the machine `⟨c, x⟩`'s programs
(`CardRunsAt`), and the headline holds at `x` (`WorkSoundAtX`). So the `x` the headline is read at is the card's own,
the one its runs refine to, and the card premise is a hypothesis of the theorem and in the lock.

Open: `G` is still any accelerator that is the devices at `x` (`OnCard`), not the canonical card-level accelerator
whose programs are exactly the card's; the witness that `SAKMatmul` and `SAKSha` hold at such a `G`, with free
guesses, is `Proofs.Pouw.DitherWitness`'s. A sampling block's commitment reads the oracle, which the card has no
instruction for (the realization premise `Assumptions.ShaRealize`). -/
def WorkSoundOnCard (c : Verity.Accel.Card) (pb : PubBest c) (hr : HostRow) : Prop :=
  ∀ can : (c.Mem → c.Mem) → ℚ≥0 → (c.Mem → Prop) → Prop,
    Verity.Assumptions.CardWithinAccel c can →
    ∃ x, c.Exact x ∧ Verity.Assumptions.AccelProductForm x ∧ CardRunsAt c x can ∧ WorkSoundAtX c pb hr x

/-- **The headline on the H200 SXM at θ**: `WorkSoundAt` on `Verity.Accel.H200.card`, with `pubBestH200`'s public
bests on the H200 and the H200 node's hosts. -/
def WorkSoundH200 : Prop := WorkSoundAt Verity.Accel.H200.card pubBestH200 hostH200

/-- **The headline on the RTX PRO 6000 (sm_120) at θ**: `WorkSoundAt` on `Verity.Accel.Sm120.card`, with
`pubBestSm120`'s public bests on the RTX PRO 6000 and its node's hosts. -/
def WorkSoundSm120 : Prop := WorkSoundAt Verity.Accel.Sm120.card pubBestSm120 hostSm120

/-- **The headline on the H200 SXM itself**: `WorkSoundOnCard` on `Verity.Accel.H200.card`, at the semantics
`card-within-accel(H200)` gives, with `pubBestH200`'s public bests on the H200 and the H200 node's hosts. -/
def WorkSoundOnCardH200 : Prop := WorkSoundOnCard Verity.Accel.H200.card pubBestH200 hostH200

/-- **The headline on the RTX PRO 6000 (sm_120) itself**: `WorkSoundOnCard` on `Verity.Accel.Sm120.card`, at the
semantics `card-within-accel(RTX PRO 6000)` gives, with `pubBestSm120`'s public bests on it and its node's hosts. -/
def WorkSoundOnCardSm120 : Prop := WorkSoundOnCard Verity.Accel.Sm120.card pubBestSm120 hostSm120

end Pouw.Guarantees.DitherInstance
