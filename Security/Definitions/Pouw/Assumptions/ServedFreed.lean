import Definitions.Pouw.ServedFreed

/-!
# The onsite PoUW protocol's freed compute: its named premises (trusted)

The premises `Definitions.Pouw.Guarantees.ServedFreed` takes besides the accelerator model's device premises (G1, G2,
P1, P2), the period's dual price and the network certifier's count
(`Definitions.Pouw.Assumptions.Dither.ImportWithinCertifier`):

* `RemoteWorkFloor`: the remote form's conclusion in this vocabulary. A prover whose whole inventory is the registered
  devices, and whose period the work check accepts, spent at least `F` of priced timed work on them, but with
  probability `κ`;
* `OutsourceWithinEgressPeriod`: an exchanged bit replaces at most `β` of priced timed work, over the period's
  programs together (RT-POOL RP21), the one premise that prices outside machines.

**Non-vacuity rests on the fixed model.** RM1 (note:compute-research/20261027T0035Z-draft-rt-model-min) shows that 𝓜
as `Definitions.Core.Accel` writes it makes `SAKMatmul` and `SAKResidual` false (free register reductions run the
four Russians' method below the public best's cost), so `RemoteWorkFloor`, which rests on them, is false there at any
useful `F`. A guarantee that takes it says something only on the model MODEL-MIN fixes (CMP-165).
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-served-work-floor` (proposed; the remote form, WORK-H1's `Properties.Pouw.ServedWork.ServedWork`, CMP-145).
**An accepted period's exchange-free work is at least `F` at the dual price**: in the model at its device premises,
for every prover whose programs exchange no bits with machines outside the registered devices (`Prover.ExchangeFree`:
its whole inventory is those devices), the probability over the oracle and the verifier's coins that the work check
`vw` accepts its transcript and its sampling blocks' timed work costs less than `F` at the prices `μ` is at most `κ`.

It is the remote form's conclusion, at one credit for the whole accepted period: at the served check (H1, J10-R,
`PouwCheck.Credit.window`) WORK-H1's `ServedWork` gives `F = p · L − Λ`, `p` the public best's priced cost a credited
word, `L` the accepted period's `(1 − ε(j))₊` share of its live words and `Λ` the windows' luck allowances, and
`κ = δ_s + n (δ_bind + δ_ks) + n κ_deg(B)` at the advice bound `B` that (G1)'s memory puts on every prover's state. It
is taken here as a premise until that guarantee's paths and window programs are this vocabulary's oracle, coins and
provers.

It rests on SAK for matrix products at the served words (`SAKServed`, `Pouw.SakLb.SAKResidual`), on the proof system's
soundness (`ProofSystemSound` in prose; J10-R's `TileServiceSound`), and, where the commitment's hashing is credited,
on SAK for SHA-512 (`SAKSha` in prose; `ShaSpeed` in Lean). Under RM1 it is false of 𝓜 as written (module doc).

It is falsified by an exchange-free strategy on the registered devices whose period the work check accepts, with
probability more than `κ`, at a priced timed work below `F`. -/
def RemoteWorkFloor (G : Accel) (P : RoundProto) (k : ℕ) (vw : Verdict P k Unit) (μ : G.Res → ℝ) (F κ : ℝ) :
    Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k, A.ExchangeFree →
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => A.Passes vw (noReadings G P k) ox.1 ox.2 ∧
      ∑ r, μ r * A.timedUse ox.1 ox.2 r < F) ≤ κ

/-- `pouw-outsource-egress-period` (proposed; class L, RT-POOL RP21; `pouw-outsource-egress` over the period's programs
together). **An exchanged bit replaces at most `β` of priced timed work, over a whole period**: in the model at its
device premises, every prover has one that exchanges no bits with machines outside the registered devices
(`Prover.ExchangeFree`), sends the same transcript on every oracle and coins, and whose sampling blocks' timed work
costs at most the prover's plus `β` a bit the prover's programs exchanged, its answers' among them (`Prover.ext`).

**Why over the period** (RP21). An answer reads every coin of the period, so it can send a later sampling block's
coins out, and an outside machine that keeps state across programs can return that block's root to its work. That
work then imports only the root, so the per-program `OutsourceWithinEgress` is false at such a node: the work's
exchange-free replacement does the block's words and hashing, worth `H_b`, against `β` times the root's 512 bits.
Here the answer's exported coins count too. The per-program form implies this one
(`Pouw.SecurityProofs.outsourcePeriod_of_perProgram`), so it asks no more of 𝓜.

**β** (RT-PROTO-2 P2, P18). The premise holds only at `β` at least a sampling block's honest priced cost over the
fewest bits that carry its coins out and its root back, `β ≥ H_b / (|coin_b| + 512)`: a block whose coins are a
512-bit salt costs about `H_b / 1024` a bit. The guarantee holds at whatever `β` this premise holds at, and its error
carries `β K`.

**The coins stay in the boundary** (RT-POOL RP4). It is a premise about exchanged bits, so it is claimed where a coin
reaches a machine outside the registered devices only through a bit their programs exchange: the coins are
confidential (`CoinsConfidential`). For the MVP that holds by placement (Daniel's RP4 ruling): the device that runs a
server's network certifier also produces that server's live coins, and private messages travel only on private links
(`WitnessPathPrivate`, Daniel, 26 Oct, 4:41 PM PDT), so no coin leaves the boundary the certifier counts but through
the links it counts. Trust in that device and its host provider is `trusted-host`. A coin readable outside the
boundary falsifies the premise: an outside machine computes the block's words from it and returns the root in 512
bits. -/
def OutsourceWithinEgressPeriod (G : Accel) (P : RoundProto) (k : ℕ) (μ : G.Res → ℝ) (β : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ A : Prover G P k, ∃ A' : Prover G P k, A'.ExchangeFree ∧ ∀ o x,
    A'.transcript o x = A.transcript o x ∧
      ∑ r, μ r * A'.timedUse o x r ≤ ∑ r, μ r * A.timedUse o x r + β * A.ext o x

end Pouw.Assumptions
