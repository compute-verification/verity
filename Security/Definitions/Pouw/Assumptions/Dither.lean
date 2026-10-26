import Definitions.Claims
import Definitions.Pouw.Dither

/-!
# The no-fold Dither protocol's work-soundness: its named premises (trusted)

The named premises of `Definitions.Pouw.Guarantees.Dither.WorkSound` besides the accelerator model (G1, G2, P1, P2) and
commit-and-open's extraction (`Definitions.Pouw.Assumptions.ShaRealize.ExtractAdv`). They are the program's trust base
(section 0) and nothing else:

* `SAKMatmul`: symmetric algorithmic knowledge for exact matrix products (R-ACCEL-6), on the round's words, for
  programs that exchange no bits with outside machines;
* `ShaSpeed`: SHA-512's implementation speed (R-ACCEL-7), on the round's commitment, in sampling form, per prover
  (RT-LEAN-2 L37);
* `OutsourceWithinEgress`: an exchanged bit replaces at most `β` of priced work (RT-PROTO-2 P2, P18; RT-LEAN-2 L32),
  the one premise that prices outside machines;
* `ServiceSound`: the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`
  (C-Flock sound and zero knowledge, live coins from the auditor's challenger);
* `ImportWithinCertifier`: an accepted period exchanged at most `K` bits, the network certifier's count (class L).

`Definitions.Pouw.Guarantees.Dither.FleetSound` adds two on the challenger's coins across a fleet's replicas (CMP-152):

* `OwnDither`: the challenger issues each replica its own coins, uniform and independent of every other replica's
  (RT-POOL's `IndependentCoins` across the fleet, from the coins' derivation, RP25);
* `CoinsConfidential`: a replica's coins reach no machine outside it before it has them, so the strategy run on it
  reads none of them (for the MVP, by one device that produces a server's coins and runs its certifier; RP4).

Only `SAKMatmul`, `ShaSpeed`, `OutsourceWithinEgress` and `ExtractAdv` say anything about which algorithms a prover
runs.

**Advice.** A round's programs read advice fixed before the round's coins, of at most `B` bits (an injection into
`Fin (2 ^ B)`). The premises hold outside a degenerate set of the round's dithers of probability at most `κdeg B`
(finding W2): where the dither is degenerate (for the no-fold round, noise that leaves the product's structure, such
as zero noise), advice that holds the un-noised product answers the check for free, so no premise of the exact form
`moment ≤ 1` holds against it. `κdeg B` is the probability of the dithers on which some advice of `B` bits beats the
public best; it enters `κ` once a round.
-/

namespace Pouw.Assumptions

open Pouw.RelativeSecurity Pouw.Dither

/-- `pouw-sak-matmul` (proposed; R-ACCEL-6, Daniel 25 Oct). **Symmetric algorithmic knowledge for exact matrix
products, SAK(matmul)**, on a round's words: in the model at its device premises, on every registrable instance
`a ∈ Reg`, every words program that exchanges no bits with outside machines, on any advice `s` of at most `B` bits
fixed before the round's coins, has moment at most 1 at the prices `μ` and rate `η a` against the public best's
priced cost `pub a`, outside a set `Deg` of the round's dithers of probability at most `κdeg B`:
`E_σ[1_{d ∉ Deg ∧ pass(σ)} · e^{η (pub − cost_μ)₊}] ≤ 1`.

`pub a` is the public best matmul method's priced cost of the instance's words (RT-LEAN L3) at the period's dual
prices: no program computes the noised products' segment prefix sums (or all but a small fraction of the words the
check could draw) more cheaply at `μ` than the best public method. Exchanged bits are priced by
`OutsourceWithinEgress`, not here (RT-LEAN-2 L32).

**What the advice holds** (RT-PROTO-2 P14). Anything computed before the round within (G1)'s memory, so `pub` must hold
against that precomputation (O-IO's lookup routes):
* the un-noised product `A B` and the periodic part of the noise;
* **the reused weight-side coins**: with weight-side coins reused across decode calls (A11), `W'` is in the instance
  (RT-PROTO-2 P17), so a later round's program holds advice that depends on `W'`, and SAK must hold for programs that
  preprocess one operand (the matrix–vector-with-preprocessing class). The known speedups are log-factor tables, and
  they lose at int8 width (A4, A11);
* **the cheapest registrable instance**: `Reg` is every instance the protocol lets a prover register (RT-LEAN-2 L27),
  so a prover that registers the instance whose words are cheapest at `μ` gets that instance's `pub`, no less;
* **D5's menu with advice** (ASK-147 item 4): `pub` is the cheapest of D5's public programs priced with their advice,
  at the dual prices.

**Degenerate dithers** (RT-PROTO-2 P17, finding W2). On a reuse round the reused coins are worst case over `Reg`, and
the fresh dither is the round's `Draw`. A dither that makes the noised product's words computable from the advice
(the zero noise, `W' = B` with `B = 0`) is in `Deg`; the operands stay worst case.

**The rate** (RT-LEAN-2 L38). A program that skips a fraction `f` of the words and guesses them passes a checked cell
it skipped with probability `p` (a free guess: `2^{-32}` for a uniform word, more where the words are structured), so
the premise holds only at `η · pub ≤ π s (1 − p)`, `π` the round's selection probability (RT-PROTO-2 P15, P20) and `s`
its checked cells. -/
def SAKMatmul (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (Reg : Set P.Inst) (pub η : P.Inst → ℝ)
    (κdeg : ℕ → ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ a ∈ Reg, ∀ (B : ℕ) (St : Type), (∃ e : St → Fin (2 ^ B), Function.Injective e) →
    ∀ (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr), (∀ i, G.ext Mw i = 0) →
      ∃ Deg : P.Draw → Prop, prob Deg ≤ κdeg B ∧ wordsMoment G P μ (pub a) (η a) a s Mw Deg ≤ 1

/-- `pouw-sha512-speed` (proposed; R-ACCEL-7, Daniel 25 Oct: "sure"). **Nobody computes SHA-512 meaningfully faster
than the best public implementation**, on a round's commitment, in sampling form and per prover (RT-LEAN-2 L37): in
the model at its device premises (with (G3): an oracle call is a step priced at the public implementation's use), for
every prover `A` on the registered instances `reg`, every advice of at most `B`
bits on which each round's programs read nothing of that round's coins or later, and every words and commitment
program of each round that exchange no bits with outside machines, where the commitment program's output is `A`'s
root as an equation (condition 1), the probability over the oracle and the verifier's coins that some round's
commitment moment exceeds 1 outside every set of dithers of probability at most `κdeg B` is at most `κsha A`
(condition 3: the oracle's luck, per prover).

The commitment moment is the sampling form (`commitMoment`): the average over the round's coins of
`1_{d ∉ Deg ∧ pass ∧ opens} · e^{η (pub − cost_μ(Mh))₊}`, where `opens` is the verifier's opening check of the cells the
check coin selects under `Mh`'s root (condition 2: from the round's `opens`, never prover-supplied). A program that
hashes part of the tree is caught where the check selects a cell outside that part. A program that hashes a fraction
`1 − f` of the leaves passes with probability about `(1 − f π)^s`, so the premise holds only at `η · pub ≤ π s`
(condition 4), `π` the round's selection probability.

`pub a` is the pinned public implementation's priced cost of the instance's commitment at the period's dual prices.
Under the random oracle it is ABSORB-THM's realization: each oracle query costs at least the public implementation's,
so the honest prover meets it with shortfall 0. MODEL-SPEC's θ floors (RT-MODEL M12, on `Open.sync`, `b1` and `madHi`)
are what keep the model from hashing faster than the card. The premise is meant for models whose programs reach the
oracle only through their calls (`Accel.OracleCalls` at the round's table, RT-LEAN-2 L28): where a program reads the
table as data it is false, and the headline says nothing. -/
def ShaSpeed (G : Accel) (P : RoundProto) (k : ℕ) (reg : Fin k → P.Inst) (μ : G.Res → ℝ) (pub η : P.Inst → ℝ)
    (κdeg : ℕ → ℝ) (κsha : Prover G P k → ℝ) : Prop :=
  G.G1 → G.G2 → G.G3 → G.P1 → G.P2 → ∀ (A : Prover G P k) (B : ℕ) (Adv : Type),
    (∃ e : Adv → Fin (2 ^ B), Function.Injective e) →
    ∀ (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv),
    (∀ o x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv o x w = adv o x' w) →
    ∀ (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root),
    (∀ w i, G.ext (Mw w) i = 0) → (∀ w i, G.ext (Mh w) i = 0) →
    (∀ o x w, G.Mc.eval (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) = A.root o x w) →
    prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => ∃ w, ¬ ∃ Deg : P.Draw → Prop, prob Deg ≤ κdeg B ∧
      commitMoment G P μ (pub (reg w)) (η (reg w)) ox.1 (reg w) (adv ox.1 ox.2 w) (Mw w) (Mh w) Deg ≤ 1) ≤ κsha A

/-- `pouw-outsource-egress` (proposed; class L, conditional on the network certifier streaming the call's coin side
`b` itself, RT-PROTO-2 P2 and P18; RT-LEAN-2 L32). **An exchanged bit replaces at most `β` of priced work**: in the
model at its device premises, every program has one that computes the same thing on every input, exchanges no bits
with outside machines, and costs at most its cost plus `β` a bit it exchanged (`Accel.ext`, in either direction,
other than the certifier's own coin stream).

A node that forwards a call's `b` (`m K` bits) and gets back only the root has the outside machine do the call's
`m K N` MACs, so `β = n_max · H/macs` with `n_max = N` for a static call and `d_v` for PV, `H = H_mm + H_sha + H_o`
(the outside machine returns the root, so it saves the round's hashing too: RT-PROTO-2 P18's addition). `SAKMatmul` and
`ShaSpeed` are stated over programs that exchange nothing, and this premise carries their bound to every program at
`β` a bit (WORKSOUND-INST-2's form, on every program type). It is per program: an answer that sends a later block's
coins out is a different program from that block's work, which is in `WorkSound`'s Open (RT-POOL RP21). -/
def OutsourceWithinEgress (G : Accel) (μ : G.Res → ℝ) (β : ℝ) : Prop :=
  G.G1 → G.G2 → G.P1 → G.P2 → ∀ {I O : Type} (f : G.Mc.Alg I O), ∃ f' : G.Mc.Alg I O, ∀ i,
    G.Mc.eval f' i = G.Mc.eval f i ∧ G.ext f' i = 0 ∧ G.cost μ f' i ≤ G.cost μ f i + β * G.ext f i

/-- `pouw-nofold-service` (proposed; `pouw-rs-staged-service`'s form, per prover, on a period of rounds). **The proof
service's verdict accepts only periods whose rounds' checks pass** on the registered instances `reg`, but with
probability `κ_S` over the oracle and the verifier's coins, for every prover: the rounds' checks run inside C-Flock's
zero-knowledge proofs (sound and zero knowledge), on live coins drawn by the auditor's challenger after each round's
commitment (`trusted-host`). The verdict reads the transcript and the devices' readings, never the prover's
code (RT-LEAN-2 L29). It assumes nothing of the prover's algorithms. -/
def ServiceSound (G : Accel) (P : RoundProto) (k : ℕ) (reg : Fin k → P.Inst) {Obs : Type}
    (verdict : Verdict P k Obs) (dev : Readings G P k Obs) (κS : ℝ) : Prop :=
  ∀ A : Prover G P k, prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
    A.Passes verdict dev ox.1 ox.2 ∧ ¬ A.Accepted reg ox.1 ox.2) ≤ κS

/-- `pouw-dither-import` (proposed; class L, the network certifier's count on `trusted-host`). **An accepted
period exchanged at most `K` bits**: the network certifier sits on the period's devices' links, on a machine the
prover doesn't control, and signs the bits that cross them in either direction (its readings `dev`); the period's
verdict reads its signed count and refuses a period past `K`. The certifier's own coin stream isn't counted
(RT-PROTO-2 P18). Open (RT-PROTO P2, O-IO): the per-period form from the certifier's counts with `n_max · E_out`. -/
def ImportWithinCertifier (G : Accel) (P : RoundProto) (k : ℕ) {Obs : Type} (verdict : Verdict P k Obs)
    (dev : Readings G P k Obs) (K : ℝ) : Prop :=
  ∀ (A : Prover G P k) (o : P.Orc) (x : Fin k → P.Draw × P.Chk), A.Passes verdict dev o x → A.ext o x ≤ K

/-- `pouw-own-dither` (proposed; the challenger on `trusted-host`, RT-POOL RP2(c), RP25, CMP-152). **Each
replica's coins are its own**: the challenger issues every replica of a fleet of `N` its own dither and check coins,
drawn fresh for it, so its randomness `Ω` gives every replica's coins together the uniform distribution (`coin`
carries the uniform measure on `Ω` to the uniform measure on `Fin N → Fin k → Draw × Chk`). With each replica's own
coins uniform, as `MomentSound`'s sample space has them, this is RT-POOL's `IndependentCoins` across the fleet.

It rests on the coins' derivation (RP25): the challenger derives each coin as
`derive(K_ch, "verity/pouw-accel/<kind>/v1", {period, replica, sampling block index, root receipt digest})`, its key
`K_ch` never leaving the challenger, so no two of the fleet's coins share a context, and the derivation is taken as
uniform. `verity.ml.randomness.derive` is deterministic, so a coin keyed by its roots alone would repeat with them.

A dither broadcast to several replicas puts a replica's own dither in the others' coins, which `FleetSound` hands to
its strategy, and then its moment fails. The check coins are in it too: `FleetSound`'s strategies read every other
replica's coins, so a check coin shared with another replica would reach a replica's strategy before the challenger
issues it. -/
def OwnDither (P : RoundProto) (k N : ℕ) {Ω : Type} [Fintype Ω] (coin : Ω → Fin N → Fin k → P.Draw × P.Chk) :
    Prop :=
  ∀ f : (Fin N → Fin k → P.Draw × P.Chk) → ℝ, avg (fun ω => f (coin ω)) = avg f

/-- `pouw-coins-confidential` (proposed; the network certifiers on `trusted-host`, RT-POOL RP4, CMP-152). **A
replica's coins reach no machine outside it before it has them**: a sampling block's dither and check coin are
readable only inside its replica's certifier boundary until its root. For the MVP it holds by one device (Daniel's
RP4 ruling, 26 Oct): the device that runs a server's network certifier also produces that server's live coins, so the
challenger issues them inside the boundary the certifier counts, and trust in that device and its host provider is
`trusted-host`.

So the strategy a fleet runs on replica `u` (its prover `A u ω` at the challenger's randomness `ω`) may read every
other replica's coins, past and future (a coalition at its worst), but none of its own: those reach `u` only through
its own programs, in the challenger's order, as `Prover` has them. A machine outside the boundary that read a coin
early could serve the block's words, work no extraction sees. -/
def CoinsConfidential {G : Accel} {P : RoundProto} {k N : ℕ} {Ω : Type} (coin : Ω → Fin N → Fin k → P.Draw × P.Chk)
    (A : Fin N → Ω → Prover G P k) : Prop :=
  ∀ u ω ω', (∀ v, v ≠ u → coin ω v = coin ω' v) → A u ω = A u ω'

end Pouw.Assumptions
