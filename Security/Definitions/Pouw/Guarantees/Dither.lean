import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline, on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, G3, P1, P2) and exactly the program's trust
  base, each a named premise (SAK(matmul), SHA-512's speed, the price of an exchanged bit, the proof service's
  soundness, the network certifier's count, and commit-and-open's extraction), for every prover that runs the period
  one sampling block at a time, each in its commit, challenge and response phases, its operands committed before its
  coins (`Prover`),
  `Pr[accepted ∧ rv + W_other > OPT_honest + ε_A C] ≤ δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm + κ_dsha)`, `C` the
  sampling blocks' honest cost at the dual price.
* `WorkMoment`: `WorkSound` before its Markov step at `δ_coin`, the per-replica moment a fleet pools.
* `WorkSoundNoFold`: the no-fold round with commit-and-open from SHA-512 as a random oracle
  (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) at the query budget the period's capacity allows, so
  one `ε` and one `κ` for every prover.
* `FleetSound`: a fleet of replicas, each its own period, whose moments are pooled by convexity on every set of
  accepted replicas at once, at `ln(N / (s δ_coin))` of luck rather than the union bound's `ln(N / δ_coin)` a replica,
  under `OwnDither` and `CoinsConfidential` (tier A).
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (freed compute). In every accelerator
model `G` of the machine model 𝓜 and for every protocol `P : RoundProto` (the no-fold protocol
is `noFoldRound` at its knobs, so protocol v1 and v2, or a SYM design's sampling blocks, are instances), every
sampling block's set `Reg w` of registrable instances (each block's operands are committed in its commit phase, after
the earlier blocks' coins, so `Reg w` holds every instance a prover may commit for it), every
service verdict on the transcript and the auditor's devices' readings (`Verdict`, `Readings`), every selection `sel`
of a block by its check coin, every period `Pd` whose credited use `hon` is the sampling blocks' honest use (their
commit phases and work and the selected blocks' answers), and every price `μ` that is the period's dual price at one
credit (`Period.DualOpt μ 1`): if
each sampling block's honest cost is `H w = H_mm w + H_sha w + H_o w` at `μ` (its words and commitment, both
positive, and the rest, `H_o w ≥ 0`, its operand root and answer among it), its public bests cost
`pub_mm w ≤ H_mm w` and `pub_sha w ≤ H_sha w` at every instance of `Reg w`, the blocks' honest costs sum to
`μ · hon`, `0 < δ ≤ 1`, `η > 0`, `β, K, ξ ≥ 0`, the degenerate dithers' probabilities are `≥ 0`, `Q` oracle calls use
at least the period's capacity of some resource `r` a call uses (`cap_r ≤ Q callUse_r`, so at most `Q` calls fit,
whatever the price), the devices meet (G1), (G2), (G3), (P1) and (P2), and

* SAK(matmul) holds on block `w`'s words at `μ` on every instance of `Reg w`, at rate `η H w/H_mm w` against
  `pub_mm w`, outside degenerate dithers of probability `κ_dmm(B)` at `B` bits of advice,
* SHA-512's speed holds on the commitment in sampling form at rate `η H w/H_sha w` against `pub_sha w` at every
  committed instance of `Reg w`, outside degenerate dithers of probability `κ_dsha(B)`, on all but a set of the oracle
  and coins of probability `κ_sha(A)`,
* an exchanged bit replaces at most `β` of priced work (`OutsourceWithinEgress`),
* the service's verdict accepts only periods whose sampling blocks' committed instances are registrable and whose
  checks pass, but with probability `κ_S`,
* the network certifier counts at most `K` exchanged bits,
* an accepted period's selected answers run within it (`AnswersFit`), and
* each sampling block's work splits into its words and their commitment on advice of at most `B_adv(A)` bits fixed
  before the block's coins, the commitment's output the prover's root; on runs within the cap `Q` (the commit phases'
  and work's calls, and each selected answer's: `Prover.WithinCap`) the two use at most the prover's timed use plus
  `ξ_A` over the period (the extractor's walk and log), and the check fails on the words at the committed instance
  with probability at most `κ_x(A)` (`ExtractAdv`; from SHA-512 as a random oracle and its realization,
  `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`),

then for every prover `A`, the probability over the oracle and the verifier's coins that the service accepts its
period and other work `z` fits on the period's devices beside its sampling blocks' timed work (`Prover.timedUse`:
their commit phases and work) with
`rv + W_other(z) > OPT_honest + ε_A C` is at most
`δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm(B_adv(A)) + κ_dsha(B_adv(A)))`, with `C = μ · hon = Σ_w H w` and
`ε_A = (Σ_w (H − pub_mm − pub_sha) w + ln(1/δ)/η + β K)/C + μ · ξ_A/C` (`epsWorkSound`).

The rate `η`, per priced cost, is one rate for every sampling block: a block selected by a
Bernoulli coin at a probability proportional to its public cost meets its premises at `η = q/C` at `q` checks, so
the luck term is `ln(1/δ)/q`. The served block is `ditherSelected` (its check coin carries the selection), with
`sel` its selection; there SAK(matmul) at this rate follows from `Pouw.SakLb.SAKResidual` at the luck allowance
`lam_w` once `η H_w / H_mm,w · (pub_mm,w + lam_w) ≤ π` (`Pouw.SakLb.workSound_sakMatmul_selected`). The terms of `ε_A`: the model's slack on the matmul and hashing parts (0 where the
honest kernel and hashing are the public best), the honest work that is neither (no lower bound, full weight), the
sampling luck (`ε_skip`'s at `δ_coin = δ`), the exchanged bits, and the walk and log the extraction adds.

The phases: each sampling block's commit phase commits its operands before any of its coins, the challenge phase's work
commits its words before its check coin, and only the response phase follows the check coin (`Prover`). The cap: the
challenger times each block's commit phase and work, from the previous block's work to the block's commitment, so on the
event bounded they fit the devices, and by (G3) make at most `Q` oracle calls (`calls_le_of_fits` in the proof); a
selected answer runs within the period (`AnswersFit`), so it makes at most `Q` too; `ExtractAdv` is needed only there.
No later program reads an answer, so no answer's use is credited or counted against the period: the honest answers' cost
is in `H_o`, at full weight. The answers' exchanged bits are counted with the work's (`Prover.ext`). `WorkSoundNoFold`
reads it at one query budget and schedule for every prover, so `ε` and `κ` are one bound. The sampling blocks of a
period are one sequence; a fleet whose replicas' coins aren't serialized is one prover per replica (`FleetSound`).

Open (stated so the red team can aim at them):
* the split of a program's use into a words part and a commitment part is `ExtractAdv`'s cost clause (the
  realization's, in `WorkSoundNoFold`), on runs within the cap: the oracle calls are priced steps (G3), but nothing in
  `Accel` says which steps of a program are its commitment path (the facts that would discharge it
  are `Definitions.Pouw.Assumptions.ShaRealize`'s Open);
* the event reads `Fits`, so a run whose timed use doesn't fit the period's devices is outside it. That an accepted
  period's sampling blocks' commit phases and work fit, and its selected answers run within it, is what the
  challenger's order and the deadlines buy: each block's operand root in before its dither coins, its root within
  `Δ` of them, and the next block's commit phase after its work, at any number of blocks:
  the card instance's `Timed`, `TimedWorkFits` and `AnswersFit`;
* `OutsourceWithinEgress` is per program: a program's outside machines see only what that program exchanges. An
  answer reads every coin of the period, so on a node whose outside machines keep state across programs, an answer
  could send out a later block's coins and that block's work import only its root. The bits are counted
  (`Prover.ext`), so the number holds, but the premise at that node would be false. Stating it over the period's
  programs together (exchange-free programs with the same outputs whose total cost is at most the total plus `β` a
  bit exchanged by any of them) would cover it;
* a block's costs and public bests are per sampling block, fixed before the period, and hold at every instance of
  `Reg w`: an instance the prover adapts to the earlier coins gets the cheapest registrable instance's `pub`. A
  registered cost `W_reg` that is itself a budget read off the committed instances is not stated;
* `noFoldRound`'s word isn't protocol v3.2's proof block: its word is a 32-bit prefix sum of the noised
  product (`Word`) with `s` cells checked, while a v3.2 proof block checks one segment's `T_s − T_{s−1}` mod `2^16`
  of the wrapped operands' product, and the last segment's also `O` mod `2^32` and its binding identity. So the
  detection rate's `p`, and which committed values are words, differ, and `WorkSoundNoFold` is not v3.2's statement.
  Until it is restated at v3.2's words, ACCEL's PROTOCOL.md prices `η` at the measured `p ≤ 2.8e-5` and the fleet
  pool's `ζ` at `2^-16`;
* `G` is any accelerator here. The card instance reads it as the card model's machine at a card's θ with its
  floors, prices an exchanged bit at `β = n_max H_mac` with the certifier streaming the dither bits and counting
  `K = E_in + E_out` (which holds only while the dither bits stay streamed), and makes the hosts and DPUs resources
  of `G` (that they are all the processors the cards reach is `PeriodInventory`, held until it is decided whether
  they are devices of the period). -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Fin k → Set P.Inst) (Obs : Type)
    (verdict : Verdict P k Obs) (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho pubmm pubsha : Fin k → ℝ) (η δ β K κS Q : ℝ) (κdmm κdsha : ℕ → ℝ) (sel : P.Chk → Prop)
    (Badv : Prover G P k → ℕ) (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ),
    Pd.DualOpt μ 1 →
    (∀ w, 0 < Hmm w ∧ 0 < Hsha w ∧ 0 ≤ Ho w ∧ pubmm w ≤ Hmm w ∧ pubsha w ≤ Hsha w) →
    ∑ w, (Hmm w + Hsha w + Ho w) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    G.G1 → G.G2 → G.G3 → G.P1 → G.P2 →
    (∀ w, SAKMatmul G P μ (Reg w) (fun _ => pubmm w) (fun _ => η * (Hmm w + Hsha w + Ho w) / Hmm w) κdmm) →
    SAKSha G P k Reg μ pubsha (fun w => η * (Hmm w + Hsha w + Ho w) / Hsha w) κdsha κsha →
    OutsourceWithinEgress G μ β → ProofSystemSound G P k Reg verdict dev κS →
    ImportWithinCertifier G P k verdict dev K → AnswersFit G P k verdict dev sel Pd →
    ExtractAdv G P k Reg sel Q Badv ξ κx →
    FreedSound G P k verdict dev Pd 1
      (fun A => epsWorkSound (fun w => Hmm w + Hsha w + Ho w) pubmm pubsha η δ β K +
        (∑ r, μ r * ξ A r) / ∑ w, (Hmm w + Hsha w + Ho w))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun A => δ + κS + κx A + κsha A + k * (κdmm (Badv A) + κdsha (Badv A)))

/-- **A replica's moment**: `WorkSound` before its Markov step, which `FleetSound` pools across a fleet's
replicas. Under `WorkSound`'s premises without `δ` (and without `η > 0`, which only the luck reads), every prover's
period is sound in moment at rate `η` (`MomentSound`): its sampling blocks' combined priced shortfall `D` against the
public bests has `e^{η D}` below their stopped supermartingale, of average at most 1, off the service's `κ_S`, the
crypto failures `δ_crypto` (`κ_x(A)` of the extraction and `κ_sha(A)` of the oracle) and the degenerate dithers'
`k (κ_dmm + κ_dsha)`; there it frees at most `(ε_A + D / C) C`, with `ε_A` the rest of `ε_skip`: `epsWorkSound` with
no luck term (`ln(1/1)/η = 0`) plus the walk and log `μ · ξ_A / C`. Markov at `δ_coin` gives `WorkSound` back, its
luck `ln(1/δ_coin)/η` added to `ε_A C`. -/
def WorkMoment : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Fin k → Set P.Inst) (Obs : Type)
    (verdict : Verdict P k Obs) (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho pubmm pubsha : Fin k → ℝ) (η β K κS Q : ℝ) (κdmm κdsha : ℕ → ℝ) (sel : P.Chk → Prop)
    (Badv : Prover G P k → ℕ) (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ),
    Pd.DualOpt μ 1 →
    (∀ w, 0 < Hmm w ∧ 0 < Hsha w ∧ 0 ≤ Ho w ∧ pubmm w ≤ Hmm w ∧ pubsha w ≤ Hsha w) →
    ∑ w, (Hmm w + Hsha w + Ho w) = ∑ r, μ r * Pd.hon r →
    0 ≤ β → 0 ≤ K → (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    G.G1 → G.G2 → G.G3 → G.P1 → G.P2 →
    (∀ w, SAKMatmul G P μ (Reg w) (fun _ => pubmm w) (fun _ => η * (Hmm w + Hsha w + Ho w) / Hmm w) κdmm) →
    SAKSha G P k Reg μ pubsha (fun w => η * (Hmm w + Hsha w + Ho w) / Hsha w) κdsha κsha →
    OutsourceWithinEgress G μ β → ProofSystemSound G P k Reg verdict dev κS →
    ImportWithinCertifier G P k verdict dev K → AnswersFit G P k verdict dev sel Pd →
    ExtractAdv G P k Reg sel Q Badv ξ κx →
    MomentSound G P k verdict dev Pd 1
      (fun A => epsWorkSound (fun w => Hmm w + Hsha w + Ho w) pubmm pubsha η 1 β K +
        (∑ r, μ r * ξ A r) / ∑ w, (Hmm w + Hsha w + Ho w))
      (∑ r, μ r * (1 * Pd.hon r)) η
      (fun A => κS + κx A + κsha A + k * (κdmm (Badv A) + κdsha (Badv A)))

/-- **The no-fold protocol is `(1 − ε)`-work-sound at one bound for every prover**.
`WorkSound` on the no-fold round `noFoldRound nf`, with commit-and-open's extraction proved from SHA-512 as a random
oracle (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) rather than assumed. The oracle is a uniform table
`eo : nf.Orc ≃ (X → D)` under which the verifier's opening check is the Merkle path check, programs reach it only
through their calls (`Accel.OracleCalls`), and the realization premise `ShaRealize` holds on runs within the cap `Q` at
one query budget and schedule for every prover. Each sampling block's commit phase and work are timed, so a run within
the cap makes at most `Q` calls in them, each selected answer at most `Q` (`AnswersFit`), and the verifier evaluates
at most `V` points. The log is `k + 1` segments of `M = ⌈Q⌉ + V` queries and `k ⌈Q⌉` more in the last: segment `w`
holds block `w`'s commit phase's and work's calls between block `w − 1`'s root and block `w`'s, padded with dummy
queries, and the last segment the answers' calls and the verifier's evaluations, which no root reads (an answer may
arrive after later roots). So the budget is `q = (k + 1) M + k ⌈Q⌉`, and block `w`'s root comes
after the first `τ_w = (w + 1) M` answers, so the walk of block `w` reads only the queries before its root. At
`τ_w = q` it would also read the queries after the root, which depend on the block's check coin and later ones, while
the words program reads advice fixed before them: a prover that commits a constant root and opens it on a path its
check coin picks has no realization at a `q` below the coin's range. The
premise still asks every prover's words program, on the advice and the dither alone, to be the walk over its log; a
prover whose words or tree read an oracle answer of the block has none at one `B_adv`, so on a machine with such a
prover the premise is false and this statement is vacuous (the walk's oracle reach,
`Definitions.Pouw.Assumptions.ShaRealize`'s Open), and `WorkSound`, at each prover's own bounds, is the one to cite.
With SHA-512's speed at one `κ_sha` and one advice size `B_adv` for every prover, every prover gets the same bound:
`ε = epsWorkSound + μ · (k ξw + q ξl) / C` and
`κ = δ + κ_S + (q² + k q ((k + 1) M + 1)) / |D| + κ_sha + k (κ_dmm(B_adv) + κ_dsha(B_adv))`, with `|D| = 2^512`; the
extraction's term is `ExtractOfRealize`'s `(q² + Σ_w q (2 τ_w + 1)) / |D|` at this schedule.
The log's term `μ · q ξl / C` is `q` log entries' upkeep against the credited cost, the padding included
(the padding's upkeep, `Definitions.Pouw.Assumptions.ShaRealize`'s Open): at `Q = cap_r / callUse_r` it is about
`(2 k + 1) (μ · ξl / shaPub_μ) (μ · cap / C)` when `r` binds, small only where a log entry's upkeep is small next to a
call (`Accel.shaPub`); `WorkSound` keeps each prover's own `q_A`. -/
def WorkSoundNoFold : Prop :=
  ∀ (G : Accel) (nf : NoFold) (k : ℕ) (Reg : Fin k → Set (noFoldRound nf).Inst)
    (Obs : Type) (verdict : Verdict (noFoldRound nf) k Obs) (dev : Readings G (noFoldRound nf) k Obs) (Pd : Period G)
    (μ : G.Res → ℝ) (Hmm Hsha Ho pubmm pubsha : Fin k → ℝ) (η δ β K κS Q κsha : ℝ)
    (κdmm κdsha : ℕ → ℝ) (sel : (noFoldRound nf).Chk → Prop) (Badv V : ℕ) (X D : Type) [Fintype X] [DecidableEq X]
    [Fintype D] [DecidableEq D]
    [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ),
    Pd.DualOpt μ 1 →
    (∀ w, 0 < Hmm w ∧ 0 < Hsha w ∧ 0 ≤ Ho w ∧ pubmm w ≤ Hmm w ∧ pubsha w ≤ Hsha w) →
    ∑ w, (Hmm w + Hsha w + Ho w) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ r, 0 ≤ ξw r) → (∀ r, 0 ≤ ξl r) →
    (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) → (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    (∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c) →
    G.G1 → G.G2 → G.G3 → G.OracleCalls eo → G.P1 → G.P2 →
    (∀ w, SAKMatmul G (noFoldRound nf) μ (Reg w) (fun _ => pubmm w)
      (fun _ => η * (Hmm w + Hsha w + Ho w) / Hmm w) κdmm) →
    SAKSha G (noFoldRound nf) k Reg μ pubsha (fun w => η * (Hmm w + Hsha w + Ho w) / Hsha w) κdsha
      (fun _ => κsha) →
    OutsourceWithinEgress G μ β → ProofSystemSound G (noFoldRound nf) k Reg verdict dev κS →
    ImportWithinCertifier G (noFoldRound nf) k verdict dev K → AnswersFit G (noFoldRound nf) k verdict dev sel Pd →
    ShaRealize G nf k Reg sel Q E eo rD pl dirs ξw ξl (fun _ => (k + 1) * (⌈Q⌉₊ + V) + k * ⌈Q⌉₊)
      (fun _ w => ((w : ℕ) + 1) * (⌈Q⌉₊ + V)) (fun _ => Badv) →
    FreedSound G (noFoldRound nf) k verdict dev Pd 1
      (fun _ => epsWorkSound (fun w => Hmm w + Hsha w + Ho w) pubmm pubsha η δ β K +
        (∑ r, μ r * (k * ξw r + ((k + 1) * (⌈Q⌉₊ + V) + k * ⌈Q⌉₊ : ℕ) * ξl r)) / ∑ w, (Hmm w + Hsha w + Ho w))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun _ => δ + κS + ((((k + 1) * (⌈Q⌉₊ + V) + k * ⌈Q⌉₊) * ((k + 1) * (⌈Q⌉₊ + V) + k * ⌈Q⌉₊) +
            k * (((k + 1) * (⌈Q⌉₊ + V) + k * ⌈Q⌉₊) * ((k + 1) * (⌈Q⌉₊ + V) + 1)) : ℕ) : ℝ) /
          Fintype.card D + κsha + k * (κdmm Badv + κdsha Badv))

/-- **A fleet of replicas, pooled by convexity** (tier A; the sampling blocks of replicas whose coins aren't
serialized are not one sequence). Replica `u` of `N` runs its own period of `k` sampling blocks on
its own devices `Pd u`, under the service's verdict `verdict u` on its transcript and its devices' readings `dev u`.
One oracle serves the fleet. The challenger issues the coins from its randomness `Ω` (`coin`), each replica its own
(`OwnDither`). The strategy the fleet runs on replica `u` (`A u`, at the challenger's randomness) reads every other
replica's coins, past and future (a coalition at its worst), and none of its own (`CoinsConfidential`); what other
replicas compute for it is work it exchanges with machines outside its devices, which the network certifier counts.
If every replica's period is sound in moment at rate `η_u` (`MomentSound`, from `WorkMoment`), with `ε_u` its `ε_skip`
without the luck and `κ_u(A u ω) ≤ κmax_u`, then the probability over the oracle and the challenger's randomness that
some nonempty set `S` of replicas is accepted, with other work `z_u` fitting beside each one's sampling blocks, while
no share `d ≥ 0` of the fleet's luck on `S` (`fleetLuck`) has each `u ∈ S` free at most `(ε_u + d_u / C_u) C_u`, is at
most `δ_coin + Σ_u κmax_u`.

**The fleet is registered before any member's first coin**: its members, their periods `Pd u` and
registered costs `C u` (each replica's `W_reg,u`, their total the fleet's `W_F = Σ_u C_u`), and so their rates `η u`,
are the statement's parameters, fixed before the challenger's randomness `Ω`, and `W_F` is published beside `W_S`. A
replica that starts later joins the next fleet period; a fleet named or grown after any coin is outside the statement,
and named after the outcomes the union over replicas returns. `N` counts the claim's replicas, not the fleet's GPUs over
a replica's size.

So the accepted share is published with its `ε`. A failed replica voids only its own period, and on the accepted set
`S` the fleet frees at most `Σ_{u ∈ S} ε_u C_u + Λ_S (ln (Λ / Λ_S) + ln (1/δ_coin))`, `Λ_S = Σ_{u ∈ S} 1/η_u`, for every
`S` at once: at equal rates `η = q / C`, `ε_u + ln(N / (s δ_coin)) / q` of the accepted credit at `s` accepted
replicas. The union bound over replicas paid `ln(N / δ_coin) / q` at every `s`: at the served parameters (ACCEL's
PROTOCOL.md: `δ_coin = 1e-6`, `q = 13,816` checks a replica-week) the luck's share is 0.1000% with every replica
accepted, against the union's 0.2177% (H200 V4) and 0.2206% (RTX PRO 6000 V4). A headline taking the smaller of this
bound and another tier's runs each at `δ_coin / 2`.

The failure is `δ_coin + Σ_u κmax_u`: pooling takes `ln N` out of the luck, not `Σ_u κ_u` out of the failure.
Each `κ_u` carries its `k` sampling blocks' degenerate dithers, `k (κ_dmm + κ_dsha)`, beside `κ_S` and `δ_crypto`
(`WorkMoment`), so the failure stays near `δ_coin` only within the budget of a replica's `κ` at most `δ_coin / N` and
a sampling block's `κ_dmm + κ_dsha` at most `δ_coin / (N k)`: at the served fleet (`δ_coin = 1e-6`, H200 V4) about
`2^-43` and `2^-73`. No premise yet gives `κ_dmm` or `κ_dsha` a value.

Its rate stays one replica's: replicas that cheat on one shared coin attain it (`Proofs.Pouw.Pool`'s
`convex_pool_tight`). A product over replicas would need (a) moments pointwise in the other replicas' dithers, which
`SAKMatmul`'s average form doesn't give (`not_pool_of_average_moments`), (b) each sampling block's selection
independent after its root and (c) independent dithers; this bound needs only (c), as `OwnDither`. -/
def FleetSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k N : ℕ) (Obs : Type) (verdict : Fin N → Verdict P k Obs)
    (dev : Fin N → Readings G P k Obs) (Pd : Fin N → Period G) (ε κ : Fin N → Prover G P k → ℝ)
    (C η κmax : Fin N → ℝ) (δcoin : ℝ) (Ω : Type) [Fintype Ω] (coin : Ω → Fin N → Fin k → P.Draw × P.Chk)
    (A : Fin N → Ω → Prover G P k),
    (∀ u, 0 < η u) → 0 < δcoin → (∀ u, 0 ≤ κmax u) →
    (∀ u, MomentSound G P k (verdict u) (dev u) (Pd u) 1 (ε u) (C u) (η u) (κ u)) →
    OwnDither P k N coin → CoinsConfidential coin A → (∀ u ω, κ u (A u ω) ≤ κmax u) →
    prob (fun oω : P.Orc × Ω => ∃ S : Finset (Fin N), S.Nonempty ∧ ∃ z : (u : Fin N) → (Pd u).J → ℝ,
      (∀ u ∈ S, (A u oω.2).Passes (verdict u) (dev u) oω.1 (coin oω.2 u) ∧
        (Pd u).Fits ((A u oω.2).timedUse oω.1 (coin oω.2 u)) (z u)) ∧
      ¬ ∃ d : Fin N → ℝ, (∀ u ∈ S, 0 ≤ d u ∧ (Pd u).FreedWithin 1 (z u) (ε u (A u oω.2) + d u / C u) (C u)) ∧
        ∑ u ∈ S, d u ≤ fleetLuck η S δcoin) ≤ δcoin + ∑ u, κmax u

/-- **Freed compute is at most `ε C` at the period's dual price**. On a period's devices (`Period`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`), frees at most
`ε C` with `C = μ · W hon`, the credited rounds' honest cost at the dual price. The floor on the whole
run would not be enough here: it would count the other work `z` toward the credited part's floor. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : Period G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * (W * Pd.hon r))

end Pouw.Guarantees.Dither
