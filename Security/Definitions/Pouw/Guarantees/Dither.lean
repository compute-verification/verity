import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline (D2), on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, G3, P1, P2) and exactly the program's trust
  base, each a named premise (SAK(matmul), SHA-512's speed, the price of an exchanged bit, the proof service's
  soundness, the network certifier's count, and commit-and-open's extraction), for every prover that runs the period
  one round at a time on instances registered before it (`Prover`),
  `Pr[accepted ∧ rv + W_other > OPT_honest + ε_A C] ≤ δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm + κ_dsha)`, `C` the
  registered rounds' honest cost at the dual price.
* `WorkSoundNoFold`: the no-fold round with commit-and-open from SHA-512 as a random oracle
  (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) at the query budget the period's capacity allows, so
  one `ε` and one `κ` for every prover.
* `FleetSound`: a fleet of units, each its own period whose coins reach only it, by the union bound over the units.
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (freed compute, `PeriodDevices.FreedWithin`).
In every accelerator model `G` and for every round protocol `P` (the no-fold protocol is `noFoldRound` at its knobs, so
protocol v1 and v2, or a SYM design's rounds, are instances), every set `Reg` of registrable instances and every
period's registered instances `reg w ∈ Reg`, every service verdict on the transcript and the auditor's devices'
readings (`Acceptance`, `Readings`), every period `Pd` whose credited unit `hon` is the registered sampling blocks' honest
use (their work and the selected blocks' answers), and every price `μ` that is the period's dual price at that unit
(`PeriodDevices.DualOpt μ 1`): if
each registrable instance's honest round costs `H = H_mm + H_sha + H_o` at `μ` (its words and commitment, both
positive, and the rest, `H_o ≥ 0`, its answers among it),
its public bests cost `pub_mm ≤ H_mm` and `pub_sha ≤ H_sha`, the registered rounds' honest costs sum to `μ · hon`,
`0 < δ ≤ 1`, `η > 0`, `β, K, ξ ≥ 0`, the degenerate dithers' probabilities are `≥ 0`, `Q` oracle calls use at least
the period's capacity of some resource `r` a call uses (`cap_r ≤ Q callUse_r`, so at most `Q` calls fit, whatever the
price), the devices meet (G1), (G2), (G3), (P1) and (P2), and

* SAK(matmul) holds on the words at `μ` on every registrable instance, at rate `η H/H_mm` against `pub_mm`, outside
  degenerate dithers of probability `κ_dmm(B)` at `B` bits of advice,
* SHA-512's speed holds on the commitment in sampling form at rate `η H/H_sha` against `pub_sha`, outside degenerate
  dithers of probability `κ_dsha(B)`, on all but a set of the oracle and coins of probability `κ_sha(A)`,
* an exchanged bit replaces at most `β` of priced work (`OutsourceWithinEgress`),
* the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`,
* the network certifier counts at most `K` exchanged bits, and
* each round's work splits into its words and their commitment on advice of at most `B_adv(A)` bits fixed before the
  round's coins, the commitment's output the prover's root; on runs whose rounds make at most `Q` oracle calls the
  two use at most the prover's timed use plus `ξ_A` over the period (the extractor's walk and log), and the check
  fails on the words with probability at most `κ_x(A)` (`ExtractAdv`; from SHA-512 as a random oracle and its
  realization,
  `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`),

then for every prover `A`, the probability over the oracle and the verifier's coins that the service accepts its
period and other work `z` fits on the period's devices beside its sampling blocks' timed work (`Prover.timedUse`)
with
`rv + W_other(z) > OPT_honest + ε_A C` is at most
`δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm(B_adv(A)) + κ_dsha(B_adv(A)))`, with `C = μ · hon = Σ_w H(reg w)` and
`ε_A = (Σ_w (H − pub_mm − pub_sha)(reg w) + ln(1/δ)/η + β K)/C + μ · ξ_A/C` (`epsWorkSound`).

The rate `η` is one rate a unit of priced cost for every round: a round selected by a Bernoulli
coin at a probability proportional to its public cost meets its premises at `η = k_draw/C`, so the luck term is
`ln(1/δ)/k_draw`. The terms of `ε_A`: the model's slack on the matmul and hashing parts (0 where the honest kernel
and hashing are the public best), the honest work that is neither (no lower bound, full weight), the sampling luck,
the exchanged bits, and the walk and log the extraction adds.

The cap: the challenger times each sampling block's work, from its dither coins to its commitment, so
on the event bounded the blocks' work fits the devices, and by (G3) it makes at most `Q` oracle calls
(`calls_le_of_fits` in the proof); `ExtractAdv` is needed only there. Answers are off the state path, so no answer's
use or calls are read: the honest answers' cost is in `H_o`, at full weight. Their exchanged bits are counted with the
work's (`Prover.ext`). `WorkSoundNoFold` reads it at one query budget and schedule for every prover,
`k + 1` segments of `⌈Q⌉ + V` queries, so `ε` and `κ` are one bound. The sampling blocks of a period are one sequence;
a fleet whose replicas' coins aren't serialized is one prover per replica (`FleetSound`).

Open (stated so the red team can aim at them):
* the split of a program's use into a words part and a commitment part is `ExtractAdv`'s cost clause (the
  realization's, in `WorkSoundNoFold`), on runs within the cap: the oracle calls are priced steps (G3), but nothing in
  `Accel` says which steps of a program are its commitment path (the facts that would discharge
  it are `Definitions.Pouw.Assumptions.ShaRealize`'s Open);
* the event reads `Fits`, so a run whose timed use doesn't fit the period's devices is outside it. That an accepted
  period's sampling blocks' work fits is what the challenger's order and the deadlines buy: each block's root within
  `Δ` of its coins, and the next block's coins only after it, at any number of blocks: the
  card instance's `Timed` and `TimedWorkFits`;
* `OutsourceWithinEgress` is per program: a program's outside machines see only what that program exchanges. An
  answer reads every coin of the period, so on a node whose outside machines keep state across programs, an answer
  could send out a later block's coins and that block's work import only its root. The bits are counted
  (`Prover.ext`), so the number holds, but the premise at that node would be false. Stating it over the period's
  programs together (exchange-free programs with the same outputs whose total cost is at most the total plus `β` a
  bit exchanged by any of them) would cover it;
* between a block's root and the next block's coins no program runs: `absorb` uses nothing and `work (b + 1)` starts
  at `t (b + 1)` (`RootAfterCoin`). A protocol that commits a block's operands in that gap (ACCEL's coin order) has an
  honest prover outside the model. A phase before each block's coins, timed from the previous root to the operand
  roots' arrival, would close it, and the windows would still telescope to `T`;
* `G` is any accelerator here. The card instance reads it as MODEL-SPEC's machine at a card's θ with its floors,
  prices an exchanged bit at `β = n_max H_mac` with the certifier streaming the
  dither bits and counting `K = E_in + E_out` (which holds only while the dither bits stay streamed), and
  makes the hosts and DPUs resources of `G` (that they are all the processors the cards reach is
  `PeriodInventory`, held open). -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Set P.Inst) (reg : Fin k → P.Inst) (Obs : Type)
    (verdict : Acceptance P k Obs) (dev : Readings G P k Obs) (Pd : PeriodDevices G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho pubmm pubsha : P.Inst → ℝ) (η δ β K κS Q : ℝ) (κdmm κdsha : ℕ → ℝ) (Badv : Prover G P k → ℕ)
    (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubmm a ≤ Hmm a ∧ pubsha a ≤ Hsha a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    G.G1 → G.G2 → G.G3 → G.P1 → G.P2 →
    SAKMatmul G P μ Reg pubmm (fun a => η * (Hmm a + Hsha a + Ho a) / Hmm a) κdmm →
    ShaSpeed G P k reg μ pubsha (fun a => η * (Hmm a + Hsha a + Ho a) / Hsha a) κdsha κsha →
    OutsourceWithinEgress G μ β → ServiceSound G P k reg verdict dev κS →
    ImportWithinCertifier G P k verdict dev K → ExtractAdv G P k reg Q Badv ξ κx →
    FreedSound G P k verdict dev Pd 1
      (fun A => epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
          (fun w => pubsha (reg w)) η δ β K +
        (∑ r, μ r * ξ A r) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun A => δ + κS + κx A + κsha A + k * (κdmm (Badv A) + κdsha (Badv A)))

/-- **The no-fold protocol is `(1 − ε)`-work-sound at one bound for every prover**.
`WorkSound` on the no-fold round `noFoldRound nf`, with commit-and-open's extraction proved from SHA-512 as a random
oracle (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) rather than assumed. The oracle is a uniform table
`eo : nf.Orc ≃ (X → D)` under which the verifier's opening check is the Merkle path check, programs reach it only
through their calls (`Accel.OracleCalls`), and the realization premise `ShaRealize` holds on runs within the cap `Q` at
one query budget and schedule for every prover. Each sampling block's work is timed, so a run within the cap makes at
most `Q` calls in its work, and the verifier evaluates at most `V` points. The log is `k + 1` segments of
`M = ⌈Q⌉ + V` queries: segment `w` holds the work's calls and the verifier's evaluations between round `w − 1`'s root
and round `w`'s (an answer may arrive after later roots, and its evaluations sit where it arrives), padded
with dummy queries, and the last segment holds those after the last root. So the budget is `q = (k + 1) M`, and
round `w`'s root comes after the first `τ_w = (w + 1) M` answers, so the walk of round `w` reads only the queries
before its root. At `τ_w = q` it would also read the queries after the root, which depend on the round's check coins
and later ones, while the words program reads advice fixed before them: a prover that commits a constant root and
opens it on a path its check coin picks has no realization at a `q` below the coin's range (WORKSOUND-WIT's WIT1).
The premise still asks every prover's words program, on the advice and the dither alone, to be the walk over its log; a
prover whose words or tree read an oracle answer of the round has none at one `B_adv`, so on a machine with such
a prover the premise is false and this statement is vacuous (the walk's oracle reach,
`Definitions.Pouw.Assumptions.ShaRealize`'s Open), and `WorkSound`, at each prover's own bounds, is the one to cite.
The same holds on a machine with a prover whose openings read oracle answers it got by calls in its answer phase:
answers are untimed, so the cap doesn't bound those calls and this budget doesn't count them (the answers' calls, the
same Open).
With SHA-512's speed at one `κ_sha` and one advice size `B_adv` for every prover, every prover gets the same bound:
`ε = epsWorkSound + μ · (k ξw + q ξl) / C` and
`κ = δ + κ_S + (q² + k q (q + 1)) / |D| + κ_sha + k (κ_dmm(B_adv) + κ_dsha(B_adv))`, with `|D| = 2^512`; the
extraction's term is `ExtractOfRealize`'s `(q² + Σ_w q (2 τ_w + 1)) / |D|` at this schedule.
The log's term `μ · q ξl / C` is `(k + 1) M` log entries' upkeep against the credited cost, the padding included
(the padding's upkeep, `Definitions.Pouw.Assumptions.ShaRealize`'s Open): at `Q = cap_r / callUse_r` it is about
`(k + 1) (μ · ξl / shaPub_μ) (μ · cap / C)` when `r` binds, small only where a log entry's upkeep is small next to a
call (`Accel.shaPub`); `WorkSound` keeps each prover's own `q_A`. -/
def WorkSoundNoFold : Prop :=
  ∀ (G : Accel) (nf : NoFold) (k : ℕ) (Reg : Set (noFoldRound nf).Inst) (reg : Fin k → (noFoldRound nf).Inst)
    (Obs : Type) (verdict : Acceptance (noFoldRound nf) k Obs) (dev : Readings G (noFoldRound nf) k Obs) (Pd : PeriodDevices G)
    (μ : G.Res → ℝ) (Hmm Hsha Ho pubmm pubsha : (noFoldRound nf).Inst → ℝ) (η δ β K κS Q κsha : ℝ)
    (κdmm κdsha : ℕ → ℝ) (Badv V : ℕ) (X D : Type) [Fintype X] [DecidableEq X] [Fintype D] [DecidableEq D]
    [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubmm a ≤ Hmm a ∧ pubsha a ≤ Hsha a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ r, 0 ≤ ξw r) → (∀ r, 0 ≤ ξl r) →
    (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) → (∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r) →
    (∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c) →
    G.G1 → G.G2 → G.G3 → G.OracleCalls eo → G.P1 → G.P2 →
    SAKMatmul G (noFoldRound nf) μ Reg pubmm (fun a => η * (Hmm a + Hsha a + Ho a) / Hmm a) κdmm →
    ShaSpeed G (noFoldRound nf) k reg μ pubsha (fun a => η * (Hmm a + Hsha a + Ho a) / Hsha a) κdsha
      (fun _ => κsha) →
    OutsourceWithinEgress G μ β → ServiceSound G (noFoldRound nf) k reg verdict dev κS →
    ImportWithinCertifier G (noFoldRound nf) k verdict dev K →
    ShaRealize G nf k reg Q E eo rD pl dirs ξw ξl (fun _ => (k + 1) * (⌈Q⌉₊ + V))
      (fun _ w => ((w : ℕ) + 1) * (⌈Q⌉₊ + V)) (fun _ => Badv) →
    FreedSound G (noFoldRound nf) k verdict dev Pd 1
      (fun _ => epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
          (fun w => pubsha (reg w)) η δ β K +
        (∑ r, μ r * (k * ξw r + ((k + 1) * (⌈Q⌉₊ + V) : ℕ) * ξl r)) /
          ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun _ => δ + κS + (((k + 1) * (⌈Q⌉₊ + V) * ((k + 1) * (⌈Q⌉₊ + V)) +
            k * ((k + 1) * (⌈Q⌉₊ + V) * ((k + 1) * (⌈Q⌉₊ + V) + 1)) : ℕ) : ℝ) /
          Fintype.card D + κsha + k * (κdmm Badv + κdsha Badv))

/-- **A fleet of units, each its own period** (the rounds of units whose coins aren't serialized are
not one sequence). Unit `u` of `N` runs its own period of `k` rounds on its own devices `Pd u`, under the service's
verdict `verdict u` on its transcript and its devices' readings `dev u`. One oracle serves the fleet, and each unit
draws its own coins, which reach no other machine before that unit has them. Unit `u`'s prover `A u` may read every
other unit's coins, past and future (a coalition of units at its worst); what other units compute for it is work it
exchanges with machines outside its devices, which the network certifier counts. If every unit's period is
`(1 − ε_u(A))`-work-sound but with probability `κ_u(A)` for every prover `A` (`WorkSound`, or `WorkSoundNoFold` at
one bound for every prover), and `κ_u(A u y) ≤ κmax_u` at every value `y` of the other units' coins, the probability
over the oracle and every unit's coins that some unit's period is accepted while it frees more than
`ε_u(A u y) C_u` is at most `Σ_u κmax_u`. Each unit at `δ/N`, so a luck term `ln(N/δ)/η` in its `ε_u`, gives the
fleet `δ` for the luck, with `k` rounds a unit. -/
def FleetSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k N : ℕ) (Obs : Type) (verdict : Fin N → Acceptance P k Obs)
    (dev : Fin N → Readings G P k Obs) (Pd : Fin N → PeriodDevices G) (ε κ : Fin N → Prover G P k → ℝ)
    (C κmax : Fin N → ℝ),
    (∀ u, 0 ≤ κmax u) → (∀ u, FreedSound G P k (verdict u) (dev u) (Pd u) 1 (ε u) (C u) (κ u)) →
    ∀ A : (u : Fin N) → ({v : Fin N // v ≠ u} → Fin k → P.Draw × P.Chk) → Prover G P k,
      (∀ u y, κ u (A u y) ≤ κmax u) →
      prob (fun ox : P.Orc × (Fin N → Fin k → P.Draw × P.Chk) => ∃ u,
        (A u fun v => ox.2 v.1).Passes (verdict u) (dev u) ox.1 (ox.2 u) ∧
          ∃ z, (Pd u).Fits ((A u fun v => ox.2 v.1).timedUse ox.1 (ox.2 u)) z ∧
            ¬ (Pd u).FreedWithin 1 z (ε u (A u fun v => ox.2 v.1)) (C u)) ≤ ∑ u, κmax u

/-- **Freed compute is at most `ε C` at the period's dual price**. On a period's devices (`PeriodDevices`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`), frees at most
`ε C` with `C = μ · W hon`, the credited rounds' honest cost at the dual price. The floor on the whole
run would not be enough here: it would count the other work `z` toward the credited part's floor. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : PeriodDevices G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * (W * Pd.hon r))

end Pouw.Guarantees.Dither
