import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline (CMP-136, D2), on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, G3, P1, P2) and exactly the program's trust
  base, each a named premise (SAK(matmul), SHA-512's speed, the price of an exchanged bit, the proof service's
  soundness, the network certifier's count, and commit-and-open's extraction), for every prover that runs the period
  one round at a time on instances registered before it (`Prover`),
  `Pr[accepted ∧ rv + W_other > OPT_honest + ε_A C] ≤ δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm + κ_dsha)`, `C` the
  registered rounds' honest cost at the dual price.
* `WorkSoundNoFold`: the no-fold round with commit-and-open from SHA-512 as a random oracle
  (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) at the query budget the period's capacity allows, so
  one `ε` and one `κ` for every prover (RT-LEAN-2 L39).
* `FleetSound`: a fleet of units, each its own period whose coins reach only it, by the union bound over the units
  (RT-PROTO-2 P16).
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Pouw.MerkleExtract

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (R-ACCEL-1's freed compute; CMP-136).
In every accelerator model `G` and for every round protocol `P` (the no-fold protocol is `noFoldRound` at its knobs, so
protocol v1 and v2, or a SYM design's rounds, are instances), every set `Reg` of registrable instances and every
period's registered instances `reg w ∈ Reg`, every service verdict on the transcript and the auditor's devices'
readings (`Verdict`, `Readings`), every period `Pd` whose credited unit `hon` is the registered rounds' honest use, and
every price `μ` that is the period's dual price at that unit (`Period.DualOpt μ 1`): if each registrable instance's
honest round costs `H = H_mm + H_sha + H_o` at `μ` (its words and commitment, both positive, and the rest, `H_o ≥ 0`),
its public bests cost `pub_mm ≤ H_mm` and `pub_sha ≤ H_sha`, the registered rounds' honest costs sum to `μ · hon`,
`0 < δ ≤ 1`, `η > 0`, `β, K, ξ ≥ 0`, the degenerate dithers' probabilities are `≥ 0`, an oracle call costs
`shaPub_μ > 0` and `Q` calls cost at least the period's capacity (`μ · cap ≤ Q shaPub_μ`, so at most `Q` calls fit:
RT-LEAN-2 L39), the devices meet (G1), (G2), (G3), (P1) and (P2), and

* SAK(matmul) holds on the words at `μ` on every registrable instance, at rate `η H/H_mm` against `pub_mm`, outside
  degenerate dithers of probability `κ_dmm(B)` at `B` bits of advice,
* SHA-512's speed holds on the commitment in sampling form at rate `η H/H_sha` against `pub_sha`, outside degenerate
  dithers of probability `κ_dsha(B)`, on all but a set of the oracle and coins of probability `κ_sha(A)`,
* an exchanged bit replaces at most `β` of priced work (`OutsourceWithinEgress`),
* the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`,
* the network certifier counts at most `K` exchanged bits, and
* each round's work splits into its words and their commitment on advice of at most `B_adv(A)` bits fixed before the
  round's coins, the commitment's output the prover's root, within the prover's work plus `ξ_A` over the period (the
  extractor's walk and log), but with probability `κ_x(A)` on runs whose work makes at most `Q` oracle calls
  (`ExtractAdv`; from SHA-512 as a random oracle and its realization,
  `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`),

then for every prover `A`, the probability over the oracle and the verifier's coins that the service accepts its
period and other work `z` fits on the period's devices beside its rounds' work (`Prover.workUse`) with
`rv + W_other(z) > OPT_honest + ε_A C` is at most `δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm(B_adv(A)) + κ_dsha(B_adv(A)))`,
with `C = μ · hon = Σ_w H(reg w)` and
`ε_A = (Σ_w (H − pub_mm − pub_sha)(reg w) + ln(1/δ)/η + β K)/C + μ · ξ_A/C` (`epsWorkSound`).

The rate `η` is one rate a unit of priced cost for every round (RT-PROTO-2 P15, P20): a round selected by a Bernoulli
coin at a probability proportional to its public cost meets its premises at `η = k_draw/C`, so the luck term is
`ln(1/δ)/k_draw`. The terms of `ε_A`: the model's slack on the matmul and hashing parts (0 where the honest kernel
and hashing are the public best), the honest work that is neither (no lower bound, full weight), the sampling luck,
the exchanged bits, and the walk and log the extraction adds.

The cap (L39): on the event bounded, the prover's work fits the devices, so by (G3) it makes at most `Q` oracle calls
(`calls_le_of_fits` in the proof); `ExtractAdv` is needed only there. `WorkSoundNoFold` reads it at a query budget
`⌈Q⌉ + V` that serves every prover, so `ε` and `κ` are one bound. The rounds of a period are one sequence; a fleet
whose units' coins aren't serialized is one prover per unit (`FleetSound`, RT-PROTO-2 P16).

Open (stated so the red team can aim at them):
* the split of a program's use into a words part and a commitment part is `ExtractAdv`'s cost clause (the
  realization's, in `WorkSoundNoFold`): the oracle calls are priced steps (G3), but the machine's composition doesn't
  yet say which steps of a program are its commitment path (RT-LEAN-2 L28's remainder);
* `G` is the abstract interface, not yet MODEL-SPEC's machine at a card's θ with its floors (RT-LEAN L4, L9;
  RT-MODEL M12);
* import is per period with no `n_max · E_out` form (RT-PROTO P2);
* host CPUs and DPUs are resources of `G` only if the model prices them (P9);
* the coin-to-root deadline and the period's boundary aren't read (P5). -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Set P.Inst) (reg : Fin k → P.Inst) (Obs : Type)
    (verdict : Verdict P k Obs) (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho pubmm pubsha : P.Inst → ℝ) (η δ β K κS Q : ℝ) (κdmm κdsha : ℕ → ℝ) (Badv : Prover G P k → ℕ)
    (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubmm a ≤ Hmm a ∧ pubsha a ≤ Hsha a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    0 < G.shaPub μ → ∑ r, μ r * Pd.cap r ≤ Q * G.shaPub μ →
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

/-- **The no-fold protocol is `(1 − ε)`-work-sound at one bound for every prover** (RT-LEAN-2 L39; CMP-136).
`WorkSound` on the no-fold round `noFoldRound nf`, with commit-and-open's extraction proved from SHA-512 as a random
oracle (`Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`) rather than assumed. The oracle is a uniform table
`eo : nf.Orc ≃ (X → D)` under which the verifier's opening check is the Merkle path check, programs reach it only
through their calls (`Accel.OracleCalls`), and the realization premise `ShaRealize` holds on runs within the cap `Q` at
the query budget and schedule `q = ⌈Q⌉ + V` for every prover, `V` the verifier's evaluations: a run within the cap
makes at most `Q` calls, so its commitment path's distinct queries, with the verifier's, are at most `q`, and a
prover can't buy a larger `q` by padding. With SHA-512's speed at one `κ_sha` and one advice size `B_adv` for every
prover, every prover gets the same bound:
`ε = epsWorkSound + μ · (k ξw + q ξl) / C` and
`κ = δ + κ_S + (q² + k q (2q + 1)) / |D| + κ_sha + k (κ_dmm(B_adv) + κ_dsha(B_adv))`, with `|D| = 2^512`.
The log's term `μ · q ξl / C` is about `(μ · ξl / shaPub_μ) (μ · cap / C)`: one bound for every prover is small only
where a log entry's upkeep is small next to a call; `WorkSound` keeps each prover's own `q_A`. -/
def WorkSoundNoFold : Prop :=
  ∀ (G : Accel) (nf : NoFold) (k : ℕ) (Reg : Set (noFoldRound nf).Inst) (reg : Fin k → (noFoldRound nf).Inst)
    (Obs : Type) (verdict : Verdict (noFoldRound nf) k Obs) (dev : Readings G (noFoldRound nf) k Obs) (Pd : Period G)
    (μ : G.Res → ℝ) (Hmm Hsha Ho pubmm pubsha : (noFoldRound nf).Inst → ℝ) (η δ β K κS Q κsha : ℝ)
    (κdmm κdsha : ℕ → ℝ) (Badv V : ℕ) (X D : Type) [Fintype X] [DecidableEq X] [Fintype D] [DecidableEq D]
    [Inhabited D] (E : Enc X D Word) (eo : nf.Orc ≃ (X → D)) (rD : nf.Root → D) (pl : nf.Path → List (Bool × D × D))
    (dirs : Fin nf.n × Fin nf.n' × Fin nf.nb → List Bool) (ξw ξl : G.Res → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubmm a ≤ Hmm a ∧ pubsha a ≤ Hsha a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ r, 0 ≤ ξw r) → (∀ r, 0 ≤ ξl r) →
    (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) → 0 < G.shaPub μ → ∑ r, μ r * Pd.cap r ≤ Q * G.shaPub μ →
    (∀ o r c v p, nf.opens o r c v p → Opens (eo o) E (rD r) (pl p) v ∧ (pl p).map Prod.fst = dirs c) →
    G.G1 → G.G2 → G.G3 → G.OracleCalls eo → G.P1 → G.P2 →
    SAKMatmul G (noFoldRound nf) μ Reg pubmm (fun a => η * (Hmm a + Hsha a + Ho a) / Hmm a) κdmm →
    ShaSpeed G (noFoldRound nf) k reg μ pubsha (fun a => η * (Hmm a + Hsha a + Ho a) / Hsha a) κdsha
      (fun _ => κsha) →
    OutsourceWithinEgress G μ β → ServiceSound G (noFoldRound nf) k reg verdict dev κS →
    ImportWithinCertifier G (noFoldRound nf) k verdict dev K →
    ShaRealize G nf k reg Q E eo rD pl dirs ξw ξl (fun _ => ⌈Q⌉₊ + V) (fun _ _ => ⌈Q⌉₊ + V) (fun _ => Badv) →
    FreedSound G (noFoldRound nf) k verdict dev Pd 1
      (fun _ => epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
          (fun w => pubsha (reg w)) η δ β K +
        (∑ r, μ r * (k * ξw r + (⌈Q⌉₊ + V : ℕ) * ξl r)) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun _ => δ + κS + (((⌈Q⌉₊ + V) * (⌈Q⌉₊ + V) + k * ((⌈Q⌉₊ + V) * (2 * (⌈Q⌉₊ + V) + 1)) : ℕ) : ℝ) /
          Fintype.card D + κsha + k * (κdmm Badv + κdsha Badv))

/-- **A fleet of units, each its own period** (RT-PROTO-2 P16: the rounds of units whose coins aren't serialized are
not one sequence). Unit `u` of `N` runs its own period of `k` rounds on its own devices `Pd u`, under the service's
verdict `verdict u` on its transcript and its devices' readings `dev u`. One oracle serves the fleet, and each unit
draws its own coins, which reach no other machine before that unit has them. Unit `u`'s prover `A u` may read every
other unit's coins, past and future (a coalition of units at its worst); what other units compute for it is work it
exchanges with machines outside its devices, which the network certifier counts. If every unit's period is
`(1 − ε_u)`-work-sound at one bound `κ_u ≥ 0` for every prover (`WorkSoundNoFold`), the probability over the oracle and
every unit's coins that some unit's period is accepted while it frees more than `ε_u C_u` is at most `Σ_u κ_u`. Each
unit at `δ/N`, so a luck term `ln(N/δ)/η` in its `ε_u`, gives the fleet `δ` for the luck, with `k` rounds a unit. -/
def FleetSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k N : ℕ) (Obs : Type) (verdict : Fin N → Verdict P k Obs)
    (dev : Fin N → Readings G P k Obs) (Pd : Fin N → Period G) (ε C κ : Fin N → ℝ),
    (∀ u, 0 ≤ κ u) → (∀ u, FreedSound G P k (verdict u) (dev u) (Pd u) 1 (fun _ => ε u) (C u) (fun _ => κ u)) →
    ∀ A : (u : Fin N) → ({v : Fin N // v ≠ u} → Fin k → P.Draw × P.Chk) → Prover G P k,
      prob (fun ox : P.Orc × (Fin N → Fin k → P.Draw × P.Chk) => ∃ u,
        (A u fun v => ox.2 v.1).Passes (verdict u) (dev u) ox.1 (ox.2 u) ∧
          ∃ z, (Pd u).Fits ((A u fun v => ox.2 v.1).workUse ox.1 (ox.2 u)) z ∧
            ¬ (Pd u).FreedWithin 1 z (ε u) (C u)) ≤ ∑ u, κ u

/-- **Freed compute is at most `ε C` at the period's dual price** (R-ACCEL-1). On a period's devices (`Period`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`), frees at most
`ε C` with `C = μ · W hon`, the credited rounds' honest cost at the dual price (RT-LEAN L25). The floor on the whole run would not be enough here:
it would count the other work `z` toward the credited part's floor. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : Period G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * (W * Pd.hon r))

end Pouw.Guarantees.Dither
