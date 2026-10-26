import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline (CMP-136, D2), on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, P1, P2) and exactly the program's trust base,
  each a named premise (SAK(matmul), SHA-512's speed, the price of an exchanged bit, the proof service's soundness,
  the network certifier's count, and commit-and-open's extraction), for every prover that runs the period one round
  at a time on instances registered before it (`Prover`),
  `Pr[accepted ∧ rv + W_other > OPT_honest + ε_A C] ≤ δ + κ_S + κ_x(A) + κ_sha(A) + k (κ_dmm + κ_dsha)`, `C` the
  registered rounds' honest cost at the dual price.
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (R-ACCEL-1's freed compute; CMP-136).
In every accelerator model `G` and for every round protocol `P` (the no-fold protocol is `noFoldRound` at its knobs, so
protocol v1 and v2, or a SYM design's rounds, are instances), every set `Reg` of registrable instances and every
period's registered instances `reg w ∈ Reg`, every service verdict on the transcript and the auditor's devices'
readings (`Verdict`, `Readings`), every period `Pd` whose credited unit `hon` is the registered rounds' honest use, and
every price `μ` that is the period's dual price at that unit (`Period.DualOpt μ 1`): if each registrable instance's
honest round costs `H = H_mm + H_sha + H_o` at `μ` (its words and commitment, both positive, and the rest, `H_o ≥ 0`),
its public bests cost `pub_mm ≤ H_mm` and `pub_sha ≤ H_sha`, the registered rounds' honest costs sum to `μ · hon`,
`0 < δ ≤ 1`, `η > 0`, `β, K, ξ ≥ 0`, the degenerate dithers' probabilities are `≥ 0`, the devices meet (G1), (G2),
(P1) and (P2), and

* SAK(matmul) holds on the words at `μ` on every registrable instance, at rate `η H/H_mm` against `pub_mm`, outside
  degenerate dithers of probability `κ_dmm(B)` at `B` bits of advice,
* SHA-512's speed holds on the commitment in sampling form at rate `η H/H_sha` against `pub_sha`, outside degenerate
  dithers of probability `κ_dsha(B)`, on all but a set of the oracle and coins of probability `κ_sha(A)`,
* an exchanged bit replaces at most `β` of priced work (`OutsourceWithinEgress`),
* the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`,
* the network certifier counts at most `K` exchanged bits, and
* each round's work splits into its words and their commitment on advice of at most `B_adv(A)` bits fixed before the
  round's coins, the commitment's output the prover's root, within the prover's work plus `ξ_A` over the period (the
  extractor's walk and log), but with probability `κ_x(A)` (`ExtractAdv`; from SHA-512 as a random oracle and its
  realization, `Definitions.Pouw.Guarantees.ShaRealize.ExtractOfRealize`),

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

Open (stated so the red team can aim at them):
* the rounds of a period are one sequence: a fleet whose units' coins aren't serialized is one prover per unit, and the
  per-unit corollary by the union bound at `δ/N` is to come (RT-PROTO-2 P16);
* the oracle calls aren't yet a separately priced step of the machine, so `ExtractAdv`'s cost split is the
  realization's premise, and the cap on a prover's own calls by its capacity is to come (RT-LEAN-2 L28, L39);
* `G` is the abstract interface, not yet MODEL-SPEC's machine at a card's θ with its floors (RT-LEAN L4, L9;
  RT-MODEL M12);
* import is per period with no `n_max · E_out` form (RT-PROTO P2);
* host CPUs and DPUs are resources of `G` only if the model prices them (P9);
* the coin-to-root deadline and the period's boundary aren't read (P5). -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (Reg : Set P.Inst) (reg : Fin k → P.Inst) (Obs : Type)
    (verdict : Verdict P k Obs) (dev : Readings G P k Obs) (Pd : Period G) (μ : G.Res → ℝ)
    (Hmm Hsha Ho pubmm pubsha : P.Inst → ℝ) (η δ β K κS : ℝ) (κdmm κdsha : ℕ → ℝ) (Badv : Prover G P k → ℕ)
    (ξ : Prover G P k → G.Res → ℝ) (κx κsha : Prover G P k → ℝ),
    (∀ w, reg w ∈ Reg) → Pd.DualOpt μ 1 →
    (∀ a ∈ Reg, 0 < Hmm a ∧ 0 < Hsha a ∧ 0 ≤ Ho a ∧ pubmm a ≤ Hmm a ∧ pubsha a ≤ Hsha a) →
    ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) = ∑ r, μ r * Pd.hon r →
    0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → (∀ A r, 0 ≤ ξ A r) → (∀ B, 0 ≤ κdmm B ∧ 0 ≤ κdsha B) →
    G.G1 → G.G2 → G.P1 → G.P2 →
    SAKMatmul G P μ Reg pubmm (fun a => η * (Hmm a + Hsha a + Ho a) / Hmm a) κdmm →
    ShaSpeed G P k reg μ pubsha (fun a => η * (Hmm a + Hsha a + Ho a) / Hsha a) κdsha κsha →
    OutsourceWithinEgress G μ β → ServiceSound G P k reg verdict dev κS →
    ImportWithinCertifier G P k verdict dev K → ExtractAdv G P k reg Badv ξ κx →
    FreedSound G P k verdict dev Pd 1
      (fun A => epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
          (fun w => pubsha (reg w)) η δ β K +
        (∑ r, μ r * ξ A r) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))
      (∑ r, μ r * (1 * Pd.hon r))
      (fun A => δ + κS + κx A + κsha A + k * (κdmm (Badv A) + κdsha (Badv A)))

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
