import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither
import Definitions.Pouw.Assumptions.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline (CMP-136, D2), on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, P1, P2) and exactly the program's trust base,
  each a named premise (SAK(matmul), SHA-512's speed, the proof service's soundness, the network certifier's import
  count, and commit-and-open's extraction), for every prover that runs the period one round at a time (`Prover`),
  `Pr[accepted ∧ W_cred + W_other > OPT_honest + ε C] ≤ δ + κ_S + κ_x`, `C = k H` the credited rounds' honest cost,
  `ε = (H_mm (1 − c_mm) + H_sha (1 − c_sha) + H_o)/H + ln(1/δ)/(η k) + β K/(k H)`.
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (R-ACCEL-1's freed compute; CMP-136).
In every accelerator model `G` and for every round protocol `P` (the no-fold protocol is `noFoldRound` at its knobs, so
protocol v1 and v2, or a SYM design's rounds, are instances), every service verdict, every period `Pd` of the devices
with its jobs, and every price `μ` that is the period's dual price at `k` credited rounds (`Period.DualOpt`): if the
honest round costs `H = H_mm + H_sha + H_o` at `μ` (its words, its commitment, both positive, and the rest, `H_o ≥ 0`),
the public bests cost `pub_mm ≤ H_mm` and `pub_sha ≤ H_sha` (so `c = pub/H ≤ 1`), `0 < δ ≤ 1`, `η > 0`, `β, K ≥ 0`, the
devices meet (G1), (G2), (P1) and (P2), and

* SAK(matmul) holds on the words at `μ` with constant `pub_mm/H_mm`, rate `η` and import weight `β` against `H_mm`,
* SHA-512's speed holds on the commitment likewise with constant `pub_sha/H_sha` against `H_sha`,
* the service's verdict accepts only periods whose rounds' checks pass, but with probability `κ_S`,
* the network certifier counts at most `K` imported bits, and
* each round's work splits into its words, on advice fixed before the round's coins, and their commitment, but with
  probability `κ_x` (`ExtractAdv`),

then for every prover, the probability over the oracle and the verifier's coins that the service accepts its period
and other work `z` fits on the period's devices beside its rounds' use with `k rv + W_other(z) > OPT_honest + ε C`
is at most `δ + κ_S + κ_x`, with `C = μ · (k hon) = k H` (RT-LEAN L25) and
`ε = (H_mm (1 − c_mm) + H_sha (1 − c_sha) + H_o)/H + ln(1/δ)/(η k) + β K/(k H)`.

The terms: the model's slack on the matmul and hashing shares (0 where the honest kernel and hashing are the public
best), the honest work that is neither (no lower bound, full weight), the sampling luck, and the network import. Open
(stated so the red team can aim at them): `G` is the abstract interface, not yet MODEL-SPEC's machine at a card's θ
with its floors (RT-LEAN L4, L9; RT-MODEL M12); import is per period with no `n_max · E_out` form (RT-PROTO P2);
host CPUs and DPUs are resources of `G` only if the model prices them (P9); the coin-to-root deadline and the
period's boundary aren't read (P5). -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (k : ℕ) (verdict : Verdict G P k) (Pd : Period G) (μ : G.Res → ℝ)
    (pubmm pubsha Hmm Hsha Ho η δ β K κS κx : ℝ),
    Pd.DualOpt μ k → 0 < Hmm → 0 < Hsha → 0 ≤ Ho → Hmm + Hsha + Ho = ∑ r, μ r * Pd.hon r →
    pubmm ≤ Hmm → pubsha ≤ Hsha → 0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K →
    G.G1 → G.G2 → G.P1 → G.P2 →
    SAKMatmul G P μ (pubmm / Hmm) η β Hmm → ShaSpeed G P μ (pubsha / Hsha) η β Hsha →
    ServiceSound G P k verdict κS → ImportWithinCertifier G P k verdict K → ExtractAdv G P k κx →
    FreedSound G P k verdict Pd k (epsWorkSound (pubmm / Hmm) (pubsha / Hsha) Hmm Hsha Ho η δ β K k)
      (∑ r, μ r * (k * Pd.hon r)) (δ + κS + κx)

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
