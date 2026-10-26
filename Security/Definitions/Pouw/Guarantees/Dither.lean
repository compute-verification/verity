import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither

/-!
# The no-fold Dither protocol's work-soundness over a monitoring period (trusted)

The ACCEL program's headline (CMP-136, D2), on the accelerator model of `Definitions.Pouw.Dither`.

* `WorkSound`: under the accelerator model's device premises (G1, G2, P1, P2) and exactly the program's trust base,
  each a named premise (SAK(matmul), SHA-512's speed, the proof service's soundness, the network certifier's import
  count, and commit-and-open's extraction), freed compute over a monitoring period of `k` rounds is at most `ε C` except
  with probability at most each adversary's `breakBound` at `δ`:
  `Pr[accepted ∧ W_cred + W_other > OPT_honest + ε C] ≤ δ + k (δb + δz) + finder terms`, with
  `ε = (1 − h)(1 − c_mm) + h (1 − c_sha) + ln(1/δ)/(η k) + β K/(k H)`.
* `FreedOfFloor`: the rounds' floor at the period's dual price bounds freed compute by `ε C` (weak duality); the step
  from the floor to `WorkSound`'s conclusion.
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **The no-fold protocol is `(1 − ε)`-work-sound over a monitoring period** (R-ACCEL-1's freed compute; CMP-136).
In every accelerator model `G` and for every round protocol `P` (the no-fold protocol is `noFoldRound` at its knobs, so
protocol v1 and v2 are instances), every service `S` whose seed is the period's `k` rounds of live coins (`e`), every
analysis map `run`, every period `Pd` of the devices with its jobs, and every price `μ` that is the period's dual
price at `k` credited rounds (`Period.DualOpt`): if the honest round costs `H = H_mm + H_sha` at `μ` (its words and its
commitment, both positive), the parameters are in range (`c_mm, c_sha ≤ 1`, the honest kernel and hashing being
public methods; `0 < δ ≤ 1`; `η > 0`; `β, K ≥ 0`), the devices meet (G1), (G2), (P1) and (P2), and

* SAK(matmul) holds on the words at `μ` with constant `c_mm`, rate `η` and import weight `β` against `H_mm`,
* SHA-512's speed holds on the commitment likewise with constant `c_sha` against `H_sha`,
* the service is sound up to its proof layer per prover,
* the network certifier counts at most `K` imported bits, and
* the analysis map extracts computing rounds from every accepted commit-and-open run, within it,

then for every adversary that commits first, the probability that its period is accepted and other work `z` fits on
the period's devices beside its credited rounds with `k + W_other(z) > OPT_honest + ε C` (`C = μ · cap`, the period's
devices at `μ`) is at most its own `breakBound` at the luck `δ`, where
`ε = (1 − h)(1 − c_mm) + h (1 − c_sha) + ln(1/δ)/(η k) + β K/(k H)`, `h = H_sha / H`.

The terms: the model's slack on the matmul share and on the hashing share (0 where the honest kernel and the honest
hashing are the public best), the sampling luck (`η = 1 − p` at the per-round luck `p`), and the network import. -/
def WorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (S : Service) (k : ℕ) (e : S.Seed ≃ Paths (P.Draw × P.Chk) k)
    (run : PeriodMap G S P k) (Pd : Period G) (μ : G.Res → ℝ) (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders G.Mc S k) (cmm csha η δ β Hmm Hsha K δb βb δz βz cb db cz dz : ℝ),
    Pd.DualOpt μ k → 0 < Hmm → 0 < Hsha → Hmm + Hsha = ∑ r, μ r * Pd.hon r →
    cmm ≤ 1 → csha ≤ 1 → 0 < η → 0 < δ → δ ≤ 1 → 0 ≤ β → 0 ≤ K → 0 ≤ βb → 0 ≤ βz →
    G.G1 → G.G2 → G.P1 → G.P2 →
    SAKMatmul G P μ cmm η β Hmm → ShaSpeed G P μ csha η β Hsha →
    ServiceSound G S P run e sha fb fz δb βb δz βz → ImportWithinCertifier G S K → Extract G S P run e →
    FreedSound G S P run e Pd k (epsWorkSound cmm csha Hmm Hsha η δ β K k) (∑ r, μ r * Pd.cap r)
      (breakBound sha fb fz cb db cz dz δ δb βb δz βz)

/-- **Freed compute is at most `ε C` at the period's dual price** (R-ACCEL-1). On a period's devices (`Period`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`), frees at most
`ε C` with `C = μ · cap`, the period's devices at the dual price. The floor on the whole run would not be enough here:
it would count the other work `z` toward the credited part's floor. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : Period G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * Pd.cap r)

end Pouw.Guarantees.Dither
