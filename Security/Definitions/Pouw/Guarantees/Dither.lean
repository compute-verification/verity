import Definitions.Pouw.Dither
import Definitions.Pouw.Assumptions.Dither

/-!
# Dither's work-soundness over a monitoring period (trusted)

The ACCEL program's final guarantee (CMP-116, DITHER-THM), on the accelerator model of `Definitions.Pouw.Dither`, and
the freed-compute bound it gives (R-ACCEL-1).

* `DitherWorkSound`: under the device premises (G1, G2, P1, P2), SAK(matmul) at the constant `c = pub/H` (the one
  computational premise, R-ACCEL-6: Theorem O's moment bound and the direct-sum cut), the
  proof service's soundness per prover (`ServiceSoundDither`: C-Flock sound and zero knowledge, SHA-512 only through
  each adversary's finders, live coins from the auditor's challenger) and the network certifier's import count
  (`ImportWithinCertifier`), a monitoring period of `k` rounds is `(1 − ε)`-work-sound at every price vector, with
  `ε = (1 − c) + D/k + β K/(k H)`: model slack, sampling luck, network import. The floor holds on the rounds the
  adversary's run is credited for (`RoundsFloor`) and so on its whole run (`WorkSound`). Each adversary's error is its
  own (`breakBound` at the luck `e^{−η D}`).
* `FreedOfFloor`: the floor at the dual price of the period's linear program bounds freed compute by `ε C`, `C` the
  period's devices at that price (`Period.FreedWithin`).

**What it replaces.** The floors of record (`StagedFloorKOneSided` and its siblings) rest on Theorem K's scope
(`KScopeOneSided`, `KScopeAffine`: which programs a prover runs) and on `ChargeVF3j16` (what the machine charges a
product). `DitherWorkSound` takes neither: its premises are the device model, the service and the network certifier,
and SAK(matmul), the one computational premise (R-ACCEL-6), whose constant `c = pub/H` reads the public best matmul
method's cost, which the ring count (`RingKCount`) proves on the tensor cores.
WORD-WORK's per-word premise (`WordCostsMacs`, CMP-102) is replaced the same way: the credit it multiplies by MACs a
word is here a priced cost, read off the model. The map from the old floors and from #4363's
`H1CheckedCreditPeriod` is in the note (note:compute-research/20261026T0130Z-draft-accel-dither-thm).
-/

namespace Pouw.Guarantees.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions

/-- **Dither is `(1 − ε)`-work-sound over a monitoring period** (DITHER-THM). In every accelerator model `G` and for
every round protocol `P` (today's Dither is `ditherRound` at its knobs), every service `S` whose seed is the period's
`k` rounds of live coins (`e`), every price vector `λ` and every analysis map `run`: if the devices meet (G1), (G2),
(P1) and (P2), SAK(matmul) holds at `λ` with constant `c`, rate `η` and import weight `β` against the honest round's
priced cost `H` (Theorem O's moment bound, and the cut `run` is a direct sum: an accepted run's rounds are pieces of
it), the service
is sound up to its proof layer per prover, and the network certifier counts at most `K` imported bits, then every
adversary that commits first has its accepted periods whose rounds are
priced below `(1 − ε) k H` with probability at most its own `breakBound` at the luck `e^{−η D}` (`RoundsFloor`), and so
its accepted periods priced below `(1 − ε) k H` as a whole (`WorkSound`), where `ε = (1 − c) + D/k + β K/(k H)`. The
rounds' floor is the one freed compute reads (`FreedOfFloor`): other work the adversary runs beside its rounds is not
counted toward it.

The three terms: the model's slack `1 − c` (SAK(matmul)'s constant; 0 where the honest kernel is the public best),
the sampling luck `D/k` (at `δ = e^{−η D}`, `D = ln(1/δ)/η`, with `η = 1 − p` at the per-round luck `p`), and the
network import `β K/(k H)`. `H` is free, so the statement holds at the honest kernel's priced cost of a round, where
it reads as `(1 − ε)`-work-soundness, and Theorem O at `c` forces `c H` below every always-passing program's cost.
Beside the data sheets' device premises, the service's cryptography, the auditor's live coins and the network
certifier, the statement reads one statement over every program of the model, SAK(matmul). Where honest time includes
SHA-512 hashing at share `h`, `c` is the matmul constant on the rest and `ShaSpeed` the hashing's (the note's
\(1 - c = (1 - h)(1 - c_{mm}) + h(1 - c_{sha})\)). -/
def DitherWorkSound : Prop :=
  ∀ (G : Accel) (P : RoundProto) (S : Service) (k : ℕ) (e : S.Seed ≃ Paths (P.Draw × P.Chk) k)
    (run : PeriodMap G S P k) (p : G.Res → ℝ) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders G.Mc S k)
    (c η β H K D δb βb δz βz cb db cz dz : ℝ),
    Simplex p → 0 < H → 0 ≤ η → 0 ≤ β → 0 ≤ δb → 0 ≤ βb → 0 ≤ δz → 0 ≤ βz →
    G.G1 → G.G2 → G.P1 → G.P2 → SAKMatmul G S P run e p c η β H →
    ServiceSoundDither G S P run e sha fb fz δb βb δz βz → ImportWithinCertifier G S K →
    let ε := epsDither c D β K H k
    let κ := breakBound sha fb fz cb db cz dz (Real.exp (-(η * D))) δb βb δz βz
    RoundsFloor G S P run e p ((1 - ε) * (k * H)) κ ∧ WorkSound G S p (k * H) ε κ

/-- **Freed compute is at most `ε C` at the period's dual price** (R-ACCEL-1). On a period's devices (`Period`), let
`μ ≥ 0` be a dual price for the other work (every job worth at most its use at `μ`) at which an honest strategy
credited `W` with other work `z'` is optimal (its other work worth exactly the capacity it leaves, at `μ`:
complementary slackness). Any strategy credited the same `W`, whose credited part uses `X` and whose other work `z`
fits beside it, and whose credited part meets the work floor at `μ` (`μ · X ≥ (1 − ε) μ · W hon`, with `X` its rounds'
use `PeriodPlay.use`: the rounds' floor `RoundsFloor` of `DitherWorkSound` at the price vector `λ_r ∝ μ_r ρ_r`), frees
at most `ε C` with `C = μ · cap`, the period's devices at the dual price. The floor on the whole run (`WorkSound`) is
not enough here: it would count the other work `z` toward the credited part's floor. Counting one resource alone
(tensor-core time) is the floor at one vertex of the simplex, which is the dual price only where that resource binds
every job. -/
def FreedOfFloor : Prop :=
  ∀ (G : Accel) (Pd : Period G) (W ε : ℝ) (X μ : G.Res → ℝ) (z z' : Pd.J → ℝ),
    (∀ r, 0 ≤ μ r) → (∀ j r, 0 ≤ Pd.needs j r) → (∀ j, Pd.val j ≤ ∑ r, μ r * Pd.needs j r) → 0 ≤ W → 0 ≤ ε →
    Pd.Fits (fun r => W * Pd.hon r) z' → ∑ j, Pd.val j * z' j = ∑ r, μ r * (Pd.cap r - W * Pd.hon r) →
    Pd.Fits X z → (1 - ε) * ∑ r, μ r * (W * Pd.hon r) ≤ ∑ r, μ r * X r →
    Pd.FreedWithin W z ε (∑ r, μ r * Pd.cap r)

end Pouw.Guarantees.Dither
