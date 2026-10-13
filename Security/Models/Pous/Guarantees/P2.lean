import Definitions.Pous.Accounting.Params
import Definitions.Pous.Model.FreeBlocks
import Definitions.Pous.Model.M1p
import Models.Pous.Assumptions.P2

/-!
# The `m1p` statements at P2's parameters (trusted)

Six statements about the abstract model `m1p` (`Security/Definitions/Pous/Model/M1p.lean`: independent uniform
permutations of `[0, p)`, forward queries only), at the block counts, widths and primes P2 uses, that `grade.sh` grades
against, in namespace `Pous.Guarantees` with the rest. The proofs are in `Security/Proofs/Pous/P2/`
(`Pous.SecurityProofs.P2MeetsM1pB19Uncond` and the others in `Security/Proofs/Pous.lean`); a docstring's
`lean/submissions/efficient-crypto/…` path names the store draft each was graded from. A seventh,
`P2SlackFamilyFreeBlocks` (the sampled setup's free blocks), is proved in `Security/Proofs/Pous/FreeBlocks/`. The last,
`P2RingAuditSecureOfReduces`, carries the `m1p` row to P2 in the ring model under the reduction the spec states.

The first seven are theorems about `m1p` only, not about the concrete P2: none models square–mask–square, SHAKE256,
roots or time. The last reaches square–mask–square only in the ring model (`p2ring`) and only under its hypothesis, a
named conjecture. So none of them says the deployed P2 is secure.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-! ## M1 on P2's exact domain `[0, p)` -/

/-- **Unconditional (no hypothesis): the POUS MVP's block count on P2's exact domain, for every payload width in
range.** `2^19` independent forward-only uniform permutations of `[0, p)`, `p = 2^16448 − 21065` (P2's recorded
16,448-bit prime), acting on 16,448-bit strings with the tail `[p, 2^16448)` fixed, each block carrying `b` payload bits
zero-extended to 16,448 bits (`Pous.m1p`), meet the requirement with sequential reveal, `2^20` forward queries per
answer, any forward-query depth, 88 challenges and `ε = 2^-128`, for every `b` with `20·16448 ≤ 21·b` (the space
bound) and `b ≤ 16447` (equivalently `2^b ≤ p`: every payload lies in `[0, p)`). Instances: `b = 16447` (one pad bit)
and `b = 16432` (2,054 bytes, the MVP's packing). Pinned as drafted
(`lean/submissions/efficient-crypto/p2-domain/P2Domain.lean`, `PousP2p.p2_meets_m1p_b19`). 88 is the least k this
certificate route gives. Storing floor(S/w) code blocks raw breaks `Meets` at k = 85 for every `b`
(`not_meets_m1p_b19_k85`).

Scope: abstract M1p only, not a concrete SMS or timing theorem. This pin resolves the ideal-domain cardinality
mismatch by using `[0,p)`, but Lean does not prove primality, the concrete byte/bit encoding or packing, the
per-block SMS-to-M1p instantiation, SeqRoot, the wall-clock-to-`Q` calibration, authentication, accounting or
fresh setup.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2MeetsM1pB19Uncond : Prop :=
  ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, Meets (m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 Params.εMax

/-- **Unconditional (no hypothesis): every block count from 2^8 to 2^23 and every payload width in range, on P2's
exact 16,448-bit domain.** For every B with `2^8 ≤ B ≤ 2^23` and every `b` with `20·16448 ≤ 21·b` and `b ≤ 16447`,
`m1p B 16448 (2^16448 − 21065) b` meets the requirement with sequential reveal, `2^20` forward queries per answer, any
forward-query depth, 88 challenges and `ε = 2^-128`. Pinned as drafted
(`lean/submissions/efficient-crypto/p2-domain/P2Domain.lean`, `PousP2p.family`). `k = 85` fails at its `2^19` instance,
for every `b` (`family_k85_fails`). Scope: as `P2MeetsM1pB19Uncond`.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2MeetsM1pFamilyUncond : Prop :=
  ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 →
    ∀ D, Meets (m1p B 16448 (2 ^ 16448 - 21065) b) .sequential D (2 ^ 20) 88 Params.εMax

/-- **Unconditional (no hypothesis): the 8,192-bit row on P2's exact domain, for every payload width in range.**
`2^24` independent forward-only uniform permutations of `[0, p)`, `p = 2^8192 − 9345` (P2's recorded 8,192-bit prime),
each block carrying `b` payload bits (`Pous.m1p`), meet the requirement with sequential reveal, `2^20` forward queries
per answer, any forward-query depth, 90 challenges and `ε = 2^-128`, for every `b` with `20·8192 ≤ 21·b` and
`b ≤ 8191`. Instances: `b = 8191` and `b = 8176` (1,022 bytes). Pinned as drafted
(`lean/submissions/efficient-crypto/p2-domain/P2Domain.lean`, `PousP2p.p2_meets_m1p_8192`).
Scope: as `P2MeetsM1pB19Uncond`; it asserts no zero-RTT or root-latency fact.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2MeetsM1pW8192Uncond : Prop :=
  ∀ b, 20 * 8192 ≤ 21 * b → b ≤ 8191 →
    ∀ D, Meets (m1p (2 ^ 24) 8192 (2 ^ 8192 - 9345) b) .sequential D (2 ^ 20) 90 Params.εMax

/-! ## Slack rows on P2's exact domain: an adversary with `λ` extra state bits per block -/

/-- **Unconditional (no hypothesis): the MVP's block count with `λ = 64`, `128` and `256` bits of state slack per
block, for every payload width in range.** `m1p (2^19) 16448 (2^16448 − 21065) b` (P2's ECPP-certified 16,448-bit
prime) is correct, holds `W`, and is within the space bound. Every adversary holding `⌊ρ·|C|⌋ + λ·2^19` bits, with
sequential reveal, `2^20` forward queries per answer and any depth, is accepted with probability at most
`1% + 2^-128`: at `k = 95` for `λ = 64`, `k = 104` for `λ = 128` and `k = 127` for `λ = 256`. This holds for every
`b` with `20·16448 ≤ 21·b` and `b ≤ 16447`. Proposed in `lean/submissions/efficient-crypto/p2-margin/P2Margin.lean`
(`PousMargin.b19_64`, `b19_128`, `b19_256`). Each k is the least this certificate route gives. Storing
⌊(S + λ·2^19)/16448⌋ code blocks raw breaks the requirement at every `k ≤ 92`, `100` and `121` respectively
(`raw_b19_64`, `raw_b19_128`, `raw_b19_256`).

Scope: abstract M1p only, not a concrete SMS or timing theorem. Lean does not prove the concrete byte/bit encoding or
packing, the per-block SMS-to-M1p instantiation, SeqRoot, the wall-clock-to-`Q` calibration, authentication,
accounting or fresh setup. The slack `λ` is a budget for structural savings an instantiation might leak; it is not
derived from any instantiation. The `λ·B` allowance conservatively covers leakage representable as at most `λ`
retained advice bits per block, aggregated into arbitrary global state. It does not cover leakage that changes the
online oracle interface, creates cross-block correlations in the instantiated maps, or exceeds `λ·B` bits of reusable
information.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2SlackB19Uncond : Prop :=
  ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    let sch := m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b
    sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 64 * sch.B) D (2 ^ 20) 95 (ENNReal.ofReal Params.δ)
        Params.εMax ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 128 * sch.B) D (2 ^ 20) 104 (ENNReal.ofReal Params.δ)
        Params.εMax ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 256 * sch.B) D (2 ^ 20) 127 (ENNReal.ofReal Params.δ)
        Params.εMax

/-- **Unconditional (no hypothesis): the slack rows for block counts from 2^8 to 2^23 and every payload width in
range.** For every `B` with `2^8 ≤ B ≤ 2^23` and every `b` with `20·16448 ≤ 21·b` and `b ≤ 16447`,
`m1p B 16448 (2^16448 − 21065) b` is correct, holds `W`, and is within the space bound. Every adversary holding
`⌊ρ·|C|⌋ + λ·B` bits, with sequential reveal, `2^20` forward queries per answer and any depth, is accepted with
probability at most `1% + 2^-128`, at:

- `λ = 64`: `k = 96`, and `k = 95` for `B ≥ 2^10`;
- `λ = 128`: `k = 105`, and `k = 104` for `B ≥ 2^9`;
- `λ = 256`: `k = 128`, and `k = 127` for `B ≥ 2^10`.

Proposed in `lean/submissions/efficient-crypto/p2-margin/P2Margin.lean` (`PousMargin.fam_*`). Below those thresholds
the uniform certificate `g = ⌊c·B⌋ + 1` does not reach the lower k. At `λ = 64` and `λ = 256` no `g` passes the
bound `logHistGammaM1_le` at some such `B` (for example `B = 275`), while the lower bound on `Γ` does not rule the
lower `k` out there, so the family's lower `k` for small `B` is open. Statement non-vacuity: as `P2SlackB19Uncond`,
at `B = 2^19`.
Scope: as `P2SlackB19Uncond`.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2SlackFamilyUncond : Prop :=
  ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    let sch := m1p B 16448 (2 ^ 16448 - 21065) b
    sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 64 * sch.B) D (2 ^ 20) 96 (ENNReal.ofReal Params.δ)
        Params.εMax ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 128 * sch.B) D (2 ^ 20) 105 (ENNReal.ofReal Params.δ)
        Params.εMax ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 256 * sch.B) D (2 ^ 20) 128 (ENNReal.ofReal Params.δ)
        Params.εMax ∧
      (2 ^ 9 ≤ B → AuditSecure sch .sequential (sch.stateBound Params.ρ + 128 * sch.B) D (2 ^ 20) 104
        (ENNReal.ofReal Params.δ) Params.εMax) ∧
      (2 ^ 10 ≤ B →
        AuditSecure sch .sequential (sch.stateBound Params.ρ + 64 * sch.B) D (2 ^ 20) 95 (ENNReal.ofReal Params.δ)
          Params.εMax ∧
        AuditSecure sch .sequential (sch.stateBound Params.ρ + 256 * sch.B) D (2 ^ 20) 127 (ENNReal.ofReal Params.δ)
          Params.εMax)

/-- **Unconditional (no hypothesis): the error split at the MVP's block count.** The rows `P2MeetsM1pB19Uncond`
(`λ = 0`, `k = 88`) and `P2SlackB19Uncond` (`λ = 64`, `128`, `256` at `k = 95`, `104`, `127`) hold with
`ε = 2^-129` in place of `2^-128`, for every payload width in range. The ideal-model term therefore needs at most
half of the `2^-128` budget. The proved `log₂(|Hist|·Γ)` bounds are about `−3.0·10^6`, `−1.0·10^6`, `−2.0·10^6` and
`−1.2·10^6` (`PousMargin.log_b19_*`). Proposed in `lean/submissions/efficient-crypto/p2-margin/P2Margin.lean`
(`PousMargin.b19_meets_129`, `b19_64_129`, `b19_128_129`, `b19_256_129`). Scope: as `P2SlackB19Uncond`. Under an
external reduction in which all unmodelled errors add and total at most `2^-129`, the combined cryptographic error is
at most `2^-128`; Lean proves neither that reduction nor the external bound.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2ErrorSplitB19Uncond : Prop :=
  ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    let sch := m1p (2 ^ 19) 16448 (2 ^ 16448 - 21065) b
    sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ) D (2 ^ 20) 88 (ENNReal.ofReal Params.δ)
        ((2⁻¹ : ENNReal) ^ 129) ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 64 * sch.B) D (2 ^ 20) 95 (ENNReal.ofReal Params.δ)
        ((2⁻¹ : ENNReal) ^ 129) ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 128 * sch.B) D (2 ^ 20) 104 (ENNReal.ofReal Params.δ)
        ((2⁻¹ : ENNReal) ^ 129) ∧
      AuditSecure sch .sequential (sch.stateBound Params.ρ + 256 * sch.B) D (2 ^ 20) 127 (ENNReal.ofReal Params.δ)
        ((2⁻¹ : ENNReal) ^ 129)

/-- **The sampled setup's free blocks.** For every `B` with `2^8 ≤ B ≤ 2^23` and every `b` with
`20·16448 ≤ 21·b` and `b ≤ 16447`, the `m1p B 16448 (2^16448 − 21065) b` audit with at most `⌊B/100⌋` free blocks
(`AuditSecureFree`: blocks a sampled setup let through, which pass every challenge) accepts every adversary holding
`⌊ρ_eff·|C|⌋ + 128·B` bits, `ρ_eff = 18/19 · 99/100 ≈ 0.938`, with sequential reveal, `2^20` forward queries per
answer and any depth, with probability at most `1% + 2^-128 + 2^-129`, at `k = 140`.

Where the numbers come from: a sampled setup of `s = 8,828` blocks (`⌈128 / −log₂ 0.99⌉`) leaves more than `B/100`
blocks not decoding to `W` with probability at most `2^-128`, which the profile's failure union carries, not this
statement. `ρ_eff` is `ρ` of the blocks that must be stored. The proof (`Pous.FreeBlocks.p2_slack_family_free_blocks`)
counts the free blocks in the 1% term, `((g − 1 + ⌊B/100⌋)/B)^140 ≤ (0.9576 + 0.01)^140 ≤ 1/100` with
`g = ⌊0.9576·B⌋ + 1`, and the state is below `P2SlackFamilyUncond`'s `λ = 0` one, so it shows `1% + 2^-129`; the `2^-128`
term is slack. Scope: as `P2SlackB19Uncond`.

Frozen (no-function, hash): the abstract `m1p` in the random-oracle model, not the deployed P2. -/
def P2SlackFamilyFreeBlocks : Prop :=
  ∀ (B : ℕ) [NeZero B], 2 ^ 8 ≤ B → B ≤ 2 ^ 23 → ∀ b, 20 * 16448 ≤ 21 * b → b ≤ 16447 → ∀ D,
    let sch := m1p B 16448 (2 ^ 16448 - 21065) b
    AuditSecureFree sch (B / 100) (sch.stateBound (Params.ρ * (99 / 100)) + 128 * sch.B) D (2 ^ 20) 140
      (ENNReal.ofReal Params.δ) (Params.εMax + (2⁻¹ : ℝ≥0∞) ^ 129)

/-! ## From the `m1p` row to P2 in the ring model -/

/-- **The bridge: the reduction the spec states gives the weak form, at every `(D, Q, L)`.** If every
`⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded responder against `p2ring (2^19) 16448 (2^16448 − 21065) 16432` at 105 challenges is
matched up to `2^-130` by an `m1p` responder with `128` more bits per block and `2^20` forward queries per answer
(`P2RingReducesToM1p`), then P2 in the ring model accepts every such responder with probability at most
`1% + 2^-129 + 2^-130` (`P2RingAuditSecure`). It rests on no assumption beyond its hypothesis: the `m1p` side is
`P2ErrorSplitB19Uncond`'s `λ = 128` row (`k = 104`), and one more challenge never raises the sequential audit's
acceptance (`PousP2Bridge.pr_seqAccepts_le`).

Frozen (no-function, hash): it relates two ring-model statements about the abstract P2, not the setup check's code. -/
def P2RingAuditSecureOfReduces : Prop :=
  ∀ D Q L, Pous.P2.Assumptions.P2RingReducesToM1p D Q L → Pous.P2.Assumptions.P2RingAuditSecure D Q L

/-! ## From the per-round conjecture: the deployed whole-audit form and the graded bound -/

/-- **The per-round form gives today's whole-audit form at `2^19` blocks.** If P2 in the ring model holds per round
with floor `p` and bad-event mass `ε₀` per round (`P2RingRoundSecure p ε₀ D Q L`), and
`(1 − p)^105 + 105·ε₀ ≤ 1% + 2^-129 + 2^-130`, then every `⌊ρ·|C|⌋`-bit, `(D, Q, L)`-bounded responder against
`p2ring (2^19) 16448 (2^16448 − 21065) 16432` at 105 challenges is accepted with probability at most
`1% + 2^-129 + 2^-130` (`P2RingAuditSecure D Q L`). A responder of today's game holds `⌊ρ·|C|⌋` bits at all 105
boundaries, so it is `P2RingGradedOfRoundSecure` at `m = k = 105`. With `ε₀ = 2^-137` the condition holds at
`p = 0.043` (`PousP2Rounds.floor_043_eps`); with any `105·ε₀ ≤ 2^-130` it fails for `p < 1 − 0.01^(1/105) ≈ 0.04291`.

Frozen (no-function, hash): it relates two ring-model statements about the abstract P2, not the setup check's code. -/
def P2RingAuditSecureOfRoundSecure : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ),
    (1 - p) ^ 105 + 105 * ε₀ ≤ ENNReal.ofReal Params.δ + ((2⁻¹ : ℝ≥0∞) ^ 129 + (2⁻¹ : ℝ≥0∞) ^ 130) →
    Pous.P2.Assumptions.P2RingRoundSecure p ε₀ D Q L → Pous.P2.Assumptions.P2RingAuditSecure D Q L

/-- **The graded bound: below `S₀` at `m` of `k` boundaries passes with probability at most `(1 − p)^m + k·ε₀`.** If
P2 in the ring model holds per round with floor `p` and bad-event mass `ε₀` per round
(`P2RingRoundSecure p ε₀ D Q L`), then for
every block count `N` from `2^19` to `2^25`, every `k`, every payload `W` and every responder against
`p2ring N 16448 (2^16448 − 21065) 16432` holding `Sz j` bits at boundary `j` (`VarResponderOn`, each round within
`(D, Q, L)`), all `k` answers are correct with probability at most `(1 − p)^m + k·ε₀`, where `m` is the number of
boundaries `j < k` at which the state is at most `S₀ = ⌊ρ·|C|⌋` bits (`lowBoundaries`). Over challenges at random
times the share of such boundaries estimates the share of time the server held at most `S₀`, which is the time
average the capacity account reads.

Frozen (no-function, hash): it relates two ring-model statements about the abstract P2, not the setup check's code. -/
def P2RingGradedOfRoundSecure : Prop :=
  ∀ (p ε₀ : ℝ≥0∞) (D Q L : ℕ), Pous.P2.Assumptions.P2RingRoundSecure p ε₀ D Q L →
    ∀ (N : ℕ) [NeZero N], 2 ^ 19 ≤ N → N ≤ 2 ^ 25 →
      ∀ (Sz : ℕ → ℕ) (k : ℕ) (W : Bits (p2ring N 16448 (2 ^ 16448 - 21065) 16432).n)
        (A₁ : Preprocessor (p2ring N 16448 (2 ^ 16448 - 21065) 16432) (Sz 0))
        (A₂ : VarResponderOn (p2ring N 16448 (2 ^ 16448 - 21065) 16432) Sz D Q L),
        pr (varPassesOn _ k k W A₁ A₂) ≤
          (1 - p) ^ lowBoundaries Sz ((p2ring N 16448 (2 ^ 16448 - 21065) 16432).stateBound Params.ρ) k + k * ε₀

end Pous.Guarantees
