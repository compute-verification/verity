import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe at any number of reads, and the H200 at 5 GB

`Pous.Guarantees.StreamedWipe` for every number `k` of sequential timed reads: a ratio `r` with `(1 − 1/r)^k ≤ 2^-129`
puts the threshold at `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130` bits of the fill, plus the server's memory outside the fill
and the egress, at `2^-128` (`StreamedWipeMeetsReads`). `StreamedWipeMeetsFamily`'s `k = 2152` is its instance at
`r = 24`, and `StreamedWipeMeetsH200` is the instance at an H200 of 141 GiB (151,397,597,184 bytes) filled whole, at
most 5 GB in all. The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`, the proofs
`Security/Proofs/Pous/StreamedWipe/Family.lean`.

**What they rest on.** No named assumption, as `StreamedWipe`: the PRF (C3) enters as its advantage in the conclusion,
and every tier counted, C5 and C6 are the game's shape, which the challenger's deployment must make true. **What they
cover:** the sequential reveal (batch size 1), the only one the verdict accepts.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed wipe at `k` reads: `512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` bits at `2^-128`.**
`StreamedWipe` at the deployed label (`b = 512`, 64 bytes), for every number `k` of sequential timed reads and every
`r ≥ 1` with `(r − 1)^k · 2^129 ≤ r^k` (that is `(1 − 1/r)^k ≤ 2^-129`, a natural-number inequality the kernel decides),
for every fill of `n ≥ 1` labels on a server of `M = n · 512 + R` bits, `R` of them outside the fill (a driver's
reserve, a tier the fill doesn't reach): a prover with `E` bits of egress that keeps at least
`512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` bits of uniform prior content through the reads (`⌈n/r⌉` is
`(n + r − 1) / r`) is accepted with probability at most `2^-128` plus the PRF's advantage.

The fill's share is about `f·n·512` with `f = 1/r + 1/512`: `r` is the sampling floor, and the least `k` for it is about
`129 · ln 2 · r ≈ 89.4 · r` (`(1 − 1/r)^k ≈ e^(−k/r)`). The memory outside the fill counts in full, as the egress does.

The error splits as `2^-129` for sampling and `2^-129` for compression, at `g = n − ⌈n/r⌉`: the prover answers at most
`(r − 1)/r` of the labels, and with the count `C(n, g+1) ≤ 2^n` and `k < 2^(⌊log₂ k⌋ + 1)` the compression term is at
most `2^-129` at the threshold. At `r = 24` and `k = 2152` (`⌊log₂ 2152⌋ + 130 = 141`) it is `StreamedWipeMeetsFamily`'s
split.

Same footing as `StreamedWipe`: no named assumption; C3 (the PRF's advantage, in the conclusion), every tier counted
and C5/C6 as there. Proof: `PousStreamedWipe.streamed_wipe_meets_reads`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsReads : Prop :=
  ∀ r k : ℕ, 0 < r → (r - 1) ^ k * 2 ^ 129 ≤ r ^ k →
    ∀ n : ℕ, 0 < n →
      ∀ (R q E κ : ℕ) (prf : Bits κ → Fin n → Bits 512) (P : StreamedWipe.Prover n k 512 q (n * 512 + R) E),
        512 * ((n + r - 1) / r - 1) + n + Nat.log2 k + 130 + R + E ≤ q →
        StreamedWipe.winReal prf P ≤
          (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **The H200 at 5 GB: 141 GiB at 2906 reads.** `StreamedWipeMeetsReads` at `r = 33` and `k = 2906` on an H200 of
141 GiB: a fill of `n = 2,365,587,456` labels, `151,397,597,184` bytes (the data sheet's 141 GB in binary units), and
`R = 0`. A prover with `E` bits of egress that keeps at least `39,068,035,213 + E` bits of uniform prior content
through the reads is accepted with probability at most `2^-128` plus the PRF's advantage. The threshold is about
4.884 GB plus the egress. NVML's total on the deployed card is `150,754,820,096` bytes (143,771 MiB), 613 MiB less; that
card filled whole has the lower threshold `38,902,167,181` bits at the same `r` and `k`. A fill that leaves some of the
device out adds that memory to `R` in full, so it needs a larger `r` for the same 5 GB: it is `StreamedWipeMeetsReads`
at the `n` and `R` it has.

`r = 33` is the least ratio whose threshold is at most 5 GB (`40·10^9` bits): at `r = 32`, with its least `k = 2817`,
it is `40,214,986,381` bits (about 5.027 GB). `k = 2906` is the least `k` with `32^k · 2^129 ≤ 33^k`. Same footing as
the family. Proof: `PousStreamedWipe.streamed_wipe_meets_H200`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsH200 : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 2365587456 → Bits 512)
    (P : StreamedWipe.Prover 2365587456 2906 512 q (2365587456 * 512) E),
    39068035213 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
