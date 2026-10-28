import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe at 2152 reads: what `verity/wipe/v0` publishes

`Pous.Guarantees.StreamedWipe` at the figure the wipe port publishes for an accepted session, `⌈M/19⌉ + E` bits at
`2^-128` with 2152 sequential timed reads, for every fill of at least one 32 MiB segment, and at 96 GiB and 141 GB.
The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`, the proofs
`Security/Proofs/Pous/StreamedWipe/Family.lean`.

**What they rest on.** No named assumption, as `StreamedWipe`: the PRF (C3) enters as its advantage in the conclusion,
and every tier counted, C5 and C6 are the game's shape, which the challenger's deployment must make true. The port's
`eps = 2^-128` leaves the PRF's advantage out. **What they cover:** the sequential reveal (batch size 1), the only one
the verdict accepts.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed wipe at 2152 reads: `⌈M/19⌉ + E` bits at `2^-128`, for every fill of a segment or more.**
`StreamedWipe` at the deployed label (`b = 512`, 64 bytes) and `k = 2152` sequential timed reads, for every fill of
`n ≥ 2^19` labels, at least one 32 MiB segment and with no upper limit (`N` whole segments are `n = N · 2^19`; a fill
that ends inside a segment is covered too), the fill being every tier of the server (`M = n · 512`): a prover with `E`
bits of egress that keeps at least `⌈M/19⌉ + E` bits of uniform prior content through the reads (`(M + 18) / 19 + E ≤ q`;
`M/19` is never whole, so this is every `q` above `f·M + E` with `f = 1/19`) is accepted with probability at most
`2^-128` plus the PRF's advantage.

It is what `verity/wipe/v0` publishes for an accepted session at `k = 2152` over a fill of at least one segment:
`foreign_bits = ⌈M/19⌉ + E` at `eps = 2^-128`, `E` the egress the network certifier counted.

The error splits as `2^-129` for sampling and `2^-129` for compression, at `g = n − ⌈n/24⌉`: the prover answers at
most `23/24` of the labels, `(23/24)^2152 ≤ e^(−2152/24) ≤ 2^-129`, and with the count `C(n, g+1) ≤ 2^n` and
`2152 ≤ 2^12` the compression term is at most `2^-129` once `q ≥ E + 512·(⌈n/24⌉ − 1) + n + 141`, which `⌈M/19⌉ + E`
exceeds (per label, `512/24 + 1 < 512/19`). Sampling cannot take all of `f`: at `g = (18/19)·n`, where the sampling term
is `(18/19)^2152 ≈ 2^-167.9`, the compression term needs the missing labels' `M/19` bits plus the count.

Same footing as `StreamedWipe`: no named assumption; C3 (the PRF's advantage, in the conclusion), every tier counted
and C5/C6 as there. Proof: `PousStreamedWipe.streamed_wipe_meets_family`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeetsFamily : Prop :=
  ∀ n : ℕ, 2 ^ 19 ≤ n →
    ∀ (q E κ : ℕ) (prf : Bits κ → Fin n → Bits 512) (P : StreamedWipe.Prover n 2152 512 q (n * 512) E),
      (n * 512 + 18) / 19 + E ≤ q →
      StreamedWipe.winReal prf P ≤
        (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **The streamed wipe of 96 GiB at 2152 reads.** `StreamedWipeMeetsFamily` at `n = 1,610,612,736` labels, `3072`
segments of 32 MiB (`M = 824,633,720,832` bits): a prover with `E` bits of egress that keeps at least
`43,401,774,781 + E` bits (`⌈M/19⌉ + E`, about 5.05 GiB plus the egress) of uniform prior content through the reads is
accepted with probability at most `2^-128` plus the PRF's advantage. 96 GiB is an RTX PRO 6000's GDDR7, which no fill
covers whole (software sees about 95 GiB); for one device, cite the family at the `n` its fill reaches, with the
memory outside the fill added to what a prover can keep. Same footing as the family. Proof:
`PousStreamedWipe.streamed_wipe_meets_96GiB`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeets96GiB : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 1610612736 → Bits 512)
    (P : StreamedWipe.Prover 1610612736 2152 512 q (1610612736 * 512) E),
    43401774781 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **The streamed wipe of 141 GB at 2152 reads.** `StreamedWipeMeetsFamily` at `n = 2,203,125,000` labels, the
`141·10^9` bytes of an H200's HBM3e (about 4,202.1 segments of 32 MiB; `M = 1,128,000,000,000` bits): a prover with `E`
bits of egress that keeps at least `59,368,421,053 + E` bits (`⌈M/19⌉ + E`, about 7.42 GB plus the egress) of uniform
prior content through the reads is accepted with probability at most `2^-128` plus the PRF's advantage. For a fill
that reaches less of the device, cite the family at the `n` it reaches, with the memory outside the fill added to what
a prover can keep. Same footing as the family. Proof: `PousStreamedWipe.streamed_wipe_meets_141GB`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeets141GB : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 2203125000 → Bits 512)
    (P : StreamedWipe.Prover 2203125000 2152 512 q (2203125000 * 512) E),
    59368421053 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
