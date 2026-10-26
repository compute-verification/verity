import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe: secure erasure from streamed PRF labels (experimental)

Added on 19 Oct 2026 for Daniel's tier 3, a proof of secure erasure for memory wipes, in place of the permutation-based
retention for wipes: the ideal-permutation premise (C8) is false for the reference Feistel10-SHAKE256 `Π₂`, where
1.53% of the fill rebuilds any label (verity#1122). The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`,
the protocol `verity/core/protocols/pous/PROTOCOL.md` ("The streamed-label wipe"). Proofs:
`Security/Proofs/Pous/StreamedWipe/`.

**What they rest on.** No named assumption. The labels' PRF (C3, claim `prf/sha-512`) enters as an advantage term in
the conclusion (`prfAdvantage`), which C3 bounds for an efficient prover; the guarantee holds for every prover. The
physical premises enter as the game's shape, and the challenger's deployment must make them true:

- *no unaccounted tier*: `M` is every bit the server can hold when the fill ends, all its memory tiers;
- C6, *egress certified over the whole wipe*: `E` bounds the bits that left the boundary from the fill's start to the
  last challenge, as the network certifier counts them, and the deadline keeps an answer from being fetched from
  outside in any other way;
- C5, *no side channel*: no label reaches the outside except through that egress, and nothing reaches the server during
  the challenges but the indices and what the outside sends, which is a function of `ξ` and the indices.

The game is not yet tied to the actual-circuit model of `Definitions.Accounting.Physical`, whose `Memory` and `Links`
constraints are these premises' forms there.

**What it covers.** The sequential reveal: round `j`'s answer reads the indices before it, not those after. A batched
reveal, where an answer can read its whole batch, is not covered.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **The streamed-label wipe is a proof of secure erasure.** For every prover of the sequential wipe of `N` labels of
`b` bits under `k` challenges, holding `M` bits inside the boundary when the fill ends, with `E` bits of egress, and
`q` bits of uniform prior content, and for every `g`:

`Pr[accepted ∧ the prior content survives] ≤ (g/N)^k + k · C(N, g+1) · 2^(M+E) / 2^(q+(g+1)·b) + Adv_PRF`,

where the labels are `prf K` under a uniform key, and `Adv_PRF` is the PRF's advantage against the distinguisher that
runs the wipe game with this prover. With `M = N·b` (the fill is the whole memory), the compression term is small once
`q` exceeds the missing labels' bits `(N − g − 1)·b` plus `E` plus `log₂ C(N, g+1) + log₂ k`, so an accepted prover
keeps at most about that much of the prior content: the labels it didn't hold, the egress and a counting term.

It is a bound in the game of `Pous.StreamedWipe`, for any `N`, `k`, `b`, `q`, `M`, `E`, `κ` and `prf`, with no
named assumption. Proof: `PousStreamedWipe.streamed_wipe`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipe : Prop :=
  ∀ (N k b q M E κ g : ℕ) (prf : Bits κ → Fin N → Bits b) (P : StreamedWipe.Prover N k b q M E),
    StreamedWipe.winReal prf P ≤
      StreamedWipe.bound N k b q M E g + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

/-- **The deployed wipe: 2.6 TB, 74,000 challenges.** `StreamedWipe` at `N = 40,625,000,000` labels of 512 bits (the
2.6 TB fill, every tier of one server, so `M = N · 512`) and `k = 74,000` sequential challenges: a prover with `E` bits
of egress that keeps at least `E + 8,600,000,000` bits (1.075 GB plus the egress) of uniform prior content through the
wipe is accepted with probability at most `2^-40` plus the PRF's advantage.

The error splits as `2^-41` for sampling (`g = N − 15,610,000`: the prover answers at most `ρ = 0.999616` of the
labels) and `2^-41` for compression, with the count `C(N, r) ≤ N^r ≤ 2^(36·r)`. The exact count gives about 1.023 GB
(`streamed_wipe.retained_bits`), and the explicit number here spends the difference on the cruder count.

Same footing as `StreamedWipe`: no named assumption; C3, the tiers and C5/C6 as there. Proof:
`PousStreamedWipe.streamed_wipe_meets_2600GB`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeets2600GB : Prop :=
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin 40625000000 → Bits 512)
    (P : StreamedWipe.Prover 40625000000 74000 512 q (40625000000 * 512) E),
    E + 8600000000 ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 40 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

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
