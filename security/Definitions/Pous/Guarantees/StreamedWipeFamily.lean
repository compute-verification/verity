module

public import Definitions.Core.Accel.Card
public import Definitions.Pous.Model.StreamedWipe

/-!
# The streamed-label wipe at 2152 reads: what `verity/wipe/v0` publishes

`Pous.Guarantees.StreamedWipe` at the figure the wipe port publishes for an accepted session, `⌈M/19⌉ + E` bits at
`2^-128` with 2152 sequential timed reads, for every fill of at least one 32 MiB segment, and at a card's device
memory (`Verity.Accel.Card`, handed in).
The game is `Security/Definitions/Pous/Model/StreamedWipe.lean`, the proofs
`Security/Proofs/Pous/StreamedWipe/Family.lean`.

**What they rest on.** No named assumption, as `StreamedWipe`: the PRF (C3) enters as its advantage in the conclusion,
and every tier counted, C5 and C6 are the game's shape, which the challenger's deployment must make true. The port's
`eps = 2^-128` leaves the PRF's advantage out. **What they cover:** the sequential reveal (batch size 1), the only one
the verdict accepts.
-/

@[expose] public section

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
`foreign_bits = ⌈M/19⌉ + E` at `eps = 2^-128`, `E` the egress the network attester counted.

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

/-- **The streamed wipe of a card's device memory at 2152 reads.** `StreamedWipeMeetsFamily` on a card `c` handed in
(`Verity.Accel.Card`), its device memory filled whole: `n = c.size .dram / 16` labels of 64 bytes (`size .dram` is in
32-bit words), at least one 32 MiB segment, and `M = n · 512` bits. The card it is stated for is the RTX PRO 6000's,
`hardware/cards/rtx-pro-6000-server.toml`, whose `store.dram` is its device memory. A prover with `E` bits of egress
that keeps at least `⌈M/19⌉ + E` bits of uniform prior content through the reads is accepted with probability at most
`2^-128` plus the PRF's advantage. No fill covers the device memory whole (software sees less of it); for one device,
cite the family at the `n` its fill reaches, with the memory outside the fill added to what a prover can keep. Same
footing as the family. Proof: `PousStreamedWipe.streamed_wipe_meets_96GiB`.

Frozen (no-function): a bound in the wipe game's model, which no Lean function computes. -/
def StreamedWipeMeets96GiB : Prop :=
  ∀ c : Verity.Accel.Card, 2 ^ 19 ≤ c.size .dram / 16 →
  ∀ (q E κ : ℕ) (prf : Bits κ → Fin (c.size .dram / 16) → Bits 512)
    (P : StreamedWipe.Prover (c.size .dram / 16) 2152 512 q (c.size .dram / 16 * 512) E),
    (c.size .dram / 16 * 512 + 18) / 19 + E ≤ q →
    StreamedWipe.winReal prf P ≤
      (2⁻¹ : ℝ≥0∞) ^ 128 + StreamedWipe.prfAdvantage prf (StreamedWipe.wipeDistinguisher P)

end Pous.Guarantees
