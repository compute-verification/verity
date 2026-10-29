import Definitions.Pous.Model.Digest
import Definitions.Pous.Model.BandChain
import Definitions.Pous.Accounting.EraseMeets

/-!
# Secure erasure: volatile memory in full use through the spot reads

Under this model the disks are unplugged, so the prover's memory is volatile
only (device memory and host DRAM), and the audit is a proof of memory utilization. The game is
`Security/Definitions/Pous/Model/SecureErasure.lean`, the requirement `EraseMeets`
(`Security/Definitions/Pous/Accounting/EraseMeets.lean`). Proofs: `Security/Proofs/Pous/SecureErasure/`.

**What they rest on.** The ideal-primitive model (false at the reference `Π₂`: see `SecureErasureMeetsFamily`) and the
`(D, Q)` game the band family is proved in, and the incompressibility of the keyed fill there (the band's `TimedINC`,
proved, `PousSpongeChain.chainExPostFactoG_v6`): a timed spot read within `(D, Q)` needs the block, or what it takes to
rebuild it within `D` rounds, in memory. No named assumption. **No bandwidth-separation premise:** none of them reads or
assumes the host-link bound `TwoTierBandwidth`, which is dropped; device memory and host DRAM count
alike.
-/

namespace Pous.Guarantees

open scoped ENNReal

/-- **Secure erasure on the band, at every memory size from 32 MiB to 2^64 segments.** For every `N` with
`1 ≤ N ≤ 2^64` and every in-degree `d ≥ 5`, the `N`-segment band `chainBandSeg N (2^9) d` (`2^9` blocks of 64 KiB per
segment, so a fill of `M = N · 32 MiB`), keyed by the memory challenger's fresh key as its salt (per-segment tags
`segDom salt`, any 192-bit salt, or one global 256-bit salt), meets `EraseMeets` at `f = 1 − ρ = 1/19`, `D = 5`,
`Q = 2^20`, `s = 2152` spot reads and `ε = 2^-128`:

- every prover whose volatile memory at each spot read's arrival is at most `⌊(18/19)·|C|⌋` bits is accepted with
  probability at most `2^-128`;
- every prover with `M` bytes of volatile memory that keeps foreign content of at least `M/19` bytes and 32 bytes
  (`f·|C| + 2λ` bits) through the spot reads wins with probability at most `2^-128`. So no prover is accepted and still
  holds that foreign content at every spot read's arrival, except with probability `2^-128`. This bounds what survives
  the reads, not what is held at one instant: a prover can hold about `M/10` bytes of foreign content at the first spot
  read's arrival and be accepted with probability about 0.9, erasing it while it rebuilds, during the first reads, the
  blocks it dropped.

Cite it as "cannot keep it through all the reads", never as "holds at the challenge", and only as a statement in the
ideal-permutation model: `Π₂` an ideal permutation called whole, `(D, Q)` rounds and counts of whole calls. **It is not
a bound on a band-fill wipe at the reference `Π₂`** (Feistel10 over SHAKE256; `feistel-indifferentiability` is refuted).
There a prover keeps the five odd rounds' 200-byte SHAKE256 states of each block (1,000 bytes a 64 KiB block, 1.53% of
`|C|`) and rebuilds any block in 481 sequential Keccak-f, a tenth of one `Π₂` call and within the `D = 5` deadline on
one CPU core. So no read deadline bounds it to `(D, Q) = (5, 2^20)`, and it is accepted at all 2152 reads with about 98%
of its memory foreign (the resume attack). For a wipe, cite `Pous.Guarantees.StreamedWipeMeetsFamily`.

`M` must be the prover's whole volatile memory, on-chip SRAM included; with host DRAM, `N` is the total over 32 MiB.
`N = 3072` is 96 GiB (`SecureErasureMeets96GiB`), the GDDR7 an RTX PRO 6000 has, but no fill covers all of it: software
sees about 95 GiB, and the calibrated fill reaches 93.86 GiB, so `N = 3003` whole segments (93.84 GiB). On one
device the guarantee holds at the `N` the fill reaches, and the memory outside the fill adds to what a prover can keep.
The read deadline `T_read` is a symbol of the model; no hardware deadline realizes it at the reference `Π₂` (above). In
the model it is any deadline within which the prover completes at most 5 rounds of calls to the ideal `Π₂` and `2^20`
calls per answer. The fill deadline does not enter.

The numbers: `9593/10000` bounds the band's certified per-read success `p₀` at `⌊ρ·|C|⌋` bits for every `N` (one
segment is the worst, `p₀ ≈ 0.959292`), and `2152` is the least `s` with `(9593/10000)^s ≤ 2^-129`. That is more than the
`1641` reads that `(18/19)^s ≤ 2^-128` alone would need, because the band's incompressibility bound `TimedINC` has slack
beyond `ρ`. The error splits as `2^-129` (sampling) `+ 2^-130` (INC's union over histories) `+ 2^-130` (the foreign
content's count, at most `2152 · 2^-256`).

It rests on the ideal-primitive model (false at the reference `Π₂`, above), the keyed fill's incompressibility (the
band's `TimedINC`, proved) and the `(D, Q)` game, as `BandMultiMeetsFamily` does; no named assumption and no
bandwidth-separation premise (`TwoTierBandwidth`, which is dropped, is not used). Proof:
`PousErase.secure_erasure_meets_family`.

Frozen (no-function, hash): an `EraseMeets` in an ideal model, which no Lean function computes. -/
def SecureErasureMeetsFamily : Prop :=
  ∀ (N : ℕ) [NeZero N], N ≤ 2 ^ 64 → ∀ d : ℕ, 5 ≤ d →
    (∀ salt : Bits 192,
      EraseMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
        (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128)) ∧
    (∀ salt : Bits 256,
      EraseMeets (Pous.BandMulti.chainBandSeg N (2 ^ 9) d 524288 512 (Pous.Sponge.dom salt))
        (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128))

/-- **Secure erasure of 96 GiB.** `SecureErasureMeetsFamily` at `N = 3072` segments of 32 MiB, with the implementation's
per-segment tags. For every `d ≥ 5` and salt, and a prover whose whole volatile memory is the 96 GiB: if its memory at
each spot read's arrival is at most `⌊(18/19)·|C|⌋` bits it is accepted with probability at most `2^-128`, and if it
keeps at least `96 GiB / 19 + 32` bytes (about 5.05 GiB) of foreign content through the 2152 spot reads it wins with
probability at most `2^-128`. 96 GiB is an RTX PRO 6000's GDDR7, but one device cannot hold this fill (about 95 GiB is
visible to software); for one device, cite the family at the `N` its fill reaches, with the memory outside the fill
added to what a prover can keep. Same footing as the family: no named assumption, no bandwidth-separation premise.
Proof: `PousErase.secure_erasure_meets_96GiB`.

Frozen (no-function, hash): an `EraseMeets` in an ideal model, which no Lean function computes. -/
def SecureErasureMeets96GiB : Prop :=
  ∀ d : ℕ, 5 ≤ d → ∀ salt : Bits 192,
    EraseMeets (Pous.BandMulti.chainBandSeg 3072 (2 ^ 9) d 524288 512 (Pous.BandMulti.segDom salt (2 ^ 9)))
      (1 - Params.ρ) 5 (2 ^ 20) 2152 ((2⁻¹ : ℝ≥0∞) ^ 128)

end Pous.Guarantees
